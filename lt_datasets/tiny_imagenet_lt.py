"""Tiny ImageNet 200, long-tailed via exponential imbalance.

Loads from the HuggingFace mirror `zh-plus/tiny-imagenet` once (cached under
`<root>/tiny-imagenet-hf/`), then exposes the same API as
`datasets/cifar_lt.py::CIFAR10LT/CIFAR100LT` so it drops into the existing
HAE training pipeline:

  - Same `__init__` kwargs (`root, imbalance_factor, train, transform`)
  - Same `__getitem__` -> (image_tensor, label)
  - Same `get_class_counts()` and `get_head_mid_tail_split()` helpers

The val/test split is kept *balanced* (50/class) so head/mid/tail accuracy
metrics remain comparable across imbalance factors — standard convention
for long-tail benchmarks.

Native resolution is 64×64. Use `make_tiny_imagenet_lt_transform(image_size=64)`
to keep that, or pass `image_size=32` to resize for the CIFAR-compatible path.
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


HF_DATASET_NAME = "zh-plus/tiny-imagenet"
NUM_CLASSES = 200
NATIVE_RESOLUTION = 64


class TinyImageNetLT(Dataset):
    """Long-tailed Tiny ImageNet 200, via exponential class-frequency decay."""

    def __init__(
        self,
        root: str = "./data",
        imbalance_factor: float = 0.01,
        train: bool = True,
        transform=None,
        image_size: int = NATIVE_RESOLUTION,
        seed: int = 0,
        download: bool = True,  # kept for API parity with CIFAR{10,100}LT
    ):
        super().__init__()
        try:
            from datasets import load_dataset  # HuggingFace `datasets`
        except ImportError as e:
            raise ImportError(
                "TinyImageNetLT requires the HuggingFace `datasets` library. "
                "Install with: pip install 'datasets>=2.14'"
            ) from e

        self.root = root
        self.train = train
        self.imbalance_factor = imbalance_factor
        self.image_size = int(image_size)
        self.seed = seed
        self.transform = transform or make_tiny_imagenet_lt_transform(
            image_size=self.image_size, train=train
        )
        self.num_classes = NUM_CLASSES

        cache_dir = os.path.join(root, "tiny-imagenet-hf")
        os.makedirs(cache_dir, exist_ok=True)

        # `zh-plus/tiny-imagenet` exposes splits "train" and "valid".
        hf_split = "train" if train else "valid"
        self._hf = load_dataset(HF_DATASET_NAME, split=hf_split, cache_dir=cache_dir)

        # Resolve column names (different HF mirrors use slightly different keys).
        self._image_key = "image" if "image" in self._hf.column_names else "img"
        if "label" in self._hf.column_names:
            self._label_key = "label"
        elif "labels" in self._hf.column_names:
            self._label_key = "labels"
        else:
            raise KeyError(
                f"Could not find a label column in {self._hf.column_names}; "
                f"expected 'label' or 'labels'."
            )

        # Set numpy/random seeds for reproducible imbalance subsampling.
        rng = np.random.RandomState(seed)

        all_labels = np.asarray(self._hf[self._label_key], dtype=np.int64)
        if train and imbalance_factor < 1.0:
            self.indices = self._build_imbalanced_indices(
                all_labels, imbalance_factor, rng,
            )
        else:
            self.indices = np.arange(len(all_labels), dtype=np.int64)

        self.targets = all_labels[self.indices].tolist()
        self.class_sample_counts = {
            int(c): int((all_labels[self.indices] == c).sum())
            for c in range(self.num_classes)
        }

    # ---- imbalance ---------------------------------------------------------

    @staticmethod
    def _build_imbalanced_indices(all_labels: np.ndarray, factor: float,
                                  rng: np.random.RandomState) -> np.ndarray:
        """Exponential-decay imbalance: n_c = N_max * factor ** (c / (C-1)).

        c=0 stays at N_max; c=C-1 ends at N_max*factor. Per-class subsets
        are sampled without replacement using a class-seeded RNG so the
        same `factor` always yields the same subset for the same `seed`.
        """
        C = int(all_labels.max()) + 1
        # Per-class count target. Use the natural N_max present in the source.
        n_max = int(np.bincount(all_labels).max())
        kept = []
        for c in range(C):
            n_c = int(round(n_max * (factor ** (c / max(C - 1, 1)))))
            n_c = max(n_c, 1)
            cls_idx = np.where(all_labels == c)[0]
            # Seed per-class so different factors don't shuffle each other.
            cls_rng = np.random.RandomState(rng.randint(0, 2**31 - 1) + c)
            cls_idx = cls_rng.permutation(cls_idx)[:n_c]
            kept.append(cls_idx)
        return np.concatenate(kept).astype(np.int64)

    # ---- standard Dataset API ---------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        row = self._hf[int(self.indices[idx])]
        img = row[self._image_key]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = int(row[self._label_key])
        return img, label

    # ---- helpers (matches CIFAR-LT / ImageNet-LT API) ---------------------

    def get_class_counts(self) -> dict:
        return dict(self.class_sample_counts)

    def get_head_mid_tail_split(self, n_groups: int = 3) -> dict:
        counts = self.get_class_counts()
        sorted_cls = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)
        per_group = len(sorted_cls) // n_groups
        groups: dict = {}
        names = ["head", "mid", "tail"]
        for g, name in enumerate(names[:n_groups]):
            start = g * per_group
            end = (g + 1) * per_group if g < n_groups - 1 else len(sorted_cls)
            groups[name] = sorted_cls[start:end]
        return groups


# ---------------------------------------------------------------------------
# Transform factory
# ---------------------------------------------------------------------------

def make_tiny_imagenet_lt_transform(image_size: int = NATIVE_RESOLUTION,
                                    train: bool = True):
    """Tiny ImageNet transforms.

    Native resolution is 64×64. We optionally resize (rare, for ablations
    that share the CIFAR backbone), then normalize to [-1, 1] to match the
    HAE pipeline convention.
    """
    ops: list = []
    if image_size != NATIVE_RESOLUTION:
        # Resize only if the user explicitly asked for a different size.
        ops.append(transforms.Resize(image_size))
        ops.append(transforms.CenterCrop(image_size))
    if train:
        ops.append(transforms.RandomCrop(image_size, padding=4,
                                          padding_mode="reflect"))
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return transforms.Compose(ops)
