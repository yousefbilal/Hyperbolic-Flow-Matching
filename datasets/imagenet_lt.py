"""ImageNet-LT dataset wrapper.

Expects the pre-arranged folder layout::

    <root>/imagenet-lt/ImageDataset_256/
        train/<class_idx>/*.png
        test/<class_idx>/*.png

Class folders are integer-named (0..999). We use ImageFolder and expose
`get_class_counts()` + `get_head_mid_tail_split()` so the same sampler /
class-weighting helpers work as for CIFAR-LT.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Optional

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageNetLT(Dataset):
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform=None,
        image_size: int = 256,
        subdir: str = "imagenet-lt/ImageDataset_256",
    ):
        super().__init__()
        split = "train" if train else "test"
        data_dir = os.path.join(root, subdir, split)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"ImageNet-LT split not found: {data_dir}")

        self.image_size = image_size
        self.transform = transform or make_imagenet_lt_transform(image_size, train)

        # Torchvision's ImageFolder sorts classes lexicographically ("0",
        # "1", "10", "100", …) which is NOT integer order. Since the
        # class folders are integer names, sort numerically so class_to_idx
        # is stable and interpretable.
        self._folder = ImageFolder(data_dir, transform=self.transform)
        class_names = sorted(self._folder.classes, key=lambda s: int(s))
        remap = {self._folder.class_to_idx[name]: i
                 for i, name in enumerate(class_names)}
        # rewrite samples + targets with numeric-sorted class ids
        self.samples = [(p, remap[old_t]) for p, old_t in self._folder.samples]
        self.targets = [remap[t] for t in self._folder.targets]
        self.num_classes = len(class_names)

        counts = Counter(self.targets)
        self.class_sample_counts = {c: counts[c] for c in range(self.num_classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self._folder.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # ---- sampler/weighting helpers (matches CIFAR-LT API) -----------------

    def get_class_counts(self) -> dict:
        return dict(self.class_sample_counts)

    def get_head_mid_tail_split(self, n_groups: int = 3) -> dict:
        counts = self.get_class_counts()
        sorted_cls = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)
        per_group = len(sorted_cls) // n_groups
        groups = {}
        names = ["head", "mid", "tail"]
        for g, name in enumerate(names[:n_groups]):
            start = g * per_group
            end = (g + 1) * per_group if g < n_groups - 1 else len(sorted_cls)
            groups[name] = sorted_cls[start:end]
        return groups


# ---------------------------------------------------------------------------
# Transforms — match SD-VAE preprocessing: [-1, 1], RGB, 256x256 square crop.
# ---------------------------------------------------------------------------

def make_imagenet_lt_transform(image_size: int = 256, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
