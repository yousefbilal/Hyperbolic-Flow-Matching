
# --------------------
# LDAM-DRW-style CIFAR-LT datasets with same API as before
# --------------------

import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from PIL import Image

class _CIFARLTBase(Dataset):
    """Base class for CIFAR-LT datasets with LDAM-DRW-style imbalance, but same API as before."""
    _NUM_CLASSES: int = 0
    _TORCHVISION_CLASS = None


    def __init__(
        self,
        root: str = "./data",
        imbalance_factor: float = 0.01,
        train: bool = True,
        transform=None,
        target_transform=None,
        download: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.imbalance_factor = imbalance_factor
        self.root = root
        self.download = download
        self.seed = seed

        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Load full CIFAR dataset
        base = self._TORCHVISION_CLASS(root=self.root, train=self.train, download=self.download)
        self.data = base.data  # numpy array (N, 32, 32, 3)
        self.targets = np.array(base.targets, dtype=np.int64)

        if self.train:
            img_num_list = self.get_img_num_per_cls(self._NUM_CLASSES, 'exp', self.imbalance_factor)
            self.gen_imbalanced_data(img_num_list)
        else:
            self.data_format_transform()

        # Build class_sample_counts for get_class_counts()
        targets_np = np.array([d['category_id'] for d in self.all_info], dtype=np.int64)
        self.class_sample_counts = {i: int(np.sum(targets_np == i)) for i in range(self._NUM_CLASSES)}

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        img = self.all_info[idx]['image']
        label = self.all_info[idx]['category_id']
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

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

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            # Use the same seed for each class for reproducibility
            rng = np.random.RandomState(self.seed + int(the_class))
            idx = rng.permutation(idx)
            selec_idx = idx[:the_img_num]
            for img in self.data[selec_idx, ...]:
                new_data.append({
                    'image': img,
                    'category_id': the_class
                })
        self.all_info = new_data

    def data_format_transform(self):
        new_data = []
        targets_np = np.array(self.targets, dtype=np.int64)
        assert len(targets_np) == len(self.data)
        for i in range(len(self.data)):
            new_data.append({
                'image': self.data[i],
                'category_id': targets_np[i],
            })
        self.all_info = new_data


class CIFAR10LT(_CIFARLTBase):
    _NUM_CLASSES = 10
    _TORCHVISION_CLASS = torchvision.datasets.CIFAR10


class CIFAR100LT(_CIFARLTBase):
    _NUM_CLASSES = 100
    _TORCHVISION_CLASS = torchvision.datasets.CIFAR100


# ---------------------------------------------------------------------------
# Transform factory
# ---------------------------------------------------------------------------

def make_cifar_lt_transform(image_size: int = 32, train: bool = True):
    """Standard CIFAR transforms for H-SiT (pixel-to-manifold pipeline).

    Images are normalized to [-1, 1] (mean=0.5, std=0.5). No VAE — raw
    32x32 RGB pixels are patchified and lifted onto the hyperboloid.
    """
    if train:
        return transforms.Compose(
            [
                # transforms.RandomApply([transforms.RandomCrop(image_size, padding=4, padding_mode='reflect')], p=0.3),
                transforms.RandomCrop(image_size, padding=4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )


# Alias for backward compatibility with existing imports
make_cifar100_lt_transform = make_cifar_lt_transform
