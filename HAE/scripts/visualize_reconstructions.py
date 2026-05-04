#!/usr/bin/env python
"""Visualize HAE reconstructions on validation images.

Loads a trained HAE checkpoint, runs it on the validation/test split, and
saves a comparison grid: original on top, reconstruction below.

Two views written to --output_dir:

    recon_grid_random.png      — N random val samples
    recon_grid_per_class.png   — `--per_class` samples from each class
                                 (head and tail visible side-by-side)

Usage:
    python scripts/visualize_reconstructions.py \
        --checkpoint /path/to/best_model.pt \
        --dataset cifar10 --imbalance_factor 0.01 \
        --data_root /path/to/data \
        --output_dir /path/to/recons \
        --n_random 32 --per_class 4
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.hae_cifar import HAECifar
from models.hae_imagenet import HAEImageNet

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from lt_datasets.cifar_lt import CIFAR10LT, CIFAR100LT, make_cifar_lt_transform
from lt_datasets.imagenet_lt import ImageNetLT, make_imagenet_lt_transform
from lt_datasets.tiny_imagenet_lt import (
    TinyImageNetLT, make_tiny_imagenet_lt_transform,
    NATIVE_RESOLUTION as TINY_IMAGENET_RESOLUTION,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100", "imagenet_lt",
                            "tiny_imagenet_lt"])
    p.add_argument("--imbalance_factor", type=float, default=0.01)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--n_random", type=int, default=32,
                   help="How many random val samples to show.")
    p.add_argument("--per_class", type=int, default=4,
                   help="How many samples per class to show in the per-class grid.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _resolve_image_size(saved_args, dataset, default):
    """Old checkpoints stored image_size=None when --image_size wasn't passed.
    `.get(k, default)` doesn't help with None values, so use `or`."""
    return int(saved_args.get("image_size") or default)


def build_dataset(args, saved_args):
    if args.dataset in ("cifar10", "cifar100"):
        cls = CIFAR10LT if args.dataset == "cifar10" else CIFAR100LT
        img_sz = _resolve_image_size(saved_args, args.dataset, 32)
        tf = make_cifar_lt_transform(image_size=img_sz, train=False)
        ds = cls(root=args.data_root, imbalance_factor=args.imbalance_factor,
                 train=False, transform=tf, download=True)
        num_classes = 10 if args.dataset == "cifar10" else 100
    elif args.dataset == "tiny_imagenet_lt":
        img_sz = _resolve_image_size(saved_args, args.dataset, TINY_IMAGENET_RESOLUTION)
        tf = make_tiny_imagenet_lt_transform(image_size=img_sz, train=False)
        # Val split is balanced per the standard LT convention.
        ds = TinyImageNetLT(root=args.data_root, imbalance_factor=1.0,
                            train=False, transform=tf, image_size=img_sz)
        num_classes = ds.num_classes
    else:    # imagenet_lt
        img_sz = _resolve_image_size(saved_args, args.dataset, 256)
        tf = make_imagenet_lt_transform(image_size=img_sz, train=False)
        ds = ImageNetLT(root=args.data_root, train=False, transform=tf)
        num_classes = ds.num_classes
    return ds, num_classes


def build_model(saved_args, num_classes, device, dataset):
    # NB: don't use `or default` for curvature — curvature=0 is the legitimate
    # Euclidean baseline and is falsy, so `0 or -1.0` evaluates to -1.0.
    _c = saved_args.get("curvature")
    curvature = float(_c) if _c is not None else -1.0
    latent_dim = saved_args.get("latent_dim") or 512
    feature_size = saved_args.get("feature_size") or 512
    variational = float(saved_args.get("kl_lambda") or 0.0) > 0.0

    default_img = {"imagenet_lt": 256, "tiny_imagenet_lt": 64}.get(dataset, 32)
    image_size = _resolve_image_size(saved_args, dataset, default_img)

    # Resolve backbone — saved value wins; legacy default per dataset.
    bb = saved_args.get("encoder_backbone")
    if bb is None:
        bb = "sd_vae" if dataset == "imagenet_lt" else "cnn_cifar"

    if bb in ("sd_vae", "taesd"):
        from models.hae_imagenet import SD_VAE_NAME, TAESD_NAME
        is_tiny = bb == "taesd"
        m = HAEImageNet(num_classes=num_classes, latent_dim=latent_dim,
                        feature_size=feature_size, curvature=curvature,
                        vae_name=TAESD_NAME if is_tiny else SD_VAE_NAME,
                        tiny_vae=is_tiny, image_size=image_size)
    else:    # cnn_cifar
        m = HAECifar(num_classes=num_classes, latent_dim=latent_dim,
                     feature_size=feature_size, curvature=curvature,
                     variational=variational, image_size=image_size)
    return m.to(device).eval()


def make_pair_grid(originals, recons, nrow):
    """Top row: originals, bottom row: recons. Returns a single tensor grid."""
    # Interleave by row: originals on even rows, recons on odd rows
    n = originals.size(0)
    assert n == recons.size(0)
    rows_orig = (n + nrow - 1) // nrow
    pairs = []
    for r in range(rows_orig):
        s, e = r * nrow, min((r + 1) * nrow, n)
        pairs.append(originals[s:e])
        pairs.append(recons[s:e])
    stacked = torch.cat(pairs, dim=0)
    return make_grid(stacked, nrow=nrow, normalize=True, value_range=(-1, 1))


@torch.no_grad()
def run_model(model, images, device):
    images = images.to(device)
    out = model(images)
    recon = out[0]                       # 6-tuple: (recon, logits, z_hyp, z_euc, z_euc_dec, kl)
    return images.cpu(), recon.cpu()


def random_grid(model, ds, args):
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(ds), generator=g)[:args.n_random].tolist()
    sub = Subset(ds, idx)
    loader = DataLoader(sub, batch_size=args.n_random, shuffle=False, num_workers=0)
    images, _ = next(iter(loader))
    orig, recon = run_model(model, images, args.device)
    nrow = min(args.n_random, 8)
    return make_pair_grid(orig, recon, nrow=nrow)


def per_class_grid(model, ds, num_classes, args):
    targets = (ds.targets if hasattr(ds, "targets")
               else [ds[i][1] for i in range(len(ds))])
    targets = np.asarray(targets)
    g = np.random.default_rng(args.seed)
    chosen = []
    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        if len(idx_c) == 0:
            print(f"  class {c}: 0 samples in val — skipping")
            continue
        pick = g.choice(idx_c, size=min(args.per_class, len(idx_c)), replace=False)
        chosen.extend(int(i) for i in pick)
    if not chosen:
        return None
    sub = Subset(ds, chosen)
    loader = DataLoader(sub, batch_size=len(chosen), shuffle=False, num_workers=0)
    images, _ = next(iter(loader))
    orig, recon = run_model(model, images, args.device)
    return make_pair_grid(orig, recon, nrow=args.per_class)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_args = ckpt.get("args", {})

    ds, num_classes = build_dataset(args, saved_args)
    print(f"Validation set: {len(ds)} images across {num_classes} classes")

    model = build_model(saved_args, num_classes, device, args.dataset)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Model loaded ({type(model).__name__}, "
          f"curvature={saved_args.get('curvature', '?')}, "
          f"variational={float(saved_args.get('kl_lambda', 0.0)) > 0.0})")

    # Random grid
    print("\nRendering random grid...")
    grid = random_grid(model, ds, args)
    out_random = os.path.join(args.output_dir, "recon_grid_random.png")
    save_image(grid, out_random)
    print(f"  saved → {out_random}")

    # Per-class grid (head/tail visualization)
    print("\nRendering per-class grid...")
    grid_pc = per_class_grid(model, ds, num_classes, args)
    if grid_pc is not None:
        out_pc = os.path.join(args.output_dir, "recon_grid_per_class.png")
        save_image(grid_pc, out_pc)
        print(f"  saved → {out_pc}")
        print("  Layout: each pair of rows is a class — top=original, bottom=recon, "
              f"{args.per_class} samples wide.")
    print("\nDone.")


if __name__ == "__main__":
    main()
