#!/usr/bin/env python
"""Export hyperbolic embeddings from a trained HAE-CIFAR checkpoint.

Outputs z_hyp.pt  (N, feature_size)  and labels.pt  (N,) for consumption by
the RFM HyperbolicImages dataset.

Usage:
    conda activate dl
    cd HAE
    python scripts/export_embeddings.py \\
        --checkpoint experiments/cifar10_c1/checkpoints/best_model.pt \\
        --dataset cifar10 \\
        --output_dir ./exported_embeddings
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.hae_cifar import HAECifar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from datasets.cifar_lt import CIFAR10LT, CIFAR100LT, make_cifar_lt_transform


def parse_args():
    p = argparse.ArgumentParser(description="Export HAE hyperbolic embeddings")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100"])
    p.add_argument("--imbalance_factor", type=float, default=0.01)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split", type=str, default="train",
                   choices=["train", "test"],
                   help="Which split to export embeddings for")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load checkpoint --------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_args = ckpt.get("args", {})

    num_classes = 10 if args.dataset == "cifar10" else 100
    model = HAECifar(
        num_classes=num_classes,
        latent_dim=saved_args.get("latent_dim", 512),
        feature_size=saved_args.get("feature_size", 512),
        curvature=saved_args.get("curvature", -1.0),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ---- Dataset (no augmentation) ----------------------------------------
    cls = CIFAR10LT if args.dataset == "cifar10" else CIFAR100LT
    tf = make_cifar_lt_transform(image_size=32, train=False)  # no augmentation
    is_train = args.split == "train"
    ds = cls(root=args.data_root, imbalance_factor=args.imbalance_factor,
             train=is_train, transform=tf, download=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    # ---- Extract embeddings -----------------------------------------------
    all_z_hyp = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _, _, z_hyp, _, _ = model(images)
            all_z_hyp.append(z_hyp.cpu())
            all_labels.append(labels)

    z_hyp = torch.cat(all_z_hyp, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # ---- Validate ---------------------------------------------------------
    norms = z_hyp.norm(dim=-1)
    frac_inside = (norms < 1.0).float().mean().item()
    print(f"Exported {z_hyp.shape[0]} embeddings  dim={z_hyp.shape[1]}")
    print(f"  Norm stats: min={norms.min():.4f}  max={norms.max():.4f}  "
          f"mean={norms.mean():.4f}")
    print(f"  Fraction inside Poincaré ball: {frac_inside*100:.1f}%")

    # ---- Save -------------------------------------------------------------
    z_path = os.path.join(args.output_dir, "z_hyp.pt")
    l_path = os.path.join(args.output_dir, "labels.pt")
    torch.save(z_hyp, z_path)
    torch.save(labels, l_path)
    print(f"  Saved: {z_path}  ({z_hyp.shape})")
    print(f"  Saved: {l_path}  ({labels.shape})")


if __name__ == "__main__":
    main()
