#!/usr/bin/env python
"""Export hyperbolic (or Euclidean) embeddings from a trained HAE checkpoint.

Outputs:
  - z_hyp.pt  (N, feature_size)
  - labels.pt (N,)
  - meta.pt   dict with num_classes, dataset, curvature

Supports CIFAR-10/100-LT (HAECifar) and ImageNet-LT (HAEImageNet w/ SD-VAE).
When the saved curvature is ~0, the ball-interior check is skipped since the
embeddings live in Euclidean space.
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

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


EUCLIDEAN_EPS = 1e-6


def parse_args():
    p = argparse.ArgumentParser(description="Export HAE embeddings")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100", "imagenet_lt",
                            "tiny_imagenet_lt"])
    p.add_argument("--imbalance_factor", type=float, default=0.01)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split", type=str, default="train",
                   choices=["train", "test"])
    return p.parse_args()


def _resolve_image_size(saved_args, dataset, default):
    """Backwards-compat: old checkpoints stored image_size=None when the
    --image_size CLI arg wasn't passed. `.get(k, default)` only kicks in
    when the key is missing, not when it's None — so use `or`."""
    return int(saved_args.get("image_size") or default)


def build_dataset(args, saved_args):
    is_train = args.split == "train"
    if args.dataset in ("cifar10", "cifar100"):
        cls = CIFAR10LT if args.dataset == "cifar10" else CIFAR100LT
        img_sz = _resolve_image_size(saved_args, args.dataset, 32)
        tf = make_cifar_lt_transform(image_size=img_sz, train=False)
        ds = cls(root=args.data_root, imbalance_factor=args.imbalance_factor,
                 train=is_train, transform=tf, download=True)
        num_classes = 10 if args.dataset == "cifar10" else 100
        return ds, num_classes
    if args.dataset == "tiny_imagenet_lt":
        img_sz = _resolve_image_size(saved_args, args.dataset, TINY_IMAGENET_RESOLUTION)
        tf = make_tiny_imagenet_lt_transform(image_size=img_sz, train=False)
        # Train split honours the imbalance factor; test split is balanced.
        imb = args.imbalance_factor if is_train else 1.0
        ds = TinyImageNetLT(root=args.data_root, imbalance_factor=imb,
                            train=is_train, transform=tf, image_size=img_sz)
        return ds, ds.num_classes
    # imagenet_lt
    tf = make_imagenet_lt_transform(image_size=256, train=False)
    ds = ImageNetLT(root=args.data_root, train=is_train, transform=tf)
    return ds, ds.num_classes


def build_model(args, saved_args, num_classes, device):
    curvature = saved_args.get("curvature") or -1.0
    latent_dim = saved_args.get("latent_dim") or 512
    feature_size = saved_args.get("feature_size") or 512
    variational = float(saved_args.get("kl_lambda") or 0.0) > 0.0

    # Default image_size depends on dataset; legacy CIFAR ckpts had no key.
    default_img = {"imagenet_lt": 256, "tiny_imagenet_lt": 64}.get(args.dataset, 32)
    image_size = _resolve_image_size(saved_args, args.dataset, default_img)
    # proj_hidden_dims may be missing (old ckpt) or None — coerce to ().
    proj_hidden = tuple(saved_args.get("proj_hidden_dims") or ())

    # Resolve backbone: prefer the explicitly saved value; fall back to
    # the legacy default for older checkpoints.
    bb = saved_args.get("encoder_backbone")
    if bb is None:
        bb = "sd_vae" if args.dataset == "imagenet_lt" else "cnn_cifar"

    if bb in ("sd_vae", "taesd"):
        from models.hae_imagenet import SD_VAE_NAME, TAESD_NAME
        is_tiny = bb == "taesd"
        model = HAEImageNet(
            num_classes=num_classes, latent_dim=latent_dim,
            feature_size=feature_size, curvature=curvature,
            vae_name=TAESD_NAME if is_tiny else SD_VAE_NAME,
            tiny_vae=is_tiny, image_size=image_size,
            proj_hidden_dims=proj_hidden,
        )
    else:   # cnn_cifar
        model = HAECifar(num_classes=num_classes, latent_dim=latent_dim,
                         feature_size=feature_size, curvature=curvature,
                         variational=variational, image_size=image_size)
    model = model.to(device)
    return model, curvature


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_args = ckpt.get("args", {})

    ds, num_classes = build_dataset(args, saved_args)
    model, curvature = build_model(args, saved_args, num_classes, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    all_z_hyp, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # 6-tuple now: (recon, logits, z_hyp, z_euc, z_euc_dec, kl).
            # In VAE mode, model.eval() makes the encoder return μ (deterministic),
            # so exported z_hyp is reproducible across calls.
            _, _, z_hyp, _, _, _ = model(images)
            all_z_hyp.append(z_hyp.cpu())
            all_labels.append(labels)

    z_hyp = torch.cat(all_z_hyp, dim=0)
    labels = torch.cat(all_labels, dim=0)

    is_euclidean = abs(float(curvature)) < EUCLIDEAN_EPS
    norms = z_hyp.norm(dim=-1)
    print(f"Exported {z_hyp.shape[0]} embeddings  dim={z_hyp.shape[1]}")
    print(f"  Norm stats: min={norms.min():.4f}  max={norms.max():.4f}  "
          f"mean={norms.mean():.4f}")
    if is_euclidean:
        print("  (Euclidean mode — skipping Poincaré-ball check)")
    else:
        frac_inside = (norms < 1.0).float().mean().item()
        print(f"  Fraction inside Poincaré ball: {frac_inside*100:.1f}%")

    z_path = os.path.join(args.output_dir, "z_hyp.pt")
    l_path = os.path.join(args.output_dir, "labels.pt")
    m_path = os.path.join(args.output_dir, "meta.pt")
    torch.save(z_hyp, z_path)
    torch.save(labels, l_path)
    torch.save({"num_classes": int(num_classes),
                "dataset": args.dataset,
                "curvature": float(curvature),
                "feature_size": int(z_hyp.shape[1])},
               m_path)
    print(f"  Saved: {z_path}  ({z_hyp.shape})")
    print(f"  Saved: {l_path}  ({labels.shape})")
    print(f"  Saved: {m_path}")


if __name__ == "__main__":
    main()
