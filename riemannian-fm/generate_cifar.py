#!/usr/bin/env python
"""End-to-end image generation via RFM + HAE decoder.

Dispatches on the HAE checkpoint's saved `dataset`:
  - cifar10/cifar100 → HAECifar's CNN decoder (32x32 images)
  - imagenet_lt      → HAEImageNet's proj_dec + frozen SD-VAE decode (256x256)

Skips logmap0 when the saved curvature is ~0 (Euclidean baseline).
Supports CFG-conditional sampling via --class_id and --cfg_scale.
"""

import argparse
import os
import sys

import torch
import geoopt.manifolds.stereographic.math as gmath
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "HAE"))

from manifm.eval_utils import load_model


EUCLIDEAN_EPS = 1e-6


def parse_args():
    p = argparse.ArgumentParser(description="Generate images from RFM + HAE")
    p.add_argument("--rfm_checkpoint", type=str, required=True)
    p.add_argument("--hae_checkpoint", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--output_dir", type=str, default="./generated")
    p.add_argument("--curvature", type=float, default=-1.0,
                   help="Must match the curvature used during HAE training")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--nrow", type=int, default=8)
    p.add_argument("--class_id", type=str, default=None,
                   help="Single int or comma list of class indices "
                        "(length 1 or n_samples). Omit for unconditional.")
    p.add_argument("--cfg_scale", type=float, default=1.0)
    return p.parse_args()


def load_hae(checkpoint_path, device):
    """Load the full HAE model (CIFAR or ImageNet) so we have encoder+decoder."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    saved_args = ckpt.get("args", {})
    dataset = saved_args.get("dataset", "cifar10")
    curvature = float(saved_args.get("curvature", -1.0))
    latent_dim = saved_args.get("latent_dim", 512)
    feature_size = saved_args.get("feature_size", 512)

    sd = ckpt["state_dict"]

    if dataset in ("cifar10", "cifar100"):
        from models.hae_cifar import HAECifar
        num_classes = 10 if dataset == "cifar10" else 100
        model = HAECifar(num_classes=num_classes, latent_dim=latent_dim,
                         feature_size=feature_size, curvature=curvature)
    elif dataset == "imagenet_lt":
        from models.hae_imagenet import HAEImageNet
        # Infer num_classes from checkpoint. Euclidean head → classifier.weight;
        # hyperbolic head → mlr.a_vals.
        num_classes = 1000
        if "head.classifier.weight" in sd:
            num_classes = sd["head.classifier.weight"].shape[0]
        elif "head.mlr.a_vals" in sd:
            num_classes = sd["head.mlr.a_vals"].shape[0]
        model = HAEImageNet(num_classes=num_classes, latent_dim=latent_dim,
                            feature_size=feature_size, curvature=curvature)
    else:
        raise ValueError(f"Unknown HAE dataset: {dataset}")

    model.load_state_dict(sd)
    model = model.to(device).eval()
    return model, dataset, curvature


def decode(model, dataset, z_euc_dec):
    """Map Euclidean feature back to image space (32x32 for CIFAR, 256x256 for ImageNet)."""
    if dataset in ("cifar10", "cifar100"):
        return model.decoder(z_euc_dec)
    # imagenet_lt
    B = z_euc_dec.shape[0]
    z_flat = model.proj_dec(z_euc_dec)
    z_spatial = z_flat.reshape(B, model.channels, model.spatial, model.spatial)
    return model.vae.decode(z_spatial)


def parse_class_id(s, n_samples, device):
    if s is None:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    ids = [int(p) for p in parts]
    if len(ids) == 1:
        return torch.full((n_samples,), ids[0], dtype=torch.long, device=device)
    assert len(ids) == n_samples, (
        f"class_id list length {len(ids)} must match n_samples {n_samples}")
    return torch.tensor(ids, dtype=torch.long, device=device)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    curvature = torch.tensor(args.curvature, dtype=torch.float32)
    is_euclidean = abs(float(args.curvature)) < EUCLIDEAN_EPS

    # ---- 1. Load RFM and sample latents -----------------------------------
    print(f"Loading RFM from: {args.rfm_checkpoint}")
    cfg, rfm_model = load_model(args.rfm_checkpoint)
    rfm_model = rfm_model.to(device).eval()

    labels = parse_class_id(args.class_id, args.n_samples, device)

    print(f"Sampling {args.n_samples} latents "
          f"(class_id={args.class_id}, cfg_scale={args.cfg_scale})...")
    with torch.no_grad():
        z_hyp = rfm_model.sample(args.n_samples, device=device,
                                 labels=labels, cfg_scale=args.cfg_scale)

    norms = z_hyp.norm(dim=-1)
    print(f"  z_hyp shape: {z_hyp.shape}  "
          f"norm min={norms.min():.4f} max={norms.max():.4f} "
          f"mean={norms.mean():.4f}")

    # ---- 2. Load HAE decoder and produce images ---------------------------
    print(f"Loading HAE from: {args.hae_checkpoint}")
    hae_model, dataset, saved_k = load_hae(args.hae_checkpoint, device)
    saved_is_euclidean = abs(saved_k) < EUCLIDEAN_EPS
    if saved_is_euclidean != is_euclidean:
        print(f"  WARNING: --curvature={args.curvature} vs saved curvature "
              f"{saved_k}. Using saved flag for logmap0 bypass.")

    with torch.no_grad():
        z = z_hyp.float()
        if not saved_is_euclidean:
            z = gmath.logmap0(z, k=curvature)
        images = decode(hae_model, dataset, z)

    # ---- 3. Save ----------------------------------------------------------
    grid_path = os.path.join(args.output_dir, "generated_grid.png")
    save_image(images, grid_path, nrow=args.nrow, normalize=True,
               value_range=(-1, 1))
    print(f"  Saved grid → {grid_path}")

    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    for i in range(images.size(0)):
        save_image(images[i], os.path.join(img_dir, f"{i:04d}.png"),
                   normalize=True, value_range=(-1, 1))
    print(f"  Saved {images.size(0)} individual images → {img_dir}")

    torch.save(z_hyp.cpu(), os.path.join(args.output_dir, "z_hyp_generated.pt"))
    print("Done.")


if __name__ == "__main__":
    main()
