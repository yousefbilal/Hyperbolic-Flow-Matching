#!/usr/bin/env python
"""End-to-end CIFAR image generation via Riemannian Flow Matching.

Pipeline:
    1. Load trained RFM → sample z_hyp on the Poincaré ball
    2. Load trained HAE → logmap0(z_hyp) → CNN decoder → images
    3. Save images as a grid / individual PNGs

Usage:
    conda activate dl
    python generate_cifar.py \\
        --rfm_checkpoint outputs/runs/.../checkpoints/last.ckpt \\
        --hae_checkpoint ../HAE/experiments/cifar10_c1/checkpoints/best_model.pt \\
        --n_samples 64 \\
        --output_dir ./generated_cifar
"""

import argparse
import os
import sys

import torch
import geoopt.manifolds.stereographic.math as gmath
from torchvision.utils import save_image

# Add project roots to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "HAE"))

from manifm.eval_utils import load_model


def parse_args():
    p = argparse.ArgumentParser(description="Generate CIFAR images from RFM + HAE")
    p.add_argument("--rfm_checkpoint", type=str, required=True,
                   help="Path to the RFM Lightning checkpoint (.ckpt)")
    p.add_argument("--hae_checkpoint", type=str, required=True,
                   help="Path to the HAE-CIFAR checkpoint (.pt)")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--output_dir", type=str, default="./generated_cifar")
    p.add_argument("--curvature", type=float, default=-1.0,
                   help="Must match the curvature used during HAE training")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--nrow", type=int, default=8,
                   help="Number of images per row in the saved grid")
    return p.parse_args()


def load_hae_decoder(checkpoint_path, device):
    """Load only the decoder portion of a trained HAE-CIFAR model."""
    from models.hae_cifar import HAECifar

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    saved_args = ckpt.get("args", {})

    num_classes = saved_args.get("num_classes", 10)
    # Infer num_classes from the MLR weight shape if available
    sd = ckpt["state_dict"]
    if "mlr.a_vals" in sd:
        num_classes = sd["mlr.a_vals"].shape[0]

    model = HAECifar(
        num_classes=num_classes,
        latent_dim=saved_args.get("latent_dim", 512),
        feature_size=saved_args.get("feature_size", 512),
        curvature=saved_args.get("curvature", -1.0),
    )
    model.load_state_dict(sd)
    model = model.to(device).eval()
    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    curvature = torch.tensor(args.curvature, dtype=torch.float32)

    # ---- 1. Load RFM and generate hyperbolic latents ----------------------
    print(f"Loading RFM from: {args.rfm_checkpoint}")
    cfg, rfm_model = load_model(args.rfm_checkpoint)
    rfm_model = rfm_model.to(device).eval()

    print(f"Sampling {args.n_samples} latents on the Poincaré ball...")
    with torch.no_grad():
        z_hyp = rfm_model.sample(args.n_samples, device=device)

    norms = z_hyp.norm(dim=-1)
    print(f"  z_hyp shape: {z_hyp.shape}")
    print(f"  Norm stats: min={norms.min():.4f}  max={norms.max():.4f}  "
          f"mean={norms.mean():.4f}")

    # ---- 2. Load HAE decoder and produce images ---------------------------
    print(f"Loading HAE decoder from: {args.hae_checkpoint}")
    hae_model = load_hae_decoder(args.hae_checkpoint, device)

    with torch.no_grad():
        z_euc = gmath.logmap0(z_hyp.float(), k=curvature)
        images = hae_model.decoder(z_euc)   # (N, 3, 32, 32) in [-1, 1]

    # ---- 3. Save ----------------------------------------------------------
    # Grid
    grid_path = os.path.join(args.output_dir, "generated_grid.png")
    save_image(images, grid_path, nrow=args.nrow, normalize=True,
               value_range=(-1, 1))
    print(f"  Saved grid → {grid_path}")

    # Individual images
    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    for i in range(images.size(0)):
        save_image(images[i], os.path.join(img_dir, f"{i:04d}.png"),
                   normalize=True, value_range=(-1, 1))
    print(f"  Saved {images.size(0)} individual images → {img_dir}")

    # Save raw latents for diagnostic plots
    torch.save(z_hyp.cpu(), os.path.join(args.output_dir, "z_hyp_generated.pt"))
    print("Done.")


if __name__ == "__main__":
    main()
