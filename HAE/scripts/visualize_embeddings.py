#!/usr/bin/env python
"""Visualize hyperbolic (or Euclidean) HAE embeddings as a 2D projection.

For hyperbolic embeddings (curvature < 0), project from the Poincaré ball
to a 2D Poincaré disk via:
    1. logmap0(z_hyp)            -> Euclidean tangent space at origin
    2. PCA(2)                     -> 2D tangent vectors
    3. expmap0(2D vec, k_2d=k)    -> 2D Poincaré disk

The 2D disk's boundary is at radius 1/sqrt(|k|), drawn as a circle.
Points are coloured by class. Optionally also plots per-class statistics:
    - mean radius
    - per-class centroid (computed in tangent space, mapped to disk)

For Euclidean embeddings, falls back to plain PCA(2) and a square plot
(no disk boundary).

Usage:
    python scripts/visualize_embeddings.py \
        --embeddings /path/to/embeddings/z_hyp.pt \
        --labels    /path/to/embeddings/labels.pt \
        --meta      /path/to/embeddings/meta.pt \
        --output    /path/to/embeddings/disk.png
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import geoopt.manifolds.stereographic.math as gmath


def parse_args():
    p = argparse.ArgumentParser(description="Visualize HAE embeddings on a 2D Poincaré disk")
    p.add_argument("--embeddings", type=str, required=True,
                   help="Path to z_hyp.pt")
    p.add_argument("--labels", type=str, required=True,
                   help="Path to labels.pt")
    p.add_argument("--meta", type=str, default=None,
                   help="Path to meta.pt (used to read curvature). "
                        "If omitted, --curvature must be passed.")
    p.add_argument("--curvature", type=float, default=None,
                   help="Override curvature (negative for hyperbolic, 0 for Euclidean).")
    p.add_argument("--output", type=str, required=True,
                   help="Where to save the figure (.png).")
    p.add_argument("--max_per_class", type=int, default=200,
                   help="Subsample at most N points per class for readability.")
    p.add_argument("--class_names", type=str, default=None,
                   help="Comma-separated class names (length = num_classes). "
                        "Defaults to integer indices.")
    p.add_argument("--no_centroids", action="store_true",
                   help="Skip the per-class centroid markers.")
    return p.parse_args()


def load_inputs(args):
    z = torch.load(args.embeddings, map_location="cpu").float()
    labels = torch.load(args.labels, map_location="cpu").long()
    if z.ndim != 2:
        z = z.reshape(z.shape[0], -1)
    if args.meta and os.path.exists(args.meta):
        meta = torch.load(args.meta, map_location="cpu")
        curvature = float(meta.get("curvature", -1.0))
    elif args.curvature is not None:
        curvature = float(args.curvature)
    else:
        raise SystemExit("Need --meta or --curvature to know the geometry")
    return z, labels, curvature


def project_2d(z: torch.Tensor, curvature: float):
    """Return (xy, is_euclidean) where xy is shape (N, 2)."""
    is_euclidean = abs(curvature) < 1e-6
    if is_euclidean:
        # Plain PCA on Euclidean embeddings
        zc = z - z.mean(0, keepdim=True)
        # SVD-based PCA, keep top 2 components
        u, s, vt = torch.linalg.svd(zc, full_matrices=False)
        xy = (zc @ vt[:2].T).numpy()
        return xy, True

    # Hyperbolic: logmap0 -> PCA(2) -> expmap0 (in 2D with same k)
    k = torch.tensor(curvature, dtype=z.dtype)
    tangent = gmath.logmap0(z, k=k)
    tc = tangent - tangent.mean(0, keepdim=True)
    u, s, vt = torch.linalg.svd(tc, full_matrices=False)
    xy_tangent = (tc @ vt[:2].T)
    # Re-embed into the 2D Poincaré disk (same curvature)
    xy_ball = gmath.expmap0(xy_tangent, k=k).numpy()
    return xy_ball, False


def class_colour_map(num_classes: int):
    cmap = plt.get_cmap("tab10" if num_classes <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(num_classes)]


def main():
    args = parse_args()
    z, labels, curvature = load_inputs(args)
    print(f"Loaded {z.shape[0]} embeddings, dim={z.shape[1]}, curvature={curvature}")
    print(f"  global ‖z‖: min={z.norm(dim=-1).min():.3f}  "
          f"mean={z.norm(dim=-1).mean():.3f}  max={z.norm(dim=-1).max():.3f}")

    num_classes = int(labels.max().item()) + 1

    # Subsample per class for readability
    chosen_idx = []
    rng = np.random.default_rng(0)
    for c in range(num_classes):
        idx = (labels == c).nonzero(as_tuple=True)[0].numpy()
        if len(idx) > args.max_per_class:
            idx = rng.choice(idx, size=args.max_per_class, replace=False)
        chosen_idx.append(idx)
    sub_idx = np.concatenate(chosen_idx)
    z_sub = z[sub_idx]
    labels_sub = labels[sub_idx]

    xy, is_euclidean = project_2d(z, curvature)         # use FULL set for PCA fit
    xy_sub = xy[sub_idx]                                 # but plot only subset

    # Per-class centroids (tangent-space mean for hyperbolic, plain mean for Euclidean)
    centroids_2d = np.zeros((num_classes, 2))
    if not args.no_centroids:
        for c in range(num_classes):
            mask = (labels == c).numpy()
            if mask.any():
                centroids_2d[c] = xy[mask].mean(0)

    # Class names
    if args.class_names:
        names = [s.strip() for s in args.class_names.split(",")]
        if len(names) != num_classes:
            print(f"Warning: --class_names has {len(names)} but {num_classes} classes; using ints.")
            names = [str(i) for i in range(num_classes)]
    else:
        names = [str(i) for i in range(num_classes)]

    colours = class_colour_map(num_classes)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                             gridspec_kw={"width_ratios": [2, 1]})
    ax_disk, ax_hist = axes

    # Disk / square plot
    if is_euclidean:
        ax_disk.set_aspect("equal")
        ax_disk.set_title("Euclidean embeddings (PCA → 2D)")
        ax_disk.axhline(0, color="lightgray", linewidth=0.5)
        ax_disk.axvline(0, color="lightgray", linewidth=0.5)
    else:
        boundary_r = 1.0 / np.sqrt(abs(curvature))
        circle = patches.Circle((0, 0), boundary_r, fill=False,
                                edgecolor="black", linewidth=1.5)
        ax_disk.add_patch(circle)
        ax_disk.set_aspect("equal")
        ax_disk.set_xlim(-boundary_r * 1.1, boundary_r * 1.1)
        ax_disk.set_ylim(-boundary_r * 1.1, boundary_r * 1.1)
        ax_disk.set_title(f"Poincaré disk projection (k={curvature}, "
                          f"boundary at r={boundary_r:.2f})")
        # Add a few concentric guide circles
        for r in [0.25, 0.5, 0.75]:
            ax_disk.add_patch(patches.Circle((0, 0), r * boundary_r,
                                             fill=False, edgecolor="lightgray",
                                             linewidth=0.5, linestyle="--"))

    for c in range(num_classes):
        mask = (labels_sub == c).numpy()
        ax_disk.scatter(xy_sub[mask, 0], xy_sub[mask, 1],
                        s=8, alpha=0.5, color=colours[c],
                        label=f"{names[c]} (n={int(mask.sum())})")
        if not args.no_centroids:
            ax_disk.scatter(centroids_2d[c, 0], centroids_2d[c, 1],
                            s=200, marker="*", color=colours[c],
                            edgecolors="black", linewidths=1.0,
                            zorder=5)

    ax_disk.legend(fontsize=8, loc="best", framealpha=0.85, ncol=2)
    ax_disk.set_xlabel("PC1")
    ax_disk.set_ylabel("PC2")

    # Per-class radius histogram
    radii = z.norm(dim=-1).numpy()
    if not is_euclidean:
        ax_hist.set_title("Per-class ‖z_hyp‖ in 512-d (full latent, not 2D)")
        ax_hist.set_xlim(0, 1.0 / np.sqrt(abs(curvature)) * 1.05)
        ax_hist.axvline(1.0 / np.sqrt(abs(curvature)), color="black",
                        linewidth=1.0, linestyle="--", label="boundary")
    else:
        ax_hist.set_title("Per-class ‖z‖ in full Euclidean latent")

    for c in range(num_classes):
        mask = (labels == c).numpy()
        if mask.any():
            ax_hist.hist(radii[mask], bins=30, alpha=0.5,
                         color=colours[c], label=f"{names[c]}")
    ax_hist.set_xlabel("‖z‖")
    ax_hist.set_ylabel("count")
    ax_hist.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {args.output}")

    # Also print per-class statistics to stdout
    print("\nPer-class stats (full 512-d latent):")
    print(f"{'class':>6}  {'name':>14}  {'n':>5}  "
          f"{'norm_mean':>9}  {'norm_std':>9}  {'pca_centroid_norm':>18}")
    for c in range(num_classes):
        mask = (labels == c).numpy()
        if not mask.any():
            continue
        z_c = z[mask]
        norms_c = z_c.norm(dim=-1)
        cent2d_norm = float(np.linalg.norm(centroids_2d[c])) if not args.no_centroids else 0.0
        print(f"{c:>6}  {names[c]:>14}  {int(mask.sum()):>5}  "
              f"{norms_c.mean().item():>9.3f}  "
              f"{norms_c.std().item():>9.3f}  "
              f"{cent2d_norm:>18.3f}")


if __name__ == "__main__":
    main()
