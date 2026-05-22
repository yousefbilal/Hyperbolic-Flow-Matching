#!/usr/bin/env python
"""Compute FID, IS, Recall, F_8, F_1/8 against the BALANCED Tiny ImageNet
training split.

Mirrors the eval setup of CBDM (Qin et al. 2023) and DiffROP (Yan et al.
2024):
  - Inception-V3 features (2048-d pool) and logits via torchmetrics, which
    uses the canonical FID-paper checkpoint (compatible with pytorch-fid
    and published FID values).
  - FID via torchmetrics.FrechetInceptionDistance.
  - IS  via torchmetrics.InceptionScore.
  - Improved Precision/Recall (Kynkäänniemi et al. 2019, K=5) computed
    manually on the same Inception pool features.
  - F_8, F_1/8 from improved P/R as F_β = (1+β²)·P·R / (β²·P + R).
  - Real reference: BALANCED Tiny ImageNet train (imbalance_factor=1.0,
    train=True, ~100k images for 200 classes), matching both papers.

Requires: pip install 'torchmetrics>=1.0' 'torch-fidelity>=0.3'
          (torch-fidelity is what torchmetrics uses internally for the
           canonical Inception-V3 weights)

Usage:
    python scripts/compute_metrics.py \
        --generated_dir /path/to/output_dir \
        --data_root /workspace/Hyperbolic-Flow-Matching/data \
        --num_classes 200 \
        --output_json /path/to/metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Datasets package — same path as the rest of the codebase.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from lt_datasets.tiny_imagenet_lt import (
        TinyImageNetLT, NATIVE_RESOLUTION as TINY_IMAGENET_RESOLUTION,
    )
except ImportError:
    from datasets.tiny_imagenet_lt import (
        TinyImageNetLT, NATIVE_RESOLUTION as TINY_IMAGENET_RESOLUTION,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generated_dir", type=str, required=True,
                   help="Output dir from generate_for_metrics.sh")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--num_classes", type=int, default=200)
    p.add_argument("--image_size", type=int, default=TINY_IMAGENET_RESOLUTION,
                   help="Resize-target before extraction. Inception will "
                        "internally resize to 299×299 anyway.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--knn_k", type=int, default=5,
                   help="K for Improved P/R manifold (Kynkäänniemi 2019).")
    p.add_argument("--real_split", type=str, default="train",
                   choices=["train", "val"],
                   help="CBDM / DiffROP both use BALANCED TRAIN as the real "
                        "reference (~100k for Tiny ImageNet). 'val' is for "
                        "quick sanity checks only (10k, 50/class).")
    p.add_argument("--max_real", type=int, default=None,
                   help="Optionally cap the real-image set.")
    p.add_argument("--max_generated", type=int, default=None,
                   help="Optionally cap the generated set.")
    p.add_argument("--is_splits", type=int, default=10,
                   help="Splits for Inception Score (paper default: 10).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class GeneratedImagesDataset(Dataset):
    """Walk class_*/images/*.png under a root.

    Returns (image, class_id). class_id is parsed from the parent dir name
    'class_NNN'. If the layout is flat (`<root>/images/*.png` — single-class
    dump from a one-shot generate), class_id falls back to -1 and per-shot
    grouping won't work for that data.
    """
    def __init__(self, root: str, transform=None, max_n=None):
        root = Path(root)
        entries = []
        for class_dir in sorted(root.glob("class_*")):
            try:
                class_id = int(class_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            for p in sorted(class_dir.glob("images/*.png")):
                entries.append((p, class_id))
        if not entries:
            for p in sorted(root.glob("images/*.png")):
                entries.append((p, -1))
        if max_n is not None:
            entries = entries[:max_n]
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, cls = self.entries[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, cls


def _build_transform(image_size: int):
    """Output: float tensor in [0, 1]. The canonical FID Inception expects
    uint8 [0, 255], handled inside the extraction loop via .byte()."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),     # PIL → float [0, 1]
    ])


def build_real_loader(args):
    tf = _build_transform(args.image_size)
    is_train = (args.real_split == "train")
    ds = TinyImageNetLT(root=args.data_root, imbalance_factor=1.0,
                        train=is_train, transform=tf, image_size=args.image_size)
    if args.max_real is not None:
        from torch.utils.data import Subset
        idx = np.random.RandomState(0).choice(len(ds), size=args.max_real,
                                              replace=False)
        ds = Subset(ds, idx.tolist())
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.workers, pin_memory=True)


def build_generated_loader(args):
    tf = _build_transform(args.image_size)
    ds = GeneratedImagesDataset(args.generated_dir, transform=tf,
                                max_n=args.max_generated)
    if len(ds) == 0:
        raise FileNotFoundError(
            f"No PNGs under {args.generated_dir}/class_*/images/. "
            f"Did you run generate_for_metrics.sh first?")
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.workers, pin_memory=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features_and_update_metrics(loader, fid_metric, is_metric,
                                          device, label, is_real=False):
    """Single forward pass per image. Updates FID, IS, and returns the
    2048-d Inception pool features, class labels, and (for generated
    images) the 1000-d Inception softmax probabilities used for per-shot
    Inception Score.

    Inputs are float [0, 1]; FID/IS metrics are built with normalize=True
    so they handle the [0,1] → uint8 conversion internally.
    Both real and generated loaders yield (image, class_id).
    """
    feats_chunks = []
    label_chunks = []
    prob_chunks  = []                                     # gen only
    n = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)                                  # float [0, 1]

        fid_metric.update(x, real=is_real)
        if not is_real and is_metric is not None:
            is_metric.update(x)

        # Pool features for P/R. The FID metric's Inception expects uint8.
        f = fid_metric.inception((x * 255).byte())
        feats_chunks.append(f.cpu())
        label_chunks.append(y.long())

        # 1000-d softmax probabilities for per-shot IS. Only needed for
        # generated images; reuses the IS metric's own Inception so we
        # don't run a third Inception pass.
        if not is_real and is_metric is not None:
            logits = is_metric.inception((x * 255).byte())
            prob_chunks.append(torch.softmax(logits, dim=-1).cpu())

        n += x.shape[0]

        if n % (loader.batch_size * 20) == 0:
            print(f"  [{label}] {n} images...", flush=True)

    probs = torch.cat(prob_chunks, dim=0) if prob_chunks else None
    return (torch.cat(feats_chunks, dim=0),
            torch.cat(label_chunks, dim=0),
            probs,
            n)


def _knn_radii(feats: torch.Tensor, k: int) -> torch.Tensor:
    """Distance to the k-th nearest neighbour (excluding self)."""
    N = feats.shape[0]
    radii = torch.empty(N, dtype=torch.float32)
    chunk = 256
    feats_sq = (feats ** 2).sum(dim=-1)
    for i in range(0, N, chunk):
        a = feats[i:i + chunk]
        a_sq = (a ** 2).sum(dim=-1, keepdim=True)
        d2 = a_sq + feats_sq.unsqueeze(0) - 2 * a @ feats.T
        # Exclude self
        for j, idx in enumerate(range(i, min(i + chunk, N))):
            d2[j, idx] = float("inf")
        topk, _ = torch.topk(d2, k=k, dim=-1, largest=False)
        radii[i:i + chunk] = topk[:, -1].clamp_min(0).sqrt()
    return radii


def _manifold_membership(feats_query: torch.Tensor,
                          feats_ref: torch.Tensor,
                          radii_ref: torch.Tensor) -> torch.Tensor:
    """For each query feature, True if it falls inside any reference k-NN ball."""
    N_q = feats_query.shape[0]
    out = torch.zeros(N_q, dtype=torch.bool)
    chunk = 256
    ref_sq = (feats_ref ** 2).sum(dim=-1)
    for i in range(0, N_q, chunk):
        a = feats_query[i:i + chunk]
        a_sq = (a ** 2).sum(dim=-1, keepdim=True)
        d2 = a_sq + ref_sq.unsqueeze(0) - 2 * a @ feats_ref.T
        d  = d2.clamp_min(0).sqrt()
        within = (d <= radii_ref.unsqueeze(0))
        out[i:i + chunk] = within.any(dim=-1)
    return out


def compute_fid_from_features(real_feats: torch.Tensor,
                               gen_feats: torch.Tensor) -> float:
    """FID computed directly from precomputed pool features. Same formula
    as torchmetrics.FrechetInceptionDistance.compute() — used for per-shot
    FID after splitting the global feature tensors by class group."""
    if len(real_feats) < 2 or len(gen_feats) < 2:
        return float("nan")
    from scipy.linalg import sqrtm
    r = real_feats.numpy().astype(np.float64)
    g = gen_feats.numpy().astype(np.float64)
    mu_r, mu_g = r.mean(axis=0), g.mean(axis=0)
    cov_r = np.cov(r, rowvar=False)
    cov_g = np.cov(g, rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = sqrtm(cov_r @ cov_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_r) + np.trace(cov_g)
                  - 2 * np.trace(covmean))


def compute_is_from_probs(probs: torch.Tensor, n_splits: int = 10):
    """Inception Score from per-image softmax probabilities.

    IS = exp(E_x[KL(p(y|x) || p(y))]),  p(y) = E_x[p(y|x)].

    Splits the input into n_splits equal chunks, computes IS on each,
    returns (mean, std). Matches the splits-based variant used by
    torchmetrics.InceptionScore. Falls back to NaN if there aren't
    enough samples to fill n_splits.
    """
    N = probs.shape[0]
    if N < n_splits or N == 0:
        return float("nan"), float("nan")
    split_size = N // n_splits
    scores = []
    for k in range(n_splits):
        chunk = probs[k * split_size : (k + 1) * split_size]
        py = chunk.mean(dim=0, keepdim=True)            # marginal p(y)
        kl = (chunk * (torch.log(chunk + 1e-16)
                       - torch.log(py + 1e-16))).sum(dim=-1)
        scores.append(torch.exp(kl.mean()).item())
    scores = np.array(scores)
    return float(scores.mean()), float(scores.std())


def get_shot_groups(args) -> dict:
    """Resolve head/mid/tail class groupings using the LT training counts."""
    ds_lt = TinyImageNetLT(root=args.data_root, imbalance_factor=0.01,
                            train=True, transform=None,
                            image_size=args.image_size)
    return ds_lt.get_head_mid_tail_split(n_groups=3)


def compute_improved_pr(real_feats: torch.Tensor, gen_feats: torch.Tensor,
                         k: int = 5) -> dict:
    """Improved Precision / Recall (Kynkäänniemi et al. 2019).

    Precision = frac. of generated samples inside the real-data K=5 manifold.
    Recall    = frac. of real samples inside the gen-data K=5 manifold.
    F_β       = (1+β²)·P·R / (β²·P + R).
    """
    real_feats = real_feats.float()
    gen_feats  = gen_feats.float()
    print(f"  K-NN radii on real (N={len(real_feats)}, k={k})...", flush=True)
    real_radii = _knn_radii(real_feats, k)
    print(f"  K-NN radii on gen  (N={len(gen_feats)},  k={k})...", flush=True)
    gen_radii  = _knn_radii(gen_feats, k)

    print(f"  precision (gen ∈ real-manifold)...", flush=True)
    in_real = _manifold_membership(gen_feats, real_feats, real_radii)
    precision = float(in_real.float().mean())

    print(f"  recall (real ∈ gen-manifold)...", flush=True)
    in_gen = _manifold_membership(real_feats, gen_feats, gen_radii)
    recall = float(in_gen.float().mean())

    def f_beta(p, r, beta):
        b2 = beta ** 2
        denom = b2 * p + r
        if denom <= 0:
            return 0.0
        return float((1 + b2) * p * r / denom)

    return {
        "precision": precision,
        "recall": recall,
        "F_8":   f_beta(precision, recall, 8.0),
        "F_1/8": f_beta(precision, recall, 1.0 / 8.0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Lazy import so the help text works without torchmetrics installed.
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.inception import InceptionScore
    except ImportError:
        raise ImportError(
            "compute_metrics.py needs torchmetrics + torch-fidelity. "
            "Install with: pip install 'torchmetrics>=1.0' 'torch-fidelity>=0.3'"
        )

    print("Building canonical-Inception FID and IS metrics...")
    # normalize=True → metrics accept float [0, 1] directly and handle
    # the uint8 conversion internally. Loader gives float [0, 1] from ToTensor.
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True,
                                           reset_real_features=True).to(device)
    is_metric  = InceptionScore(normalize=True, splits=args.is_splits).to(device)

    print(f"\nLoading real images (Tiny ImageNet {args.real_split.upper()} "
          f"split, balanced)...")
    real_loader = build_real_loader(args)
    print(f"  N_real = {len(real_loader.dataset)}")

    print(f"\nLoading generated images from {args.generated_dir}...")
    gen_loader = build_generated_loader(args)
    print(f"  N_gen = {len(gen_loader.dataset)}")

    print("\nExtracting Inception features for REAL "
          "(updates FID's real stats simultaneously)...")
    real_feats, real_labels, _, n_real = extract_features_and_update_metrics(
        real_loader, fid_metric, is_metric=None, device=device,
        label="real", is_real=True,
    )

    print("\nExtracting Inception features for GENERATED "
          "(updates FID's fake stats + IS simultaneously)...")
    gen_feats, gen_labels, gen_probs, n_gen = (
        extract_features_and_update_metrics(
            gen_loader, fid_metric, is_metric=is_metric, device=device,
            label="gen", is_real=False,
        )
    )

    print("\n--- Global metrics ---")
    metrics = {}

    print("FID (canonical Inception, comparable to published)...")
    metrics["FID"] = float(fid_metric.compute())
    print(f"  FID = {metrics['FID']:.3f}")

    print(f"Inception Score ({args.is_splits} splits)...")
    is_mean, is_std = is_metric.compute()
    metrics["IS_mean"] = float(is_mean)
    metrics["IS_std"]  = float(is_std)
    print(f"  IS  = {metrics['IS_mean']:.3f} ± {metrics['IS_std']:.3f}")

    print(f"Improved Precision/Recall (Kynkäänniemi 2019, K={args.knn_k})...")
    pr = compute_improved_pr(real_feats, gen_feats, k=args.knn_k)
    metrics.update(pr)
    print(f"  precision = {pr['precision']:.3f}")
    print(f"  recall    = {pr['recall']:.3f}")
    print(f"  F_8       = {pr['F_8']:.3f}    (diversity-leaning)")
    print(f"  F_1/8     = {pr['F_1/8']:.3f}  (fidelity-leaning)")

    metrics["n_real"] = n_real
    metrics["n_gen"]  = n_gen

    # ----- Per-shot (head / mid / tail) ------------------------------------
    print("\n--- Per-shot metrics (head / mid / tail) ---")
    shots = get_shot_groups(args)        # {"head": [...], "mid": [...], "tail": [...]}
    metrics["per_shot"] = {}
    if (gen_labels < 0).any():
        print("  Skipping per-shot: some generated images have unknown class "
              "(flat output_dir/images/*.png layout). Re-run generation with "
              "the per-class layout (class_NNN/images/*.png).")
    else:
        for shot, class_ids in shots.items():
            cls_set = torch.tensor(class_ids, dtype=torch.long)
            real_mask = torch.isin(real_labels, cls_set)
            gen_mask  = torch.isin(gen_labels, cls_set)
            real_sub = real_feats[real_mask]
            gen_sub  = gen_feats[gen_mask]
            shot_fid = compute_fid_from_features(real_sub, gen_sub)
            shot_pr  = compute_improved_pr(real_sub, gen_sub, k=args.knn_k)
            shot_is_mean, shot_is_std = (
                compute_is_from_probs(gen_probs[gen_mask],
                                       n_splits=args.is_splits)
                if gen_probs is not None
                else (float("nan"), float("nan"))
            )
            metrics["per_shot"][shot] = {
                "n_classes": len(class_ids),
                "n_real":    int(real_mask.sum()),
                "n_gen":     int(gen_mask.sum()),
                "FID":       shot_fid,
                "IS_mean":   shot_is_mean,
                "IS_std":    shot_is_std,
                "precision": shot_pr["precision"],
                "recall":    shot_pr["recall"],
                "F_8":       shot_pr["F_8"],
                "F_1/8":     shot_pr["F_1/8"],
            }
            print(f"  [{shot:5s}]  n_cls={len(class_ids):3d}  "
                  f"n_real={int(real_mask.sum()):5d}  "
                  f"n_gen={int(gen_mask.sum()):5d}  "
                  f"FID={shot_fid:7.3f}  "
                  f"IS={shot_is_mean:5.3f}±{shot_is_std:.3f}  "
                  f"prec={shot_pr['precision']:.3f}  "
                  f"recall={shot_pr['recall']:.3f}  "
                  f"F_8={shot_pr['F_8']:.3f}  "
                  f"F_1/8={shot_pr['F_1/8']:.3f}")

    print("\n=== Summary ===")
    print(f"  Global    FID={metrics['FID']:.3f}  IS={metrics['IS_mean']:.3f}±{metrics['IS_std']:.3f}  "
          f"Recall={metrics['recall']:.3f}  F_8={metrics['F_8']:.3f}  F_1/8={metrics['F_1/8']:.3f}")
    if metrics["per_shot"]:
        for shot in ("head", "mid", "tail"):
            if shot in metrics["per_shot"]:
                ps = metrics["per_shot"][shot]
                print(f"  {shot:<8s}  FID={ps['FID']:.3f}  "
                      f"IS={ps['IS_mean']:.3f}±{ps['IS_std']:.3f}  "
                      f"P={ps['precision']:.3f}  R={ps['recall']:.3f}  "
                      f"F_8={ps['F_8']:.3f}  F_1/8={ps['F_1/8']:.3f}")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)),
                    exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
