#!/usr/bin/env python
"""Train the Hyperbolic Autoencoder on CIFAR-10/100-LT or ImageNet-LT.

- CIFAR: trained end-to-end with a small CNN encoder/decoder (HAECifar).
- ImageNet-LT: uses a frozen SD-VAE (stabilityai/sd-vae-ft-mse) as encoder +
  decoder; only the two projection layers and hyperbolic head are learned
  (HAEImageNet).

Self-contained — no dependency on the legacy coach.py or pSp pipeline.

Usage (CIFAR example):
    conda activate dl
    cd HAE
    python scripts/train_hae.py \\
        --dataset cifar10 --imbalance_factor 0.01 \\
        --num_epochs 200 --batch_size 128 \\
        --exp_dir ./experiments/cifar10_c1 \\
        --curvature -1.0

Usage (ImageNet-LT example):
    python scripts/train_hae.py \\
        --dataset imagenet_lt --batch_size 64 \\
        --num_epochs 40 --exp_dir ./experiments/imagenet_sdvae \\
        --curvature -1.0 --lpips_lambda 0.0 --ssim_lambda 0.0
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

# -- project imports (HAE is the working dir) --------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.hae_cifar import HAECifar
from models.hae_imagenet import HAEImageNet

# datasets live one level above HAE/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from lt_datasets.cifar_lt import CIFAR10LT, CIFAR100LT, make_cifar_lt_transform
from lt_datasets.imagenet_lt import ImageNetLT, make_imagenet_lt_transform
from lt_datasets.tiny_imagenet_lt import (
    TinyImageNetLT, make_tiny_imagenet_lt_transform,
    NATIVE_RESOLUTION as TINY_IMAGENET_RESOLUTION,
)


# ---------------------------------------------------------------------------
# Arg-parse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train HAE-CIFAR")

    # dataset
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100", "imagenet_lt",
                            "tiny_imagenet_lt"])
    p.add_argument("--imbalance_factor", type=float, default=0.01,
                   help="Exponential imbalance factor (CIFAR only; ImageNet-LT "
                        "is already imbalanced on disk)")
    p.add_argument("--image_size", type=int, default=None,
                   help="Override input size (default: 32 for CIFAR, 256 for ImageNet-LT)")
    p.add_argument("--data_root", type=str, default="../data")

    # model
    p.add_argument("--latent_dim", type=int, default=512)
    p.add_argument("--feature_size", type=int, default=512)
    p.add_argument("--curvature", type=float, default=-1.0,
                   help="Negative curvature k for the Poincaré ball")
    p.add_argument("--encoder_backbone", type=str, default=None,
                   choices=["cnn_cifar", "sd_vae", "taesd"],
                   help="Image encoder backbone. If unset, picks a sensible "
                        "default per dataset: cnn_cifar for CIFAR-10/100, "
                        "taesd for tiny_imagenet_lt (cheap pretrained), "
                        "sd_vae for imagenet_lt (full SD-VAE).")

    # imbalance handling
    p.add_argument("--sampler", type=str, default="instance",
                   choices=["instance", "balanced", "sqrt"],
                   help="Training sampler: natural (instance), class-balanced, "
                        "or sqrt-frequency reweighted")
    p.add_argument("--class_weighting", type=str, default="none",
                   choices=["none", "inv_freq", "effective_number"],
                   help="Per-class weights for the NLL term")
    p.add_argument("--eff_num_beta", type=float, default=0.9999,
                   help="Beta for effective-number weighting (Cui et al. 2019)")

    # hyperbolic contrastive (alternative to weighted NLL)
    p.add_argument("--contrastive_mode", type=str, default="none",
                   choices=["none", "supcon_hyp", "align_unif_hyp",
                            "supcon_hyp_radius"],
                   help="Hyperbolic contrastive loss on z_hyp")
    p.add_argument("--contrastive_lambda", type=float, default=0.0,
                   help="Weight of the contrastive term")
    p.add_argument("--contrastive_tau", type=float, default=0.5,
                   help="Temperature for Poincaré SupCon")
    p.add_argument("--radius_prior_lambda", type=float, default=0.0,
                   help="Weight of the radius-prior term (only used when "
                        "contrastive_mode = supcon_hyp_radius)")

    # training
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=None,
                   help="Stop after this many optimiser steps (overrides epochs)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)

    # loss weights
    p.add_argument("--pretrain_epochs", type=int, default=0,
                   help="Train as a pure AE for this many epochs before "
                        "the hyperbolic-head loss starts to ramp in. "
                        "0 = no pretraining stage (default).")
    p.add_argument("--hyperbolic_lambda", type=float, default=0.05,
                   help="Weight of NLL hyperbolic loss (warm-up over first 20 epochs)")
    p.add_argument("--hyper_warmup_epochs", type=int, default=20)
    p.add_argument("--lpips_lambda", type=float, default=0.8,
                   help="Weight of LPIPS perceptual loss (0 = off)")
    p.add_argument("--ssim_lambda", type=float, default=0.1,
                   help="Weight of SSIM loss (0 = off)")
    p.add_argument("--reverse_lambda", type=float, default=0.0,
                   help="Weight of reverse/cycle-consistency loss MSE(z_euc, z_euc_dec) "
                        "(inner manifold cycle — through expmap0/logmap0 only). 0 = off.")
    p.add_argument("--feat_recon_lambda", type=float, default=0.0,
                   help="Weight of the W+-cycle feature-reconstruction loss "
                        "(paper's L_rec, coach.py:247): MSE(z_flat, z_flat_dec). "
                        "Only active for HAEImageNet (sd_vae/taesd). 0 = off. "
                        "When `--feat_recon_adaptive` is set this is the BASE "
                        "weight; the effective weight is bumped on every step "
                        "as the loss decreases.")
    p.add_argument("--feat_recon_adaptive", action="store_true",
                   help="Use the coach.py:248-263 step-wise adaptive schedule: "
                        "as the W+-cycle MSE drops below successive thresholds "
                        "(0.2, 0.1, 0.05, ...), the effective lambda is bumped "
                        "(3, 6, 12, 24, 48, 96, 150). Keeps the loss "
                        "contribution roughly constant across training so the "
                        "term doesn't fade as the model fits the W+ cycle.")
    p.add_argument("--ms_ssim_lambda", type=float, default=0.0,
                   help="Weight of MS-SSIM perceptual loss (0 = off, needs kernel_size tuning)")
    p.add_argument("--lpips_bb", type=str, default="alex",
                   choices=["alex", "vgg", "squeeze"])
    p.add_argument("--kl_lambda", type=float, default=0.0,
                   help="If > 0, train HAECifar as a VAE with KL weight "
                        "(Stable-Diffusion-style ~1e-6 .. 1e-4 keeps recons "
                        "sharp while making the latent space samplable). "
                        "0 = deterministic AE (default). CIFAR only — "
                        "ignored for ImageNet path.")

    # resume / logging / checkpoints
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint .pt file to resume from. Loads "
                        "model state, optimizer, scheduler, global_step, "
                        "epoch, and RNG states so the next iteration continues "
                        "where the run stopped.")
    p.add_argument("--reset_lr_schedule", action="store_true",
                   help="On --resume, build a fresh cosine schedule with the "
                        "*new* --num_epochs / --lr instead of restoring the "
                        "saved scheduler state. Use when extending a run "
                        "whose original cosine has already decayed to ~0, "
                        "or when you want a fresh LR boost. "
                        "Does NOT reset global_step / epoch / model weights — "
                        "training continues from the same point, just with a "
                        "new LR trajectory.")
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--log_interval", type=int, default=50,
                   help="Print metrics every N steps")
    p.add_argument("--image_interval", type=int, default=500,
                   help="Save reconstruction grid every N steps")
    p.add_argument("--save_interval", type=int, default=5000)
    p.add_argument("--val_interval", type=int, default=1000)
    p.add_argument("--val_subset_size", type=int, default=5000,
                   help="Use a class-stratified subset of this size for periodic "
                        "validation. Set to 0 to use the full test set every time. "
                        "A full-test pass is always run at the end of training.")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--use_l1", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_datasets(args):
    """Return (train_dataset, test_dataset, num_classes, image_size)."""
    if args.dataset in ("cifar10", "cifar100"):
        cls = CIFAR10LT if args.dataset == "cifar10" else CIFAR100LT
        image_size = args.image_size or 32
        train_tf = make_cifar_lt_transform(image_size=image_size, train=True)
        test_tf = make_cifar_lt_transform(image_size=image_size, train=False)
        train_ds = cls(root=args.data_root, imbalance_factor=args.imbalance_factor,
                       train=True, transform=train_tf, download=True)
        test_ds = cls(root=args.data_root, imbalance_factor=args.imbalance_factor,
                      train=False, transform=test_tf, download=True)
        num_classes = 10 if args.dataset == "cifar10" else 100
        return train_ds, test_ds, num_classes, image_size

    if args.dataset == "imagenet_lt":
        image_size = args.image_size or 256
        train_tf = make_imagenet_lt_transform(image_size=image_size, train=True)
        test_tf = make_imagenet_lt_transform(image_size=image_size, train=False)
        train_ds = ImageNetLT(root=args.data_root, train=True,
                              transform=train_tf, image_size=image_size)
        test_ds = ImageNetLT(root=args.data_root, train=False,
                             transform=test_tf, image_size=image_size)
        return train_ds, test_ds, train_ds.num_classes, image_size

    if args.dataset == "tiny_imagenet_lt":
        # Native res is 64; allow override via --image_size for ablations
        # that want to share the 32×32 CIFAR backbone.
        image_size = args.image_size or TINY_IMAGENET_RESOLUTION
        train_tf = make_tiny_imagenet_lt_transform(image_size=image_size, train=True)
        test_tf = make_tiny_imagenet_lt_transform(image_size=image_size, train=False)
        train_ds = TinyImageNetLT(root=args.data_root,
                                  imbalance_factor=args.imbalance_factor,
                                  train=True, transform=train_tf,
                                  image_size=image_size)
        test_ds = TinyImageNetLT(root=args.data_root,
                                 imbalance_factor=1.0,    # val stays balanced
                                 train=False, transform=test_tf,
                                 image_size=image_size)
        return train_ds, test_ds, train_ds.num_classes, image_size

    raise ValueError(f"unknown dataset: {args.dataset}")


def _labels_array(ds) -> np.ndarray:
    """Return (N,) int64 labels array for either CIFAR-LT or ImageNet-LT."""
    if hasattr(ds, "targets") and ds.targets is not None:
        return np.asarray(ds.targets, dtype=np.int64)
    # CIFAR-LT stores per-item dicts under all_info
    return np.asarray([d["category_id"] for d in ds.all_info], dtype=np.int64)


def build_sampler(train_ds, mode: str):
    """Build a WeightedRandomSampler for 'balanced' or 'sqrt'.

    Returns None for 'instance' (natural frequency, use shuffle=True instead).
    """
    if mode == "instance":
        return None
    counts = train_ds.get_class_counts()
    n_cls = len(counts)
    cnt_arr = np.array([counts[c] for c in range(n_cls)], dtype=np.float64)
    if mode == "balanced":
        class_w = 1.0 / cnt_arr
    elif mode == "sqrt":
        class_w = 1.0 / np.sqrt(cnt_arr)
    else:
        raise ValueError(mode)
    # sample weight per example = class_w[label]
    labels = _labels_array(train_ds)
    sample_w = class_w[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True,
    )


def build_class_weights(train_ds, mode: str, beta: float, num_classes: int,
                        device):
    """Return a (num_classes,) tensor of per-class NLL weights, or None."""
    if mode == "none":
        return None
    counts = train_ds.get_class_counts()
    cnt_arr = np.array([counts[c] for c in range(num_classes)], dtype=np.float64)
    if mode == "inv_freq":
        w = 1.0 / cnt_arr
    elif mode == "effective_number":
        # Cui et al. 2019: w_c = (1 - beta) / (1 - beta**n_c)
        eff_num = 1.0 - np.power(beta, cnt_arr)
        w = (1.0 - beta) / np.maximum(eff_num, 1e-12)
    else:
        raise ValueError(mode)
    # normalise so mean weight = 1 (keeps loss scale stable)
    w = w / w.mean()
    return torch.as_tensor(w, dtype=torch.float32, device=device)


def hyperbolic_weight(epoch: int, warmup_epochs: int, target: float,
                      pretrain_epochs: int = 0) -> float:
    """Schedule for the hyperbolic loss weight.

    Phases:
      [0, pretrain_epochs)               -> 0   (pure AE pretraining)
      [pretrain_epochs, +warmup_epochs)  -> linear ramp 0 -> target
      [pretrain_epochs+warmup_epochs, ∞) -> target
    """
    if epoch < pretrain_epochs:
        return 0.0
    e = epoch - pretrain_epochs
    if warmup_epochs <= 0:
        return target
    return min(1.0, e / warmup_epochs) * target


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(args.exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Dataset ----------------------------------------------------------
    train_ds, test_ds, num_classes, image_size = get_datasets(args)
    print(f"Dataset: {args.dataset}  |  classes: {num_classes}  |  "
          f"train: {len(train_ds)}  |  test: {len(test_ds)}  |  "
          f"image_size: {image_size}")

    sampler = build_sampler(train_ds, args.sampler)
    shuffle = sampler is None
    print(f"Sampler: {args.sampler}  |  class_weighting: {args.class_weighting}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, shuffle=shuffle,
                              num_workers=args.workers, drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, drop_last=False, pin_memory=True)

    # ---- Quick-val subset (for periodic in-training validation) -----------
    # Stratified: at least one sample per class so head/mid/tail metrics
    # remain meaningful, then top up uniformly to val_subset_size.
    if args.val_subset_size and args.val_subset_size < len(test_ds):
        val_targets = (test_ds.targets if hasattr(test_ds, "targets")
                       else [test_ds[i][1] for i in range(len(test_ds))])
        val_targets = np.asarray(val_targets)
        rng = np.random.default_rng(0)
        per_class = {}
        for idx, c in enumerate(val_targets):
            per_class.setdefault(int(c), []).append(idx)
        # one stratified sample per class
        stratified = [int(rng.choice(idxs)) for idxs in per_class.values()]
        chosen = set(stratified)
        # top up uniformly without replacement
        remaining = [i for i in range(len(test_ds)) if i not in chosen]
        n_extra = max(0, args.val_subset_size - len(stratified))
        if n_extra > 0 and remaining:
            extra = rng.choice(remaining,
                               size=min(n_extra, len(remaining)),
                               replace=False)
            stratified.extend(int(i) for i in extra)
        val_subset = Subset(test_ds, sorted(stratified))
        val_loader_quick = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False,
            num_workers=max(2, args.workers // 2), drop_last=False,
            pin_memory=True,
        )
        print(f"Quick val: {len(val_subset)} samples "
              f"({len(per_class)} classes, ≥1 per class)")
    else:
        val_loader_quick = test_loader
        print(f"Quick val: full test set ({len(test_ds)} samples)")

    class_weights = build_class_weights(
        train_ds, args.class_weighting, args.eff_num_beta, num_classes, device
    )

    # ---- Contrastive loss (optional) -------------------------------------
    from criteria.hyperbolic_contrastive import build_contrastive
    contrastive_fn, radius_fn = build_contrastive(
        args.contrastive_mode,
        temperature=args.contrastive_tau,
        class_counts=train_ds.get_class_counts(),
        num_classes=num_classes,
    )
    if contrastive_fn is not None:
        contrastive_fn = contrastive_fn.to(device)
    if radius_fn is not None:
        radius_fn = radius_fn.to(device)
    print(f"Contrastive: {args.contrastive_mode}  (λ={args.contrastive_lambda}, "
          f"radius_λ={args.radius_prior_lambda})")

    # ---- Model ------------------------------------------------------------
    # Resolve encoder backbone — explicit flag wins, else dataset default.
    bb = args.encoder_backbone
    if bb is None:
        bb = {
            "cifar10": "cnn_cifar",
            "cifar100": "cnn_cifar",
            "tiny_imagenet_lt": "taesd",       # pretrained, cheap
            "imagenet_lt": "sd_vae",
        }[args.dataset]
    args.encoder_backbone = bb   # so it lands in the saved checkpoint args

    if bb in ("sd_vae", "taesd"):
        from models.hae_imagenet import (
            HAEImageNet as _HAEImageNet, SD_VAE_NAME, TAESD_NAME,
        )
        is_tiny = bb == "taesd"
        vae_name = TAESD_NAME if is_tiny else SD_VAE_NAME
        model = _HAEImageNet(
            num_classes=num_classes,
            latent_dim=args.latent_dim,
            feature_size=args.feature_size,
            curvature=args.curvature,
            vae_name=vae_name,
            tiny_vae=is_tiny,
            image_size=image_size,
        ).to(device)
        print(f"Using HAEImageNet [{bb.upper()}] @ {image_size}× "
              f"(frozen pretrained AE + hyperbolic head, "
              f"latent_grid={image_size // 8}×{image_size // 8}×4)")
    elif bb == "cnn_cifar":
        variational = args.kl_lambda > 0.0
        model = HAECifar(
            num_classes=num_classes,
            latent_dim=args.latent_dim,
            feature_size=args.feature_size,
            curvature=args.curvature,
            variational=variational,
            image_size=image_size,
        ).to(device)
        mode = "VAE" if variational else "AE"
        print(f"Using HAECifar [{mode}] @ {image_size}× (trained CNN + "
              f"hyperbolic head, kl_λ={args.kl_lambda})")
    else:
        raise ValueError(f"Unknown encoder_backbone: {bb}")
    print(model)

    # ---- Optimiser --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * len(train_loader)
    )

    # ---- Losses -----------------------------------------------------------
    l1_loss_fn = nn.L1Loss() if args.use_l1 else nn.MSELoss()

    lpips_fn = None
    if args.lpips_lambda > 0:
        from criteria.lpips.lpips import LPIPS
        lpips_fn = LPIPS(net_type=args.lpips_bb, device=str(device)).to(device).eval()

    # Single-scale SSIM (works at 32x32 natively)
    ssim_fn = None
    if args.ssim_lambda > 0:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_fn = StructuralSimilarityIndexMeasure(
            data_range=2.0, kernel_size=7
        ).to(device)

    # MS-SSIM (optional, reduced scales for 32x32)
    ms_ssim_fn = None
    if args.ms_ssim_lambda > 0:
        from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
        ms_ssim_fn = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=2.0,
            betas=(0.0448, 0.3001),  # 2 scales only for 32x32
            kernel_size=5,
        ).to(device)

    # ---- Logging ----------------------------------------------------------
    writer = SummaryWriter(log_dir=log_dir)
    if args.use_wandb:
        import wandb
        wandb.init(project="HAE-CIFAR", config=vars(args))

    # ---- Training loop ----------------------------------------------------
    global_step = 0
    best_val_loss = float("inf")
    start_epoch = 0

    # ---- Resume from checkpoint ------------------------------------------
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume path does not exist: {args.resume}")
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        # state_dict — strict by default; if the saved ckpt doesn't have a
        # field that the current model expects (e.g. legacy ckpts without
        # the new VAE heads), fall back to non-strict and warn loudly.
        try:
            model.load_state_dict(ckpt["state_dict"], strict=True)
        except RuntimeError as e:
            print(f"  strict load failed ({e}); retrying with strict=False")
            missing, unexpected = model.load_state_dict(ckpt["state_dict"],
                                                         strict=False)
            if missing:
                print(f"  missing  keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"  unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and not args.reset_lr_schedule:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
                print(f"  scheduler restored (T_max={scheduler.T_max}, "
                      f"last_epoch={scheduler.last_epoch})")
            except Exception as e:
                print(f"  scheduler restore failed ({e}) — using freshly "
                      f"built schedule from --num_epochs")
        elif args.reset_lr_schedule:
            # Reset cosine: schedule starts from step 0 of the new T_max
            # (= args.num_epochs * len(train_loader)) at base lr=args.lr.
            # The optimizer's saved LR state is overwritten — we explicitly
            # set every param group's lr back to args.lr so the cosine
            # starts at the requested peak.
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr
            print(f"  scheduler RESET — fresh cosine over "
                  f"{scheduler.T_max} steps at base lr={args.lr}")
        global_step = int(ckpt.get("global_step", 0))
        start_epoch = int(ckpt.get("epoch", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        rng = ckpt.get("rng", None)
        if rng is not None:
            try:
                torch.set_rng_state(rng["torch"])
                if rng.get("torch_cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["torch_cuda"])
                np.random.set_state(rng["numpy"])
            except Exception as e:
                print(f"  RNG restore failed ({e}) — continuing")
        print(f"  Resumed @ epoch={start_epoch} step={global_step} "
              f"best_val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        lam_hyper = hyperbolic_weight(epoch, args.hyper_warmup_epochs,
                                      args.hyperbolic_lambda,
                                      pretrain_epochs=args.pretrain_epochs)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            recon, logits, z_hyp, z_euc, z_euc_dec, kl = model(images)

            # --- Losses ---
            loss_recon = l1_loss_fn(recon, images)
            loss = loss_recon

            # KL divergence (VAE mode only — kl is a zero scalar otherwise)
            if args.kl_lambda > 0:
                loss = loss + args.kl_lambda * kl

            # Hyperbolic classification (optionally class-weighted)
            loss_hyper = F.nll_loss(logits, labels, weight=class_weights)
            loss = loss + lam_hyper * loss_hyper

            # LPIPS perceptual loss (upsample CIFAR 32→64 for AlexNet; leave ImageNet as is)
            loss_lpips = torch.tensor(0.0, device=device)
            if lpips_fn is not None and args.lpips_lambda > 0:
                if image_size < 64:
                    recon_in = F.interpolate(recon, size=64, mode='bilinear', align_corners=False)
                    images_in = F.interpolate(images, size=64, mode='bilinear', align_corners=False)
                else:
                    recon_in, images_in = recon, images
                loss_lpips = lpips_fn(recon_in, images_in)
                loss = loss + args.lpips_lambda * loss_lpips

            # SSIM loss
            loss_ssim = torch.tensor(0.0, device=device)
            if ssim_fn is not None and args.ssim_lambda > 0:
                loss_ssim = 1.0 - ssim_fn(recon, images)
                loss = loss + args.ssim_lambda * loss_ssim

            # MS-SSIM loss (optional)
            loss_ms_ssim = torch.tensor(0.0, device=device)
            if ms_ssim_fn is not None and args.ms_ssim_lambda > 0:
                loss_ms_ssim = 1.0 - ms_ssim_fn(recon, images)
                loss = loss + args.ms_ssim_lambda * loss_ms_ssim

            # Reverse / cycle-consistency loss: MSE(z_euc, z_euc_dec)
            # — the *inner* manifold cycle through expmap0/logmap0 only.
            # For HAEImageNet this is approximately a no-op (logmap0 ∘
            # expmap0 ≈ identity); kept for back-compat and because it
            # softly penalises encoder magnitudes that saturate expmap0.
            loss_reverse = torch.tensor(0.0, device=device)
            if args.reverse_lambda > 0:
                loss_reverse = F.mse_loss(z_euc, z_euc_dec)
                loss = loss + args.reverse_lambda * loss_reverse

            # Feature-reconstruction loss (paper's L_rec, coach.py:247):
            # MSE(z_flat, z_flat_dec) — across both proj_enc and proj_dec
            # plus the manifold round-trip. Real bottleneck reconstruction
            # term; only active for HAEImageNet (which stashes the W+
            # analogues on `_z_flat_target` / `_z_flat_recon`). HAECifar
            # has no analogous pre-MLP bottleneck so this is skipped.
            loss_feat_recon = torch.tensor(0.0, device=device)
            eff_feat_recon_lambda = args.feat_recon_lambda
            if args.feat_recon_lambda > 0:
                z_flat_t = getattr(model, "_z_flat_target", None)
                z_flat_r = getattr(model, "_z_flat_recon", None)
                if z_flat_t is not None and z_flat_r is not None:
                    loss_feat_recon = F.mse_loss(z_flat_r, z_flat_t)
                    if args.feat_recon_adaptive:
                        eff_feat_recon_lambda = adaptive_feat_recon_lambda(
                            float(loss_feat_recon.item()),
                            args.feat_recon_lambda,
                        )
                    loss = loss + eff_feat_recon_lambda * loss_feat_recon

            # Hyperbolic contrastive (+ optional radius prior)
            loss_con = torch.tensor(0.0, device=device)
            loss_radius = torch.tensor(0.0, device=device)
            if contrastive_fn is not None and args.contrastive_lambda > 0:
                loss_con = contrastive_fn(z_hyp, labels)
                loss = loss + args.contrastive_lambda * loss_con
            if radius_fn is not None and args.radius_prior_lambda > 0:
                loss_radius = radius_fn(z_hyp, labels)
                loss = loss + args.radius_prior_lambda * loss_radius

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            # stats
            pred = logits.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item()
            global_step += 1

            # --- Logging ---
            if global_step % args.log_interval == 0:
                with torch.no_grad():
                    z_norms = z_euc.norm(dim=-1)
                    z_hyp_norms = z_hyp.norm(dim=-1)  # < 1 in Poincaré mode; unbounded in Euclidean mode

                lr_now = scheduler.get_last_lr()[0]
                print(f"[Epoch {epoch+1}/{args.num_epochs}  step {global_step}]  "
                      f"loss={loss.item():.4f}  recon={loss_recon.item():.4f}  "
                      f"hyper={loss_hyper.item():.4f}  "
                      f"lpips={loss_lpips.item():.4f}  "
                      f"ssim={loss_ssim.item():.4f}  "
                      f"ms_ssim={loss_ms_ssim.item():.4f}  "
                      f"reverse={loss_reverse.item():.4f}  "
                      f"feat_rec={loss_feat_recon.item():.4f}  "
                      f"con={loss_con.item():.4f}  "
                      f"rad={loss_radius.item():.4f}  "
                      f"kl={kl.item():.4f}  "
                      f"acc={100.*correct/total:.1f}% lr={lr_now:.2e}")
                print(f"  z_euc norm — mean: {z_norms.mean():.3f}  max: {z_norms.max():.3f}")
                print(f"  z_hyp norm — mean: {z_hyp_norms.mean():.3f}  max: {z_hyp_norms.max():.3f}")
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_recon", loss_recon.item(), global_step)
                writer.add_scalar("train/loss_hyper", loss_hyper.item(), global_step)
                writer.add_scalar("train/loss_lpips", loss_lpips.item(), global_step)
                writer.add_scalar("train/loss_ssim", loss_ssim.item(), global_step)
                writer.add_scalar("train/loss_ms_ssim", loss_ms_ssim.item(), global_step)
                writer.add_scalar("train/loss_reverse", loss_reverse.item(), global_step)
                writer.add_scalar("train/loss_feat_recon", loss_feat_recon.item(), global_step)
                writer.add_scalar("train/lam_feat_recon", float(eff_feat_recon_lambda), global_step)
                writer.add_scalar("train/loss_contrastive", loss_con.item(), global_step)
                writer.add_scalar("train/loss_radius", loss_radius.item(), global_step)
                writer.add_scalar("train/loss_kl", kl.item(), global_step)
                writer.add_scalar("train/accuracy", 100. * correct / total, global_step)
                writer.add_scalar("train/lr", lr_now, global_step)
                writer.add_scalar("train/lam_hyper", lam_hyper, global_step)
                writer.add_scalar("train/z_euc_norm_mean", z_norms.mean(), global_step)
                writer.add_scalar("train/z_euc_norm_max", z_norms.max(), global_step)
                writer.add_scalar("train/z_hyp_norm_mean", z_hyp_norms.mean(), global_step)
                writer.add_scalar("train/z_hyp_norm_max", z_hyp_norms.max(), global_step)
                if args.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/loss_recon": loss_recon.item(),
                        "train/loss_hyper": loss_hyper.item(),
                        "train/loss_lpips": loss_lpips.item(),
                        "train/loss_ssim": loss_ssim.item(),
                        "train/loss_reverse": loss_reverse.item(),
                        "train/loss_feat_recon": loss_feat_recon.item(),
                        "train/accuracy": 100. * correct / total,
                        "train/lr": lr_now,
                        "global_step": global_step,
                    })

            # --- Save reconstruction images ---
            if global_step % args.image_interval == 0:
                img_dir = os.path.join(log_dir, "images")
                os.makedirs(img_dir, exist_ok=True)
                n = min(8, images.size(0))
                comparison = torch.cat([images[:n], recon[:n]])
                save_image(comparison, os.path.join(img_dir, f"recon_{global_step:06d}.png"),
                           nrow=n, normalize=True, value_range=(-1, 1))

            # --- Validation (periodic; uses quick subset when configured) ---
            if global_step % args.val_interval == 0:
                val_loss, val_recon = validate(model, val_loader_quick, device, l1_loss_fn,
                                    writer, global_step, lam_hyper,
                                    class_weights)
                if val_recon < best_val_loss:
                    best_val_loss = val_recon
                    save_checkpoint(model, optimizer, args, global_step, epoch,
                                    ckpt_dir, "best_model.pt",
                                    scheduler=scheduler,
                                    best_val_loss=best_val_loss)
                model.train()

            # --- Periodic checkpoint ---
            if global_step % args.save_interval == 0:
                save_checkpoint(model, optimizer, args, global_step, epoch,
                                ckpt_dir, f"step_{global_step:06d}.pt",
                                scheduler=scheduler,
                                best_val_loss=best_val_loss)

            # early stop on max_steps
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        dt = time.time() - t0
        print(f"Epoch {epoch+1} done in {dt:.1f}s  "
              f"avg_loss={epoch_loss/len(train_loader):.4f}  "
              f"train_acc={100.*correct/total:.1f}%")

        if args.max_steps is not None and global_step >= args.max_steps:
            print(f"Reached max_steps={args.max_steps}, stopping.")
            break

    # final checkpoint
    save_checkpoint(model, optimizer, args, global_step, epoch,
                    ckpt_dir, "final_model.pt",
                    scheduler=scheduler,
                    best_val_loss=best_val_loss)

    # one full-test pass at the end so the headline number is on all 50k
    if val_loader_quick is not test_loader:
        print("Running final validation on FULL test set...")
        validate(model, test_loader, device, l1_loss_fn, writer, global_step,
                 lam_hyper, class_weights)

    writer.close()
    print("Training complete.")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, l1_loss_fn, writer, global_step,
             lam_hyper, class_weights):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    correct = 0
    total = 0

    # per-class accuracy for head/mid/tail tracking (C inferred from logits)
    per_class_correct = None
    per_class_total = None

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        recon, logits, z_hyp, z_euc, z_euc_dec, _kl = model(images)

        loss_recon = l1_loss_fn(recon, images)
        # val NLL is unweighted to keep metric comparable across runs
        loss_hyper = F.nll_loss(logits, labels)
        # NB: KL deliberately excluded from val loss so the "best_model"
        # selection compares pure recon+nll across AE / VAE runs.
        loss = loss_recon + lam_hyper * loss_hyper

        total_loss += loss.item() * images.size(0)
        total_recon += loss_recon.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        # per-class
        if per_class_correct is None:
            C = logits.size(1)
            per_class_correct = torch.zeros(C, device=device)
            per_class_total = torch.zeros(C, device=device)
        for c in labels.unique():
            m = labels == c
            per_class_correct[c] += pred[m].eq(labels[m]).sum()
            per_class_total[c] += m.sum()

    avg_loss = total_loss / total
    avg_recon = total_recon / total
    acc = 100.0 * correct / total
    print(f"  [VAL step {global_step}]  loss={avg_loss:.4f}  recon_loss={avg_recon:.4f}  acc={acc:.1f}%")
    writer.add_scalar("val/loss", avg_loss, global_step)
    writer.add_scalar("val/loss_recon", avg_recon, global_step)
    writer.add_scalar("val/accuracy", acc, global_step)

    # per-class accuracy (log mean of bottom 1/3 as tail acc)
    if per_class_total is not None:
        per_cls_acc = (per_class_correct / per_class_total.clamp(min=1)).cpu().numpy()
        # tail = classes with lowest count — since CIFAR-LT orders classes
        # by descending frequency, bottom third by index is tail.
        C = len(per_cls_acc)
        third = max(1, C // 3)
        head_acc = 100. * per_cls_acc[:third].mean()
        mid_acc  = 100. * per_cls_acc[third:2*third].mean()
        tail_acc = 100. * per_cls_acc[2*third:].mean()
        print(f"    head={head_acc:.1f}%  mid={mid_acc:.1f}%  tail={tail_acc:.1f}%")
        writer.add_scalar("val/acc_head", head_acc, global_step)
        writer.add_scalar("val/acc_mid", mid_acc, global_step)
        writer.add_scalar("val/acc_tail", tail_acc, global_step)

    return avg_loss, avg_recon


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def adaptive_feat_recon_lambda(loss_val: float, base: float) -> float:
    """coach.py:248-263 step-wise schedule.

    As the W+-cycle MSE drops, increase λ to keep the contribution roughly
    constant. The thresholds are powers of 2 from 0.2 down to 0.003125; the
    matching multipliers are 3, 6, 12, 24, 48, 96, 150 (the paper's choice
    — not exactly geometric but close).
    """
    if loss_val <= 0.003125:
        return 150.0
    if loss_val <= 0.00625:
        return 96.0
    if loss_val <= 0.0125:
        return 48.0
    if loss_val <= 0.025:
        return 24.0
    if loss_val <= 0.05:
        return 12.0
    if loss_val <= 0.1:
        return 6.0
    if loss_val <= 0.2:
        return 3.0
    return base


def save_checkpoint(model, optimizer, args, step, epoch, ckpt_dir, name,
                    scheduler=None, best_val_loss=None):
    path = os.path.join(ckpt_dir, name)
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "epoch": epoch,
        "args": vars(args),
        "rng": {
            "torch": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
            "numpy": np.random.get_state(),
        },
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if best_val_loss is not None:
        payload["best_val_loss"] = float(best_val_loss)
    torch.save(payload, path)
    print(f"  Saved checkpoint → {path}")


if __name__ == "__main__":
    main()
