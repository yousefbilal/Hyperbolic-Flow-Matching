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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

# -- project imports (HAE is the working dir) --------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.hae_cifar import HAECifar
from models.hae_imagenet import HAEImageNet

# datasets live one level above HAE/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from datasets.cifar_lt import CIFAR10LT, CIFAR100LT, make_cifar_lt_transform
from datasets.imagenet_lt import ImageNetLT, make_imagenet_lt_transform


# ---------------------------------------------------------------------------
# Arg-parse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train HAE-CIFAR")

    # dataset
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100", "imagenet_lt"])
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
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)

    # loss weights
    p.add_argument("--hyperbolic_lambda", type=float, default=0.05,
                   help="Weight of NLL hyperbolic loss (warm-up over first 20 epochs)")
    p.add_argument("--hyper_warmup_epochs", type=int, default=20)
    p.add_argument("--lpips_lambda", type=float, default=0.8,
                   help="Weight of LPIPS perceptual loss (0 = off)")
    p.add_argument("--ssim_lambda", type=float, default=0.1,
                   help="Weight of SSIM loss (0 = off)")
    p.add_argument("--reverse_lambda", type=float, default=0.0,
                   help="Weight of reverse/cycle-consistency loss MSE(z_euc, z_euc_dec) (0 = off)")
    p.add_argument("--ms_ssim_lambda", type=float, default=0.0,
                   help="Weight of MS-SSIM perceptual loss (0 = off, needs kernel_size tuning)")
    p.add_argument("--lpips_bb", type=str, default="alex",
                   choices=["alex", "vgg", "squeeze"])

    # logging / checkpoints
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--log_interval", type=int, default=50,
                   help="Print metrics every N steps")
    p.add_argument("--image_interval", type=int, default=500,
                   help="Save reconstruction grid every N steps")
    p.add_argument("--save_interval", type=int, default=5000)
    p.add_argument("--val_interval", type=int, default=1000)
    p.add_argument("--use_wandb", action="store_true")

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


def hyperbolic_weight(epoch: int, warmup_epochs: int, target: float) -> float:
    """Linear warmup of the hyperbolic loss weight."""
    if warmup_epochs <= 0:
        return target
    return min(1.0, epoch / warmup_epochs) * target


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
    if args.dataset == "imagenet_lt":
        model = HAEImageNet(
            num_classes=num_classes,
            latent_dim=args.latent_dim,
            feature_size=args.feature_size,
            curvature=args.curvature,
        ).to(device)
        print("Using HAEImageNet (frozen SD-VAE + hyperbolic head)")
    else:
        model = HAECifar(
            num_classes=num_classes,
            latent_dim=args.latent_dim,
            feature_size=args.feature_size,
            curvature=args.curvature,
        ).to(device)
        print("Using HAECifar (trained CNN + hyperbolic head)")
    print(model)

    # ---- Optimiser --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * len(train_loader)
    )

    # ---- Losses -----------------------------------------------------------
    l1_loss_fn = nn.MSELoss()

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

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        lam_hyper = hyperbolic_weight(epoch, args.hyper_warmup_epochs,
                                      args.hyperbolic_lambda)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            recon, logits, z_hyp, z_euc, z_euc_dec = model(images)

            # --- Losses ---
            loss_recon = l1_loss_fn(recon, images)
            loss = loss_recon

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
            loss_reverse = torch.tensor(0.0, device=device)
            if args.reverse_lambda > 0:
                loss_reverse = F.mse_loss(z_euc, z_euc_dec)
                loss = loss + args.reverse_lambda * loss_reverse

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
                      f"reverse={loss_reverse.item():.4f}  "
                      f"con={loss_con.item():.4f}  "
                      f"rad={loss_radius.item():.4f}  "
                      f"acc={100.*correct/total:.1f}% lr={lr_now:.2e}")
                print(f"  z_euc norm — mean: {z_norms.mean():.3f}  max: {z_norms.max():.3f}")
                print(f"  z_hyp norm — mean: {z_hyp_norms.mean():.3f}  max: {z_hyp_norms.max():.3f}")
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_recon", loss_recon.item(), global_step)
                writer.add_scalar("train/loss_hyper", loss_hyper.item(), global_step)
                writer.add_scalar("train/loss_lpips", loss_lpips.item(), global_step)
                writer.add_scalar("train/loss_ssim", loss_ssim.item(), global_step)
                writer.add_scalar("train/loss_reverse", loss_reverse.item(), global_step)
                writer.add_scalar("train/loss_contrastive", loss_con.item(), global_step)
                writer.add_scalar("train/loss_radius", loss_radius.item(), global_step)
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

            # --- Validation ---
            if global_step % args.val_interval == 0:
                val_loss = validate(model, test_loader, device, l1_loss_fn,
                                    writer, global_step, lam_hyper,
                                    class_weights)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, args, global_step, epoch,
                                    ckpt_dir, "best_model.pt")
                model.train()

            # --- Periodic checkpoint ---
            if global_step % args.save_interval == 0:
                save_checkpoint(model, optimizer, args, global_step, epoch,
                                ckpt_dir, f"step_{global_step:06d}.pt")

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
                    ckpt_dir, "final_model.pt")
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
    correct = 0
    total = 0

    # per-class accuracy for head/mid/tail tracking (C inferred from logits)
    per_class_correct = None
    per_class_total = None

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        recon, logits, z_hyp, z_euc, z_euc_dec = model(images)

        loss_recon = l1_loss_fn(recon, images)
        # val NLL is unweighted to keep metric comparable across runs
        loss_hyper = F.nll_loss(logits, labels)
        loss = loss_recon + lam_hyper * loss_hyper

        total_loss += loss.item() * images.size(0)
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
    acc = 100.0 * correct / total
    print(f"  [VAL step {global_step}]  loss={avg_loss:.4f}  acc={acc:.1f}%")
    writer.add_scalar("val/loss", avg_loss, global_step)
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

    return avg_loss


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, args, step, epoch, ckpt_dir, name):
    path = os.path.join(ckpt_dir, name)
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "epoch": epoch,
        "args": vars(args),
    }, path)
    print(f"  Saved checkpoint → {path}")


if __name__ == "__main__":
    main()
