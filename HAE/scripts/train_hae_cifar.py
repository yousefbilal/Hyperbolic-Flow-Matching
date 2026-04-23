#!/usr/bin/env python
"""Train the Hyperbolic Autoencoder on CIFAR-10 / CIFAR-100-LT.

Self-contained script — no dependency on the legacy coach.py or pSp pipeline.

Usage (example):
    conda activate dl
    cd HAE
    python scripts/train_hae_cifar.py \\
        --dataset cifar10 --imbalance_factor 0.01 \\
        --num_epochs 200 --batch_size 128 \\
        --exp_dir ./experiments/cifar10_c1 \\
        --curvature -1.0
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

# -- project imports (HAE is the working dir) --------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.hae_cifar import HAECifar, AECifar

# datasets live one level above HAE/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from datasets.cifar_lt import CIFAR10LT, CIFAR100LT, make_cifar_lt_transform


# ---------------------------------------------------------------------------
# Arg-parse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train HAE-CIFAR")

    # dataset
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100"])
    p.add_argument("--imbalance_factor", type=float, default=0.01,
                   help="Exponential imbalance factor (lower = more imbalanced)")
    p.add_argument("--data_root", type=str, default="./data")

    # model
    p.add_argument("--model", type=str, default="hae",
                   choices=["hae", "ae"],
                   help="Model type: 'hae' (hyperbolic) or 'ae' (plain autoencoder)")
    p.add_argument("--latent_dim", type=int, default=512)
    p.add_argument("--feature_size", type=int, default=512)
    p.add_argument("--curvature", type=float, default=-1.0,
                   help="Negative curvature k for the Poincaré ball (only for HAE)")

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
    """Return (train_dataset, test_dataset)."""
    cls = CIFAR10LT if args.dataset == "cifar10" else CIFAR100LT
    train_tf = make_cifar_lt_transform(image_size=32, train=True)
    test_tf = make_cifar_lt_transform(image_size=32, train=False)

    train_ds = cls(root=args.data_root, imbalance_factor=args.imbalance_factor,
                   train=True, transform=train_tf, download=True)
    test_ds = cls(root=args.data_root, imbalance_factor=args.imbalance_factor,
                  train=False, transform=test_tf, download=True)
    return train_ds, test_ds


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
    train_ds, test_ds = get_datasets(args)
    num_classes = 10 if args.dataset == "cifar10" else 100
    print(f"Dataset: {args.dataset}  |  classes: {num_classes}  |  "
          f"train: {len(train_ds)}  |  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, drop_last=False, pin_memory=True)


    # ---- Model ------------------------------------------------------------
    if args.model == "hae":
        model = HAECifar(
            num_classes=num_classes,
            latent_dim=args.latent_dim,
            feature_size=args.feature_size,
            curvature=args.curvature,
        ).to(device)
        print("Using HAECifar (hyperbolic autoencoder)")
    else:
        model = AECifar(
            num_classes=num_classes,
            latent_dim=args.latent_dim,
            feature_size=args.feature_size,
        ).to(device)
        print("Using AECifar (plain autoencoder)")
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

    # LPIPS (upsample to 64x64 for AlexNet compatibility)
    lpips_fn = None
    if args.lpips_lambda > 0:
        from criteria.lpips.lpips import LPIPS
        lpips_fn = LPIPS(net_type='alex', device=str(device)).to(device).eval()

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

            if args.model == "hae":
                recon, logits, z_hyp, z_euc, z_euc_dec = model(images)
            else:
                recon, z_euc = model(images)
                logits = None
                z_hyp = None
                z_euc_dec = None

            # --- Losses ---
            # L1 reconstruction (replaces MSE for sharper outputs)
            loss_recon = l1_loss_fn(recon, images)

            loss = loss_recon

            # Hyperbolic classification (only for HAE)
            loss_hyper = torch.tensor(0.0, device=device)
            if args.model == "hae":
                loss_hyper = F.nll_loss(logits, labels)
                loss = loss + lam_hyper * loss_hyper

            # LPIPS perceptual loss (upsample to 64x64)
            loss_lpips = torch.tensor(0.0, device=device)
            if lpips_fn is not None and args.lpips_lambda > 0:
                recon_up = F.interpolate(recon, size=64, mode='bilinear', align_corners=False)
                images_up = F.interpolate(images, size=64, mode='bilinear', align_corners=False)
                loss_lpips = lpips_fn(recon_up, images_up)
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

            # Reverse / cycle-consistency loss: MSE(z_euc, z_euc_dec) (only for HAE)
            loss_reverse = torch.tensor(0.0, device=device)
            if args.model == "hae" and args.reverse_lambda > 0:
                loss_reverse = F.mse_loss(z_euc, z_euc_dec)
                loss = loss + args.reverse_lambda * loss_reverse

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            # stats
            if args.model == "hae":
                pred = logits.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
            epoch_loss += loss.item()
            global_step += 1

            # --- Logging ---
            if global_step % args.log_interval == 0:
                with torch.no_grad():
                    z_norms = z_euc.norm(dim=-1)
                    if z_hyp is not None:
                        z_hyp_norms = z_hyp.norm(dim=-1)  # should stay < 1.0 in Poincaré ball

                
                lr_now = scheduler.get_last_lr()[0]
                acc_str = f"acc={100.*correct/total:.1f}%" if args.model == "hae" else "" 
                print(f"[Epoch {epoch+1}/{args.num_epochs}  step {global_step}]  "
                      f"loss={loss.item():.4f}  recon={loss_recon.item():.4f}  "
                      f"hyper={loss_hyper.item():.4f}  "
                      f"lpips={loss_lpips.item():.4f}  "
                      f"ssim={loss_ssim.item():.4f}  "
                      f"reverse={loss_reverse.item():.4f}  "
                      f"{acc_str} lr={lr_now:.2e}")
                print(f"  z_euc norm — mean: {z_norms.mean():.3f}  max: {z_norms.max():.3f}")
                if z_hyp is not None:
                    print(f"  z_hyp norm — mean: {z_hyp_norms.mean():.3f}  max: {z_hyp_norms.max():.3f}")
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/loss_recon", loss_recon.item(), global_step)
                writer.add_scalar("train/loss_hyper", loss_hyper.item(), global_step)
                writer.add_scalar("train/loss_lpips", loss_lpips.item(), global_step)
                writer.add_scalar("train/loss_ssim", loss_ssim.item(), global_step)
                writer.add_scalar("train/loss_reverse", loss_reverse.item(), global_step)
                if args.model == "hae":
                    writer.add_scalar("train/accuracy", 100. * correct / total, global_step)
                writer.add_scalar("train/lr", lr_now, global_step)
                writer.add_scalar("train/lam_hyper", lam_hyper, global_step)
                writer.add_scalar("train/z_euc_norm_mean", z_norms.mean(), global_step)
                writer.add_scalar("train/z_euc_norm_max", z_norms.max(), global_step)
                if z_hyp is not None:
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
                        "train/accuracy": (100. * correct / total) if args.model=="hae" else 0,
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
                if args.model=="hae":
                    val_loss = validate(model, test_loader, device, l1_loss_fn, writer,
                                    global_step, lam_hyper)
                else:
                    val_loss = validate_ae(model, test_loader, device, l1_loss_fn, writer, global_step)
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
        acc_str = f"train_acc={100.*correct/total:.1f}%" if args.model =="hae" else ""
        print(f"Epoch {epoch+1} done in {dt:.1f}s  "
              f"avg_loss={epoch_loss/len(train_loader):.4f}  " + acc_str)

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
def validate(model, loader, device, l1_loss_fn, writer, global_step, lam_hyper):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        recon, logits, z_hyp, z_euc, z_euc_dec = model(images)

        loss_recon = l1_loss_fn(recon, images)
        loss_hyper = F.nll_loss(logits, labels)
        loss = loss_recon + lam_hyper * loss_hyper

        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    print(f"  [VAL step {global_step}]  loss={avg_loss:.4f}  acc={acc:.1f}%")
    writer.add_scalar("val/loss", avg_loss, global_step)
    writer.add_scalar("val/accuracy", acc, global_step)
    return avg_loss

@torch.no_grad()
def validate_ae(model, loader, device, l1_loss_fn, writer, global_step):
    model.eval()
    total_loss = 0.0
    total = 0

    for images, _ in loader:
        images = images.to(device)
        recon, z_euc = model(images)
        loss_recon = l1_loss_fn(recon, images)
        total_loss += loss_recon.item() * images.size(0)
        total += images.size(0)

    avg_loss = total_loss / total
    print(f"  [VAL step {global_step}]  loss={avg_loss:.4f}")
    writer.add_scalar("val/loss", avg_loss, global_step)
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
