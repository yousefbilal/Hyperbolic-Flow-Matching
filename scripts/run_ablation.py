#!/usr/bin/env python
"""Ablation sweep runner for Hyperbolic Flow Matching.

Orchestrates the full pipeline for each ablation config:
    HAE train → export embeddings → RFM train → generate → compute metrics

Usage:
    conda activate dl
    python scripts/run_ablation.py \\
        --sweep geometry \\
        --base_exp_dir ./experiments/ablations \\
        --data_root ./data

    # Or run a single config from YAML:
    python scripts/run_ablation.py \\
        --config_yaml ./my_experiment.yaml \\
        --base_exp_dir ./experiments/custom
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "HAE"))
from configs.ablation_config import AblationConfig, ALL_SWEEPS


def parse_args():
    p = argparse.ArgumentParser(description="Run ablation sweep")
    p.add_argument("--sweep", type=str, default=None,
                   choices=list(ALL_SWEEPS.keys()),
                   help="Named sweep to run")
    p.add_argument("--config_yaml", type=str, default=None,
                   help="Single experiment config YAML file")
    p.add_argument("--base_exp_dir", type=str, required=True)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--rfm_dir", type=str,
                   default=os.path.join(os.path.dirname(__file__), "..", "riemannian-fm"),
                   help="Path to riemannian-fm/ directory")
    p.add_argument("--hae_dir", type=str,
                   default=os.path.join(os.path.dirname(__file__), "..", "HAE"),
                   help="Path to HAE/ directory")
    p.add_argument("--skip_hae", action="store_true", help="Skip HAE training")
    p.add_argument("--skip_rfm", action="store_true", help="Skip RFM training")
    p.add_argument("--skip_generate", action="store_true", help="Skip generation")
    p.add_argument("--dry_run", action="store_true", help="Print commands only")
    return p.parse_args()


def run_cmd(cmd: list, cwd: str, dry_run: bool):
    """Run a command, printing it first."""
    cmd_str = " ".join(cmd)
    print(f"\n{'='*60}")
    print(f"[CMD] {cmd_str}")
    print(f"[CWD] {cwd}")
    print(f"{'='*60}")
    if dry_run:
        print("  (dry run — skipping)")
        return 0
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"  !! Command FAILED with code {result.returncode}")
    return result.returncode


def run_experiment(cfg: AblationConfig, args):
    """Run one full experiment (HAE → export → RFM → generate)."""
    exp_name = cfg.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.base_exp_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save config
    cfg.exp_dir = exp_dir
    cfg.data_root = args.data_root
    cfg.to_yaml(os.path.join(exp_dir, "config.yaml"))

    hae_dir = os.path.abspath(args.hae_dir)
    rfm_dir = os.path.abspath(args.rfm_dir)
    hae_exp = os.path.join(exp_dir, "hae")
    rfm_exp = os.path.join(exp_dir, "rfm")
    emb_dir = os.path.join(exp_dir, "embeddings")
    gen_dir = os.path.join(exp_dir, "generated")

    print(f"\n{'#'*60}")
    print(f"# Experiment: {exp_name}")
    print(f"{'#'*60}")

    # ---- Step 1: Train HAE ------------------------------------------------
    if not args.skip_hae:
        hae_cmd = [
            sys.executable, "scripts/train_hae.py",
            "--dataset", cfg.dataset,
            "--imbalance_factor", str(cfg.imbalance_factor),
            "--curvature", str(cfg.curvature),
            "--latent_dim", str(cfg.latent_dim),
            "--feature_size", str(cfg.feature_size),
            "--hyperbolic_lambda", str(cfg.hyperbolic_lambda),
            "--hyper_warmup_epochs", str(cfg.hyper_warmup_epochs),
            "--ms_ssim_lambda", str(cfg.ms_ssim_lambda),
            "--kl_lambda", str(cfg.kl_lambda),
            "--num_epochs", str(cfg.hae_epochs),
            "--batch_size", str(cfg.hae_batch_size),
            "--lr", str(cfg.hae_lr),
            "--sampler", cfg.sampler,
            "--class_weighting", cfg.class_weighting,
            "--eff_num_beta", str(cfg.eff_num_beta),
            "--contrastive_mode", cfg.contrastive_mode,
            "--contrastive_lambda", str(cfg.contrastive_lambda),
            "--contrastive_tau", str(cfg.contrastive_tau),
            "--radius_prior_lambda", str(cfg.radius_prior_lambda),
            "--exp_dir", hae_exp,
            "--data_root", args.data_root,
        ]
        rc = run_cmd(hae_cmd, cwd=hae_dir, dry_run=args.dry_run)
        if rc != 0 and not args.dry_run:
            print(f"HAE training failed for {exp_name}, skipping rest.")
            return

    # ---- Step 2: Export embeddings ----------------------------------------
    hae_ckpt = os.path.join(hae_exp, "checkpoints", "best_model.pt")
    if not args.skip_hae:
        export_cmd = [
            sys.executable, "scripts/export_embeddings.py",
            "--checkpoint", hae_ckpt,
            "--dataset", cfg.dataset,
            "--imbalance_factor", str(cfg.imbalance_factor),
            "--data_root", args.data_root,
            "--output_dir", emb_dir,
        ]
        rc = run_cmd(export_cmd, cwd=hae_dir, dry_run=args.dry_run)
        if rc != 0 and not args.dry_run:
            print(f"Embedding export failed for {exp_name}, skipping rest.")
            return

    # ---- Step 3: Train RFM -----------------------------------------------
    is_euclidean = abs(float(cfg.curvature)) < 1e-6
    is_imagenet = cfg.dataset == "imagenet_lt"
    if is_imagenet:
        exp_name_rfm = "imagenet_hae_euclidean" if is_euclidean else "imagenet_hae"
    else:
        exp_name_rfm = "cifar_hae_euclidean" if is_euclidean else "cifar_hae"
    num_classes = {"cifar10": 10, "cifar100": 100, "imagenet_lt": 1000}.get(
        cfg.dataset, 10)

    if not args.skip_rfm:
        z_hyp_path = os.path.join(emb_dir, "z_hyp.pt")
        labels_path = os.path.join(emb_dir, "labels.pt")
        rfm_cmd = [
            sys.executable, "train.py",
            f"experiment={exp_name_rfm}",
            f"images_datadir={z_hyp_path}",
            f"images_labels={labels_path}",
            f"num_classes={num_classes}",
            f"fm_conditioning={cfg.fm_conditioning}",
            f"cfg_label_dropout={cfg.cfg_label_dropout}",
            f"optim.num_iterations={cfg.rfm_iterations}",
            f"optim.batch_size={cfg.rfm_batch_size}",
            f"optim.lr={cfg.rfm_lr}",
            f"hydra.run.dir={rfm_exp}",
        ]
        rc = run_cmd(rfm_cmd, cwd=rfm_dir, dry_run=args.dry_run)
        if rc != 0 and not args.dry_run:
            print(f"RFM training failed for {exp_name}, skipping generation.")
            return

    # ---- Step 4: Generate -------------------------------------------------
    if not args.skip_generate:
        rfm_ckpt = os.path.join(rfm_exp, "checkpoints", "last.ckpt")
        gen_cmd = [
            sys.executable, "generate_cifar.py",
            "--rfm_checkpoint", rfm_ckpt,
            "--hae_checkpoint", hae_ckpt,
            "--n_samples", "1000",
            "--output_dir", gen_dir,
            "--curvature", str(cfg.curvature),
            "--cfg_scale", str(cfg.cfg_scale),
        ]
        rc = run_cmd(gen_cmd, cwd=rfm_dir, dry_run=args.dry_run)

    print(f"\nExperiment {exp_name} done.")


def main():
    args = parse_args()

    if args.config_yaml:
        configs = [AblationConfig.from_yaml(args.config_yaml)]
    elif args.sweep:
        configs = ALL_SWEEPS[args.sweep]()
    else:
        print("ERROR: Specify either --sweep or --config_yaml")
        sys.exit(1)

    print(f"Running {len(configs)} experiment(s)...")
    for i, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(configs)}: {cfg.name}")
        print(f"{'='*60}")
        run_experiment(cfg, args)

    print(f"\n\nAll {len(configs)} experiments complete.")


if __name__ == "__main__":
    main()
