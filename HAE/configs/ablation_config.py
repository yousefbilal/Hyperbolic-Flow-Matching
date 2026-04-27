"""Ablation configuration for Hyperbolic Flow Matching experiments.

This module defines all ablation knobs as a dataclass so they can be
serialised to / from YAML sweep files and passed to training + inference
scripts.
"""

from dataclasses import dataclass, asdict
from typing import List
import yaml


@dataclass
class AblationConfig:
    """One experiment configuration (one row of an ablation table)."""

    # --- Geometry ablation -------------------------------------------------
    # full              = HAE (hyp) + RFM (hyp)     [proposed model]
    # euclidean_fm      = HAE (hyp) + RFM (euclidean)
    # euclidean_ae_hyp_fm = AE (euclidean, no L_hyper) + RFM (hyp, via expmap0)
    # hyp_ae_euclidean_fm = HAE (hyp) + FM on logmap0(z_hyp) flat vectors
    geometry_mode: str = "full"

    # --- Dataset -----------------------------------------------------------
    dataset: str = "cifar10"            # cifar10 | cifar100
    imbalance_factor: float = 0.01      # maps to IR=100 (lower = more imbalance)

    # --- Model hyper-parameters -------------------------------------------
    latent_dim: int = 512
    feature_size: int = 512
    curvature: float = -1.0             # negative k for the Poincaré ball

    # --- Loss weights ------------------------------------------------------
    hyperbolic_lambda: float = 0.05
    hyper_warmup_epochs: int = 20
    lpips_lambda: float = 0.8
    ssim_lambda: float = 0.1
    reverse_lambda: float = 0.0
    ms_ssim_lambda: float = 0.0
    kl_lambda: float = 0.0          # > 0 turns HAECifar into a VAE (CIFAR only)

    # --- Imbalance handling ------------------------------------------------
    # Sampler: instance (natural) | balanced (1/n_c) | sqrt (1/sqrt(n_c))
    sampler: str = "instance"
    # Class-weighted NLL: none | inv_freq | effective_number
    class_weighting: str = "none"
    eff_num_beta: float = 0.9999

    # --- Contrastive loss (hyperbolic) ------------------------------------
    # none              = standard NLL head only (current behaviour)
    # supcon_hyp        = supervised contrastive w/ Poincaré distance kernel
    # align_unif_hyp    = alignment + uniformity on the ball (freq-free, unsup negatives)
    # supcon_hyp_radius = supcon + per-class radius prior ‖z_c‖ ∝ log(N_max/n_c)
    contrastive_mode: str = "none"
    contrastive_lambda: float = 0.0
    contrastive_tau: float = 0.5
    radius_prior_lambda: float = 0.0    # only used when contrastive_mode = *_radius

    # --- Encoder backbone (for ImageNet-LT) -------------------------------
    # cnn_cifar   = small CIFAR CNN (current)
    # sd_vae      = frozen Stable Diffusion KL-f8 VAE (ImageNet-LT only)
    encoder_backbone: str = "cnn_cifar"
    # How to collapse SD-VAE's 4xH/8xW/8 spatial latent → flat vector
    # flatten | mean_pool | cls_token
    sd_vae_pool: str = "mean_pool"

    # --- Class conditioning in FM -----------------------------------------
    # none = unconditional FM
    # cfg  = classifier-free guidance with 10% label dropout (Ho & Salimans 2022)
    fm_conditioning: str = "none"
    cfg_scale: float = 1.0              # inference-time guidance strength
    cfg_label_dropout: float = 0.1

    # --- Training ----------------------------------------------------------
    hae_epochs: int = 200
    hae_batch_size: int = 128
    hae_lr: float = 3e-4
    rfm_iterations: int = 100000
    rfm_batch_size: int = 128
    rfm_lr: float = 1e-4

    # --- Paths (filled in by sweep runner) ---------------------------------
    exp_dir: str = ""
    data_root: str = "./data"

    # --- Meta --------------------------------------------------------------
    seed: int = 0
    name: str = ""                      # human-readable experiment name

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> "AblationConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)


# ---------------------------------------------------------------------------
# Pre-built sweep definitions (convenience for common ablation tables)
# ---------------------------------------------------------------------------

def geometry_sweep() -> List[AblationConfig]:
    """§3.1 — Geometry ablations."""
    modes = ["full", "euclidean_fm", "euclidean_ae_hyp_fm", "hyp_ae_euclidean_fm"]
    return [AblationConfig(geometry_mode=m, name=f"geom_{m}") for m in modes]


def curvature_sweep() -> List[AblationConfig]:
    """§3.2 — Curvature sweep. curvature=0 = Euclidean baseline via GeometryHead."""
    cfgs = [AblationConfig(curvature=0.0, name="curv_euclidean")]
    for c in [0.1, 0.5, 1.0, 2.0, 3.0]:
        cfgs.append(AblationConfig(curvature=-c, name=f"curv_{c}"))
    return cfgs


def imbalance_sweep() -> List[AblationConfig]:
    """§3.3 — Long-tail sensitivity."""
    # imbalance_factor → imbalance_ratio mapping (approx):
    #   0.1  → IR≈10,  0.02 → IR≈50,  0.01 → IR≈100,  0.005 → IR≈200
    factors = [0.1, 0.02, 0.01, 0.005]
    return [AblationConfig(imbalance_factor=f, name=f"imb_{f}") for f in factors]


def hyper_lambda_sweep() -> List[AblationConfig]:
    """§3.4 — Hyperbolic loss weight."""
    vals = [0.0, 0.01, 0.05, 0.1, 0.5]
    return [AblationConfig(hyperbolic_lambda=v, name=f"lam_{v}") for v in vals]


def conditioning_sweep() -> List[AblationConfig]:
    """§3.5 — Class conditioning in FM. Only CFG vs unconditional."""
    cfgs = [AblationConfig(fm_conditioning="none", name="cond_none")]
    for w in [1.0, 2.0, 4.0, 7.5]:
        cfgs.append(AblationConfig(
            fm_conditioning="cfg", cfg_scale=w, name=f"cfg_w{w}"
        ))
    return cfgs


def latent_dim_sweep() -> List[AblationConfig]:
    """§3.6 — Latent dimension."""
    dims = [128, 256, 512]
    return [AblationConfig(latent_dim=d, feature_size=d, name=f"dim_{d}") for d in dims]


def reverse_lambda_sweep() -> List[AblationConfig]:
    """§3.7 — Reverse / cycle-consistency loss weight."""
    vals = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    return [AblationConfig(reverse_lambda=v, name=f"rev_{v}") for v in vals]


def sampler_sweep() -> List[AblationConfig]:
    """§3.8 — Resampling strategies for long-tail."""
    return [AblationConfig(sampler=s, name=f"samp_{s}")
            for s in ["instance", "sqrt", "balanced"]]


def class_weighting_sweep() -> List[AblationConfig]:
    """§3.9 — Frequency-weighted NLL variants."""
    cfgs = [AblationConfig(class_weighting="none", name="cw_none"),
            AblationConfig(class_weighting="inv_freq", name="cw_inv_freq")]
    for b in [0.99, 0.999, 0.9999]:
        cfgs.append(AblationConfig(class_weighting="effective_number",
                                   eff_num_beta=b, name=f"cw_eff_{b}"))
    return cfgs


def contrastive_sweep() -> List[AblationConfig]:
    """§3.10 — Hyperbolic contrastive AS AN ALTERNATIVE to the NLL head.

    In each non-baseline config we zero out hyperbolic_lambda so the
    contrastive term is the *only* class-separation signal — otherwise the
    comparison is confounded by NLL still training in parallel.

    First three modes are frequency-free; supcon_hyp_radius uses class counts
    to place heads near origin and tails near the ball boundary.
    """
    base_lam = 0.1
    cfgs = [AblationConfig(contrastive_mode="none", name="con_none")]
    for m in ["supcon_hyp", "align_unif_hyp", "supcon_hyp_radius"]:
        cfgs.append(AblationConfig(
            contrastive_mode=m,
            contrastive_lambda=base_lam,
            radius_prior_lambda=(0.05 if m.endswith("_radius") else 0.0),
            hyperbolic_lambda=0.0,          # disable NLL head — alternative, not additive
            name=f"con_{m}",
        ))
    return cfgs


def encoder_backbone_sweep() -> List[AblationConfig]:
    """§3.11 — Encoder choice.

    cnn_cifar is used for CIFAR-LT; sd_vae (flatten) is used for ImageNet-LT.
    mean_pool / cls_token would require a learned upsampler back to the VAE's
    (4, 32, 32) spatial latent before the frozen decoder — left as future work.
    """
    return [
        AblationConfig(dataset="cifar10",     encoder_backbone="cnn_cifar",
                       name="enc_cnn_cifar"),
        AblationConfig(dataset="imagenet_lt", encoder_backbone="sd_vae",
                       sd_vae_pool="flatten",
                       hae_batch_size=64, hae_epochs=40,
                       name="enc_sdvae_imagenet"),
    ]


ALL_SWEEPS = {
    "geometry": geometry_sweep,
    "curvature": curvature_sweep,
    "imbalance": imbalance_sweep,
    "hyper_lambda": hyper_lambda_sweep,
    "conditioning": conditioning_sweep,
    "latent_dim": latent_dim_sweep,
    "reverse_lambda": reverse_lambda_sweep,
    "sampler": sampler_sweep,
    "class_weighting": class_weighting_sweep,
    "contrastive": contrastive_sweep,
    "encoder_backbone": encoder_backbone_sweep,
}
