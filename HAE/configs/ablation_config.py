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

    # --- Class conditioning in FM -----------------------------------------
    # none        = unconditional FM
    # conditional = class embedding added to time embedding
    # cfg         = classifier-free guidance with 10% label dropout
    fm_conditioning: str = "none"
    cfg_scale: float = 1.0              # only when fm_conditioning=cfg
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
    """§3.2 — Curvature sweep."""
    vals = [0.1, 0.5, 1.0, 2.0, 3.0]
    return [AblationConfig(curvature=-c, name=f"curv_{c}") for c in vals]


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
    """§3.5 — Class conditioning in FM."""
    cfgs = [
        AblationConfig(fm_conditioning="none", name="cond_none"),
        AblationConfig(fm_conditioning="conditional", name="cond_class"),
    ]
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


ALL_SWEEPS = {
    "geometry": geometry_sweep,
    "curvature": curvature_sweep,
    "imbalance": imbalance_sweep,
    "hyper_lambda": hyper_lambda_sweep,
    "conditioning": conditioning_sweep,
    "latent_dim": latent_dim_sweep,
    "reverse_lambda": reverse_lambda_sweep,
}
