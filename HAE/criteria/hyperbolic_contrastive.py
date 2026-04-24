"""Hyperbolic contrastive losses for long-tail classification.

All three operate on z_hyp already inside the Poincaré ball (‖z‖ < 1):

    supcon_hyp        Supervised contrastive with Poincaré distance kernel.
                      Frequency-free; imbalance only enters via batch composition.

    align_unif_hyp    Alignment (pull same-class means together) + uniformity
                      (-log E exp(-t·d_H)) on the ball. Frequency-free.

    supcon_hyp_radius supcon_hyp + a per-class radius prior
                      ‖mean_c z_hyp‖ → r_c  where r_c ∝ log(N_max / n_c),
                      scaled into [0, R_MAX]. Head classes near origin,
                      tail classes near boundary.

Poincaré distance reuses the numerically-stable helper in contrastive_loss.py.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn

from .contrastive_loss import poincare_distance


# ---------------------------------------------------------------------------
# Single-view supervised contrastive with Poincaré distance
# ---------------------------------------------------------------------------

class PoincareSupCon(nn.Module):
    """SupCon (Khosla et al. 2020) with -d_H² / τ as the similarity kernel.

    One view per sample (no augmentation pair required). Positives are
    same-class, negatives are different-class within the batch.
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_hyp: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = z_hyp.size(0)
        device = z_hyp.device

        d = poincare_distance(z_hyp, z_hyp)                 # (B, B)
        logits = -(d.pow(2)) / self.temperature
        # numerical stability
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # same-class mask, minus diagonal
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.T).float()
        self_mask = torch.eye(B, device=device)
        pos_mask = pos_mask - self_mask
        valid_mask = 1.0 - self_mask

        exp_logits = torch.exp(logits) * valid_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_per_anchor = pos_mask.sum(dim=1)
        # anchors with no positives contribute zero
        has_pos = pos_per_anchor > 0
        if not has_pos.any():
            return torch.zeros((), device=device)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[has_pos] \
                            / pos_per_anchor[has_pos]
        return -mean_log_prob_pos.mean()


# ---------------------------------------------------------------------------
# Alignment + uniformity on the Poincaré ball (Wang & Isola, 2020)
# ---------------------------------------------------------------------------

class PoincareAlignUniformity(nn.Module):
    """Supervised alignment + unsupervised uniformity, both with d_H.

    alignment   = E_{(i,j): y_i=y_j, i!=j} d_H(z_i, z_j)^α
    uniformity  = log E_{i!=j} exp(-t * d_H(z_i, z_j)^2)

    Total: alignment + unif_weight * uniformity.
    """
    def __init__(self, alpha: float = 2.0, t: float = 2.0,
                 unif_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.t = t
        self.unif_weight = unif_weight

    def forward(self, z_hyp: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = z_hyp.size(0)
        device = z_hyp.device
        d = poincare_distance(z_hyp, z_hyp)                 # (B, B)

        # alignment
        labels = labels.view(-1, 1)
        same = (labels == labels.T).float()
        self_mask = torch.eye(B, device=device)
        pos_mask = same - self_mask
        n_pos = pos_mask.sum().clamp(min=1.0)
        align = (pos_mask * d.pow(self.alpha)).sum() / n_pos

        # uniformity (all off-diagonal pairs)
        off = 1.0 - self_mask
        unif = torch.log(
            (off * torch.exp(-self.t * d.pow(2))).sum() / off.sum().clamp(min=1.0)
            + 1e-12
        )
        return align + self.unif_weight * unif


# ---------------------------------------------------------------------------
# Radius prior: head classes near origin, tail classes near boundary
# ---------------------------------------------------------------------------

class PoincareRadiusPrior(nn.Module):
    """MSE( ‖mean_c z_hyp‖ , r_c ) where r_c ∝ log(N_max/n_c).

    Targets are precomputed from training-set class counts so this is a
    static per-class regulariser. Scaled so the largest target = r_max.
    """
    def __init__(self, class_counts: dict, num_classes: int,
                 r_max: float = 0.9):
        super().__init__()
        counts = np.array([class_counts[c] for c in range(num_classes)],
                          dtype=np.float64)
        n_max = counts.max()
        raw = np.log(n_max / np.maximum(counts, 1.0))       # 0 for head, grows for tail
        if raw.max() > 0:
            targets = raw / raw.max() * r_max
        else:
            targets = np.zeros_like(raw)
        self.register_buffer(
            "radius_targets",
            torch.as_tensor(targets, dtype=torch.float32),
        )

    def forward(self, z_hyp: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = z_hyp.device
        norms = z_hyp.norm(dim=-1)                          # (B,)
        targets = self.radius_targets.to(device)[labels]    # (B,)
        return torch.mean((norms - targets).pow(2))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_contrastive(mode: str, temperature: float,
                      class_counts: dict | None,
                      num_classes: int | None):
    """Return (main_loss_module, radius_prior_or_None) for the given mode."""
    if mode == "none":
        return None, None
    if mode == "supcon_hyp":
        return PoincareSupCon(temperature=temperature), None
    if mode == "align_unif_hyp":
        return PoincareAlignUniformity(), None
    if mode == "supcon_hyp_radius":
        assert class_counts is not None and num_classes is not None
        return (PoincareSupCon(temperature=temperature),
                PoincareRadiusPrior(class_counts, num_classes))
    raise ValueError(f"unknown contrastive_mode: {mode}")
