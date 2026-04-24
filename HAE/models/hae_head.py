"""Shared hyperbolic/Euclidean head used by HAECifar and HAEImageNet.

When |curvature| < EUCLIDEAN_EPS we build a plain Linear + Linear classifier
so the model is a strict Euclidean baseline (no Möbius, no logmap0, no
HyperbolicMLR — the latter divides by √c and would NaN at c=0).

Forward returns (logits [log-softmax], z_hyp, z_euc_dec) where:
  - hyperbolic mode: z_hyp in the Poincaré ball, z_euc_dec = logmap0(z_hyp)
  - euclidean  mode: z_hyp = z_euc_dec = the linear projection of z_euc

Downstream code treats z_hyp as the "geometry-space" latent and z_euc_dec as
the decoder input — both interpretations are consistent with the Euclidean
limit of the stereographic model (logmap0 → identity as k→0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt.manifolds.stereographic.math as gmath

from models.hyper_nets import MobiusLinear, HyperbolicMLR


EUCLIDEAN_EPS = 1e-6


class GeometryHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        feature_size: int,
        num_classes: int,
        curvature: float,
        max_norm: float = 0.95,
    ):
        super().__init__()
        self.curvature_value = float(curvature)
        self.euclidean = abs(self.curvature_value) < EUCLIDEAN_EPS
        self.max_norm = max_norm
        # Buffer so .to(device) moves it alongside the module (used by logmap0).
        self.register_buffer(
            "curvature",
            torch.tensor(curvature, dtype=torch.float32),
        )

        if self.euclidean:
            self.proj = nn.Linear(latent_dim, feature_size)
            self.classifier = nn.Linear(feature_size, num_classes)
        else:
            self.hyperbolic_linear = MobiusLinear(
                latent_dim, feature_size,
                hyperbolic_input=False, hyperbolic_bias=True,
                nonlin=None, k=curvature,
            )
            self.mlr = HyperbolicMLR(
                ball_dim=feature_size, n_classes=num_classes,
                c=abs(curvature),
            )

    def forward(self, z_euc: torch.Tensor):
        """Returns (logits, z_hyp, z_euc_dec). logits are log-softmaxed."""
        if self.euclidean:
            z = self.proj(z_euc)
            logits = F.log_softmax(self.classifier(z), dim=-1)
            return logits, z, z

        z_hyp = self.hyperbolic_linear(z_euc)
        # Keep points off the absolute edge (1.0) for numerical stability.
        norm = z_hyp.norm(dim=-1, keepdim=True)
        cond = norm > self.max_norm
        z_hyp = torch.where(cond, z_hyp * (self.max_norm / (norm + 1e-6)), z_hyp)
        logits = F.log_softmax(self.mlr(z_hyp, self.mlr.c), dim=-1)
        z_euc_dec = gmath.logmap0(z_hyp, k=self.curvature)
        return logits, z_hyp, z_euc_dec
