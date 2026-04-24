"""Hyperbolic (or Euclidean) Autoencoder for CIFAR-10 / CIFAR-100-LT.

When |curvature| < EUCLIDEAN_EPS the geometry head falls back to plain
Linear layers — this is the clean Euclidean baseline used by the geometry
ablation. See models/hae_head.py for the branching logic.
"""

import torch
import torch.nn as nn

from models.cifar_backbone import CIFAREncoder, CIFARDecoder
from models.hae_head import GeometryHead


class HAECifar(nn.Module):
    """Hyperbolic Autoencoder for CIFAR-sized (32x32) images.

    Forward::

        x → encoder → z_euc → GeometryHead → (logits, z_hyp, z_euc_dec)
                                                                │
                                                            decoder → recon

    Returns (recon, logits, z_hyp, z_euc, z_euc_dec).
    """

    def __init__(
        self,
        num_classes: int = 10,
        latent_dim: int = 512,
        feature_size: int = 512,
        curvature: float = -1.0,
    ):
        super().__init__()
        self.encoder = CIFAREncoder(latent_dim=latent_dim)
        self.decoder = CIFARDecoder(latent_dim=feature_size)
        self.head = GeometryHead(
            latent_dim=latent_dim,
            feature_size=feature_size,
            num_classes=num_classes,
            curvature=curvature,
        )

    def forward(self, x: torch.Tensor):
        z_euc = self.encoder(x)
        logits, z_hyp, z_euc_dec = self.head(z_euc)
        recon = self.decoder(z_euc_dec)
        return recon, logits, z_hyp, z_euc, z_euc_dec


if __name__ == "__main__":
    x = torch.randn((2, 3, 32, 32))
    for k in (-1.0, 0.0):
        m = HAECifar(curvature=k)
        recon, logits, z_hyp, z_euc, z_euc_dec = m(x)
        print(f"k={k}: recon={recon.shape}, logits={logits.shape}, "
              f"z_hyp={z_hyp.shape}, z_euc_dec={z_euc_dec.shape}, "
              f"euclidean_head={m.head.euclidean}")
