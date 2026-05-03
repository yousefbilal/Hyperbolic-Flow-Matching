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

        x → encoder → (z_euc, kl) → GeometryHead → (logits, z_hyp, z_euc_dec)
                                                                  │
                                                              decoder → recon

    Returns (recon, logits, z_hyp, z_euc, z_euc_dec, kl).

    `variational=True` makes the encoder a VAE: z_euc is sampled via reparam
    during training (mean at eval), and `kl` carries the KL term that the
    trainer multiplies by --kl_lambda. With variational=False, kl is a zero
    scalar and the model is a deterministic AE.
    """

    def __init__(
        self,
        num_classes: int = 10,
        latent_dim: int = 512,
        feature_size: int = 512,
        curvature: float = -1.0,
        variational: bool = False,
        image_size: int = 32,
    ):
        super().__init__()
        self.variational = variational
        self.image_size = int(image_size)
        self.encoder = CIFAREncoder(latent_dim=latent_dim, variational=variational,
                                    image_size=self.image_size)
        self.decoder = CIFARDecoder(latent_dim=feature_size,
                                    image_size=self.image_size)
        self.head = GeometryHead(
            latent_dim=latent_dim,
            feature_size=feature_size,
            num_classes=num_classes,
            curvature=curvature,
        )

    def forward(self, x: torch.Tensor):
        z_euc, kl = self.encoder(x)
        logits, z_hyp, z_euc_dec = self.head(z_euc)
        recon = self.decoder(z_euc_dec)
        return recon, logits, z_hyp, z_euc, z_euc_dec, kl


if __name__ == "__main__":
    x = torch.randn((2, 3, 32, 32))
    for k in (-1.0, 0.0):
        for variational in (False, True):
            m = HAECifar(curvature=k, variational=variational)
            recon, logits, z_hyp, z_euc, z_euc_dec, kl = m(x)
            print(f"k={k} variational={variational}: recon={recon.shape}, "
                  f"logits={logits.shape}, z_hyp={z_hyp.shape}, "
                  f"z_euc_dec={z_euc_dec.shape}, kl={kl.item():.4f}, "
                  f"euclidean_head={m.head.euclidean}")
