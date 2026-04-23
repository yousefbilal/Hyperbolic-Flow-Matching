"""
Hyperbolic Autoencoder for CIFAR-10 / CIFAR-100-LT.

Replaces the pSp + StyleGAN backbone with a lightweight CNN encoder-decoder.
All hyperbolic geometry components (MobiusLinear, HyperbolicMLR, gmath ops)
are kept unchanged from the original HAE codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt.manifolds.stereographic.math as gmath

from models.cifar_backbone import CIFAREncoder, CIFARDecoder
from models.hyper_nets import MobiusLinear, HyperbolicMLR


class AECifar(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        latent_dim: int = 512,
        feature_size: int = 512,
    ):
        super().__init__()

        # ---- Image backbone (CIFAR-specific) ----
        self.encoder = CIFAREncoder(latent_dim=latent_dim)
        self.decoder = CIFARDecoder(latent_dim=feature_size)

        # ---- Hyperbolic layers (unchanged from original HAE) ----

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input images  (B, 3, 32, 32)

        Returns:
            recon:     reconstructed images  (B, 3, 32, 32)
        """
        # Encode to flat Euclidean vector
        z_euc = self.encoder(x)                         # (B, latent_dim)

        recon = self.decoder(z_euc)              # (B, 3, 32, 32)

        return recon, z_euc

class HAECifar(nn.Module):
    """Hyperbolic Autoencoder for CIFAR-sized (32x32) images.

    Forward pass::

        x  →  encoder  →  z_euc  →  MobiusLinear  →  z_hyp  →  MLR  →  logits
                                                        ↓
                                                   logmap0  →  z_euc_dec  →  decoder  →  recon

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
        self.curvature = torch.tensor(curvature, dtype=torch.float32)

        # ---- Image backbone (CIFAR-specific) ----
        self.encoder = CIFAREncoder(latent_dim=latent_dim)
        self.decoder = CIFARDecoder(latent_dim=feature_size)

        # ---- Hyperbolic layers (unchanged from original HAE) ----
        self.hyperbolic_linear = MobiusLinear(
            latent_dim,
            feature_size,
            hyperbolic_input=False,
            hyperbolic_bias=True,
            nonlin=None,
            k=curvature ,
        )
        self.mlr = HyperbolicMLR(
            ball_dim=feature_size,
            n_classes=num_classes,
            c=abs(curvature),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input images  (B, 3, 32, 32)

        Returns:
            recon:     reconstructed images  (B, 3, 32, 32)
            logits:    log-softmax class logits  (B, num_classes)
            z_hyp:     hyperbolic latent codes  (B, feature_size)
            z_euc:     Euclidean encoder output  (B, latent_dim)
            z_euc_dec: logmap0(z_hyp) fed to decoder  (B, feature_size)
        """
        # Encode to flat Euclidean vector
        z_euc = self.encoder(x)                         # (B, latent_dim)

        # Map to Poincaré ball
        z_hyp = self.hyperbolic_linear(z_euc)

# Safety clamp: keep points away from the absolute edge (1.0)
        norm = z_hyp.norm(dim=-1, keepdim=True)
        max_norm = 0.95 # Stay slightly away from the boundary
        cond = norm > max_norm
        z_hyp_clamped = torch.where(cond, z_hyp * (max_norm / (norm + 1e-6)), z_hyp)

        logits = F.log_softmax(self.mlr(z_hyp_clamped, self.mlr.c), dim=-1)
        z_euc_dec = gmath.logmap0(z_hyp_clamped, k=self.curvature)
        recon = self.decoder(z_euc_dec)              # (B, 3, 32, 32)

        return recon, logits, z_hyp_clamped, z_euc, z_euc_dec


if __name__ == "__main__":
    x = torch.randn((2, 3, 32, 32))
    model = HAECifar()
    recon, logits, z_hyp, z_euc, z_euc_dec = model(x)
    print(f"recon: {recon.shape}, logits: {logits.shape}, "
          f"z_hyp: {z_hyp.shape}, z_euc: {z_euc.shape}, z_euc_dec: {z_euc_dec.shape}")