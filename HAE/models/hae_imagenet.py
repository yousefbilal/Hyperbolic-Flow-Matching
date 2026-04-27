"""Hyperbolic Autoencoder for ImageNet-LT using a frozen Stable Diffusion VAE.

Pipeline:

    x (B,3,256,256)
      │
      │ SD-VAE encode (frozen)     ↓ * scaling_factor
      ▼
    z_spatial (B,4,32,32) ── flatten ── Linear → z_euc (B,latent_dim)
                                                    │
                                                    │ MobiusLinear
                                                    ▼
                                                z_hyp → MLR → logits
                                                    │
                                                    │ logmap0
                                                    ▼
                                              z_euc_dec (B,latent_dim)
                                                    │
                                                    │ Linear → reshape
                                                    ▼
                                           z_spatial_dec (B,4,32,32)
                                                    │
                                                    │ / scaling_factor + SD-VAE decode (frozen)
                                                    ▼
                                                 recon (B,3,256,256)

Only the two projection layers + the hyperbolic head are trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.hae_head import GeometryHead


SD_VAE_NAME = "stabilityai/sd-vae-ft-mse"
SD_VAE_LATENT_CHANNELS = 4
SD_VAE_SPATIAL = 32          # for 256x256 input  (/8 downsample)
SD_VAE_SCALING = 0.18215     # SD v1-style latent scale factor


class FrozenSDVae(nn.Module):
    """Thin wrapper around the diffusers AutoencoderKL in eval mode.

    - encode returns scaled latent (deterministic: posterior mean, not sampled)
    - decode inverts the scaling
    All parameters have requires_grad=False and the module is kept in eval()
    so BN/dropout stay frozen.
    """
    def __init__(self, name: str = SD_VAE_NAME):
        super().__init__()
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(name)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        # Keep the VAE in eval regardless of outer train() calls.
        super().train(mode)
        self.vae.eval()
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # posterior mean (deterministic); (B, 4, 32, 32) for 256² input
        latent = self.vae.encode(x).latent_dist.mean
        return latent * SD_VAE_SCALING

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # NB: gradient flows through decode so the upstream projection can learn,
        # but the decoder's own params stay frozen via requires_grad=False.
        return self.vae.decode(latent / SD_VAE_SCALING).sample


class HAEImageNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        latent_dim: int = 512,
        feature_size: int = 512,
        curvature: float = -1.0,
        vae_name: str = SD_VAE_NAME,
        spatial: int = SD_VAE_SPATIAL,
        channels: int = SD_VAE_LATENT_CHANNELS,
    ):
        super().__init__()
        self.spatial = spatial
        self.channels = channels
        self.flat_dim = channels * spatial * spatial

        # Frozen SD-VAE
        self.vae = FrozenSDVae(vae_name)

        # Learned projections between VAE latent and Euclidean feature
        self.proj_enc = nn.Linear(self.flat_dim, latent_dim)
        self.proj_dec = nn.Linear(feature_size, self.flat_dim)

        # Shared geometry head (hyperbolic or Euclidean depending on curvature)
        self.head = GeometryHead(
            latent_dim=latent_dim,
            feature_size=feature_size,
            num_classes=num_classes,
            curvature=curvature,
        )

    def forward(self, x: torch.Tensor):
        B = x.size(0)

        # Frozen encode (no grad through VAE encoder)
        z_spatial = self.vae.encode(x)                       # (B, 4, 32, 32)
        z_flat = z_spatial.reshape(B, -1)                    # (B, 4096)
        z_euc = self.proj_enc(z_flat)                        # (B, latent_dim)

        # Geometry head (hyperbolic or Euclidean)
        logits, z_hyp, z_euc_dec = self.head(z_euc)

        # Back to spatial latent → frozen decode
        z_flat_dec = self.proj_dec(z_euc_dec)                # (B, 4096)
        z_spatial_dec = z_flat_dec.reshape(B, self.channels, self.spatial, self.spatial)
        recon = self.vae.decode(z_spatial_dec)               # (B, 3, 256, 256)

        # ImageNet path is currently AE-only; return zero KL for interface parity
        kl = torch.zeros((), device=x.device)
        return recon, logits, z_hyp, z_euc, z_euc_dec, kl


if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)
    m = HAEImageNet(num_classes=1000).eval()
    with torch.no_grad():
        recon, logits, z_hyp, z_euc, z_euc_dec = m(x)
    print("recon:", recon.shape, "logits:", logits.shape,
          "z_hyp:", z_hyp.shape, "z_euc:", z_euc.shape)
