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
TAESD_NAME = "madebyollin/taesd"
SD_VAE_LATENT_CHANNELS = 4
SD_VAE_SPATIAL_DEFAULT = 32          # for 256x256 input  (/8 downsample)
SD_VAE_DOWNSAMPLE = 8                # both SD-VAE and TAESD downsample by 8
SD_VAE_SCALING = 0.18215             # SD v1-style latent scale factor


class FrozenSDVae(nn.Module):
    """Thin wrapper around a frozen pretrained image AE in eval mode.

    Two backbones supported (both /8 downsample, 4-channel latent, same
    scale factor — identical interface):
      - "sd_vae"  → diffusers `AutoencoderKL` from `stabilityai/sd-vae-ft-mse`
                    (~83M params, ~330MB) — best recon quality
      - "taesd"   → diffusers `AutoencoderTiny` from `madebyollin/taesd`
                    (~2.4M params, ~10MB) — ~30× faster, slightly softer
                    recon. Drop-in replacement for SD-VAE.

    All parameters have requires_grad=False and the module is kept in eval()
    so BN/dropout stay frozen. encode is wrapped in @torch.no_grad();
    decode lets gradients pass through to upstream projections (the
    decoder's own params stay frozen via requires_grad=False).
    """

    def __init__(self, name: str = SD_VAE_NAME, tiny: bool = False):
        super().__init__()
        self.tiny = tiny
        if tiny:
            from diffusers import AutoencoderTiny
            self.vae = AutoencoderTiny.from_pretrained(name)
        else:
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
        # AutoencoderKL returns a posterior with .latent_dist.mean;
        # AutoencoderTiny returns latents directly via .latents.
        out = self.vae.encode(x)
        latent = out.latents if self.tiny else out.latent_dist.mean
        return latent * SD_VAE_SCALING

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent / SD_VAE_SCALING).sample


def _build_mlp(in_dim: int, hidden_dims, out_dim: int) -> nn.Module:
    """Linear → LeakyReLU stack ending with Linear(out_dim).

    If `hidden_dims` is empty (or falsy), returns a plain Linear so the
    saved state_dict keys match the legacy single-Linear setup — old
    checkpoints (no proj_hidden_dims arg) load without remapping.
    """
    if not hidden_dims:
        return nn.Linear(in_dim, out_dim)
    layers: list = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class HAEImageNet(nn.Module):
    """HAE with a frozen pretrained image AE (SD-VAE or TAESD).

    The pretrained VAE downsamples the input by 8× spatially with 4 latent
    channels, so for `image_size=S` the latent grid is `(4, S/8, S/8)` and
    flattens to `4·(S/8)²` features. Default S=256 (the SD-VAE training
    resolution); set S=64 for Tiny ImageNet etc.

    `proj_hidden_dims` controls the size of the MLP between flat VAE
    features and the latent space (and its mirror on the decoder side):
      []                  →  flat ──Linear──▶ latent          (default; legacy)
      [1024]              →  flat ─Linear─ReLU─Linear─▶ latent
      [1024, 512, 256]    →  3-layer MLP, similar to the original pSp
                              `EqualLinear_encoder_*` projections.
    The decoder mirror reverses the hidden list, so the architecture is
    symmetric:  feature_size ─Linear─ReLU─…─▶ flat.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        latent_dim: int = 512,
        feature_size: int = 512,
        curvature: float = -1.0,
        vae_name: str = SD_VAE_NAME,
        tiny_vae: bool = False,
        image_size: int = 256,
        channels: int = SD_VAE_LATENT_CHANNELS,
        proj_hidden_dims=(),
    ):
        super().__init__()
        if image_size % SD_VAE_DOWNSAMPLE != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by {SD_VAE_DOWNSAMPLE} "
                f"(SD-VAE/TAESD downsample by 8×)."
            )
        self.image_size = int(image_size)
        self.spatial = self.image_size // SD_VAE_DOWNSAMPLE
        self.channels = channels
        self.flat_dim = channels * self.spatial * self.spatial
        self.proj_hidden_dims = tuple(proj_hidden_dims) if proj_hidden_dims else ()

        # Frozen pretrained AE
        self.vae = FrozenSDVae(vae_name, tiny=tiny_vae)

        # Learned projections between VAE latent and Euclidean feature.
        # Empty hidden_dims → plain nn.Linear (legacy behaviour, old ckpts).
        self.proj_enc = _build_mlp(self.flat_dim, self.proj_hidden_dims, latent_dim)
        self.proj_dec = _build_mlp(feature_size, tuple(reversed(self.proj_hidden_dims)),
                                   self.flat_dim)

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
        z_spatial = self.vae.encode(x)                       # (B, 4, S/8, S/8)
        z_flat = z_spatial.reshape(B, -1)                    # (B, flat_dim)
        z_euc = self.proj_enc(z_flat)                        # (B, latent_dim)

        # Geometry head (hyperbolic or Euclidean)
        logits, z_hyp, z_euc_dec = self.head(z_euc)

        # Back to spatial latent → frozen decode
        z_flat_dec = self.proj_dec(z_euc_dec)                # (B, flat_dim)
        z_spatial_dec = z_flat_dec.reshape(B, self.channels, self.spatial, self.spatial)
        recon = self.vae.decode(z_spatial_dec)               # (B, 3, S, S)

        # Frozen-VAE path is AE-only; return zero KL for interface parity
        kl = torch.zeros((), device=x.device)
        return recon, logits, z_hyp, z_euc, z_euc_dec, kl


if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)
    m = HAEImageNet(num_classes=1000).eval()
    with torch.no_grad():
        recon, logits, z_hyp, z_euc, z_euc_dec = m(x)
    print("recon:", recon.shape, "logits:", logits.shape,
          "z_hyp:", z_hyp.shape, "z_euc:", z_euc.shape)
