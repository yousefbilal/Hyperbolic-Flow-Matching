"""
CIFAR-compatible symmetric encoder and decoder for the Hyperbolic Autoencoder.

CIFAREncoder  - Custom ResBlock encoder (GroupNorm) → flat Euclidean vector
CIFARDecoder  - Symmetric ResBlock decoder → (B, 3, 32, 32) in [-1, 1]

Both sides use the same ResBlock structure for balanced gradient flow.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block: Conv3x3 → GN → LeakyReLU → Conv3x3 → GN + skip."""

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, channels), channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, channels), channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class DownBlock(nn.Module):
    """Conv stride-2 downsample → 2 × ResBlock."""

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, out_ch), out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res1 = ResBlock(out_ch, num_groups)
        self.res2 = ResBlock(out_ch, num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class UpBlock(nn.Module):
    """Upsample (nearest) → Conv → 2 × ResBlock."""

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, out_ch), out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res1 = ResBlock(out_ch, num_groups)
        self.res2 = ResBlock(out_ch, num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

# class UpBlock(nn.Module):
#     """Upsample via pixel shuffle (learned) → 2 × ResBlock."""

#     def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
#         super().__init__()
#         # Conv expands to out_ch*4 channels, PixelShuffle folds 4 into 2×2 spatial
#         self.up = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1, bias=False),
#             nn.PixelShuffle(upscale_factor=2),   # (B, out_ch*4, H, W) → (B, out_ch, 2H, 2W)
#             nn.GroupNorm(min(num_groups, out_ch), out_ch),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.res1 = ResBlock(out_ch, num_groups)
#         self.res2 = ResBlock(out_ch, num_groups)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.res2(self.res1(self.up(x)))

# class UpBlock(nn.Module):
#     """Upsample via pixel shuffle (learned) → 2 × ResBlock."""

#     def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
#         super().__init__()
#         self.up = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1, bias=False),
#             nn.PixelShuffle(upscale_factor=2),
#             # GroupNorm removed — it amplifies checkerboard after shuffle
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         # ICNR init: makes pixel shuffle start as nearest-neighbor upsampling
#         # prevents checkerboard artifacts from uneven sub-pixel group weights
#         self._icnr_init(self.up[0], upscale=2)

#         self.res1 = ResBlock(out_ch, num_groups)
#         self.res2 = ResBlock(out_ch, num_groups)

#     @staticmethod
#     def _icnr_init(conv: nn.Conv2d, upscale: int = 2):
#         """Fill conv weight so every upscale²-group starts identical.
#         Each spatial filter is copied across all sub-pixel positions,
#         which is equivalent to starting from nearest-neighbor upsampling."""
#         with torch.no_grad():
#             new_weight = conv.weight.clone()
#             # out_ch*4 filters, grouped in blocks of upscale²=4
#             # copy the first filter in each block to the rest
#             for i in range(0, conv.out_channels, upscale ** 2):
#                 new_weight[i: i + upscale ** 2] = new_weight[i].unsqueeze(0)
#             conv.weight.copy_(new_weight)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.res2(self.res1(self.up(x)))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CIFAREncoder(nn.Module):
    """Symmetric ResBlock encoder for 32×32 / 64×64 / 128×128 RGB images.

    Architecture (image_size=32):
        Conv3x3 3→64      (32×32)
        DownBlock 64→128   (16×16)
        DownBlock 128→256  (8×8)
        DownBlock 256→512  (4×4)
        GAP → Dropout → Linear(512, latent_dim)

    For image_size=64 / 128 / 256 we add additional DownBlocks before GAP
    so capacity scales with input size. Channels saturate at 512 after the
    third downblock.

    Returns (z, kl) — see VAE/AE branch comments below.
    """

    # Channel ladder: stem at 64, double per downblock, cap at 512.
    _CHANNELS_BY_SIZE = {
        32:  [64, 128, 256, 512],            # 3 downblocks
        64:  [64, 128, 256, 512, 512],       # 4 downblocks
        128: [64, 128, 256, 512, 512, 512],  # 5 downblocks
        256: [64, 128, 256, 512, 512, 512, 512],  # 6 downblocks
    }

    def __init__(self, latent_dim: int = 512, num_groups: int = 8,
                 dropout: float = 0.1, variational: bool = False,
                 image_size: int = 32):
        super().__init__()
        self.variational = variational
        self.latent_dim = latent_dim
        self.image_size = int(image_size)

        if self.image_size not in self._CHANNELS_BY_SIZE:
            raise ValueError(
                f"image_size={image_size} not supported. "
                f"Supported: {sorted(self._CHANNELS_BY_SIZE)}")
        chs = self._CHANNELS_BY_SIZE[self.image_size]

        self.stem = nn.Sequential(
            nn.Conv2d(3, chs[0], 3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, chs[0]), chs[0]),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 2 ResBlocks at input resolution
        self.stem_res1 = ResBlock(chs[0], num_groups)
        self.stem_res2 = ResBlock(chs[0], num_groups)

        # Build downblocks: image_size → image_size/2 → ... → 4
        self.downs = nn.ModuleList([
            DownBlock(chs[i], chs[i + 1], num_groups)
            for i in range(len(chs) - 1)
        ])
        self.bottleneck_channels = chs[-1]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

        if variational:
            self.fc_mu = nn.Linear(self.bottleneck_channels, latent_dim)
            self.fc_logvar = nn.Linear(self.bottleneck_channels, latent_dim)
            # init logvar head small so initial σ ≈ 1, avoids early-training blowup
            nn.init.zeros_(self.fc_logvar.weight)
            nn.init.zeros_(self.fc_logvar.bias)
        else:
            self.fc = nn.Linear(self.bottleneck_channels, latent_dim)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.stem_res1(h)
        h = self.stem_res2(h)
        for down in self.downs:
            h = down(h)
        h = self.pool(h).flatten(1)
        return self.dropout(h)

    def forward(self, x: torch.Tensor):
        """x: (B, 3, 32, 32) → (z, kl)"""
        h = self._backbone(x)
        if not self.variational:
            return self.fc(h), torch.zeros((), device=x.device)

        mu = self.fc_mu(h)
        # clamp logvar for numerical safety (σ ∈ [~6.7e-3, ~22])
        logvar = self.fc_logvar(h).clamp(-10.0, 6.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        # KL( N(μ, σ²) || N(0, I) ), per-sample sum over latent dims, then mean over batch
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return z, kl


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class CIFARDecoder(nn.Module):
    """Symmetric ResBlock decoder for 32 / 64 / 128 / 256 output sizes.

    Architecture (image_size=32):
        Linear(latent_dim, 512*4*4) → reshape (B, 512, 4, 4)
        UpBlock 512→256   (4→8)
        UpBlock 256→128   (8→16)
        UpBlock 128→64    (16→32)
        2 × ResBlock at 64
        Conv2d 64→3, tanh

    For larger image_size we add UpBlocks before the residual head.
    Always reshapes from a (B, base_channels, 4, 4) feature map; number
    of UpBlocks = log2(image_size / 4).
    """

    # Mirror of CIFAREncoder._CHANNELS_BY_SIZE (reversed direction).
    _CHANNELS_BY_SIZE = {
        32:  [512, 256, 128, 64],
        64:  [512, 512, 256, 128, 64],
        128: [512, 512, 512, 256, 128, 64],
        256: [512, 512, 512, 512, 256, 128, 64],
    }

    def __init__(self, latent_dim: int = 512, num_groups: int = 8,
                 image_size: int = 32):
        super().__init__()
        self.image_size = int(image_size)

        if self.image_size not in self._CHANNELS_BY_SIZE:
            raise ValueError(
                f"image_size={image_size} not supported. "
                f"Supported: {sorted(self._CHANNELS_BY_SIZE)}")
        chs = self._CHANNELS_BY_SIZE[self.image_size]
        self.base_channels = chs[0]

        # Always reshape to (B, base_channels, 4, 4); UpBlocks grow it 2× each.
        self.fc = nn.Linear(latent_dim, self.base_channels * 4 * 4)

        self.ups = nn.ModuleList([
            UpBlock(chs[i], chs[i + 1], num_groups)
            for i in range(len(chs) - 1)
        ])

        # 2 ResBlocks at output resolution (mirror of encoder stem)
        out_ch = chs[-1]
        self.out_res1 = ResBlock(out_ch, num_groups)
        self.out_res2 = ResBlock(out_ch, num_groups)

        self.to_rgb = nn.Conv2d(out_ch, 3, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → (B, 3, image_size, image_size)"""
        h = self.fc(z)
        h = h.view(-1, self.base_channels, 4, 4)
        for up in self.ups:
            h = up(h)
        h = self.out_res1(h)
        h = self.out_res2(h)
        return torch.tanh(self.to_rgb(h))
