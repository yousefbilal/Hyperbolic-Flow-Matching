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


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CIFAREncoder(nn.Module):
    """Symmetric ResBlock encoder for 32×32 CIFAR images.

    Architecture:
        Conv3x3 3→64      (32×32)
        DownBlock 64→128   (16×16)
        DownBlock 128→256  (8×8)
        DownBlock 256→512  (4×4)
        GAP → Dropout → Linear(512, latent_dim)
    """

    def __init__(self, latent_dim: int = 512, num_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, 64), 64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 2 ResBlocks at input resolution
        self.stem_res1 = ResBlock(64, num_groups)
        self.stem_res2 = ResBlock(64, num_groups)

        self.down1 = DownBlock(64, 128, num_groups)      # 32→16
        self.down2 = DownBlock(128, 256, num_groups)      # 16→8
        self.down3 = DownBlock(256, 512, num_groups)      # 8→4

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 32, 32) → (B, latent_dim)"""
        h = self.stem(x)
        h = self.stem_res1(h)
        h = self.stem_res2(h)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.pool(h).flatten(1)
        h = self.dropout(h)
        return self.fc(h)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class CIFARDecoder(nn.Module):
    """Symmetric ResBlock decoder: flat vector → (B, 3, 32, 32) image in [-1, 1].

    Architecture (mirror of encoder):
        Linear(latent_dim, 512*4*4) → reshape (B, 512, 4, 4)
        UpBlock 512→256   (4→8)
        UpBlock 256→128   (8→16)
        UpBlock 128→64    (16→32)
        2 × ResBlock at 64
        Conv2d 64→3, tanh
    """

    def __init__(self, latent_dim: int = 512, num_groups: int = 8):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        self.up1 = UpBlock(512, 256, num_groups)     # 4→8
        self.up2 = UpBlock(256, 128, num_groups)     # 8→16
        self.up3 = UpBlock(128, 64, num_groups)      # 16→32

        # 2 ResBlocks at output resolution (mirror of encoder stem)
        self.out_res1 = ResBlock(64, num_groups)
        self.out_res2 = ResBlock(64, num_groups)

        self.to_rgb = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → (B, 3, 32, 32)"""
        h = self.fc(z)
        h = h.view(-1, 512, 4, 4)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.out_res1(h)
        h = self.out_res2(h)
        return torch.tanh(self.to_rgb(h))
