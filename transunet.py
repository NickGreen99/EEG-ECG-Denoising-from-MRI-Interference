###############################################
# TransUNet‑style 2‑D U‑Net with ViT bottleneck
# ------------------------------------------------
# A drop‑in replacement for the 3‑D `Unet3D_ViT` that
# works on 2‑D tensors: (B, C_in, H, W) → (B, C_out, H, W).
#
# Designed for EEG denoising use‑case where the two spatial
# dimensions can be (channels × time).
###############################################

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────
# Helper blocks
# ────────────────────────────────────────────────

class ConvBlock2D(nn.Module):
    """Two consecutive Conv‑InstanceNorm‑GELU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock2D(nn.Module):
    """Upsample → concat with skip → ConvBlock2D."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock2D(in_ch, out_ch)  # after concat: channels double

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # handle odd input sizes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ────────────────────────────────────────────────
# Vision Transformer bottleneck
# ────────────────────────────────────────────────

class ViTBlock(nn.Module):
    """A lightweight ViT encoder built from nn.TransformerEncoderLayer."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, depth: int, dropout: float = 0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        return self.encoder(x)


# ────────────────────────────────────────────────
# Main model
# ────────────────────────────────────────────────

class TransUNet2D(nn.Module):
    """2‑D TransUNet: U‑Net backbone with ViT bottleneck.

    Args:
        in_channels:  # input feature maps (e.g. 3 for [dirty, noise1, noise2])
        out_channels: # predicted maps (e.g. 1 clean channel)
        base_ch:      # #filters after first conv (scales as 2× per level)
        embed_dim:    # token dimension inside ViT
        num_heads:    # self‑attention heads
        mlp_dim:      # feed‑forward hidden dim inside ViT blocks
        vit_depth:    # #TransformerEncoder layers
        max_tokens:   # pre‑allocated size for learnable positional embeddings
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_ch: int = 32,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_dim: int = 1024,
        vit_depth: int = 4,
        max_tokens: int = 4096,
    ):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock2D(in_channels, base_ch)
        self.enc2 = ConvBlock2D(base_ch, base_ch * 2)
        self.enc3 = ConvBlock2D(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock2D(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ViT bottleneck
        self.patch_proj = nn.Linear(base_ch * 8, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, embed_dim))
        self.vit = ViTBlock(embed_dim, num_heads, mlp_dim, vit_depth)
        self.unproj = nn.Linear(embed_dim, base_ch * 8)

        # Decoder
        self.up3 = UpBlock2D(base_ch * 8, base_ch * 4)
        self.up2 = UpBlock2D(base_ch * 4, base_ch * 2)
        self.up1 = UpBlock2D(base_ch * 2, base_ch)

        self.final = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    # ────────────────────────────────────────────
    # helpers
    # ────────────────────────────────────────────
    @staticmethod
    def _to_sequence(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Flatten spatial dims → sequence of tokens."""
        b, c, h, w = x.shape
        n = h * w
        x = x.flatten(2).permute(0, 2, 1)  # (B, N, C)
        return x, (h, w)

    @staticmethod
    def _to_2d(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """Rebuild 2‑D feature map from token sequence."""
        b, n, c = x.shape
        h, w = hw
        return x.permute(0, 2, 1).contiguous().view(b, c, h, w)

    # ────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        s1 = self.enc1(x)          # (B, C, H, W)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        x  = self.enc4(self.pool(s3))  # bottleneck feature map

        # ViT bottleneck
        seq, hw = self._to_sequence(x)          # (B, N, C_enc)
        seq = self.patch_proj(seq)              # (B, N, D)

        # positional embeddings (crop or interpolate)
        N = seq.shape[1]
        if N <= self.pos_embed.size(1):
            pos = self.pos_embed[:, :N]
        else:  # interpolate if input bigger than pre‑alloc
            pos = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=N,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
        seq = seq + pos

        seq = self.vit(seq)                    # Transformer encoder
        x   = self._to_2d(self.unproj(seq), hw)  # back to (B, C_enc, h, w)

        # Decoder path
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        return self.final(x)
