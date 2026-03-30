"""U-Net architecture for binary brain tumour segmentation.

Architecture:
  Encoder  (4 x DoubleConv + MaxPool): 64 -> 128 -> 256 -> 512
  Bottleneck:                          1024
  Decoder  (4 x Upsample + skip + DoubleConv): 512 -> 256 -> 128 -> 64
  Head:    Conv2d 64 -> 1, sigmoid

Reference: Ronneberger et al. (2015) "U-Net: Convolutional Networks for
Biomedical Image Segmentation", MICCAI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Two consecutive Conv2d 3x3 -> BatchNorm -> ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """One encoder step: MaxPool2d (halves H and W) then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.step = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.step(x)


class Up(nn.Module):
    """One decoder step: bilinear upsample, concat skip, then DoubleConv.

    Args:
        in_channels:  skip_channels + upsampled_channels (before DoubleConv)
        out_channels: feature maps produced by DoubleConv
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Pad x to match skip dimensions if H or W is off by 1 pixel
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x  = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])

        x = torch.cat([skip, x], dim=1)   # concatenate along channel axis
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    """U-Net for binary segmentation.

    Input:  [B, 3,   H, W]
    Output: [B, 1,   H, W]  -- tumour probability in [0, 1]
    """

    def __init__(self, in_channels: int = 3, base_features: int = 64):
        super().__init__()
        f = base_features   # 64

        # Encoder
        self.enc1 = DoubleConv(in_channels, f)     # [B,  64, H,    W   ]
        self.enc2 = Down(f,     f * 2)             # [B, 128, H/2,  W/2 ]
        self.enc3 = Down(f * 2, f * 4)             # [B, 256, H/4,  W/4 ]
        self.enc4 = Down(f * 4, f * 8)             # [B, 512, H/8,  W/8 ]

        # Bottleneck
        self.bottleneck = Down(f * 8, f * 16)      # [B, 1024, H/16, W/16]

        # Decoder
        # in_channels = upsampled_channels + skip_channels
        self.dec4 = Up(f * 16 + f * 8, f * 8)     # 1024 + 512 -> 512
        self.dec3 = Up(f * 8  + f * 4, f * 4)     #  512 + 256 -> 256
        self.dec2 = Up(f * 4  + f * 2, f * 2)     #  256 + 128 -> 128
        self.dec1 = Up(f * 2  + f,     f)          #  128 +  64 ->  64

        # Output head: 1x1 conv, no bias (sigmoid handles offset)
        self.head = nn.Conv2d(f, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder — save outputs as skip connections
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder — each step receives the previous output + its skip
        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return torch.sigmoid(self.head(d1))   # [B, 1, H, W] in [0, 1]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def create_model(in_channels: int = 3, base_features: int = 64) -> UNet:
    """Return a freshly initialised U-Net."""
    return UNet(in_channels=in_channels, base_features=base_features)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
