"""A modular U-Net implementation (Ronneberger et al., 2015) in plain PyTorch.

Depth and channel width are configurable so the same class serves as both the lightweight
baseline used for CPU smoke tests and a larger model for a real training run.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        if diff_h != 0 or diff_w != 0:
            x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Standard encoder-decoder U-Net with skip connections.

    Args:
        in_channels: input image channels (3 for RGB).
        num_classes: number of output segmentation classes (including background).
        base_channels: channel width of the first encoder stage; doubles at each downsampling.
        depth: number of downsampling stages (4 reproduces the original paper's architecture).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10, base_channels: int = 32, depth: int = 4) -> None:
        super().__init__()
        self.depth = depth
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.in_conv = DoubleConv(in_channels, channels[0])
        self.downs = nn.ModuleList([Down(channels[i], channels[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList(
            [Up(channels[i + 1], channels[i], channels[i]) for i in reversed(range(depth))]
        )
        self.out_conv = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.in_conv(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        out = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]
            out = up(out, skip)

        return self.out_conv(out)
