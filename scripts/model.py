"""Model definitions for CNN (WideResNet) and ViT (Vision Transformer) in PyTorch."""

import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ConvBnReLU(nn.Sequential):
    """Basic conv-batchnorm-relu block without risky in-place operations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),  # Safe non-inplace activation
        )


class ResidualBlock(nn.Module):
    """Residual block with automatic shortcut projection matching channel dimensions."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.relu = nn.ReLU(inplace=False)  # Safe non-inplace activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.dropout(self.conv1(x))) + self.shortcut(x))


class WideResNet(nn.Module):
    """Compact WideResNet CNN for CIFAR-100 / ImageNet."""

    def __init__(self, num_classes: int = 100, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.stem = ConvBnReLU(3, 16)

        # Stage 1: 16 -> 160 channels (dropout_p correctly passed down)
        self.stage1 = nn.Sequential(
            ResidualBlock(16, 160, dropout_p=dropout_p),
            ResidualBlock(160, 160, dropout_p=dropout_p),
            nn.MaxPool2d(2),
        )
        # Stage 2: 160 -> 320 channels
        self.stage2 = nn.Sequential(
            ResidualBlock(160, 320, dropout_p=dropout_p),
            ResidualBlock(320, 320, dropout_p=dropout_p),
            nn.MaxPool2d(2),
        )
        # Stage 3: 320 -> 640 channels
        self.stage3 = nn.Sequential(
            ResidualBlock(320, 640, dropout_p=dropout_p),
            ResidualBlock(640, 640, dropout_p=dropout_p),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(640, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


class ViTModel(nn.Module):
    """Vision Transformer wrapper with dataset-specific classification head."""

    def __init__(self, num_classes: int = 100, pretrained: bool = True) -> None:
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)

        # Replace default ImageNet head (1000 classes) with target classification head
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)