"""Model definitions for CNN and ViT scaling experiments."""

import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ConvBnReLU(nn.Module):
    """Basic convolutional feature block used by WideResNet."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block with optional channel projection on the shortcut path."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.01) -> None:
        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout_p)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        return self.activation(out + residual)


class WideResNet(nn.Module):
    """Compact WideResNet-like CNN used for CIFAR-100 scaling tests."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        channels = [3, 16, 160, 320, 640]
        self.stem = ConvBnReLU(channels[0], channels[1])
        self.block1 = ResidualBlock(channels[1], channels[2])
        self.block2 = ResidualBlock(channels[2], channels[2])
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.block3 = ResidualBlock(channels[2], channels[3])
        self.block4 = ResidualBlock(channels[3], channels[3])
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.block5 = ResidualBlock(channels[3], channels[4])
        self.block6 = ResidualBlock(channels[4], channels[4])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[4], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.pool1(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool2(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)


class ViTModel(nn.Module):
    """Vision Transformer wrapper with dataset-specific classification head."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)
