"""ResNet-18 for CIFAR-10: standard and depthwise-separable convolutions."""

from __future__ import annotations

import io
from typing import Type

import torch
import torch.nn as nn


class DepthwiseSeparableConv2d(nn.Module):
    """MobileNet-style block: depthwise k×k + pointwise 1×1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    *,
    separable: bool = False,
) -> nn.Module:
    if separable:
        return DepthwiseSeparableConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
    return nn.Conv2d(
        in_channels, out_channels, 3, stride=stride, padding=1, bias=False
    )


def conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    *,
    separable: bool = False,
) -> nn.Module:
    if separable:
        return DepthwiseSeparableConv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
    return nn.Conv2d(
        in_channels, out_channels, 1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        *,
        separable: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride, separable=separable)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, separable=separable)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: list[int],
        num_classes: int = 10,
        *,
        separable: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = 64
        self.separable = separable

        self.conv1 = conv3x3(3, 64, separable=separable)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # CIFAR-10: 32×32, no early pooling
        self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(
                    self.in_channels,
                    out_channels,
                    stride,
                    separable=self.separable,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = [
            block(
                self.in_channels,
                out_channels,
                stride,
                downsample,
                separable=self.separable,
            )
        ]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, separable=self.separable)
            )

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18(num_classes: int = 10, *, separable: bool = False) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, separable=separable)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_bytes(model: nn.Module) -> int:
    """Serialized checkpoint size (weights + buffers)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()
