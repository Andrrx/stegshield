from __future__ import annotations

import torch
from torch import nn


class StegShieldCNN(nn.Module):
    """Steganalysis-oriented CNN for three-class image risk classification.

    The model is runnable now, while leaving clear extension points for
    Yedroudj-Net/SRNet-style high-pass residual preprocessing and truncation.
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.preprocess = SteganalysisPreprocessor(input_channels=3)
        self.stem = nn.Sequential(
            nn.Conv2d(self.preprocess.output_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            _residual_stage(32, 32, downsample=False),
            _residual_stage(32, 64, downsample=True),
            _residual_stage(64, 128, downsample=True),
            _residual_stage(128, 256, downsample=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual_view = self.preprocess(inputs)
        features = self.features(self.stem(residual_view))
        return self.classifier(features)


class SteganalysisPreprocessor(nn.Module):
    """Fixed high-pass residual preprocessing for steganalysis.

    The same high-pass filter bank is applied separately to each RGB channel.
    This follows the residual-filtering idea from steganalysis CNNs while
    preserving channel-specific artifacts that grayscale conversion can hide.
    """

    def __init__(
        self,
        input_channels: int,
        residual_clip: float = 3.0,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.residual_clip = residual_clip
        self.filter_count = len(_high_pass_kernels())
        self.output_channels = input_channels * self.filter_count
        self.high_pass = nn.Conv2d(
            input_channels,
            self.output_channels,
            kernel_size=5,
            padding=2,
            groups=input_channels,
            bias=False,
        )
        self._initialize_high_pass_filters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residuals = self.high_pass(inputs)
        return torch.clamp(residuals, min=-self.residual_clip, max=self.residual_clip)

    def _initialize_high_pass_filters(self) -> None:
        kernels = torch.stack(_high_pass_kernels())
        weights = kernels.repeat(self.input_channels, 1, 1, 1)
        with torch.no_grad():
            self.high_pass.weight.copy_(weights)
        self.high_pass.weight.requires_grad_(False)

    # TODO: Trainable high-pass warm-up.
    # Papers often start from constrained/fixed residual filters. Later you can
    # test unfreezing this layer after a few epochs and compare validation data.

    # TODO: Grayscale/Y-channel ablation.
    # Do not use it by default for StegShield. If thesis experiments need it,
    # add it as a controlled ablation and compare against this RGB residual path.

    # TODO: Dataset normalization experiment.
    # The current dataset uses ImageNet normalization before this layer. Compare
    # it against raw ToTensor inputs or dataset-specific mean/std normalization.


class ResidualBlock(nn.Module):
    """Small residual block for StegShield's feature extractor."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.shortcut(inputs))


def _residual_stage(in_channels: int, out_channels: int, downsample: bool) -> nn.Sequential:
    stride = 2 if downsample else 1
    return nn.Sequential(
        ResidualBlock(in_channels, out_channels, stride=stride),
        ResidualBlock(out_channels, out_channels),
    )


def _high_pass_kernels() -> list[torch.Tensor]:
    """Small SRM-inspired high-pass filter bank."""
    horizontal = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ) / 2
    vertical = horizontal.t()
    square = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ) / 4
    laplacian = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0],
            [0, -1, 4, -1, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ) / 4
    edge_residual = torch.tensor(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=torch.float32,
    ) / 12
    return [
        horizontal.unsqueeze(0),
        vertical.unsqueeze(0),
        square.unsqueeze(0),
        laplacian.unsqueeze(0),
        edge_residual.unsqueeze(0),
    ]
