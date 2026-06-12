from __future__ import annotations

import torch
from torch import nn


MODEL_NAMES = ("yedroudj", "steganalysis")


def create_cnn_model(
    model_name: str = "steganalysis",
    num_classes: int = 3,
    payload_head: bool = False,
) -> nn.Module:
    if payload_head and model_name != "steganalysis":
        raise ValueError("The payload regression head is only available on the steganalysis model.")
    if model_name == "yedroudj":
        return YedroudjNet(num_classes=num_classes)
    if model_name == "steganalysis":
        return StegShieldCNN(num_classes=num_classes, payload_head=payload_head)
    if model_name == "baseline":
        raise ValueError(
            "The plain 'baseline' CNN was replaced by the Yedroudj-Net steganalysis "
            "baseline. Retrain with --model yedroudj. Old 'baseline' checkpoints are "
            "no longer loadable."
        )
    raise ValueError(f"Unknown CNN model: {model_name}. Expected one of: {', '.join(MODEL_NAMES)}")


class YedroudjNet(nn.Module):
    """Yedroudj-Net spatial steganalysis CNN, used as the literature baseline.

    Reference:
        M. Yedroudj, F. Comby, M. Chaumont, "Yedroudj-Net: An Efficient CNN for
        Spatial Steganalysis", IEEE ICASSP 2018, pp. 2092-2096,
        doi:10.1109/ICASSP.2018.8461438 (HAL: lirmm-01717550).

    Faithful to the paper: fixed 30-filter SRM preprocessing (unnormalized kernels),
    five convolutional blocks with no biases, ABS in block 1, truncation activations
    (T=3, T=2) in blocks 1-2, ReLU in blocks 3-5, average pooling from block 2 on,
    global average pooling in block 5, and a 256/1024 fully connected classifier.

    Adaptation for StegShield: the paper uses single-channel 256x256 inputs from
    BOSSBase; StegShield analyzes RGB images, so the SRM bank is applied to each
    color channel separately (grouped convolution, 3x30 residual maps). Inputs are
    expected in the raw 0-255 pixel range (normalization mode ``raw255``) so the
    truncation thresholds keep the meaning they have in the paper.
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 3) -> None:
        super().__init__()
        self.preprocess = SRMPreprocessor(in_channels=in_channels)
        self.block1 = nn.Sequential(
            nn.Conv2d(self.preprocess.output_channels, 30, kernel_size=5, padding=2, bias=False),
            AbsoluteValue(),
            nn.BatchNorm2d(30),
            TruncationLinearUnit(threshold=3.0),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(30, 30, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(30),
            TruncationLinearUnit(threshold=2.0),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(30, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residuals = self.preprocess(inputs)
        features = self.block5(self.block4(self.block3(self.block2(self.block1(residuals)))))
        return self.classifier(features)


class StegShieldCNN(nn.Module):
    """StegShield's proposed steganalysis CNN for binary clean/stego estimation.

    Shares the fixed 30-filter SRM front-end with the Yedroudj-Net baseline so the
    comparison isolates the feature extractor: residual blocks with strided
    downsampling here versus the paper's five plain convolutional blocks.

    With ``payload_head=True`` the network becomes multi-task: a small regression
    MLP on the shared pooled features estimates the sequential-LSB payload size
    (as a log2 target) alongside the clean/stego logits. The flag defaults to
    False so detection-only checkpoints load and behave unchanged.
    """

    def __init__(self, num_classes: int = 3, payload_head: bool = False) -> None:
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
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )
        self.payload_head = (
            nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
            if payload_head
            else None
        )

    def pooled_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(self.stem(self.preprocess(inputs))))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pooled_features(inputs))

    def forward_multitask(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (clean/stego logits, payload log2 estimate or None).

        The payload estimate is a per-sample scalar; it is None when the model
        was built without a payload head.
        """
        features = self.pooled_features(inputs)
        logits = self.classifier(features)
        if self.payload_head is None:
            return logits, None
        return logits, self.payload_head(features).squeeze(-1)


class SRMPreprocessor(nn.Module):
    """Fixed SRM high-pass residual extraction.

    Applies the 30 basic SRM high-pass filters (unnormalized 5x5 kernels, as in
    Yedroudj-Net) to each input channel separately via a grouped convolution, so
    channel-specific embedding artifacts survive instead of being averaged away.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        kernels = srm_filter_bank()
        self.filter_count = kernels.shape[0]
        self.output_channels = in_channels * self.filter_count
        self.high_pass = nn.Conv2d(
            in_channels,
            self.output_channels,
            kernel_size=5,
            padding=2,
            groups=in_channels,
            bias=False,
        )
        with torch.no_grad():
            self.high_pass.weight.copy_(kernels.repeat(in_channels, 1, 1, 1))
        self.high_pass.weight.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.high_pass(inputs)


class SteganalysisPreprocessor(nn.Module):
    """SRM residual extraction plus truncation, used by StegShieldCNN.

    Same fixed 30-filter SRM bank as the Yedroudj-Net baseline, followed by a
    truncation that clamps residuals so strong image content does not dominate the
    weak stego signal. Expects inputs in the raw 0-255 range (``raw255``).
    """

    def __init__(
        self,
        input_channels: int,
        residual_clip: float = 3.0,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.srm = SRMPreprocessor(in_channels=input_channels)
        self.filter_count = self.srm.filter_count
        self.output_channels = self.srm.output_channels
        self.truncation = TruncationLinearUnit(threshold=residual_clip)

    @property
    def high_pass(self) -> nn.Conv2d:
        return self.srm.high_pass

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.truncation(self.srm(inputs))

    # TODO: Trainable high-pass warm-up.
    # Papers often start from constrained/fixed residual filters. Later you can
    # test unfreezing this layer after a few epochs and compare validation data.


class AbsoluteValue(nn.Module):
    """ABS activation from Xu-Net/Yedroudj-Net: enforce residual sign symmetry."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.abs(inputs)


class TruncationLinearUnit(nn.Module):
    """Clamp residual values so large image content does not dominate stego noise."""

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(inputs, min=-self.threshold, max=self.threshold)


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


def srm_filter_bank() -> torch.Tensor:
    """The 30 basic SRM high-pass filters as 5x5 kernels, shape (30, 1, 5, 5).

    Filter classes (Fridrich & Kodovsky's Spatial Rich Models), reused unnormalized
    as in Yedroudj-Net and Ye-Net:

    - 8 first-order directional differences
    - 4 second-order differences
    - 8 third-order differences
    - 1 SQUARE 3x3 and 4 EDGE 3x3
    - 1 SQUARE 5x5 and 4 EDGE 5x5
    """
    kernels: list[torch.Tensor] = []
    center = 2
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

    for dy, dx in directions:
        kernel = torch.zeros(5, 5)
        kernel[center, center] = -1.0
        kernel[center + dy, center + dx] = 1.0
        kernels.append(kernel)

    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        kernel = torch.zeros(5, 5)
        kernel[center - dy, center - dx] = 1.0
        kernel[center, center] = -2.0
        kernel[center + dy, center + dx] = 1.0
        kernels.append(kernel)

    for dy, dx in directions:
        kernel = torch.zeros(5, 5)
        kernel[center - dy, center - dx] = 1.0
        kernel[center, center] = -3.0
        kernel[center + dy, center + dx] = 3.0
        kernel[center + 2 * dy, center + 2 * dx] = -1.0
        kernels.append(kernel)

    square_3x3 = torch.zeros(5, 5)
    square_3x3[1:4, 1:4] = torch.tensor(
        [
            [-1.0, 2.0, -1.0],
            [2.0, -4.0, 2.0],
            [-1.0, 2.0, -1.0],
        ]
    )
    kernels.append(square_3x3)

    edge_3x3 = torch.zeros(5, 5)
    edge_3x3[1:4, 1:4] = torch.tensor(
        [
            [-1.0, 2.0, -1.0],
            [2.0, -4.0, 2.0],
            [0.0, 0.0, 0.0],
        ]
    )
    for rotation in range(4):
        kernels.append(torch.rot90(edge_3x3, k=rotation))

    square_5x5 = torch.tensor(
        [
            [-1.0, 2.0, -2.0, 2.0, -1.0],
            [2.0, -6.0, 8.0, -6.0, 2.0],
            [-2.0, 8.0, -12.0, 8.0, -2.0],
            [2.0, -6.0, 8.0, -6.0, 2.0],
            [-1.0, 2.0, -2.0, 2.0, -1.0],
        ]
    )
    kernels.append(square_5x5)

    edge_5x5 = square_5x5.clone()
    edge_5x5[3:, :] = 0.0
    for rotation in range(4):
        kernels.append(torch.rot90(edge_5x5, k=rotation))

    return torch.stack(kernels).unsqueeze(1)
