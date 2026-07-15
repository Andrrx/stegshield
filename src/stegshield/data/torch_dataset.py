from __future__ import annotations

import math
import random
from functools import partial
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from stegshield.data.splits import read_samples_csv, resolve_sample_path
from stegshield.data.synth_lsb import crop_capacity_bytes
from stegshield.labels import label_to_index_for_task
from stegshield.processing import sample_augmentation

CROP_MODES = ("top-left", "center")


def payload_regression_target(payload_bytes: int, capacity_bytes: int) -> float:
    """log2 of the crop-capped payload size; the well-posed regression target.

    The network can only see payload within its crop, so the target is capped
    at the crop's LSB capacity. log2(x + 1) compresses the 16 B .. ~24 KB range
    (three orders of magnitude) and maps a zero payload to 0.
    """
    capped = min(max(payload_bytes, 0), capacity_bytes)
    return math.log2(capped + 1)


def payload_bytes_from_target(target: float, capacity_bytes: int) -> int:
    """Invert payload_regression_target back to a byte count."""
    bytes_estimate = int(round(2.0**target - 1.0))
    return min(max(bytes_estimate, 0), capacity_bytes)


class ImageRiskDataset(Dataset):
    """PyTorch dataset for StegShield image classification."""

    def __init__(
        self,
        csv_path: Path,
        image_size: int = 256,
        raw_dir: Path | None = None,
        normalization: str = "raw255",
        task: str = "stego",
        crop: str = "top-left",
        with_payload_target: bool = False,
        augment: bool = False,
    ) -> None:
        self.samples = read_samples_csv(csv_path)
        self.raw_dir = raw_dir
        self.task = task
        self.with_payload_target = with_payload_target
        self.capacity_bytes = crop_capacity_bytes(image_size)
        self.transform = _build_transform(
            image_size=image_size, normalization=normalization, crop=crop, augment=augment
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int] | tuple[Tensor, int, float]:
        sample = self.samples[index]
        with Image.open(resolve_sample_path(sample.path, self.raw_dir)) as image:
            image_tensor = self.transform(image.convert("RGB"))
        label_index = label_to_index_for_task(sample.label, self.task)
        if not self.with_payload_target:
            return image_tensor, label_index

        # NaN marks a masked sample (clean, or stego without a known size): the
        # detection head still trains on it, the regression head ignores it.
        if sample.payload_bytes is None:
            target = math.nan
        else:
            target = payload_regression_target(sample.payload_bytes, self.capacity_bytes)
        return image_tensor, label_index, target


def _ensure_min_size(image: Image.Image, target_size: int) -> Image.Image:
    """Upscale only when needed; avoid downscaling away steganographic pixel artifacts."""
    width, height = image.size
    smallest_side = min(width, height)
    if smallest_side >= target_size:
        return image

    scale = target_size / smallest_side
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, resample=Image.Resampling.NEAREST)


def _crop_top_left(image: Image.Image, target_size: int) -> Image.Image:
    """Keep the top-left corner.

    Sequential LSB embedders (including the tool behind the Kaggle Stego Images
    Dataset) write payload bits row by row starting at pixel (0, 0), so the stego
    signal concentrates in the first rows. A center crop can discard it entirely.
    """
    return image.crop((0, 0, target_size, target_size))


def _scale_to_255(tensor: Tensor) -> Tensor:
    return tensor * 255.0


def _augment_pil(image: Image.Image) -> Image.Image:
    # Fresh per-call Random so DataLoader worker processes diverge and augmentation
    # varies across epochs; exact reproducibility is not needed for augmentation.
    return sample_augmentation(image, random.Random())


def _build_transform(
    image_size: int,
    normalization: str,
    crop: str = "top-left",
    augment: bool = False,
) -> transforms.Compose:
    steps: list[object] = []
    if augment:
        # Payload-preserving processing on the full-resolution image, before crop.
        steps.append(transforms.Lambda(_augment_pil))
    steps.append(transforms.Lambda(partial(_ensure_min_size, target_size=image_size)))
    if crop == "top-left":
        steps.append(transforms.Lambda(partial(_crop_top_left, target_size=image_size)))
    elif crop == "center":
        steps.append(transforms.CenterCrop((image_size, image_size)))
    else:
        raise ValueError(f"Unsupported crop mode: {crop}. Expected one of: {', '.join(CROP_MODES)}")

    steps.append(transforms.ToTensor())
    if normalization == "raw255":
        # Steganalysis CNNs (Yedroudj-Net, Ye-Net) operate on raw 0-255 pixel
        # values; their truncation thresholds are meaningless on [0, 1] inputs.
        steps.append(transforms.Lambda(_scale_to_255))
    elif normalization == "imagenet":
        steps.append(transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    elif normalization != "none":
        raise ValueError(f"Unsupported normalization mode: {normalization}")

    return transforms.Compose(steps)
