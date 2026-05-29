from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from stegshield.data.splits import read_samples_csv, resolve_sample_path
from stegshield.labels import LABEL_TO_INDEX


class ImageRiskDataset(Dataset):
    """PyTorch dataset for StegShield image classification."""

    def __init__(self, csv_path: Path, image_size: int = 256, raw_dir: Path | None = None) -> None:
        self.samples = read_samples_csv(csv_path)
        self.raw_dir = raw_dir
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda image: _ensure_min_size(image, image_size)),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        with Image.open(resolve_sample_path(sample.path, self.raw_dir)) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, LABEL_TO_INDEX[sample.label]


def _ensure_min_size(image: Image.Image, target_size: int) -> Image.Image:
    """Upscale only when needed; avoid downscaling away steganographic pixel artifacts."""
    width, height = image.size
    smallest_side = min(width, height)
    if smallest_side >= target_size:
        return image

    scale = target_size / smallest_side
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, resample=Image.Resampling.NEAREST)
