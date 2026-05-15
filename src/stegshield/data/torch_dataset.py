from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from stegshield.data.splits import read_samples_csv
from stegshield.labels import LABEL_TO_INDEX


class ImageRiskDataset(Dataset):
    """PyTorch dataset for StegShield image classification."""

    def __init__(self, csv_path: Path, image_size: int = 224) -> None:
        self.samples = read_samples_csv(csv_path)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, LABEL_TO_INDEX[sample.label]
