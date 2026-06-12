from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from stegshield.data.torch_dataset import _build_transform, payload_bytes_from_target
from stegshield.labels import STEGO_LABEL_TO_INDEX
from stegshield.models.cnn import create_cnn_model


class StegoPredictor:
    """Reusable binary clean/stego CNN predictor with optional payload estimation."""

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        self.checkpoint = torch.load(model_path, map_location="cpu")
        task = self.checkpoint.get("task", "risk")
        if task != "stego":
            raise ValueError("CNN fusion requires a checkpoint trained with --task stego.")

        model_name = self.checkpoint.get("model_name", "steganalysis")
        image_size = int(self.checkpoint.get("image_size", 256))
        normalization = self.checkpoint.get("normalization", "none")
        crop = self.checkpoint.get("crop", "center")
        labels = tuple(self.checkpoint.get("labels", ("clean", "stego")))
        self.has_payload_head = bool(self.checkpoint.get("payload_head", False))
        self.payload_capacity_bytes = int(
            self.checkpoint.get("payload_capacity_bytes") or (image_size * image_size * 3 // 8)
        )

        self.stego_index = (
            labels.index("stego") if "stego" in labels else STEGO_LABEL_TO_INDEX["stego"]
        )
        self.transform = _build_transform(
            image_size=image_size, normalization=normalization, crop=crop
        )

        self.device = torch.device(device)
        self.model = create_cnn_model(
            model_name=model_name,
            num_classes=len(labels),
            payload_head=self.has_payload_head,
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

    def _load_tensor(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict(self, image_path: Path) -> float:
        with torch.no_grad():
            logits = self.model(self._load_tensor(image_path))
            probabilities = torch.softmax(logits, dim=1)
        return float(probabilities[0, self.stego_index].item())

    def predict_with_payload(self, image_path: Path) -> tuple[float, int | None]:
        """Return (stego probability, estimated payload bytes or None).

        The payload estimate is None when the checkpoint has no regression head.
        The estimate inverts the log2 target and is clamped to the crop capacity.
        """
        if not self.has_payload_head:
            return self.predict(image_path), None

        with torch.no_grad():
            logits, payload_log2 = self.model.forward_multitask(self._load_tensor(image_path))
            probabilities = torch.softmax(logits, dim=1)

        stego_probability = float(probabilities[0, self.stego_index].item())
        estimated_bytes = payload_bytes_from_target(
            float(payload_log2[0].item()), self.payload_capacity_bytes
        )
        return stego_probability, estimated_bytes


def predict_stego_probability(
    image_path: Path,
    model_path: Path,
    device: str = "cpu",
) -> float:
    predictor = StegoPredictor(model_path=model_path, device=device)
    return predictor.predict(image_path)
