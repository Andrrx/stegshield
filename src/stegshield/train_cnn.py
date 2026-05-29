from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from stegshield.data.torch_dataset import ImageRiskDataset
from stegshield.labels import LABELS
from stegshield.models.cnn import StegShieldCNN


@dataclass(frozen=True)
class TrainingConfig:
    train_csv: Path
    val_csv: Path
    output_model: Path
    output_metrics: Path
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 0.001
    image_size: int = 256
    device: str = "cpu"


def train_cnn(config: TrainingConfig) -> dict[str, object]:
    device = torch.device(config.device)
    train_dataset = ImageRiskDataset(config.train_csv, image_size=config.image_size)
    val_dataset = ImageRiskDataset(config.val_csv, image_size=config.image_size)

    if len(train_dataset) == 0:
        raise ValueError("Training split is empty.")
    if len(val_dataset) == 0:
        raise ValueError("Validation split is empty.")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = StegShieldCNN(num_classes=len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[dict[str, float | int]] = []
    best_val_accuracy = 0.0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_accuracy = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        best_val_accuracy = max(best_val_accuracy, val_accuracy)

    config.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": LABELS,
            "image_size": config.image_size,
        },
        config.output_model,
    )

    metrics = {
        "config": {key: str(value) for key, value in asdict(config).items()},
        "best_val_accuracy": best_val_accuracy,
        "history": history,
    }
    config.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    config.output_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if optimizer is not None:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total
