from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from stegshield.data.torch_dataset import ImageRiskDataset
from stegshield.labels import labels_for_task
from stegshield.ml_metrics import (
    auc_from_roc_points,
    classification_report_from_confusion,
    empty_confusion,
    roc_curve_points,
    update_confusion,
)
from stegshield.models.cnn import create_cnn_model


@dataclass(frozen=True)
class EvaluationConfig:
    model_path: Path
    split_csv: Path
    output_report: Path
    raw_dir: Path | None = None
    batch_size: int = 16
    device: str = "cpu"
    num_workers: int = 0
    model_name: str | None = None
    image_size: int | None = None
    normalization: str | None = None
    crop: str | None = None
    task: str | None = None


def evaluate_cnn(config: EvaluationConfig) -> dict[str, Any]:
    checkpoint = torch.load(config.model_path, map_location="cpu")
    model_name = config.model_name or checkpoint.get("model_name", "steganalysis")
    image_size = config.image_size or int(checkpoint.get("image_size", 256))
    normalization = config.normalization or checkpoint.get("normalization", "none")
    crop = config.crop or checkpoint.get("crop", "center")
    task = config.task or checkpoint.get("task", "risk")
    labels = tuple(checkpoint.get("labels", labels_for_task(task)))

    device = torch.device(config.device)
    dataset = ImageRiskDataset(
        config.split_csv,
        image_size=image_size,
        raw_dir=config.raw_dir,
        normalization=normalization,
        task=task,
        crop=crop,
    )
    if len(dataset) == 0:
        raise ValueError("Evaluation split is empty.")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Build with the payload head if the checkpoint has one, so multi-task
    # checkpoints load; forward() still uses only the detection head here.
    model = create_cnn_model(
        model_name=model_name,
        num_classes=len(labels),
        payload_head=bool(checkpoint.get("payload_head", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    confusion = empty_confusion(len(labels))
    total_loss = 0.0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    positive_index = labels.index("stego") if "stego" in labels else None
    positive_scores: list[float] = []
    positive_actuals: list[int] = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(images)
            loss = criterion(logits, batch_labels)
            predictions = logits.argmax(dim=1)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

            if positive_index is not None:
                probabilities = torch.softmax(logits, dim=1)[:, positive_index]
                positive_scores.extend(
                    round(float(score), 6) for score in probabilities.cpu().tolist()
                )
                positive_actuals.extend(
                    int(actual == positive_index) for actual in batch_labels.cpu().tolist()
                )

            for actual, predicted in zip(batch_labels.cpu(), predictions.cpu(), strict=True):
                update_confusion(confusion, int(actual), int(predicted))

    report = _build_report(
        config=config,
        model_name=model_name,
        image_size=image_size,
        normalization=normalization,
        crop=crop,
        task=task,
        labels=labels,
        checkpoint=checkpoint,
        confusion=confusion,
        average_loss=total_loss / total,
    )

    if positive_scores and 0 < sum(positive_actuals) < len(positive_actuals):
        fpr_points, tpr_points = roc_curve_points(positive_scores, positive_actuals)
        report["roc_auc"] = round(auc_from_roc_points(fpr_points, tpr_points), 6)
        report["binary_scores"] = {
            "positive_label": "stego",
            "scores": positive_scores,
            "actuals": positive_actuals,
        }

    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _build_report(
    config: EvaluationConfig,
    model_name: str,
    image_size: int,
    normalization: str,
    crop: str,
    task: str,
    labels: tuple[str, ...],
    checkpoint: dict[str, Any],
    confusion: list[list[int]],
    average_loss: float,
) -> dict[str, Any]:
    metrics = classification_report_from_confusion(confusion, labels)

    return {
        "config": {key: str(value) for key, value in asdict(config).items()},
        "split_csv": str(config.split_csv),
        "task": task,
        "model_name": model_name,
        "labels": list(labels),
        "checkpoint": {
            "model_name": model_name,
            "image_size": image_size,
            "normalization": normalization,
            "crop": crop,
            "task": task,
            "labels": list(labels),
            "class_weights": checkpoint.get("class_weights"),
            "balanced_sampler": checkpoint.get("balanced_sampler"),
            "selection_metric": checkpoint.get("selection_metric"),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_selection_metric": checkpoint.get("best_selection_metric"),
            "best_selection_metric_name": checkpoint.get("best_selection_metric_name"),
            "train_csv": checkpoint.get("train_csv"),
            "val_csv": checkpoint.get("val_csv"),
            "output_model": checkpoint.get("output_model"),
            "output_metrics": checkpoint.get("output_metrics"),
            "batch_size": checkpoint.get("batch_size"),
            "epochs": checkpoint.get("epochs"),
            "learning_rate": checkpoint.get("learning_rate"),
            "weight_decay": checkpoint.get("weight_decay"),
            "device": checkpoint.get("device"),
            "num_workers": checkpoint.get("num_workers"),
            "timestamp": checkpoint.get("timestamp"),
            "started_at": checkpoint.get("started_at"),
            "finished_at": checkpoint.get("finished_at"),
        },
        "average_loss": round(average_loss, 6),
        **metrics,
    }
