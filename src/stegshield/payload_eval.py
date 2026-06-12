from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from stegshield.data.splits import resolve_sample_path
from stegshield.data.torch_dataset import ImageRiskDataset, payload_bytes_from_target
from stegshield.metadata.lsb_payload import estimate_sequential_lsb_payload
from stegshield.models.cnn import create_cnn_model


@dataclass(frozen=True)
class PayloadRegressionConfig:
    model_path: Path
    split_csv: Path
    output_report: Path
    raw_dir: Path | None = None
    batch_size: int = 16
    device: str = "cpu"
    num_workers: int = 0


@dataclass(frozen=True)
class PayloadAgreementConfig:
    model_path: Path
    split_csv: Path
    output_report: Path
    raw_dir: Path | None = None
    batch_size: int = 16
    device: str = "cpu"
    num_workers: int = 0
    stego_threshold: float = 0.5


def _load_payload_model(checkpoint: dict[str, Any], device: torch.device):
    if not checkpoint.get("payload_head", False):
        raise ValueError("Checkpoint has no payload head; retrain with --payload-head.")
    labels = tuple(checkpoint.get("labels", ("clean", "stego")))
    model = create_cnn_model(
        model_name=checkpoint.get("model_name", "steganalysis"),
        num_classes=len(labels),
        payload_head=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stego_index = labels.index("stego") if "stego" in labels else len(labels) - 1
    return model, stego_index


def _checkpoint_settings(checkpoint: dict[str, Any]) -> tuple[int, str, str, int]:
    image_size = int(checkpoint.get("image_size", 256))
    normalization = checkpoint.get("normalization", "raw255")
    crop = checkpoint.get("crop", "top-left")
    capacity = int(checkpoint.get("payload_capacity_bytes") or (image_size * image_size * 3 // 8))
    return image_size, normalization, crop, capacity


def _run_cnn_payload(
    config: PayloadRegressionConfig | PayloadAgreementConfig,
    checkpoint: dict[str, Any],
    with_target: bool,
) -> tuple[list[float], list[int], list[float | None], ImageRiskDataset]:
    """Batched forward pass: returns stego probs, payload byte estimates, log2 targets, dataset."""
    device = torch.device(config.device)
    image_size, normalization, crop, capacity = _checkpoint_settings(checkpoint)
    dataset = ImageRiskDataset(
        config.split_csv,
        image_size=image_size,
        raw_dir=config.raw_dir,
        normalization=normalization,
        task="stego",
        crop=crop,
        with_payload_target=with_target,
    )
    if len(dataset) == 0:
        raise ValueError("Payload evaluation split is empty.")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model, stego_index = _load_payload_model(checkpoint, device)

    probs: list[float] = []
    pred_bytes: list[int] = []
    targets_log2: list[float | None] = []

    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            logits, payload_log2 = model.forward_multitask(images)
            batch_probs = torch.softmax(logits, dim=1)[:, stego_index]
            probs.extend(float(value) for value in batch_probs.cpu().tolist())
            pred_bytes.extend(
                payload_bytes_from_target(float(value), capacity)
                for value in payload_log2.cpu().tolist()
            )
            if with_target:
                targets_log2.extend(
                    None if math.isnan(value) else float(value)
                    for value in batch[2].cpu().tolist()
                )

    return probs, pred_bytes, targets_log2, dataset


def evaluate_payload_regression(config: PayloadRegressionConfig) -> dict[str, Any]:
    """Payload-size regression quality on a labeled regression split (MAE/median in bytes)."""
    checkpoint = torch.load(config.model_path, map_location="cpu")
    _, _, _, capacity = _checkpoint_settings(checkpoint)
    _, pred_bytes, targets_log2, _ = _run_cnn_payload(config, checkpoint, with_target=True)

    true_bytes: list[int] = []
    matched_pred: list[int] = []
    for target, predicted in zip(targets_log2, pred_bytes, strict=True):
        if target is None:
            continue
        true_bytes.append(payload_bytes_from_target(target, capacity))
        matched_pred.append(predicted)

    if not true_bytes:
        raise ValueError("Regression split has no supervised (known-payload) samples.")

    abs_errors = [abs(p - t) for p, t in zip(matched_pred, true_bytes, strict=True)]
    log2_abs_errors = [
        abs(math.log2(p + 1) - math.log2(t + 1))
        for p, t in zip(matched_pred, true_bytes, strict=True)
    ]

    report = {
        "report_type": "payload_regression",
        "split_csv": str(config.split_csv),
        "model_path": str(config.model_path),
        "capacity_bytes": capacity,
        "supervised_count": len(true_bytes),
        "mae_bytes": round(sum(abs_errors) / len(abs_errors), 3),
        "median_absolute_error_bytes": round(_median(abs_errors), 3),
        "mae_log2": round(sum(log2_abs_errors) / len(log2_abs_errors), 4),
        "points": {"true_bytes": true_bytes, "pred_bytes": matched_pred},
    }
    _write(config.output_report, report)
    return report


def evaluate_payload_agreement(config: PayloadAgreementConfig) -> dict[str, Any]:
    """CNN vs statistical (Westfeld-Pfitzmann) payload estimates on real stego images.

    Both estimators run on independent logic; the statistical one is capped at the
    crop capacity so the comparison is on the same scale the CNN can represent.
    """
    checkpoint = torch.load(config.model_path, map_location="cpu")
    _, _, _, capacity = _checkpoint_settings(checkpoint)
    probs, pred_bytes, _, dataset = _run_cnn_payload(config, checkpoint, with_target=False)

    cnn_bytes: list[int] = []
    statistical_bytes: list[int] = []
    for sample, prob, cnn_estimate in zip(dataset.samples, probs, pred_bytes, strict=True):
        if sample.label == "safe" or prob < config.stego_threshold:
            continue
        statistical = estimate_sequential_lsb_payload(
            resolve_sample_path(sample.path, config.raw_dir)
        )
        if statistical is None:
            continue
        cnn_bytes.append(cnn_estimate)
        statistical_bytes.append(min(statistical.estimated_payload_bytes, capacity))

    if len(cnn_bytes) < 2:
        raise ValueError("Not enough comparable stego samples for agreement analysis.")

    cnn_log2 = [math.log2(value + 1) for value in cnn_bytes]
    statistical_log2 = [math.log2(value + 1) for value in statistical_bytes]

    report = {
        "report_type": "payload_agreement",
        "split_csv": str(config.split_csv),
        "model_path": str(config.model_path),
        "capacity_bytes": capacity,
        "compared_count": len(cnn_bytes),
        "pearson_log2": round(_pearson(cnn_log2, statistical_log2), 4),
        "spearman_log2": round(_spearman(cnn_log2, statistical_log2), 4),
        "median_abs_diff_bytes": round(
            _median([abs(c - s) for c, s in zip(cnn_bytes, statistical_bytes, strict=True)]), 3
        ),
        "points": {"cnn_bytes": cnn_bytes, "statistical_bytes": statistical_bytes},
    }
    _write(config.output_report, report)
    return report


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _pearson(xs: list[float], ys: list[float]) -> float:
    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    var_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if var_x == 0 or var_y == 0:
        return 0.0
    return covariance / (var_x * var_y)


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average_rank = (index + end) / 2.0
        for position in range(index, end + 1):
            ranks[order[position]] = average_rank
        index = end + 1
    return ranks


def _write(output_report: Path, report: dict[str, Any]) -> None:
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")


def config_as_dict(config: PayloadRegressionConfig | PayloadAgreementConfig) -> dict[str, Any]:
    return asdict(config)
