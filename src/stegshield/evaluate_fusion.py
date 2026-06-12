from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stegshield.data.splits import read_samples_csv, resolve_sample_path
from stegshield.fusion import fuse_cnn_and_metadata
from stegshield.labels import LABELS, LABEL_TO_INDEX
from stegshield.ml_metrics import classification_report_from_confusion, empty_confusion
from stegshield.metadata.extract import extract_image_metadata
from stegshield.metadata.risk_rules import assess_risk, build_assessment, cnn_payload_indicators
from stegshield.predict_cnn import StegoPredictor

PAYLOAD_SOURCES = ("statistical", "cnn", "both")
_STEGO_THRESHOLD = 0.5


@dataclass(frozen=True)
class FusionEvaluationConfig:
    model_path: Path
    split_csv: Path
    output_report: Path
    raw_dir: Path | None = None
    device: str = "cpu"
    cnn_suspicious_threshold: float = 0.3
    limit: int | None = None
    payload_source: str = "statistical"


def evaluate_fusion(config: FusionEvaluationConfig) -> dict[str, Any]:
    if config.payload_source not in PAYLOAD_SOURCES:
        raise ValueError(f"payload_source must be one of {PAYLOAD_SOURCES}.")

    samples = read_samples_csv(config.split_csv)
    if config.limit is not None:
        samples = samples[: config.limit]
    if not samples:
        raise ValueError("Fusion evaluation split is empty.")

    predictor = StegoPredictor(model_path=config.model_path, device=config.device)
    use_cnn_payload = config.payload_source in ("cnn", "both")
    if use_cnn_payload and not predictor.has_payload_head:
        raise ValueError(
            f"payload_source='{config.payload_source}' needs a checkpoint trained with "
            "--payload-head."
        )
    include_statistical_lsb = config.payload_source in ("statistical", "both")

    confusion_matrices = {
        "metadata_only": _empty_confusion(),
        "cnn_only": _empty_confusion(),
        "fused": _empty_confusion(),
    }

    for sample in samples:
        if sample.label not in LABEL_TO_INDEX:
            raise ValueError(f"Unknown risk label in split: {sample.label}")

        image_path = resolve_sample_path(sample.path, raw_dir=config.raw_dir)
        metadata = extract_image_metadata(image_path)
        metadata_assessment = assess_risk(metadata, include_statistical_lsb=include_statistical_lsb)

        if use_cnn_payload:
            stego_probability, payload_bytes = predictor.predict_with_payload(image_path)
            # The regression head is only trained on stego, so only trust its estimate
            # when the detector itself considers the image stego.
            if stego_probability >= _STEGO_THRESHOLD:
                metadata_assessment = build_assessment(
                    metadata_assessment.indicators + cnn_payload_indicators(payload_bytes)
                )
        else:
            stego_probability = predictor.predict(image_path)

        fused_assessment = fuse_cnn_and_metadata(
            cnn_stego_probability=stego_probability,
            metadata_assessment=metadata_assessment,
        )

        actual_index = LABEL_TO_INDEX[sample.label]
        _record_prediction(
            confusion_matrices["metadata_only"],
            actual_index,
            metadata_assessment.label,
        )
        _record_prediction(
            confusion_matrices["cnn_only"],
            actual_index,
            _cnn_only_risk_label(stego_probability, config.cnn_suspicious_threshold),
        )
        _record_prediction(confusion_matrices["fused"], actual_index, fused_assessment.label)

    report = {
        "config": {key: str(value) for key, value in asdict(config).items()},
        "split_csv": str(config.split_csv),
        "model_path": str(config.model_path),
        "sample_count": len(samples),
        "payload_source": config.payload_source,
        "labels": list(LABELS),
        "checkpoint": _checkpoint_metadata(predictor.checkpoint),
        "cnn_only_mapping": {
            "safe_below_stego_probability": config.cnn_suspicious_threshold,
            "suspicious_at_or_above_stego_probability": config.cnn_suspicious_threshold,
            "dangerous": "not produced by CNN-only; dangerous requires metadata/fusion evidence",
        },
        "methods": {
            name: classification_report_from_confusion(confusion, LABELS)
            for name, confusion in confusion_matrices.items()
        },
    }

    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _empty_confusion() -> list[list[int]]:
    return empty_confusion(len(LABELS))


def _record_prediction(confusion: list[list[int]], actual_index: int, predicted_label: str) -> None:
    if predicted_label not in LABEL_TO_INDEX:
        raise ValueError(f"Unknown predicted risk label: {predicted_label}")
    confusion[actual_index][LABEL_TO_INDEX[predicted_label]] += 1


def _cnn_only_risk_label(stego_probability: float, suspicious_threshold: float) -> str:
    if stego_probability >= suspicious_threshold:
        return "suspicious"
    return "safe"


def _checkpoint_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "task",
        "labels",
        "model_name",
        "normalization",
        "class_weights",
        "balanced_sampler",
        "selection_metric",
        "best_epoch",
        "best_selection_metric",
        "best_selection_metric_name",
        "train_csv",
        "val_csv",
        "output_model",
        "output_metrics",
        "image_size",
        "batch_size",
        "epochs",
        "learning_rate",
        "weight_decay",
        "device",
        "num_workers",
        "timestamp",
        "started_at",
        "finished_at",
    )
    return {key: checkpoint.get(key) for key in keys}
