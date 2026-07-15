from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from stegshield.data.splits import read_samples_csv, resolve_sample_path
from stegshield.predict_cnn import StegoPredictor
from stegshield.processing import ProcessingOp, benchmark_operations


@dataclass(frozen=True)
class RobustnessConfig:
    model_path: Path
    split_csv: Path
    output_report: Path
    raw_dir: Path | None = None
    device: str = "cpu"
    threshold: float = 0.5
    limit_per_class: int | None = 200
    batch_size: int = 32
    seed: int = 0


def evaluate_robustness(config: RobustnessConfig) -> dict[str, Any]:
    """Measure how detection survives benign and lossy image processing.

    For each processing operation, reports the stego detection rate and the
    clean false-positive rate, so the JPEG cliff and the resilience to benign
    transforms are both quantified.
    """
    predictor = StegoPredictor(model_path=config.model_path, device=config.device)
    rng = random.Random(config.seed)

    samples = read_samples_csv(config.split_csv)
    clean_paths = [s.path for s in samples if s.label == "safe"]
    stego_paths = [s.path for s in samples if s.label != "safe"]
    rng.shuffle(clean_paths)
    rng.shuffle(stego_paths)
    if config.limit_per_class is not None:
        clean_paths = clean_paths[: config.limit_per_class]
        stego_paths = stego_paths[: config.limit_per_class]
    if not clean_paths or not stego_paths:
        raise ValueError("Robustness benchmark needs both clean and stego samples in the split.")

    operations = benchmark_operations()
    results = []
    for op in operations:
        stego_probs = _probabilities(predictor, stego_paths, op, config, rng)
        clean_probs = _probabilities(predictor, clean_paths, op, config, rng)
        stego_detection_rate = _rate(stego_probs, config.threshold, positive=True)
        clean_fpr = _rate(clean_probs, config.threshold, positive=True)
        # Balanced accuracy exposes the trap where a detector "detects" everything
        # under noise by flagging clean images too (high recall AND high FPR).
        balanced_accuracy = (stego_detection_rate + (1.0 - clean_fpr)) / 2.0
        results.append(
            {
                "name": op.name,
                "lossy": op.lossy,
                "stego_detection_rate": round(stego_detection_rate, 4),
                "clean_fpr": round(clean_fpr, 4),
                "balanced_accuracy": round(balanced_accuracy, 4),
                "mean_stego_probability": round(sum(stego_probs) / len(stego_probs), 4),
            }
        )

    report = {
        "report_type": "robustness_benchmark",
        "model_path": str(config.model_path),
        "split_csv": str(config.split_csv),
        "threshold": config.threshold,
        "clean_count": len(clean_paths),
        "stego_count": len(stego_paths),
        "operations": results,
    }
    config.output_report.parent.mkdir(parents=True, exist_ok=True)
    config.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _probabilities(
    predictor: StegoPredictor,
    paths: list[str],
    op: ProcessingOp,
    config: RobustnessConfig,
    rng: random.Random,
) -> list[float]:
    probs: list[float] = []
    batch: list[torch.Tensor] = []

    def flush() -> None:
        if not batch:
            return
        tensors = torch.stack(batch).to(predictor.device)
        with torch.no_grad():
            batch_probs = torch.softmax(predictor.model(tensors), dim=1)[:, predictor.stego_index]
        probs.extend(float(value) for value in batch_probs.cpu().tolist())
        batch.clear()

    for path in paths:
        with Image.open(resolve_sample_path(path, config.raw_dir)) as image:
            processed = op.apply(image.convert("RGB"), rng)
        batch.append(predictor.transform(processed))
        if len(batch) >= config.batch_size:
            flush()
    flush()
    return probs


def _rate(probs: list[float], threshold: float, positive: bool) -> float:
    if not probs:
        return 0.0
    hits = sum(1 for value in probs if (value >= threshold) == positive)
    return hits / len(probs)
