from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from stegshield.labels import LABELS

SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class DatasetSample:
    path: str
    label: str
    payload_bytes: int | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def collect_labeled_images(raw_dir: Path) -> list[DatasetSample]:
    """Collect images from data/raw/{safe,suspicious,dangerous}."""
    raw_dir = raw_dir.expanduser().resolve()
    samples: list[DatasetSample] = []

    for label in LABELS:
        label_dir = raw_dir / label
        if not label_dir.exists():
            continue

        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                samples.append(DatasetSample(path=path.relative_to(raw_dir).as_posix(), label=label))

    return samples


def create_stratified_splits(
    samples: list[DatasetSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[DatasetSample]]:
    """Create deterministic train/validation/test splits per label."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    grouped: dict[str, list[DatasetSample]] = {label: [] for label in LABELS}
    for sample in samples:
        if sample.label not in grouped:
            raise ValueError(f"Unknown label: {sample.label}")
        grouped[sample.label].append(sample)

    rng = random.Random(seed)
    splits: dict[str, list[DatasetSample]] = {"train": [], "val": [], "test": []}

    for label_samples in grouped.values():
        shuffled = label_samples[:]
        rng.shuffle(shuffled)

        total = len(shuffled)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        splits["train"].extend(shuffled[:train_count])
        splits["val"].extend(shuffled[train_count : train_count + val_count])
        splits["test"].extend(shuffled[train_count + val_count :])

    for split_samples in splits.values():
        split_samples.sort(key=lambda sample: sample.path)

    return splits


def write_split_csvs(splits: dict[str, list[DatasetSample]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, samples in splits.items():
        write_samples_csv(samples, output_dir / f"{split_name}.csv")


def write_samples_csv(
    samples: list[DatasetSample],
    output_path: Path,
    raw_dir: Path | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({"path": _portable_sample_path(sample.path, raw_dir), "label": sample.label})


def read_samples_csv(csv_path: Path) -> list[DatasetSample]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return [
            DatasetSample(
                path=row["path"],
                label=row["label"],
                payload_bytes=_parse_payload_bytes(row.get("payload_bytes")),
            )
            for row in reader
        ]


def _parse_payload_bytes(value: str | None) -> int | None:
    """Empty or missing payload size means unknown (masked during training)."""
    if value is None or value.strip() == "":
        return None
    return int(value)


def resolve_sample_path(sample_path: str, raw_dir: Path | None = None) -> Path:
    path = Path(sample_path)
    if path.is_absolute():
        return path

    base_dir = raw_dir if raw_dir is not None else project_root() / "data" / "raw"
    return base_dir.expanduser().resolve() / path


def _portable_sample_path(sample_path: str, raw_dir: Path | None) -> str:
    path = Path(sample_path)
    if not path.is_absolute():
        return path.as_posix()
    if raw_dir is None:
        return str(path)

    try:
        return path.resolve().relative_to(raw_dir.expanduser().resolve()).as_posix()
    except ValueError:
        return str(path)
