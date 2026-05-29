from __future__ import annotations

from pathlib import Path

from stegshield.data.splits import DatasetSample, write_samples_csv

DATASET_DIR_NAME = "kaggle_stegoimages"


def create_kaggle_stegoimage_splits(
    raw_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> dict[str, list[DatasetSample]]:
    """Create CSV files that preserve the Kaggle Stego Images train/val/test layout."""
    raw_dir = raw_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    output_paths = {
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test_standard": output_dir / "test_standard.csv",
        "test_adversarial": output_dir / "test_adversarial.csv",
        "test_full": output_dir / "test_full.csv",
    }
    if not force and any(path.exists() for path in output_paths.values()):
        raise FileExistsError("Kaggle split CSV files already exist. Use --force to overwrite them.")

    splits = {
        "train": _split_samples(raw_dir, "train"),
        "val": _split_samples(raw_dir, "val"),
        "test_standard": _standard_test_samples(raw_dir),
        "test_adversarial": _adversarial_test_samples(raw_dir),
    }
    splits["test_full"] = sorted(
        [*splits["test_standard"], *splits["test_adversarial"]],
        key=lambda sample: (sample.label, sample.path),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, samples in splits.items():
        write_samples_csv(samples, output_paths[split_name], raw_dir=raw_dir)

    return splits


def _split_samples(raw_dir: Path, split_name: str) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    for label in ("safe", "suspicious", "dangerous"):
        samples.extend(_collect_pngs(raw_dir, label, split_name))
    return sorted(samples, key=lambda sample: (sample.label, sample.path))


def _standard_test_samples(raw_dir: Path) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    samples.extend(_collect_pngs(raw_dir, "safe", "test"))
    samples.extend(_collect_pngs(raw_dir, "suspicious", "test"))
    samples.extend(_collect_pngs(raw_dir, "dangerous", "test/plain"))
    return sorted(samples, key=lambda sample: (sample.label, sample.path))


def _adversarial_test_samples(raw_dir: Path) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    samples.extend(_collect_pngs(raw_dir, "dangerous", "test/b64"))
    samples.extend(_collect_pngs(raw_dir, "dangerous", "test/zip"))
    return sorted(samples, key=lambda sample: (sample.label, sample.path))


def _collect_pngs(raw_dir: Path, label: str, dataset_subdir: str) -> list[DatasetSample]:
    base_dir = raw_dir / label / DATASET_DIR_NAME / Path(dataset_subdir)
    if not base_dir.exists():
        return []

    return [
        DatasetSample(path=path.relative_to(raw_dir).as_posix(), label=label)
        for path in sorted(base_dir.rglob("*.png"))
        if path.is_file()
    ]
