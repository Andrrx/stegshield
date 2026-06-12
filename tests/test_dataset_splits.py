from pathlib import Path

import pytest
from PIL import Image

from stegshield.data.kaggle_stegoimages import create_kaggle_stegoimage_splits
from stegshield.data.splits import (
    collect_labeled_images,
    create_stratified_splits,
    read_samples_csv,
    resolve_sample_path,
    write_split_csvs,
)
from stegshield.data.torch_dataset import ImageRiskDataset
from stegshield.train_cnn import _balanced_sampler


def test_collect_labeled_images(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    safe_dir = raw_dir / "safe"
    suspicious_dir = raw_dir / "suspicious"
    dangerous_dir = raw_dir / "dangerous"
    safe_dir.mkdir(parents=True)
    suspicious_dir.mkdir(parents=True)
    dangerous_dir.mkdir(parents=True)

    Image.new("RGB", (4, 4)).save(safe_dir / "safe.png")
    Image.new("RGB", (4, 4)).save(suspicious_dir / "suspicious.jpg")
    Image.new("RGB", (4, 4)).save(dangerous_dir / "dangerous.webp")
    (safe_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    samples = collect_labeled_images(raw_dir)

    assert len(samples) == 3
    assert {sample.label for sample in samples} == {"safe", "suspicious", "dangerous"}
    assert {sample.path for sample in samples} == {
        "safe/safe.png",
        "suspicious/suspicious.jpg",
        "dangerous/dangerous.webp",
    }


def test_create_and_write_splits(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    for label in ("safe", "suspicious", "dangerous"):
        label_dir = raw_dir / label
        label_dir.mkdir(parents=True)
        for index in range(10):
            Image.new("RGB", (4, 4)).save(label_dir / f"{label}-{index}.png")

    samples = collect_labeled_images(raw_dir)
    splits = create_stratified_splits(samples, seed=123)

    assert len(splits["train"]) == 21
    assert len(splits["val"]) == 3
    assert len(splits["test"]) == 6

    output_dir = tmp_path / "splits"
    write_split_csvs(splits, output_dir)

    assert (output_dir / "train.csv").exists()
    assert (output_dir / "val.csv").exists()
    assert (output_dir / "test.csv").exists()


def test_read_samples_csv_handles_utf8_bom(tmp_path: Path) -> None:
    csv_path = tmp_path / "samples.csv"
    csv_path.write_text(
        "\ufeffpath,label\nC:\\images\\sample.png,safe\n",
        encoding="utf-8",
    )

    samples = read_samples_csv(csv_path)

    assert len(samples) == 1
    assert samples[0].path == "C:\\images\\sample.png"
    assert samples[0].label == "safe"


def test_resolve_sample_path_uses_raw_dir_for_relative_paths(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"

    resolved = resolve_sample_path("safe/example.png", raw_dir=raw_dir)

    assert resolved == raw_dir.resolve() / "safe" / "example.png"


def test_create_kaggle_stegoimage_splits_preserves_official_layout(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    for relative_dir in (
        "safe/kaggle_stegoimages/train",
        "safe/kaggle_stegoimages/val",
        "safe/kaggle_stegoimages/test",
        "suspicious/kaggle_stegoimages/train",
        "suspicious/kaggle_stegoimages/val",
        "suspicious/kaggle_stegoimages/test",
        "dangerous/kaggle_stegoimages/train",
        "dangerous/kaggle_stegoimages/val",
        "dangerous/kaggle_stegoimages/test/plain",
        "dangerous/kaggle_stegoimages/test/b64",
        "dangerous/kaggle_stegoimages/test/zip",
    ):
        image_dir = raw_dir / relative_dir
        image_dir.mkdir(parents=True)
        Image.new("RGB", (4, 4)).save(image_dir / "sample.png")

    output_dir = tmp_path / "splits"
    splits = create_kaggle_stegoimage_splits(raw_dir=raw_dir, output_dir=output_dir)

    assert len(splits["train"]) == 3
    assert len(splits["val"]) == 3
    assert len(splits["test_standard"]) == 3
    assert len(splits["test_adversarial"]) == 2
    assert len(splits["test_full"]) == 5
    assert (output_dir / "test_standard.csv").exists()
    assert (output_dir / "test_adversarial.csv").exists()
    assert (output_dir / "test_full.csv").exists()

    train_samples = read_samples_csv(output_dir / "train.csv")
    assert all(not Path(sample.path).is_absolute() for sample in train_samples)


def test_image_dataset_can_disable_imagenet_normalization(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    image_dir = raw_dir / "safe"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "white.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)

    csv_path = tmp_path / "samples.csv"
    csv_path.write_text("path,label\nsafe/white.png,safe\n", encoding="utf-8")

    dataset = ImageRiskDataset(csv_path, image_size=8, raw_dir=raw_dir, normalization="none")
    image_tensor, label = dataset[0]

    assert label == 0
    assert float(image_tensor.min()) == 1.0
    assert float(image_tensor.max()) == 1.0


def test_image_dataset_maps_risk_labels_to_binary_stego_task(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    for label in ("safe", "suspicious", "dangerous"):
        image_dir = raw_dir / label
        image_dir.mkdir(parents=True)
        Image.new("RGB", (8, 8), color="white").save(image_dir / f"{label}.png")

    csv_path = tmp_path / "samples.csv"
    csv_path.write_text(
        "\n".join(
            [
                "path,label",
                "safe/safe.png,safe",
                "suspicious/suspicious.png,suspicious",
                "dangerous/dangerous.png,dangerous",
            ]
        ),
        encoding="utf-8",
    )

    dataset = ImageRiskDataset(csv_path, image_size=8, raw_dir=raw_dir, task="stego")

    assert dataset[0][1] == 0
    assert dataset[1][1] == 1
    assert dataset[2][1] == 1


def test_balanced_sampler_weights_minority_class_more_heavily(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    safe_dir = raw_dir / "safe"
    suspicious_dir = raw_dir / "suspicious"
    safe_dir.mkdir(parents=True)
    suspicious_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="white").save(safe_dir / "safe.png")
    for index in range(3):
        Image.new("RGB", (8, 8), color="white").save(suspicious_dir / f"suspicious-{index}.png")

    csv_path = tmp_path / "samples.csv"
    csv_path.write_text(
        "\n".join(
            [
                "path,label",
                "safe/safe.png,safe",
                "suspicious/suspicious-0.png,suspicious",
                "suspicious/suspicious-1.png,suspicious",
                "suspicious/suspicious-2.png,suspicious",
            ]
        ),
        encoding="utf-8",
    )

    dataset = ImageRiskDataset(csv_path, image_size=8, raw_dir=raw_dir, task="stego")
    sampler = _balanced_sampler(dataset.samples, task="stego")
    weights = sampler.weights.tolist()

    assert weights[0] == 1.0
    assert weights[1:] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
