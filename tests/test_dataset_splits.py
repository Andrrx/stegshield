from pathlib import Path

from PIL import Image

from stegshield.data.splits import collect_labeled_images, create_stratified_splits, write_split_csvs


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
