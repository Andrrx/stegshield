import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from PIL import Image  # noqa: E402

from stegshield.data.synth_lsb import (  # noqa: E402
    RegressionSetConfig,
    crop_capacity_bytes,
    embed_sequential_lsb,
    make_payload_regression_set,
    sample_payload_size,
)
from stegshield.metadata.lsb_payload import estimate_sequential_lsb_payload  # noqa: E402


def test_crop_capacity_matches_one_bit_per_channel() -> None:
    assert crop_capacity_bytes(256) == 24576
    assert crop_capacity_bytes(64) == 1536


def test_embed_does_not_mutate_input_and_writes_only_lsb() -> None:
    base = np.full((32, 32, 3), 200, dtype=np.uint8)  # even -> LSB 0 everywhere
    original = base.copy()
    stego = embed_sequential_lsb(base, b"hello world")

    assert np.array_equal(base, original)  # input untouched
    # Only LSBs may change, so the high 7 bits are identical everywhere.
    assert np.array_equal(stego & 0xFE, base & 0xFE)


@pytest.mark.parametrize("payload_bytes", [48, 96, 480, 4800])
def test_embed_roundtrip_recovers_size_via_statistical_estimator(
    tmp_path: Path, payload_bytes: int
) -> None:
    # Smooth image -> quiet LSBs after the payload, so the estimator sees a
    # clean transition and recovers the embedded size (48-byte block resolution).
    base = np.zeros((256, 256, 3), dtype=np.uint8)
    stego = embed_sequential_lsb(base, os.urandom(payload_bytes))
    path = tmp_path / "stego.png"
    Image.fromarray(stego, mode="RGB").save(path)

    estimate = estimate_sequential_lsb_payload(path)
    assert estimate is not None
    assert abs(estimate.estimated_payload_bytes - payload_bytes) <= 48


def test_embed_rejects_oversized_payload() -> None:
    base = np.zeros((16, 16, 3), dtype=np.uint8)
    capacity = crop_capacity_bytes(16)
    with pytest.raises(ValueError, match="exceeds image capacity"):
        embed_sequential_lsb(base, os.urandom(capacity + 1))


def test_sample_payload_size_stays_in_bounds() -> None:
    import random

    rng = random.Random(0)
    capacity = crop_capacity_bytes(256)
    sizes = [sample_payload_size(rng, capacity) for _ in range(500)]
    assert min(sizes) >= 16
    assert max(sizes) <= capacity


def test_make_regression_set_has_known_sizes_and_no_source_leakage(tmp_path: Path) -> None:
    # Two tiny source pools standing in for Kaggle train/test clean images.
    raw = tmp_path / "raw"
    (raw / "safe").mkdir(parents=True)
    train_rows = ["path,label"]
    test_rows = ["path,label"]
    for index in range(6):
        name = f"train_{index}.png"
        Image.new("RGB", (64, 64), color=(index * 10, 20, 30)).save(raw / "safe" / name)
        train_rows.append(f"safe/{name},safe")
    for index in range(4):
        name = f"test_{index}.png"
        Image.new("RGB", (64, 64), color=(5, index * 10, 40)).save(raw / "safe" / name)
        test_rows.append(f"safe/{name},safe")
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    train_csv.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    test_csv.write_text("\n".join(test_rows) + "\n", encoding="utf-8")

    config = RegressionSetConfig(
        train_csv=train_csv,
        test_csv=test_csv,
        output_image_dir=tmp_path / "img",
        output_split_dir=tmp_path / "splits",
        image_size=64,
        val_fraction=0.34,
        clean_fraction=0.25,
        seed=3,
        raw_dir_for_sources=raw,
    )
    splits = make_payload_regression_set(config)

    assert set(splits) == {"train", "val", "test"}
    capacity = crop_capacity_bytes(64)
    for samples in splits.values():
        for sample in samples:
            if sample.payload_bytes is not None:
                assert 16 <= sample.payload_bytes <= capacity
    # Test images come only from the Kaggle-test pool -> 4 generated test rows.
    assert len(splits["test"]) == 4
    assert (tmp_path / "splits" / "regress_train.csv").exists()
