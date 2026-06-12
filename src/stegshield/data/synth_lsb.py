from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

from stegshield.data.splits import DatasetSample, read_samples_csv, resolve_sample_path

# Synthetic sequential-LSB embedding for payload-size regression ground truth.
#
# This module exists to break a circular-dependency trap: if the CNN payload
# regressor were trained on labels produced by the statistical estimator
# (stegshield.metadata.lsb_payload), the later "CNN vs Westfeld-Pfitzmann
# agreement" experiment would only measure how well the student mimics the
# teacher. Here we instead embed payloads of *known* size with our own
# embedder, giving both estimators an independent ground truth.
#
# Embedding scheme (replicates the Invoke-PSImage-style sequential LSB used by
# the Kaggle Stego Images Dataset, see [[kaggle-stego-dataset-payload-location]]):
#   - payload bits are written into pixel least-significant bits,
#   - in row-major order with R, G, B channels interleaved per pixel,
#   - starting at pixel (0, 0),
#   - one payload bit per channel value, MSB-first within each byte.
# This is exactly the bit order that estimate_sequential_lsb_payload reads, so
# an embed -> estimate round-trip recovers the embedded size (up to the
# estimator's 128-pixel block resolution).
#
# Images are generated at the model's crop size, so the top-left crop applied
# at load time is a no-op and the embedded payload is a contiguous prefix of
# the crop. The payload byte count stored as ground truth is therefore exactly
# what the network can see (payload_bytes_in_crop), with no 512-vs-256 raster
# mismatch. Payload bytes are inert os.urandom data, never real file content.

REGRESS_SPLITS = ("train", "val", "test")
MIN_PAYLOAD_BYTES = 16


@dataclass(frozen=True)
class RegressionSample:
    path: str
    label: str
    payload_bytes: int | None


def crop_capacity_bytes(image_size: int) -> int:
    """LSB capacity of a square crop: one bit per RGB channel value."""
    return image_size * image_size * 3 // 8


def embed_sequential_lsb(pixels, payload: bytes):
    """Embed ``payload`` into the LSBs of ``pixels`` (H x W x 3 uint8 array).

    Bits are written row-major with R, G, B interleaved, MSB-first per byte,
    starting at pixel (0, 0). Returns a new array; the input is not mutated.
    """
    import numpy as np

    flat = pixels.reshape(-1).copy()
    payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    bit_count = int(payload_bits.size)
    if bit_count > flat.size:
        raise ValueError(
            f"Payload of {len(payload)} bytes exceeds image capacity of {flat.size // 8} bytes."
        )
    flat[:bit_count] = (flat[:bit_count] & 0xFE) | payload_bits
    return flat.reshape(pixels.shape)


def sample_payload_size(rng: random.Random, capacity_bytes: int) -> int:
    """Log-uniform payload size in [MIN_PAYLOAD_BYTES, capacity_bytes]."""
    low = math.log2(MIN_PAYLOAD_BYTES)
    high = math.log2(capacity_bytes)
    size = int(round(2 ** rng.uniform(low, high)))
    return max(MIN_PAYLOAD_BYTES, min(size, capacity_bytes))


def _label_for_payload(payload_bytes: int) -> str:
    # Mirrors the metadata severity gate (LARGE_LSB_PAYLOAD_BYTES = 128): the
    # binary task maps both suspicious and dangerous to "stego", so the exact
    # split only matters if the regression set is ever used for the risk task.
    if payload_bytes >= 128:
        return "dangerous"
    return "suspicious"


@dataclass(frozen=True)
class RegressionSetConfig:
    train_csv: Path
    test_csv: Path
    output_image_dir: Path
    output_split_dir: Path
    image_size: int = 256
    val_fraction: float = 0.15
    clean_fraction: float = 0.2
    seed: int = 1234
    limit_per_split: int | None = None
    raw_dir_for_sources: Path | None = None


def make_payload_regression_set(config: RegressionSetConfig) -> dict[str, list[RegressionSample]]:
    """Build regress_{train,val,test}.csv with known sequential-LSB payload sizes.

    Clean source images come ONLY from the Kaggle train split for
    regress_train/val and ONLY from the Kaggle test split for regress_test, so
    no source image leaks across the regression splits.
    """
    import numpy as np
    from PIL import Image

    capacity = crop_capacity_bytes(config.image_size)
    rng = random.Random(config.seed)

    train_clean = _clean_sources(config.train_csv)
    test_clean = _clean_sources(config.test_csv)
    source_raw_dir = config.raw_dir_for_sources
    rng.shuffle(train_clean)
    rng.shuffle(test_clean)

    if config.limit_per_split is not None:
        train_clean = train_clean[: 2 * config.limit_per_split]
        test_clean = test_clean[: config.limit_per_split]

    val_count = int(len(train_clean) * config.val_fraction)
    source_pools = {
        "val": train_clean[:val_count],
        "train": train_clean[val_count:],
        "test": test_clean,
    }

    config.output_split_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[RegressionSample]] = {}

    for split_name in REGRESS_SPLITS:
        split_image_dir = config.output_image_dir / split_name
        split_image_dir.mkdir(parents=True, exist_ok=True)
        samples: list[RegressionSample] = []

        for index, source in enumerate(source_pools[split_name]):
            with Image.open(resolve_sample_path(source.path, source_raw_dir)) as image:
                crop = image.convert("RGB").crop(
                    (0, 0, config.image_size, config.image_size)
                )
            pixels = np.array(crop)

            embed = rng.random() >= config.clean_fraction
            if embed:
                payload_bytes = sample_payload_size(rng, capacity)
                pixels = embed_sequential_lsb(pixels, os.urandom(payload_bytes))
                label = _label_for_payload(payload_bytes)
                payload_value: int | None = payload_bytes
            else:
                label = "safe"
                payload_value = None

            out_path = (split_image_dir / f"{split_name}_{index:06d}.png").resolve()
            Image.fromarray(pixels, mode="RGB").save(out_path)
            samples.append(
                RegressionSample(path=str(out_path), label=label, payload_bytes=payload_value)
            )

        write_regression_csv(samples, config.output_split_dir / f"regress_{split_name}.csv")
        splits[split_name] = samples

    return splits


def _clean_sources(csv_path: Path) -> list[DatasetSample]:
    return [sample for sample in read_samples_csv(csv_path) if sample.label == "safe"]


def write_regression_csv(samples: list[RegressionSample], output_path: Path) -> None:
    import csv

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label", "payload_bytes"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "path": sample.path,
                    "label": sample.label,
                    "payload_bytes": "" if sample.payload_bytes is None else sample.payload_bytes,
                }
            )
