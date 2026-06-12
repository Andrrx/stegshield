from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Defensive static analysis of sequential LSB embedding.
#
# Tools in the Invoke-PSImage family (including the generator behind the Kaggle
# Stego Images Dataset) write payload bits into pixel least-significant bits in
# scan order, starting near the first pixel. The embedded region looks like a
# Bernoulli(0.5) bit stream, while the original LSB plane keeps the spatial
# structure of the image, so the payload shows up as a leading run of
# noise-like LSB blocks followed by an abrupt transition back to image-like
# statistics. This is the structural idea behind classical sequential-LSB
# steganalysis (Westfeld & Pfitzmann, "Attacks on Steganographic Systems",
# Information Hiding 1999): sequential embedding randomizes the start of the
# LSB stream and leaves the remainder untouched.
#
# The detector reports the length of the leading noise-like run as an estimate
# of the payload size. It never decodes or executes payload content.

# Pixel formats whose LSB plane is meaningful to analyze. Lossy formats (JPEG,
# WebP) decode to pixels whose LSBs are codec artifacts, not embedded data.
LOSSLESS_FORMATS = {"PNG", "BMP", "TIFF"}

NOISE_RATIO_LOW = 0.25
NOISE_RATIO_HIGH = 0.75
BLOCK_PIXELS = 128


@dataclass(frozen=True)
class LsbPayloadEstimate:
    leading_noisy_blocks: int
    block_pixels: int
    estimated_payload_bytes: int


def estimate_sequential_lsb_payload(
    path: str | Path,
    block_pixels: int = BLOCK_PIXELS,
) -> LsbPayloadEstimate | None:
    """Estimate a sequential LSB payload from the leading noise-like LSB run.

    Returns None when no payload-like run is found, when the whole image is
    noise-like (no transition, so sequential embedding cannot be inferred),
    when the format is lossy, or when numpy/Pillow are unavailable.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None

    try:
        with Image.open(path) as image:
            if (image.format or "").upper() not in LOSSLESS_FORMATS:
                return None
            pixels = np.array(image.convert("RGB"))
    except Exception:
        return None

    bits = (pixels & 1).reshape(-1)
    bits_per_block = block_pixels * 3
    block_count = bits.shape[0] // bits_per_block
    if block_count < 2:
        return None

    ratios = bits[: block_count * bits_per_block].reshape(block_count, bits_per_block).mean(axis=1)
    noisy = (ratios > NOISE_RATIO_LOW) & (ratios < NOISE_RATIO_HIGH)

    leading_run = 0
    for is_noisy in noisy:
        if not is_noisy:
            break
        leading_run += 1

    if leading_run == 0 or leading_run >= block_count:
        # No leading run, or no transition back to image-like statistics.
        return None

    return LsbPayloadEstimate(
        leading_noisy_blocks=leading_run,
        block_pixels=block_pixels,
        estimated_payload_bytes=leading_run * bits_per_block // 8,
    )
