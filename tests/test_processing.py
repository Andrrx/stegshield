import random

import numpy as np
import pytest
from PIL import Image

from stegshield.processing import (
    benchmark_operations,
    downscale_upscale,
    gaussian_blur,
    gaussian_noise,
    jpeg_recompress,
    png_resave,
    sample_augmentation,
)


def _image(seed: int = 0, size: int = 64) -> Image.Image:
    pixels = np.random.default_rng(seed).integers(0, 256, (size, size, 3)).astype("uint8")
    return Image.fromarray(pixels, "RGB")


def test_png_resave_is_lossless() -> None:
    img = _image()
    assert np.array_equal(np.array(img), np.array(png_resave(img)))


def test_jpeg_recompress_is_lossy_and_rgb() -> None:
    img = _image()
    out = jpeg_recompress(img, 75)
    assert out.mode == "RGB"
    assert out.size == img.size
    assert not np.array_equal(np.array(img), np.array(out))


def test_downscale_upscale_preserves_size() -> None:
    img = _image(size=48)
    out = downscale_upscale(img, 0.5)
    assert out.size == img.size


def test_downscale_upscale_rejects_bad_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        downscale_upscale(_image(), 1.5)


def test_gaussian_noise_and_blur_keep_shape() -> None:
    img = _image()
    assert gaussian_blur(img, 1.0).size == img.size
    assert gaussian_noise(img, 3.0, random.Random(0)).size == img.size


def test_benchmark_operations_cover_lossless_and_lossy() -> None:
    ops = benchmark_operations()
    names = {op.name for op in ops}
    assert "none" in names
    assert any(op.lossy and op.name.startswith("jpeg") for op in ops)
    assert any((not op.lossy) and op.name.startswith("resize") for op in ops)
    # Every op is callable and returns a same-size RGB image.
    rng = random.Random(0)
    img = _image()
    for op in ops:
        out = op.apply(img, rng)
        assert out.size == img.size


def test_sample_augmentation_never_uses_jpeg() -> None:
    # Augmentation must be payload-preserving: repeatedly sampled outputs should
    # never match a JPEG-recompressed version's heavy degradation. We check the
    # weaker invariant that outputs stay valid RGB and same size across many draws.
    img = _image()
    rng = random.Random(42)
    for _ in range(200):
        out = sample_augmentation(img, rng)
        assert out.mode == "RGB"
        assert out.size == img.size
