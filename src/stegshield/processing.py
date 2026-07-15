from __future__ import annotations

import io
import random
from dataclasses import dataclass

from PIL import Image, ImageFilter

# Image-processing primitives shared by the robustness benchmark and by
# training-time augmentation.
#
# Two distinct uses, one important semantic difference:
#
# * The robustness benchmark applies EVERY transform, including JPEG, to measure
#   how detection degrades under processing an uploaded image may have undergone.
#
# * Training augmentation applies only PAYLOAD-PRESERVING transforms. Empirically
#   (see docs/robustness_deployment.md), resize / blur / additive noise / lossless
#   re-save keep the sequential-LSB fingerprint detectable, while JPEG re-encoding
#   destroys both the payload and the fingerprint. Augmenting stego images with
#   JPEG would therefore relabel payload-free images as "stego" and poison the
#   detector, so the router (not the detector) handles the lossy case.
#
# All functions take and return an RGB PIL image and operate at full resolution
# (before the model's crop), matching how real processing hits an uploaded file.


def jpeg_recompress(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode through JPEG at the given quality, then decode back to RGB.

    Lossy: destroys the spatial LSB plane. Used by the benchmark only.
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def downscale_upscale(image: Image.Image, scale: float, resample: int = Image.BICUBIC) -> Image.Image:
    """Downscale by ``scale`` (0 < scale < 1) then back to the original size.

    Models a platform that normalizes image dimensions. Payload-preserving in the
    statistical sense: the LSB plane stays noise-like even though exact bytes are
    lost (verified empirically).
    """
    if not 0.0 < scale < 1.0:
        raise ValueError("scale must be in (0, 1).")
    width, height = image.size
    small = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), resample)
    return small.resize((width, height), resample)


def gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius))


def gaussian_noise(image: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    """Add zero-mean Gaussian noise (std ``sigma`` in 0-255 units), clipped."""
    import numpy as np

    seed = rng.randrange(2**32)
    generator = np.random.default_rng(seed)
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    noise = generator.normal(0.0, sigma, size=pixels.shape)
    noisy = np.clip(pixels + noise, 0, 255).astype("uint8")
    return Image.fromarray(noisy, mode="RGB")


def png_resave(image: Image.Image) -> Image.Image:
    """Lossless round-trip through PNG. Control transform: must not change pixels."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


@dataclass(frozen=True)
class ProcessingOp:
    """A named processing transform at a fixed strength, for the benchmark."""

    name: str
    lossy: bool
    apply: object  # Callable[[Image.Image, random.Random], Image.Image]


def benchmark_operations() -> list[ProcessingOp]:
    """The transforms and strengths swept by the robustness benchmark."""
    return [
        ProcessingOp("none", False, lambda image, rng: image),
        ProcessingOp("png_resave", False, lambda image, rng: png_resave(image)),
        ProcessingOp("resize_0.75", False, lambda image, rng: downscale_upscale(image, 0.75)),
        ProcessingOp("resize_0.50", False, lambda image, rng: downscale_upscale(image, 0.50)),
        ProcessingOp("blur_0.6", False, lambda image, rng: gaussian_blur(image, 0.6)),
        ProcessingOp("blur_1.0", False, lambda image, rng: gaussian_blur(image, 1.0)),
        ProcessingOp("noise_2", False, lambda image, rng: gaussian_noise(image, 2.0, rng)),
        ProcessingOp("noise_5", False, lambda image, rng: gaussian_noise(image, 5.0, rng)),
        ProcessingOp("jpeg_95", True, lambda image, rng: jpeg_recompress(image, 95)),
        ProcessingOp("jpeg_90", True, lambda image, rng: jpeg_recompress(image, 90)),
        ProcessingOp("jpeg_75", True, lambda image, rng: jpeg_recompress(image, 75)),
        ProcessingOp("jpeg_60", True, lambda image, rng: jpeg_recompress(image, 60)),
    ]


# Payload-preserving augmentation menu (training). No JPEG: see module docstring.
def sample_augmentation(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply a random payload-preserving transform (or identity) to a PIL image.

    Called per-sample during training when augmentation is enabled. Roughly half
    the samples are left untouched so the detector still sees pristine images.

    Note: a heavier-noise variant of this menu was tested (identity 40%, noise
    25%) and did not improve robustness — at each model's best threshold the two
    were within noise (see docs/robustness_deployment.md). Robustness to additive
    noise is limited by the signal (the +/-1 LSB perturbation sits below a
    sigma>=2 noise floor), not by the amount of augmentation.
    """
    choice = rng.random()
    if choice < 0.5:
        return image
    if choice < 0.65:
        return downscale_upscale(image, rng.uniform(0.5, 0.9))
    if choice < 0.80:
        return gaussian_blur(image, rng.uniform(0.3, 1.0))
    if choice < 0.92:
        return gaussian_noise(image, rng.uniform(1.0, 5.0), rng)
    return png_resave(image)
