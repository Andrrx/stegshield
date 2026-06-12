from pathlib import Path

import pytest
from PIL import Image

from stegshield.metadata.extract import extract_image_metadata
from stegshield.metadata.risk_rules import assess_risk

np = pytest.importorskip("numpy")


def _smooth_image(width: int = 256, height: int = 64) -> "np.ndarray":
    # Smooth gradient: even pixel values, so the LSB plane is structured (all zero).
    column = (np.arange(width, dtype=np.uint8) // 2 * 2)
    return np.repeat(column[None, :, None], 3, axis=2).repeat(height, axis=0)


def _embed_sequential_lsb(pixels: "np.ndarray", payload_bytes: int) -> "np.ndarray":
    rng = np.random.default_rng(42)
    stego = pixels.copy().reshape(-1, 3)
    bits = payload_bytes * 8
    payload = rng.integers(0, 2, size=bits, dtype=np.uint8)
    flat = stego.reshape(-1)
    flat[:bits] = (flat[:bits] & 0xFE) | payload
    return flat.reshape(pixels.shape)


def _save(tmp_path: Path, name: str, pixels: "np.ndarray") -> Path:
    path = tmp_path / name
    Image.fromarray(pixels, mode="RGB").save(path)
    return path


def test_clean_smooth_png_has_no_indicators(tmp_path: Path) -> None:
    path = _save(tmp_path, "clean.png", _smooth_image())

    assessment = assess_risk(extract_image_metadata(path))

    assert assessment.label == "safe"
    assert assessment.indicators == []


def test_small_sequential_lsb_payload_is_medium_severity(tmp_path: Path) -> None:
    stego = _embed_sequential_lsb(_smooth_image(), payload_bytes=48)
    path = _save(tmp_path, "small_payload.png", stego)

    assessment = assess_risk(extract_image_metadata(path))

    codes = {indicator.code for indicator in assessment.indicators}
    assert "sequential_lsb_payload" in codes
    assert "sequential_lsb_payload_large" not in codes
    assert assessment.label == "suspicious"


def test_large_sequential_lsb_payload_is_high_severity(tmp_path: Path) -> None:
    stego = _embed_sequential_lsb(_smooth_image(), payload_bytes=2048)
    path = _save(tmp_path, "large_payload.png", stego)

    assessment = assess_risk(extract_image_metadata(path))

    codes = {indicator.code for indicator in assessment.indicators}
    assert "sequential_lsb_payload_large" in codes


def test_trailing_data_after_png_end_is_flagged(tmp_path: Path) -> None:
    path = _save(tmp_path, "trailing.png", _smooth_image())
    path.write_bytes(path.read_bytes() + b"appended-payload-bytes")

    assessment = assess_risk(extract_image_metadata(path))

    codes = {indicator.code for indicator in assessment.indicators}
    assert "trailing_data_after_image_end" in codes


def test_appended_zip_is_flagged_as_embedded_signature(tmp_path: Path) -> None:
    path = _save(tmp_path, "polyglot.png", _smooth_image())
    path.write_bytes(path.read_bytes() + b"PK\x03\x04" + b"\x00" * 32)

    assessment = assess_risk(extract_image_metadata(path))

    codes = {indicator.code for indicator in assessment.indicators}
    assert "embedded_binary_signature" in codes
    assert "trailing_data_after_image_end" in codes
