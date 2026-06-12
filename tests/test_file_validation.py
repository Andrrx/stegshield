import random
from pathlib import Path

from PIL import Image

from stegshield.utils.file_validation import (
    detect_file_identity,
    find_embedded_signatures,
    find_trailing_png_bytes,
)


def _write_png(path: Path, size: int = 16) -> Path:
    Image.new("RGB", (size, size), color="white").save(path)
    return path


def test_detect_file_identity_recognizes_png(tmp_path: Path) -> None:
    png = _write_png(tmp_path / "image.png")

    identity = detect_file_identity(png)

    assert identity.detected_type == "png"
    assert identity.extension_matches_type is True


def test_random_mz_bytes_do_not_trigger_executable_signature(tmp_path: Path) -> None:
    # Compressed image data contains "MZ" by chance; only a validated PE
    # header (e_lfanew pointer to "PE\x00\x00") may count as an executable.
    random.seed(7)
    noise = bytes(random.randrange(256) for _ in range(200_000)) + b"MZ" + bytes(64)
    path = tmp_path / "noise.bin"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + noise)

    assert "Windows executable" not in find_embedded_signatures(path)


def test_validated_pe_header_is_detected(tmp_path: Path) -> None:
    pe_offset = 0x80
    blob = bytearray(b"MZ" + bytes(pe_offset + 8 - 2))
    blob[0x3C:0x40] = pe_offset.to_bytes(4, "little")
    blob[pe_offset : pe_offset + 4] = b"PE\x00\x00"
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(blob))

    assert "Windows executable" in find_embedded_signatures(path)


def test_embedded_zip_signature_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(100) + b"PK\x03\x04payload")

    assert "ZIP archive" in find_embedded_signatures(path)


def test_trailing_png_bytes_zero_for_clean_png(tmp_path: Path) -> None:
    png = _write_png(tmp_path / "clean.png")

    assert find_trailing_png_bytes(png) == 0


def test_trailing_png_bytes_counts_appended_data(tmp_path: Path) -> None:
    png = _write_png(tmp_path / "appended.png")
    png.write_bytes(png.read_bytes() + b"X" * 123)

    assert find_trailing_png_bytes(png) == 123
