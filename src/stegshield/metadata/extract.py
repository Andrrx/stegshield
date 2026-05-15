from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from stegshield.utils.file_validation import detect_file_identity, find_trailing_jpeg_bytes
from stegshield.utils.hashing import sha256_file


@dataclass(frozen=True)
class ImageMetadata:
    path: str
    file_name: str
    extension: str
    file_size_bytes: int
    sha256: str
    detected_type: str
    detected_mime: str
    extension_matches_type: bool
    width: int | None
    height: int | None
    mode: str | None
    image_format: str | None
    metadata_fields: dict[str, str]
    metadata_text_size: int
    trailing_bytes_after_jpeg_eoi: int
    parse_error: str | None


def extract_image_metadata(path: Path) -> ImageMetadata:
    """Extract safe, static information from an image file."""
    path = path.expanduser().resolve()
    identity = detect_file_identity(path)
    file_size = path.stat().st_size
    metadata_fields: dict[str, str] = {}
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    image_format: str | None = None
    parse_error: str | None = None

    try:
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            image_format = image.format
            metadata_fields.update(_stringify_mapping(image.info))

            exif = image.getexif()
            if exif:
                metadata_fields.update({f"exif:{key}": str(value) for key, value in exif.items()})
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        parse_error = f"{type(exc).__name__}: {exc}"

    metadata_text_size = sum(len(key) + len(value) for key, value in metadata_fields.items())

    return ImageMetadata(
        path=str(path),
        file_name=path.name,
        extension=path.suffix.lower(),
        file_size_bytes=file_size,
        sha256=sha256_file(path),
        detected_type=identity.detected_type,
        detected_mime=identity.detected_mime,
        extension_matches_type=identity.extension_matches_type,
        width=width,
        height=height,
        mode=mode,
        image_format=image_format,
        metadata_fields=metadata_fields,
        metadata_text_size=metadata_text_size,
        trailing_bytes_after_jpeg_eoi=find_trailing_jpeg_bytes(path)
        if identity.detected_type == "jpeg"
        else 0,
        parse_error=parse_error,
    )


def _stringify_mapping(mapping: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in mapping.items():
        if isinstance(value, bytes):
            result[key] = value[:128].hex()
        else:
            result[key] = str(value)
    return result
