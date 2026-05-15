from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileIdentity:
    detected_type: str
    detected_mime: str
    extension_matches_type: bool


IMAGE_SIGNATURES: tuple[tuple[str, str, tuple[str, ...], bytes], ...] = (
    ("jpeg", "image/jpeg", (".jpg", ".jpeg"), b"\xff\xd8\xff"),
    ("png", "image/png", (".png",), b"\x89PNG\r\n\x1a\n"),
    ("gif", "image/gif", (".gif",), b"GIF8"),
    ("bmp", "image/bmp", (".bmp",), b"BM"),
    ("tiff", "image/tiff", (".tif", ".tiff"), b"II*\x00"),
    ("tiff", "image/tiff", (".tif", ".tiff"), b"MM\x00*"),
    ("webp", "image/webp", (".webp",), b"RIFF"),
)

EMBEDDED_SIGNATURES: tuple[tuple[str, bytes], ...] = (
    ("Windows executable", b"MZ"),
    ("ZIP archive", b"PK\x03\x04"),
    ("RAR archive", b"Rar!\x1a\x07"),
    ("7z archive", b"7z\xbc\xaf\x27\x1c"),
    ("ELF executable", b"\x7fELF"),
)


def detect_file_identity(path: Path) -> FileIdentity:
    header = _read_header(path, 32)
    suffix = path.suffix.lower()

    for detected_type, mime, extensions, signature in IMAGE_SIGNATURES:
        if header.startswith(signature):
            if detected_type == "webp" and b"WEBP" not in header[:16]:
                continue
            return FileIdentity(
                detected_type=detected_type,
                detected_mime=mime,
                extension_matches_type=suffix in extensions,
            )

    return FileIdentity(
        detected_type="unknown",
        detected_mime="application/octet-stream",
        extension_matches_type=False,
    )


def find_trailing_jpeg_bytes(path: Path) -> int:
    data = path.read_bytes()
    marker_index = data.rfind(b"\xff\xd9")
    if marker_index == -1:
        return 0
    return max(len(data) - (marker_index + 2), 0)


def find_embedded_signatures(path: str | Path) -> list[str]:
    data = Path(path).read_bytes()
    matches: list[str] = []

    for name, signature in EMBEDDED_SIGNATURES:
        index = data.find(signature)
        if index > 0:
            matches.append(name)

    return matches


def _read_header(path: Path, size: int) -> bytes:
    with path.open("rb") as file:
        return file.read(size)
