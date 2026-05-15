from __future__ import annotations

from dataclasses import dataclass

from stegshield.metadata.extract import ImageMetadata
from stegshield.utils.file_validation import find_embedded_signatures


@dataclass(frozen=True)
class RiskIndicator:
    code: str
    severity: str
    description: str


@dataclass(frozen=True)
class RiskAssessment:
    label: str
    risk_score: float
    indicators: list[RiskIndicator]


SUSPICIOUS_METADATA_TERMS = (
    "powershell",
    "cmd.exe",
    "/bin/sh",
    "bash -c",
    "eval(",
    "<script",
    "javascript:",
    "base64",
    "http://",
    "https://",
    "wget ",
    "curl ",
)


def assess_risk(metadata: ImageMetadata) -> RiskAssessment:
    indicators: list[RiskIndicator] = []

    if metadata.detected_type == "unknown":
        indicators.append(
            RiskIndicator(
                code="unknown_file_signature",
                severity="high",
                description="File does not match a supported image signature.",
            )
        )

    if not metadata.extension_matches_type:
        indicators.append(
            RiskIndicator(
                code="extension_type_mismatch",
                severity="medium",
                description="File extension does not match detected image type.",
            )
        )

    if metadata.parse_error:
        indicators.append(
            RiskIndicator(
                code="image_parse_error",
                severity="high",
                description=f"Image parser could not safely read the file: {metadata.parse_error}",
            )
        )

    if metadata.trailing_bytes_after_jpeg_eoi > 0:
        indicators.append(
            RiskIndicator(
                code="jpeg_trailing_data",
                severity="medium",
                description=(
                    "JPEG contains data after the end-of-image marker "
                    f"({metadata.trailing_bytes_after_jpeg_eoi} bytes)."
                ),
            )
        )

    embedded_signatures = find_embedded_signatures(metadata.path)
    for signature in embedded_signatures:
        indicators.append(
            RiskIndicator(
                code="embedded_binary_signature",
                severity="high",
                description=f"Found embedded {signature} signature inside the file.",
            )
        )

    if metadata.metadata_text_size > 16_384:
        indicators.append(
            RiskIndicator(
                code="large_metadata",
                severity="medium",
                description=f"Metadata text is unusually large ({metadata.metadata_text_size} chars).",
            )
        )

    metadata_text = " ".join(metadata.metadata_fields.values()).lower()
    for term in SUSPICIOUS_METADATA_TERMS:
        if term in metadata_text:
            indicators.append(
                RiskIndicator(
                    code="suspicious_metadata_text",
                    severity="medium",
                    description=f"Metadata contains suspicious text pattern: {term}",
                )
            )
            break

    if metadata.width and metadata.height:
        bytes_per_pixel = metadata.file_size_bytes / max(metadata.width * metadata.height, 1)
        if bytes_per_pixel > 100:
            indicators.append(
                RiskIndicator(
                    code="large_file_for_dimensions",
                    severity="low",
                    description=(
                        "File is unusually large for its pixel dimensions "
                        f"({bytes_per_pixel:.2f} bytes per pixel)."
                    ),
                )
            )

    score = min(sum(_severity_weight(indicator.severity) for indicator in indicators), 1.0)
    label = _label_from_score(score)

    return RiskAssessment(label=label, risk_score=round(score, 3), indicators=indicators)


def _severity_weight(severity: str) -> float:
    weights = {
        "low": 0.15,
        "medium": 0.3,
        "high": 0.5,
    }
    return weights.get(severity, 0.0)


def _label_from_score(score: float) -> str:
    if score >= 0.7:
        return "dangerous"
    if score >= 0.3:
        return "suspicious"
    return "safe"
