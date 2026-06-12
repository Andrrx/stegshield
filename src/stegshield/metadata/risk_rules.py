from __future__ import annotations

from dataclasses import dataclass

from stegshield.metadata.extract import ImageMetadata
from stegshield.metadata.lsb_payload import estimate_sequential_lsb_payload
from stegshield.utils.file_validation import find_embedded_signatures

# Sequential LSB payloads above this size are script/binary-scale rather than
# marker-scale (addresses, URLs), so they carry high severity.
LARGE_LSB_PAYLOAD_BYTES = 128


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


def assess_risk(metadata: ImageMetadata, include_statistical_lsb: bool = True) -> RiskAssessment:
    """Assess static file risk.

    ``include_statistical_lsb`` toggles the statistical sequential-LSB payload
    estimator (stegshield.metadata.lsb_payload). It is turned off when fusion is
    configured to use the CNN payload estimate instead, so the two payload
    sources can be compared without double-counting.
    """
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

    if metadata.trailing_bytes_after_image_end > 0:
        indicators.append(
            RiskIndicator(
                code="trailing_data_after_image_end",
                severity="medium",
                description=(
                    "File contains data after the image end marker "
                    f"({metadata.trailing_bytes_after_image_end} bytes)."
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

    lsb_payload = estimate_sequential_lsb_payload(metadata.path) if include_statistical_lsb else None
    if lsb_payload is not None:
        if lsb_payload.estimated_payload_bytes >= LARGE_LSB_PAYLOAD_BYTES:
            indicators.append(
                RiskIndicator(
                    code="sequential_lsb_payload_large",
                    severity="high",
                    description=(
                        "Sequential LSB embedding detected with a script/binary-scale payload "
                        f"(~{lsb_payload.estimated_payload_bytes} bytes)."
                    ),
                )
            )
        else:
            indicators.append(
                RiskIndicator(
                    code="sequential_lsb_payload",
                    severity="medium",
                    description=(
                        "Sequential LSB embedding detected with a small payload "
                        f"(~{lsb_payload.estimated_payload_bytes} bytes)."
                    ),
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

    return build_assessment(indicators)


def build_assessment(indicators: list[RiskIndicator]) -> RiskAssessment:
    """Sum severity weights into a clamped risk score and a 3-way label."""
    score = min(sum(_severity_weight(indicator.severity) for indicator in indicators), 1.0)
    return RiskAssessment(
        label=_label_from_score(score), risk_score=round(score, 3), indicators=indicators
    )


def cnn_payload_indicators(payload_bytes: int | None) -> list[RiskIndicator]:
    """Turn a CNN payload-size estimate into risk indicators (same 128-byte gate).

    Returns an empty list when there is no estimate or it is non-positive. The
    caller is responsible for only passing an estimate when the CNN considers the
    image stego, since the regression head is never supervised on clean images.
    """
    if payload_bytes is None or payload_bytes <= 0:
        return []
    if payload_bytes >= LARGE_LSB_PAYLOAD_BYTES:
        return [
            RiskIndicator(
                code="cnn_lsb_payload_large",
                severity="high",
                description=(
                    "CNN estimates a script/binary-scale sequential LSB payload "
                    f"(~{payload_bytes} bytes)."
                ),
            )
        ]
    return [
        RiskIndicator(
            code="cnn_lsb_payload",
            severity="medium",
            description=f"CNN estimates a small sequential LSB payload (~{payload_bytes} bytes).",
        )
    ]


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
