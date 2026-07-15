from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from stegshield.fusion import fuse_cnn_and_metadata
from stegshield.metadata.extract import extract_image_metadata
from stegshield.metadata.risk_rules import RiskAssessment, assess_risk
from stegshield.utils.file_validation import detect_file_identity

# Deployment router: a format/processing-aware front-end for scanning uploaded
# images, the "defense layer" contribution.
#
# The threat model has an asymmetry that a single classifier ignores: spatial
# LSB steganalysis (the CNN) is only valid on losslessly-stored images. JPEG (and
# other lossy) re-encoding destroys the LSB plane, so both the hidden payload and
# its detectability vanish (measured: 100% -> ~1% detection under JPEG-75). The
# router therefore inspects each file's stored format and routes:
#
#   lossless (PNG/BMP/TIFF) -> live LSB threat: run the spatial CNN + static
#                              analysis, fuse into a risk label.
#   lossy   (JPEG/WebP/GIF) -> LSB already neutralized by re-encoding: run static
#                              structural/metadata analysis only; report that
#                              spatial stego detection is not applicable.
#
# This turns the JPEG limitation into an explicit, defensible design: re-encoding
# uploads to JPEG is itself a mitigation against LSB exfiltration.

SPATIAL_LSB_FORMATS = frozenset({"png", "bmp", "tiff"})


@dataclass(frozen=True)
class ScanVerdict:
    path: str
    detected_type: str
    processing_state: str  # "lossless" | "lossy" | "unknown"
    spatial_lsb_applicable: bool
    analyses_run: list[str]
    label: str
    risk_score: float
    cnn_stego_probability: float | None
    metadata_risk_score: float
    explanation: str
    indicators: list[dict[str, str]] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def processing_state(detected_type: str) -> tuple[str, bool]:
    """Map a detected image type to (processing_state, spatial_lsb_applicable)."""
    if detected_type in SPATIAL_LSB_FORMATS:
        return "lossless", True
    if detected_type == "unknown":
        return "unknown", False
    return "lossy", False


def scan_image(
    path: Path,
    predictor: object | None = None,
    device: str = "cpu",
) -> ScanVerdict:
    """Route one image through the appropriate analyses and return a verdict.

    ``predictor`` is an optional preloaded StegoPredictor (reused across a batch).
    When it is None, the router runs static-only; the CNN branch is skipped and
    lossless images are scored from metadata alone.
    """
    started = time.perf_counter()
    path = Path(path)

    identity = detect_file_identity(path)
    state, spatial_applicable = processing_state(identity.detected_type)

    metadata = extract_image_metadata(path)
    analyses = ["static_metadata"]

    if spatial_applicable and predictor is not None:
        metadata_assessment = assess_risk(metadata, include_statistical_lsb=True)
        stego_probability = float(predictor.predict(path))
        analyses.extend(["statistical_lsb", "spatial_cnn"])
        fused = fuse_cnn_and_metadata(
            cnn_stego_probability=stego_probability,
            metadata_assessment=metadata_assessment,
        )
        label, risk_score, explanation = fused.label, fused.risk_score, fused.explanation
        cnn_probability: float | None = stego_probability
    else:
        # Lossy / unknown / no model: the statistical LSB estimator is skipped
        # (invalid on lossy pixels), so static structural analysis stands alone.
        metadata_assessment = assess_risk(metadata, include_statistical_lsb=spatial_applicable)
        if spatial_applicable:
            analyses.append("statistical_lsb")
        label, risk_score = metadata_assessment.label, metadata_assessment.risk_score
        explanation = _static_explanation(state, spatial_applicable, metadata_assessment)
        cnn_probability = None

    latency_ms = (time.perf_counter() - started) * 1000.0
    return ScanVerdict(
        path=str(path),
        detected_type=identity.detected_type,
        processing_state=state,
        spatial_lsb_applicable=spatial_applicable,
        analyses_run=analyses,
        label=label,
        risk_score=risk_score,
        cnn_stego_probability=cnn_probability,
        metadata_risk_score=metadata_assessment.risk_score,
        explanation=explanation,
        indicators=[asdict(indicator) for indicator in metadata_assessment.indicators],
        latency_ms=round(latency_ms, 2),
    )


def _static_explanation(
    state: str, spatial_applicable: bool, assessment: RiskAssessment
) -> str:
    if state == "lossy":
        base = (
            "Lossy format: spatial LSB steganalysis not applicable (re-encoding "
            "destroys the LSB plane). Static structural analysis only."
        )
    elif state == "unknown":
        base = "Unrecognized image signature; static structural analysis only."
    else:
        base = "No CNN model supplied; static structural analysis only."
    if assessment.indicators:
        return f"{base} {assessment.indicators[0].description}"
    return base
