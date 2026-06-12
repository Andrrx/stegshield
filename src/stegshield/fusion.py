from __future__ import annotations

from dataclasses import dataclass

from stegshield.metadata.risk_rules import RiskAssessment


@dataclass(frozen=True)
class FusedRiskAssessment:
    label: str
    risk_score: float
    cnn_stego_probability: float
    metadata_risk_score: float
    explanation: str


def fuse_cnn_and_metadata(
    cnn_stego_probability: float,
    metadata_assessment: RiskAssessment,
    cnn_weight: float = 0.55,
    metadata_weight: float = 0.45,
) -> FusedRiskAssessment:
    """Combine visual stego evidence and metadata/file-structure evidence.

    With the default weights, confident CNN stego evidence alone scores 0.55
    (suspicious); a medium-severity static indicator keeps it suspicious
    (0.685); high-severity static evidence such as a script-scale sequential
    LSB payload pushes it to dangerous (>= 0.7).
    """
    metadata_score = metadata_assessment.risk_score
    score = (cnn_weight * cnn_stego_probability) + (metadata_weight * metadata_score)
    score = min(max(score, 0.0), 1.0)

    has_high_severity = any(
        indicator.severity == "high" for indicator in metadata_assessment.indicators
    )
    has_embedded_signature = any(
        indicator.code == "embedded_binary_signature"
        for indicator in metadata_assessment.indicators
    )
    has_large_lsb_payload = any(
        indicator.code in ("sequential_lsb_payload_large", "cnn_lsb_payload_large")
        for indicator in metadata_assessment.indicators
    )
    has_strong_metadata = has_high_severity or metadata_score >= 0.5
    if has_embedded_signature:
        score = max(score, 0.7)
    elif has_high_severity:
        score = max(score, 0.5)

    if has_high_severity and cnn_stego_probability >= 0.5:
        score = max(score, 0.7)

    return FusedRiskAssessment(
        label=_label_from_score(score),
        risk_score=round(score, 3),
        cnn_stego_probability=round(cnn_stego_probability, 6),
        metadata_risk_score=metadata_score,
        explanation=_explanation(
            cnn_stego_probability,
            metadata_score,
            has_high_severity,
            has_embedded_signature,
            has_strong_metadata,
            has_large_lsb_payload,
        ),
    )


def _label_from_score(score: float) -> str:
    if score >= 0.7:
        return "dangerous"
    if score >= 0.3:
        return "suspicious"
    return "safe"


def _explanation(
    cnn_stego_probability: float,
    metadata_score: float,
    has_high_severity: bool,
    has_embedded_signature: bool,
    has_strong_metadata: bool,
    has_large_lsb_payload: bool,
) -> str:
    if has_embedded_signature:
        return "Embedded binary/archive signature is a high-severity file-structure indicator."
    if has_large_lsb_payload and cnn_stego_probability >= 0.5:
        return "CNN stego evidence with a script/binary-scale sequential LSB payload."
    if has_high_severity and cnn_stego_probability >= 0.5:
        return "High-severity metadata indicators combined with CNN stego evidence."
    if cnn_stego_probability >= 0.7 and has_strong_metadata:
        return "Strong CNN stego evidence and metadata risk indicators."
    if cnn_stego_probability >= 0.7:
        return "Strong CNN stego evidence."
    if has_high_severity:
        return "High-severity metadata or file-structure indicators."
    if metadata_score >= 0.3:
        return "Metadata or file-structure risk indicators."
    return "Low CNN and metadata risk evidence."
