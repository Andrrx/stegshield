from stegshield.fusion import fuse_cnn_and_metadata
from stegshield.metadata.risk_rules import RiskAssessment, RiskIndicator


def test_fusion_maps_low_cnn_and_metadata_risk_to_safe() -> None:
    metadata = RiskAssessment(label="safe", risk_score=0.0, indicators=[])

    fused = fuse_cnn_and_metadata(cnn_stego_probability=0.1, metadata_assessment=metadata)

    assert fused.label == "safe"
    assert fused.risk_score == 0.055


def test_fusion_maps_strong_cnn_evidence_to_suspicious() -> None:
    metadata = RiskAssessment(label="safe", risk_score=0.0, indicators=[])

    fused = fuse_cnn_and_metadata(cnn_stego_probability=0.8, metadata_assessment=metadata)

    assert fused.label == "suspicious"


def test_fusion_high_metadata_and_cnn_evidence_forces_dangerous() -> None:
    metadata = RiskAssessment(
        label="suspicious",
        risk_score=0.3,
        indicators=[
            RiskIndicator(
                code="embedded_binary_signature",
                severity="high",
                description="Found embedded ZIP signature.",
            )
        ],
    )

    fused = fuse_cnn_and_metadata(cnn_stego_probability=0.55, metadata_assessment=metadata)

    assert fused.label == "dangerous"
    assert fused.risk_score >= 0.7
    assert "Embedded binary" in fused.explanation


def test_fusion_embedded_signature_forces_dangerous_even_with_low_cnn() -> None:
    metadata = RiskAssessment(
        label="suspicious",
        risk_score=0.3,
        indicators=[
            RiskIndicator(
                code="embedded_binary_signature",
                severity="high",
                description="Found embedded executable signature.",
            )
        ],
    )

    fused = fuse_cnn_and_metadata(cnn_stego_probability=0.1, metadata_assessment=metadata)

    assert fused.label == "dangerous"
    assert fused.risk_score == 0.7
    assert "high-severity" in fused.explanation


def test_fusion_stego_with_small_lsb_payload_is_suspicious() -> None:
    metadata = RiskAssessment(
        label="suspicious",
        risk_score=0.3,
        indicators=[
            RiskIndicator(
                code="sequential_lsb_payload",
                severity="medium",
                description="Sequential LSB embedding detected with a small payload (~48 bytes).",
            )
        ],
    )

    fused = fuse_cnn_and_metadata(cnn_stego_probability=0.99, metadata_assessment=metadata)

    assert fused.label == "suspicious"
    assert fused.risk_score < 0.7


def test_fusion_stego_with_large_lsb_payload_is_dangerous() -> None:
    metadata = RiskAssessment(
        label="suspicious",
        risk_score=0.5,
        indicators=[
            RiskIndicator(
                code="sequential_lsb_payload_large",
                severity="high",
                description=(
                    "Sequential LSB embedding detected with a script/binary-scale payload "
                    "(~15000 bytes)."
                ),
            )
        ],
    )

    fused = fuse_cnn_and_metadata(cnn_stego_probability=0.99, metadata_assessment=metadata)

    assert fused.label == "dangerous"
    assert fused.risk_score >= 0.7
    assert "sequential LSB payload" in fused.explanation


def test_fusion_high_severity_metadata_without_signature_is_suspicious() -> None:
    metadata = RiskAssessment(
        label="suspicious",
        risk_score=0.3,
        indicators=[
            RiskIndicator(
                code="image_parse_error",
                severity="high",
                description="Image parser could not safely read the file.",
            )
        ],
    )

    fused = fuse_cnn_and_metadata(cnn_stego_probability=0.1, metadata_assessment=metadata)

    assert fused.label == "suspicious"
    assert fused.risk_score == 0.5
    assert "High-severity" in fused.explanation
