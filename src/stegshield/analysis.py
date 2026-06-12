from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from stegshield.fusion import fuse_cnn_and_metadata
from stegshield.metadata.extract import extract_image_metadata
from stegshield.metadata.risk_rules import assess_risk


def analyze_image(
    path: Path,
    cnn_model_path: Path | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    """Run metadata analysis, optionally fused with a binary stego CNN."""
    metadata = extract_image_metadata(path)
    assessment = assess_risk(metadata)
    result = {
        "file": asdict(metadata),
        "risk": asdict(assessment),
    }

    if cnn_model_path is not None:
        from stegshield.predict_cnn import predict_stego_probability

        stego_probability = predict_stego_probability(
            image_path=path,
            model_path=cnn_model_path,
            device=device,
        )
        result["fusion"] = asdict(
            fuse_cnn_and_metadata(
                cnn_stego_probability=stego_probability,
                metadata_assessment=assessment,
            )
        )

    return result
