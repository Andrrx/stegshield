from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from stegshield.metadata.extract import extract_image_metadata
from stegshield.metadata.risk_rules import assess_risk


def analyze_image(path: Path) -> dict[str, object]:
    """Run the current non-ML image analysis pipeline."""
    metadata = extract_image_metadata(path)
    assessment = assess_risk(metadata)

    return {
        "file": asdict(metadata),
        "risk": asdict(assessment),
    }
