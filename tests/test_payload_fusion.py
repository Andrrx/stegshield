from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from stegshield.data.synth_lsb import crop_capacity_bytes, embed_sequential_lsb  # noqa: E402
from stegshield.evaluate_fusion import FusionEvaluationConfig, evaluate_fusion  # noqa: E402
from stegshield.labels import STEGO_LABELS  # noqa: E402
from stegshield.metadata.risk_rules import cnn_payload_indicators  # noqa: E402
from stegshield.models.cnn import create_cnn_model  # noqa: E402


def test_cnn_payload_indicators_severity_gate() -> None:
    assert cnn_payload_indicators(None) == []
    assert cnn_payload_indicators(0) == []

    small = cnn_payload_indicators(48)
    assert [indicator.code for indicator in small] == ["cnn_lsb_payload"]
    assert small[0].severity == "medium"

    large = cnn_payload_indicators(5000)
    assert [indicator.code for indicator in large] == ["cnn_lsb_payload_large"]
    assert large[0].severity == "high"


def _make_payload_checkpoint(tmp_path: Path, image_size: int) -> Path:
    model = create_cnn_model("steganalysis", num_classes=2, payload_head=True)
    path = tmp_path / "multitask.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": STEGO_LABELS,
            "image_size": image_size,
            "model_name": "steganalysis",
            "normalization": "raw255",
            "crop": "top-left",
            "task": "stego",
            "payload_head": True,
            "payload_loss_weight": 0.5,
            "payload_capacity_bytes": crop_capacity_bytes(image_size),
        },
        path,
    )
    return path


def _make_split(tmp_path: Path, image_size: int) -> Path:
    raw = tmp_path / "raw"
    (raw / "safe").mkdir(parents=True)
    (raw / "dangerous").mkdir(parents=True)
    base = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    Image.fromarray(base, mode="RGB").save(raw / "safe" / "clean.png")
    Image.fromarray(embed_sequential_lsb(base, b"\x07" * 600), mode="RGB").save(
        raw / "dangerous" / "stego.png"
    )
    split_csv = tmp_path / "split.csv"
    split_csv.write_text(
        "path,label\nsafe/clean.png,safe\ndangerous/stego.png,dangerous\n", encoding="utf-8"
    )
    return split_csv


@pytest.mark.parametrize("payload_source", ["statistical", "cnn", "both"])
def test_fusion_runs_with_each_payload_source(tmp_path: Path, payload_source: str) -> None:
    image_size = 64
    model_path = _make_payload_checkpoint(tmp_path, image_size)
    split_csv = _make_split(tmp_path, image_size)

    report = evaluate_fusion(
        FusionEvaluationConfig(
            model_path=model_path,
            split_csv=split_csv,
            output_report=tmp_path / f"fusion_{payload_source}.json",
            raw_dir=tmp_path / "raw",
            device="cpu",
            payload_source=payload_source,
        )
    )

    assert report["payload_source"] == payload_source
    assert set(report["methods"]) == {"metadata_only", "cnn_only", "fused"}


def test_cnn_payload_source_requires_payload_head(tmp_path: Path) -> None:
    image_size = 64
    detection_model = create_cnn_model("steganalysis", num_classes=2)
    model_path = tmp_path / "detect.pt"
    torch.save(
        {
            "model_state_dict": detection_model.state_dict(),
            "labels": STEGO_LABELS,
            "image_size": image_size,
            "model_name": "steganalysis",
            "normalization": "raw255",
            "crop": "top-left",
            "task": "stego",
        },
        model_path,
    )
    split_csv = _make_split(tmp_path, image_size)

    with pytest.raises(ValueError, match="--payload-head"):
        evaluate_fusion(
            FusionEvaluationConfig(
                model_path=model_path,
                split_csv=split_csv,
                output_report=tmp_path / "out.json",
                raw_dir=tmp_path / "raw",
                device="cpu",
                payload_source="cnn",
            )
        )
