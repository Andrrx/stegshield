from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from stegshield.evaluate_fusion import FusionEvaluationConfig, evaluate_fusion  # noqa: E402
from stegshield.labels import STEGO_LABELS  # noqa: E402
from stegshield.models.cnn import create_cnn_model  # noqa: E402


def test_evaluate_fusion_compares_metadata_cnn_and_fused_outputs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    safe_dir = raw_dir / "safe"
    safe_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), color="white").save(safe_dir / "safe.png")

    split_csv = tmp_path / "split.csv"
    split_csv.write_text("path,label\nsafe/safe.png,safe\n", encoding="utf-8")

    model = create_cnn_model("yedroudj", num_classes=len(STEGO_LABELS))
    model_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": STEGO_LABELS,
            "image_size": 16,
            "model_name": "yedroudj",
            "normalization": "none",
            "task": "stego",
            "class_weights": True,
            "train_csv": "train.csv",
            "val_csv": "val.csv",
            "output_model": "model.pt",
            "output_metrics": "metrics.json",
        },
        model_path,
    )

    output_report = tmp_path / "fusion_report.json"
    report = evaluate_fusion(
        FusionEvaluationConfig(
            model_path=model_path,
            split_csv=split_csv,
            output_report=output_report,
            raw_dir=raw_dir,
            device="cpu",
        )
    )

    assert output_report.exists()
    assert report["sample_count"] == 1
    assert report["checkpoint"]["task"] == "stego"
    assert report["checkpoint"]["model_name"] == "yedroudj"
    assert set(report["methods"]) == {"metadata_only", "cnn_only", "fused"}
    for method_report in report["methods"].values():
        assert "accuracy" in method_report
        assert "macro_f1" in method_report
        assert "balanced_accuracy" in method_report
        assert "majority_class_baseline_accuracy" in method_report
        assert "false_negatives_by_class" in method_report
        assert method_report["confusion_matrix"]["labels"] == ["safe", "suspicious", "dangerous"]
