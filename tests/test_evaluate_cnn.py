from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from stegshield.evaluate_cnn import EvaluationConfig, evaluate_cnn  # noqa: E402
from stegshield.labels import STEGO_LABELS  # noqa: E402
from stegshield.models.cnn import create_cnn_model  # noqa: E402


def test_evaluate_cnn_writes_metrics_report(tmp_path: Path) -> None:
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
            "balanced_sampler": True,
            "selection_metric": "macro_f1",
            "train_csv": "train.csv",
            "val_csv": "val.csv",
            "output_model": "model.pt",
            "output_metrics": "metrics.json",
            "batch_size": 1,
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "device": "cpu",
            "num_workers": 0,
            "timestamp": "2026-06-04T00:00:00+00:00",
        },
        model_path,
    )

    output_report = tmp_path / "report.json"
    report = evaluate_cnn(
        EvaluationConfig(
            model_path=model_path,
            split_csv=split_csv,
            output_report=output_report,
            raw_dir=raw_dir,
            batch_size=1,
            device="cpu",
        )
    )

    assert output_report.exists()
    assert report["checkpoint"]["model_name"] == "yedroudj"
    assert report["checkpoint"]["task"] == "stego"
    assert report["checkpoint"]["labels"] == list(STEGO_LABELS)
    assert report["checkpoint"]["class_weights"] is True
    assert report["checkpoint"]["balanced_sampler"] is True
    assert report["checkpoint"]["selection_metric"] == "macro_f1"
    assert report["checkpoint"]["train_csv"] == "train.csv"
    assert report["checkpoint"]["output_model"] == "model.pt"
    assert report["checkpoint"]["output_metrics"] == "metrics.json"
    assert report["checkpoint"]["timestamp"] == "2026-06-04T00:00:00+00:00"
    assert report["split_csv"] == str(split_csv)
    assert report["task"] == "stego"
    assert report["model_name"] == "yedroudj"
    assert report["labels"] == list(STEGO_LABELS)
    assert "accuracy" in report
    assert "macro_f1" in report
    assert "balanced_accuracy" in report
    assert "majority_class_baseline_accuracy" in report
    assert report["majority_class_baseline_accuracy"] == 1.0
    assert report["false_negatives_by_class"]["clean"] in {0, 1}
    assert report["confusion_matrix"]["labels"] == list(STEGO_LABELS)
    assert report["per_class"]["clean"]["support"] == 1


def test_evaluate_cnn_loads_multitask_checkpoint(tmp_path: Path) -> None:
    # A payload-head (multi-task) checkpoint must still evaluate on detection:
    # the head is built so load_state_dict succeeds, but forward() ignores it.
    raw_dir = tmp_path / "raw"
    safe_dir = raw_dir / "safe"
    safe_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), color="white").save(safe_dir / "safe.png")

    split_csv = tmp_path / "split.csv"
    split_csv.write_text("path,label\nsafe/safe.png,safe\n", encoding="utf-8")

    model = create_cnn_model("steganalysis", num_classes=len(STEGO_LABELS), payload_head=True)
    model_path = tmp_path / "multitask.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": STEGO_LABELS,
            "image_size": 16,
            "model_name": "steganalysis",
            "normalization": "raw255",
            "crop": "top-left",
            "task": "stego",
            "payload_head": True,
            "payload_loss_weight": 0.5,
            "payload_capacity_bytes": 16 * 16 * 3 // 8,
        },
        model_path,
    )

    report = evaluate_cnn(
        EvaluationConfig(
            model_path=model_path,
            split_csv=split_csv,
            output_report=tmp_path / "report.json",
            raw_dir=raw_dir,
            batch_size=1,
            device="cpu",
        )
    )
    assert "accuracy" in report
    assert report["model_name"] == "steganalysis"
