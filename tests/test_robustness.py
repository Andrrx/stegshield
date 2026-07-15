import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from stegshield.data.synth_lsb import embed_sequential_lsb  # noqa: E402
from stegshield.labels import STEGO_LABELS  # noqa: E402
from stegshield.models.cnn import create_cnn_model  # noqa: E402
from stegshield.robustness import RobustnessConfig, evaluate_robustness  # noqa: E402


def _checkpoint(tmp_path: Path, image_size: int) -> Path:
    model = create_cnn_model("steganalysis", num_classes=len(STEGO_LABELS))
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": STEGO_LABELS,
            "image_size": image_size,
            "model_name": "steganalysis",
            "normalization": "raw255",
            "crop": "top-left",
            "task": "stego",
        },
        path,
    )
    return path


def _split(tmp_path: Path, image_size: int, count: int = 3) -> Path:
    raw = tmp_path / "raw"
    (raw / "safe").mkdir(parents=True)
    (raw / "dangerous").mkdir(parents=True)
    base = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    rows = ["path,label"]
    for i in range(count):
        Image.fromarray(base, "RGB").save(raw / "safe" / f"c{i}.png")
        Image.fromarray(embed_sequential_lsb(base, b"\x05" * 300), "RGB").save(
            raw / "dangerous" / f"s{i}.png"
        )
        rows.append(f"safe/c{i}.png,safe")
        rows.append(f"dangerous/s{i}.png,dangerous")
    csv = tmp_path / "split.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return csv


def test_robustness_report_has_all_operations(tmp_path: Path) -> None:
    image_size = 64
    model_path = _checkpoint(tmp_path, image_size)
    split_csv = _split(tmp_path, image_size)

    report = evaluate_robustness(
        RobustnessConfig(
            model_path=model_path,
            split_csv=split_csv,
            output_report=tmp_path / "robust.json",
            raw_dir=tmp_path / "raw",
            device="cpu",
            limit_per_class=3,
            batch_size=2,
        )
    )

    assert report["report_type"] == "robustness_benchmark"
    op_names = [op["name"] for op in report["operations"]]
    assert "none" in op_names and "jpeg_75" in op_names
    for op in report["operations"]:
        assert 0.0 <= op["stego_detection_rate"] <= 1.0
        assert 0.0 <= op["clean_fpr"] <= 1.0
        assert 0.0 <= op["balanced_accuracy"] <= 1.0
    assert (tmp_path / "robust.json").exists()


def test_robustness_requires_both_classes(tmp_path: Path) -> None:
    image_size = 64
    model_path = _checkpoint(tmp_path, image_size)
    raw = tmp_path / "raw"
    (raw / "safe").mkdir(parents=True)
    Image.new("RGB", (image_size, image_size), "white").save(raw / "safe" / "c.png")
    csv = tmp_path / "clean_only.csv"
    csv.write_text("path,label\nsafe/c.png,safe\n", encoding="utf-8")

    with pytest.raises(ValueError, match="both clean and stego"):
        evaluate_robustness(
            RobustnessConfig(
                model_path=model_path,
                split_csv=csv,
                output_report=tmp_path / "out.json",
                raw_dir=raw,
                device="cpu",
            )
        )


def test_robustness_plot_renders(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from stegshield.plots import PlotConfig, generate_plots

    report = {
        "report_type": "robustness_benchmark",
        "threshold": 0.5,
        "operations": [
            {"name": "none", "lossy": False, "stego_detection_rate": 1.0, "clean_fpr": 0.0,
             "balanced_accuracy": 1.0},
            {"name": "resize_0.50", "lossy": False, "stego_detection_rate": 0.98, "clean_fpr": 0.0,
             "balanced_accuracy": 0.99},
            {"name": "jpeg_75", "lossy": True, "stego_detection_rate": 0.02, "clean_fpr": 0.0,
             "balanced_accuracy": 0.51},
        ],
    }
    path_a = tmp_path / "robustness_baseline.json"
    path_b = tmp_path / "robustness_hardened.json"
    path_a.write_text(json.dumps(report), encoding="utf-8")
    hardened = json.loads(json.dumps(report))
    hardened["operations"][2]["stego_detection_rate"] = 0.10
    path_b.write_text(json.dumps(hardened), encoding="utf-8")

    written = generate_plots(
        PlotConfig(report_paths=(path_a, path_b), output_dir=tmp_path / "figs", dpi=72)
    )
    names = {p.name for p in written}
    assert "robustness_baseline_robustness.png" in names
    assert "robustness_hardened_robustness.png" in names
    assert "robustness_overlay.png" in names
