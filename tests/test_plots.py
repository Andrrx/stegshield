import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from stegshield.plots import PlotConfig, generate_plots  # noqa: E402


def _classification_metrics(labels: list[str], matrix: list[list[int]]) -> dict[str, object]:
    return {
        "accuracy": 0.8,
        "macro_f1": 0.75,
        "balanced_accuracy": 0.78,
        "confusion_matrix": {
            "labels": labels,
            "rows_actual_columns_predicted": matrix,
        },
    }


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_generate_plots_from_all_report_types(tmp_path: Path) -> None:
    training_report = _write(
        tmp_path / "yedroudj_training.json",
        {
            "best_epoch": 2,
            "history": [
                {
                    "epoch": epoch,
                    "train_loss": 1.0 / epoch,
                    "val_loss": 1.2 / epoch,
                    "val_metrics": {"macro_f1": 0.3 * epoch, "balanced_accuracy": 0.31 * epoch},
                }
                for epoch in (1, 2, 3)
            ],
        },
    )
    cnn_report = _write(
        tmp_path / "yedroudj_test.json",
        {
            **_classification_metrics(["clean", "stego"], [[8, 2], [3, 7]]),
            "binary_scores": {
                "positive_label": "stego",
                "scores": [0.9, 0.8, 0.7, 0.4, 0.3, 0.2],
                "actuals": [1, 1, 0, 1, 0, 0],
            },
        },
    )
    fusion_report = _write(
        tmp_path / "fusion_test.json",
        {
            "methods": {
                "metadata_only": _classification_metrics(
                    ["safe", "suspicious", "dangerous"],
                    [[5, 1, 0], [1, 4, 1], [0, 2, 6]],
                ),
                "fused": _classification_metrics(
                    ["safe", "suspicious", "dangerous"],
                    [[6, 0, 0], [1, 5, 0], [0, 1, 7]],
                ),
            }
        },
    )

    output_dir = tmp_path / "figures"
    written = generate_plots(
        PlotConfig(
            report_paths=(training_report, cnn_report, fusion_report),
            output_dir=output_dir,
            dpi=72,
        )
    )

    written_names = {path.name for path in written}
    assert written_names == {
        "yedroudj_training_training_curves.png",
        "yedroudj_test_confusion_matrix.png",
        "fusion_test_metadata_only_confusion_matrix.png",
        "fusion_test_fused_confusion_matrix.png",
        "model_comparison.png",
        "roc_curves.png",
    }
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0


def test_generate_plots_rejects_unknown_report(tmp_path: Path) -> None:
    unknown = _write(tmp_path / "unknown.json", {"something": 1})

    with pytest.raises(ValueError, match="Unrecognized report format"):
        generate_plots(PlotConfig(report_paths=(unknown,), output_dir=tmp_path / "figures"))
