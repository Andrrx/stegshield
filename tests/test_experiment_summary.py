import json
from pathlib import Path

from typer.testing import CliRunner

from stegshield.cli import app
from stegshield.experiment_summary import ExperimentSummaryConfig, summarize_experiments


def test_summarize_experiments_combines_cnn_and_fusion_reports(tmp_path: Path) -> None:
    cnn_report = tmp_path / "cnn.json"
    cnn_report.write_text(
        json.dumps(
            {
                "split_csv": "data/splits/test_standard.csv",
                "model_name": "steganalysis",
                "labels": ["clean", "stego"],
                "accuracy": 0.6,
                "macro_f1": 0.448,
                "balanced_accuracy": 0.633,
                "per_class": {
                    "clean": {"recall": 1.0},
                    "stego": {"recall": 0.266},
                },
            }
        ),
        encoding="utf-8",
    )
    fusion_report = tmp_path / "fusion.json"
    fusion_report.write_text(
        json.dumps(
            {
                "split_csv": "data/splits/test_standard.csv",
                "checkpoint": {"model_name": "steganalysis"},
                "methods": {
                    "metadata_only": {
                        "accuracy": 0.7,
                        "macro_f1": 0.5,
                        "balanced_accuracy": 0.55,
                        "per_class": {
                            "safe": {"recall": 1.0},
                            "dangerous": {"recall": 0.0},
                        },
                    },
                    "fused": {
                        "accuracy": 0.8,
                        "macro_f1": 0.65,
                        "balanced_accuracy": 0.7,
                        "per_class": {
                            "safe": {"recall": 1.0},
                            "dangerous": {"recall": 0.5},
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "summary.md"
    rows = summarize_experiments(
        ExperimentSummaryConfig(
            report_paths=(cnn_report, fusion_report),
            output_path=output_path,
        )
    )

    assert len(rows) == 3
    assert rows[0]["method"] == "cnn_binary"
    assert rows[0]["safe_clean_recall"] == 1.0
    assert rows[0]["stego_recall"] == 0.266
    assert rows[2]["method"] == "fused"
    assert rows[2]["dangerous_recall"] == 0.5
    assert "Experiment Summary" in output_path.read_text(encoding="utf-8")


def test_summarize_experiments_cli_writes_markdown(tmp_path: Path) -> None:
    report_path = tmp_path / "cnn.json"
    report_path.write_text(
        json.dumps(
            {
                "split_csv": "data/splits/test_standard.csv",
                "model_name": "baseline",
                "labels": ["clean", "stego"],
                "accuracy": 0.25,
                "macro_f1": 0.2,
                "balanced_accuracy": 0.5,
                "per_class": {
                    "clean": {"recall": 1.0},
                    "stego": {"recall": 0.0},
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.md"

    result = CliRunner().invoke(
        app,
        [
            "summarize-experiments",
            str(report_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "CNN misses all stego samples" in output_path.read_text(encoding="utf-8")
