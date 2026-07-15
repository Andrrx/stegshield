from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from stegshield.ml_metrics import auc_from_roc_points, roc_curve_points  # noqa: E402


@dataclass(frozen=True)
class PlotConfig:
    report_paths: tuple[Path, ...]
    output_dir: Path = Path("outputs/figures")
    dpi: int = 300


@dataclass(frozen=True)
class _ComparisonEntry:
    name: str
    accuracy: float | None
    macro_f1: float | None
    balanced_accuracy: float | None


@dataclass(frozen=True)
class _RocEntry:
    name: str
    fpr_points: list[float]
    tpr_points: list[float]
    auc: float


def generate_plots(config: PlotConfig) -> list[Path]:
    """Create thesis figures from training/evaluation JSON reports.

    Per report: training curves (training reports), confusion matrix heatmaps
    (CNN and fusion evaluation reports). Across reports: a metric comparison bar
    chart and overlaid ROC curves for binary CNN reports that include scores.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    comparison_entries: list[_ComparisonEntry] = []
    roc_entries: list[_RocEntry] = []
    robustness_reports: list[tuple[str, dict[str, Any]]] = []

    for report_path in config.report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        name = report_path.stem

        report_type = report.get("report_type")
        if report_type == "payload_regression":
            written.append(_plot_payload_regression(name, report, config))
            continue
        if report_type == "payload_agreement":
            written.append(_plot_payload_agreement(name, report, config))
            continue
        if report_type == "robustness_benchmark":
            written.append(_plot_robustness(name, report, config))
            robustness_reports.append((name, report))
            continue

        if "history" in report:
            written.append(_plot_training_curves(name, report, config))
            continue

        if "methods" in report:
            for method_name, metrics in report["methods"].items():
                written.append(
                    _plot_confusion_matrix(f"{name}_{method_name}", metrics, config)
                )
                comparison_entries.append(_comparison_entry(f"{name}:{method_name}", metrics))
            continue

        if "confusion_matrix" in report:
            written.append(_plot_confusion_matrix(name, report, config))
            comparison_entries.append(_comparison_entry(name, report))
            binary_scores = report.get("binary_scores")
            if isinstance(binary_scores, dict):
                fpr_points, tpr_points = roc_curve_points(
                    binary_scores["scores"], binary_scores["actuals"]
                )
                roc_entries.append(
                    _RocEntry(
                        name=name,
                        fpr_points=fpr_points,
                        tpr_points=tpr_points,
                        auc=auc_from_roc_points(fpr_points, tpr_points),
                    )
                )
            continue

        raise ValueError(f"Unrecognized report format: {report_path}")

    if len(comparison_entries) >= 2:
        written.append(_plot_comparison(comparison_entries, config))
    if roc_entries:
        written.append(_plot_roc_curves(roc_entries, config))
    if len(robustness_reports) >= 2:
        written.append(_plot_robustness_overlay(robustness_reports, config))
    return written


def _comparison_entry(name: str, metrics: dict[str, Any]) -> _ComparisonEntry:
    return _ComparisonEntry(
        name=name,
        accuracy=metrics.get("accuracy"),
        macro_f1=metrics.get("macro_f1"),
        balanced_accuracy=metrics.get("balanced_accuracy"),
    )


def _plot_training_curves(name: str, report: dict[str, Any], config: PlotConfig) -> Path:
    history = report["history"]
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    val_macro_f1 = [entry["val_metrics"]["macro_f1"] for entry in history]
    val_balanced_accuracy = [entry["val_metrics"]["balanced_accuracy"] for entry in history]
    best_epoch = report.get("best_epoch")

    figure, (loss_axis, metric_axis) = plt.subplots(1, 2, figsize=(11, 4.2))

    loss_axis.plot(epochs, train_loss, marker="o", markersize=3, label="Train loss")
    loss_axis.plot(epochs, val_loss, marker="o", markersize=3, label="Validation loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Cross-entropy loss")
    loss_axis.set_title("Loss")
    loss_axis.grid(alpha=0.3)
    loss_axis.legend()

    metric_axis.plot(epochs, val_macro_f1, marker="o", markersize=3, label="Validation macro F1")
    metric_axis.plot(
        epochs,
        val_balanced_accuracy,
        marker="s",
        markersize=3,
        label="Validation balanced accuracy",
    )
    if best_epoch is not None:
        metric_axis.axvline(
            best_epoch, color="gray", linestyle="--", linewidth=1, label=f"Best epoch ({best_epoch})"
        )
    metric_axis.set_xlabel("Epoch")
    metric_axis.set_ylabel("Score")
    metric_axis.set_ylim(0.0, 1.0)
    metric_axis.set_title("Validation metrics")
    metric_axis.grid(alpha=0.3)
    metric_axis.legend()

    figure.suptitle(name)
    return _save(figure, config, f"{name}_training_curves.png")


def _plot_confusion_matrix(name: str, metrics: dict[str, Any], config: PlotConfig) -> Path:
    confusion = metrics["confusion_matrix"]
    labels = confusion["labels"]
    matrix = confusion["rows_actual_columns_predicted"]
    row_totals = [max(sum(row), 1) for row in matrix]
    normalized = [
        [cell / row_totals[row_index] for cell in row] for row_index, row in enumerate(matrix)
    ]

    figure, axis = plt.subplots(figsize=(4.6, 4.0))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    figure.colorbar(image, ax=axis, fraction=0.046, label="Row-normalized rate")

    for row_index, row in enumerate(matrix):
        for column_index, cell in enumerate(row):
            color = "white" if normalized[row_index][column_index] > 0.5 else "black"
            axis.text(
                column_index,
                row_index,
                f"{cell}\n({normalized[row_index][column_index]:.2f})",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(name)
    return _save(figure, config, f"{name}_confusion_matrix.png")


def _plot_comparison(entries: list[_ComparisonEntry], config: PlotConfig) -> Path:
    metric_names = ("accuracy", "macro_f1", "balanced_accuracy")
    metric_labels = ("Accuracy", "Macro F1", "Balanced accuracy")
    bar_width = 0.8 / len(metric_names)

    figure, axis = plt.subplots(figsize=(max(6.0, 1.8 * len(entries)), 4.4))
    for metric_index, (metric_name, metric_label) in enumerate(
        zip(metric_names, metric_labels, strict=True)
    ):
        values = [getattr(entry, metric_name) or 0.0 for entry in entries]
        positions = [
            entry_index + (metric_index - (len(metric_names) - 1) / 2) * bar_width
            for entry_index in range(len(entries))
        ]
        bars = axis.bar(positions, values, width=bar_width, label=metric_label)
        axis.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)

    axis.set_xticks(range(len(entries)), [entry.name for entry in entries], rotation=20, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Model and method comparison")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    return _save(figure, config, "model_comparison.png")


def _plot_roc_curves(entries: list[_RocEntry], config: PlotConfig) -> Path:
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    for entry in entries:
        axis.plot(
            entry.fpr_points,
            entry.tpr_points,
            linewidth=1.6,
            label=f"{entry.name} (AUC = {entry.auc:.3f})",
        )
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate (stego recall)")
    axis.set_title("ROC curves: binary clean/stego CNNs")
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right", fontsize=8)
    return _save(figure, config, "roc_curves.png")


def _plot_payload_regression(name: str, report: dict[str, Any], config: PlotConfig) -> Path:
    true_bytes = report["points"]["true_bytes"]
    pred_bytes = report["points"]["pred_bytes"]
    capacity = report.get("capacity_bytes", max(max(true_bytes), max(pred_bytes)))

    figure, axis = plt.subplots(figsize=(5.0, 4.8))
    # +1 keeps zero-byte points on the log axis.
    axis.scatter(
        [value + 1 for value in true_bytes],
        [value + 1 for value in pred_bytes],
        s=8,
        alpha=0.3,
        edgecolors="none",
    )
    limit = capacity + 1
    axis.plot([1, limit], [1, limit], color="gray", linestyle="--", linewidth=1, label="y = x")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("True payload bytes (+1)")
    axis.set_ylabel("CNN estimated payload bytes (+1)")
    axis.set_title(
        f"Payload regression\nMAE {report['mae_bytes']:.0f} B, "
        f"median |err| {report['median_absolute_error_bytes']:.0f} B "
        f"(n={report['supervised_count']})"
    )
    axis.grid(alpha=0.3, which="both")
    axis.legend(loc="upper left", fontsize=8)
    return _save(figure, config, f"{name}_scatter.png")


def _plot_payload_agreement(name: str, report: dict[str, Any], config: PlotConfig) -> Path:
    cnn_bytes = report["points"]["cnn_bytes"]
    statistical_bytes = report["points"]["statistical_bytes"]
    capacity = report.get("capacity_bytes", max(max(cnn_bytes), max(statistical_bytes)))

    figure, axis = plt.subplots(figsize=(5.0, 4.8))
    axis.scatter(
        [value + 1 for value in statistical_bytes],
        [value + 1 for value in cnn_bytes],
        s=8,
        alpha=0.3,
        edgecolors="none",
    )
    limit = capacity + 1
    axis.plot([1, limit], [1, limit], color="gray", linestyle="--", linewidth=1, label="y = x")
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("Statistical estimate bytes (+1)")
    axis.set_ylabel("CNN estimate bytes (+1)")
    axis.set_title(
        f"CNN vs statistical payload estimate\n"
        f"Pearson(log2) {report['pearson_log2']:.3f}, "
        f"Spearman {report['spearman_log2']:.3f} (n={report['compared_count']})"
    )
    axis.grid(alpha=0.3, which="both")
    axis.legend(loc="upper left", fontsize=8)
    return _save(figure, config, f"{name}_scatter.png")


def _plot_robustness(name: str, report: dict[str, Any], config: PlotConfig) -> Path:
    ops = report["operations"]
    labels = [op["name"] for op in ops]
    detection = [op["stego_detection_rate"] for op in ops]
    fpr = [op["clean_fpr"] for op in ops]
    positions = list(range(len(ops)))
    bar_width = 0.4

    figure, axis = plt.subplots(figsize=(max(7.0, 0.7 * len(ops)), 4.6))
    detect_bars = axis.bar(
        [p - bar_width / 2 for p in positions], detection, bar_width, label="Stego detection rate"
    )
    axis.bar(
        [p + bar_width / 2 for p in positions], fpr, bar_width, label="Clean false-positive rate"
    )
    axis.bar_label(detect_bars, fmt="%.2f", fontsize=7, padding=1)
    # Shade the lossy (JPEG) operations to mark where spatial detection collapses.
    for index, op in enumerate(ops):
        if op["lossy"]:
            axis.axvspan(index - 0.5, index + 0.5, color="tab:red", alpha=0.06)

    axis.set_xticks(positions, labels, rotation=40, ha="right")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Rate")
    axis.set_title(f"Robustness to image processing\n{name}")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(loc="center right", fontsize=8)
    return _save(figure, config, f"{name}_robustness.png")


def _plot_robustness_overlay(
    reports: list[tuple[str, dict[str, Any]]], config: PlotConfig
) -> Path:
    # Balanced accuracy, not raw detection rate: under noise a detector can hit
    # 100% detection by also flagging clean images, so detection rate alone
    # overstates robustness. Balanced accuracy penalizes that.
    figure, axis = plt.subplots(figsize=(8.0, 4.6))
    labels = [op["name"] for op in reports[0][1]["operations"]]
    positions = list(range(len(labels)))
    for name, report in reports:
        balanced = [op.get("balanced_accuracy", 0.0) for op in report["operations"]]
        axis.plot(positions, balanced, marker="o", markersize=4, linewidth=1.6, label=name)
    axis.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="chance")
    for index, op in enumerate(reports[0][1]["operations"]):
        if op["lossy"]:
            axis.axvspan(index - 0.5, index + 0.5, color="tab:red", alpha=0.06)

    axis.set_xticks(positions, labels, rotation=40, ha="right")
    axis.set_ylim(0.4, 1.05)
    axis.set_ylabel("Balanced accuracy")
    axis.set_title("Robustness (balanced accuracy): baseline vs hardened")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    return _save(figure, config, "robustness_overlay.png")


def _save(figure: plt.Figure, config: PlotConfig, filename: str) -> Path:
    output_path = config.output_dir / filename
    figure.tight_layout()
    figure.savefig(output_path, dpi=config.dpi)
    plt.close(figure)
    return output_path
