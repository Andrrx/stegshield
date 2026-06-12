from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentSummaryConfig:
    report_paths: tuple[Path, ...]
    output_path: Path
    output_format: str = "markdown"


def summarize_experiments(config: ExperimentSummaryConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for report_path in config.report_paths:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows.extend(_rows_from_report(report_path, report))

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    if config.output_format == "json":
        config.output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    elif config.output_format == "markdown":
        config.output_path.write_text(_markdown_table(rows), encoding="utf-8")
    else:
        raise ValueError("output_format must be markdown or json.")
    return rows


def _rows_from_report(report_path: Path, report: dict[str, Any]) -> list[dict[str, object]]:
    if "methods" in report:
        return _fusion_rows(report_path, report)
    return [_cnn_row(report_path, report)]


def _cnn_row(report_path: Path, report: dict[str, Any]) -> dict[str, object]:
    labels = tuple(report.get("labels", ()))
    model_name = str(report.get("model_name") or report.get("checkpoint", {}).get("model_name", "cnn"))
    return _summary_row(
        method="cnn_binary",
        model=model_name,
        split=Path(str(report.get("split_csv", report_path))).stem,
        metrics=report,
        safe_or_clean_label="clean" if "clean" in labels else "safe",
        stego_label="stego" if "stego" in labels else None,
        dangerous_label="dangerous" if "dangerous" in labels else None,
        notes=_cnn_notes(report),
    )


def _fusion_rows(report_path: Path, report: dict[str, Any]) -> list[dict[str, object]]:
    checkpoint = report.get("checkpoint", {})
    model_name = str(checkpoint.get("model_name", "cnn"))
    split = Path(str(report.get("split_csv", report_path))).stem
    payload_source = report.get("payload_source")
    rows = []
    for method_name, metrics in report.get("methods", {}).items():
        # Tag the method with the payload source so statistical/cnn/both fusion
        # variants on the same split appear as distinct rows.
        labelled_method = (
            f"{method_name}[{payload_source}]"
            if payload_source is not None and method_name == "fused"
            else method_name
        )
        rows.append(
            _summary_row(
                method=labelled_method,
                model=model_name,
                split=split,
                metrics=metrics,
                safe_or_clean_label="safe",
                stego_label=None,
                dangerous_label="dangerous",
                notes=_fusion_notes(method_name, payload_source),
            )
        )
    return rows


def _summary_row(
    method: str,
    model: str,
    split: str,
    metrics: dict[str, Any],
    safe_or_clean_label: str,
    stego_label: str | None,
    dangerous_label: str | None,
    notes: str,
) -> dict[str, object]:
    return {
        "method": method,
        "model": model,
        "split": split,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "safe_clean_recall": _recall(metrics, safe_or_clean_label),
        "stego_recall": _recall(metrics, stego_label) if stego_label else "",
        "dangerous_recall": _recall(metrics, dangerous_label) if dangerous_label else "",
        "notes": notes,
    }


def _recall(metrics: dict[str, Any], label: str | None) -> float | str:
    if label is None:
        return ""
    per_class = metrics.get("per_class", {})
    if not isinstance(per_class, dict) or label not in per_class:
        return ""
    label_metrics = per_class[label]
    if not isinstance(label_metrics, dict):
        return ""
    return label_metrics.get("recall", "")


def _cnn_notes(report: dict[str, Any]) -> str:
    per_class = report.get("per_class", {})
    if isinstance(per_class, dict):
        clean = per_class.get("clean", {})
        stego = per_class.get("stego", {})
        if isinstance(clean, dict) and clean.get("recall") == 0:
            return "CNN misses all clean samples."
        if isinstance(stego, dict) and stego.get("recall") == 0:
            return "CNN misses all stego samples."
    return "Binary CNN stego-evidence estimator."


def _fusion_notes(method_name: str, payload_source: str | None = None) -> str:
    notes = {
        "metadata_only": "Static metadata/file-structure baseline.",
        "cnn_only": "Binary CNN probability converted to risk; cannot produce dangerous alone.",
        "fused": "Final hybrid classifier combining CNN and metadata evidence.",
    }
    note = notes.get(method_name, "Evaluation method.")
    if method_name == "fused" and payload_source is not None:
        note += f" Payload severity source: {payload_source}."
    return note


def _markdown_table(rows: list[dict[str, object]]) -> str:
    headers = (
        "method",
        "model",
        "split",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "safe_clean_recall",
        "stego_recall",
        "dangerous_recall",
        "notes",
    )
    lines = [
        "# Experiment Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(header, "")) for header in headers) + " |")
    lines.append("")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "/")


def config_as_dict(config: ExperimentSummaryConfig) -> dict[str, object]:
    return asdict(config)
