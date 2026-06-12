from __future__ import annotations

from typing import Any


def classification_report_from_confusion(
    confusion: list[list[int]],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[index][index] for index in range(len(labels)))
    per_class = {}
    false_negatives_by_class = {}

    for index, label in enumerate(labels):
        true_positive = confusion[index][index]
        false_positive = sum(row[index] for row in confusion) - true_positive
        false_negative = sum(confusion[index]) - true_positive
        support = sum(confusion[index])
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1_score = _safe_divide(2 * precision * recall, precision + recall)

        false_negatives_by_class[label] = false_negative
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1_score, 6),
            "support": support,
        }

    macro_f1 = sum(metrics["f1_score"] for metrics in per_class.values()) / len(per_class)
    balanced_accuracy = (
        sum(metrics["recall"] for metrics in per_class.values()) / len(per_class)
    )
    majority_support = max((sum(row) for row in confusion), default=0)

    return {
        "accuracy": round(_safe_divide(correct, total), 6),
        "majority_class_baseline_accuracy": round(_safe_divide(majority_support, total), 6),
        "macro_f1": round(macro_f1, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "per_class": per_class,
        "false_negatives_by_class": false_negatives_by_class,
        "confusion_matrix": {
            "labels": list(labels),
            "rows_actual_columns_predicted": confusion,
        },
    }


def empty_confusion(label_count: int) -> list[list[int]]:
    return [[0 for _ in range(label_count)] for _ in range(label_count)]


def update_confusion(confusion: list[list[int]], actual: int, predicted: int) -> None:
    confusion[actual][predicted] += 1


def better_selection_metric(
    candidate: float,
    current_best: float | None,
    candidate_epoch: int,
    best_epoch: int | None,
) -> bool:
    if current_best is None:
        return True
    if candidate > current_best:
        return True
    return candidate == current_best and (best_epoch is None or candidate_epoch < best_epoch)


def roc_curve_points(
    scores: list[float],
    actuals: list[int],
) -> tuple[list[float], list[float]]:
    """ROC curve from per-sample positive-class scores.

    ``actuals`` holds 1 for positive samples and 0 for negatives. Returns
    (false_positive_rates, true_positive_rates) sorted from (0, 0) to (1, 1),
    with one point per distinct score threshold.
    """
    if len(scores) != len(actuals):
        raise ValueError("scores and actuals must have the same length.")
    positives = sum(actuals)
    negatives = len(actuals) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC requires at least one positive and one negative sample.")

    ranked = sorted(zip(scores, actuals, strict=True), key=lambda pair: pair[0], reverse=True)
    fpr_points = [0.0]
    tpr_points = [0.0]
    true_positives = 0
    false_positives = 0
    previous_score: float | None = None

    for score, actual in ranked:
        if previous_score is not None and score != previous_score:
            fpr_points.append(false_positives / negatives)
            tpr_points.append(true_positives / positives)
        if actual == 1:
            true_positives += 1
        else:
            false_positives += 1
        previous_score = score

    fpr_points.append(1.0)
    tpr_points.append(1.0)
    return fpr_points, tpr_points


def auc_from_roc_points(fpr_points: list[float], tpr_points: list[float]) -> float:
    """Area under the ROC curve via trapezoidal integration."""
    area = 0.0
    for index in range(1, len(fpr_points)):
        width = fpr_points[index] - fpr_points[index - 1]
        height = (tpr_points[index] + tpr_points[index - 1]) / 2
        area += width * height
    return area


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
