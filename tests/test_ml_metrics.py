from stegshield.ml_metrics import better_selection_metric, classification_report_from_confusion


def test_classification_report_detects_majority_class_collapse() -> None:
    report = classification_report_from_confusion(
        confusion=[
            [0, 25],
            [0, 75],
        ],
        labels=("clean", "stego"),
    )

    assert report["accuracy"] == 0.75
    assert report["majority_class_baseline_accuracy"] == 0.75
    assert report["macro_f1"] == 0.428571
    assert report["balanced_accuracy"] == 0.5
    assert report["per_class"]["clean"]["recall"] == 0.0
    assert report["per_class"]["stego"]["recall"] == 1.0
    assert report["false_negatives_by_class"]["clean"] == 25


def test_classification_report_balanced_accuracy_averages_recalls() -> None:
    report = classification_report_from_confusion(
        confusion=[
            [8, 2],
            [3, 7],
        ],
        labels=("clean", "stego"),
    )

    assert report["accuracy"] == 0.75
    assert report["balanced_accuracy"] == 0.75
    assert report["macro_f1"] == 0.749374


def test_selection_metric_prefers_metric_value_over_accuracy() -> None:
    assert better_selection_metric(
        candidate=0.55,
        current_best=0.42,
        candidate_epoch=2,
        best_epoch=1,
    )
    assert not better_selection_metric(
        candidate=0.42,
        current_best=0.55,
        candidate_epoch=2,
        best_epoch=1,
    )


def test_roc_curve_points_and_auc_perfect_separation() -> None:
    from stegshield.ml_metrics import auc_from_roc_points, roc_curve_points

    fpr_points, tpr_points = roc_curve_points(
        scores=[0.9, 0.8, 0.2, 0.1], actuals=[1, 1, 0, 0]
    )

    assert fpr_points[0] == 0.0 and tpr_points[0] == 0.0
    assert fpr_points[-1] == 1.0 and tpr_points[-1] == 1.0
    assert auc_from_roc_points(fpr_points, tpr_points) == 1.0


def test_roc_auc_uninformative_scores_is_half() -> None:
    from stegshield.ml_metrics import auc_from_roc_points, roc_curve_points

    # Half of the positive/negative pairs are ranked correctly -> AUC 0.5.
    fpr_points, tpr_points = roc_curve_points(
        scores=[0.9, 0.8, 0.2, 0.1], actuals=[1, 0, 0, 1]
    )

    assert auc_from_roc_points(fpr_points, tpr_points) == 0.5


def test_roc_curve_requires_both_classes() -> None:
    import pytest

    from stegshield.ml_metrics import roc_curve_points

    with pytest.raises(ValueError, match="positive and one negative"):
        roc_curve_points(scores=[0.5, 0.4], actuals=[1, 1])
