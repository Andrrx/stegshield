import json
from pathlib import Path

import pytest

from stegshield.payload_eval import _pearson, _ranks, _spearman


def test_pearson_perfect_positive() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    assert _pearson(xs, [2.0, 4.0, 6.0, 8.0]) == pytest.approx(1.0)
    assert _pearson(xs, [8.0, 6.0, 4.0, 2.0]) == pytest.approx(-1.0)


def test_spearman_monotonic_nonlinear_is_one() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 4.0, 9.0, 16.0]  # monotonic but nonlinear
    assert _spearman(xs, ys) == pytest.approx(1.0)


def test_ranks_average_ties() -> None:
    assert _ranks([10.0, 10.0, 20.0]) == [0.5, 0.5, 2.0]


def test_pearson_zero_variance_is_zero() -> None:
    assert _pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


def test_plots_render_payload_reports(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from stegshield.plots import PlotConfig, generate_plots

    regression = tmp_path / "payload_regression_test.json"
    regression.write_text(
        json.dumps(
            {
                "report_type": "payload_regression",
                "capacity_bytes": 24576,
                "supervised_count": 4,
                "mae_bytes": 123.0,
                "median_absolute_error_bytes": 80.0,
                "mae_log2": 0.5,
                "points": {
                    "true_bytes": [16, 200, 2000, 20000],
                    "pred_bytes": [20, 180, 2200, 19000],
                },
            }
        ),
        encoding="utf-8",
    )
    agreement = tmp_path / "payload_agreement_test.json"
    agreement.write_text(
        json.dumps(
            {
                "report_type": "payload_agreement",
                "capacity_bytes": 24576,
                "compared_count": 3,
                "pearson_log2": 0.95,
                "spearman_log2": 0.93,
                "median_abs_diff_bytes": 50.0,
                "points": {"cnn_bytes": [48, 500, 9000], "statistical_bytes": [48, 480, 9600]},
            }
        ),
        encoding="utf-8",
    )

    written = generate_plots(
        PlotConfig(report_paths=(regression, agreement), output_dir=tmp_path / "figs", dpi=72)
    )
    names = {path.name for path in written}
    assert names == {
        "payload_regression_test_scatter.png",
        "payload_agreement_test_scatter.png",
    }
    for path in written:
        assert path.stat().st_size > 0
