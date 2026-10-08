import numpy as np
import pytest
from matplotlib.figure import Figure

from experiments.tree.plot_tree_budget_gain import plot_budget_gain
from experiments.tree.random_tree_budget_gain import write_rows


@pytest.mark.parametrize("options, expected_gain", [({}, 40), ({"threshold": 0.05}, 20)])
def test_plot_selects_one_threshold_without_reachability_fields(
    tmp_path, monkeypatch, options, expected_gain
):
    settings = {
        "n_nodes": 8,
        "n_trees": 50,
        "p": 0.9,
        "q": 0.1,
        "physical_error": 0.2,
        "social_error": 0.05,
        "max_budget": 8,
        "random_seed": 7,
        "r": 0.5,
    }
    input_path = tmp_path / "summary.csv"
    output_path = tmp_path / "gain.png"
    write_rows(
        input_path,
        [
            {
                **settings,
                "threshold": threshold,
                "mean_budget_gain": gain,
                "ci95_low": gain - 0.05,
                "ci95_high": gain + 0.05,
            }
            for threshold, gain in ((0.1, 0.4), (0.05, 0.2))
        ],
    )
    figures = []
    original_savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        figures.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    plot_budget_gain(input_path, output_path, **options)

    assert output_path.stat().st_size > 0
    assert len(figures) == 1
    assert len(figures[0].axes) == 1
    gain_line = figures[0].axes[0].lines[0]
    np.testing.assert_equal(gain_line.get_xdata(), [0.5])
    np.testing.assert_equal(gain_line.get_ydata(), [expected_gain])


def test_plot_rejects_missing_threshold(tmp_path):
    input_path = tmp_path / "summary.csv"
    output_path = tmp_path / "gain.png"
    write_rows(input_path, [{"threshold": 0.05, "ci95_low": 0.1, "ci95_high": 0.2}])

    with pytest.raises(ValueError, match="no rows for threshold 0.1"):
        plot_budget_gain(input_path, output_path)

    assert not output_path.exists()
