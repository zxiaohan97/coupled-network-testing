import math

import numpy as np

from experiments.tree.random_tree_budget_gain import (
    finite_budget_gain,
    run_budget_gain_experiment,
    summarize_trials,
)


def test_gain_keeps_losses_and_excludes_undefined_ratios():
    assert finite_budget_gain(3, 2) == -0.5
    assert finite_budget_gain(0, 0) is None
    assert finite_budget_gain(2, float("inf")) is None
    assert finite_budget_gain(float("inf"), 2) is None


def test_summary_keeps_unreached_and_initially_satisfied_trees_visible():
    settings = {
        "n_nodes": 8,
        "p": 0.9,
        "q": 0.1,
        "r": 0.5,
        "physical_error": 0.2,
        "social_error": 0.05,
        "threshold": 0.1,
        "max_budget": 8,
        "random_seed": 7,
    }
    trials = [
        {**settings, "greedy_budget": g, "contact_budget": c, "budget_gain": gain}
        for g, c, gain in (
            (2, 4, 0.5),
            (6, 5, -0.2),
            (3, float("inf"), float("nan")),
            (float("inf"), 4, float("nan")),
            (float("inf"), float("inf"), float("nan")),
            (0, 0, float("nan")),
        )
    ]

    summary = summarize_trials(trials)[0]

    assert summary["n_trees"] == 6
    assert summary["num_reached"] == 2
    assert summary["num_both_reached"] == 3
    assert summary["num_initially_satisfied"] == 1
    assert summary["num_greedy_reached"] == 4
    assert summary["num_contact_reached"] == 4
    assert summary["num_neither_reached"] == 1
    assert math.isclose(summary["mean_budget_gain"], 0.15)
    assert summary["ci95_low"] < 0.15 < summary["ci95_high"]
    assert summary["mean_greedy_budget"] == 4
    assert summary["mean_contact_budget"] == 4.5


def test_sweep_matches_between_serial_and_parallel_execution(tmp_path):
    settings = dict(
        n_nodes=4,
        n_trees=3,
        p=0.9,
        q=0.1,
        r_values=[0, 0.5],
        physical_error=0.2,
        social_errors=[0.05],
        thresholds=[0.1, 0.05],
        max_budget=2,
        random_seed=7,
    )
    serial = run_budget_gain_experiment(
        **settings,
        workers=1,
        details_output=tmp_path / "serial.csv",
    )
    parallel = run_budget_gain_experiment(
        **settings,
        workers=2,
        details_output=tmp_path / "parallel.csv",
    )

    assert len(serial) == len(parallel) == 4
    for first, second in zip(serial, parallel, strict=True):
        assert first.keys() == second.keys()
        np.testing.assert_equal(list(first.values()), list(second.values()))
    assert (tmp_path / "serial.csv").read_bytes() == (tmp_path / "parallel.csv").read_bytes()


def test_default_sweep_uses_only_threshold_point_one():
    rows = run_budget_gain_experiment(
        n_nodes=4,
        n_trees=2,
        p=0.9,
        q=0.1,
        r_values=[0, 1],
        physical_error=0.2,
        social_errors=[0.05],
        max_budget=2,
        random_seed=7,
    )

    assert len(rows) == 2
    assert {row["threshold"] for row in rows} == {0.1}
