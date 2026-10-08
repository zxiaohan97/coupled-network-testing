import math

from coupled_network_testing.benchmarks.star_dp import (
    StarBenchmarkConfig,
    StarDecision,
    StarTreeSymmetryBenchmark,
    _state_to_test_sequence,
    run_star_tree_benchmark,
)


def test_star_state_uses_center_then_fresh_side_nodes():
    sequence = _state_to_test_sequence((1, 1, 1, 1, 1, 1, 1, 1))

    assert sequence == (
        (1, 1, 1),
        (1, 1, 0),
        (1, 0, 1),
        (1, 0, 0),
        (2, 1, 1),
        (3, 1, 0),
        (4, 0, 1),
        (5, 0, 0),
    )


def test_star_tree_optimal_is_no_worse_than_greedy():
    config = StarBenchmarkConfig(n_nodes=7, budget=3)

    result = run_star_tree_benchmark(config)

    assert result.optimal_uncertainty <= result.greedy_uncertainty + 1e-12
    assert result.initial_greedy_decision in set(StarDecision)
    assert result.initial_optimal_decision in set(StarDecision)
    assert 0.0 <= result.relative_gap <= 1.0


def test_star_tree_symmetry_benchmark_is_reproducible():
    config = StarBenchmarkConfig(n_nodes=6, budget=2)

    first = StarTreeSymmetryBenchmark(config).evaluate()
    second = StarTreeSymmetryBenchmark(config).evaluate()

    assert math.isclose(first.greedy_uncertainty, second.greedy_uncertainty)
    assert math.isclose(first.optimal_uncertainty, second.optimal_uncertainty)
