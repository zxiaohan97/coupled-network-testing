import math

from coupled_network_testing.benchmarks.complete_dp import (
    CompleteGraphBenchmarkConfig,
    CompleteGraphDecision,
    CompleteGraphSymmetryBenchmark,
    _state_to_test_sequence,
    run_complete_graph_benchmark,
)
from coupled_network_testing.general.simulation_model import GeneralModelParameters


def test_complete_state_uses_canonical_fresh_nodes():
    sequence = _state_to_test_sequence((1, 1, 1, 1))

    assert sequence == (
        (1, 1, 1),
        (2, 1, 0),
        (3, 0, 1),
        (4, 0, 0),
    )


def test_complete_graph_optimal_is_no_worse_than_greedy():
    config = CompleteGraphBenchmarkConfig(
        n_nodes=7,
        budget=3,
        n_particles=2_000,
        parameters=GeneralModelParameters(
            p_infect_careless=0.35,
            q_infect_careful=0.05,
            social_correlation=0.2,
            physical_error=0.1,
            social_error=0.05,
        ),
    )

    result = run_complete_graph_benchmark(config)

    assert result.optimal_uncertainty <= result.greedy_uncertainty + 1e-12
    assert result.initial_greedy_decision in {
        CompleteGraphDecision.PHYSICAL,
        CompleteGraphDecision.SOCIAL,
    }
    assert result.initial_optimal_decision in {
        CompleteGraphDecision.PHYSICAL,
        CompleteGraphDecision.SOCIAL,
    }
    assert 0.0 <= result.relative_gap <= 1.0


def test_complete_graph_symmetry_benchmark_is_reproducible():
    config = CompleteGraphBenchmarkConfig(n_nodes=6, budget=2, n_particles=1_000)

    first = CompleteGraphSymmetryBenchmark(config).evaluate()
    second = CompleteGraphSymmetryBenchmark(config).evaluate()

    assert math.isclose(first.greedy_uncertainty, second.greedy_uncertainty)
    assert math.isclose(first.optimal_uncertainty, second.optimal_uncertainty)
