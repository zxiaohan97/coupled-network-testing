import math

from coupled_network_testing.tree.exact_model import ModelParameters
from coupled_network_testing.tree.policies import (
    analyze_threshold_times,
    compare_strategies,
    run_contact_tracing,
    run_greedy_testing,
    select_next_test_greedy,
)
from coupled_network_testing.tree.posterior_updates import ExactTreePosteriorModel
from coupled_network_testing.tree.validation import DiseaseSpreadSimulation


def make_small_model():
    return ExactTreePosteriorModel.from_edges(
        edges=[(0, 1), (0, 2), (1, 3)],
        seed=0,
        parameters=ModelParameters(
            p_infect_careless=0.8,
            q_infect_careful=0.2,
            social_correlation=0.5,
            physical_error=0.01,
            social_error=0.05,
        ),
    )


def test_exact_posterior_update_sequence_remains_normalized():
    model = make_small_model()

    probabilities = model.update_with_test_sequence(((2, 1, 0), (1, 0, 1)))

    assert len(probabilities) == 3
    for node_table in probabilities[-1].values():
        assert math.isclose(sum(node_table.values()), 1.0)


def test_greedy_policy_selects_a_physical_or_social_test():
    model = make_small_model()

    best_test, expected_uncertainty = select_next_test_greedy(model)

    assert best_test is not None
    assert best_test[0] in model.G.nodes
    assert best_test[1] in (0, 1)
    assert expected_uncertainty >= 0


def test_tree_policy_runners_return_uncertainty_sequences():
    model = make_small_model()

    _, greedy_uncertainties = run_greedy_testing(model, num_tests=2)
    _, contact_uncertainties = run_contact_tracing(model, num_tests=2)
    comparison = compare_strategies(model, num_tests=1)

    assert len(greedy_uncertainties) == 3
    assert len(contact_uncertainties) == 3
    assert set(comparison) == {"greedy", "contact_tracing"}


def test_validation_simulation_returns_empirical_probabilities():
    sim = DiseaseSpreadSimulation(
        edges=[(0, 1), (0, 2), (1, 3)],
        seed_node=0,
        p=0.8,
        q=0.2,
        r=0.5,
        ep=0.01,
        es=0.05,
        n_simulations=200,
        random_seed=7,
    )

    probabilities = sim.run_simulation_sequence(((2, 1, 0),))

    assert probabilities[0] is not None
    assert set(probabilities[0]) == {0, 1, 2, 3}


def test_threshold_times_match_full_expected_uncertainty_curves():
    model = make_small_model()
    curves = compare_strategies(model, num_tests=3)
    targets = [0.25, 0.1, 0.05, 0.00001]
    times = analyze_threshold_times(model, targets, max_budget=3)

    for strategy, result in curves.items():
        for target in targets:
            expected = next(
                (index for index, value in enumerate(result["uncertainties"]) if value <= target),
                float("inf"),
            )
            assert times[strategy][target] == expected


def test_threshold_times_include_zero_budget_and_restore_current_posterior():
    model = make_small_model()
    prior = model.get_physical_uncertainty()
    model.update_with_test((2, 1, 0))
    before = model.snapshot()

    times = analyze_threshold_times(model, [prior, prior / 2], max_budget=0)

    for strategy in ("greedy", "contact_tracing"):
        assert times[strategy][prior] == 0
        assert math.isinf(times[strategy][prior / 2])
    assert model.snapshot() == before
