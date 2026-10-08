import math

import networkx as nx
import numpy as np

from coupled_network_testing.common.testing_types import TestType as ObservationType
from coupled_network_testing.general.experiments import (
    GeneralExperimentConfig,
    GeneralGraphModel,
    PolicyStrategy,
    TopologyComparisonConfig,
    run_policy_comparison,
    run_topology_comparison,
)
from coupled_network_testing.general.graph_generators import (
    SeedSelection,
    generate_er_graph,
    select_seed_nodes,
    split_overlapping_subgraphs,
)
from coupled_network_testing.general.noncooperation import (
    NonCooperationMode,
    NonCooperationParameters,
    NonCooperativeStrategy,
    noncooperative_steps_to_dataframe,
    refusal_probability,
    run_noncooperative_policy_comparison,
    sample_cooperation_status,
)
from coupled_network_testing.general.old_rejection_filter import (
    OldRejectionMonteCarloPosterior,
    expected_uncertainty_after_test,
)
from coupled_network_testing.general.policies import (
    select_contact_tracing_test,
    select_weighted_greedy_test,
)
from coupled_network_testing.general.simulation_model import (
    GeneralCascadeSimulator,
    GeneralModelParameters,
)
from coupled_network_testing.general.weighted_particle_filter import (
    WeightedParticlePosterior,
    WeightedPosteriorEstimate,
    observation_likelihood,
)
from coupled_network_testing.tree.exact_model import ModelParameters
from coupled_network_testing.tree.posterior_updates import ExactTreePosteriorModel


def test_graph_generators_preserve_nodes_and_seed_modes():
    rng = np.random.default_rng(7)
    graph = generate_er_graph(n_nodes=10, edge_probability=0.4, rng=rng)
    pair = split_overlapping_subgraphs(graph, overlap=0.5, rng=rng)

    assert set(pair.physical.nodes()) == set(graph.nodes())
    assert set(pair.social.nodes()) == set(graph.nodes())

    top_seed = select_seed_nodes(
        graph,
        n_seeds=1,
        rng=rng,
        mode=SeedSelection.HIGHEST_DEGREE,
    )[0]
    max_degree = max(dict(graph.degree()).values())
    assert graph.degree(top_seed) == max_degree


def test_simulator_samples_valid_states_and_error_free_tests():
    rng = np.random.default_rng(11)
    graph = nx.path_graph(4)
    params = GeneralModelParameters(
        p_infect_careless=0.8,
        q_infect_careful=0.2,
        social_correlation=0.5,
        physical_error=0.0,
        social_error=0.0,
    )
    simulator = GeneralCascadeSimulator(graph, graph, seeds=(0,), parameters=params, rng=rng)

    realization = simulator.sample_realization()

    assert realization.physical_states[0] == 1
    assert set(realization.physical_states.values()) <= {0, 1}
    assert set(realization.social_states.values()) <= {0, 1}
    for node in graph.nodes:
        assert realization.physical_tests[node] == realization.physical_states[node]
        assert realization.social_tests[node] == realization.social_states[node]


def test_monte_carlo_posteriors_match_exact_tree_update_on_tree_graph():
    rng = np.random.default_rng(23)
    edges = [(0, 1), (1, 2), (1, 3)]
    graph = nx.Graph(edges)
    general_params = GeneralModelParameters(
        p_infect_careless=0.8,
        q_infect_careful=0.2,
        social_correlation=0.5,
        physical_error=0.05,
        social_error=0.1,
    )
    simulator = GeneralCascadeSimulator(
        graph,
        graph,
        seeds=(0,),
        parameters=general_params,
        rng=rng,
    )
    realizations = simulator.sample_realizations(30_000)
    posterior = OldRejectionMonteCarloPosterior(realizations, nodes=tuple(graph.nodes()))
    test_sequence = ((2, int(ObservationType.PHYSICAL), 0), (1, int(ObservationType.SOCIAL), 1))

    exact = ExactTreePosteriorModel.from_edges(
        edges=edges,
        seed=0,
        parameters=ModelParameters(
            p_infect_careless=0.8,
            q_infect_careful=0.2,
            social_correlation=0.5,
            physical_error=0.05,
            social_error=0.1,
        ),
    )
    exact_probs = exact.update_with_test_sequence(test_sequence)[-1]
    monte_carlo_probs = posterior.estimate(test_sequence)

    assert monte_carlo_probs.n_samples > 1_000
    for node in graph.nodes:
        exact_infection_probability = exact_probs[node][(1, 0)] + exact_probs[node][(1, 1)]
        assert math.isclose(
            monte_carlo_probs.physical_posteriors[node],
            exact_infection_probability,
            abs_tol=0.035,
        )


def test_expected_uncertainty_after_candidate_test_is_finite():
    rng = np.random.default_rng(17)
    graph = nx.path_graph(5)
    params = GeneralModelParameters(0.8, 0.2, 0.4, physical_error=0.1, social_error=0.1)
    simulator = GeneralCascadeSimulator(graph, graph, seeds=(0,), parameters=params, rng=rng)
    realizations = simulator.sample_realizations(500)

    expected_uncertainty = expected_uncertainty_after_test(
        realizations,
        node=2,
        test_type=ObservationType.PHYSICAL,
    )

    assert 0.0 <= expected_uncertainty <= 0.25


def test_weighted_particle_posterior_matches_exact_tree_after_long_sequence():
    rng = np.random.default_rng(920)
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (5, 6), (5, 7)]
    graph = nx.Graph(edges)
    params = GeneralModelParameters(
        p_infect_careless=0.8,
        q_infect_careful=0.2,
        social_correlation=0.5,
        physical_error=0.05,
        social_error=0.1,
    )
    simulator = GeneralCascadeSimulator(graph, graph, seeds=(0,), parameters=params, rng=rng)
    particles = simulator.sample_realizations(5_000)
    posterior = WeightedParticlePosterior(particles, parameters=params, nodes=tuple(graph.nodes()))

    # A 15-observation history exercises the failure mode where hard rejection
    # would leave very few matching samples. The weighted estimator conditions
    # on likelihoods of the observed results instead of exact noisy-test draws.
    test_sequence = (
        (2, int(ObservationType.PHYSICAL), 0),
        (1, int(ObservationType.SOCIAL), 0),
        (4, int(ObservationType.PHYSICAL), 0),
        (6, int(ObservationType.PHYSICAL), 0),
        (7, int(ObservationType.SOCIAL), 0),
        (3, int(ObservationType.PHYSICAL), 0),
        (5, int(ObservationType.SOCIAL), 0),
        (2, int(ObservationType.SOCIAL), 0),
        (4, int(ObservationType.SOCIAL), 0),
        (6, int(ObservationType.SOCIAL), 0),
        (7, int(ObservationType.PHYSICAL), 1),
        (1, int(ObservationType.PHYSICAL), 0),
        (3, int(ObservationType.SOCIAL), 0),
        (5, int(ObservationType.PHYSICAL), 0),
        (0, int(ObservationType.SOCIAL), 0),
    )

    exact = ExactTreePosteriorModel.from_edges(
        edges=edges,
        seed=0,
        parameters=ModelParameters(
            p_infect_careless=0.8,
            q_infect_careful=0.2,
            social_correlation=0.5,
            physical_error=0.05,
            social_error=0.1,
        ),
    )
    exact_probs = exact.update_with_test_sequence(test_sequence)[-1]
    estimate = posterior.estimate(test_sequence)

    assert estimate.effective_sample_size > 300
    for node in graph.nodes:
        exact_infection_probability = exact_probs[node][(1, 0)] + exact_probs[node][(1, 1)]
        assert math.isclose(
            estimate.physical_posteriors[node],
            exact_infection_probability,
            abs_tol=0.025,
        )


def test_weighted_particle_likelihood_uses_hidden_state_not_presampled_test():
    rng = np.random.default_rng(31)
    graph = nx.path_graph(3)
    params = GeneralModelParameters(0.8, 0.2, 0.5, physical_error=0.05, social_error=0.1)
    simulator = GeneralCascadeSimulator(graph, graph, seeds=(0,), parameters=params, rng=rng)
    realization = simulator.sample_realization()
    node = 0

    positive_likelihood = observation_likelihood(
        realization,
        (node, int(ObservationType.PHYSICAL), 1),
        params,
    )

    assert realization.physical_states[node] == 1
    assert positive_likelihood == 1.0 - params.physical_error


def test_weighted_greedy_policy_returns_diagnostics_for_valid_candidate():
    rng = np.random.default_rng(41)
    graph = nx.path_graph(5)
    params = GeneralModelParameters(0.8, 0.2, 0.5, physical_error=0.05, social_error=0.1)
    simulator = GeneralCascadeSimulator(graph, graph, seeds=(0,), parameters=params, rng=rng)
    posterior = WeightedParticlePosterior(
        simulator.sample_realizations(800),
        parameters=params,
        nodes=tuple(graph.nodes()),
    )

    decision = select_weighted_greedy_test(
        posterior,
        test_sequence=((2, int(ObservationType.PHYSICAL), 0),),
        seeds=(0,),
        min_branch_ess=10,
    )

    assert decision.node in graph.nodes
    assert decision.test_type in {ObservationType.SOCIAL, ObservationType.PHYSICAL}
    assert decision.minimum_branch_ess is not None
    assert decision.minimum_branch_ess > 0
    assert 0.0 <= decision.expected_uncertainty <= 0.25


def test_contact_tracing_policy_uses_observed_positive_frontier():
    graph = nx.path_graph(4)
    estimate = WeightedPosteriorEstimate(
        physical_posteriors={0: 1.0, 1: 0.9, 2: 0.5, 3: 0.2},
        social_posteriors={0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
        physical_uncertainty=0.1,
        n_particles=100,
        effective_sample_size=100.0,
        max_weight=0.01,
    )

    decision = select_contact_tracing_test(
        graph,
        estimate,
        test_sequence=((1, int(ObservationType.PHYSICAL), 1),),
        seeds=(0,),
    )

    assert decision.node == 2
    assert decision.test_type == ObservationType.PHYSICAL


def test_general_policy_comparison_runs_both_strategies():
    config = GeneralExperimentConfig(
        n_nodes=8,
        edge_probability=0.35,
        graph_model=GeneralGraphModel.ERDOS_RENYI,
        n_particles=600,
        max_tests=3,
        ess_warning_threshold=20,
        min_branch_ess=20,
    )

    result = run_policy_comparison(config, rng=np.random.default_rng(51))
    table = result.to_dataframe()

    assert set(table["strategy"]) == {
        PolicyStrategy.WEIGHTED_GREEDY.value,
        PolicyStrategy.CONTACT_TRACING.value,
    }
    assert len(table) == 2 * config.max_tests
    assert table["posterior_uncertainty"].between(0.0, 0.25).all()
    assert (table["effective_sample_size"] > 0).all()


def test_topology_comparison_runs_er_scale_free_and_watts_strogatz():
    config = TopologyComparisonConfig(
        base_config=GeneralExperimentConfig(
            n_nodes=8,
            edge_probability=0.2,
            n_particles=300,
            max_tests=2,
            ess_warning_threshold=5,
            min_branch_ess=5,
            parameters=GeneralModelParameters(
                p_infect_careless=0.6,
                q_infect_careful=0.2,
                social_correlation=0.75,
                physical_error=0.15,
                social_error=0.1,
            ),
        ),
        graph_models=(
            GeneralGraphModel.ERDOS_RENYI,
            GeneralGraphModel.SCALE_FREE,
            GeneralGraphModel.WATTS_STROGATZ,
        ),
        edge_probabilities=(0.2,),
        n_replicates=1,
        random_seed=70,
    )

    result = run_topology_comparison(config)

    assert set(result.summary["graph_model"]) == {"er", "scale_free", "watts_strogatz"}
    assert set(result.summary["strategy"]) == {
        PolicyStrategy.WEIGHTED_GREEDY.value,
        PolicyStrategy.CONTACT_TRACING.value,
    }
    assert len(result.summary) == 6
    assert result.summary["final_uncertainty"].between(0.0, 0.25).all()


def test_noncooperation_status_mode_one_only_selects_careless_nodes():
    rng = np.random.default_rng(81)
    social_states = {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0}
    params = NonCooperationParameters(
        cooperation_fraction=0.5,
        mode=NonCooperationMode.CARELESS_PHYSICAL,
    )

    status = sample_cooperation_status(social_states, params, rng)
    noncooperative = {node for node, cooperates in status.items() if not cooperates}

    assert len(noncooperative) == 2
    assert noncooperative <= {0, 1, 2, 3}
    assert status[4]
    assert status[5]


def test_noncooperation_refusal_probability_uses_social_posterior():
    estimate = WeightedPosteriorEstimate(
        physical_posteriors={2: 0.5},
        social_posteriors={2: 0.8},
        physical_uncertainty=0.25,
        n_particles=100,
        effective_sample_size=100.0,
        max_weight=0.01,
    )
    params = NonCooperationParameters(
        cooperation_fraction=0.6,
        mode=NonCooperationMode.CARELESS_PHYSICAL,
    )
    social_params = NonCooperationParameters(
        cooperation_fraction=0.6,
        mode=NonCooperationMode.CARELESS_PHYSICAL_AND_SOCIAL,
    )

    assert math.isclose(
        refusal_probability(estimate, 2, ObservationType.PHYSICAL, params),
        0.32,
    )
    assert refusal_probability(estimate, 2, ObservationType.SOCIAL, params) == 0.0
    assert math.isclose(
        refusal_probability(estimate, 2, ObservationType.SOCIAL, social_params),
        0.32,
    )


def test_noncooperative_policy_comparison_runs_with_refusals():
    rng = np.random.default_rng(91)
    graph = nx.path_graph(6)
    model_params = GeneralModelParameters(
        p_infect_careless=0.8,
        q_infect_careful=0.2,
        social_correlation=0.5,
        physical_error=0.05,
        social_error=0.1,
    )
    simulator = GeneralCascadeSimulator(graph, graph, seeds=(0,), parameters=model_params, rng=rng)
    ground_truth = simulator.sample_realization()
    particles = simulator.sample_realizations(500)
    noncoop_params = NonCooperationParameters(
        cooperation_fraction=0.0,
        mode=NonCooperationMode.CARELESS_PHYSICAL,
    )
    cooperative_status = {node: False for node in graph.nodes()}

    steps = run_noncooperative_policy_comparison(
        graph,
        particles,
        ground_truth,
        model_params,
        noncoop_params,
        seeds=(0,),
        max_tests=3,
        rng=rng,
        cooperative_status=cooperative_status,
        min_branch_ess=5,
        ess_warning_threshold=5,
    )
    table = noncooperative_steps_to_dataframe(steps)

    assert set(table["strategy"]) == {
        NonCooperativeStrategy.WEIGHTED_GREEDY.value,
        NonCooperativeStrategy.CONTACT_TRACING.value,
    }
    assert table["posterior_uncertainty"].between(0.0, 0.25).all()
    assert table["effective_sample_size"].gt(0).all()
    assert set(table["result"]) <= {-1, 0, 1}
    assert table["refused"].any()
