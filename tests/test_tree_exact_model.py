import math

import pytest

from coupled_network_testing.tree.exact_model import (
    ModelParameters,
    TreeOpinionDiseaseModel,
    TreeStructure,
)


def test_path_tree_initial_probabilities_are_normalized():
    params = ModelParameters(
        p_infect_careless=0.8,
        q_infect_careful=0.2,
        social_correlation=0.6,
        physical_error=0.05,
        social_error=0.1,
    )
    model = TreeOpinionDiseaseModel(TreeStructure.path(4, seed=0), params)

    for table in model.joint_probabilities.values():
        assert math.isclose(sum(table.values()), 1.0)


def test_seed_is_always_infected_and_social_state_is_balanced():
    params = ModelParameters(0.8, 0.2, 0.6)
    model = TreeOpinionDiseaseModel(TreeStructure.star(5, seed=0), params)

    assert model.infection_probability(0) == 1.0
    assert model.social_probability(0) == 0.5


def test_rejects_non_tree_edges():
    params = ModelParameters(0.8, 0.2, 0.6)

    with pytest.raises(ValueError, match="connected acyclic tree"):
        TreeOpinionDiseaseModel.from_edges(
            edges=[(0, 1), (1, 2), (2, 0)],
            seed=0,
            parameters=params,
        )


def test_branching_factor_generator_matches_expected_size():
    structure = TreeStructure.from_branching_factors([2, 3], seed=0)

    assert structure.graph.number_of_nodes() == 9
    assert structure.children_of[0] == (1, 2)
