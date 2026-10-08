"""Exact posterior updates for tree networks.

This module is adapted from ``Tree/Tree_Testing_Model.py`` in the original
research workspace. It intentionally keeps the original posterior-update
structure close to the source code while giving it a package-friendly API.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import TypeAlias

import networkx as nx

from coupled_network_testing.tree.exact_model import (
    ModelParameters,
    NodeState,
    TreeOpinionDiseaseModel,
    TreeStructure,
)

JointProbabilities: TypeAlias = defaultdict[int, defaultdict[NodeState, float]]
ConditionalProbabilities: TypeAlias = defaultdict[
    int,
    defaultdict[int, defaultdict[NodeState, defaultdict[NodeState, float]]],
]


def _joint_container() -> JointProbabilities:
    return defaultdict(lambda: defaultdict(float))


def _conditional_container() -> ConditionalProbabilities:
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))


class ExactTreePosteriorModel:
    """Exact tree posterior model with sequential test updates.

    Test records use the original convention ``(node, test_type, result)``,
    where ``test_type == 1`` means a physical disease test and ``test_type == 0``
    means a social opinion test.
    """

    def __init__(self, structure: TreeStructure, parameters: ModelParameters):
        self.structure = structure
        self.parameters = parameters
        self.G = structure.graph
        self.seed = structure.seed
        self.p = parameters.p_infect_careless
        self.q = parameters.q_infect_careful
        self.r = parameters.social_correlation
        self.ep = parameters.physical_error
        self.es = parameters.social_error

        if not nx.is_tree(self.G):
            raise ValueError("Input graph must be a tree")

        self.parent_child = list(structure.bfs_edges)
        self.parent_of = dict(structure.parent_of)
        self.child_of = defaultdict(set)
        for parent, child in self.parent_child:
            self.child_of[parent].add(child)

        self.probs: JointProbabilities = _joint_container()
        self.conditional_probs: ConditionalProbabilities = _conditional_container()
        self.joint_dict: dict[tuple[tuple[int, int, int], ...], dict] = {}
        self.conditional_dict: dict[tuple[tuple[int, int, int], ...], dict] = {}

        self.calculate_joint_probabilities()
        self.store_current_state(())

    @classmethod
    def from_edges(
        cls,
        edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        seed: int,
        parameters: ModelParameters,
    ) -> ExactTreePosteriorModel:
        """Build the exact posterior model directly from tree edges."""

        return cls(TreeStructure.from_edges(edges, seed=seed), parameters)

    def calculate_joint_probabilities(self) -> None:
        """Calculate initial joint and conditional probabilities."""

        initial_model = TreeOpinionDiseaseModel(self.structure, self.parameters)
        self.probs = _joint_container()
        self.conditional_probs = _conditional_container()

        for node, table in initial_model.joint_probabilities.items():
            self.probs[node] = defaultdict(float, table)

        for (parent, child), parent_table in initial_model.conditional_probabilities.items():
            for parent_state, child_table in parent_table.items():
                self.conditional_probs[child][parent][parent_state] = defaultdict(
                    float,
                    child_table,
                )

    def _get_path_to_seed(self, node: int) -> list[int]:
        """Get path from node to seed, starting at ``node``."""

        path = [node]
        current = node
        while current != self.seed:
            current = self.parent_of[current]
            path.append(current)
        return path

    def _update_child_from_parent(self, parent: int, child: int) -> None:
        """Update child probabilities using the current parent posterior."""

        new_child_probs = defaultdict(float)
        for parent_state, parent_prob in self.probs[parent].items():
            for child_state, cond_prob in self.conditional_probs[child][parent][
                parent_state
            ].items():
                # Forward propagation after an upstream posterior changed.
                new_child_probs[child_state] += parent_prob * cond_prob
        self.probs[child] = new_child_probs

    def update_with_test(self, test_info: tuple[int, int, int] | list[int]) -> None:
        """Update posterior probabilities after one test result."""

        node, test_type, result = test_info
        original_probs = {n: dict(probs) for n, probs in self.probs.items()}

        new_probs = defaultdict(float)
        total_prob = 0.0
        for (physical_state, social_state), prior_prob in self.probs[node].items():
            # First update only the tested node by Bayes' rule:
            # posterior(state) proportional to likelihood(test | state) * prior(state).
            if test_type == 1:
                likelihood = (1 - self.ep) if physical_state == result else self.ep
            else:
                likelihood = (1 - self.es) if social_state == result else self.es
            new_prob = likelihood * prior_prob
            new_probs[(physical_state, social_state)] = new_prob
            total_prob += new_prob

        if total_prob == 0:
            raise ValueError("test has zero probability under the current posterior")

        for state in new_probs:
            new_probs[state] /= total_prob
        self.probs[node] = new_probs

        path_to_seed = self._get_path_to_seed(node)
        for i in range(1, len(path_to_seed)):
            child = path_to_seed[i - 1]
            parent = path_to_seed[i]
            new_parent_probs = defaultdict(float)

            # The tested node can change beliefs about its ancestors. On a tree,
            # ancestors are exactly the nodes on the unique path back to the seed.
            for parent_state in original_probs[parent]:
                prob = 0.0
                for child_state, child_posterior in self.probs[child].items():
                    if original_probs[child][child_state] > 0:
                        # Reverse the stored edge conditional using the
                        # pre-test marginals: P(parent | child).
                        reverse_cond = (
                            self.conditional_probs[child][parent][parent_state][child_state]
                            * original_probs[parent][parent_state]
                            / original_probs[child][child_state]
                        )
                        prob += reverse_cond * child_posterior
                new_parent_probs[parent_state] = prob

            total = sum(new_parent_probs.values())
            if total > 0:
                for state in new_parent_probs:
                    new_parent_probs[state] /= total
            self.probs[parent] = new_parent_probs

            for parent_state in self.probs[parent]:
                for child_state in self.probs[child]:
                    if (
                        self.probs[parent][parent_state] > 0
                        and original_probs[child][child_state] > 0
                    ):
                        # Refresh P(child | parent, observations) so later
                        # descendant propagation uses conditionals consistent
                        # with the updated marginals.
                        reverse_cond = (
                            self.conditional_probs[child][parent][parent_state][child_state]
                            * original_probs[parent][parent_state]
                            / original_probs[child][child_state]
                        )
                        self.conditional_probs[child][parent][parent_state][child_state] = (
                            reverse_cond
                            * self.probs[child][child_state]
                            / self.probs[parent][parent_state]
                        )

        updated_nodes = set(path_to_seed)
        queue = []
        # Every off-path subtree hangs from a node whose posterior may have
        # changed. Push those changes outward with the current edge conditionals.
        for path_node in path_to_seed:
            for neighbor in self.G.neighbors(path_node):
                if neighbor not in updated_nodes:
                    queue.append(neighbor)
                    updated_nodes.add(neighbor)
                    self._update_child_from_parent(self.parent_of[neighbor], neighbor)

        while queue:
            current = queue.pop(0)
            for neighbor in self.G.neighbors(current):
                if neighbor not in updated_nodes:
                    queue.append(neighbor)
                    updated_nodes.add(neighbor)
                    self._update_child_from_parent(self.parent_of[neighbor], neighbor)

    def update_with_test_sequence(
        self,
        test_sequence: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
    ) -> list[dict]:
        """Update probabilities with a sequence of test results."""

        sequence_probs = [deepcopy(dict(self.probs))]
        for test in test_sequence:
            self.update_with_test(test)
            sequence_probs.append(deepcopy(dict(self.probs)))
        return sequence_probs

    def store_current_state(self, test_sequence: tuple[tuple[int, int, int], ...]) -> None:
        """Store current joint and conditional probabilities for a test sequence."""

        self.joint_dict[test_sequence] = deepcopy(dict(self.probs))
        self.conditional_dict[test_sequence] = deepcopy(dict(self.conditional_probs))

    def load_state(self, test_sequence: tuple[tuple[int, int, int], ...]) -> None:
        """Load joint and conditional probabilities for a test sequence."""

        self.probs = _joint_container()
        self.conditional_probs = _conditional_container()

        for node, probs in self.joint_dict[test_sequence].items():
            self.probs[node] = defaultdict(float, probs)

        for node, parent_dict in self.conditional_dict[test_sequence].items():
            for parent, state_dict in parent_dict.items():
                for parent_state, child_dict in state_dict.items():
                    self.conditional_probs[node][parent][parent_state] = defaultdict(
                        float,
                        child_dict,
                    )

    def snapshot(self) -> tuple[dict, dict]:
        """Return a deep copy of the current posterior state."""

        return deepcopy(dict(self.probs)), deepcopy(dict(self.conditional_probs))

    def restore_snapshot(self, snapshot: tuple[dict, dict]) -> None:
        """Restore a posterior state previously returned by ``snapshot``."""

        joint, conditional = snapshot
        self.probs = _joint_container()
        self.conditional_probs = _conditional_container()

        for node, probs in joint.items():
            self.probs[node] = defaultdict(float, probs)

        for node, parent_dict in conditional.items():
            for parent, state_dict in parent_dict.items():
                for parent_state, child_dict in state_dict.items():
                    self.conditional_probs[node][parent][parent_state] = defaultdict(
                        float,
                        child_dict,
                    )

    def reset_to_initial_state(self) -> None:
        """Restore the initial no-observation posterior state."""

        self.load_state(())

    def get_physical_uncertainty(self) -> float:
        """Calculate average posterior disease uncertainty across all nodes."""

        uncertainties = []
        for node in self.G.nodes():
            # The objective is disease-state uncertainty only, so marginalize
            # over social state before computing p(1-p).
            p_infected = sum(
                prob
                for (physical_state, _), prob in self.probs[node].items()
                if physical_state == 1
            )
            uncertainties.append(p_infected * (1 - p_infected))
        return sum(uncertainties) / len(uncertainties)

    def get_test_outcome_probability(
        self,
        node: int,
        test_type: int,
        test_result: int,
    ) -> float:
        """Calculate the probability of one physical or social test outcome."""

        prob = 0.0
        for (physical_state, social_state), state_prob in self.probs[node].items():
            if test_type == 1:
                likelihood = (1 - self.ep) if physical_state == test_result else self.ep
            else:
                likelihood = (1 - self.es) if social_state == test_result else self.es
            prob += likelihood * state_prob
        return prob
