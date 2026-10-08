"""Validation helpers comparing exact tree probabilities with simulation."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np


class DiseaseSpreadSimulation:
    """Monte Carlo simulator for validating exact tree posterior updates.

    This is adapted from ``Tree/Tree_Posterior_Validation.py``. It keeps the
    original state convention ``(physical_state, social_state)`` and test-type
    convention where ``1`` is physical and ``0`` is social.
    """

    def __init__(
        self,
        edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        seed_node: int,
        p: float,
        q: float,
        r: float,
        ep: float,
        es: float,
        n_simulations: int = 1_000_000,
        random_seed: int | None = None,
    ):
        self.G = nx.Graph(edges)
        self.seed = seed_node
        self.p = p
        self.q = q
        self.r = r
        self.ep = ep
        self.es = es
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)

        self.parent_child = self._get_parent_child_pairs()
        self.states = defaultdict(list)
        self.test_results = defaultdict(list)

    def _get_parent_child_pairs(self) -> list[tuple[int, int]]:
        pairs = []
        visited = {self.seed}
        queue = [self.seed]
        while queue:
            parent = queue.pop(0)
            for neighbor in self.G.neighbors(parent):
                if neighbor not in visited:
                    pairs.append((parent, neighbor))
                    visited.add(neighbor)
                    queue.append(neighbor)
        return pairs

    def simulate_one_instance(self) -> dict[int, tuple[int, int]]:
        """Simulate one opinion-disease cascade instance."""

        states = {}
        states[self.seed] = (1, int(self.rng.integers(2)))

        for parent, child in self.parent_child:
            edge_exists = self.rng.random() < self.r
            parent_physical, parent_social = states[parent]

            # Simulation mirrors the exact model: social correlation is sampled
            # first, then infection risk is based on the child's social label.
            child_social = parent_social if edge_exists else int(self.rng.integers(2))

            if parent_physical == 0:
                child_physical = 0
            else:
                infection_prob = self.p if child_social == 1 else self.q
                child_physical = 1 if self.rng.random() < infection_prob else 0

            states[child] = (child_physical, child_social)

        return states

    def simulate_test(self, states: dict[int, tuple[int, int]], node: int, test_type: int) -> int:
        """Simulate one noisy physical or social test result."""

        physical_state, social_state = states[node]
        true_state = physical_state if test_type == 1 else social_state
        error_rate = self.ep if test_type == 1 else self.es

        if self.rng.random() < error_rate:
            return 1 - true_state
        return true_state

    def run_simulations(self, test_info: tuple[int, int, int] | None = None) -> None:
        """Run simulations, optionally conditioning on one observed test."""

        self.states.clear()
        self.test_results.clear()

        for _ in range(self.n_simulations):
            states = self.simulate_one_instance()
            if test_info:
                node, test_type, expected_result = test_info
                test_result = self.simulate_test(states, node, test_type)
                if test_result != expected_result:
                    continue

            for node, state in states.items():
                self.states[node].append(state)

    def get_probabilities(self) -> dict[int, dict[tuple[int, int], float]]:
        """Calculate empirical state probabilities from stored simulations."""

        probs = {}
        for node in self.states:
            node_probs = defaultdict(float)
            n_samples = len(self.states[node])
            if n_samples == 0:
                continue

            for physical_state, social_state in self.states[node]:
                node_probs[(physical_state, social_state)] += 1
            for state in node_probs:
                node_probs[state] /= n_samples
            probs[node] = dict(node_probs)

        return probs

    def run_simulation_sequence(
        self,
        test_sequence: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
    ) -> list[dict[int, dict[tuple[int, int], float]] | None]:
        """Run simulations conditioned on a sequence of observed tests."""

        sequence_probs = []
        initial_states = [
            self.simulate_one_instance()
            for _ in range(self.n_simulations)
        ]
        valid_states_list = [initial_states]

        probs = self._calculate_probs_from_states(initial_states)
        sequence_probs.append(probs)

        current_valid_states = initial_states
        for node, test_type, expected_result in test_sequence:
            next_valid_states = []
            for states in current_valid_states:
                test_result = self.simulate_test(states, node, test_type)
                if test_result == expected_result:
                    next_valid_states.append(states)

            valid_states_list.append(next_valid_states)
            probs = self._calculate_probs_from_states(next_valid_states)
            sequence_probs.append(probs)
            current_valid_states = next_valid_states

        return sequence_probs

    def _calculate_probs_from_states(
        self,
        states_list: list[dict[int, tuple[int, int]]],
    ) -> dict[int, dict[tuple[int, int], float]] | None:
        if not states_list:
            return None

        probs = {}
        n_samples = len(states_list)
        for node in self.G.nodes():
            node_probs = defaultdict(float)
            for states in states_list:
                physical_state, social_state = states[node]
                node_probs[(physical_state, social_state)] += 1
            for state in node_probs:
                node_probs[state] /= n_samples
            probs[node] = dict(node_probs)

        return probs


def print_model_sim_comparison(model_probs, sim_probs, test_sequence=None) -> None:
    """Print exact and simulation probabilities side by side."""

    print("\nInitial probabilities:")
    for node in sorted(model_probs[0].keys()):
        print(f"\nNode {node}:")
        print("Model:      ", end="")
        probs_str = ", ".join(
            [
                f"P(P={physical},S={social})={prob:.4f}"
                for (physical, social), prob in sorted(model_probs[0][node].items())
            ]
        )
        print(probs_str)
        print("Simulation: ", end="")
        probs_str = ", ".join(
            [
                f"P(P={physical},S={social})={prob:.4f}"
                for (physical, social), prob in sorted(sim_probs[0][node].items())
            ]
        )
        print(probs_str)

    if test_sequence:
        for i, (m_probs, s_probs, test) in enumerate(
            zip(model_probs[1:], sim_probs[1:], test_sequence, strict=False),
            1,
        ):
            test_label = "physical" if test[1] == 1 else "social"
            print(f"\nAfter test {i} - node {test[0]}, {test_label} test, result {test[2]}:")
            for node in sorted(m_probs.keys()):
                print(f"\nNode {node}:")
                print("Model:      ", end="")
                probs_str = ", ".join(
                    [
                        f"P(P={physical},S={social})={prob:.4f}"
                        for (physical, social), prob in sorted(m_probs[node].items())
                    ]
                )
                print(probs_str)
                print("Simulation: ", end="")
                probs_str = ", ".join(
                    [
                        f"P(P={physical},S={social})={prob:.4f}"
                        for (physical, social), prob in sorted(s_probs[node].items())
                    ]
                )
                print(probs_str)
