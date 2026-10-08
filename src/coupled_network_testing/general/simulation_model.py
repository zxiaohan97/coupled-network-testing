"""Monte Carlo simulation model for general cyclic networks.

The exact posterior update in this project is available only on trees. For
cyclic graphs, the original experiments approximated the posterior by sampling
many complete opinion-disease cascades and conditioning those samples on the
observed test history. This module contains only the model-generation part of
that workflow: social states, disease states, and noisy test observations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TypeAlias

import networkx as nx
import numpy as np

from coupled_network_testing.common.testing_types import TestType

PhysicalStates: TypeAlias = dict[int, int]
SocialStates: TypeAlias = dict[int, int]
TestResults: TypeAlias = dict[int, int]
TestRecord: TypeAlias = tuple[int, int, int]


@dataclass(frozen=True)
class GeneralModelParameters:
    """Parameters of the simulation-based coupled-network model."""

    p_infect_careless: float
    q_infect_careful: float
    social_correlation: float
    physical_error: float = 0.0
    social_error: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "p_infect_careless": self.p_infect_careless,
            "q_infect_careful": self.q_infect_careful,
            "social_correlation": self.social_correlation,
            "physical_error": self.physical_error,
            "social_error": self.social_error,
        }
        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class CascadeRealization:
    """One sampled complete world used by Monte Carlo posterior updates."""

    social_states: SocialStates
    social_components: tuple[frozenset[int], ...]
    physical_states: PhysicalStates
    infection_paths: dict[int, tuple[int, ...]]
    physical_tests: TestResults
    social_tests: TestResults

    def test_result(self, node: int, test_type: int | TestType) -> int:
        """Return the pre-sampled noisy result for one physical/social test."""

        if int(test_type) == TestType.PHYSICAL:
            return self.physical_tests[node]
        return self.social_tests[node]


class GeneralCascadeSimulator:
    """Sample coupled opinion-disease cascades on arbitrary graphs.

    The physical graph carries disease transmission. The social graph carries
    correlated opinion states. The two graphs may be identical or partially
    overlapping.
    """

    def __init__(
        self,
        physical_graph: nx.Graph,
        social_graph: nx.Graph,
        seeds: tuple[int, ...] | list[int],
        parameters: GeneralModelParameters,
        rng: np.random.Generator | None = None,
    ):
        if set(physical_graph.nodes()) != set(social_graph.nodes()):
            raise ValueError("physical and social graphs must have the same node set")
        if not seeds:
            raise ValueError("at least one infection seed is required")
        missing_seeds = set(seeds) - set(physical_graph.nodes())
        if missing_seeds:
            raise ValueError(f"seeds are not present in graph: {sorted(missing_seeds)}")

        self.physical_graph = physical_graph.copy()
        self.social_graph = social_graph.copy()
        self.seeds = tuple(seeds)
        self.parameters = parameters
        self.rng = rng or np.random.default_rng()

    @property
    def nodes(self) -> tuple[int, ...]:
        """Return graph nodes in deterministic order when labels are sortable."""

        try:
            return tuple(sorted(self.physical_graph.nodes()))
        except TypeError:
            return tuple(self.physical_graph.nodes())

    def sample_social_states(self) -> tuple[SocialStates, tuple[frozenset[int], ...]]:
        """Sample social opinions by percolating the social graph.

        Each social edge is retained with probability ``social_correlation``.
        Every connected component of the retained graph receives a shared
        careful/careless state, matching the original simulation logic.
        """

        retained_social = self.social_graph.copy()
        edges_to_remove = [
            edge
            for edge in retained_social.edges()
            if self.rng.random() > self.parameters.social_correlation
        ]
        retained_social.remove_edges_from(edges_to_remove)

        components = tuple(
            frozenset(component)
            for component in nx.connected_components(retained_social)
        )
        social_states: SocialStates = {}
        for component in components:
            state = int(self.rng.integers(0, 2))
            for node in component:
                social_states[node] = state

        return social_states, components

    def sample_physical_states(
        self,
        social_states: SocialStates,
    ) -> tuple[PhysicalStates, dict[int, tuple[int, ...]]]:
        """Sample the independent-cascade disease process.

        The infection probability is determined by the *target* node's social
        state. Careless targets use ``p_infect_careless`` and careful targets
        use ``q_infect_careful``.
        """

        physical_states = {node: 0 for node in self.nodes}
        infection_paths: dict[int, tuple[int, ...]] = {}
        queue: deque[tuple[int, tuple[int, ...]]] = deque()

        for seed in self.seeds:
            physical_states[seed] = 1
            infection_paths[seed] = (seed,)
            queue.append((seed, (seed,)))

        while queue:
            current, path = queue.popleft()
            for neighbor in self.physical_graph.neighbors(current):
                if physical_states[neighbor] == 1:
                    continue

                infection_probability = (
                    self.parameters.p_infect_careless
                    if social_states[neighbor] == 1
                    else self.parameters.q_infect_careful
                )
                if self.rng.random() < infection_probability:
                    physical_states[neighbor] = 1
                    infection_path = path + (neighbor,)
                    infection_paths[neighbor] = infection_path
                    queue.append((neighbor, infection_path))

        return physical_states, infection_paths

    def sample_test_results(
        self,
        social_states: SocialStates,
        physical_states: PhysicalStates,
    ) -> tuple[TestResults, TestResults]:
        """Sample noisy physical and social observations for every node."""

        physical_tests: TestResults = {}
        social_tests: TestResults = {}
        for node in self.nodes:
            physical_tests[node] = self._sample_noisy_binary_observation(
                true_state=physical_states[node],
                error_rate=self.parameters.physical_error,
            )
            social_tests[node] = self._sample_noisy_binary_observation(
                true_state=social_states[node],
                error_rate=self.parameters.social_error,
            )
        return physical_tests, social_tests

    def sample_realization(self) -> CascadeRealization:
        """Sample one complete cascade and its test outcomes."""

        social_states, social_components = self.sample_social_states()
        physical_states, infection_paths = self.sample_physical_states(social_states)
        physical_tests, social_tests = self.sample_test_results(social_states, physical_states)
        return CascadeRealization(
            social_states=social_states,
            social_components=social_components,
            physical_states=physical_states,
            infection_paths=infection_paths,
            physical_tests=physical_tests,
            social_tests=social_tests,
        )

    def sample_realizations(self, n_samples: int) -> list[CascadeRealization]:
        """Sample many complete worlds for Monte Carlo posterior estimation."""

        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        return [self.sample_realization() for _ in range(n_samples)]

    def _sample_noisy_binary_observation(self, true_state: int, error_rate: float) -> int:
        """Flip a binary state with the given test-error probability."""

        if self.rng.random() < error_rate:
            return 1 - true_state
        return true_state
