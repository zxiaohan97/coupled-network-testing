"""Symmetry dynamic-programming benchmark for complete graphs.

On a complete graph, all untested non-seed nodes are exchangeable. The policy
state can therefore be represented by four counts:

``(physical_positive, physical_negative, social_positive, social_negative)``.

This module preserves the idea from ``Optimal/Complete_Graph.py`` while using
the package's likelihood-weighted posterior engine. The benchmark is small and
intended for validation: it compares a one-step greedy policy against the
optimal finite-horizon policy over the same symmetry-reduced state space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

import networkx as nx
import numpy as np

from coupled_network_testing.common.testing_types import TestType
from coupled_network_testing.general.simulation_model import (
    CascadeRealization,
    GeneralCascadeSimulator,
    GeneralModelParameters,
    TestRecord,
)
from coupled_network_testing.general.weighted_particle_filter import WeightedParticlePosterior

CompleteState: TypeAlias = tuple[int, int, int, int]


class CompleteGraphDecision(StrEnum):
    """Decision labels for the complete-graph symmetry DP."""

    PHYSICAL = "P"
    SOCIAL = "S"


@dataclass(frozen=True)
class CompleteGraphBenchmarkConfig:
    """Configuration for a complete-graph greedy-vs-optimal benchmark."""

    n_nodes: int
    budget: int
    parameters: GeneralModelParameters = field(
        default_factory=lambda: GeneralModelParameters(
            p_infect_careless=0.35,
            q_infect_careful=0.05,
            social_correlation=0.2,
            physical_error=0.1,
            social_error=0.05,
        )
    )
    n_particles: int = 20_000
    infection_seed: int = 0
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.n_nodes < 2:
            raise ValueError("n_nodes must be at least 2")
        if self.budget < 0:
            raise ValueError("budget cannot be negative")
        if self.budget > self.n_nodes - 1:
            raise ValueError("complete-graph benchmark tests fresh non-seed nodes only")
        if self.infection_seed != 0:
            raise ValueError("complete-graph symmetry benchmark currently expects seed node 0")
        if self.n_particles < 1:
            raise ValueError("n_particles must be positive")


@dataclass(frozen=True)
class CompleteGraphBenchmarkResult:
    """Greedy and optimal benchmark values under one sampled particle set."""

    config: CompleteGraphBenchmarkConfig
    greedy_uncertainty: float
    optimal_uncertainty: float
    greedy_expected_social_tests: float
    optimal_expected_social_tests: float
    initial_greedy_decision: CompleteGraphDecision
    initial_optimal_decision: CompleteGraphDecision

    @property
    def absolute_gap(self) -> float:
        """Greedy final uncertainty minus optimal final uncertainty."""

        return self.greedy_uncertainty - self.optimal_uncertainty

    @property
    def relative_gap(self) -> float:
        """Relative gap using greedy uncertainty as the denominator."""

        if self.greedy_uncertainty <= 0.0:
            return 0.0
        return self.absolute_gap / self.greedy_uncertainty


class CompleteGraphSymmetryBenchmark:
    """Compare greedy and optimal policies on complete graphs by symmetry DP."""

    def __init__(
        self,
        config: CompleteGraphBenchmarkConfig,
        particles: list[CascadeRealization] | None = None,
    ):
        self.config = config
        graph = nx.complete_graph(config.n_nodes)
        self.graph = graph
        if particles is None:
            simulator = GeneralCascadeSimulator(
                graph,
                graph,
                seeds=(config.infection_seed,),
                parameters=config.parameters,
                rng=np.random.default_rng(config.random_seed),
            )
            particles = simulator.sample_realizations(config.n_particles)
        self.posterior = WeightedParticlePosterior(
            particles,
            parameters=config.parameters,
            nodes=tuple(graph.nodes()),
        )
        self.optimal_decisions: dict[CompleteState, CompleteGraphDecision] = {}
        self.greedy_decisions: dict[CompleteState, CompleteGraphDecision] = {}
        self._optimal_cache: dict[CompleteState, tuple[float, float]] = {}
        self._greedy_cache: dict[CompleteState, tuple[float, float]] = {}
        self._terminal_cache: dict[CompleteState, float] = {}
        self._branch_cache: dict[tuple[CompleteState, TestType], object] = {}

    def evaluate(self) -> CompleteGraphBenchmarkResult:
        """Return greedy and optimal expected final uncertainty."""

        initial_state = (0, 0, 0, 0)
        greedy_uncertainty, greedy_social = self._greedy_value(initial_state)
        optimal_uncertainty, optimal_social = self._optimal_value(initial_state)
        return CompleteGraphBenchmarkResult(
            config=self.config,
            greedy_uncertainty=greedy_uncertainty,
            optimal_uncertainty=optimal_uncertainty,
            greedy_expected_social_tests=greedy_social,
            optimal_expected_social_tests=optimal_social,
            initial_greedy_decision=self.greedy_decisions[initial_state],
            initial_optimal_decision=self.optimal_decisions[initial_state],
        )

    def _optimal_value(self, state: CompleteState) -> tuple[float, float]:
        """Optimal expected final uncertainty and expected social-test count."""

        if state in self._optimal_cache:
            return self._optimal_cache[state]

        if sum(state) == self.config.budget:
            value = (self._terminal_uncertainty(state), 0.0)
            self._optimal_cache[state] = value
            return value

        physical_value = self._future_value(state, CompleteGraphDecision.PHYSICAL, optimal=True)
        social_value = self._future_value(state, CompleteGraphDecision.SOCIAL, optimal=True)
        if social_value[0] < physical_value[0]:
            self.optimal_decisions[state] = CompleteGraphDecision.SOCIAL
            self._optimal_cache[state] = social_value
            return social_value

        self.optimal_decisions[state] = CompleteGraphDecision.PHYSICAL
        self._optimal_cache[state] = physical_value
        return physical_value

    def _greedy_value(self, state: CompleteState) -> tuple[float, float]:
        """Final value when each state uses the one-step greedy decision."""

        if state in self._greedy_cache:
            return self._greedy_cache[state]

        if sum(state) == self.config.budget:
            value = (self._terminal_uncertainty(state), 0.0)
            self._greedy_cache[state] = value
            return value

        physical_branch = self._branch(state, TestType.PHYSICAL)
        social_branch = self._branch(state, TestType.SOCIAL)
        if social_branch.expected_uncertainty < physical_branch.expected_uncertainty:
            decision = CompleteGraphDecision.SOCIAL
        else:
            decision = CompleteGraphDecision.PHYSICAL

        self.greedy_decisions[state] = decision
        value = self._future_value(state, decision, optimal=False)
        self._greedy_cache[state] = value
        return value

    def _future_value(
        self,
        state: CompleteState,
        decision: CompleteGraphDecision,
        optimal: bool,
    ) -> tuple[float, float]:
        """Expected final value after choosing one physical or social test."""

        test_type = _decision_to_test_type(decision)
        branch = self._branch(state, test_type)
        positive_state = _advance_state(state, decision, observed_result=1)
        negative_state = _advance_state(state, decision, observed_result=0)

        positive_value, positive_social = (
            self._optimal_value(positive_state)
            if optimal
            else self._greedy_value(positive_state)
        )
        negative_value, negative_social = (
            self._optimal_value(negative_state)
            if optimal
            else self._greedy_value(negative_state)
        )

        p_positive = branch.positive_probability
        expected_uncertainty = p_positive * positive_value + (1.0 - p_positive) * negative_value
        expected_social = p_positive * positive_social + (1.0 - p_positive) * negative_social
        if decision is CompleteGraphDecision.SOCIAL:
            expected_social += 1.0
        return float(expected_uncertainty), float(expected_social)

    def _terminal_uncertainty(self, state: CompleteState) -> float:
        """Posterior infection uncertainty after a canonical count state."""

        if state not in self._terminal_cache:
            self._terminal_cache[state] = self.posterior.estimate(
                _state_to_test_sequence(state)
            ).physical_uncertainty
        return self._terminal_cache[state]

    def _branch(self, state: CompleteState, test_type: TestType):
        """One-step branch estimate for the next fresh node."""

        key = (state, test_type)
        if key in self._branch_cache:
            return self._branch_cache[key]

        next_node = sum(state) + 1
        branch = self.posterior.branch_estimate(
            _state_to_test_sequence(state),
            node=next_node,
            test_type=test_type,
        )
        self._branch_cache[key] = branch
        return branch


def run_complete_graph_benchmark(
    config: CompleteGraphBenchmarkConfig,
) -> CompleteGraphBenchmarkResult:
    """Convenience wrapper for one complete-graph benchmark run."""

    return CompleteGraphSymmetryBenchmark(config).evaluate()


def run_complete_graph_validation_grid(
    configs: list[CompleteGraphBenchmarkConfig] | tuple[CompleteGraphBenchmarkConfig, ...],
) -> list[CompleteGraphBenchmarkResult]:
    """Evaluate several complete-graph benchmark configurations."""

    return [run_complete_graph_benchmark(config) for config in configs]


def _state_to_test_sequence(state: CompleteState) -> tuple[TestRecord, ...]:
    """Convert a count state into a canonical complete-graph test history."""

    physical_positive, physical_negative, social_positive, social_negative = state
    sequence: list[TestRecord] = []
    node = 1
    for _ in range(physical_positive):
        sequence.append((node, int(TestType.PHYSICAL), 1))
        node += 1
    for _ in range(physical_negative):
        sequence.append((node, int(TestType.PHYSICAL), 0))
        node += 1
    for _ in range(social_positive):
        sequence.append((node, int(TestType.SOCIAL), 1))
        node += 1
    for _ in range(social_negative):
        sequence.append((node, int(TestType.SOCIAL), 0))
        node += 1
    return tuple(sequence)


def _advance_state(
    state: CompleteState,
    decision: CompleteGraphDecision,
    observed_result: int,
) -> CompleteState:
    """Advance a count state after one observed binary result."""

    physical_positive, physical_negative, social_positive, social_negative = state
    if decision is CompleteGraphDecision.PHYSICAL:
        if observed_result == 1:
            return (physical_positive + 1, physical_negative, social_positive, social_negative)
        return (physical_positive, physical_negative + 1, social_positive, social_negative)

    if observed_result == 1:
        return (physical_positive, physical_negative, social_positive + 1, social_negative)
    return (physical_positive, physical_negative, social_positive, social_negative + 1)


def _decision_to_test_type(decision: CompleteGraphDecision) -> TestType:
    """Convert a DP decision label to the shared test-type enum."""

    if decision is CompleteGraphDecision.PHYSICAL:
        return TestType.PHYSICAL
    return TestType.SOCIAL
