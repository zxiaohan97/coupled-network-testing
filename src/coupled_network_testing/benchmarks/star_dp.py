"""Exact symmetry dynamic-programming benchmark for star-side trees.

The benchmark tree has a known infected seed ``0``, one center node ``1``, and
exchangeable side nodes ``2..N-1``. The DP action space mirrors the original
star benchmark: physical/social tests on the center, or physical/social tests on
a fresh side node. Posterior probabilities are evaluated with the exact tree
update code rather than with the older standalone formulas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from coupled_network_testing.common.testing_types import TestType
from coupled_network_testing.tree import ExactTreePosteriorModel, ModelParameters

StarState: TypeAlias = tuple[int, int, int, int, int, int, int, int]
TestSequence: TypeAlias = tuple[tuple[int, int, int], ...]


class StarDecision(StrEnum):
    """Decision labels for the star-side symmetry DP."""

    CENTER_PHYSICAL = "PC"
    CENTER_SOCIAL = "SC"
    SIDE_PHYSICAL = "PS"
    SIDE_SOCIAL = "SS"


@dataclass(frozen=True)
class StarBenchmarkConfig:
    """Configuration for a star-side greedy-vs-optimal benchmark."""

    n_nodes: int
    budget: int
    parameters: ModelParameters = field(
        default_factory=lambda: ModelParameters(
            p_infect_careless=0.8,
            q_infect_careful=0.2,
            social_correlation=1.0,
            physical_error=0.2,
            social_error=0.02,
        )
    )
    infection_seed: int = 0

    def __post_init__(self) -> None:
        if self.n_nodes < 3:
            raise ValueError("n_nodes must include a seed, a center, and at least one side node")
        if self.budget < 0:
            raise ValueError("budget cannot be negative")
        if self.infection_seed != 0:
            raise ValueError("star benchmark currently expects seed node 0")


@dataclass(frozen=True)
class StarBenchmarkResult:
    """Greedy and optimal benchmark values for one star-side tree."""

    config: StarBenchmarkConfig
    greedy_uncertainty: float
    optimal_uncertainty: float
    greedy_expected_social_tests: float
    optimal_expected_social_tests: float
    initial_greedy_decision: StarDecision
    initial_optimal_decision: StarDecision

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


@dataclass(frozen=True)
class _BranchEstimate:
    positive_probability: float
    expected_uncertainty: float


class StarTreeSymmetryBenchmark:
    """Compare greedy and optimal policies on star-side trees by exact DP."""

    def __init__(self, config: StarBenchmarkConfig):
        self.config = config
        self.edges = _star_side_edges(config.n_nodes)
        self.model = ExactTreePosteriorModel.from_edges(
            self.edges,
            seed=config.infection_seed,
            parameters=config.parameters,
        )
        self.optimal_decisions: dict[StarState, StarDecision] = {}
        self.greedy_decisions: dict[StarState, StarDecision] = {}
        self._optimal_cache: dict[StarState, tuple[float, float]] = {}
        self._greedy_cache: dict[StarState, tuple[float, float]] = {}
        self._terminal_cache: dict[StarState, float] = {}
        self._branch_cache: dict[tuple[StarState, StarDecision], _BranchEstimate] = {}

    def evaluate(self) -> StarBenchmarkResult:
        """Return greedy and optimal expected final uncertainty."""

        initial_state = (0, 0, 0, 0, 0, 0, 0, 0)
        greedy_uncertainty, greedy_social = self._greedy_value(initial_state)
        optimal_uncertainty, optimal_social = self._optimal_value(initial_state)
        return StarBenchmarkResult(
            config=self.config,
            greedy_uncertainty=greedy_uncertainty,
            optimal_uncertainty=optimal_uncertainty,
            greedy_expected_social_tests=greedy_social,
            optimal_expected_social_tests=optimal_social,
            initial_greedy_decision=self.greedy_decisions[initial_state],
            initial_optimal_decision=self.optimal_decisions[initial_state],
        )

    def _optimal_value(self, state: StarState) -> tuple[float, float]:
        """Optimal expected final uncertainty and expected social-test count."""

        if state in self._optimal_cache:
            return self._optimal_cache[state]
        if sum(state) == self.config.budget:
            value = (self._terminal_uncertainty(state), 0.0)
            self._optimal_cache[state] = value
            return value

        best_decision = None
        best_value = (float("inf"), 0.0)
        for decision in self._available_decisions(state):
            value = self._future_value(state, decision, optimal=True)
            if value[0] < best_value[0]:
                best_decision = decision
                best_value = value

        if best_decision is None:
            best_value = (self._terminal_uncertainty(state), 0.0)
        else:
            self.optimal_decisions[state] = best_decision
        self._optimal_cache[state] = best_value
        return best_value

    def _greedy_value(self, state: StarState) -> tuple[float, float]:
        """Final value when each state uses the one-step greedy decision."""

        if state in self._greedy_cache:
            return self._greedy_cache[state]
        if sum(state) == self.config.budget:
            value = (self._terminal_uncertainty(state), 0.0)
            self._greedy_cache[state] = value
            return value

        best_decision = None
        best_uncertainty = float("inf")
        for decision in self._available_decisions(state):
            branch = self._branch(state, decision)
            if branch.expected_uncertainty < best_uncertainty:
                best_decision = decision
                best_uncertainty = branch.expected_uncertainty

        if best_decision is None:
            value = (self._terminal_uncertainty(state), 0.0)
        else:
            self.greedy_decisions[state] = best_decision
            value = self._future_value(state, best_decision, optimal=False)
        self._greedy_cache[state] = value
        return value

    def _future_value(
        self,
        state: StarState,
        decision: StarDecision,
        optimal: bool,
    ) -> tuple[float, float]:
        """Expected final value after choosing one center or side test."""

        branch = self._branch(state, decision)
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
        if decision in {StarDecision.CENTER_SOCIAL, StarDecision.SIDE_SOCIAL}:
            expected_social += 1.0
        return float(expected_uncertainty), float(expected_social)

    def _terminal_uncertainty(self, state: StarState) -> float:
        """Exact posterior uncertainty after a canonical star count state."""

        if state not in self._terminal_cache:
            self._load_state(state)
            self._terminal_cache[state] = self.model.get_physical_uncertainty()
        return self._terminal_cache[state]

    def _branch(self, state: StarState, decision: StarDecision) -> _BranchEstimate:
        """One-step branch probabilities and immediate expected uncertainty."""

        key = (state, decision)
        if key in self._branch_cache:
            return self._branch_cache[key]

        node, test_type = _decision_to_node_test(state, decision)
        self._load_state(state)
        p_positive = self.model.get_test_outcome_probability(node, int(test_type), 1)
        snapshot = self.model.snapshot()

        self.model.update_with_test((node, int(test_type), 1))
        positive_uncertainty = self.model.get_physical_uncertainty()
        self.model.restore_snapshot(snapshot)

        self.model.update_with_test((node, int(test_type), 0))
        negative_uncertainty = self.model.get_physical_uncertainty()

        expected_uncertainty = (
            p_positive * positive_uncertainty + (1.0 - p_positive) * negative_uncertainty
        )
        branch = _BranchEstimate(
            positive_probability=float(p_positive),
            expected_uncertainty=float(expected_uncertainty),
        )
        self._branch_cache[key] = branch
        return branch

    def _load_state(self, state: StarState) -> None:
        """Load the exact posterior for the canonical sequence of a count state."""

        self.model.reset_to_initial_state()
        self.model.update_with_test_sequence(_state_to_test_sequence(state))

    def _available_decisions(self, state: StarState) -> tuple[StarDecision, ...]:
        """Return decisions that are valid for the current count state."""

        decisions = [StarDecision.CENTER_PHYSICAL, StarDecision.CENTER_SOCIAL]
        if _side_tests_used(state) < self.config.n_nodes - 2:
            decisions.extend([StarDecision.SIDE_PHYSICAL, StarDecision.SIDE_SOCIAL])
        return tuple(decisions)


def run_star_tree_benchmark(config: StarBenchmarkConfig) -> StarBenchmarkResult:
    """Convenience wrapper for one star-side benchmark run."""

    return StarTreeSymmetryBenchmark(config).evaluate()


def run_star_tree_validation_grid(
    configs: list[StarBenchmarkConfig] | tuple[StarBenchmarkConfig, ...],
) -> list[StarBenchmarkResult]:
    """Evaluate several star-side benchmark configurations."""

    return [run_star_tree_benchmark(config) for config in configs]


def _star_side_edges(n_nodes: int) -> tuple[tuple[int, int], ...]:
    """Build a seed-center-side star tree."""

    return ((0, 1), *tuple((1, node) for node in range(2, n_nodes)))


def _state_to_test_sequence(state: StarState) -> TestSequence:
    """Convert a symmetry count state into one canonical exact-test sequence."""

    center_physical_pos, center_physical_neg, center_social_pos, center_social_neg = state[:4]
    side_physical_pos, side_physical_neg, side_social_pos, side_social_neg = state[4:]
    sequence: list[tuple[int, int, int]] = []

    sequence.extend((1, int(TestType.PHYSICAL), 1) for _ in range(center_physical_pos))
    sequence.extend((1, int(TestType.PHYSICAL), 0) for _ in range(center_physical_neg))
    sequence.extend((1, int(TestType.SOCIAL), 1) for _ in range(center_social_pos))
    sequence.extend((1, int(TestType.SOCIAL), 0) for _ in range(center_social_neg))

    node = 2
    for test_type, result, count in (
        (TestType.PHYSICAL, 1, side_physical_pos),
        (TestType.PHYSICAL, 0, side_physical_neg),
        (TestType.SOCIAL, 1, side_social_pos),
        (TestType.SOCIAL, 0, side_social_neg),
    ):
        for _ in range(count):
            sequence.append((node, int(test_type), result))
            node += 1

    return tuple(sequence)


def _advance_state(
    state: StarState,
    decision: StarDecision,
    observed_result: int,
) -> StarState:
    """Advance a star count state after one observed binary result."""

    counts = list(state)
    index_by_decision = {
        StarDecision.CENTER_PHYSICAL: 0 if observed_result == 1 else 1,
        StarDecision.CENTER_SOCIAL: 2 if observed_result == 1 else 3,
        StarDecision.SIDE_PHYSICAL: 4 if observed_result == 1 else 5,
        StarDecision.SIDE_SOCIAL: 6 if observed_result == 1 else 7,
    }
    counts[index_by_decision[decision]] += 1
    return tuple(counts)  # type: ignore[return-value]


def _decision_to_node_test(state: StarState, decision: StarDecision) -> tuple[int, TestType]:
    """Map a symmetry decision to a canonical node and test type."""

    if decision is StarDecision.CENTER_PHYSICAL:
        return 1, TestType.PHYSICAL
    if decision is StarDecision.CENTER_SOCIAL:
        return 1, TestType.SOCIAL
    next_side_node = 2 + _side_tests_used(state)
    if decision is StarDecision.SIDE_PHYSICAL:
        return next_side_node, TestType.PHYSICAL
    return next_side_node, TestType.SOCIAL


def _side_tests_used(state: StarState) -> int:
    """Return how many fresh side nodes have already been tested."""

    return sum(state[4:])
