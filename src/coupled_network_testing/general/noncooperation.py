"""Non-cooperation extensions for general-network testing experiments.

The original non-cooperation scripts model refusals as a social-state-dependent
testing response: careless nodes are more likely to refuse physical tests. This
module keeps that idea, but separates it from the base cascade simulator so the
fully cooperative model remains the clean default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum

import networkx as nx
import numpy as np
import pandas as pd

from coupled_network_testing.common.metrics import bernoulli_variance
from coupled_network_testing.common.testing_types import TestType
from coupled_network_testing.general.policies import candidate_tests
from coupled_network_testing.general.simulation_model import (
    CascadeRealization,
    GeneralModelParameters,
    SocialStates,
    TestRecord,
)
from coupled_network_testing.general.weighted_particle_filter import (
    WeightedParticlePosterior,
    WeightedPosteriorEstimate,
)


class NonCooperationMode(IntEnum):
    """Refusal scenarios used by the original cooperation experiments."""

    CARELESS_PHYSICAL = 1
    MIXED_PHYSICAL = 2
    CARELESS_PHYSICAL_AND_SOCIAL = 3


class NonCooperativeStrategy(StrEnum):
    """Strategy labels emitted by non-cooperation runners."""

    WEIGHTED_GREEDY = "noncooperative_weighted_greedy"
    CONTACT_TRACING = "noncooperative_contact_tracing"


@dataclass(frozen=True)
class NonCooperationParameters:
    """Parameters controlling social-state-dependent test refusal.

    ``cooperation_fraction`` is the probability that a careless node cooperates.
    In mode 2, careful nodes can also refuse, but the original code made them
    more cooperative by default: ``coop + (1 - coop) / 2``.
    """

    cooperation_fraction: float = 1.0
    mode: NonCooperationMode | int = NonCooperationMode.CARELESS_PHYSICAL
    careful_cooperation_fraction: float | None = None

    def __post_init__(self) -> None:
        NonCooperationMode(self.mode)
        _validate_probability(self.cooperation_fraction, "cooperation_fraction")
        if self.careful_cooperation_fraction is not None:
            _validate_probability(
                self.careful_cooperation_fraction,
                "careful_cooperation_fraction",
            )

    @property
    def mode_enum(self) -> NonCooperationMode:
        """Return ``mode`` as a named enum value."""

        return NonCooperationMode(self.mode)

    @property
    def effective_careful_cooperation_fraction(self) -> float:
        """Cooperation fraction for careful nodes in mode 2."""

        if self.careful_cooperation_fraction is not None:
            return self.careful_cooperation_fraction
        return self.cooperation_fraction + (1.0 - self.cooperation_fraction) / 2.0


@dataclass(frozen=True)
class NonCooperativePolicyDecision:
    """One non-cooperative policy decision with refusal-aware diagnostics."""

    node: int
    test_type: TestType
    expected_uncertainty: float
    positive_probability: float | None
    minimum_branch_ess: float | None
    refusal_probability: float
    is_stable: bool


@dataclass(frozen=True)
class NonCooperativePolicyStep:
    """One observed step from a non-cooperative policy run."""

    strategy: str
    test_index: int
    node: int
    test_type: int
    result: int
    refused: bool
    posterior_uncertainty: float
    effective_sample_size: float
    ess_fraction: float
    expected_uncertainty_before_test: float | None
    positive_probability: float | None
    minimum_branch_ess: float | None
    refusal_probability: float
    known_noncooperative_count: int
    is_stable: bool


def sample_cooperation_status(
    social_states: SocialStates,
    parameters: NonCooperationParameters,
    rng: np.random.Generator,
) -> dict[int, bool]:
    """Sample which nodes will cooperate, matching the original mode logic."""

    status = {node: True for node in social_states}
    if parameters.cooperation_fraction >= 1.0:
        return status

    careless_nodes = [node for node, state in social_states.items() if state == 1]
    careful_nodes = [node for node, state in social_states.items() if state == 0]
    n_careless_refusers = int((1.0 - parameters.cooperation_fraction) * len(careless_nodes))

    refusers = _choose_without_replacement(careless_nodes, n_careless_refusers, rng)
    if parameters.mode_enum is NonCooperationMode.MIXED_PHYSICAL:
        careful_fraction = parameters.effective_careful_cooperation_fraction
        n_careful_refusers = int((1.0 - careful_fraction) * len(careful_nodes))
        refusers.extend(_choose_without_replacement(careful_nodes, n_careful_refusers, rng))

    for node in refusers:
        status[node] = False
    return status


def refusal_probability(
    posterior_estimate: WeightedPosteriorEstimate,
    node: int,
    test_type: int | TestType,
    parameters: NonCooperationParameters,
) -> float:
    """Estimate refusal probability from the current posterior social state."""

    p_careless = posterior_estimate.social_posteriors[node]
    mode = parameters.mode_enum
    if int(test_type) == TestType.PHYSICAL:
        if mode is NonCooperationMode.MIXED_PHYSICAL:
            p_refuse_careless = 1.0 - parameters.cooperation_fraction
            p_refuse_careful = 1.0 - parameters.effective_careful_cooperation_fraction
            return _clip_probability(
                p_refuse_careless * p_careless + p_refuse_careful * (1.0 - p_careless)
            )
        return _clip_probability((1.0 - parameters.cooperation_fraction) * p_careless)

    if mode is NonCooperationMode.CARELESS_PHYSICAL_AND_SOCIAL:
        return _clip_probability((1.0 - parameters.cooperation_fraction) * p_careless)
    return 0.0


def is_test_refused(
    cooperative_status: dict[int, bool],
    node: int,
    test_type: int | TestType,
    parameters: NonCooperationParameters,
) -> bool:
    """Return whether a chosen test is refused in the sampled ground truth."""

    if cooperative_status.get(node, True):
        return False
    if int(test_type) == TestType.PHYSICAL:
        return True
    return parameters.mode_enum is NonCooperationMode.CARELESS_PHYSICAL_AND_SOCIAL


def select_weighted_greedy_test_with_noncooperation(
    posterior: WeightedParticlePosterior,
    noncooperation_parameters: NonCooperationParameters,
    test_sequence: tuple[TestRecord, ...] | list[TestRecord] = (),
    known_noncooperative: set[int] | frozenset[int] | None = None,
    seeds: tuple[int, ...] | list[int] = (),
    min_branch_ess: float = 300.0,
    allow_retests: bool = False,
    include_social_tests: bool = True,
) -> NonCooperativePolicyDecision:
    """Select a greedy test after accounting for possible refusal.

    A refusal gives no biological/social measurement, so its branch keeps the
    current posterior uncertainty. Non-refusal branches use the normal weighted
    posterior update for positive and negative outcomes.
    """

    current_estimate = posterior.estimate(test_sequence)
    known_noncooperative = set(known_noncooperative or set())
    best: tuple[tuple[int, TestType], float, float, float, float] | None = None

    for node, test_type in candidate_tests(
        posterior.nodes,
        test_sequence=test_sequence,
        seeds=seeds,
        allow_retests=allow_retests,
        include_social_tests=include_social_tests,
    ):
        if _is_blocked_by_known_refusal(
            node,
            test_type,
            known_noncooperative,
            noncooperation_parameters,
        ):
            continue

        branch = posterior.branch_estimate(test_sequence, node=node, test_type=test_type)
        p_refuse = refusal_probability(
            current_estimate,
            node,
            test_type,
            noncooperation_parameters,
        )
        expected_uncertainty = (
            p_refuse * current_estimate.physical_uncertainty
            + (1.0 - p_refuse) * branch.expected_uncertainty
        )
        candidate = (
            (node, test_type),
            float(expected_uncertainty),
            float(branch.positive_probability),
            float(branch.minimum_branch_ess),
            float(p_refuse),
        )
        if best is None or _is_better_noncooperative_candidate(candidate, best):
            best = candidate

    if best is None:
        raise ValueError("no available candidate tests")

    (node, test_type), expected_uncertainty, positive_probability, minimum_ess, p_refuse = best
    return NonCooperativePolicyDecision(
        node=node,
        test_type=test_type,
        expected_uncertainty=expected_uncertainty,
        positive_probability=positive_probability,
        minimum_branch_ess=minimum_ess,
        refusal_probability=p_refuse,
        is_stable=minimum_ess >= min_branch_ess,
    )


def select_contact_tracing_test_with_noncooperation(
    physical_graph: nx.Graph,
    posterior_estimate: WeightedPosteriorEstimate,
    test_sequence: tuple[TestRecord, ...] | list[TestRecord] = (),
    known_noncooperative: set[int] | frozenset[int] | None = None,
    seeds: tuple[int, ...] | list[int] = (),
) -> NonCooperativePolicyDecision:
    """Select a contact-tracing physical test while skipping known refusers."""

    known_noncooperative = set(known_noncooperative or set())
    seed_set = set(seeds)
    tested_physical = {
        node for node, test_type, _ in test_sequence if int(test_type) == TestType.PHYSICAL
    }
    positive_physical = {
        node
        for node, test_type, result in test_sequence
        if int(test_type) == TestType.PHYSICAL and result == 1
    }

    unavailable = tested_physical | seed_set | known_noncooperative
    frontier = set()
    for node in positive_physical:
        frontier.update(physical_graph.neighbors(node))
    frontier.difference_update(unavailable)

    if frontier:
        node = _most_uncertain_node(frontier, posterior_estimate)
    else:
        candidates = set(physical_graph.nodes()) - unavailable
        if not candidates:
            raise ValueError("no available physical tests")
        node = _most_uncertain_node(candidates, posterior_estimate)

    return NonCooperativePolicyDecision(
        node=int(node),
        test_type=TestType.PHYSICAL,
        expected_uncertainty=posterior_estimate.physical_uncertainty,
        positive_probability=None,
        minimum_branch_ess=None,
        refusal_probability=0.0,
        is_stable=True,
    )


def run_noncooperative_weighted_greedy_policy(
    physical_graph: nx.Graph,
    particles: list[CascadeRealization],
    ground_truth: CascadeRealization,
    model_parameters: GeneralModelParameters,
    noncooperation_parameters: NonCooperationParameters,
    seeds: tuple[int, ...],
    max_tests: int,
    rng: np.random.Generator | None = None,
    cooperative_status: dict[int, bool] | None = None,
    min_branch_ess: float = 300.0,
    ess_warning_threshold: float = 300.0,
) -> tuple[NonCooperativePolicyStep, ...]:
    """Run refusal-aware weighted greedy testing on a sampled cascade."""

    posterior = WeightedParticlePosterior(
        particles,
        parameters=model_parameters,
        nodes=tuple(physical_graph.nodes()),
    )
    return _run_noncooperative_policy(
        strategy=NonCooperativeStrategy.WEIGHTED_GREEDY,
        physical_graph=physical_graph,
        posterior=posterior,
        ground_truth=ground_truth,
        noncooperation_parameters=noncooperation_parameters,
        seeds=seeds,
        max_tests=max_tests,
        rng=rng,
        cooperative_status=cooperative_status,
        min_branch_ess=min_branch_ess,
        ess_warning_threshold=ess_warning_threshold,
    )


def run_noncooperative_contact_tracing_policy(
    physical_graph: nx.Graph,
    particles: list[CascadeRealization],
    ground_truth: CascadeRealization,
    model_parameters: GeneralModelParameters,
    noncooperation_parameters: NonCooperationParameters,
    seeds: tuple[int, ...],
    max_tests: int,
    rng: np.random.Generator | None = None,
    cooperative_status: dict[int, bool] | None = None,
    ess_warning_threshold: float = 300.0,
) -> tuple[NonCooperativePolicyStep, ...]:
    """Run refusal-aware physical contact tracing on a sampled cascade."""

    posterior = WeightedParticlePosterior(
        particles,
        parameters=model_parameters,
        nodes=tuple(physical_graph.nodes()),
    )
    return _run_noncooperative_policy(
        strategy=NonCooperativeStrategy.CONTACT_TRACING,
        physical_graph=physical_graph,
        posterior=posterior,
        ground_truth=ground_truth,
        noncooperation_parameters=noncooperation_parameters,
        seeds=seeds,
        max_tests=max_tests,
        rng=rng,
        cooperative_status=cooperative_status,
        min_branch_ess=0.0,
        ess_warning_threshold=ess_warning_threshold,
    )


def run_noncooperative_policy_comparison(
    physical_graph: nx.Graph,
    particles: list[CascadeRealization],
    ground_truth: CascadeRealization,
    model_parameters: GeneralModelParameters,
    noncooperation_parameters: NonCooperationParameters,
    seeds: tuple[int, ...],
    max_tests: int,
    rng: np.random.Generator | None = None,
    cooperative_status: dict[int, bool] | None = None,
    min_branch_ess: float = 300.0,
    ess_warning_threshold: float = 300.0,
) -> tuple[NonCooperativePolicyStep, ...]:
    """Run greedy and contact tracing with the same sampled cooperation status."""

    rng = rng or np.random.default_rng()
    if cooperative_status is None:
        cooperative_status = sample_cooperation_status(
            ground_truth.social_states,
            noncooperation_parameters,
            rng,
        )
    greedy_steps = run_noncooperative_weighted_greedy_policy(
        physical_graph,
        particles,
        ground_truth,
        model_parameters,
        noncooperation_parameters,
        seeds,
        max_tests,
        rng=rng,
        cooperative_status=cooperative_status,
        min_branch_ess=min_branch_ess,
        ess_warning_threshold=ess_warning_threshold,
    )
    contact_steps = run_noncooperative_contact_tracing_policy(
        physical_graph,
        particles,
        ground_truth,
        model_parameters,
        noncooperation_parameters,
        seeds,
        max_tests,
        rng=rng,
        cooperative_status=cooperative_status,
        ess_warning_threshold=ess_warning_threshold,
    )
    return (*greedy_steps, *contact_steps)


def noncooperative_steps_to_dataframe(
    steps: tuple[NonCooperativePolicyStep, ...] | list[NonCooperativePolicyStep],
) -> pd.DataFrame:
    """Return non-cooperative policy steps as a tidy table."""

    return pd.DataFrame([asdict(step) for step in steps])


def _run_noncooperative_policy(
    strategy: NonCooperativeStrategy,
    physical_graph: nx.Graph,
    posterior: WeightedParticlePosterior,
    ground_truth: CascadeRealization,
    noncooperation_parameters: NonCooperationParameters,
    seeds: tuple[int, ...],
    max_tests: int,
    rng: np.random.Generator | None,
    cooperative_status: dict[int, bool] | None,
    min_branch_ess: float,
    ess_warning_threshold: float,
) -> tuple[NonCooperativePolicyStep, ...]:
    """Shared sequential loop for non-cooperative policies."""

    rng = rng or np.random.default_rng()
    if cooperative_status is None:
        cooperative_status = sample_cooperation_status(
            ground_truth.social_states,
            noncooperation_parameters,
            rng,
        )

    posterior_sequence: list[TestRecord] = []
    known_noncooperative: set[int] = set()
    steps: list[NonCooperativePolicyStep] = []

    for test_index in range(1, max_tests + 1):
        estimate_before = posterior.estimate(posterior_sequence)
        if strategy is NonCooperativeStrategy.WEIGHTED_GREEDY:
            decision = select_weighted_greedy_test_with_noncooperation(
                posterior,
                noncooperation_parameters,
                test_sequence=posterior_sequence,
                known_noncooperative=known_noncooperative,
                seeds=seeds,
                min_branch_ess=min_branch_ess,
            )
        else:
            decision = select_contact_tracing_test_with_noncooperation(
                physical_graph,
                estimate_before,
                test_sequence=posterior_sequence,
                known_noncooperative=known_noncooperative,
                seeds=seeds,
            )

        refused = is_test_refused(
            cooperative_status,
            decision.node,
            decision.test_type,
            noncooperation_parameters,
        )
        if refused:
            result = -1
            known_noncooperative.add(decision.node)
        else:
            result = ground_truth.test_result(decision.node, decision.test_type)
            posterior_sequence.append((decision.node, int(decision.test_type), result))

        estimate_after = posterior.estimate(posterior_sequence)
        steps.append(
            _make_noncooperative_step(
                strategy,
                test_index,
                decision,
                result,
                refused,
                estimate_after,
                known_noncooperative,
                ess_warning_threshold,
            )
        )

    return tuple(steps)


def _make_noncooperative_step(
    strategy: NonCooperativeStrategy,
    test_index: int,
    decision: NonCooperativePolicyDecision,
    result: int,
    refused: bool,
    estimate_after: WeightedPosteriorEstimate,
    known_noncooperative: set[int],
    ess_warning_threshold: float,
) -> NonCooperativePolicyStep:
    """Build a step row after either an observed result or a refusal."""

    is_stable = (
        decision.is_stable
        and estimate_after.effective_sample_size >= ess_warning_threshold
    )
    return NonCooperativePolicyStep(
        strategy=strategy.value,
        test_index=test_index,
        node=decision.node,
        test_type=int(decision.test_type),
        result=result,
        refused=refused,
        posterior_uncertainty=estimate_after.physical_uncertainty,
        effective_sample_size=estimate_after.effective_sample_size,
        ess_fraction=estimate_after.ess_fraction,
        expected_uncertainty_before_test=decision.expected_uncertainty,
        positive_probability=decision.positive_probability,
        minimum_branch_ess=decision.minimum_branch_ess,
        refusal_probability=decision.refusal_probability,
        known_noncooperative_count=len(known_noncooperative),
        is_stable=is_stable,
    )


def _is_blocked_by_known_refusal(
    node: int,
    test_type: TestType,
    known_noncooperative: set[int],
    parameters: NonCooperationParameters,
) -> bool:
    """Return whether a known refuser makes this candidate unproductive."""

    if node not in known_noncooperative:
        return False
    if int(test_type) == TestType.PHYSICAL:
        return True
    return parameters.mode_enum is NonCooperationMode.CARELESS_PHYSICAL_AND_SOCIAL


def _is_better_noncooperative_candidate(
    candidate: tuple[tuple[int, TestType], float, float, float, float],
    best: tuple[tuple[int, TestType], float, float, float, float],
) -> bool:
    """Compare refusal-aware candidates with deterministic tie-breaking."""

    candidate_pair, expected, _, minimum_ess, p_refuse = candidate
    best_pair, best_expected, _, best_minimum_ess, best_p_refuse = best
    if expected != best_expected:
        return expected < best_expected
    if minimum_ess != best_minimum_ess:
        return minimum_ess > best_minimum_ess
    if p_refuse != best_p_refuse:
        return p_refuse < best_p_refuse
    return (candidate_pair[0], int(candidate_pair[1])) < (
        best_pair[0],
        int(best_pair[1]),
    )


def _most_uncertain_node(
    candidates: set[int],
    posterior_estimate: WeightedPosteriorEstimate,
) -> int:
    """Choose the highest-uncertainty candidate with deterministic ties."""

    return min(
        candidates,
        key=lambda node: (
            -bernoulli_variance(posterior_estimate.physical_posteriors[node]),
            node,
        ),
    )


def _choose_without_replacement(
    values: list[int],
    n_selected: int,
    rng: np.random.Generator,
) -> list[int]:
    """Choose values without assuming node labels are NumPy-compatible."""

    if n_selected <= 0:
        return []
    indices = rng.choice(len(values), size=n_selected, replace=False)
    return [values[int(index)] for index in indices]


def _validate_probability(value: float, name: str) -> None:
    """Raise if ``value`` is outside ``[0, 1]``."""

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _clip_probability(value: float) -> float:
    """Clip tiny floating-point drift to a probability."""

    return float(min(1.0, max(0.0, value)))
