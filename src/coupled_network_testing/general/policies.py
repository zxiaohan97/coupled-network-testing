"""Testing policies for simulation-based general-network experiments."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from coupled_network_testing.common.metrics import bernoulli_variance
from coupled_network_testing.common.testing_types import TestType
from coupled_network_testing.general.simulation_model import TestRecord
from coupled_network_testing.general.weighted_particle_filter import (
    WeightedBranchEstimate,
    WeightedParticlePosterior,
    WeightedPosteriorEstimate,
)


@dataclass(frozen=True)
class PolicyDecision:
    """One selected node/test pair with lookahead diagnostics."""

    node: int
    test_type: TestType
    expected_uncertainty: float
    positive_probability: float | None
    minimum_branch_ess: float | None
    is_stable: bool


def candidate_tests(
    nodes: tuple[int, ...] | list[int],
    test_sequence: tuple[TestRecord, ...] | list[TestRecord] = (),
    seeds: tuple[int, ...] | list[int] = (),
    allow_retests: bool = False,
    include_social_tests: bool = True,
) -> list[tuple[int, TestType]]:
    """Return policy candidates without using hidden states.

    Repeated tests are disabled by default because the public simulator stores
    one noisy observation per node/test type. Keeping candidates unique avoids
    accidentally treating the same pre-sampled noisy draw as a fresh retest.
    """

    tested_pairs = {(node, int(test_type)) for node, test_type, _ in test_sequence}
    seed_set = set(seeds)
    types = [TestType.PHYSICAL]
    if include_social_tests:
        types.insert(0, TestType.SOCIAL)

    candidates: list[tuple[int, TestType]] = []
    for node in sorted(nodes):
        for test_type in types:
            if test_type == TestType.PHYSICAL and node in seed_set:
                continue
            if not allow_retests and (node, int(test_type)) in tested_pairs:
                continue
            candidates.append((node, test_type))
    return candidates


def select_weighted_greedy_test(
    posterior: WeightedParticlePosterior,
    test_sequence: tuple[TestRecord, ...] | list[TestRecord] = (),
    seeds: tuple[int, ...] | list[int] = (),
    min_branch_ess: float = 300.0,
    allow_retests: bool = False,
    include_social_tests: bool = True,
) -> PolicyDecision:
    """Select the candidate test with smallest expected posterior uncertainty."""

    best_candidate: tuple[int, TestType] | None = None
    best_branch: WeightedBranchEstimate | None = None

    for node, test_type in candidate_tests(
        posterior.nodes,
        test_sequence=test_sequence,
        seeds=seeds,
        allow_retests=allow_retests,
        include_social_tests=include_social_tests,
    ):
        branch = posterior.branch_estimate(test_sequence, node=node, test_type=test_type)
        if best_branch is None or _is_better_branch(
            (node, test_type),
            branch,
            best_candidate,
            best_branch,
        ):
            best_candidate = (node, test_type)
            best_branch = branch

    if best_candidate is None or best_branch is None:
        raise ValueError("no available candidate tests")

    node, test_type = best_candidate
    return PolicyDecision(
        node=node,
        test_type=test_type,
        expected_uncertainty=best_branch.expected_uncertainty,
        positive_probability=best_branch.positive_probability,
        minimum_branch_ess=best_branch.minimum_branch_ess,
        is_stable=best_branch.minimum_branch_ess >= min_branch_ess,
    )


def select_contact_tracing_test(
    physical_graph: nx.Graph,
    posterior_estimate: WeightedPosteriorEstimate,
    test_sequence: tuple[TestRecord, ...] | list[TestRecord] = (),
    seeds: tuple[int, ...] | list[int] = (),
) -> PolicyDecision:
    """Select a posterior-only contact-tracing physical test.

    The baseline expands from nodes with observed positive physical tests. When
    there is no frontier, it tests the untested node with highest posterior
    infection uncertainty. It never inspects hidden infection states.
    """

    seed_set = set(seeds)
    tested_physical = {
        node for node, test_type, _ in test_sequence if int(test_type) == TestType.PHYSICAL
    }
    positive_physical = {
        node
        for node, test_type, result in test_sequence
        if int(test_type) == TestType.PHYSICAL and result == 1
    }

    frontier = set()
    for node in positive_physical:
        frontier.update(physical_graph.neighbors(node))
    frontier.difference_update(tested_physical)
    frontier.difference_update(seed_set)

    if frontier:
        node = _most_uncertain_node(frontier, posterior_estimate)
    else:
        candidates = set(physical_graph.nodes()) - tested_physical - seed_set
        if not candidates:
            raise ValueError("no available physical tests")
        node = _most_uncertain_node(candidates, posterior_estimate)

    return PolicyDecision(
        node=int(node),
        test_type=TestType.PHYSICAL,
        expected_uncertainty=posterior_estimate.physical_uncertainty,
        positive_probability=None,
        minimum_branch_ess=None,
        is_stable=True,
    )


def _most_uncertain_node(
    candidates: set[int],
    posterior_estimate: WeightedPosteriorEstimate,
) -> int:
    """Choose the highest-uncertainty candidate with deterministic tie-breaking."""

    return min(
        candidates,
        key=lambda node: (
            -bernoulli_variance(posterior_estimate.physical_posteriors[node]),
            node,
        ),
    )


def _is_better_branch(
    candidate: tuple[int, TestType],
    branch: WeightedBranchEstimate,
    best_candidate: tuple[int, TestType] | None,
    best_branch: WeightedBranchEstimate,
) -> bool:
    """Compare branches with deterministic tie-breaking."""

    if branch.expected_uncertainty != best_branch.expected_uncertainty:
        return branch.expected_uncertainty < best_branch.expected_uncertainty
    if branch.minimum_branch_ess != best_branch.minimum_branch_ess:
        return branch.minimum_branch_ess > best_branch.minimum_branch_ess
    if best_candidate is None:
        return True
    return (candidate[0], int(candidate[1])) < (best_candidate[0], int(best_candidate[1]))
