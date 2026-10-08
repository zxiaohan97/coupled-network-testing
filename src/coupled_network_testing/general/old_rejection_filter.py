"""Old hard-rejection Monte Carlo posterior updates.

This module preserves the first general-graph posterior approximation from the
original research scripts. It conditions simulated worlds by rejecting every
sample whose pre-sampled noisy test result does not exactly match the observed
history. That baseline is useful for comparison, but it can suffer severe
sample collapse on long adaptive test sequences.

New general-graph code should usually prefer
``coupled_network_testing.general.weighted_particle_filter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from coupled_network_testing.common.metrics import bernoulli_variance
from coupled_network_testing.common.testing_types import TestType
from coupled_network_testing.general.simulation_model import (
    CascadeRealization,
    TestRecord,
)

RealizationSequence: TypeAlias = list[CascadeRealization] | tuple[CascadeRealization, ...]


@dataclass(frozen=True)
class PosteriorEstimate:
    """Monte Carlo estimate of posterior node probabilities."""

    physical_posteriors: dict[int, float]
    social_posteriors: dict[int, float]
    physical_uncertainty: float
    n_samples: int


@dataclass(frozen=True)
class OutcomeSplit:
    """Samples split by the two possible outcomes of one candidate test."""

    positive: list[CascadeRealization]
    negative: list[CascadeRealization]

    @property
    def n_total(self) -> int:
        return len(self.positive) + len(self.negative)

    @property
    def positive_probability(self) -> float:
        if self.n_total == 0:
            return 0.0
        return len(self.positive) / self.n_total


def filter_realizations(
    realizations: RealizationSequence,
    test_sequence: tuple[TestRecord, ...] | list[TestRecord],
) -> list[CascadeRealization]:
    """Keep sampled worlds consistent with all observed tests.

    This is the core approximation used in the original general-graph scripts:
    samples play the role of posterior particles, and conditioning is performed
    by rejecting samples whose pre-sampled test results disagree with the
    observation history.
    """

    current = list(realizations)
    for node, test_type, observed_result in test_sequence:
        current = [
            realization
            for realization in current
            if realization.test_result(node, test_type) == observed_result
        ]
        if not current:
            break
    return current


def split_by_test_outcome(
    realizations: RealizationSequence,
    node: int,
    test_type: int | TestType,
) -> OutcomeSplit:
    """Split samples according to the positive/negative result of one test."""

    positive: list[CascadeRealization] = []
    negative: list[CascadeRealization] = []
    for realization in realizations:
        if realization.test_result(node, test_type) == 1:
            positive.append(realization)
        else:
            negative.append(realization)
    return OutcomeSplit(positive=positive, negative=negative)


def estimate_posteriors(
    realizations: RealizationSequence,
    nodes: tuple[int, ...] | list[int] | None = None,
) -> PosteriorEstimate:
    """Estimate disease and social posteriors from retained samples."""

    if not realizations:
        return PosteriorEstimate(
            physical_posteriors={},
            social_posteriors={},
            physical_uncertainty=0.0,
            n_samples=0,
        )

    if nodes is None:
        try:
            nodes = tuple(sorted(realizations[0].physical_states))
        except TypeError:
            nodes = tuple(realizations[0].physical_states)

    physical_matrix = np.array(
        [[realization.physical_states[node] for node in nodes] for realization in realizations],
        dtype=float,
    )
    social_matrix = np.array(
        [[realization.social_states[node] for node in nodes] for realization in realizations],
        dtype=float,
    )
    physical_means = np.mean(physical_matrix, axis=0)
    social_means = np.mean(social_matrix, axis=0)

    physical_posteriors = dict(zip(nodes, physical_means, strict=True))
    social_posteriors = dict(zip(nodes, social_means, strict=True))
    physical_uncertainty = float(np.mean([bernoulli_variance(p) for p in physical_means]))

    return PosteriorEstimate(
        physical_posteriors=physical_posteriors,
        social_posteriors=social_posteriors,
        physical_uncertainty=physical_uncertainty,
        n_samples=len(realizations),
    )


def expected_uncertainty_after_test(
    realizations: RealizationSequence,
    node: int,
    test_type: int | TestType,
    nodes: tuple[int, ...] | list[int] | None = None,
) -> float:
    """Estimate expected posterior uncertainty after one candidate test."""

    split = split_by_test_outcome(realizations, node=node, test_type=test_type)
    if split.n_total == 0:
        return 0.0

    positive_uncertainty = (
        estimate_posteriors(split.positive, nodes=nodes).physical_uncertainty
        if split.positive
        else 0.0
    )
    negative_uncertainty = (
        estimate_posteriors(split.negative, nodes=nodes).physical_uncertainty
        if split.negative
        else 0.0
    )
    p_positive = split.positive_probability
    return p_positive * positive_uncertainty + (1.0 - p_positive) * negative_uncertainty


class OldRejectionMonteCarloPosterior:
    """Cache old hard-rejection estimates for many test histories."""

    def __init__(
        self,
        realizations: RealizationSequence,
        nodes: tuple[int, ...] | list[int] | None = None,
    ):
        self.realizations = list(realizations)
        if nodes is None and self.realizations:
            try:
                nodes = tuple(sorted(self.realizations[0].physical_states))
            except TypeError:
                nodes = tuple(self.realizations[0].physical_states)
        self.nodes = tuple(nodes or ())
        self._filtered_cache: dict[tuple[TestRecord, ...], list[CascadeRealization]] = {
            (): self.realizations
        }
        self._posterior_cache: dict[tuple[TestRecord, ...], PosteriorEstimate] = {}

    def filter(
        self,
        test_sequence: tuple[TestRecord, ...] | list[TestRecord],
    ) -> list[CascadeRealization]:
        """Return samples matching ``test_sequence``, using cached prefixes."""

        key = tuple(test_sequence)
        if key in self._filtered_cache:
            return self._filtered_cache[key]

        prefix = key[:-1]
        base = self.filter(prefix)
        self._filtered_cache[key] = filter_realizations(base, key[-1:])
        return self._filtered_cache[key]

    def estimate(
        self,
        test_sequence: tuple[TestRecord, ...] | list[TestRecord] = (),
    ) -> PosteriorEstimate:
        """Estimate posteriors after a sequence of observed tests."""

        key = tuple(test_sequence)
        if key not in self._posterior_cache:
            self._posterior_cache[key] = estimate_posteriors(
                self.filter(key),
                nodes=self.nodes,
            )
        return self._posterior_cache[key]

    def expected_uncertainty(
        self,
        test_sequence: tuple[TestRecord, ...] | list[TestRecord],
        node: int,
        test_type: int | TestType,
    ) -> float:
        """Estimate expected uncertainty of testing ``node`` next."""

        return expected_uncertainty_after_test(
            self.filter(test_sequence),
            node=node,
            test_type=test_type,
            nodes=self.nodes,
        )


# Backward-compatible name used by the first packaged version.
MonteCarloPosterior = OldRejectionMonteCarloPosterior
