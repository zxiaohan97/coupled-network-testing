"""Weighted particle posterior updates for general cyclic networks.

The original general-graph scripts condition Monte Carlo samples by hard
rejection: a sampled world is kept only if its pre-sampled noisy test result
matches the observed result. This module keeps the hidden worlds and updates
their weights by the likelihood of the observed tests instead. That makes every
sample contribute to the posterior, while effective sample size reports when
the weighted approximation is becoming fragile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from coupled_network_testing.common.metrics import bernoulli_variance
from coupled_network_testing.common.testing_types import TestType
from coupled_network_testing.general.simulation_model import (
    CascadeRealization,
    GeneralModelParameters,
    TestRecord,
)

ParticleSequence: TypeAlias = list[CascadeRealization] | tuple[CascadeRealization, ...]


@dataclass(frozen=True)
class WeightedPosteriorEstimate:
    """Posterior estimate with particle-filter diagnostics."""

    physical_posteriors: dict[int, float]
    social_posteriors: dict[int, float]
    physical_uncertainty: float
    n_particles: int
    effective_sample_size: float
    max_weight: float

    @property
    def ess_fraction(self) -> float:
        """Effective sample size as a fraction of the particle count."""

        if self.n_particles == 0:
            return 0.0
        return self.effective_sample_size / self.n_particles


@dataclass(frozen=True)
class WeightedBranchEstimate:
    """Expected uncertainty and diagnostics for one candidate test."""

    expected_uncertainty: float
    positive_probability: float
    positive: WeightedPosteriorEstimate
    negative: WeightedPosteriorEstimate

    @property
    def minimum_branch_ess(self) -> float:
        """Smallest ESS among the positive and negative hypothetical branches."""

        return min(
            self.positive.effective_sample_size,
            self.negative.effective_sample_size,
        )


def observation_likelihood(
    realization: CascadeRealization,
    test: TestRecord,
    parameters: GeneralModelParameters,
) -> float:
    """Return ``P(observed test result | hidden states in realization)``.

    This deliberately ignores the realization's pre-sampled noisy test arrays.
    The posterior should condition on the observed result's likelihood under
    the hidden state, not on whether one auxiliary noisy draw happened to match.
    """

    node, test_type, observed_result = test
    if int(test_type) == TestType.PHYSICAL:
        true_state = realization.physical_states[node]
        error_rate = parameters.physical_error
    else:
        true_state = realization.social_states[node]
        error_rate = parameters.social_error

    return (1.0 - error_rate) if true_state == observed_result else error_rate


def effective_sample_size(weights: np.ndarray) -> float:
    """Return the usual particle-filter effective sample size."""

    if weights.size == 0:
        return 0.0
    weight_sum = float(np.sum(weights))
    squared_sum = float(np.sum(np.square(weights)))
    if weight_sum <= 0.0 or squared_sum <= 0.0:
        return 0.0
    return weight_sum * weight_sum / squared_sum


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize nonnegative weights, raising when all mass is lost."""

    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("observed test sequence has zero likelihood under all particles")
    return weights / total


def estimate_weighted_posteriors(
    particles: ParticleSequence,
    weights: np.ndarray,
    nodes: tuple[int, ...] | list[int] | None = None,
) -> WeightedPosteriorEstimate:
    """Estimate disease and social posteriors from weighted particles."""

    if not particles:
        return WeightedPosteriorEstimate(
            physical_posteriors={},
            social_posteriors={},
            physical_uncertainty=0.0,
            n_particles=0,
            effective_sample_size=0.0,
            max_weight=0.0,
        )

    normalized = normalize_weights(np.asarray(weights, dtype=float))
    if normalized.shape != (len(particles),):
        raise ValueError("weights must have one entry per particle")

    if nodes is None:
        try:
            nodes = tuple(sorted(particles[0].physical_states))
        except TypeError:
            nodes = tuple(particles[0].physical_states)

    physical_posteriors: dict[int, float] = {}
    social_posteriors: dict[int, float] = {}
    for node in nodes:
        physical_posteriors[node] = float(
            np.dot(normalized, [particle.physical_states[node] for particle in particles])
        )
        social_posteriors[node] = float(
            np.dot(normalized, [particle.social_states[node] for particle in particles])
        )

    physical_uncertainty = float(
        np.mean([bernoulli_variance(p) for p in physical_posteriors.values()])
    )

    return WeightedPosteriorEstimate(
        physical_posteriors=physical_posteriors,
        social_posteriors=social_posteriors,
        physical_uncertainty=physical_uncertainty,
        n_particles=len(particles),
        effective_sample_size=effective_sample_size(normalized),
        max_weight=float(np.max(normalized)),
    )


def systematic_resample_indices(
    weights: np.ndarray,
    rng: np.random.Generator,
    n_particles: int | None = None,
) -> np.ndarray:
    """Draw particle indices by systematic resampling.

    This is provided for future sequential particle filtering/rejuvenation.
    The current posterior estimator can often avoid resampling entirely because
    the hidden cascade is static and full-sequence likelihood weighting is exact
    for the sampled particles.
    """

    normalized = normalize_weights(np.asarray(weights, dtype=float))
    n_draws = int(n_particles or len(normalized))
    positions = (rng.random() + np.arange(n_draws)) / n_draws
    cumulative = np.cumsum(normalized)
    return np.searchsorted(cumulative, positions, side="right")


class WeightedParticlePosterior:
    """Likelihood-weighted posterior cache for one fixed particle population."""

    def __init__(
        self,
        particles: ParticleSequence,
        parameters: GeneralModelParameters,
        nodes: tuple[int, ...] | list[int] | None = None,
    ):
        if not particles:
            raise ValueError("at least one particle is required")

        self.particles = list(particles)
        self.parameters = parameters
        if nodes is None:
            try:
                nodes = tuple(sorted(self.particles[0].physical_states))
            except TypeError:
                nodes = tuple(self.particles[0].physical_states)
        self.nodes = tuple(nodes)
        self._node_to_column = {node: index for index, node in enumerate(self.nodes)}
        self._physical_matrix = np.array(
            [
                [particle.physical_states[node] for node in self.nodes]
                for particle in self.particles
            ],
            dtype=float,
        )
        self._social_matrix = np.array(
            [
                [particle.social_states[node] for node in self.nodes]
                for particle in self.particles
            ],
            dtype=float,
        )

        initial_weights = np.full(len(self.particles), 1.0 / len(self.particles))
        self._weights_cache: dict[tuple[TestRecord, ...], np.ndarray] = {
            (): initial_weights
        }
        self._posterior_cache: dict[tuple[TestRecord, ...], WeightedPosteriorEstimate] = {}

    def weights(
        self,
        test_sequence: tuple[TestRecord, ...] | list[TestRecord] = (),
    ) -> np.ndarray:
        """Return normalized particle weights after an observed test sequence."""

        key = tuple(test_sequence)
        if key in self._weights_cache:
            return self._weights_cache[key].copy()

        prefix = key[:-1]
        weights = self.weights(prefix)
        likelihoods = self._likelihoods(key[-1])
        updated = normalize_weights(weights * likelihoods)
        self._weights_cache[key] = updated
        return updated.copy()

    def estimate(
        self,
        test_sequence: tuple[TestRecord, ...] | list[TestRecord] = (),
    ) -> WeightedPosteriorEstimate:
        """Estimate posteriors after a sequence of observed tests."""

        key = tuple(test_sequence)
        if key not in self._posterior_cache:
            self._posterior_cache[key] = self._estimate_from_weights(self.weights(key))
        return self._posterior_cache[key]

    def branch_estimate(
        self,
        test_sequence: tuple[TestRecord, ...] | list[TestRecord],
        node: int,
        test_type: int | TestType,
    ) -> WeightedBranchEstimate:
        """Estimate expected uncertainty for one candidate next test."""

        current_weights = self.weights(test_sequence)
        branches: dict[int, tuple[float, WeightedPosteriorEstimate]] = {}
        for result in (0, 1):
            test = (node, int(test_type), result)
            likelihoods = self._likelihoods(test)
            unnormalized = current_weights * likelihoods
            outcome_probability = float(np.sum(unnormalized))
            branch_weights = normalize_weights(unnormalized)
            branches[result] = (
                outcome_probability,
                self._estimate_from_weights(branch_weights),
            )

        negative_probability, negative = branches[0]
        positive_probability, positive = branches[1]
        probability_total = negative_probability + positive_probability
        if probability_total <= 0.0:
            raise ValueError("candidate test has zero probability under all particles")

        positive_probability /= probability_total
        negative_probability /= probability_total
        expected_uncertainty = (
            positive_probability * positive.physical_uncertainty
            + negative_probability * negative.physical_uncertainty
        )

        return WeightedBranchEstimate(
            expected_uncertainty=float(expected_uncertainty),
            positive_probability=float(positive_probability),
            positive=positive,
            negative=negative,
        )

    def _likelihoods(self, test: TestRecord) -> np.ndarray:
        """Vectorized likelihoods for one observed test across all particles."""

        node, test_type, observed_result = test
        column = self._node_to_column[node]
        if int(test_type) == TestType.PHYSICAL:
            states = self._physical_matrix[:, column]
            error_rate = self.parameters.physical_error
        else:
            states = self._social_matrix[:, column]
            error_rate = self.parameters.social_error
        return np.where(states == observed_result, 1.0 - error_rate, error_rate)

    def _estimate_from_weights(self, weights: np.ndarray) -> WeightedPosteriorEstimate:
        """Vectorized posterior estimate for this particle population."""

        normalized = normalize_weights(np.asarray(weights, dtype=float))
        physical_means = normalized @ self._physical_matrix
        social_means = normalized @ self._social_matrix
        physical_posteriors = dict(zip(self.nodes, physical_means, strict=True))
        social_posteriors = dict(zip(self.nodes, social_means, strict=True))
        physical_uncertainty = float(np.mean([bernoulli_variance(p) for p in physical_means]))
        return WeightedPosteriorEstimate(
            physical_posteriors=physical_posteriors,
            social_posteriors=social_posteriors,
            physical_uncertainty=physical_uncertainty,
            n_particles=len(self.particles),
            effective_sample_size=effective_sample_size(normalized),
            max_weight=float(np.max(normalized)),
        )
