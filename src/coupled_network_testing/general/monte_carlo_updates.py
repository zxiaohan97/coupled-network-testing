"""Compatibility imports for the old hard-rejection Monte Carlo estimator.

Prefer importing old baseline code from ``old_rejection_filter`` and improved
posterior code from ``weighted_particle_filter``. This module remains only so
early package examples and notebooks using ``monte_carlo_updates`` do not break.
"""

from coupled_network_testing.general.old_rejection_filter import (
    MonteCarloPosterior,
    OldRejectionMonteCarloPosterior,
    OutcomeSplit,
    PosteriorEstimate,
    estimate_posteriors,
    expected_uncertainty_after_test,
    filter_realizations,
    split_by_test_outcome,
)

__all__ = [
    "MonteCarloPosterior",
    "OldRejectionMonteCarloPosterior",
    "OutcomeSplit",
    "PosteriorEstimate",
    "estimate_posteriors",
    "expected_uncertainty_after_test",
    "filter_realizations",
    "split_by_test_outcome",
]
