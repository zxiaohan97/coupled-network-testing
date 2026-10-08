"""Metrics for sequential testing experiments."""


def bernoulli_variance(probability: float) -> float:
    """Return p(1-p), the uncertainty of a Bernoulli posterior."""

    return probability * (1.0 - probability)
