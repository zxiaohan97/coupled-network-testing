"""Utilities for sequential testing in coupled opinion-disease networks."""

__version__ = "0.1.0"

from coupled_network_testing.tree.exact_model import (
    ModelParameters,
    TreeOpinionDiseaseModel,
    TreeStructure,
)
from coupled_network_testing.tree.posterior_updates import ExactTreePosteriorModel

__all__ = [
    "ExactTreePosteriorModel",
    "ModelParameters",
    "TreeOpinionDiseaseModel",
    "TreeStructure",
]
