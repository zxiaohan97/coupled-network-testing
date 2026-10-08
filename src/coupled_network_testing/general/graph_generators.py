"""Graph generators for simulation-based general-network experiments.

The original workspace generated ER, Barabasi-Albert/scale-free, and
Watts-Strogatz graphs in several separate scripts. This module keeps the common
behavior in one place and makes randomness explicit through a
``numpy.random.Generator``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

import networkx as nx
import numpy as np


class SeedSelection(StrEnum):
    """Supported infection-seed choices for public experiments."""

    RANDOM = "random"
    HIGHEST_DEGREE = "highest_degree"


@dataclass(frozen=True)
class GeneratedNetworkPair:
    """Physical and social graphs used by one coupled-network experiment."""

    physical: nx.Graph
    social: nx.Graph


def _networkx_seed(rng: np.random.Generator) -> int:
    """Return an integer seed suitable for NetworkX graph generators."""

    return int(rng.integers(0, np.iinfo(np.int32).max))


def generate_er_graph(
    n_nodes: int,
    edge_probability: float,
    rng: np.random.Generator,
) -> nx.Graph:
    """Generate an Erdos-Renyi graph with nodes relabeled to ``0..n-1``."""

    if n_nodes < 1:
        raise ValueError("n_nodes must be positive")
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("edge_probability must be between 0 and 1")

    graph = nx.erdos_renyi_graph(
        n=n_nodes,
        p=edge_probability,
        seed=_networkx_seed(rng),
    )
    return nx.convert_node_labels_to_integers(graph)


def calculate_scale_free_attachment_degree(n_nodes: int, edge_probability: float) -> int:
    """Approximate the BA attachment degree matching an ER edge density.

    This is the cleaned version of ``calculate_sf_degree`` in the original
    scripts. A Barabasi-Albert graph has about ``m * n`` edges, so this solves
    the corresponding quadratic approximation and rounds upward.
    """

    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("edge_probability must be between 0 and 1")

    target_edges = n_nodes * (n_nodes - 1) * edge_probability / 2.0
    if target_edges <= 0:
        return 1

    # Original derivation: -m^2/2 - m/2 + mn = target_edges.
    a = -0.5
    b = n_nodes - 0.5
    c = -target_edges
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("no real BA attachment degree matches this density")

    roots = [(-b + np.sqrt(discriminant)) / (2 * a), (-b - np.sqrt(discriminant)) / (2 * a)]
    degree = int(np.ceil(min(root for root in roots if root > 0)))
    return max(1, min(degree, n_nodes - 1))


def generate_scale_free_graph(
    n_nodes: int,
    edge_probability: float,
    rng: np.random.Generator,
) -> nx.Graph:
    """Generate a BA graph with edge density roughly comparable to ER."""

    attachment_degree = calculate_scale_free_attachment_degree(n_nodes, edge_probability)
    if attachment_degree == 1:
        graph = nx.barabasi_albert_graph(
            n_nodes,
            attachment_degree,
            seed=_networkx_seed(rng),
        )
    else:
        graph = nx.barabasi_albert_graph(
            n_nodes,
            attachment_degree,
            initial_graph=nx.complete_graph(attachment_degree),
            seed=_networkx_seed(rng),
        )
    return _trim_edges_to_expected_density(graph, edge_probability, rng)


def generate_watts_strogatz_graph(
    n_nodes: int,
    edge_probability: float,
    rewiring_probability: float,
    rng: np.random.Generator,
) -> nx.Graph:
    """Generate a WS graph with edge density roughly comparable to ER."""

    if not 0.0 <= rewiring_probability <= 1.0:
        raise ValueError("rewiring_probability must be between 0 and 1")

    expected_edges = n_nodes * (n_nodes - 1) * edge_probability / 2.0
    degree = int(np.ceil(2 * expected_edges / n_nodes))
    degree = max(2, min(degree, n_nodes - 1))
    if degree % 2 == 1:
        degree += 1
    degree = min(degree, n_nodes - 1 if (n_nodes - 1) % 2 == 0 else n_nodes - 2)

    graph = nx.watts_strogatz_graph(
        n=n_nodes,
        k=degree,
        p=rewiring_probability,
        seed=_networkx_seed(rng),
    )
    return _trim_edges_to_expected_density(graph, edge_probability, rng)


def _trim_edges_to_expected_density(
    graph: nx.Graph,
    edge_probability: float,
    rng: np.random.Generator,
) -> nx.Graph:
    """Randomly remove surplus edges to match the target ER expected count."""

    result = nx.convert_node_labels_to_integers(graph.copy())
    expected_edges = result.number_of_nodes() * (result.number_of_nodes() - 1)
    expected_edges *= edge_probability / 2.0
    surplus = result.number_of_edges() - int(expected_edges)
    if surplus <= 0:
        return result

    edges = list(result.edges())
    remove_indices = rng.choice(len(edges), size=surplus, replace=False)
    result.remove_edges_from(edges[index] for index in remove_indices)
    return result


def split_overlapping_subgraphs(
    graph: nx.Graph,
    overlap: float,
    rng: np.random.Generator,
) -> GeneratedNetworkPair:
    """Split one graph into physical/social layers with controlled overlap.

    Each layer receives half of the original edges. A fraction ``overlap`` of
    those layer edges is shared by both graphs. This preserves the original
    experiment convention while keeping the raw node set identical.
    """

    if not 0.0 <= overlap <= 1.0:
        raise ValueError("overlap must be between 0 and 1")

    edges = list(graph.edges())
    rng.shuffle(edges)
    edges_per_graph = len(edges) // 2
    n_shared = int(overlap * edges_per_graph)
    n_unique = edges_per_graph - n_shared

    shared_edges = edges[:n_shared]
    physical_unique = edges[n_shared : n_shared + n_unique]
    social_unique = edges[n_shared + n_unique : n_shared + 2 * n_unique]

    physical = nx.Graph()
    social = nx.Graph()
    physical.add_nodes_from(graph.nodes())
    social.add_nodes_from(graph.nodes())
    physical.add_edges_from(shared_edges + physical_unique)
    social.add_edges_from(shared_edges + social_unique)
    return GeneratedNetworkPair(physical=physical, social=social)


def select_seed_nodes(
    graph: nx.Graph,
    n_seeds: int,
    rng: np.random.Generator,
    mode: SeedSelection | str = SeedSelection.RANDOM,
) -> tuple[int, ...]:
    """Select infection seeds without using hidden disease states."""

    if n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    if n_seeds > graph.number_of_nodes():
        raise ValueError("n_seeds cannot exceed the number of graph nodes")

    mode = SeedSelection(mode)
    nodes = list(graph.nodes())
    if mode is SeedSelection.RANDOM:
        return tuple(int(node) for node in rng.choice(nodes, size=n_seeds, replace=False))

    if mode is SeedSelection.HIGHEST_DEGREE:
        ranked_nodes = sorted(graph.degree(), key=lambda item: (-item[1], item[0]))
        return tuple(int(node) for node, _ in ranked_nodes[:n_seeds])

    raise ValueError(f"unsupported seed selection mode: {mode}")


def make_coupled_network_pair(
    graph: nx.Graph,
    rng: np.random.Generator,
    overlap: float | None = None,
) -> GeneratedNetworkPair:
    """Return identical or overlap-controlled physical/social graph layers."""

    if overlap is None:
        return GeneratedNetworkPair(physical=graph.copy(), social=graph.copy())
    return split_overlapping_subgraphs(graph, overlap=overlap, rng=rng)


def relabel_to_contiguous_integers(graph: nx.Graph, nodes: Iterable[int] | None = None) -> nx.Graph:
    """Return a copy whose nodes are contiguous integers.

    The simulation code supports arbitrary NetworkX node labels, but the public
    examples use contiguous integers because they are easier to read in outputs.
    """

    result = nx.convert_node_labels_to_integers(graph)
    if nodes is not None and result.number_of_nodes() != len(tuple(nodes)):
        raise ValueError("node relabeling changed the expected node count")
    return result
