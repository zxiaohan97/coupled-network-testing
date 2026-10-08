"""Spanning-tree approximation experiment.

This script is a compact public version of the tree approximation experiment in
``Tree/Spanning_Tree.py``. It compares a cyclic ER graph with a BFS spanning
tree by running the same sample-based greedy policy on both graphs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class CascadeSamples:
    """Precomputed cascade states and noisy observations."""

    physical_state: np.ndarray
    social_state: np.ndarray
    physical_observation: np.ndarray
    social_observation: np.ndarray


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated list such as ``0.04,0.08,0.12``."""

    return [float(item.strip()) for item in value.split(",") if item.strip()]


def cyclomatic_number(graph: nx.Graph) -> int:
    """Return the number of independent cycles."""

    return graph.number_of_edges() - graph.number_of_nodes() + nx.number_connected_components(graph)


def largest_component_with_relabeling(graph: nx.Graph) -> nx.Graph:
    """Use the largest connected component and relabel nodes to ``0..n-1``."""

    component_nodes = max(nx.connected_components(graph), key=len)
    component = graph.subgraph(component_nodes).copy()
    return nx.convert_node_labels_to_integers(component)


def bfs_spanning_tree(graph: nx.Graph, root: int) -> nx.Graph:
    """Create an undirected BFS spanning tree rooted at ``root``."""

    tree = nx.Graph(nx.bfs_tree(graph, source=root))
    tree.add_nodes_from(graph.nodes())
    return tree


def sample_social_states(graph: nx.Graph, r: float, rng: np.random.Generator) -> np.ndarray:
    """Sample correlated social states by retaining social edges with probability ``r``."""

    retained = nx.Graph()
    retained.add_nodes_from(graph.nodes())
    for edge in graph.edges():
        if rng.random() < r:
            retained.add_edge(*edge)

    social = np.zeros(graph.number_of_nodes(), dtype=int)
    for component in nx.connected_components(retained):
        label = int(rng.integers(2))
        for node in component:
            social[node] = label
    return social


def sample_physical_states(
    graph: nx.Graph,
    social: np.ndarray,
    seed: int,
    p: float,
    q: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample an independent-cascade disease realization on a general graph."""

    infected = np.zeros(graph.number_of_nodes(), dtype=int)
    infected[seed] = 1
    active = [seed]

    while active:
        parent = active.pop(0)
        for child in graph.neighbors(parent):
            if infected[child] == 1:
                continue
            infection_prob = p if social[child] == 1 else q
            if rng.random() < infection_prob:
                infected[child] = 1
                active.append(child)

    return infected


def draw_samples(
    graph: nx.Graph,
    *,
    seed_node: int,
    p: float,
    q: float,
    r: float,
    physical_error: float,
    social_error: float,
    n_samples: int,
    rng: np.random.Generator,
) -> CascadeSamples:
    """Generate cascade states and all possible noisy node observations."""

    n_nodes = graph.number_of_nodes()
    physical_state = np.zeros((n_samples, n_nodes), dtype=int)
    social_state = np.zeros((n_samples, n_nodes), dtype=int)

    for sample_idx in range(n_samples):
        social = sample_social_states(graph, r, rng)
        physical = sample_physical_states(graph, social, seed_node, p, q, rng)
        social_state[sample_idx] = social
        physical_state[sample_idx] = physical

    physical_flip = rng.random(size=physical_state.shape) < physical_error
    social_flip = rng.random(size=social_state.shape) < social_error
    physical_observation = np.where(physical_flip, 1 - physical_state, physical_state)
    social_observation = np.where(social_flip, 1 - social_state, social_state)

    return CascadeSamples(
        physical_state=physical_state,
        social_state=social_state,
        physical_observation=physical_observation,
        social_observation=social_observation,
    )


def posterior_uncertainty(samples: CascadeSamples, sample_indices: np.ndarray) -> float:
    """Estimate average disease posterior variance from valid samples."""

    if len(sample_indices) == 0:
        return 0.0
    infection_probs = samples.physical_state[sample_indices].mean(axis=0)
    return float(np.mean(infection_probs * (1.0 - infection_probs)))


def observations_for(samples: CascadeSamples, node: int, test_type: int) -> np.ndarray:
    """Return precomputed noisy observations for one candidate test."""

    if test_type == 1:
        return samples.physical_observation[:, node]
    return samples.social_observation[:, node]


def expected_uncertainty_after_test(
    samples: CascadeSamples,
    sample_indices: np.ndarray,
    node: int,
    test_type: int,
) -> float:
    """Evaluate the sample-based expected uncertainty for one candidate test."""

    observations = observations_for(samples, node, test_type)[sample_indices]
    expected = 0.0
    for result in (0, 1):
        branch_indices = sample_indices[observations == result]
        if len(branch_indices) == 0:
            continue
        branch_probability = len(branch_indices) / len(sample_indices)
        expected += branch_probability * posterior_uncertainty(samples, branch_indices)
    return expected


def select_greedy_test(samples: CascadeSamples, sample_indices: np.ndarray) -> tuple[int, int]:
    """Select the physical or social test with minimum expected uncertainty."""

    n_nodes = samples.physical_state.shape[1]
    best_test = (0, 1)
    best_uncertainty = float("inf")
    for node in range(n_nodes):
        for test_type in (0, 1):
            expected = expected_uncertainty_after_test(samples, sample_indices, node, test_type)
            if expected < best_uncertainty:
                best_uncertainty = expected
                best_test = (node, test_type)
    return best_test


def run_sample_greedy_policy(samples: CascadeSamples, budget: int) -> list[float]:
    """Run the sample-based greedy policy while averaging over both outcomes."""

    initial_indices = np.arange(samples.physical_state.shape[0])
    branches = [(initial_indices, 1.0)]
    uncertainties = [posterior_uncertainty(samples, initial_indices)]

    for _ in range(budget):
        next_branches = []
        averaged_uncertainty = 0.0
        for sample_indices, branch_prob in branches:
            node, test_type = select_greedy_test(samples, sample_indices)
            observations = observations_for(samples, node, test_type)[sample_indices]
            for result in (0, 1):
                child_indices = sample_indices[observations == result]
                if len(child_indices) == 0:
                    continue
                child_prob = branch_prob * len(child_indices) / len(sample_indices)
                next_branches.append((child_indices, child_prob))
                averaged_uncertainty += child_prob * posterior_uncertainty(samples, child_indices)
        branches = next_branches
        uncertainties.append(averaged_uncertainty)

    return uncertainties


def run_spanning_tree_experiment(
    *,
    n_nodes: int,
    edge_probabilities: list[float],
    n_graphs: int,
    n_samples: int,
    budget: int,
    p: float,
    q: float,
    r: float,
    physical_error: float,
    social_error: float,
    random_seed: int,
) -> list[dict[str, float]]:
    """Compare ER graphs against their BFS spanning trees."""

    rng = np.random.default_rng(random_seed)
    rows = []

    for edge_probability in edge_probabilities:
        relative_differences = []
        cycle_counts = []
        component_sizes = []
        edge_counts = []

        for _ in range(n_graphs):
            graph_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            graph = nx.erdos_renyi_graph(n_nodes, edge_probability, seed=graph_seed)
            if graph.number_of_edges() == 0:
                continue
            graph = largest_component_with_relabeling(graph)
            if graph.number_of_nodes() < 3:
                continue

            root = max(graph.degree(), key=lambda item: item[1])[0]
            tree = bfs_spanning_tree(graph, root)
            cycle_counts.append(cyclomatic_number(graph))
            component_sizes.append(graph.number_of_nodes())
            edge_counts.append(graph.number_of_edges())

            graph_samples = draw_samples(
                graph,
                seed_node=root,
                p=p,
                q=q,
                r=r,
                physical_error=physical_error,
                social_error=social_error,
                n_samples=n_samples,
                rng=rng,
            )
            tree_samples = draw_samples(
                tree,
                seed_node=root,
                p=p,
                q=q,
                r=r,
                physical_error=physical_error,
                social_error=social_error,
                n_samples=n_samples,
                rng=rng,
            )

            graph_uncertainty = run_sample_greedy_policy(graph_samples, budget)[-1]
            tree_uncertainty = run_sample_greedy_policy(tree_samples, budget)[-1]
            if graph_uncertainty > 0:
                relative_differences.append(
                    abs(tree_uncertainty - graph_uncertainty) / graph_uncertainty
                )

        relative_array = np.array(relative_differences, dtype=float)
        cycle_array = np.array(cycle_counts, dtype=float)
        component_array = np.array(component_sizes, dtype=float)
        edge_array = np.array(edge_counts, dtype=float)
        relative_std = (
            float(np.std(relative_array, ddof=1))
            if len(relative_array) > 1
            else float("nan")
        )

        rows.append(
            {
                "n_nodes": n_nodes,
                "edge_probability": edge_probability,
                "n_graphs": n_graphs,
                "n_samples": n_samples,
                "budget": budget,
                "p": p,
                "q": q,
                "r": r,
                "physical_error": physical_error,
                "social_error": social_error,
                "num_valid_graphs": len(relative_differences),
                "mean_relative_difference": (
                    float(np.mean(relative_array))
                    if len(relative_array) > 0
                    else float("nan")
                ),
                "median_relative_difference": (
                    float(np.median(relative_array))
                    if len(relative_array) > 0
                    else float("nan")
                ),
                "std_relative_difference": relative_std,
                "stderr_relative_difference": (
                    relative_std / float(np.sqrt(len(relative_array)))
                    if len(relative_array) > 1
                    else float("nan")
                ),
                "mean_cyclomatic_number": (
                    float(np.mean(cycle_array)) if len(cycle_array) > 0 else float("nan")
                ),
                "mean_component_nodes": (
                    float(np.mean(component_array))
                    if len(component_array) > 0
                    else float("nan")
                ),
                "mean_edge_count": (
                    float(np.mean(edge_array)) if len(edge_array) > 0 else float("nan")
                ),
            }
        )

    return rows


def write_rows(path: Path, rows: list[dict[str, float]]) -> None:
    """Write experiment rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-nodes", type=int, default=16)
    parser.add_argument(
        "--edge-probabilities",
        type=parse_float_list,
        default=parse_float_list("0.08,0.12,0.16,0.20"),
    )
    parser.add_argument("--n-graphs", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--q", type=float, default=0.2)
    parser.add_argument("--r", type=float, default=0.5)
    parser.add_argument("--physical-error", type=float, default=0.15)
    parser.add_argument("--social-error", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/spanning_tree_approximation.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_spanning_tree_experiment(
        n_nodes=args.n_nodes,
        edge_probabilities=args.edge_probabilities,
        n_graphs=args.n_graphs,
        n_samples=args.n_samples,
        budget=args.budget,
        p=args.p,
        q=args.q,
        r=args.r,
        physical_error=args.physical_error,
        social_error=args.social_error,
        random_seed=args.random_seed,
    )
    write_rows(args.output, rows)
    for row in rows:
        print(row)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
