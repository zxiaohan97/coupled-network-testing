"""Random-tree budget-gain experiment.

This is a small, reproducible public version of the original
``Tree/Tree_Budget_Gain.py`` experiment. The manuscript-scale run used many
more trees and a wider parameter grid; the defaults here are intentionally small.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

import networkx as nx
import numpy as np

from coupled_network_testing.tree import ExactTreePosteriorModel, ModelParameters
from coupled_network_testing.tree.policies import analyze_threshold_times


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated list such as ``0,0.5,1``."""

    return [float(item.strip()) for item in value.split(",") if item.strip()]


def random_tree_edges(n_nodes: int, seed: int) -> list[tuple[int, int]]:
    """Generate a random labeled tree using the current NetworkX API."""

    graph = nx.generators.trees.random_labeled_tree(n_nodes, seed=seed)
    return list(graph.edges())


def finite_budget_gain(greedy_budget: float, contact_budget: float) -> float | None:
    """Return budget gain when both policies reach the target threshold."""

    if not np.isfinite(greedy_budget) or not np.isfinite(contact_budget):
        return None
    if contact_budget <= 0:
        return None
    return (contact_budget - greedy_budget) / contact_budget


def run_budget_gain_experiment(
    *,
    n_nodes: int,
    n_trees: int,
    p: float,
    q: float,
    r_values: Iterable[float],
    physical_error: float,
    social_errors: Iterable[float],
    threshold: float,
    max_budget: int,
    random_seed: int,
) -> list[dict[str, float]]:
    """Estimate average budget gain over random trees."""

    rows: list[dict[str, float]] = []
    tree_seeds = np.random.default_rng(random_seed).integers(
        0,
        np.iinfo(np.int32).max,
        size=n_trees,
    )

    for social_error in social_errors:
        for social_correlation in r_values:
            gains = []
            greedy_budgets = []
            contact_budgets = []

            for tree_seed in tree_seeds:
                params = ModelParameters(
                    p_infect_careless=p,
                    q_infect_careful=q,
                    social_correlation=social_correlation,
                    physical_error=physical_error,
                    social_error=social_error,
                )
                model = ExactTreePosteriorModel.from_edges(
                    random_tree_edges(n_nodes, seed=int(tree_seed)),
                    seed=0,
                    parameters=params,
                )
                threshold_times = analyze_threshold_times(
                    model,
                    thresholds=[threshold],
                    max_budget=max_budget,
                )
                greedy_budget = threshold_times["greedy"][threshold]
                contact_budget = threshold_times["contact_tracing"][threshold]
                gain = finite_budget_gain(greedy_budget, contact_budget)
                if gain is None:
                    continue

                gains.append(gain)
                greedy_budgets.append(greedy_budget)
                contact_budgets.append(contact_budget)

            rows.append(
                {
                    "n_nodes": n_nodes,
                    "n_trees": n_trees,
                    "p": p,
                    "q": q,
                    "r": social_correlation,
                    "physical_error": physical_error,
                    "social_error": social_error,
                    "threshold": threshold,
                    "max_budget": max_budget,
                    "num_reached": len(gains),
                    "mean_budget_gain": float(np.mean(gains)) if gains else float("nan"),
                    "std_budget_gain": float(np.std(gains)) if gains else float("nan"),
                    "mean_greedy_budget": (
                        float(np.mean(greedy_budgets)) if greedy_budgets else float("nan")
                    ),
                    "mean_contact_budget": (
                        float(np.mean(contact_budgets)) if contact_budgets else float("nan")
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
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--n-trees", type=int, default=5)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--q", type=float, default=0.2)
    parser.add_argument("--r-values", type=parse_float_list, default=parse_float_list("0,0.5,1"))
    parser.add_argument("--physical-error", type=float, default=0.15)
    parser.add_argument("--social-errors", type=parse_float_list, default=parse_float_list("0.1"))
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--max-budget", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("outputs/tree_budget_gain.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_budget_gain_experiment(
        n_nodes=args.n_nodes,
        n_trees=args.n_trees,
        p=args.p,
        q=args.q,
        r_values=args.r_values,
        physical_error=args.physical_error,
        social_errors=args.social_errors,
        threshold=args.threshold,
        max_budget=args.max_budget,
        random_seed=args.random_seed,
    )
    write_rows(args.output, rows)
    for row in rows:
        print(row)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
