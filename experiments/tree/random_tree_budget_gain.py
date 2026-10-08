"""Random-tree budget-gain experiment.

This is a small, reproducible public version of the original
``Tree/Tree_Budget_Gain.py`` experiment. The manuscript-scale run used many
more trees and a wider parameter grid. The public sweep pairs policies on the
same trees and records unreached targets instead of silently discarding them.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import t as student_t

from coupled_network_testing.tree import ExactTreePosteriorModel, ModelParameters
from coupled_network_testing.tree.policies import analyze_threshold_times

ExperimentRow = dict[str, int | float]


@dataclass(frozen=True)
class _TreeTrial:
    n_nodes: int
    tree_index: int
    tree_seed: int
    parameters: ModelParameters
    thresholds: tuple[float, ...]
    max_budget: int


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


def _evaluate_tree(trial: _TreeTrial) -> list[ExperimentRow]:
    """Evaluate both policies and every target on one fixed tree."""

    model = ExactTreePosteriorModel.from_edges(
        random_tree_edges(trial.n_nodes, seed=trial.tree_seed),
        seed=0,
        parameters=trial.parameters,
    )
    initial_uncertainty = model.get_physical_uncertainty()
    times = analyze_threshold_times(model, trial.thresholds, trial.max_budget)
    rows = []
    for threshold in trial.thresholds:
        greedy_budget = times["greedy"][threshold]
        contact_budget = times["contact_tracing"][threshold]
        gain = finite_budget_gain(greedy_budget, contact_budget)
        rows.append(
            {
                "n_nodes": trial.n_nodes,
                "tree_index": trial.tree_index,
                "tree_seed": trial.tree_seed,
                "p": trial.parameters.p_infect_careless,
                "q": trial.parameters.q_infect_careful,
                "r": trial.parameters.social_correlation,
                "physical_error": trial.parameters.physical_error,
                "social_error": trial.parameters.social_error,
                "threshold": threshold,
                "max_budget": trial.max_budget,
                "initial_uncertainty": initial_uncertainty,
                "greedy_budget": greedy_budget,
                "contact_budget": contact_budget,
                "budget_gain": float("nan") if gain is None else gain,
            },
        )
    return rows


def summarize_trials(trials: list[ExperimentRow]) -> list[ExperimentRow]:
    """Summarize paired gains and report censoring for each parameter setting.

    Confidence intervals describe variation across sampled trees, conditional
    on both policies reaching the target with a positive baseline budget. They
    do not turn unreached targets into observations or quantify model error.
    """

    grouped: defaultdict[tuple[float, float, float], list[ExperimentRow]] = defaultdict(list)
    for trial in trials:
        key = (trial["social_error"], trial["r"], trial["threshold"])
        grouped[key].append(trial)

    rows = []
    for group in grouped.values():
        first = group[0]
        greedy = np.array([trial["greedy_budget"] for trial in group], dtype=float)
        contact = np.array([trial["contact_budget"] for trial in group], dtype=float)
        gains = np.array([trial["budget_gain"] for trial in group], dtype=float)
        valid = np.isfinite(gains)
        n_valid = int(valid.sum())
        mean_gain = float(np.mean(gains[valid])) if n_valid else float("nan")
        # Use the sample standard deviation for an approximate Student-t CI.
        std_gain = float(np.std(gains[valid], ddof=1)) if n_valid > 1 else float("nan")
        standard_error = std_gain / np.sqrt(n_valid) if n_valid else float("nan")
        margin = (
            float(student_t.ppf(0.975, n_valid - 1)) * standard_error
            if n_valid > 1
            else float("nan")
        )
        greedy_reached = np.isfinite(greedy)
        contact_reached = np.isfinite(contact)
        rows.append(
            {
                **{
                    key: first[key]
                    for key in (
                        "n_nodes",
                        "p",
                        "q",
                        "r",
                        "physical_error",
                        "social_error",
                        "threshold",
                        "max_budget",
                        "random_seed",
                    )
                },
                "n_trees": len(group),
                "num_reached": n_valid,
                "num_both_reached": int((greedy_reached & contact_reached).sum()),
                "num_greedy_reached": int(greedy_reached.sum()),
                "num_contact_reached": int(contact_reached.sum()),
                "num_neither_reached": int((~greedy_reached & ~contact_reached).sum()),
                "num_initially_satisfied": int(((greedy == 0) & (contact == 0)).sum()),
                "greedy_reach_fraction": float(greedy_reached.mean()),
                "contact_reach_fraction": float(contact_reached.mean()),
                "mean_budget_gain": mean_gain,
                "std_budget_gain": std_gain,
                "se_budget_gain": float(standard_error),
                "ci95_low": mean_gain - margin,
                "ci95_high": mean_gain + margin,
                "mean_greedy_budget": float(greedy[valid].mean()) if n_valid else float("nan"),
                "mean_contact_budget": float(contact[valid].mean()) if n_valid else float("nan"),
            },
        )
    return rows


def run_budget_gain_experiment(
    *,
    n_nodes: int,
    n_trees: int,
    p: float,
    q: float,
    r_values: Iterable[float],
    physical_error: float,
    social_errors: Iterable[float],
    threshold: float | None = None,
    max_budget: int,
    random_seed: int,
    thresholds: Iterable[float] | None = None,
    workers: int = 1,
    details_output: Path | None = None,
    progress: bool = False,
) -> list[ExperimentRow]:
    """Estimate paired budget gain with reproducible seeds and reachability counts.

    ``threshold`` retains the single-target interface. ``thresholds`` evaluates
    multiple targets from the same policy traversal. Worker count changes only
    execution order, not graph generation or policy decisions.
    """

    if threshold is not None and thresholds is not None:
        raise ValueError("provide threshold or thresholds, not both")
    targets = (
        tuple(thresholds)
        if thresholds is not None
        else ((threshold,) if threshold is not None else (0.1, 0.05))
    )
    correlations = tuple(r_values)
    errors = tuple(social_errors)
    if n_nodes < 2 or n_trees < 1 or workers < 1 or max_budget < 0:
        raise ValueError("require n_nodes >= 2, n_trees/workers >= 1, and max_budget >= 0")
    if not targets or any(not 0.0 < target <= 0.25 for target in targets):
        raise ValueError("thresholds must be nonempty and in (0, 0.25]")
    if not correlations or not errors:
        raise ValueError("r_values and social_errors cannot be empty")
    if any(len(values) != len(set(values)) for values in (targets, correlations, errors)):
        raise ValueError("parameter grids cannot contain duplicates")
    tree_seeds = np.random.default_rng(random_seed).integers(
        0,
        np.iinfo(np.int32).max,
        size=n_trees,
    )

    # Reuse exactly the same sampled trees across every correlation and target.
    tasks = [
        _TreeTrial(
            n_nodes,
            index,
            int(tree_seed),
            ModelParameters(p, q, correlation, physical_error, social_error),
            targets,
            max_budget,
        )
        for social_error in errors
        for correlation in correlations
        for index, tree_seed in enumerate(tree_seeds)
    ]
    trials = []

    def collect(results: Iterable[list[ExperimentRow]]) -> None:
        for completed, group in enumerate(results, start=1):
            trials.extend({**row, "random_seed": random_seed} for row in group)
            if progress and (completed % n_trees == 0 or completed == len(tasks)):
                print(f"Completed {completed}/{len(tasks)} tree/parameter evaluations", flush=True)

    if workers == 1:
        collect(map(_evaluate_tree, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            collect(executor.map(_evaluate_tree, tasks))

    if details_output is not None:
        write_rows(details_output, trials)
    return summarize_trials(trials)


def write_rows(path: Path, rows: list[ExperimentRow]) -> None:
    """Write experiment rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-nodes", type=int, default=8)
    parser.add_argument("--n-trees", type=int, default=50)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--q", type=float, default=0.1)
    parser.add_argument(
        "--r-values",
        type=parse_float_list,
        default=parse_float_list("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"),
    )
    parser.add_argument("--physical-error", type=float, default=0.2)
    parser.add_argument("--social-errors", type=parse_float_list, default=parse_float_list("0.05"))
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--threshold", type=float, help="One uncertainty target")
    target_group.add_argument(
        "--thresholds",
        type=parse_float_list,
        help="Comma-separated targets (default: 0.1,0.05)",
    )
    parser.add_argument("--max-budget", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/tree_budget_gain.csv"))
    parser.add_argument("--details-output", type=Path, help="Optional per-tree CSV path")
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
        thresholds=args.thresholds,
        max_budget=args.max_budget,
        random_seed=args.random_seed,
        workers=args.workers,
        details_output=args.details_output
        or args.output.with_name(f"{args.output.stem}_trials.csv"),
        progress=True,
    )
    write_rows(args.output, rows)
    for row in rows:
        print(
            f"r={row['r']:.1f}, target={row['threshold']:g}: "
            f"gain={100 * row['mean_budget_gain']:.1f}%, "
            f"paired={row['num_reached']}/{row['n_trees']}, "
            f"greedy reached={row['num_greedy_reached']}, "
            f"contact reached={row['num_contact_reached']}",
        )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
