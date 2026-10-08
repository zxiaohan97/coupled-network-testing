"""Fixed-tree budget-gain example.

This script gives a compact, readable case where social information helps at
moderate social correlation. It is an illustrative complement to the
random-tree budget-gain experiment, not a replacement for a broad random-tree
parameter sweep.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

from coupled_network_testing.tree import ExactTreePosteriorModel, ModelParameters
from coupled_network_testing.tree.policies import run_contact_tracing, run_greedy_testing

DEFAULT_EDGES = ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6))


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated list such as ``0.16,0.14,0.13``."""

    return [float(item.strip()) for item in value.split(",") if item.strip()]


def first_budget_at_or_below(uncertainties: list[float], threshold: float) -> int | None:
    """Return the first test budget where uncertainty reaches the threshold."""

    for budget, uncertainty in enumerate(uncertainties):
        if uncertainty <= threshold:
            return budget
    return None


def run_fixed_tree_budget_gain(
    *,
    budget: int,
    thresholds: Iterable[float],
    p: float,
    q: float,
    social_correlation: float,
    physical_error: float,
    social_error: float,
) -> list[dict[str, float | int | str | None]]:
    """Compute threshold budgets for greedy and contact tracing on one tree."""

    params = ModelParameters(
        p_infect_careless=p,
        q_infect_careful=q,
        social_correlation=social_correlation,
        physical_error=physical_error,
        social_error=social_error,
    )
    model = ExactTreePosteriorModel.from_edges(DEFAULT_EDGES, seed=0, parameters=params)
    _, greedy_uncertainties = run_greedy_testing(model, budget)

    model.reset_to_initial_state()
    model.store_current_state(())
    _, contact_uncertainties = run_contact_tracing(model, budget)

    rows: list[dict[str, float | int | str | None]] = []
    for threshold in thresholds:
        greedy_budget = first_budget_at_or_below(greedy_uncertainties, threshold)
        contact_budget = first_budget_at_or_below(contact_uncertainties, threshold)
        reached_by_both = greedy_budget is not None and contact_budget is not None
        budget_gain = None
        if reached_by_both and contact_budget and greedy_budget is not None:
            # Positive values mean greedy reached the same target in fewer tests.
            budget_gain = (contact_budget - greedy_budget) / contact_budget

        rows.append(
            {
                "threshold": threshold,
                "max_budget": budget,
                "n_nodes": len({node for edge in DEFAULT_EDGES for node in edge}),
                "p": p,
                "q": q,
                "r": social_correlation,
                "physical_error": physical_error,
                "social_error": social_error,
                "greedy_budget": greedy_budget,
                "contact_tracing_budget": contact_budget,
                "budget_gain": budget_gain,
                "reached_by_both": str(reached_by_both).lower(),
            }
        )

    return rows


def write_rows(path: Path, rows: list[dict[str, float | int | str | None]]) -> None:
    """Write fixed-tree budget-gain rows to CSV."""

    if not rows:
        raise ValueError("no fixed-tree budget-gain rows were generated")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument(
        "--thresholds",
        type=parse_float_list,
        default=parse_float_list("0.16,0.14,0.13"),
    )
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--q", type=float, default=0.1)
    parser.add_argument("--r", type=float, default=0.5)
    parser.add_argument("--physical-error", type=float, default=0.2)
    parser.add_argument("--social-error", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/fixed_tree_budget_gain.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_fixed_tree_budget_gain(
        budget=args.budget,
        thresholds=args.thresholds,
        p=args.p,
        q=args.q,
        social_correlation=args.r,
        physical_error=args.physical_error,
        social_error=args.social_error,
    )
    write_rows(args.output, rows)
    for row in rows:
        print(row)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
