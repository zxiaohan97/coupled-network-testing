"""Illustrative fixed-tree comparison of greedy and contact-tracing policies.

This experiment is intentionally descriptive. Greedy minimizes expected
one-step posterior uncertainty, so it is not a proof of global optimality and it
need not dominate contact tracing on every tree or parameter setting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from coupled_network_testing.tree import ExactTreePosteriorModel, ModelParameters
from coupled_network_testing.tree.policies import compare_strategies

DEFAULT_EDGES = ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6))


def run_fixed_tree_comparison(
    *,
    budget: int,
    p: float,
    q: float,
    social_correlation: float,
    physical_error: float,
    social_error: float,
) -> list[dict[str, float | int | str]]:
    """Return policy uncertainty trajectories on one fixed tree."""

    params = ModelParameters(
        p_infect_careless=p,
        q_infect_careful=q,
        social_correlation=social_correlation,
        physical_error=physical_error,
        social_error=social_error,
    )
    model = ExactTreePosteriorModel.from_edges(DEFAULT_EDGES, seed=0, parameters=params)
    comparison = compare_strategies(model, num_tests=budget)
    greedy = comparison["greedy"]["uncertainties"]
    contact = comparison["contact_tracing"]["uncertainties"]

    rows: list[dict[str, float | int | str]] = []
    for tests_used, (greedy_uncertainty, contact_uncertainty) in enumerate(
        zip(greedy, contact, strict=True),
    ):
        if greedy_uncertainty < contact_uncertainty:
            lower_uncertainty_policy = "greedy"
        elif contact_uncertainty < greedy_uncertainty:
            lower_uncertainty_policy = "contact_tracing"
        else:
            lower_uncertainty_policy = "tie"

        # Positive values mean greedy has lower average posterior disease
        # uncertainty; negative values honestly expose contact-tracing wins.
        relative_difference = (
            (contact_uncertainty - greedy_uncertainty) / contact_uncertainty
            if contact_uncertainty > 0
            else 0.0
        )
        rows.append(
            {
                "tests_used": tests_used,
                "n_nodes": len({node for edge in DEFAULT_EDGES for node in edge}),
                "p": p,
                "q": q,
                "r": social_correlation,
                "physical_error": physical_error,
                "social_error": social_error,
                "greedy_uncertainty": greedy_uncertainty,
                "contact_tracing_uncertainty": contact_uncertainty,
                "relative_difference": relative_difference,
                "lower_uncertainty_policy": lower_uncertainty_policy,
            }
        )

    return rows


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Write comparison rows to CSV."""

    if not rows:
        raise ValueError("no policy-comparison rows were generated")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--q", type=float, default=0.1)
    parser.add_argument("--r", type=float, default=0.5)
    parser.add_argument("--physical-error", type=float, default=0.2)
    parser.add_argument("--social-error", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/fixed_tree_policy_comparison.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_fixed_tree_comparison(
        budget=args.budget,
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
