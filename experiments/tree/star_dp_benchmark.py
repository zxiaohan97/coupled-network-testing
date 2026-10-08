"""Star-side exact-DP benchmark comparing greedy and optimal policies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from coupled_network_testing.benchmarks.star_dp import (
    StarBenchmarkConfig,
    run_star_tree_benchmark,
)
from coupled_network_testing.tree import ModelParameters


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated integer list such as ``3,4,5``."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def run_star_dp_experiment(
    *,
    n_nodes: int,
    budgets: list[int],
    p: float,
    q: float,
    r: float,
    physical_error: float,
    social_error: float,
) -> list[dict[str, float | int | str]]:
    """Evaluate greedy-vs-optimal gaps for a few star-side budgets."""

    rows: list[dict[str, float | int | str]] = []
    parameters = ModelParameters(
        p_infect_careless=p,
        q_infect_careful=q,
        social_correlation=r,
        physical_error=physical_error,
        social_error=social_error,
    )
    for budget in budgets:
        result = run_star_tree_benchmark(
            StarBenchmarkConfig(
                n_nodes=n_nodes,
                budget=budget,
                parameters=parameters,
            )
        )
        rows.append(
            {
                "n_nodes": n_nodes,
                "budget": budget,
                "p": p,
                "q": q,
                "r": r,
                "physical_error": physical_error,
                "social_error": social_error,
                "greedy_uncertainty": result.greedy_uncertainty,
                "optimal_uncertainty": result.optimal_uncertainty,
                "absolute_gap": result.absolute_gap,
                "relative_gap": result.relative_gap,
                "greedy_expected_social_tests": result.greedy_expected_social_tests,
                "optimal_expected_social_tests": result.optimal_expected_social_tests,
                "initial_greedy_decision": result.initial_greedy_decision.value,
                "initial_optimal_decision": result.initial_optimal_decision.value,
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Write star benchmark rows to CSV."""

    if not rows:
        raise ValueError("no star-DP benchmark rows were generated")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-nodes", type=int, default=12)
    parser.add_argument("--budgets", type=parse_int_list, default=parse_int_list("2,3,4,5"))
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--q", type=float, default=0.2)
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--physical-error", type=float, default=0.2)
    parser.add_argument("--social-error", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/star_dp_benchmark.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_star_dp_experiment(
        n_nodes=args.n_nodes,
        budgets=args.budgets,
        p=args.p,
        q=args.q,
        r=args.r,
        physical_error=args.physical_error,
        social_error=args.social_error,
    )
    write_rows(args.output, rows)
    for row in rows:
        print(row)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
