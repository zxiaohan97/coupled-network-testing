"""Small fixed-tree examples that stress-test the budget-gain intuition.

The goal is not to prove a theorem. Instead, this script checks whether the
mechanism is visible under controlled settings:

* larger separation between ``p`` and ``q`` makes social state more predictive;
* larger physical-test error makes physical-only contact tracing less decisive;
* low social-test error makes social tests useful;
* concentrated trees can make one social observation informative for many nodes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from coupled_network_testing.tree import ExactTreePosteriorModel, ModelParameters
from coupled_network_testing.tree.policies import run_contact_tracing, run_greedy_testing

TREES = {
    "balanced_7": ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)),
    "star_7": tuple((0, node) for node in range(1, 7)),
    "path_7": tuple((node, node + 1) for node in range(6)),
    "two_level_10": (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 6),
        (2, 7),
        (3, 8),
        (3, 9),
    ),
}

SCENARIOS = {
    "low_contrast_low_physical_error": {
        "p": 0.65,
        "q": 0.35,
        "r": 0.5,
        "physical_error": 0.05,
        "social_error": 0.02,
    },
    "high_contrast_low_physical_error": {
        "p": 0.9,
        "q": 0.1,
        "r": 0.5,
        "physical_error": 0.05,
        "social_error": 0.02,
    },
    "high_contrast_high_physical_error": {
        "p": 0.9,
        "q": 0.1,
        "r": 0.5,
        "physical_error": 0.20,
        "social_error": 0.02,
    },
    "high_contrast_noisy_social": {
        "p": 0.9,
        "q": 0.1,
        "r": 0.5,
        "physical_error": 0.20,
        "social_error": 0.15,
    },
}

THRESHOLDS = (0.18, 0.16, 0.14, 0.13, 0.12, 0.10)


def first_budget_at_or_below(uncertainties: list[float], threshold: float) -> int | None:
    """Return the first budget that reaches an uncertainty threshold."""

    for budget, uncertainty in enumerate(uncertainties):
        if uncertainty <= threshold:
            return budget
    return None


def run_intuition_examples(max_budget: int) -> list[dict[str, float | int | str | None]]:
    """Evaluate a small grid of tree shapes and mechanism-focused scenarios."""

    rows: list[dict[str, float | int | str | None]] = []
    for tree_name, edges in TREES.items():
        n_nodes = len({node for edge in edges for node in edge})
        for scenario_name, scenario in SCENARIOS.items():
            params = ModelParameters(
                p_infect_careless=scenario["p"],
                q_infect_careful=scenario["q"],
                social_correlation=scenario["r"],
                physical_error=scenario["physical_error"],
                social_error=scenario["social_error"],
            )
            model = ExactTreePosteriorModel.from_edges(edges, seed=0, parameters=params)
            _, greedy_uncertainties = run_greedy_testing(model, max_budget)

            model.reset_to_initial_state()
            model.store_current_state(())
            _, contact_uncertainties = run_contact_tracing(model, max_budget)

            contact_final = contact_uncertainties[-1]
            final_relative_reduction = (
                (contact_final - greedy_uncertainties[-1]) / contact_final
                if contact_final > 0
                else 0.0
            )

            for threshold in THRESHOLDS:
                greedy_budget = first_budget_at_or_below(greedy_uncertainties, threshold)
                contact_budget = first_budget_at_or_below(contact_uncertainties, threshold)
                budget_gain = None
                if (
                    greedy_budget is not None
                    and contact_budget is not None
                    and contact_budget > 0
                ):
                    # Positive values mean greedy reaches the same target sooner.
                    budget_gain = (contact_budget - greedy_budget) / contact_budget

                rows.append(
                    {
                        "tree": tree_name,
                        "scenario": scenario_name,
                        "threshold": threshold,
                        "n_nodes": n_nodes,
                        "max_budget": max_budget,
                        "p": scenario["p"],
                        "q": scenario["q"],
                        "r": scenario["r"],
                        "physical_error": scenario["physical_error"],
                        "social_error": scenario["social_error"],
                        "greedy_budget": greedy_budget,
                        "contact_tracing_budget": contact_budget,
                        "budget_gain": budget_gain,
                        "greedy_final_uncertainty": greedy_uncertainties[-1],
                        "contact_final_uncertainty": contact_final,
                        "final_relative_reduction": final_relative_reduction,
                    }
                )

    return rows


def write_rows(path: Path, rows: list[dict[str, float | int | str | None]]) -> None:
    """Write intuition-example rows to CSV."""

    if not rows:
        raise ValueError("no intuition-example rows were generated")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-budget", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/fixed_tree_intuition_examples.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_intuition_examples(max_budget=args.max_budget)
    write_rows(args.output, rows)

    positive_rows = [
        row
        for row in rows
        if row["budget_gain"] is not None and float(row["budget_gain"]) > 0
    ]
    print(f"rows={len(rows)}")
    print(f"positive_budget_gain_rows={len(positive_rows)}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
