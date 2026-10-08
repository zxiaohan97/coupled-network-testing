"""Small sensitivity grid for the star-side exact-DP benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

from coupled_network_testing.benchmarks.star_dp import (
    StarBenchmarkConfig,
    run_star_tree_benchmark,
)
from coupled_network_testing.tree import ModelParameters

SCENARIOS = (
    {
        "label": "physical_accurate_social_noisy",
        "n_nodes": 12,
        "budgets": (3, 5),
        "p": 0.8,
        "q": 0.2,
        "r": 1.0,
        "physical_error": 0.01,
        "social_error": 0.1,
    },
    {
        "label": "social_reliable_physical_noisy",
        "n_nodes": 12,
        "budgets": (3, 5),
        "p": 0.8,
        "q": 0.2,
        "r": 1.0,
        "physical_error": 0.20,
        "social_error": 0.02,
    },
    {
        "label": "social_reliable_moderate_corr",
        "n_nodes": 12,
        "budgets": (3, 5),
        "p": 0.8,
        "q": 0.2,
        "r": 0.6,
        "physical_error": 0.20,
        "social_error": 0.02,
    },
    {
        "label": "social_reliable_strong_contrast",
        "n_nodes": 12,
        "budgets": (3, 5),
        "p": 0.9,
        "q": 0.1,
        "r": 1.0,
        "physical_error": 0.20,
        "social_error": 0.02,
    },
    {
        "label": "larger_star",
        "n_nodes": 16,
        "budgets": (5, 7),
        "p": 0.8,
        "q": 0.2,
        "r": 1.0,
        "physical_error": 0.20,
        "social_error": 0.02,
    },
)


def run_sensitivity_grid() -> list[dict[str, float | int | str]]:
    """Run a few public-size star DP configurations."""

    rows: list[dict[str, float | int | str]] = []
    for scenario in SCENARIOS:
        parameters = ModelParameters(
            p_infect_careless=scenario["p"],
            q_infect_careful=scenario["q"],
            social_correlation=scenario["r"],
            physical_error=scenario["physical_error"],
            social_error=scenario["social_error"],
        )
        for budget in scenario["budgets"]:
            result = run_star_tree_benchmark(
                StarBenchmarkConfig(
                    n_nodes=scenario["n_nodes"],
                    budget=budget,
                    parameters=parameters,
                )
            )
            rows.append(
                {
                    "label": scenario["label"],
                    "n_nodes": scenario["n_nodes"],
                    "budget": budget,
                    "p": scenario["p"],
                    "q": scenario["q"],
                    "r": scenario["r"],
                    "physical_error": scenario["physical_error"],
                    "social_error": scenario["social_error"],
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
    """Write the sensitivity grid to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = run_sensitivity_grid()
    output = Path("outputs/star_dp_sensitivity.csv")
    write_rows(output, rows)
    for row in rows:
        print(row)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
