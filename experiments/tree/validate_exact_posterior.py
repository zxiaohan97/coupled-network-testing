"""Validate exact tree posterior updates against Monte Carlo simulation.

This is the public, lightweight version of the original
``Tree/Tree_Posterior_Validation.py`` check. It conditions both methods on the
same observed test sequence, then reports exact and empirical node-state
probabilities side by side.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from coupled_network_testing.tree import ExactTreePosteriorModel, ModelParameters
from coupled_network_testing.tree.exact_model import STATES
from coupled_network_testing.tree.validation import DiseaseSpreadSimulation

DEFAULT_EDGES = ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6))
DEFAULT_TEST_SEQUENCE = ((2, 1, 0), (1, 0, 1))


def parse_test_sequence(value: str) -> tuple[tuple[int, int, int], ...]:
    """Parse ``node:test_type:result`` records separated by commas."""

    tests = []
    for item in value.split(","):
        if not item.strip():
            continue
        node, test_type, result = (int(part) for part in item.split(":"))
        tests.append((node, test_type, result))
    return tuple(tests)


def run_validation(
    *,
    p: float,
    q: float,
    social_correlation: float,
    physical_error: float,
    social_error: float,
    n_simulations: int,
    random_seed: int,
    test_sequence: tuple[tuple[int, int, int], ...],
) -> list[dict[str, float | int | str]]:
    """Compare exact posterior probabilities with empirical estimates."""

    params = ModelParameters(
        p_infect_careless=p,
        q_infect_careful=q,
        social_correlation=social_correlation,
        physical_error=physical_error,
        social_error=social_error,
    )
    model = ExactTreePosteriorModel.from_edges(DEFAULT_EDGES, seed=0, parameters=params)
    exact_probs = model.update_with_test_sequence(test_sequence)

    simulator = DiseaseSpreadSimulation(
        edges=DEFAULT_EDGES,
        seed_node=0,
        p=p,
        q=q,
        r=social_correlation,
        ep=physical_error,
        es=social_error,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )
    simulated_probs = simulator.run_simulation_sequence(test_sequence)

    rows: list[dict[str, float | int | str]] = []
    for step, (exact_step, simulated_step) in enumerate(
        zip(exact_probs, simulated_probs, strict=True),
    ):
        if simulated_step is None:
            continue

        test_label = "initial" if step == 0 else str(test_sequence[step - 1])
        for node in sorted(exact_step):
            for physical_state, social_state in STATES:
                exact_probability = exact_step[node].get((physical_state, social_state), 0.0)
                simulated_probability = simulated_step[node].get(
                    (physical_state, social_state),
                    0.0,
                )
                rows.append(
                    {
                        "step": step,
                        "conditioned_on": test_label,
                        "node": node,
                        "physical_state": physical_state,
                        "social_state": social_state,
                        "exact_probability": exact_probability,
                        "simulated_probability": simulated_probability,
                        "absolute_error": abs(exact_probability - simulated_probability),
                    }
                )

    return rows


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    """Write validation rows to CSV."""

    if not rows:
        raise ValueError("no validation rows were generated")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--q", type=float, default=0.2)
    parser.add_argument("--r", type=float, default=0.5)
    parser.add_argument("--physical-error", type=float, default=0.15)
    parser.add_argument("--social-error", type=float, default=0.1)
    parser.add_argument("--n-simulations", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=11)
    parser.add_argument(
        "--test-sequence",
        type=parse_test_sequence,
        default=DEFAULT_TEST_SEQUENCE,
        help="Comma-separated tests like 2:1:0,1:0:1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/tree_posterior_validation.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = run_validation(
        p=args.p,
        q=args.q,
        social_correlation=args.r,
        physical_error=args.physical_error,
        social_error=args.social_error,
        n_simulations=args.n_simulations,
        random_seed=args.random_seed,
        test_sequence=args.test_sequence,
    )
    write_rows(args.output, rows)

    errors = np.array([float(row["absolute_error"]) for row in rows])
    print(f"rows={len(rows)}")
    print(f"mean_absolute_error={errors.mean():.6f}")
    print(f"max_absolute_error={errors.max():.6f}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
