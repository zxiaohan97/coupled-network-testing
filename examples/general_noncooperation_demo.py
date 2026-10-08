"""Small non-cooperation demo for general-network simulations.

This script compares refusal-aware greedy testing with physical contact tracing
on the same sampled cascades. It is intentionally small enough for a laptop
smoke run; manuscript-scale sweeps used many more particles and graph samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from coupled_network_testing.general import (
    GeneralCascadeSimulator,
    GeneralModelParameters,
    NonCooperationMode,
    NonCooperationParameters,
    generate_er_graph,
    make_coupled_network_pair,
    noncooperative_steps_to_dataframe,
    run_noncooperative_policy_comparison,
    select_seed_nodes,
)


def main() -> None:
    rng = np.random.default_rng(2026)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    rows = []
    for run_index, cooperation_fraction in enumerate((1.0, 0.6, 0.0)):
        run_rng = np.random.default_rng(2026 + run_index)
        base_graph = generate_er_graph(n_nodes=12, edge_probability=0.2, rng=run_rng)
        network_pair = make_coupled_network_pair(base_graph, rng=run_rng)
        seeds = select_seed_nodes(network_pair.physical, n_seeds=1, rng=run_rng)
        model_parameters = GeneralModelParameters(
            p_infect_careless=0.6,
            q_infect_careful=0.2,
            social_correlation=0.75,
            physical_error=0.15,
            social_error=0.1,
        )
        simulator = GeneralCascadeSimulator(
            network_pair.physical,
            network_pair.social,
            seeds=seeds,
            parameters=model_parameters,
            rng=run_rng,
        )
        ground_truth = simulator.sample_realization()
        particles = simulator.sample_realizations(1_000)
        noncooperation_parameters = NonCooperationParameters(
            cooperation_fraction=cooperation_fraction,
            mode=NonCooperationMode.CARELESS_PHYSICAL,
        )
        steps = run_noncooperative_policy_comparison(
            network_pair.physical,
            particles,
            ground_truth,
            model_parameters,
            noncooperation_parameters,
            seeds=seeds,
            max_tests=6,
            rng=rng,
            min_branch_ess=25,
            ess_warning_threshold=25,
        )
        table = noncooperative_steps_to_dataframe(steps)
        table["run_index"] = run_index
        table["cooperation_fraction"] = cooperation_fraction
        rows.append(table)

    steps = pd.concat(rows, ignore_index=True)
    summary = (
        steps.sort_values("test_index")
        .groupby(["cooperation_fraction", "strategy"], as_index=False)
        .agg(
            final_uncertainty=("posterior_uncertainty", "last"),
            refused_tests=("refused", "sum"),
            social_tests=("test_type", lambda values: int((values == 0).sum())),
            stable_fraction=("is_stable", "mean"),
        )
    )

    steps.to_csv(output_dir / "general_noncooperation_steps.csv", index=False)
    summary.to_csv(output_dir / "general_noncooperation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
