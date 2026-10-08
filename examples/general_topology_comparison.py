"""Run a small ER/scale-free/Watts-Strogatz policy comparison.

This example is intentionally lightweight. It validates the general-graph
pipeline and writes tidy CSV files, but it is not a manuscript-scale sweep.
"""

from __future__ import annotations

from pathlib import Path

from coupled_network_testing.general import (
    GeneralExperimentConfig,
    GeneralGraphModel,
    GeneralModelParameters,
    TopologyComparisonConfig,
    run_topology_comparison,
)


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    config = TopologyComparisonConfig(
        base_config=GeneralExperimentConfig(
            n_nodes=16,
            n_particles=2_000,
            max_tests=6,
            ess_warning_threshold=100,
            min_branch_ess=100,
            parameters=GeneralModelParameters(
                p_infect_careless=0.6,
                q_infect_careful=0.2,
                social_correlation=0.75,
                physical_error=0.15,
                social_error=0.1,
            ),
        ),
        graph_models=(
            GeneralGraphModel.ERDOS_RENYI,
            GeneralGraphModel.SCALE_FREE,
            GeneralGraphModel.WATTS_STROGATZ,
        ),
        edge_probabilities=(0.04, 0.08, 0.12),
        n_replicates=2,
        random_seed=2026,
    )
    result = run_topology_comparison(config)

    steps_path = output_dir / "general_topology_steps.csv"
    summary_path = output_dir / "general_topology_summary.csv"
    result.steps.to_csv(steps_path, index=False)
    result.summary.to_csv(summary_path, index=False)

    print(result.summary.to_string(index=False))
    print(f"\nWrote {steps_path} and {summary_path}")


if __name__ == "__main__":
    main()
