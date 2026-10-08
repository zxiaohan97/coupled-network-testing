"""Experiment orchestration for general networks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum

import networkx as nx
import numpy as np
import pandas as pd

from coupled_network_testing.common.testing_types import TestType
from coupled_network_testing.general.graph_generators import (
    SeedSelection,
    generate_er_graph,
    generate_scale_free_graph,
    generate_watts_strogatz_graph,
    make_coupled_network_pair,
    select_seed_nodes,
)
from coupled_network_testing.general.policies import (
    PolicyDecision,
    select_contact_tracing_test,
    select_weighted_greedy_test,
)
from coupled_network_testing.general.simulation_model import (
    CascadeRealization,
    GeneralCascadeSimulator,
    GeneralModelParameters,
    TestRecord,
)
from coupled_network_testing.general.weighted_particle_filter import WeightedParticlePosterior


class GeneralGraphModel(StrEnum):
    """Synthetic graph families used by lightweight public experiments."""

    ERDOS_RENYI = "er"
    SCALE_FREE = "scale_free"
    WATTS_STROGATZ = "watts_strogatz"


class PolicyStrategy(StrEnum):
    """Policy names emitted by the experiment runner."""

    WEIGHTED_GREEDY = "weighted_greedy"
    CONTACT_TRACING = "contact_tracing"


@dataclass(frozen=True)
class GeneralExperimentConfig:
    """Configuration for a small reproducible general-graph policy comparison."""

    n_nodes: int = 20
    edge_probability: float = 0.15
    graph_model: GeneralGraphModel | str = GeneralGraphModel.ERDOS_RENYI
    parameters: GeneralModelParameters = field(
        default_factory=lambda: GeneralModelParameters(
            p_infect_careless=0.8,
            q_infect_careful=0.2,
            social_correlation=0.5,
            physical_error=0.05,
            social_error=0.1,
        )
    )
    n_particles: int = 5_000
    max_tests: int = 10
    n_seeds: int = 1
    seed_selection: SeedSelection | str = SeedSelection.HIGHEST_DEGREE
    overlap: float | None = None
    watts_strogatz_rewiring: float = 0.2
    ess_warning_threshold: float = 300.0
    min_branch_ess: float = 300.0


@dataclass(frozen=True)
class TopologyComparisonConfig:
    """Small grid of graph topologies and parameters for public experiments."""

    base_config: GeneralExperimentConfig = field(
        default_factory=lambda: GeneralExperimentConfig(
            n_nodes=20,
            edge_probability=0.08,
            n_particles=5_000,
            max_tests=10,
            parameters=GeneralModelParameters(
                p_infect_careless=0.6,
                q_infect_careful=0.2,
                social_correlation=0.75,
                physical_error=0.15,
                social_error=0.1,
            ),
        )
    )
    graph_models: tuple[GeneralGraphModel | str, ...] = (
        GeneralGraphModel.ERDOS_RENYI,
        GeneralGraphModel.SCALE_FREE,
        GeneralGraphModel.WATTS_STROGATZ,
    )
    edge_probabilities: tuple[float, ...] = (0.04, 0.08, 0.12)
    parameter_sets: tuple[GeneralModelParameters, ...] = (
        GeneralModelParameters(
            p_infect_careless=0.6,
            q_infect_careful=0.2,
            social_correlation=0.75,
            physical_error=0.15,
            social_error=0.1,
        ),
    )
    n_replicates: int = 5
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.n_replicates < 1:
            raise ValueError("n_replicates must be positive")
        if not self.graph_models:
            raise ValueError("at least one graph model is required")
        if not self.edge_probabilities:
            raise ValueError("at least one edge probability is required")
        if not self.parameter_sets:
            raise ValueError("at least one parameter set is required")


@dataclass(frozen=True)
class PolicyStep:
    """One observed step from a policy run."""

    strategy: str
    test_index: int
    node: int
    test_type: int
    result: int
    posterior_uncertainty: float
    effective_sample_size: float
    ess_fraction: float
    expected_uncertainty_before_test: float | None
    positive_probability: float | None
    minimum_branch_ess: float | None
    is_stable: bool


@dataclass(frozen=True)
class PolicyComparisonResult:
    """Outputs from one paired policy comparison."""

    physical_graph: nx.Graph
    social_graph: nx.Graph
    seeds: tuple[int, ...]
    ground_truth: CascadeRealization
    steps: tuple[PolicyStep, ...]

    def to_dataframe(self) -> pd.DataFrame:
        """Return step records as a tidy table for plotting or saving."""

        return pd.DataFrame([asdict(step) for step in self.steps])


@dataclass(frozen=True)
class TopologyComparisonResult:
    """Step-level and summarized outputs from a topology comparison grid."""

    steps: pd.DataFrame
    summary: pd.DataFrame


def make_synthetic_graph(
    config: GeneralExperimentConfig,
    rng: np.random.Generator,
) -> nx.Graph:
    """Generate one synthetic graph from the public experiment config."""

    graph_model = GeneralGraphModel(config.graph_model)
    if graph_model is GeneralGraphModel.ERDOS_RENYI:
        return generate_er_graph(config.n_nodes, config.edge_probability, rng)
    if graph_model is GeneralGraphModel.SCALE_FREE:
        return generate_scale_free_graph(config.n_nodes, config.edge_probability, rng)
    if graph_model is GeneralGraphModel.WATTS_STROGATZ:
        return generate_watts_strogatz_graph(
            config.n_nodes,
            config.edge_probability,
            config.watts_strogatz_rewiring,
            rng,
        )
    raise ValueError(f"unsupported graph model: {graph_model}")


def run_policy_comparison(
    config: GeneralExperimentConfig | None = None,
    rng: np.random.Generator | None = None,
) -> PolicyComparisonResult:
    """Run weighted greedy and contact tracing on the same sampled cascade."""

    config = config or GeneralExperimentConfig()
    rng = rng or np.random.default_rng()
    base_graph = make_synthetic_graph(config, rng)
    network_pair = make_coupled_network_pair(base_graph, rng=rng, overlap=config.overlap)
    seeds = select_seed_nodes(
        network_pair.physical,
        n_seeds=config.n_seeds,
        rng=rng,
        mode=config.seed_selection,
    )
    simulator = GeneralCascadeSimulator(
        network_pair.physical,
        network_pair.social,
        seeds=seeds,
        parameters=config.parameters,
        rng=rng,
    )
    ground_truth = simulator.sample_realization()
    particles = simulator.sample_realizations(config.n_particles)

    steps = [
        *run_weighted_greedy_policy(
            network_pair.physical,
            particles,
            ground_truth,
            config,
            seeds,
        ),
        *run_contact_tracing_policy(
            network_pair.physical,
            particles,
            ground_truth,
            config,
            seeds,
        ),
    ]
    return PolicyComparisonResult(
        physical_graph=network_pair.physical,
        social_graph=network_pair.social,
        seeds=seeds,
        ground_truth=ground_truth,
        steps=tuple(steps),
    )


def run_topology_comparison(
    config: TopologyComparisonConfig | None = None,
) -> TopologyComparisonResult:
    """Run a lightweight ER/BA/WS comparison over selected parameters.

    The default grid is intentionally small enough for a laptop demo. Increase
    ``n_replicates`` and ``n_particles`` only for manuscript-scale reruns.
    """

    config = config or TopologyComparisonConfig()
    step_tables: list[pd.DataFrame] = []

    run_index = 0
    for parameter_index, parameters in enumerate(config.parameter_sets):
        for graph_model in config.graph_models:
            graph_model = GeneralGraphModel(graph_model)
            for edge_probability in config.edge_probabilities:
                for replicate in range(config.n_replicates):
                    run_seed = config.random_seed + run_index
                    run_index += 1
                    run_config = replace(
                        config.base_config,
                        graph_model=graph_model,
                        edge_probability=edge_probability,
                        parameters=parameters,
                    )
                    comparison = run_policy_comparison(
                        run_config,
                        rng=np.random.default_rng(run_seed),
                    )
                    step_table = comparison.to_dataframe()
                    _add_run_metadata(
                        step_table,
                        run_config=run_config,
                        graph_model=graph_model,
                        parameter_index=parameter_index,
                        replicate=replicate,
                        run_seed=run_seed,
                    )
                    step_tables.append(step_table)

    steps = pd.concat(step_tables, ignore_index=True) if step_tables else pd.DataFrame()
    summary = summarize_topology_steps(steps)
    return TopologyComparisonResult(steps=steps, summary=summary)


def summarize_topology_steps(steps: pd.DataFrame) -> pd.DataFrame:
    """Summarize final uncertainty, ESS, and social-test use by run/strategy."""

    if steps.empty:
        return pd.DataFrame()

    group_columns = [
        "graph_model",
        "edge_probability",
        "parameter_index",
        "replicate",
        "strategy",
    ]
    records = []
    for keys, group in steps.groupby(group_columns, sort=True):
        group = group.sort_values("test_index")
        final = group.iloc[-1]
        record = dict(zip(group_columns, keys, strict=True))
        record.update(
            {
                "final_uncertainty": float(final["posterior_uncertainty"]),
                "final_ess": float(final["effective_sample_size"]),
                "area_under_uncertainty": float(group["posterior_uncertainty"].sum()),
                "social_tests": int((group["test_type"] == int(TestType.SOCIAL)).sum()),
                "stable_fraction": float(group["is_stable"].mean()),
                "n_tests": int(group["test_index"].max()),
                "p_infect_careless": float(final["p_infect_careless"]),
                "q_infect_careful": float(final["q_infect_careful"]),
                "social_correlation": float(final["social_correlation"]),
                "physical_error": float(final["physical_error"]),
                "social_error": float(final["social_error"]),
                "n_particles": int(final["n_particles"]),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def run_weighted_greedy_policy(
    physical_graph: nx.Graph,
    particles: list[CascadeRealization],
    ground_truth: CascadeRealization,
    config: GeneralExperimentConfig,
    seeds: tuple[int, ...],
) -> list[PolicyStep]:
    """Run the weighted greedy policy for ``config.max_tests`` observations."""

    posterior = WeightedParticlePosterior(
        particles,
        parameters=config.parameters,
        nodes=tuple(physical_graph.nodes()),
    )
    test_sequence: list[TestRecord] = []
    steps: list[PolicyStep] = []

    for test_index in range(1, config.max_tests + 1):
        decision = select_weighted_greedy_test(
            posterior,
            test_sequence=test_sequence,
            seeds=seeds,
            min_branch_ess=config.min_branch_ess,
        )
        result = ground_truth.test_result(decision.node, decision.test_type)
        test_sequence.append((decision.node, int(decision.test_type), result))
        steps.append(
            _make_policy_step(
                PolicyStrategy.WEIGHTED_GREEDY,
                test_index,
                decision,
                result,
                posterior,
                test_sequence,
                config,
            )
        )

    return steps


def run_contact_tracing_policy(
    physical_graph: nx.Graph,
    particles: list[CascadeRealization],
    ground_truth: CascadeRealization,
    config: GeneralExperimentConfig,
    seeds: tuple[int, ...],
) -> list[PolicyStep]:
    """Run posterior-only contact tracing for ``config.max_tests`` observations."""

    posterior = WeightedParticlePosterior(
        particles,
        parameters=config.parameters,
        nodes=tuple(physical_graph.nodes()),
    )
    test_sequence: list[TestRecord] = []
    steps: list[PolicyStep] = []

    for test_index in range(1, config.max_tests + 1):
        estimate_before = posterior.estimate(test_sequence)
        decision = select_contact_tracing_test(
            physical_graph,
            estimate_before,
            test_sequence=test_sequence,
            seeds=seeds,
        )
        result = ground_truth.test_result(decision.node, decision.test_type)
        test_sequence.append((decision.node, int(decision.test_type), result))
        steps.append(
            _make_policy_step(
                PolicyStrategy.CONTACT_TRACING,
                test_index,
                decision,
                result,
                posterior,
                test_sequence,
                config,
            )
        )

    return steps


def _make_policy_step(
    strategy: PolicyStrategy,
    test_index: int,
    decision: PolicyDecision,
    result: int,
    posterior: WeightedParticlePosterior,
    test_sequence: list[TestRecord],
    config: GeneralExperimentConfig,
) -> PolicyStep:
    """Build a step record after appending the observed result."""

    estimate_after = posterior.estimate(test_sequence)
    is_stable = (
        decision.is_stable
        and estimate_after.effective_sample_size >= config.ess_warning_threshold
    )
    return PolicyStep(
        strategy=strategy.value,
        test_index=test_index,
        node=decision.node,
        test_type=int(decision.test_type),
        result=result,
        posterior_uncertainty=estimate_after.physical_uncertainty,
        effective_sample_size=estimate_after.effective_sample_size,
        ess_fraction=estimate_after.ess_fraction,
        expected_uncertainty_before_test=decision.expected_uncertainty,
        positive_probability=decision.positive_probability,
        minimum_branch_ess=decision.minimum_branch_ess,
        is_stable=is_stable,
    )


def _add_run_metadata(
    step_table: pd.DataFrame,
    run_config: GeneralExperimentConfig,
    graph_model: GeneralGraphModel,
    parameter_index: int,
    replicate: int,
    run_seed: int,
) -> None:
    """Attach graph/model settings to a step table in place."""

    parameters = run_config.parameters
    step_table["graph_model"] = graph_model.value
    step_table["edge_probability"] = run_config.edge_probability
    step_table["parameter_index"] = parameter_index
    step_table["replicate"] = replicate
    step_table["run_seed"] = run_seed
    step_table["n_nodes"] = run_config.n_nodes
    step_table["n_particles"] = run_config.n_particles
    step_table["max_tests"] = run_config.max_tests
    step_table["seed_selection"] = str(run_config.seed_selection)
    step_table["overlap"] = run_config.overlap
    step_table["watts_strogatz_rewiring"] = run_config.watts_strogatz_rewiring
    step_table["p_infect_careless"] = parameters.p_infect_careless
    step_table["q_infect_careful"] = parameters.q_infect_careful
    step_table["social_correlation"] = parameters.social_correlation
    step_table["physical_error"] = parameters.physical_error
    step_table["social_error"] = parameters.social_error
