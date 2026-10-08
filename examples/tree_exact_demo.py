"""Small exact tree posterior demo.

Run from the public-release root after installing the package:

    python examples/tree_exact_demo.py
"""

from coupled_network_testing.tree import ExactTreePosteriorModel, ModelParameters
from coupled_network_testing.tree.policies import compare_strategies


def format_node_probabilities(model: ExactTreePosteriorModel) -> str:
    """Create a compact table of node-level posterior probabilities."""

    lines = ["node,p_infected,p_careless"]
    for node in sorted(model.G.nodes()):
        p_infected = sum(
            probability
            for (physical_state, _), probability in model.probs[node].items()
            if physical_state == 1
        )
        p_careless = sum(
            probability
            for (_, social_state), probability in model.probs[node].items()
            if social_state == 1
        )
        lines.append(f"{node},{p_infected:.4f},{p_careless:.4f}")
    return "\n".join(lines)


def main() -> None:
    params = ModelParameters(
        p_infect_careless=0.8,
        q_infect_careful=0.2,
        social_correlation=0.5,
        physical_error=0.05,
        social_error=0.10,
    )
    model = ExactTreePosteriorModel.from_edges(
        edges=[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)],
        seed=0,
        parameters=params,
    )

    print("Initial posterior")
    print(format_node_probabilities(model))

    test_sequence = ((2, 1, 0), (1, 0, 1))
    model.update_with_test_sequence(test_sequence)

    print("\nAfter tests: (node 2, physical, negative), (node 1, social, positive)")
    print(format_node_probabilities(model))

    model.reset_to_initial_state()
    comparison = compare_strategies(model, num_tests=2)
    print("\nExpected physical uncertainty by budget")
    print(f"greedy:          {comparison['greedy']['uncertainties']}")
    print(f"contact tracing: {comparison['contact_tracing']['uncertainties']}")


if __name__ == "__main__":
    main()
