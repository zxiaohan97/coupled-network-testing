"""Testing policies for exact tree networks.

These functions are adapted from the policy methods in the original
``Tree/Tree_Testing_Model.py`` file.
"""

from __future__ import annotations

from coupled_network_testing.tree.posterior_updates import ExactTreePosteriorModel

TestSequence = tuple[tuple[int, int, int], ...]


def select_next_test_greedy(
    model: ExactTreePosteriorModel,
    test_sequence: TestSequence = (),
) -> tuple[tuple[int, int] | None, float]:
    """Choose the next test minimizing expected one-step uncertainty."""

    model.load_state(test_sequence)
    min_expected_uncertainty = float("inf")
    best_test = None

    for node in model.G.nodes():
        for test_type in (0, 1):
            expected_uncertainty = 0.0
            for result in (0, 1):
                p_outcome = model.get_test_outcome_probability(node, test_type, result)
                if p_outcome > 0:
                    # Evaluate both possible outcomes without committing either
                    # branch to the live posterior state.
                    snapshot = model.snapshot()
                    model.update_with_test((node, test_type, result))
                    expected_uncertainty += p_outcome * model.get_physical_uncertainty()
                    model.restore_snapshot(snapshot)

            if expected_uncertainty < min_expected_uncertainty:
                min_expected_uncertainty = expected_uncertainty
                best_test = (node, test_type)

    return best_test, min_expected_uncertainty


def run_greedy_testing(
    model: ExactTreePosteriorModel,
    num_tests: int,
) -> tuple[list[tuple[TestSequence, float]], list[float]]:
    """Run the original exhaustive greedy testing tree for ``num_tests`` rounds."""

    model.reset_to_initial_state()
    sequences_to_explore: list[tuple[TestSequence, float]] = [((), 1.0)]
    uncertainty_by_time = [model.get_physical_uncertainty()]

    for _ in range(num_tests):
        new_sequences: list[tuple[TestSequence, float]] = []
        time_uncertainty = 0.0

        for test_sequence, sequence_prob in sequences_to_explore:
            next_test, _ = select_next_test_greedy(model, test_sequence)
            if next_test is None:
                continue
            node, test_type = next_test

            for result in (0, 1):
                model.load_state(test_sequence)
                p_outcome = model.get_test_outcome_probability(node, test_type, result)
                if p_outcome > 0:
                    # Keep the full binary outcome tree because the next greedy
                    # decision depends on the observations seen so far.
                    model.update_with_test((node, test_type, result))
                    new_sequence = test_sequence + ((node, test_type, result),)
                    new_prob = sequence_prob * p_outcome
                    model.store_current_state(new_sequence)
                    new_sequences.append((new_sequence, new_prob))
                    time_uncertainty += new_prob * model.get_physical_uncertainty()

        sequences_to_explore = new_sequences
        uncertainty_by_time.append(time_uncertainty)

    return sequences_to_explore, uncertainty_by_time


def contact_tracing_order(model: ExactTreePosteriorModel) -> list[int]:
    """Return the original BFS contact-tracing order from seed neighbors."""

    bfs_order = []
    visited = {model.seed}
    queue = list(model.G.neighbors(model.seed))
    visited.update(queue)
    bfs_order.extend(queue)

    while queue:
        node = queue.pop(0)
        for neighbor in model.G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                bfs_order.append(neighbor)

    return bfs_order


def run_contact_tracing(
    model: ExactTreePosteriorModel,
    num_tests: int,
) -> tuple[list[tuple[TestSequence, float]], list[float]]:
    """Run physical-test-only contact tracing in the original BFS order."""

    model.reset_to_initial_state()
    bfs_order = contact_tracing_order(model)
    sequences_to_explore: list[tuple[TestSequence, float]] = [((), 1.0)]
    uncertainty_by_time = [model.get_physical_uncertainty()]

    if not bfs_order:
        return sequences_to_explore, uncertainty_by_time

    for _ in range(num_tests):
        new_sequences: list[tuple[TestSequence, float]] = []
        time_uncertainty = 0.0

        for current_sequence, sequence_prob in sequences_to_explore:
            # Baseline policy: fixed BFS order from seed neighbors, physical
            # tests only. It does not use hidden true infection states.
            test_idx = len(current_sequence)
            if test_idx >= len(bfs_order):
                test_idx %= len(bfs_order)
            node = bfs_order[test_idx]
            test_type = 1

            for result in (0, 1):
                model.load_state(current_sequence)
                p_outcome = model.get_test_outcome_probability(node, test_type, result)
                if p_outcome > 0:
                    model.update_with_test((node, test_type, result))
                    new_sequence = current_sequence + ((node, test_type, result),)
                    new_prob = sequence_prob * p_outcome
                    model.store_current_state(new_sequence)
                    time_uncertainty += new_prob * model.get_physical_uncertainty()
                    new_sequences.append((new_sequence, new_prob))

        sequences_to_explore = new_sequences
        uncertainty_by_time.append(time_uncertainty)

    return sequences_to_explore, uncertainty_by_time


def compare_strategies(model: ExactTreePosteriorModel, num_tests: int) -> dict:
    """Compare greedy and contact-tracing strategies."""

    initial_snapshot = model.snapshot()
    greedy_sequences, greedy_uncertainties = run_greedy_testing(model, num_tests)

    model.restore_snapshot(initial_snapshot)
    model.store_current_state(())
    contact_sequences, contact_uncertainties = run_contact_tracing(model, num_tests)

    return {
        "greedy": {
            "uncertainties": greedy_uncertainties,
            "final_sequences": greedy_sequences,
        },
        "contact_tracing": {
            "uncertainties": contact_uncertainties,
            "final_sequences": contact_sequences,
        },
    }


def analyze_threshold_times(
    model: ExactTreePosteriorModel,
    thresholds: list[float] | tuple[float, ...],
    max_budget: int,
) -> dict:
    """Track when each policy first reaches each uncertainty threshold."""

    threshold_times = {
        "greedy": {d: float("inf") for d in thresholds},
        "contact_tracing": {d: float("inf") for d in thresholds},
    }

    initial_snapshot = model.snapshot()
    remaining_thresholds = set(thresholds)
    t = 0
    while t < max_budget and remaining_thresholds:
        _, uncertainties = run_greedy_testing(model, t + 1)
        current_uncertainty = uncertainties[-1]
        reached_thresholds = {
            threshold
            for threshold in remaining_thresholds
            if current_uncertainty <= threshold
        }
        for threshold in reached_thresholds:
            threshold_times["greedy"][threshold] = t + 1
        remaining_thresholds -= reached_thresholds
        t += 1

    model.restore_snapshot(initial_snapshot)
    model.store_current_state(())

    remaining_thresholds = set(thresholds)
    t = 0
    while t < max_budget and remaining_thresholds:
        _, uncertainties = run_contact_tracing(model, t + 1)
        current_uncertainty = uncertainties[-1]
        reached_thresholds = {
            threshold
            for threshold in remaining_thresholds
            if current_uncertainty <= threshold
        }
        for threshold in reached_thresholds:
            threshold_times["contact_tracing"][threshold] = t + 1
        remaining_thresholds -= reached_thresholds
        t += 1

    return threshold_times
