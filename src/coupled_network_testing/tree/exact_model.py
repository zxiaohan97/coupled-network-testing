"""Tree-structured opinion-disease model.

This module is a cleaned starting point for the exact tree track. It preserves
the model-generation behavior used in the original tree scripts:

* the infection seed is always infected;
* the seed's social state is equally likely to be careful or careless;
* along each rooted tree edge, social correlation is controlled by ``r``;
* an infected parent infects a child with probability ``q`` if the child is
  careful and ``p`` if the child is careless.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

import networkx as nx

PhysicalState: TypeAlias = int
SocialState: TypeAlias = int
NodeState: TypeAlias = tuple[PhysicalState, SocialState]
ProbabilityTable: TypeAlias = dict[NodeState, float]

# State encoding is (physical disease state, social opinion state).
# physical=1 means infected; social=1 means careless.
STATES: tuple[NodeState, ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


@dataclass(frozen=True)
class ModelParameters:
    """Parameters of the coupled opinion-disease cascade model.

    Attributes:
        p_infect_careless: Infection probability for a careless child.
        q_infect_careful: Infection probability for a careful child.
        social_correlation: Probability that a child inherits its parent's
            social state across a social edge.
        physical_error: Physical test error probability.
        social_error: Social test error probability.
    """

    p_infect_careless: float
    q_infect_careful: float
    social_correlation: float
    physical_error: float = 0.0
    social_error: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "p_infect_careless": self.p_infect_careless,
            "q_infect_careful": self.q_infect_careful,
            "social_correlation": self.social_correlation,
            "physical_error": self.physical_error,
            "social_error": self.social_error,
        }
        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class TreeStructure:
    """A tree rooted at the infection seed."""

    graph: nx.Graph
    seed: int
    parent_of: dict[int, int]
    children_of: dict[int, tuple[int, ...]]
    bfs_edges: tuple[tuple[int, int], ...]

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[int, int]], seed: int) -> TreeStructure:
        """Build and root a tree from an edge list."""

        graph = nx.Graph()
        graph.add_edges_from(edges)
        return cls.from_graph(graph, seed=seed)

    @classmethod
    def from_graph(cls, graph: nx.Graph, seed: int) -> TreeStructure:
        """Build and root a tree from a NetworkX graph."""

        if seed not in graph:
            raise ValueError("seed must be present in the tree")
        if not nx.is_tree(graph):
            raise ValueError("graph must be a connected acyclic tree")

        parent_of: dict[int, int] = {}
        children_of: defaultdict[int, list[int]] = defaultdict(list)
        bfs_edges: list[tuple[int, int]] = []
        visited = {seed}
        queue: deque[int] = deque([seed])

        while queue:
            parent = queue.popleft()
            for child in sorted(graph.neighbors(parent)):
                if child in visited:
                    continue
                visited.add(child)
                parent_of[child] = parent
                children_of[parent].append(child)
                bfs_edges.append((parent, child))
                queue.append(child)

        frozen_children = {
            node: tuple(children)
            for node, children in children_of.items()
        }
        for node in graph.nodes:
            frozen_children.setdefault(node, ())

        return cls(
            graph=graph,
            seed=seed,
            parent_of=parent_of,
            children_of=frozen_children,
            bfs_edges=tuple(bfs_edges),
        )

    @classmethod
    def path(cls, n_nodes: int, seed: int = 0) -> TreeStructure:
        """Create a path tree with nodes ``0`` through ``n_nodes - 1``."""

        if n_nodes < 1:
            raise ValueError("n_nodes must be positive")
        graph = nx.path_graph(n_nodes)
        return cls.from_graph(graph, seed=seed)

    @classmethod
    def star(cls, n_nodes: int, seed: int = 0) -> TreeStructure:
        """Create a star tree with ``seed`` as the center."""

        if n_nodes < 2:
            raise ValueError("n_nodes must be at least 2")
        edges = [(seed, node) for node in range(n_nodes) if node != seed]
        return cls.from_edges(edges, seed=seed)

    @classmethod
    def from_branching_factors(
        cls,
        branching_factors: Iterable[int],
        seed: int = 0,
    ) -> TreeStructure:
        """Create a layered tree from branching factors.

        This is the public version of the original ``GraphPred`` helper. For
        example, ``[2, 3]`` creates two children of the seed and three children
        for each node in the next layer.
        """

        factors = list(branching_factors)
        if any(factor < 1 for factor in factors):
            raise ValueError("branching factors must be positive integers")

        next_node = seed + 1
        current_layer = [seed]
        edges: list[tuple[int, int]] = []
        for factor in factors:
            next_layer: list[int] = []
            for parent in current_layer:
                for _ in range(factor):
                    child = next_node
                    next_node += 1
                    edges.append((parent, child))
                    next_layer.append(child)
            current_layer = next_layer

        if not edges:
            graph = nx.Graph()
            graph.add_node(seed)
            return cls.from_graph(graph, seed=seed)
        return cls.from_edges(edges, seed=seed)

    def path_to_seed(self, node: int) -> tuple[int, ...]:
        """Return the path from ``node`` back to the seed."""

        if node not in self.graph:
            raise ValueError("node must be present in the tree")
        path = [node]
        current = node
        while current != self.seed:
            current = self.parent_of[current]
            path.append(current)
        return tuple(path)


class TreeOpinionDiseaseModel:
    """Initial exact probability model on a rooted tree."""

    def __init__(self, structure: TreeStructure, parameters: ModelParameters):
        self.structure = structure
        self.parameters = parameters
        self.joint_probabilities: dict[int, ProbabilityTable] = {}
        self.conditional_probabilities: dict[
            tuple[int, int],
            dict[NodeState, ProbabilityTable],
        ] = {}
        self.calculate_initial_probabilities()

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[int, int]],
        seed: int,
        parameters: ModelParameters,
    ) -> TreeOpinionDiseaseModel:
        """Build a model directly from tree edges."""

        return cls(TreeStructure.from_edges(edges, seed=seed), parameters)

    def transition_probabilities(self, parent_state: NodeState) -> ProbabilityTable:
        """Return ``P(child_state | parent_state)`` for one rooted tree edge."""

        parent_physical, parent_social = parent_state
        p = self.parameters.p_infect_careless
        q = self.parameters.q_infect_careful
        r = self.parameters.social_correlation

        table: dict[NodeState, float] = {}
        for child_social in (0, 1):
            # With probability r the child inherits the parent's social state;
            # otherwise the child receives an independent fair social label.
            social_prob = (
                r + (1.0 - r) * 0.5
                if child_social == parent_social
                else (1.0 - r) * 0.5
            )

            if parent_physical == 0:
                # The independent cascade model only allows infection to move
                # from an infected parent to its child along the rooted tree.
                table[(0, child_social)] = social_prob
                continue

            # Infection risk depends on the child's social state, not the
            # parent's. This is an important model assumption to preserve.
            infection_prob = p if child_social == 1 else q
            table[(1, child_social)] = social_prob * infection_prob
            table[(0, child_social)] = social_prob * (1.0 - infection_prob)

        return {state: table.get(state, 0.0) for state in STATES}

    def calculate_initial_probabilities(self) -> None:
        """Calculate initial ``P(physical_state, social_state)`` for every node."""

        seed = self.structure.seed
        self.joint_probabilities = {
            seed: {
                (0, 0): 0.0,
                (0, 1): 0.0,
                (1, 0): 0.5,
                (1, 1): 0.5,
            }
        }
        self.conditional_probabilities = {}

        for parent, child in self.structure.bfs_edges:
            child_table = {state: 0.0 for state in STATES}
            self.conditional_probabilities[(parent, child)] = {}
            for parent_state, parent_probability in self.joint_probabilities[parent].items():
                # Tree factorization: P(child) = sum_parent P(parent) P(child | parent).
                transition = self.transition_probabilities(parent_state)
                self.conditional_probabilities[(parent, child)][parent_state] = transition
                for child_state, transition_probability in transition.items():
                    child_table[child_state] += parent_probability * transition_probability
            self.joint_probabilities[child] = child_table

    def infection_probability(self, node: int) -> float:
        """Return the marginal probability that ``node`` is infected."""

        return sum(
            probability
            for (physical_state, _), probability in self.joint_probabilities[node].items()
            if physical_state == 1
        )

    def social_probability(self, node: int) -> float:
        """Return the marginal probability that ``node`` is careless."""

        return sum(
            probability
            for (_, social_state), probability in self.joint_probabilities[node].items()
            if social_state == 1
        )

    def physical_uncertainty(self) -> float:
        """Return average disease-state posterior variance across nodes."""

        node_count = self.structure.graph.number_of_nodes()
        return sum(
            self.infection_probability(node) * (1.0 - self.infection_probability(node))
            for node in self.structure.graph.nodes
        ) / node_count
