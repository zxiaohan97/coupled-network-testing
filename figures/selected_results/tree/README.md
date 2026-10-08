Selected synthetic figures for the README and experiment notes.

Run from the repository root after installing the package:

```bash
python experiments/tree/random_tree_budget_gain.py
python experiments/tree/plot_tree_budget_gain.py
python experiments/tree/spanning_tree_approximation.py
python experiments/tree/plot_spanning_tree_approximation.py
```

- `tree_budget_gain.png`: five 8-node random trees, seed 7, uncertainty threshold
  0.15, and maximum budget 5. The average includes only trees where both policies
  reach the threshold and contact tracing uses a positive budget. Check
  `num_reached` in the generated CSV before interpreting the mean.
- `spanning_tree_approximation.png`: sparse cyclic ER networks compared with BFS
  spanning trees using the legacy hard-rejection estimator. This is an
  approximation study, not a validation of the current weighted posterior engine.

See [Tree Experiments](../../../docs/tree_experiments.md) for parameter details.
