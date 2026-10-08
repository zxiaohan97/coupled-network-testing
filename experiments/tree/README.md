Tree-network experiments.

Use this folder for exact posterior validation, tree budget-gain experiments,
and selected tree figure reproduction scripts.

```bash
python experiments/tree/validate_exact_posterior.py
python experiments/tree/fixed_tree_policy_comparison.py
python experiments/tree/fixed_tree_budget_gain.py
python experiments/tree/fixed_tree_intuition_examples.py
python experiments/tree/star_dp_benchmark.py
python experiments/tree/star_dp_sensitivity.py
python experiments/tree/random_tree_budget_gain.py
python experiments/tree/plot_tree_budget_gain.py
python experiments/tree/spanning_tree_approximation.py
python experiments/tree/plot_spanning_tree_approximation.py
```

The defaults are small public smoke runs. Manuscript-scale runs should increase
the number of graphs, samples, budgets, and parameter-grid values.

Experiment roles:

- `validate_exact_posterior.py` checks the exact tree posterior update against
  Monte Carlo simulation on a fixed tree and observed test sequence.
- `fixed_tree_policy_comparison.py` is an illustrative policy comparison. It is
  deliberately not presented as proof that greedy is globally optimal.
- `fixed_tree_budget_gain.py` shows a small fixed-tree threshold example where
  social observations create a positive budget gain at moderate social
  correlation.
- `fixed_tree_intuition_examples.py` checks a few controlled examples that vary
  disease contrast, physical-test error, social-test error, and tree shape.
- `star_dp_benchmark.py` compares the star-side symmetry greedy policy with the
  exact dynamic-programming optimum using the tree posterior update.
- `star_dp_sensitivity.py` runs a few small star-side DP configurations to show
  how the greedy-optimal gap changes with budget and parameters.
- `random_tree_budget_gain.py` is the public version of the manuscript's
  random-tree budget-gain experiment.
- `spanning_tree_approximation.py` studies when a BFS spanning tree is a
  reasonable approximation to a sparse cyclic ER graph.
