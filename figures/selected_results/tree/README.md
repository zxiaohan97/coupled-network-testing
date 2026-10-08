Selected synthetic figures for the README and experiment notes.

Run from the repository root after installing the package:

```bash
python experiments/tree/random_tree_budget_gain.py --workers 4 \
  --output figures/selected_results/tree/tree_budget_gain_summary.csv \
  --details-output figures/selected_results/tree/tree_budget_gain_trials.csv
python experiments/tree/plot_tree_budget_gain.py \
  --input figures/selected_results/tree/tree_budget_gain_summary.csv
python experiments/tree/spanning_tree_approximation.py
python experiments/tree/plot_spanning_tree_approximation.py
```

- `tree_budget_gain.png`: 50 paired 8-node random trees, seed 7, 11 correlation
  values from 0 to 1, targets 0.1 and 0.05, and maximum budget 8. Parameters are
  p=0.9, q=0.1, physical error=0.2, and social error=0.05. Gain means and approximate
  95% Student-t intervals use trees where both policies reach the target and
  contact tracing uses a positive budget. Reachability panels include all trees.
- `tree_budget_gain_summary.csv`: all 22 parameter/target summaries, including
  confidence intervals, eligible counts, and target reach rates.
- `tree_budget_gain_trials.csv`: per-tree budgets, graph seeds, and model settings.
  Infinite budgets mean the target was not reached by the cap; NaN gains mean
  the ratio is undefined. These are small synthetic results, not real-world data.
- `spanning_tree_approximation.png`: sparse cyclic ER networks compared with BFS
  spanning trees using the legacy hard-rejection estimator. This is an
  approximation study, not a validation of the current weighted posterior engine.

See [Tree Experiments](../../../docs/tree_experiments.md) for parameter details.

The sweep can take several minutes; `--workers` controls parallelism without
changing results. Exact outcome-tree enumeration grows exponentially with the
maximum test budget. The default run evaluates both thresholds in one traversal
per policy, and a target already met before testing has budget zero.
