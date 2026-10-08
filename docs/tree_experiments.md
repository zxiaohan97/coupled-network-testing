# Tree Experiments

The tree section has two goals: verify that the exact posterior update matches
simulation on genuine trees, and reproduce the manuscript-style tree results in
a lightweight form suitable for GitHub.

## Public Experiments

Run commands from the repository root after installing the package. The
`PYTHONPATH=src` prefix below also supports running directly from the source tree.

```bash
PYTHONPATH=src python experiments/tree/validate_exact_posterior.py
PYTHONPATH=src python experiments/tree/fixed_tree_policy_comparison.py
PYTHONPATH=src python experiments/tree/fixed_tree_budget_gain.py
PYTHONPATH=src python experiments/tree/fixed_tree_intuition_examples.py
PYTHONPATH=src python experiments/tree/star_dp_benchmark.py
PYTHONPATH=src python experiments/tree/star_dp_sensitivity.py
PYTHONPATH=src python experiments/tree/random_tree_budget_gain.py
PYTHONPATH=src python experiments/tree/plot_tree_budget_gain.py
PYTHONPATH=src python experiments/tree/spanning_tree_approximation.py
PYTHONPATH=src python experiments/tree/plot_spanning_tree_approximation.py
```

## Interpretation

`validate_exact_posterior.py` is the correctness check. Exact posterior
probabilities should agree with Monte Carlo estimates up to sampling error.

`fixed_tree_policy_comparison.py` is an illustrative comparison only. The greedy
policy minimizes the next-step expected posterior uncertainty; it is not a
global dynamic-programming optimum. The script therefore reports ties and
contact-tracing wins honestly instead of filtering them out.

`fixed_tree_budget_gain.py` is the clearest small example for interviews. It
uses a fixed 7-node tree and parameters chosen so moderate social correlation
still makes social testing informative. This is a demonstration of mechanism,
not evidence that every random tree has positive budget gain at the same
parameter values.

`fixed_tree_intuition_examples.py` checks the qualitative mechanism more
directly. It varies disease contrast, physical-test error, social-test error,
and tree shape while keeping the runs small enough for a quick public demo.

`star_dp_benchmark.py` compares the one-step greedy policy with the exact
finite-horizon dynamic-programming optimum on a star-side tree. The benchmark
uses the exact tree posterior update and a symmetry-reduced action space:
physical/social tests on the center or physical/social tests on a fresh side
node.

`star_dp_sensitivity.py` runs a few small star-side DP configurations. These
checks are useful because the optimal benchmark is exact but grows quickly with
budget.

`random_tree_budget_gain.py` is the broader policy experiment. It estimates how
many tests the greedy policy saves, relative to contact tracing, when both
policies reach the same uncertainty threshold within a finite budget.

`spanning_tree_approximation.py` is the approximation experiment. It compares a
Monte Carlo posterior policy on sparse cyclic ER graphs with the same procedure
on a BFS spanning tree extracted from each graph.

## Manuscript-Scale Settings

The public defaults are intentionally small. Manuscript-scale runs should use
more random trees or graphs, larger node counts, larger Monte Carlo sample
sizes, and the full parameter grid. Keep the output CSV files out of version
control unless they are small curated examples.

## Demo Results Generated for This Release

These results are small reproducibility checks, not manuscript-scale estimates.

Posterior validation used 20,000 Monte Carlo samples on a 7-node tree with test
sequence `(2, physical, 0)` followed by `(1, social, 1)`. The exact posterior
and simulation agreed closely: mean absolute probability error was `0.003660`,
and the maximum absolute error over all nodes, states, and conditioning steps
was `0.028865`.

The fixed-tree policy comparison used the same 7-node tree with `p=0.9`,
`q=0.1`, `r=0.5`, physical error `0.20`, and social error `0.02`. The two
policies start tied before testing. After four tests, greedy reached average
posterior disease uncertainty `0.085678`, while contact tracing reached
`0.125844`, a `31.92%` relative reduction in this illustrative case.

The fixed-tree budget-gain example used the same parameters. At threshold
`0.16`, greedy reached the target after one test while contact tracing needed
two tests, giving a `50%` budget gain. At threshold `0.14`, greedy needed two
tests and contact tracing needed three, giving a `33.33%` gain. At threshold
`0.13`, greedy again needed two tests while contact tracing needed four, giving
a `50%` gain. This example is useful because it shows the role of social
testing even at moderate social correlation `r=0.5`.

The intuition examples confirm the expected mechanism. The sweep generated 96
small threshold checks and found 8 positive budget-gain cases. All 8 positive
cases occurred in the high-contrast, high-physical-error, reliable-social-test
scenario: `p=0.9`, `q=0.1`, physical error `0.20`, social error `0.02`, and
`r=0.5`. Positive gains appeared on the balanced, star, path, and two-level
fixed trees for selected thresholds. When disease contrast was smaller,
physical tests were accurate, or social tests were noisy, greedy could still
have lower final uncertainty, but the policies often crossed the demo
thresholds at the same test budget. This supports the interpretation that
budget gain appears when social observations become more useful than noisy
physical-only contact tracing and when the threshold is small enough to expose
the difference between trajectories.

The star-side DP benchmark provides an exact optimal-policy comparison for a
highly symmetric tree. The default public run now uses a setting where social
testing is valuable: `N=12`, `p=0.8`, `q=0.2`, `r=1.0`, physical error `0.20`,
and social error `0.02`. In this regime, greedy is nearly identical to the DP
optimum: the relative gaps are `0.11%` at budget `3` and `0.06%` at budget `5`.
With stronger disease contrast (`p=0.9`, `q=0.1`) and the same testing errors,
greedy is exactly optimal at budget `3` in the demo and has only a `0.12%` gap
at budget `5`. This supports the intuition that when physical tests are noisy
and social tests are reliable, the one-step greedy policy finds the same useful
social tests that the finite-horizon optimum wants. The larger public-size
social-favorable case with `N=16` also stays close, with gaps of `0.20%` at
budget `5` and `0.12%` at budget `7`.

As a contrast, the older physical-accurate/social-noisy setting (`physical
error=0.01`, `social error=0.10`) downplays social testing. In that setting,
the public exact-tree gaps were `7.82%` at budget `3` and `15.53%` at budget
`5`. This is a useful caveat: greedy can look less close to optimal when the
best finite-horizon behavior is mostly careful physical probing rather than
obvious social testing. The manuscript reports a star-graph DP comparison at
larger scale; that should be treated as manuscript-scale evidence rather than a
quick demo.

The earlier random-tree smoke demo used five 8-node random trees, threshold
`0.15`, and max budget `5`. Greedy and contact tracing tied on average at
`r=0` and `r=0.5`; at `r=1`, the mean budget gain was `20%`. This matches the
expected qualitative story that social information helps most when social
states are strongly correlated, but the run is too small to support a formal
claim by itself. Those are historical outputs from the earlier one-based
threshold counter, not the current README figure.

## Expanded Random-Tree Sweep

The current runner uses 50 random labeled trees with 8 nodes each, infection seed
node 0, and random seed 7. It reuses the same tree seeds for every social
correlation value `0, 0.1, ..., 1` and both uncertainty targets `0.1` and `0.05`.
The selected social-informative regime is `p=0.9`, `q=0.1`, physical test error
`0.2`, and social test error `0.05`, with a maximum budget of 8 tests.

The policies and posterior updates are unchanged. Greedy selects physical or
social tests using exact one-step expected disease uncertainty. The original
tree contact-tracing baseline uses physical tests in a fixed BFS order starting
from the known infection seed, cycling if necessary. Exact-tree retests are
independent noisy measurements, unlike the fixed observation arrays used in the
general-network simulator.

Each policy's uncertainty is averaged over its full binary outcome tree. A
reported budget is the first round where that expected uncertainty is at or
below the target. It is not the expectation of history-dependent stopping times.
Targets already met by the prior now correctly have budget zero; the previous
counter started at round one. Policy traversal also now advances one round at a
time instead of recalculating every shorter history from scratch.

The summary reports the mean per-tree ratio
`(contact_budget - greedy_budget) / contact_budget` only when both budgets are
finite and the contact budget is positive. Negative gains remain in the sample.
Unreached targets are recorded as infinite and zero-denominator gains as NaN.
Counts and reachability fractions include all sampled trees, including excluded
cases, so the conditional mean is not mistaken for a population-wide effect.

Confidence intervals are approximate pointwise 95% Student-t intervals using
the sample standard deviation across eligible paired trees. They describe
random-tree sampling variation conditional on reaching the target, not posterior
Monte Carlo error or uncertainty about the disease model. An interval is
undefined when fewer than two trees qualify. Reusing trees across correlations
makes the curve paired; its individual points are not independent replicates.

The completed target-0.1 sweep gives the following selected points:

| Social correlation | Mean budget gain | Approx. 95% CI | Eligible paired trees | Greedy reaches target | Contact reaches target |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 31.2% | 25.3-37.2% | 38/50 | 50/50 | 44/50 |
| 0.5 | 37.6% | 33.8-41.5% | 43/50 | 50/50 | 43/50 |
| 1.0 | 53.3% | 51.3-55.4% | 50/50 | 50/50 | 50/50 |

At r=0, six trees already meet target 0.1 without any tests; they count as
reaching the target but have no defined relative budget gain. For target 0.05,
only 0-8 trees per correlation have two finite budgets by the cap. At r=0.5,
greedy reaches that target on 22/50 trees while contact tracing reaches it on
0/50; at r=1, the counts are 50/50 and 8/50. Thus the stricter target is mainly
informative about reachability under the cap. Its conditional gain curve uses
small, changing subsets and should not be read as a population-wide trend.

The plot leaves gain values missing when no trees qualify. At r=1 and target
0.05, all eight eligible pairs have budgets 5 and 8, so the empirical variance
and t-interval width are zero. That reflects identical observed ratios in a
small conditional sample, not certainty about the population effect.

Reproduce the complete figure and small synthetic data files:

```bash
python experiments/tree/random_tree_budget_gain.py --workers 4 \
  --output figures/selected_results/tree/tree_budget_gain_summary.csv \
  --details-output figures/selected_results/tree/tree_budget_gain_trials.csv
python experiments/tree/plot_tree_budget_gain.py \
  --input figures/selected_results/tree/tree_budget_gain_summary.csv
```

Use `--workers 1` for serial execution; the results are unchanged. For a smoke
run, keep the default ignored output location and use
`--n-trees 3 --r-values 0,0.5,1 --max-budget 3`. Budgets substantially above 8 can
be expensive because exact policy enumeration grows exponentially.

## Spanning-Tree Results

The spanning-tree approximation demo now uses 20 ER graphs per edge
probability, 1,500 Monte Carlo samples, and budget `2`. This is still a public
demo rather than a manuscript-scale run, but it is large enough to show the
expected aggregate trend. Mean relative differences increased with ER edge
probability: `3.22%`, `3.57%`, `4.74%`, and `10.25%` for edge probabilities
`0.08`, `0.12`, `0.16`, and `0.20`. The mean cyclomatic number increased at the
same time: `0.74`, `2.30`, `4.80`, and `8.45` excess edges. This makes
conceptual sense because a BFS spanning tree discards more cycle edges as the
ER graph becomes denser, so the tree posterior is approximating a graph that is
less tree-like. Small runs can still be non-monotone, so the plotted demo
includes standard-error bars.
