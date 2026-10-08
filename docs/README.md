# Documentation

Start with the repository [README](../README.md) for installation, the model,
and limitations. [Tree Experiments](tree_experiments.md) describes each tree
runner and the small demonstration results generated during development.

## Current and Historical Methods

The general-policy examples use `WeightedParticlePosterior`, which multiplies
particle weights by the likelihood of each observed physical or social result.
ESS and maximum weight help diagnose concentration in that fixed population.
The resampling helper is available, but the runner does not automatically
resample or regenerate particles. Longer and rarer histories still need
additional accuracy checks.

`OldRejectionMonteCarloPosterior` preserves the earlier method of retaining only
particles whose pre-sampled noisy observations match the entire history.
`monte_carlo_updates.py` imports this legacy method for compatibility. The
spanning-tree approximation experiment also retains this older estimator;
its figure is not validation of the newer weighted engine.

The complete-graph benchmark uses four outcome counts and tests fresh non-seed
nodes. Its DP is optimal for that approximate decision model and action class;
finite Monte Carlo populations need not preserve exact exchangeability. The
star benchmark instead uses exact tree posterior updates within its documented
symmetry-reduced action space.

## Scope of the Public Release

The public repository contains the reusable package, synthetic examples,
tests, small experiment runners, and selected figures. The original research
workspace's `Optimal/`, `Tree/`, `General/`, `General_Simulation/`, `Cooperation/`,
and `Waste/` folders are historical sources, not installation dependencies.
`Real/` and `RealCoop/` and their datasets are not included.

The full manuscript parameter sweeps and distributed execution setup are not
reproduced by the quick-start examples. Model extensions, particle rejuvenation,
and broader policy validation remain future work.
