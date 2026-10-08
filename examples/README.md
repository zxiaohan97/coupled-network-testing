# Examples

Run from the repository root after installing the package with
`python -m pip install -e .` (or `python -m pip install -e ".[dev]"` for tests).

```bash
python examples/tree_exact_demo.py
python examples/general_topology_comparison.py
python examples/general_noncooperation_demo.py
```

The demos use synthetic networks and do not require raw real-world data.

| Example | Settings and output |
| --- | --- |
| `tree_exact_demo.py` | A fixed 6-node tree, two observed tests, and exact expected policy uncertainty; prints results |
| `general_topology_comparison.py` | 16 nodes, 2,000 particles, 6 tests, three topologies, three target densities, two replicates, seed 2026; writes step and summary CSVs to `outputs/` |
| `general_noncooperation_demo.py` | 12-node ER networks, 1,000 particles, 6 tests, cooperation fractions 1.0/0.6/0.0, seeds starting at 2026; writes step and summary CSVs to `outputs/` |

The topology comparison takes longer than the single-tree example. These are
pipeline demonstrations, not manuscript-scale policy comparisons. Inspect ESS
and `is_stable` in the general step tables when interpreting results.

The non-cooperation example pairs the two policies on the same world within
each cooperation setting. Different cooperation settings use different graphs
and worlds, so the output is not a controlled estimate of the effect of
cooperation fraction alone. Refusals do not contribute evidence to the posterior.
