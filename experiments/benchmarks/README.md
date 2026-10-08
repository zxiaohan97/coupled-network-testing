Small greedy-versus-DP benchmarks are implemented in
`src/coupled_network_testing/benchmarks/`.

Run the star-tree benchmark from the repository root:

```bash
python experiments/tree/star_dp_benchmark.py
```

It uses exact tree posterior updates. The complete-graph benchmark is available
through `run_complete_graph_benchmark` and is exercised by
`tests/test_complete_graph_benchmark.py`. It uses Monte Carlo posterior and
transition estimates, with physical/social actions on fresh non-seed nodes.
Its DP reference is therefore approximate and limited to that action class.
