# ABBO Bayes Greedy and DP overhead

This report profiles the same `PerCriticCostDPPlanner` used for the earlier LCB-Hard/gpt-oss-120b estimate of 1.45% overhead. It is not the smaller 51-point fitted-live controller.

Two budgets are reported: `G=5,V=2`, matching the completed two-model runs, and `G=20,V=10`, matching the earlier 120B LCB-Hard stress configuration. Both use `C=101`. `K` is the number of critics with non-null fitted likelihoods in that cell; the earlier 120B run had `K=4`.

| Model | Benchmark | n | K | Wall/instance (s) | Belief (us) | Greedy (us) | G5/V2 build (ms) | States | Current overhead | Cached overhead | G20/V10 build (s) | States | Current overhead |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | lcb_easy | 101 | 3 | 20.56 | 0.27 | 3.68 | 24.3 | 4,787 | 0.118% | 0.0012% | 0.44 | 81,285 | 2.13% |
| gpt-oss-20b | lcb_medium | 155 | 3 | 50.82 | 0.27 | 3.65 | 22.3 | 4,419 | 0.044% | 0.0003% | 0.41 | 76,490 | 0.81% |
| gpt-oss-20b | lcb_hard | 76 | 3 | 86.72 | 0.27 | 3.71 | 24.3 | 4,771 | 0.028% | 0.0004% | 0.45 | 82,369 | 0.52% |
| gpt-oss-20b | humaneval | 123 | 3 | 23.34 | 0.28 | 3.72 | 25.6 | 4,826 | 0.110% | 0.0009% | 0.45 | 81,657 | 1.91% |
| gpt-oss-20b | mbpp | 284 | 3 | 21.14 | 0.27 | 3.78 | 30.3 | 4,628 | 0.144% | 0.0005% | 0.43 | 78,854 | 2.05% |
| gpt-oss-20b | humanevalfix | 123 | 3 | 31.30 | 0.27 | 3.74 | 24.0 | 4,661 | 0.077% | 0.0006% | 0.41 | 76,569 | 1.32% |
| gpt-oss-20b | codecontests | 124 | 3 | 87.47 | 0.28 | 3.79 | 24.7 | 4,739 | 0.028% | 0.0002% | 0.45 | 81,920 | 0.52% |
| gpt-oss-20b | swebench_lite | 225 | 3 | 263.84 | 0.27 | 3.75 | 21.3 | 3,787 | 0.008% | 0.0000% | 0.33 | 62,978 | 0.13% |
| gpt-oss-20b | swebench_verified | 375 | 3 | 303.06 | 0.27 | 3.79 | 18.4 | 3,491 | 0.006% | 0.0000% | 0.29 | 54,824 | 0.09% |
| Qwen2.5-Coder-32B | lcb_easy | 101 | 3 | 8.55 | 0.27 | 3.71 | 23.1 | 4,519 | 0.270% | 0.0027% | 0.41 | 75,742 | 4.76% |
| Qwen2.5-Coder-32B | lcb_medium | 155 | 3 | 30.86 | 0.27 | 3.71 | 15.5 | 2,931 | 0.050% | 0.0004% | 0.22 | 42,618 | 0.72% |
| Qwen2.5-Coder-32B | lcb_hard | 76 | 3 | 61.01 | 0.27 | 3.77 | 20.1 | 3,813 | 0.033% | 0.0005% | 0.35 | 65,280 | 0.57% |
| Qwen2.5-Coder-32B | humaneval | 123 | 3 | 8.82 | 0.27 | 3.69 | 23.2 | 4,414 | 0.263% | 0.0022% | 0.38 | 73,871 | 4.35% |
| Qwen2.5-Coder-32B | mbpp | 284 | 3 | 6.62 | 0.27 | 3.64 | 21.5 | 4,121 | 0.325% | 0.0012% | 0.36 | 68,569 | 5.46% |
| Qwen2.5-Coder-32B | humanevalfix | 123 | 3 | 7.85 | 0.27 | 3.69 | 22.9 | 4,110 | 0.292% | 0.0025% | 0.37 | 68,948 | 4.70% |
| Qwen2.5-Coder-32B | codecontests | 124 | 3 | 117.06 | 0.27 | 3.70 | 22.6 | 4,404 | 0.019% | 0.0002% | 0.42 | 73,926 | 0.36% |
| Qwen2.5-Coder-32B | swebench_lite | 225 | 3 | 280.73 | 0.27 | 3.91 | 18.9 | 3,321 | 0.007% | 0.0000% | 0.29 | 54,183 | 0.10% |
| Qwen2.5-Coder-32B | swebench_verified | 375 | 3 | 304.49 | 0.27 | 3.71 | 17.6 | 3,231 | 0.006% | 0.0000% | 0.26 | 49,333 | 0.09% |

## Aggregate

Across all 18 cells (`419,199.9 s` observed wall-clock):

- Bayes Greedy: `0.118 s` (`0.00003%`).
- DP at matched `G=5,V=2`, rebuilt per instance: `68.9 s` (`0.016%`).
- DP at matched `G=5,V=2`, cached once per cell: `0.44 s` (`0.00010%`).
- DP at legacy `G=20,V=10`, rebuilt per instance: `1121.9 s` (`0.27%`).

## Interpretation

The earlier `1.45%` is still correct for its exact LCB-Hard/120B run: 77 per-instance builds at about 1.49 s each over 7,922 s total wall-clock. The matrix-matched budget is smaller, so its DP table has fewer reachable states and lower construction cost. The legacy column is the direct same-planner/same-budget sensitivity comparison.

## Reproduction

```bash
conda run -n agents python experiments/orchestration_hypothesis_testing/scripts/profile_abbo_bayes_overhead_matrix.py
```
