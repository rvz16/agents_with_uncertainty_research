# Fitted-live Bayes Greedy and DP computational overhead

This profiles the compact 51-point fitted-live controller, not the ABBO `PerCriticCostDPPlanner` used for the earlier 1.45% LCB-Hard estimate.

## Setup

The policy code is profiled on CPU using the fitted critic likelihoods for each benchmark/model cell. The DP implementation builds a 51-point belief grid over a five-generation horizon (255 belief-step states). Non-SWE cells evaluate L0, L2, and L3; SWE cells evaluate L0 and L3. Timings are medians of repeated calls.

End-to-end denominators come from the completed SAGE/UQ runs. Critic and verifier actions contain explicit timers. Generation and LLM routing calls do not, so `LLM residual` is `total wall-clock - explicitly timed actions`; it is real elapsed time but is not pure engine latency.

`DP current` reconstructs the current fitted-live implementation, which builds the policy once per instance. `DP cached` builds one table per benchmark/model cell and then performs policy lookups.

## Results

| Model | Benchmark | n | Wall / instance (s) | Belief update (us) | Greedy (us/decision) | DP build (ms) | DP lookup (us) | Greedy overhead | DP current | DP cached | LLM residual | Verifier | OpenRouter L3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | lcb_easy | 101 | 20.56 | 0.34 | 2.90 | 3.39 | 1.74 | 0.00007% | 0.017% | 0.00021% | 96.1% | 3.1% | 0.6% |
| gpt-oss-20b | lcb_medium | 155 | 50.82 | 0.34 | 2.89 | 3.40 | 1.74 | 0.00004% | 0.007% | 0.00007% | 97.5% | 2.4% | 0.1% |
| gpt-oss-20b | lcb_hard | 76 | 86.72 | 0.33 | 2.89 | 3.39 | 1.74 | 0.00004% | 0.004% | 0.00008% | 96.9% | 2.9% | 0.1% |
| gpt-oss-20b | humaneval | 123 | 23.34 | 0.34 | 2.89 | 3.39 | 1.83 | 0.00007% | 0.015% | 0.00016% | 98.9% | 0.8% | 0.2% |
| gpt-oss-20b | mbpp | 284 | 21.14 | 0.33 | 2.88 | 3.39 | 1.74 | 0.00008% | 0.016% | 0.00011% | 98.3% | 1.3% | 0.2% |
| gpt-oss-20b | humanevalfix | 123 | 31.30 | 0.34 | 2.97 | 3.39 | 1.78 | 0.00006% | 0.011% | 0.00013% | 99.6% | 0.3% | 0.1% |
| gpt-oss-20b | codecontests | 124 | 87.47 | 0.34 | 2.98 | 3.54 | 1.80 | 0.00004% | 0.004% | 0.00006% | 96.8% | 3.2% | 0.1% |
| gpt-oss-20b | swebench_lite | 225 | 263.84 | 0.34 | 1.96 | 2.16 | 1.81 | 0.00001% | 0.001% | 0.00001% | 21.9% | 78.0% | 0.0% |
| gpt-oss-20b | swebench_verified | 375 | 303.06 | 0.35 | 1.95 | 2.14 | 1.79 | 0.00001% | 0.001% | 0.00001% | 20.5% | 79.4% | 0.0% |
| Qwen2.5-Coder-32B | lcb_easy | 101 | 8.55 | 0.34 | 2.97 | 3.48 | 1.79 | 0.00015% | 0.041% | 0.00050% | 74.4% | 13.8% | 10.0% |
| Qwen2.5-Coder-32B | lcb_medium | 155 | 30.86 | 0.34 | 2.98 | 3.51 | 1.77 | 0.00012% | 0.011% | 0.00015% | 70.2% | 17.7% | 10.6% |
| Qwen2.5-Coder-32B | lcb_hard | 76 | 61.01 | 0.35 | 2.99 | 3.51 | 1.79 | 0.00009% | 0.006% | 0.00013% | 60.1% | 29.0% | 9.9% |
| Qwen2.5-Coder-32B | humaneval | 123 | 8.82 | 0.34 | 2.98 | 3.51 | 1.78 | 0.00016% | 0.040% | 0.00043% | 59.2% | 25.3% | 13.8% |
| Qwen2.5-Coder-32B | mbpp | 284 | 6.62 | 0.34 | 2.98 | 3.53 | 1.82 | 0.00027% | 0.053% | 0.00036% | 74.5% | 5.7% | 18.2% |
| Qwen2.5-Coder-32B | humanevalfix | 123 | 7.85 | 0.35 | 3.00 | 3.54 | 1.81 | 0.00032% | 0.045% | 0.00057% | 76.2% | 0.6% | 22.7% |
| Qwen2.5-Coder-32B | codecontests | 124 | 117.06 | 0.34 | 2.96 | 3.50 | 1.77 | 0.00005% | 0.003% | 0.00006% | 71.3% | 21.4% | 6.4% |
| Qwen2.5-Coder-32B | swebench_lite | 225 | 280.73 | 0.35 | 1.93 | 2.20 | 1.80 | 0.00001% | 0.001% | 0.00001% | 13.5% | 85.4% | 1.1% |
| Qwen2.5-Coder-32B | swebench_verified | 375 | 304.49 | 0.34 | 1.94 | 2.17 | 1.87 | 0.00001% | 0.001% | 0.00001% | 13.9% | 86.1% | 0.0% |

## Aggregate

- **gpt-oss-20b:** observed wall-clock 213,125.7 s; Greedy CPU overhead 0.0326 s (0.00002%); DP current 4.672 s (0.002%); DP cached 0.052 s (0.00002%).
- **Qwen2.5-Coder-32B:** observed wall-clock 206,074.2 s; Greedy CPU overhead 0.0478 s (0.00002%); DP current 4.810 s (0.002%); DP cached 0.065 s (0.00003%).

Across all 18 cells, observed wall-clock is **419,199.9 s**. Greedy policy computation takes **0.0804 s (0.00002%)**. DP takes **9.482 s (0.002%)** with the current per-instance build, or **0.118 s (0.00003%)** when cached once per cell.

## Paper-ready summary

> Across nine benchmarks and two local models, a Bayesian belief update takes 0.33-0.35 microseconds and Bayes-Greedy action selection takes 1.93-3.00 microseconds per step. The DP policy is computed over 255 belief-step states in 2.14-3.54 ms; subsequent action selection takes 1.74-1.87 microseconds. Across all 18 cells, policy computation accounts for 0.00002% of measured wall-clock for Bayes Greedy and 0.002% for the current per-instance DP construction (0.00003% when the DP table is cached once per cell).

## Reproduction

```bash
conda run -n agents python experiments/orchestration_hypothesis_testing/scripts/profile_bayes_overhead_matrix.py
```
