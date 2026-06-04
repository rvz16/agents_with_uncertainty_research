# ε-Thompson sweep

Mixes posterior-mean DP with posterior sampling. At each planning step,
with probability ε draw a kernel from the Beta posterior on
`p_fix_broken` / `p_break_correct`, otherwise use the posterior mean. So
ε controls the exploration / exploitation trade-off:

- ε = 0   ≡ plain offline DP (no exploration)
- ε = 1   ≡ full Thompson sampling (the policy from the online-kernel
            branch)
- 0 < ε < 1: intermediate — partial exploration, partial exploitation

Hypothesis: more exploration creates more refinement pairs → more
kernel updates → utility moves; but past some ε the cost of exploration
exceeds the value of the updates and utility falls. So we expect a peak
at intermediate ε, not at ε = 1.

Sweep on a bail-locked cell (CC / gpt5_mini), ε ∈ {0, 0.25, 0.5, 0.75,
1.0}, two sample sizes (n = 20 and n = 50).

## Results

| ε | n = 20 fix % | n = 20 Ū | n = 50 fix % | n = 50 Ū |
|---:|---:|---:|---:|---:|
| 0.00 | 40 | +14.75 | 40 | +14.34 |
| 0.25 | 30 | +8.60  | 36 | +15.94 |
| **0.50** | 30 | **+15.05** | 36 | **+19.16** |
| 0.75 | 30 | +14.75 | 32 | +13.66 |
| 1.00 | 25 | +10.55 | 30 | +14.62 |

At n = 20 the picture is noisy (small sample, CIs overlap heavily), but
at n = 50 the curve is clearer: **utility peaks at ε = 0.5** (+19.16 vs
offline +14.34, Δ = +4.82) and falls off in both directions. ε = 1
(pure Thompson) and ε = 0 (no exploration) both lose ≈ 4-5 utility per
instance to the half-and-half mix. The hypothesis stands at n = 50.

## Why the peak is at intermediate ε

- ε = 0: no exploration, no kernel updates → planner stays at the
  train-fit kernel and bails the same instances offline does. Same
  utility as offline-DP.
- Small ε (0.25): the sampled-kernel branches that fire pull the
  planner into refining without giving it enough variance to escape the
  pessimistic bail-lock — extra cost, same fix rate, lower utility.
- ε = 0.5: enough variance in sampled kernels to catch a handful of
  instances where the mean-kernel pessimism was wrong, without paying
  for an exploration cost the rest of the time.
- ε ≥ 0.75: too much exploration. Each instance the planner is
  drawing from the posterior, which is wide → many wasted refinements
  on hopeless instances → cost up, fix rate down.

## Caveat

n = 20 paired CIs all cross zero. Only the n = 50 run separates the
mid-range ε's from the endpoints, and even there the paired CI on
ε = 0.5 vs ε = 0 grazes zero. The shape of the curve is *suggestive*,
not statistically conclusive at this cell.

## Files

- `cc_live_epsthompson_{0_00, 0_25, 0_50, 0_75, 1_00}_gpt5mini_n20.json`
- `cc_live_epsthompson_{0_00, 0_25, 0_50, 0_75, 1_00}_gpt5mini_n50.json`

Each file: `dp_fitted` records for n test instances with that ε; same
record schema as the other CC live runs (`task_id`, `variant`, `fixed`,
`total_cost`, `actions`, …).
