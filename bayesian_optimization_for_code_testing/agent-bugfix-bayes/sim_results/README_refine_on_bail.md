# Refine-on-bail decomposition

When the DP planner picks `bail_out`, force one full refinement step
(generate + verify) before bailing. Every bail trajectory then
contributes one observed (Y_t = 0, Y_{t+1}) transition pair, so the
online estimator finally sees the data it needs.

**Two questions:**

1. How much of the win comes from **extra retry value** — i.e. simply
   correcting decisions where the planner bailed on a fix that one
   refinement would have caught?
2. How much comes from **online learning value** — i.e. the kernel
   posterior moves on those forced observations, and that better
   kernel changes downstream decisions?

We separate them by comparing offline + refine-on-bail (frozen kernel,
forced refines only correct decisions) against online + refine-on-bail
(updates posterior from forced observations).

## Cell

CC / gpt5_mini, n = 20.

## Headline (paired Δ_Ū vs offline-measured, bootstrap 95 % CI)

| Variant | n | fix % | cost | Ū | Δ vs offline | 95 % CI |
|---|---:|---:|---:|---:|---:|---:|
| offline (measured) | 20 | 20 % | 6.40 | +13.60 | — | — |
| offline + refine-on-bail | 20 | 40 % | 16.90 | +23.10 | **+9.50** | [−5.50, +26.50] |
| **online + refine-on-bail** | 20 | **45 %** | 16.90 | **+28.10** | **+14.50** | [−2.50, +33.00] |

**Within-variant paired Δ_Ū (online − offline, same refine policy,
matched instance IDs):**

- online + refine-on-bail − offline + refine-on-bail: **+5.00**
  [+0.00, +15.00] — CI grazes 0 on the lower bound

## Decomposition

The +14.50 utility/instance from online + refine-on-bail breaks down as:

- **Extra retry value: +9.50** — the +9.50 that offline + refine-on-bail
  also gets. This is purely "correcting conservative bail decisions"
  — the planner bails, the forced refine catches a hidden fix, you
  pocket the reward and move on.
- **Online learning value: +5.00** — the additional gain that online
  earns over offline on the *same* forced-observation set. This is the
  pure kernel-adaptation contribution: the Beta posterior updates
  bring the planner to slightly different decisions on subsequent
  instances, and on net those decisions are better.

So roughly **2/3 of the refine-on-bail win is retry value, 1/3 is online
adaptation**, on this cell.

## Why this matters

Refine-on-bail forces the controller to generate the (Y_t, Y_{t+1})
transitions that the diagnostic table (see `online-kernel-tests`
branch) said were missing — mean online updates jumped from ~0.1 to
~0.8 per instance. Once the data flows, the online estimator does its
job, and we get +5.00 utility/instance over offline + refine-on-bail
just from the kernel adapting. The within-variant CI on that +5.00
just touches zero at the bottom; the result is suggestive at n = 20,
not conclusive.

## Files

- `cc_live_measured_gpt5mini_n20.json` — offline baseline
- `cc_live_offline_rob_gpt5mini_n20.json` — offline + refine-on-bail
- `cc_live_online_rob_gpt5mini_n20.json` — online + refine-on-bail
- `cc_live_online_rob_gpt5mini_n50.json` — same as above at n = 50

## Caveat

n = 20 paired CIs vs offline baseline cross zero (sample is small).
The within-variant +5.00 just touches 0. Decomposition is suggestive,
not conclusive.

## Related

- `online-kernel-tests` — diagnostic showing online updates ≈ 0 without
  forced refines (the gap this experiment fills).
- `gap-gated-refine` — variant that only forces refine when the Q-gap
  between bail and refine is small; reduces cost but cancels the
  online-learning benefit.
