# K-shot active calibration + bail-risk UQ

This directory holds the K-shot sweep on CC / gpt5_mini (n=20). The
experiment tests two related ideas:

1. **K-shot active calibration**: when the DP planner picks `bail_out`,
   force one (generate, verify) pair for the first K bails per cell —
   regardless of the planner's remaining budget. Each forced (Y_t=0,
   Y_{t+1}) pair updates a Beta posterior on `p_fix_broken` (online mode)
   or is logged only (offline mode). Hypothesis: a small amount of
   guaranteed transition acquisition unlocks adaptation — online+small-K
   should beat offline+small-K.

2. **Bail-risk UQ**: action-entropy UQ from TTS was degenerate on this
   cell (top-action probability ≈ 1.0). The forced bail audits give us a
   different UQ signal: a Beta posterior on `p_catch` (probability that a
   forced refine recovers a fix), compared against the threshold
   `C_retry / R = (c_gen + c_ver) / R = 15 / 100 = 0.15`. **P(p_catch >
   threshold)** quantifies the posterior probability that bailing is
   *unsafe* (i.e., expected reward of a refine exceeds its cost).

## Mechanic

In `run_kshot_cc.py`:

- Cell-level `KShotState` carries: K, mode (online/offline), Beta(α, β)
  on `p_fix_broken`, `n_forced` counter, and a per-instance `obs_log`.
- Initial Beta(α=1.2, β=2.8) → mean = 0.3 with prior ESS = 4 (light
  prior; one-two observations move the posterior noticeably).
- When `planner.choose_action()` returns `bail_out`:
  - If `n_forced < K` and not yet forced on this episode → force one
    `generate_on_bail` + one `verify_on_bail`. Log the outcome. If
    online, update Beta. Continue (planner will likely bail again next
    step; the second bail is accepted).
  - Else accept the bail.
- The planner is re-built between instances (online mode) with the
  current Beta mean as `transition_kernel["p_fix_broken"]`. Cached by
  rounded `(p_fix, p_break)`.

## Headline numbers

### K-shot sweep (paired Δ_Ū = online − offline, bootstrap 95% CI)

| K | offline Ū | online Ū | Δ (on − off) | 95% CI |
|---|---:|---:|---:|---:|
| 0 | +13.60 | +13.60 | +0.00 | tie |
| **2** | +17.10 | **+25.60** | **+8.50** | [−1.95, +23.50] (borderline) |
| 5 | +14.85 | +9.45 | −5.40 | [−15.70, +0.15] (borderline) |
| 10 | +16.10 | +14.20 | −1.90 | [−14.15, +6.15] |
| all | +14.60 | +12.40 | −2.20 | [−20.45, +17.20] |

Small-K signal (K=2 → +8.50) is consistent with the hypothesis but
borderline at n=20. Larger K hurts online because each failed forced
refine drives the Beta posterior down, making the planner pessimistic on
subsequent instances — it bails earlier and skips marginal opportunities
that would have paid off.

Offline derivations (K=0..all) come from the K=all offline log truncated
to the first K forced refines per cell — same kernel, same forced refine
ordering, just truncated.

### Bail-risk UQ (from the K=all offline run)

Forced bail audits on n=20 instances: **12 forced refines, 2 catches,
10 misses**. Uninformative-prior posterior on `p_catch`:

- Beta(α=3, β=11)
- mean = **0.214**, 95% credible interval = [0.050, 0.454]
- Threshold C_retry / R = 15 / 100 = **0.15**
- **P(p_catch > 0.15) = 0.692**

Reading: the posterior expected catch rate (0.214) **exceeds** the
break-even threshold (0.15) by a factor of 1.4. There is a 69%
posterior probability that bail is *unsafe* on this cell — the DP
planner has been bailing on instances where, in posterior expectation,
one more refine would have positive expected utility:

$$
E[\Delta U \mid \text{refine}] = E[p_\text{catch}] \cdot R - C_\text{retry} = 0.214 \cdot 100 - 15 = +6.4 \text{ per refine}
$$

This is the **mechanism behind the K=2 online win**. Forcing the first
two bails harvested two free-money opportunities the planner had walked
away from; subsequent online updates then started moving in the wrong
direction (because most refines after the first two missed), so larger K
turned the mechanism into a drag.

## Caveats

- Borderline statistical significance at n=20 (Δ_K=2 CI = [−1.95,
  +23.50] grazes zero). The conclusion is *suggestive*, not conclusive.
- The bail-risk UQ posterior is wide (CrI [0.05, 0.45]). The 69%
  unsafe-probability is honest about that uncertainty: 31% of the
  posterior mass still says bail was safe. A larger force-all run would
  tighten this.
- Online mode's K-shot result is sensitive to *which* of the early bails
  catch — the K=5 cell had 0/5 catches due to instance-order luck, which
  flipped the posterior downward and pushed Ū to +9.45. With a
  different instance permutation the K=5 row might land closer to the
  K=2 result. (This itself is a useful finding: online adaptation is
  noisy at very small forced-refine counts.)

## Files

- `cc_live_kshot_K2_online.json` — K=2 online run (n=20)
- `cc_live_kshot_K5_online.json` — K=5 online run
- `cc_live_kshot_K10_online.json` — K=10 online run
- `cc_live_kshot_Kall_online.json` — K=all (capped at 999) online run
- `cc_live_kshot_Kall_offline.json` — K=all offline (frozen kernel)
  run; also the source of the bail-risk UQ posterior. Stores the
  per-instance `obs_log` of forced bail outcomes in
  `kshot_state.obs_log`.
- `cc_live_kshot_analysis.json` — `analyze_kshot.py` output (table +
  UQ summary in machine-readable form).

## Reproducing

Each online cell (~30 min, ~$1):

```bash
python bayesian_optimization_for_code_testing/agent-bugfix-bayes/scripts/run_kshot_cc.py \
    --model openai/gpt-5-mini \
    --results bayesian_optimization_for_code_testing/agent-bugfix-bayes/sim_results/cc_live_kshot_K2_online.json \
    --K 2 --mode online --n-tasks 20
```

The offline K=all cell (also ~30 min, ~$1) is needed for the UQ
analysis:

```bash
python bayesian_optimization_for_code_testing/agent-bugfix-bayes/scripts/run_kshot_cc.py \
    --model openai/gpt-5-mini \
    --results bayesian_optimization_for_code_testing/agent-bugfix-bayes/sim_results/cc_live_kshot_Kall_offline.json \
    --K 999 --mode offline --n-tasks 20
```

Analyze:

```bash
python bayesian_optimization_for_code_testing/agent-bugfix-bayes/scripts/analyze_kshot.py
```
