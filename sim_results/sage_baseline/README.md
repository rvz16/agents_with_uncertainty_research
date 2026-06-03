# Sage baseline — TTS self-consistency on BDP's action space

This directory holds the Sage baseline runs that put our Bayesian DP
planner against an LLM-policy comparator on **the same tool set**.

## What is Sage here

Sage (Algorithm 1 from the Sage paper) is an LLM-driven agent that, at
each step, samples an action from the model, optionally clarifies via
EVPI, then executes. For this baseline we configured Sage to be a fair
comparator to our Bayesian DP planner:

- **Same tools as BDP**: `generate`, `verify`, four critics
  (`L0_lint`, `L1_smoke_tests`, `L2_public_tests`, `L3_critic_llm`),
  and `bail`. No tool the BDP planner cannot use.
- **TTS self-consistency, N=5, temperature=0.7**: instead of one greedy
  LLM decision per step, we draw 5 samples and execute the most-frequent
  action. This is Sage's principled UQ mechanism.
- **Clarification disabled** (`tau_exec=0, max_questions=0,
  enable_escalation=False`): this is an automated benchmark; no human is
  available to answer EVPI questions, so the planner is forced to
  execute the top candidate from the TTS distribution.

The result is "Sage runtime + TTS UQ + majority-vote policy over the
same tool/observation space as BDP" — any utility gap is attributable
to **planning**, not to a richer or poorer tool set.

## How Sage differs from BDP

| Axis | BDP (ours) | Sage (this baseline) |
|---|---|---|
| Decision rule | $\arg\max_a [R(b,a) - c(a) + \mathbb{E}_{b'} V(b')]$ from offline-fitted value function | LLM samples action given step state + recent observation (majority of N=5) |
| Uncertainty | Beta posterior on kernel transitions (optionally Thompson-sampled) | TTS frequency distribution over actions |
| Cost discipline | Explicit: action cost $c(a)$ is subtracted from expected value at every step | Implicit: no cost term in the prompt; LLM keeps acting until budget exhausted |
| Bail criterion | $E[V_\text{bail}] > \max_a E[V_a]$ — bails when posterior fix-probability × reward drops below remaining action cost | Whatever the LLM majority decides (and on hard tasks the LLM rarely picks bail) |
| Exploration | Posterior sampling (Thompson) or ε-mix | TTS sampling diversity (turns out to be ≈ 0 — N=5 samples agree on a single action ~100% of the time) |

## Results

### CC / gpt5_mini (n=20, **paired** — same instance IDs as BDP, bootstrap 95% CI)

| Policy | Ū | fix rate | cost | Δ_Ū (Sage − X) | 95% CI |
|---|---:|---:|---:|---:|---:|
| **Sage (TTS, N=5)** | **−96.60** | **40%** | **136.60** | — | — |
| Always-verify (`simple`) | +0.50 | 35% | 34.50 | **−97.10** | [−126.65, −65.65] ✓ |
| Offline BDP (`dp_fitted`) | +13.60 | 20% | 6.40 | **−110.20** | [−163.25, −55.35] ✓ |
| ε-Thompson (ε=0.5) | +15.05 | 30% | 14.95 | **−111.65** | [−159.35, −59.65] ✓ |
| Thompson BDP | +22.45 | 35% | 12.55 | **−119.05** | [−173.90, −62.90] ✓ |
| Refine-on-bail (offline) | +23.10 | 40% | 16.90 | **−119.70** | [−165.10, −74.80] ✓ |
| **Refine-on-bail (online)** | **+28.10** | **45%** | 16.90 | **−124.70** | [−167.15, −83.20] ✓ |

✓ = paired 95% CI excludes 0. **All six comparators beat Sage with
statistically tight margins**, including the naive always-verify
baseline that has no critic budget at all.

### LCB-hard / haiku45 (n_test=10, unpaired — different split pool than BDP)

| Policy | Ū | fix rate | mean cost | n |
|---|---:|---:|---:|---:|
| Sage (TTS, N=5) | −20.40 | 10% | 30.40 | 10 |
| Offline BDP (`dp_fitted`) | −11.45 | 5% | 16.45 | 20 |
| Greedy BDP (`greedy_fitted`) | −11.70 | 5% | 16.70 | 20 |
| Always-verify (`simple`) | −32.75 | 10% | 42.75 | 20 |

Unpaired bootstrap 95% CI on Δ_Ū (Sage − BDP):
- vs `dp_fitted`: **−8.95** [−29.70, +18.55] (crosses 0)
- vs `simple`: **+12.35** [−11.70, +40.95] (crosses 0)

(The Sage runner uses a 50/50 train/test split of the loaded pool to
calibrate its prior, giving n_test = 10 vs BDP's n_test = 20; the two
pools share `split_seed=42` but were drawn at different pool sizes so
only 2 instance IDs overlap — we report unpaired means.)

## How good is Sage as a baseline?

**Headline:** Sage with the same tools as BDP achieves
competitive-or-better fix coverage (40% on CC vs BDP 20–45%) but burns
**3–20× the cost** because the LLM planner has no explicit
cost-vs-reward calculus. On CC every paired CI vs every BDP variant
excludes 0 — Sage is unambiguously dominated by even the simplest BDP
(`dp_fitted`) and by plain `always-verify`.

### Mechanism

Per-instance action traces show Sage:

- on LCB-hard: typically `generate → critic_L0 → critic_L1 → verify →
  think → critic_L2 → critic_L3 → generate → verify → finish`
  (2 generates + 4–6 critics + 1–2 verifies, ~30 cost units), ending at
  `finish` rather than `bail`;
- on CC: ~8 generations + 11 critics + 1–2 verifies = ~210 cost units
  on failing instances, ending at `bail` or `exhausted` only after
  burning the full 12-step budget.

The TTS top-action probability is essentially 1.0 across decisions
(the N=5 samples agree on a single action), so TTS injects zero
exploration — Sage's LLM is decisive but **decisively wrong about when
to stop spending**.

### Takeaway for the paper

The Bayesian DP planner's advantage over a same-tool LLM policy is
**not** better fix coverage — it is **knowing when not to spend**. The
explicit value function $V_t(b_t) = \max_a [R(b_t,a) - c(a) +
\mathbb{E}_{b_{t+1}} V_{t+1}]$ trades reward for cost at every step;
Sage's prompt-driven planner has no such mechanism and does the natural
thing — it keeps trying. This is exactly the gap the Bayesian framing is
designed to close, and on CC the paired CIs make it statistically tight.

## Files in this directory

- `lcb_hard_haiku45_n20.jsonl` — per-instance LCB-hard results (10 test
  + 10 train calibration), `fixed`, `total_cost`, `final_action`,
  `api_cost_usd`
- `lcb_hard_haiku45_n20.actions.jsonl` — per-step action trace for every
  episode (used to compute action mix + cost in BDP units)
- `lcb_hard_haiku45_n20.split.json` — split seed and instance IDs
  (matches `split_seed=42, train_fraction=0.5`)
- `lcb_hard_haiku45_n20.train_prior_calibration.jsonl` — calibration set
  used to fit Sage's prior $p_\text{fix}$

The CC Sage results live at
`bayesian_optimization_for_code_testing/agent-bugfix-bayes/sim_results/cc_live_sage_gpt5mini_n20.json`
(same format as the other CC runs in that directory: dict keyed by
`task_id|variant`, `variant ∈ {simple, sage}`).

## Reproducing

LCB-hard (n=20 total → 10 test after calibration split):

```bash
python experiments/orchestration_hypothesis_testing/scripts/run_sage_baseline.py \
    --tts-samples 5 --tts-temperature 0.7 --passthrough \
    --benchmark lcb_hard --generator haiku45 --n-instances 20 \
    --output sim_results/sage_baseline/lcb_hard_haiku45_n20.jsonl
```

CC (n=20 paired with BDP):

```bash
python bayesian_optimization_for_code_testing/agent-bugfix-bayes/scripts/run_codecontests_full.py \
    --llm-model openai/gpt-5-mini --variants sage simple --n-tasks 20 \
    --output bayesian_optimization_for_code_testing/agent-bugfix-bayes/sim_results/cc_live_sage_gpt5mini_n20.json
```
