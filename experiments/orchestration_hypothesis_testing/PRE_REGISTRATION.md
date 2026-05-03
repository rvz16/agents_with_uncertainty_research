# Pre-registration: Bayesian orchestration vs heuristic baselines

This document fixes the experimental protocol **before** running any new
evaluation, so that null and negative results count as findings rather
than gaps to paper over. Items marked TBD must be resolved before the
first run.

## 1. Question

Does a Bayesian decision-theoretic controller for agentic code-bug
fixing produce higher cost-adjusted utility than a tuned heuristic
baseline family, and if so, in what regime?

We do **not** pre-commit to a headline claim. Three candidate framings
remain on the table after the first SWE-bench Lite run:

- (a) Bayesian orchestration beats heuristics.
- (b) Bayesian orchestration beats heuristics in a characterized regime.
- (c) Sequential hypothesis testing is the right framework; here is the
  regime map.

Selection rule: the framing is fixed once the first evaluation reports
results across baselines (see §6).

## 2. Hypotheses

For each (dataset × generator × controller) cell we evaluate:

- **H0:** Per-instance utility difference between the controller and the
  best heuristic baseline is zero.
- **H1 (two-sided):** It is non-zero.

The controller is treated as the test statistic; the baseline family is
the null reference. We do not assume direction in advance.

## 3. Datasets and splits

| Run | Dataset | Role | Calibration split | Eval split |
|-----|---------|------|--------------------|------------|
| R1 (first eval) | SWE-bench Lite | Primary | 70% (210 instances) | 30% (90 instances) |
| R2 | SWE-bench Verified | Secondary | reuse R1 θ | full (500) |
| R3 | LiveCodeBench | Regime check (low cost-ratio) | reuse R1 θ | full (~500) |
| R4 | HumanEval / MBPP | Regime check (no cost gap) | none (use R1 θ) | full |
| R5 | SWE-bench Pro | Final eval | new calibration on Lite-style split | 731 public |

Splits are **stratified by repo** where the dataset exposes repo IDs,
otherwise random with seed 42. Splits are computed once and committed to
the repo as `splits.json`.

Calibration instances are **disjoint** from eval instances within a
dataset. Across datasets we may reuse the calibration's θ but never
reuse calibration *instances* as eval data.

## 4. Generators

Per §D1 of the planning discussion: different generators per benchmark
run for the baseline + controller comparison. For R1 (SWE-bench Lite):

- Default generator: **TBD** (candidates: qwen2.5:7b, gpt-oss-120b,
  Claude Haiku 4.5, Claude Sonnet 4.5).
- A second generator is run if the base rate of the first is outside
  [0.30, 0.70] — extreme base rates (e.g., the LCB 0.71 case) make the
  Bellman planning regime trivial.

Generator parameters: temperature, max tokens, retry policy fixed once
and recorded in `run_config.json`.

## 5. Baseline family

Pre-committed family (per §B1):

| ID | Baseline | Action policy |
|----|----------|---------------|
| B1 | Single-shot | Generate → Verify, no retries |
| B2 | Simple retry | Generate → Verify, retry up to N=3 on fail |
| B3 | Escalating | B2 with prompt change per retry |
| B4 | Best-of-N + majority vote | Generate N candidates, vote, verify modal |
| B5 | Threshold-on-critic (tuned) | Generate → Critic → Verify if score ≥ τ else regenerate |

**Threshold tuning protocol (B5):**

- Tuning split: same 70% calibration split (no contamination of eval).
- Loss: maximize utility on the tuning split (same metric as controller).
- Search: grid τ ∈ {0.05, 0.10, …, 0.95}, then golden-section refine
  inside the winning bin.
- One threshold per critic; for multi-critic, separate variants:
  B5-AND, B5-OR, B5-sum-evidence.
- Refit when calibration set changes; never tune on eval split.

**Budget matching (per §B2):** all baselines and the controller are
budget-matched on **total cost** (sum of action costs in the cost
model defined in §8). The unit is wall-clock seconds; $-equivalents
are reported in appendix.

## 6. Controller variants

- C1: Bayesian Greedy (one-step lookahead).
- C2: Bayesian DP (full Bellman recursion over the 19K-state grid).

Both are reported separately. The Greedy>DP inversion observed on the
20-bug benchmark is treated as evidence of θ-misspecification, not
hidden.

State space, action space, bail action: per §F2/F3 — no changes from
the current implementation.

## 7. Calibration protocol

- Estimator: **Beta–Binomial** with Beta(1,1) prior (Laplace smoothing).
- Fit on the 70% calibration split, frozen for eval.
- Critics included in the headline run: **TBD** (slide 6 says only
  `mid` has a real signal gap of 0.44; `syntax` and `lint` have 0.05;
  `early` has 0.22). Default plan: include all five (syntax, lint,
  early, mid, L3 LLM reviewer); drop one at a time as ablations.
- Conditional independence is **not** assumed — we measure
  P(L_i | L_j, Y) on the calibration split and report the joint where
  it differs from the product of marginals (§D3 of the planning doc,
  TODO #1 of slide 13).

Calibration sensitivity ablation: re-run controller with (a) hand-tuned
θ from prior work, (b) fitted θ, (c) fitted θ + ±0.1 per-cell noise.
Reported in the main results table.

## 8. Cost model

Two cost models reported side-by-side:

| Action | Toy cost | Real cost (SWE-bench Lite) |
|--------|----------|----------------------------|
| Generate (LLM call) | 1 | wall-time seconds (median) |
| Critic (lint, syntax, fast test) | 0.1 | 0.1–5 s |
| Critic (LLM-based, e.g., L3) | 1 | wall-time seconds |
| Verify (full test suite in Docker) | 5 | 30–120 s |
| Bail | 0 | 0 |

Reward for correct final patch: R = 100 (toy) and R = $-equivalent of
verify cost × 100 (real). Real costs are **measured during the run**
and reported per (dataset, generator) in `cost_table.json`.

## 9. Metrics

**Reported on every run:**

1. Fix rate = fraction of episodes ending in a verified-correct patch.
2. Mean utility per episode = R · 1[correct] − Σ action costs.
3. Pass-vs-cost curve = cumulative pass rate vs cumulative cost across
   the eval split, sorted by cost.
4. Verifier efficiency = average verifier calls per solved problem.
5. Bail rate (controllers only).

**Headline metric for run R1:** all four reported. Headline metric for
R2 onward is **fixed** to the metric chosen post-R1.

## 10. Statistical test

- Paired bootstrap CI on per-instance utility difference (controller
  − best baseline), 10 000 resamples, 95% CI.
- Decision rule: H0 is rejected if 95% CI excludes 0.
- Minimum detectable effect (MDE): we declare in advance that an
  absolute utility difference < 1.0 (at R = 100, this is < 1% of
  reward) is "practically null" even if statistically detectable.
- Multiple comparisons: when comparing the controller against multiple
  baselines, we report the comparison against the **best** baseline by
  tuning-split utility, and apply Holm correction to the family.

## 11. Sample size

- R1 (SWE-bench Lite eval split = 90 instances): with σ ≈ 8.0 utility
  per instance (slide 11 LCB benchmark), N = 90 gives MDE ≈ 1.7 utility
  at α = 0.05, power = 0.8. This is at the edge of our practical-null
  threshold.
- If R1 is inconclusive, we extend to the full Lite (300 instances),
  raising MDE to ~0.9.
- For R5 (SWE-bench Pro), N = 731 gives MDE ≈ 0.6.

## 12. What counts as a positive / negative result

| Outcome | Interpretation | Implied framing |
|---------|----------------|------------------|
| Controller > best baseline by ≥ 1.0 utility, 95% CI excludes 0 | Strong positive | (a) or (b) |
| Controller > best baseline, 95% CI excludes 0, effect < 1.0 | Statistically positive, practically null | (b) or (c) |
| 95% CI includes 0 | Null result | (b) or (c) — regime characterization |
| Controller < best baseline, 95% CI excludes 0 | Negative | (c) — regime map shows this benchmark is outside the regime |

All four outcomes are reported in the paper's main table. None are
papered over. This is the entire point of pre-registration.

## 13. Stop conditions and amendments

- Amendments to this document are allowed but must be (a) committed
  to git with a dated entry in §14 below and (b) noted in the paper's
  appendix.
- We do **not** amend the metric, baseline family, or statistical test
  after seeing eval results from a given run.

## 14. Amendments log

(empty — initial version)

## 15. Open items (TBD before first run)

- §4: pick R1 default generator.
- §7: confirm critic stack for R1 headline (drop dead critics or keep
  all five and ablate?).
- §H3 of planning doc: author ownership (calibration, controller,
  experiments, writing).
- §H2: confirm NeurIPS 2026 deadline as the binding scope ceiling.
