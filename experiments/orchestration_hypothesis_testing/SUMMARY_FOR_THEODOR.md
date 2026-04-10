# Empirical validation of Bayesian orchestration — status and next steps

*Status report for Theodor Papamarkou, co-author of "Agentic AI
Orchestration as Sequential Hypothesis Testing for Code Generation".*

This document summarizes the experiments run on the paper's hypothesis
so far, the results we currently have, and what we plan to do next. It
assumes familiarity with sections 2–4 of the paper.

---

## 1. What we set out to test

The paper's Section 4 synthetic benchmark shows a Bayesian
two-diagnostic controller beating a fixed one-diagnostic heuristic
with a clean gap:

| Policy                                   | Expected utility |
|------------------------------------------|-----------------:|
| Bayesian two-diagnostic `V_2`            | **130.33**       |
| One-diagnostic fixed workflow            | 89.33            |
| Gap                                      | **+41.0**        |

The question we wanted to answer is whether this gap survives on a
real code-generation benchmark, where the clean `P(Z=fail | H, T)`
matrix is replaced by messy, correlated critic outcomes drawn from
real code and real test suites. Concretely: is the
sequential-hypothesis-testing formulation genuinely useful in practice,
or does its advantage depend on the specific informational structure
of the toy benchmark.

---

## 2. Infrastructure built

All code sits under
`experiments/orchestration_hypothesis_testing/` on the
`feature/orchestration-hypothesis-testing-experiment` branch of
`rvz16/agents_with_uncertainty_research`.

- **Calibration pipeline** — runs the generator agent against benchmark
  problems at temperature > 0, produces multiple candidate patches per
  instance, runs every critic on every patch, runs the full hidden
  test suite to get ground truth `Y`, and stores everything as JSONL.
  - SWE-bench Lite (sympy subset, 69 instances × 3 patches):
    `calibration/data/raw_results_v3.jsonl`.
  - LiveCodeBench (119 LeetCode hard+medium instances × 3 patches):
    `calibration/data/lcb_results_v2.jsonl`.
- **Critics implemented**
  - `L0_syntax` — Python `ast.parse`.
  - `L1_lint` — Ruff with a minimal ruleset.
  - `L2_fast_test` — public test cases on LCB; fast pytest subset on
    SWE-bench.
  - `L3_llm_review` — Haiku-4.5 asked to judge correctness with a
    "VERDICT: PASS/FAIL" prompt.
  - `L4_mypy` — mypy with error-delta comparison (count before/after).
- **Likelihood estimation** — `calibration/compute_likelihoods.py`
  reads the JSONL files and produces `P(z_k | Y=1)` and
  `P(z_k | Y=0)` with Laplace smoothing. These are the confusion
  matrices that play the role of the paper's `P(Z | H, T)` table.
- **Bayesian controller** — `controller/bayesian_controller.py`
  implements the single-agent Bellman solver over `(belief, step)`,
  solving the same recursion as the paper's `V_r(b)` via backward
  induction on a 1001-point belief grid. Loaded with
  `from_likelihood_tables` and supports an `iid_kernel` option that
  derives the generator transition `p_fix = base_rate,
  p_break = 1 − base_rate` directly from the calibrated base rate,
  which matches our iid-sampling calibration regime.
- **Multi-critic controller** — `controller/multi_critic_controller.py`
  extends the state to `(belief, used_critics_mask, step)` so the
  policy can plan multi-critic sequences on the same patch (e.g. L3
  then L2 to combine independent evidence) rather than repeatedly
  querying the single best critic.
- **Simulator** — `evaluation/run_simulation.py` replays the
  calibration data as episodes so we can compare policies without
  re-paying LLM costs. Supports Bayesian (single and multi-critic),
  fixed pipeline, and threshold baselines at L1 / L2 / L3.
- **Ablation scripts**
  - `evaluation/lcb_cost_ablation.py` — sweep `(c_crit_l2, c_ver)` to
    check robustness to the cost model.
  - `evaluation/l2_noise_ablation.py` — deliberately degrade L2's
    informativeness by replacing a fraction of outcomes with fresh
    coin flips, seeded per `(instance, patch)` for determinism.
- **Full math writeup for onboarding new contributors** —
  `experiments/orchestration_hypothesis_testing/MATH_EXPLAINED.md`.

---

## 3. Current results

### 3.1 Calibrated critic likelihoods on LCB (v2)

These are the confusion matrices our Bayesian controller loads on LCB:

| Critic            | TPR  | FPR  | Gap    |
|-------------------|-----:|-----:|-------:|
| `L0_syntax`       | 0.99 | 0.98 | 0.01   |
| `L1_lint`         | 0.99 | 0.98 | 0.01   |
| `L2_fast_test`    | 0.98 | 0.44 | **0.54** |
| `L3_llm_review`   | 0.60 | 0.32 | 0.28   |
| `L4_mypy`         | 0.50 | 0.50 | 0.00   |

The LCB base rate is 0.714 (255 / 357 patches correct).

### 3.2 LCB default cost point (n = 119 instances)

| Policy                                      | Utility per episode |
|----------------------------------------------|--------------------:|
| Threshold(L2)                                | **+126.85 ± 7.79**  |
| Fixed pipeline (L1 + verify)                 | +125.42 ± 9.60      |
| Bayesian (single-critic, iid kernel)         | +124.75 ± 8.23      |
| Threshold(L1 lint)                           | +103.50 ± 8.71      |
| Threshold(L3 LLM reviewer)                   | +87.70  ± 8.93      |

Paired difference Bayesian − Threshold(L2) is `−2.10 ± 0.64` with
`t = −3.29`: a statistically consistent but tiny 1.6% effect.
Bayesian and Threshold(L2) are a near-tie.

### 3.3 LCB cost sensitivity sweep

Seven configurations of `(c_crit_l2, c_ver)` spanning
`c_l2 ∈ {2, 3, 4, 5}` and `c_ver ∈ {20, 25, 30, 40}`. Paired
Bayesian − Threshold(L2) difference was in the range
`[−2.10, −1.60]` across every configuration, with stable
t-statistics around −3. The ranking of policies never flips:
Threshold(L2) on top by a small margin, Bayesian second, Fixed
pipeline third, lower-gap thresholds far below.

### 3.4 L2 noise ablation

To test the hypothesis "Bayesian wins only when no single critic
dominates", we injected controlled noise into L2 outcomes with a
parameter α ∈ [0, 1]. At noise level α each L2 outcome is replaced by
a fair coin flip with probability α, which drives L2's gap linearly
from 0.54 at α = 0 down to 0 at α = 1. At roughly α = 0.5 the L2 gap
drops below L3's gap (0.28), so L3 becomes the more informative
single critic — the regime where, in theory, a multi-critic policy
should shine.

| α    | L2 gap | L3 gap | Bay(single) | Bay(multi-critic) | Thr(L2) | Thr(L3) |
|-----:|-------:|-------:|------------:|------------------:|--------:|--------:|
| 0.00 | 0.52   | 0.28   | +124.7      | +112.1            | +126.8  | +87.7   |
| 0.20 | 0.43   | 0.28   | +123.1      | +101.9            | +124.7  | +87.7   |
| 0.40 | 0.29   | 0.28   | +81.0       | +81.0             | +123.9  | +87.7   |
| 0.50 | 0.24   | 0.28   | +81.0       | +81.0             | +125.3  | +87.7   |
| 0.70 | 0.19   | 0.28   | +81.0       | +81.0             | +116.5  | +87.7   |
| 1.00 | 0.08   | 0.28   | +81.0       | +81.0             | +103.5  | +87.7   |

Two observations. First, the multi-critic Bellman does not improve
over single-critic on this data at any noise level. Second,
Threshold(L2) degrades gracefully with noise: even at α = 1 where L2
is pure noise, it still scores +103 because the high base rate (0.714)
turns "verify any patch whose L2 happened to pass" into a 71% success
gate without needing real information from L2.

### 3.5 Why multi-critic does not improve on LCB — diagnosis

The multi-critic Bellman's Q-values at the prior `b = 0.714` (raw LCB,
no noise) are:

```
Q_L3  at prior = -2 + 0.505·V(0.826, L3_used) + 0.495·V(0.601, L3_used)
              = 144.95    ← Bellman's best action here
Q_L2  at prior = -5 + 0.835·V(0.842, L2_used) + 0.165·V(0.068, L2_used)
              = 142.58
Q_verify at prior = 0.714·200 - 20 = 122.86
```

The multi-critic Bellman correctly picks **L3 first** because L3 is
cheaper and leaves L2 available as a stronger follow-up. In
expectation under iid critic outcomes, that is the better sequence.
The solver finds the right answer to the optimization problem it is
being asked.

Empirically, however, the L3-first policy scores +112 while L2-first
scores +125. Two factors explain the gap between the Bellman's
expected value (144.95) and the realized average:

1. **Conditional dependence.** The Bellman assumes critic outcomes on
   a new patch are fresh iid draws from the calibrated marginal.
   Within a single LCB instance, they are not — problems that fool L3
   tend to also fool L2, and the same patch has a single deterministic
   outcome for each critic across the whole simulation.
2. **L3 fail-bias.** `P(L3 pass | Y=1) = 0.58`, meaning L3 rejects
   42% of actually-correct patches. On the subpopulation of LCB
   instances where L3 is systematically pessimistic on a given
   problem, a multi-critic policy burns half its step budget querying
   L3 before it ever runs L2. A single-critic policy that never
   considered L3 does not pay this tax.

In short, the paper's +41 gap comes from a scenario where the
diagnostics are conditionally independent given the latent class, and
where the test matrix is both strong and symmetric. Our real critics
on LCB do not satisfy conditional independence, and L3 is fail-biased,
so the value-of-information argument does not cash out on this
dataset.

---

## 4. Next steps

In priority order. Each step is independent and can be executed on
its own timeline.

### 4.1 Conditional-independence diagnostic on LCB

Compute, per ground truth `Y`, the 2×2 joint confusion table of L2
and L3:

```
P(L2=pass | L3=pass, Y=1) vs P(L2=pass | L3=fail, Y=1)
P(L2=pass | L3=pass, Y=0) vs P(L2=pass | L3=fail, Y=0)
```

If the two conditional distributions differ substantially — that is,
if `L2 ⊥̸ L3 | Y` on this data — the number quantifies exactly how far
our critics are from the paper's assumption and explains the
multi-critic result. A short script; half a day of work.

### 4.2 Lower base rate by switching the generator

Regenerate the LCB calibration set using Haiku-4.5 as the generator
instead of Sonnet-4.5. We expect base rate to drop from 0.714 to
roughly 0.35–0.45, which moves us into the regime where:

- "Always verify" has negative expected utility.
- Regeneration actually matters (`Q_gen > Q_ver` becomes binding).
- The Bellman's multi-step planning over `(belief, step)` has real
  room to differ from simple threshold behaviour.

This is closer to the operating point the orchestration framework is
designed for, and is where we expect the largest potential gap. One
day of calibration plus re-running the simulator.

### 4.3 Reproduce the paper's synthetic +41 gap in our framework

Instantiate the exact `P(Z=fail | H, T)` matrix from Section 4 of the
paper (the 0.9 diagonal, 0.2 off-diagonal), the exact costs
(`C_test = 1, C_patch = 3, C_ver = 20, R = 200`), and the three-way
prior `H ∼ Uniform({A, B, C})` inside our simulator, and check that
our single-critic Bellman recovers the paper's `V_2 = 130.33,
V_1 = 89.33, gap = 41` numbers. This is a direct sanity check that
our solver is calibrated correctly against the paper's own math, and
pins down whether any remaining discrepancy on LCB is a property of
the data or of the implementation.

### 4.4 Inject a synthetically-independent third critic on LCB

For each LCB patch, compute an auxiliary critic that depends on a
feature neither L2 nor L3 has access to — e.g. cyclomatic complexity,
normalized code length, presence of a docstring, or a small linear
probe trained on patch embeddings. Calibrate its TPR and FPR against
the stored ground truths. If its conditional correlation with L2 and
L3 given `Y` is low, the multi-critic Bellman should be able to
combine it with L2 or L3 to produce a sequence that outperforms any
single-critic threshold. This lets us test the paper's thesis on LCB
without rebuilding a whole new benchmark.

### 4.5 Characterisation-result reframing

If steps 4.1–4.4 do not surface a regime where Bayesian strictly
outperforms the best single-critic threshold on real data, the
empirically honest contribution is a **characterisation**: Bayesian
orchestration strictly improves over single-critic thresholds exactly
when (a) no single critic has a gap above some analytically-derivable
`g*` at the operating prior, (b) critics are conditionally independent
given `Y`, and (c) the reward to verify-cost ratio exceeds some
threshold. We derive `g*` in closed form from the Q-value inequalities
and show, on SWE-bench Lite, LCB, and a synthetic conditionally-
independent benchmark, that the empirical win correlates tightly with
those three conditions being simultaneously met. This reframes the
current LCB numbers as part of the story rather than as a problem to
explain away.

---

*All numbers in this report are reproducible from the current head of
the `feature/orchestration-hypothesis-testing-experiment` branch.
Scripts and instructions live in
`experiments/orchestration_hypothesis_testing/`.*
