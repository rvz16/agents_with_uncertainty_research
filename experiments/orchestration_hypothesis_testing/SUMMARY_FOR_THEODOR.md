# Empirical validation of Bayesian orchestration — status and next steps

*Status report for Theodor Papamarkou, co-author of "Agentic AI
Orchestration as Sequential Hypothesis Testing for Code Generation".*

This document summarizes the experiments run on the orchestration paper's
hypothesis so far, the infrastructure built around them, the findings
(positive and negative), and what we propose to do next. It assumes you
have read sections 2–4 of the paper.

---

## 1. What we set out to test

The paper's synthetic Section 4 benchmark shows a Bayesian two-diagnostic
controller beating a one-diagnostic heuristic with a clean gap:

| Policy                                   | Expected utility |
|------------------------------------------|-----------------:|
| Bayesian two-diagnostic `V_2`            | **130.33**       |
| One-diagnostic fixed workflow            | 89.33            |
| Gap                                      | **+41.0**        |

We wanted to answer **"does this gap survive on a real code-generation
benchmark with a real LLM generator and real critics?"** That is, is the
sequential-hypothesis-testing formulation genuinely useful in practice,
or does it collapse once the clean `P(Z=fail|H=k,T_i)` matrix is replaced
by messy, correlated critic outcomes drawn from actual code and actual
test suites.

---

## 2. Infrastructure we built

All code is under
`experiments/orchestration_hypothesis_testing/` on the
`feature/orchestration-hypothesis-testing-experiment` branch of
`rvz16/agents_with_uncertainty_research`.

- **Calibration pipeline** — runs the generator agent against benchmark
  problems at temperature > 0, produces multiple candidate patches per
  instance, runs every critic on every patch, runs the full hidden test
  suite to get ground truth `Y`, and stores everything as JSONL.
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
- **Likelihood estimation** — `calibration/compute_likelihoods.py` reads
  the JSONL files and produces `P(z_k | Y=1)` and `P(z_k | Y=0)` with
  Laplace smoothing. These are the confusion matrices the paper's
  `P(Z|H,T)` table plays.
- **Bayesian controller** — `controller/bayesian_controller.py`
  implements the single-agent Bellman solver over `(belief, step)`,
  solving the same recursion as the paper's `V_r(b)` via backward
  induction on a 1001-point belief grid. Loaded with
  `from_likelihood_tables`.
- **Multi-critic controller** — `controller/multi_critic_controller.py`
  extends the state to `(belief, used_critics_mask, step)` so the policy
  can plan multi-critic sequences on the same patch (e.g. L3 then L2 to
  combine independent evidence) rather than repeatedly querying the
  single best critic.
- **Simulator** — `evaluation/run_simulation.py` replays the calibration
  data as episodes so we can compare policies without re-paying LLM
  costs. Supports Bayesian (single and multi-critic), fixed pipeline,
  and threshold baselines at L1 / L2 / L3.
- **Ablation scripts**
  - `evaluation/lcb_cost_ablation.py` — sweep `(c_crit_l2, c_ver)` to
    check robustness to the cost model.
  - `evaluation/l2_noise_ablation.py` — deliberately degrade L2's
    informativeness by replacing a fraction of outcomes with coin
    flips, seeded per (instance, patch) for determinism.
- **Full math writeup for onboarding new contributors** —
  `experiments/orchestration_hypothesis_testing/MATH_EXPLAINED.md`.

---

## 3. Two bugs we found and fixed along the way

Both were in the comparison harness, not in the Bellman solver itself,
but they mattered enough that the first round of negative results on
LCB was actually driven by them, not by the method.

### Bug 1 — threshold baseline got multiple verifies, Bayesian got one

`run_threshold_policy` used to loop through up to 3 patches and verify
on every critic-pass, only returning on a successful verify. That meant
if L2 passed but the hidden test suite returned `Y=0`, threshold paid
the verify cost and *continued* the loop, generating a new patch and
possibly verifying again. `run_bayesian_policy` treated verify as
terminal (single commit), which is what the paper assumes. So the two
baselines were playing different games. Fix: make verify terminal for
all policies; 16 / 119 LCB episodes had been silently running >1 verify
in the old threshold. This fix alone moved the paired Bayesian−Threshold
gap on LCB from `−9.96` to `−6.18` utility.

### Bug 2 — transition kernel measured the wrong thing

`compute_generator_transition` estimated `p_fix` and `p_break` by
counting `Y` transitions between consecutive stored patches. But our
calibration patches are iid samples at temperature > 0, not actual
refinement outputs. What the counting procedure ended up measuring is
**within-problem autocorrelation** — hard problems stay hard, easy
problems stay easy — which has a fixed point near the base rate but a
tiny gradient (one generate only nudges belief by ~0.1). Loaded into
the Bellman, this kernel told the controller "generating is almost a
no-op, don't bother", which made it skip good patches in the sequence.

Fix: added an `iid_kernel=True` option to
`BayesianController.from_likelihood_tables` that replaces the stored
kernel with `p_fix = base_rate`, `p_break = 1 − base_rate`, i.e. "a
fresh generate draws an iid patch from the marginal distribution". With
this kernel belief snaps to the base rate after a single generate,
which matches what the simulator is actually doing when it advances
`patch_idx`.

After both fixes, the raw LCB result at the default cost point is:

```
Bayesian (iid kernel) = +124.75 ± 8.23
Threshold(L2)         = +126.85 ± 7.79
Paired difference     = −2.10 ± 0.64  (t = −3.29)
```

Statistically significant but a ~1.6% effect on the reward. The story
is "tie within a hair" rather than "Bayesian is broken".

---

## 4. The main empirical finding so far

On our current calibration data, **the Bayesian controller does not
reproduce the +41 gap from the paper's synthetic example on LCB or on
SWE-bench Lite**. Concretely:

### 4.1 LCB default cost point (n = 119 instances, 357 patches, base rate 0.714)

| Policy                                      | Utility per episode | Paired diff vs best threshold |
|----------------------------------------------|--------------------:|------------------------------:|
| Threshold(L2)                                | **+126.85**         | —                             |
| Fixed pipeline (L1+verify)                   | +125.42             | −1.43                         |
| Bayesian (iid kernel)                        | +124.75             | **−2.10 ± 0.64**              |
| Bayesian (calibrated kernel)                 | +120.67             | −6.18 ± 2.60                  |
| Threshold(L3 LLM reviewer)                   | +87.70              | —                             |

### 4.2 LCB cost sweep (7 configurations of `c_crit_l2, c_ver`)

Across every cost configuration we tried — lowering L2 cost, raising
verifier cost, making the verifier 2× more expensive than L2 — the
paired difference Bayesian − Threshold(L2) was always in the range
`[−2.10, −1.60]` with t-statistics around −3. **Threshold wins
marginally everywhere, never dramatically.** The ranking is completely
stable.

### 4.3 L2-noise ablation

To test the hypothesis "Bayesian wins only when no single critic
dominates", we injected controlled noise into L2 outcomes with
parameter α ∈ [0, 1]. At α we replace each L2 outcome with a fair coin
flip with probability α, which drives L2's gap linearly from 0.54 down
to 0. At roughly α = 0.5 the L2 gap drops below L3's gap (0.28), so L3
becomes the more informative single critic. This is the regime where,
in theory, a multi-critic Bayesian policy should shine.

| α    | L2 gap | L3 gap | Bay(single) | Bay(multi-critic) | Thr(L2) | Thr(L3) |
|-----:|-------:|-------:|------------:|------------------:|--------:|--------:|
| 0.00 | 0.52   | 0.28   | +124.7      | **+112.1**        | +126.8  | +87.7   |
| 0.20 | 0.43   | 0.28   | +123.1      | **+101.9**        | +124.7  | +87.7   |
| 0.40 | 0.29   | 0.28   | +81.0       | +81.0             | +123.9  | +87.7   |
| 0.50 | 0.24   | 0.28   | +81.0       | +81.0             | +125.3  | +87.7   |
| 0.70 | 0.19   | 0.28   | +81.0       | +81.0             | +116.5  | +87.7   |
| 1.00 | 0.08   | 0.28   | +81.0       | +81.0             | +103.5  | +87.7   |

Two things to notice:

1. Multi-critic Bayesian is *strictly worse* than single-critic Bayesian
   at low α, and they collapse to the same number at α ≥ 0.4.
2. Threshold(L2) stays near +120 across the whole sweep. Even at α = 1
   (L2 is pure noise), Threshold(L2) still scores +103, essentially by
   using the high base rate as a free 71% success gate.

We have not yet found a regime on this data where Bayesian beats the
best single-critic threshold.

---

## 5. Why multi-critic doesn't help on LCB — diagnosis

The multi-critic Bellman is correctly implemented (we hand-verified the
Q-values at the prior). At `b = 0.714` with `c_crit_l3 = 2`,
`c_crit_l2 = 5`:

```
Q_L3  at prior = -2 + 0.505·V(0.826, L3_used, step=1)
                   + 0.495·V(0.601, L3_used, step=1)
              = -2 + 0.505·154.50 + 0.495·139.24
              = 144.95     ← Bellman's best action here
Q_L2  at prior = -5 + 0.835·V(0.842, L2_used, step=1)
                   + 0.165·V(0.068, L2_used, step=1)
              = -5 + 0.835·149.22 + 0.165·139.24
              = 142.58
Q_verify at prior = 0.714·200 - 20 = 122.86
```

So multi-critic Bayesian picks **L3 first**, because L3 is cheaper and
leaves L2 available as a stronger follow-up if L3 passes. In
expectation (iid critics), that is the better sequence. The Bellman
gets the right answer to the optimization problem it is being asked.

**But empirically it loses by 12 utility per episode**, because of a
mismatch between the Bellman's stochastic model and the actual data:

1. The Bellman assumes critic outcomes on each new patch are fresh iid
   draws from the calibrated marginal. Within an LCB instance, they
   are not — problems that fool L3 tend to fool L2 as well, and the
   same patch has a single deterministic L3 outcome across the whole
   simulation.
2. L3 (Haiku-4.5 reviewer) has a TPR of only 0.58, i.e. it rejects 42%
   of actually-correct patches. On the subpopulation of LCB episodes
   where L3 is systematically pessimistic on a given problem,
   multi-critic burns half its step budget querying L3 before it ever
   runs L2. Single-critic, which only runs L2, never pays that tax.
3. At high α, both L2 and L3 are weak, and multi-critic correctly
   falls back to running one cheap critic and verifying — so it
   produces the same trajectories as single-critic. Hence the identical
   `+81.0` at α ≥ 0.4.

In short: **the paper's +41 gap comes from a scenario where the
diagnostics are conditionally independent given the latent class
(`H ∈ {A,B,C}`), and the test matrix is both strong and symmetric. Our
real critics on LCB are not conditionally independent, and L3 is
fail-biased, so the value-of-information argument no longer pays off
on this dataset.** We do not believe the method is wrong. We believe
the data does not satisfy its informational preconditions.

---

## 6. What we plan to check next

In priority order — each step is independent of the others, and each
one can disprove or confirm a specific concern.

### 6.1 Direct conditional-independence diagnostic on LCB (half a day)

Compute, per `Y`, the 2×2 joint confusion table of L2 and L3:

```
P(L2=pass | L3=pass, Y=1) vs P(L2=pass | L3=fail, Y=1)
P(L2=pass | L3=pass, Y=0) vs P(L2=pass | L3=fail, Y=0)
```

If the conditional distributions differ substantially (i.e. `L2 ⊥̸ L3 | Y`
on our data), that quantifies exactly how far we are from the paper's
assumption and explains the multi-critic collapse. This is a ~10-line
script and a sanity check we should have run earlier.

### 6.2 Lower base rate by switching the generator (one day)

Regenerate the calibration set using Haiku-4.5 as the generator instead
of Sonnet-4.5. Base rate should drop from 0.714 to around 0.35–0.45,
bringing us into the regime where:
- "Always verify" has negative expected utility.
- Regeneration actually matters (`Q_gen > Q_ver` becomes meaningful).
- The Bellman's multi-step planning over `(belief, step)` has real room
  to differ from Threshold.

This is closer to the operating point that motivates the whole
framework.

### 6.3 Rebuild the paper's synthetic benchmark in our framework
### and verify we reproduce the +41 gap

The paper specifies the `P(Z=fail | H, T)` matrix (the 0.9 diagonal,
0.2 off-diagonal one) and exact costs explicitly. We can instantiate
it in our simulator, run our Bayesian controller against it, and check
that we recover `130.3 vs 89.3 = +41`. If we do not, there is a bug
in our solver. If we do, our code is sound and the gap on real data
is a property of the data, not the implementation. This is maybe a
few hours of work and is the cleanest way to separate "is our code
correct?" from "does the theory apply to LCB?"

### 6.4 Add a synthetically-independent critic to LCB

For each LCB patch, compute an "independent" critic that depends on
a feature L2 and L3 don't see — e.g. code length, cyclomatic
complexity, presence of a docstring, or a trained linear probe on patch
embeddings. Calibrate its `TPR/FPR` on the stored ground truths. If
its conditional correlation with L2 and L3 given `Y` is low, we can
check whether the multi-critic Bellman now finds a sequence that beats
Threshold(L2). This would let us test the paper's thesis on LCB
without needing a whole new benchmark.

### 6.5 Characterisation result instead of a win

If none of the above produces a Bayesian win, the honest paper-shaped
story is **a characterisation theorem with empirical backing**:
"Bayesian orchestration strictly improves over single-critic thresholds
iff (a) no single critic has a gap ≥ `g*` at the given prior, (b)
critics are conditionally independent given `Y`, and (c) reward /
verify-cost ratio exceeds some threshold". We derive `g*` analytically
and show on SWE-bench Lite, LCB, and a synthetic conditionally-
independent benchmark that the empirical win correlates with those
three conditions being simultaneously met. That is still a publishable
contribution and it turns the current negative results on LCB into
part of the story rather than a problem.

---

## 7. What we would like from you

Any of the following would help us move faster:

1. **Is the paper's synthetic benchmark meant to be taken as an
   *existence proof* (multi-critic planning *can* win with the right
   data) or as a *practical claim* (it *should* win on realistic
   code-generation pipelines)?** The answer changes which of the next
   steps above is the most urgent.
2. **Do you have intuition for which real-world benchmarks have the
   conditional-independence structure** that the theory needs? We have
   ruled in/out SWE-bench Lite and LCB; we suspect competition-style
   coding might be the worst case because every critic sees the same
   tests, but we would love a pointer before we spend a day
   regenerating calibration data on a new benchmark.
3. **Willingness to review a characterisation-style reframing** if
   plan 6.5 turns out to be where we land. We would not want to write
   that up without your input on whether it matches the story the
   full paper wants to tell.

---

*All numbers in this report are reproducible from commits `c334a70`
through `8f4bbe1` on the `feature/orchestration-hypothesis-testing-experiment`
branch. Scripts and instructions live in
`experiments/orchestration_hypothesis_testing/`.*
