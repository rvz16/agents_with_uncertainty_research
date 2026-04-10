# Bayesian Orchestration for Code Generation — Math Explained

This document explains the core mathematics behind the orchestration
experiment in plain language, with worked numerical examples using the real
LiveCodeBench (LCB) calibration numbers in this repository. Read it
top-to-bottom and you should understand what every script in this folder is
computing and why.

If you only have five minutes, read sections 1, 3, 4, and 6 — those four
cover the POMDP setup, the critic model, the Bayes update, and the Bellman
equation, which is enough to read the code.

---

## 1. The problem we are trying to solve

We have a code-generation pipeline. Given a programming problem, a language
model writes a candidate "patch". Before paying the full cost of running the
hidden test suite (the **verifier**), we want to use cheaper diagnostics
(**critics** like lint, a handful of public unit tests, an LLM reviewer, a
type checker) to estimate whether the patch is actually correct. If it
looks correct, we submit it. If it looks broken, we regenerate and try
again.

Two questions:

1. Which critics should we run, in what order, and when should we stop and
   either verify or give up?
2. Is a **Bayesian decision-theoretic controller** (one that maintains a
   probability that the current patch is correct and solves a Bellman
   equation for the optimal action at each belief level) actually better
   than simple hand-coded heuristics like "run lint, if it passes submit,
   else regenerate"?

The paper ["Agentic AI Orchestration as Sequential Hypothesis
Testing"](../../orchestration_as_hypothesis_testing_paper%20%281%29.pdf)
argues yes. This experiment tests that claim on real benchmarks.

---

## 2. States, actions, rewards, and costs

The system is modeled as a Partially Observable Markov Decision Process
(POMDP).

**Hidden state.** For each patch there is a binary "ground truth"
`Y ∈ {0, 1}`: 1 means the patch is actually correct (would pass the hidden
test suite), 0 means it is broken. The controller cannot directly observe
`Y`. It only sees noisy critic outcomes and, at the end, the verifier's
final answer.

**Belief state.** Since `Y` is hidden, the controller maintains a
probability `b = P(Y = 1 | everything observed so far)`. This single number
is the "state" the controller plans against. `b = 0.7` means "I am 70%
confident the current patch is correct."

**Actions.** At every step the controller chooses one of:

| Action         | Meaning                                          | Cost             |
|----------------|--------------------------------------------------|------------------|
| `VERIFY`       | Run the full hidden test suite. Terminal.        | `c_ver`  (e.g. 20) |
| `GENERATE`     | Ask the LLM for a new patch.                     | `c_gen`  (e.g. 5)  |
| `CRITIC_Lk`    | Run diagnostic level `k` on the current patch.   | `c_crit_k` (e.g. L0=0.1, L1=1, L2=5, L3=2, L4=3) |

**Reward.** If a VERIFY action confirms `Y = 1`, we collect
`reward = 200`. Otherwise the reward is 0 (the patch fails, nothing
earned). "Give up" (submit nothing) earns 0.

**Episode utility.** The quantity we want to maximize over a whole episode
is the expected

```
U = reward * 1{final verify succeeded} - sum of all action costs
```

Concretely, if a policy runs `L2 (cost 5)` then `verify (cost 20)` on a
correct patch, the utility is `200 - 5 - 20 = +175`. If the same policy
runs `L2, gen, L2, verify` on a broken patch, the utility is
`0 - 5 - 5 - 5 - 20 = -35`.

---

## 3. How critics are described mathematically

Every critic `k` is characterized by **two probabilities**:

- `p_pass_given_correct = P(critic_k outputs PASS | Y = 1)` — the True
  Positive Rate (TPR).
- `p_pass_given_incorrect = P(critic_k outputs PASS | Y = 0)` — the False
  Positive Rate (FPR).

The **informativeness gap** is `TPR - FPR`. A gap of 1.0 means the critic
is a perfect oracle; a gap of 0.0 means the critic is pure noise. Real
critics fall somewhere between.

### The actual LCB numbers (from `lcb_likelihood_tables_v2.json`)

| Critic                | TPR  | FPR  | Gap  |
|-----------------------|------|------|------|
| `L0_syntax`           | 0.99 | 0.98 | **0.01** |
| `L1_lint`             | 0.99 | 0.98 | **0.01** |
| `L2_fast_test`        | 0.98 | 0.44 | **0.54** |
| `L3_llm_review`       | 0.60 | 0.32 | **0.28** |
| `L4_mypy`             | 0.50 | 0.50 | **0.00** |

Interpretation: L0 and L1 almost always pass, regardless of correctness,
so they carry no information on LCB. L2 (running public test cases) is
almost an oracle. L3 (asking Haiku-4.5 "is this solution correct?") is
moderately informative. L4 (mypy) adds nothing on LCB because most
solutions are functional, self-contained scripts without type annotations
that mypy has strong opinions about.

### How we estimated these numbers (calibration)

The file `calibration/generate_calibration_data.py` (and the LCB variant
`livecodebench_calibration.py`) does the following for each benchmark
instance:

1. Prompt the model to generate several candidate patches at a moderate
   temperature.
2. For each patch, run every critic and record a pass/fail outcome.
3. Run the full hidden test suite to get the true `Y`.
4. Save everything into `raw_results.jsonl` / `lcb_results_v2.jsonl`.

Then `compute_likelihoods.py` counts, per critic `k`, the number of
patches where:

```
tp_k = # (passed k  AND  Y = 1)
fp_k = # (passed k  AND  Y = 0)
fn_k = # (failed k  AND  Y = 1)
tn_k = # (failed k  AND  Y = 0)
```

and estimates

```
p_pass_given_correct_k   = (tp_k + 1) / (tp_k + fn_k + 2)
p_pass_given_incorrect_k = (fp_k + 1) / (fp_k + tn_k + 2)
```

The `+1` and `+2` are **Laplace smoothing**: they keep probabilities
strictly between 0 and 1 even for critics with very few observations,
which matters because Bayes' rule divides by these numbers. Without
smoothing, a critic that happened to pass every correct patch in our
sample would get `TPR = 1.0`, and the Bayesian update would make belief
jump to `b = 1.0` on a single pass, which is over-confident.

### Worked example

On LCB v2, `L2_fast_test` has the counts `tp = 148, fp = 27, fn = 2,
tn = 34` from 150 correct and 61 incorrect patches. The smoothed
likelihoods are

```
TPR = (148 + 1) / (150 + 2) = 149 / 152 ≈ 0.980
FPR = ( 27 + 1) / ( 61 + 2) =  28 /  63 ≈ 0.444
gap = 0.980 - 0.444 = 0.536
```

which matches the `0.9803 / 0.4444 / 0.5359` stored in the JSON file.

---

## 4. Bayes update — what a critic observation does to belief

Before observing a critic outcome, the belief is `b`. After observing
outcome `z ∈ {pass, fail}`, Bayes' rule gives the new belief:

```
b_new = P(Y = 1 | z)
      = P(z | Y = 1) * P(Y = 1) / P(z)
      = P(z | Y = 1) * b / [ P(z | Y = 1) * b + P(z | Y = 0) * (1 - b) ]
```

The denominator `P(z) = P(z | Y = 1) b + P(z | Y = 0) (1 - b)` is the
total probability of seeing that outcome. It is also the probability the
controller uses to weight the two branches in the Bellman equation — see
section 6.

### Worked example

LCB prior `b = 0.714` (base rate of correct patches). Run `L2`. Two
possible outcomes:

**If L2 passes:**

```
P(pass | Y=1) = 0.9803
P(pass | Y=0) = 0.4444
P(pass) = 0.9803 * 0.714 + 0.4444 * 0.286
        = 0.6999 + 0.1271
        = 0.8270
b_new = 0.9803 * 0.714 / 0.8270
      = 0.6999 / 0.8270
      ≈ 0.846
```

So a single L2 pass lifts belief from 0.714 to 0.846. That is a big
jump — L2 is informative.

**If L2 fails:**

```
P(fail | Y=1) = 1 - 0.9803 = 0.0197
P(fail | Y=0) = 1 - 0.4444 = 0.5556
P(fail) = 0.0197 * 0.714 + 0.5556 * 0.286
        = 0.01406 + 0.15890
        = 0.17296
b_new = 0.0197 * 0.714 / 0.17296
      = 0.01406 / 0.17296
      ≈ 0.081
```

An L2 fail crashes belief from 0.714 to 0.081. Also a big swing — this is
what we mean by "L2 has a gap of 0.54, it is very informative."

Contrast with `L0_syntax` on the same prior:

```
P(L0 pass | Y=1) = 0.9961
P(L0 pass | Y=0) = 0.9904
P(L0 pass) = 0.9961 * 0.714 + 0.9904 * 0.286 ≈ 0.9945
b_new      = 0.9961 * 0.714 / 0.9945 ≈ 0.7151
```

A pass only moves belief from 0.714 to 0.715. That is why the gap-0.01
critics are useless: seeing a pass changes nothing, so the controller
never bothers running them.

---

## 5. Transition kernel — what GENERATE does to belief

Running the generator produces a new candidate patch. Let `Y_t` be the
correctness of the old patch and `Y_{t+1}` the correctness of the new one.
The **transition kernel** is the conditional probability

```
p_fix   = P(Y_{t+1} = 1 | Y_t = 0)   # probability of fixing a broken patch
p_break = P(Y_{t+1} = 0 | Y_t = 1)   # probability of breaking a correct patch
```

From the perspective of the belief state, `GENERATE` replaces `b` with the
deterministic quantity

```
b_next = b * (1 - p_break) + (1 - b) * p_fix
```

(We are marginalizing over the random `Y_{t+1}` because we did not
*observe* the new `Y_{t+1}` — we just generated the patch.)

### Two very different kernel interpretations

The same formula can describe very different situations.

**Interpretation A: refinement chain.** The model is being given the
previous patch plus feedback and asked to improve it. Here `p_fix > base
rate` (you fix things more often than chance) and `p_break` is low
(refinements rarely break already-working code). The fixed point of the
recursion is somewhere above the prior — successive refinements walk
belief upward.

**Interpretation B: iid sampling.** Each "generate" call actually draws a
fresh iid sample from the same model at temperature > 0 on the same
problem, independent of the previous patch. Here the new `Y` is independent
of the old `Y`, so

```
p_fix   = P(Y_new = 1 | Y_old = 0) = P(Y_new = 1) = base_rate
p_break = P(Y_new = 0 | Y_old = 1) = P(Y_new = 0) = 1 - base_rate
```

Plugging into the update formula:

```
b_next = b * (1 - (1 - base_rate)) + (1 - b) * base_rate
       = b * base_rate + (1 - b) * base_rate
       = base_rate
```

So after a single generate, belief snaps to the base rate regardless of
where it started. That matches the intuition: if you redraw a fresh
independent patch, your prior about its correctness is just the base rate.

### Which one is our calibration data?

Our LCB calibration pipeline (`livecodebench_calibration.py`) generates
several patches per problem at temperature > 0, each independently. So it
is iid sampling on a *per-problem* basis. **Interpretation B is the right
kernel.**

However, the original `compute_likelihoods.py:compute_generator_transition`
counted transitions between consecutive patches in the same instance and
computed `p_fix ≈ 0.14, p_break ≈ 0.09`. That is *neither* of the two
interpretations above — it is measuring "within-problem autocorrelation"
(hard problems stay hard, easy problems stay easy). The fixed point of
that kernel happens to sit at the base rate, but between the fixed point
the gradient is tiny, so one generate only nudges belief by ~0.1. The
Bayesian controller loaded this kernel and concluded "generating is
almost a no-op, do not bother". That was Bug 2 we fixed (see
`BayesianController.from_likelihood_tables(..., iid_kernel=True)`).

---

## 6. The Bellman equation — how the controller picks actions

Given a belief `b`, how does the controller decide what to do? It
computes the expected utility of each candidate action (its **Q-value**)
and picks the best.

### Q-values

**Verify** is terminal. Expected reward minus cost:

```
Q_ver(b) = b * reward - c_ver
```

At `b = 0.714, reward = 200, c_ver = 20`:

```
Q_ver(0.714) = 0.714 * 200 - 20 = 142.8 - 20 = +122.8
```

**Generate** pays `c_gen`, then we are at belief `b_next` with one fewer
step remaining, and we will continue optimally from there (value `V`):

```
Q_gen(b) = -c_gen + V_{t+1}( b * (1 - p_break) + (1 - b) * p_fix )
```

**Critic `k`** pays `c_crit_k`, then we observe either pass or fail and
continue optimally from the two resulting beliefs, each weighted by its
probability:

```
p_pass  = b * TPR_k + (1 - b) * FPR_k
b_pass  = b * TPR_k / p_pass                        (Bayes)
b_fail  = b * (1 - TPR_k) / (1 - p_pass)            (Bayes)

Q_crit_k(b) = -c_crit_k
            + p_pass   * V_{t+1}(b_pass)
            + (1-p_pass) * V_{t+1}(b_fail)
```

The controller picks the maximum Q, plus always considers "give up"
(value 0) as a fallback:

```
V_t(b) = max ( Q_gen(b), Q_ver(b), Q_crit_0(b), ..., Q_crit_n(b), 0 )
policy_t(b) = argmax of the same list
```

### Backward induction

We cannot evaluate `V_t` directly because it depends on `V_{t+1}`. The
trick is to start from the horizon and walk backward:

```
V_T(b) = max(Q_ver(b), 0)                                (terminal)
V_{T-1}(b) = max over actions of Q_{T-1}(b), using V_T
V_{T-2}(b) = max over actions of Q_{T-2}(b), using V_{T-1}
...
V_0(b)
```

Because belief is a continuous number in `[0, 1]`, we discretize it onto
a grid of 1001 points. At each step we fill in a table of
`V_t(b_i), policy_t(b_i)` for every grid point. This is the "solve the
Bellman" pass that `BayesianController.__init__` runs once at startup.

### Worked example: why L2 dominates on LCB

At `b = 0.714` with horizon-10, the Q-values on LCB are roughly:

```
Q_ver       = 0.714 * 200 - 20             = 122.8
Q_gen       = -5 + V_next(0.714)           ≈ -5 + 123   = 118
Q_crit_L2   = -5 + 0.827 * V_next(0.846)
                 + 0.173 * V_next(0.081)   ≈ -5 + 0.827*148 + 0.173*0
                                           ≈ -5 + 122          ≈ 117
Q_crit_L3   = -2 + 0.505 * V_next(0.825)
                 + 0.495 * V_next(0.600)   ≈ -2 + 0.505*145 + 0.495*100
                                           ≈ -2 + 73 + 49.5    ≈ 120
```

With horizon 10 there is enough future budget that L2 becomes optimal
(the belief region `[0.35, 0.86)` maps to "run L2"). The controller
runs L2, observes either pass (verifies at `b = 0.846`) or fail
(generates a new patch and retries). This is essentially the same
sequence the Threshold(L2) heuristic performs by hand — which is why
on LCB the two policies are a statistical tie.

---

## 7. The simulation — comparing policies fairly

We cannot run the real benchmark for every policy variation (each LLM
call costs money and takes minutes). Instead we **replay the calibration
data as episodes**.

### Setup

Each LCB instance has `n` candidate patches (typically 3) stored in
`lcb_results_v2.jsonl`, with all their critic outcomes and ground truth
precomputed. One instance = one episode.

The simulation runs a policy on the episode:

1. Initialize `b = prior` (base rate), `patch_idx = 0`.
2. Loop until the policy decides to verify, give up, or hit the horizon.
3. When the policy says VERIFY, we look up `ground_truth` on
   `patches[patch_idx]`, charge `c_ver`, and score the episode.
4. When the policy says GENERATE, we advance to the next stored patch
   (simulating "the generator produces a new candidate"), charge `c_gen`,
   and apply the transition kernel to belief.
5. When the policy says CRITIC_Lk, we look up the stored outcome for
   that critic on the current patch, charge `c_crit_k`, and apply the
   Bayes update.

### Comparing against baselines

`run_simulation.py` compares the Bayesian controller to three baselines:

- `run_threshold_policy(level)` — "run critic `level`, if pass verify,
  if fail regenerate, up to 3 attempts". The intended "smart hand-coded
  heuristic".
- `run_fixed_pipeline()` — "always run L1 lint then verify if it passes".
  The dumbest possible baseline.
- Partial-info regime (`--exclude-l2`) — rebuild the Bayesian controller
  without access to L2, so no single critic is near-oracle.

### The two bugs we found and fixed

**Bug 1: threshold gets multiple verifies, Bayesian gets one.** The
original `run_threshold_policy` returned only on a *successful* verify.
If L2 passed but the hidden test suite returned `Y = 0`, threshold paid
the verify cost and continued the for loop, generating a new patch and
running L2+verify again. Bayesian, in contrast, returned after the first
verify regardless of outcome. So threshold played a multi-shot game
while Bayesian played a single-shot game — the two policies were not
actually comparable. Fix: make both terminate on the first verify.

**Bug 2: transition kernel measured the wrong thing.** Described in
section 5. Fix: add `iid_kernel=True` to
`BayesianController.from_likelihood_tables`, which replaces the stored
within-problem kernel with the iid kernel derived from the base rate.

After both fixes, on LCB the Bayesian and Threshold(L2) policies are
within 2 utility per episode of each other (paired difference
`-2.10 ± 0.64`) — a statistical tie, not the -10 loss we saw before.

---

## 8. The L2 noise ablation — when does Bayesian matter?

The natural worry about the LCB result is that L2 is "too good": its
gap is 0.54, which means a single L2 run is already a near-perfect
predictor. Any sensible policy ("run L2, verify on pass") will score
well. A multi-critic Bellman solver has nothing to optimize over.

The `l2_noise_ablation.py` script deliberately degrades L2 to create
a regime where no single critic dominates.

### The noise model

For a noise level `alpha ∈ [0, 1]`, each stored L2 outcome is replaced
with a fresh coin flip with probability `alpha`:

```
with prob alpha:      new L2 = random coin flip (50/50)
with prob (1-alpha):  new L2 = original L2 outcome
```

This is applied *deterministically* via a hash-seeded RNG, so every
run sees the same noised outcomes.

### Effect on the likelihoods

If we blend the original outcomes with fair coin flips at rate `alpha`:

```
new_TPR = (1 - alpha) * 0.9803 + alpha * 0.5  = 0.9803 - 0.4803 * alpha
new_FPR = (1 - alpha) * 0.4444 + alpha * 0.5  = 0.4444 + 0.0556 * alpha
new_gap = (1 - alpha) * 0.5359                = 0.5359 - 0.5359 * alpha
```

So the gap shrinks linearly from 0.54 at `alpha = 0` to 0 at
`alpha = 1`. At `alpha = 0.5` the gap drops to roughly 0.27, which is
below L3's gap of 0.28 — so L3 becomes the more informative single
critic. This is the regime where a multi-critic policy should matter.

### What we observed (single-critic Bellman)

Running the current single-critic Bayesian controller across
`alpha ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}`:

| alpha | L2 gap | L3 gap | Bayesian | Threshold(L2) | Threshold(L3) |
|-------|--------|--------|----------|---------------|---------------|
| 0.0   | 0.52   | 0.28   | +124.7   | +126.8        | +87.7         |
| 0.4   | 0.29   | 0.28   | +81.0    | +123.9        | +87.7         |
| 0.5   | 0.24   | 0.28   | +81.0    | +125.3        | +87.7         |
| 0.9   | 0.14   | 0.28   | +81.0    | +103.7        | +87.7         |

Two things jump out:

1. **Bayesian collapses at `alpha ≥ 0.4`.** It drops to +81 and stays
   there. What happened: as L2's gap shrinks below L3's, the controller
   switches its favorite critic to L3 (which is cheaper *and*
   more informative once L2 is noisy). Then on every patch it runs L3,
   and on L3-fail it generates, consuming the 10-step horizon without
   ever verifying anything.
2. **Threshold(L2) *does not* collapse with alpha.** Even at `alpha = 1`
   (L2 is pure noise) it scores +103, because the base rate of 0.714 is
   high enough that "verify any patch whose L2 happened to pass" is still
   a decent policy. You are not really using L2 as a signal anymore, you
   are using it as a randomized gate.

### The underlying limitation (why Bayesian stays stuck at +81)

Look at the single-critic Bellman policy at `alpha = 0.5`:

```
b in [0.000, 0.575): generate
b in [0.575, 0.876): critic_L3
b in [0.876, 0.904): critic_L0
b in [0.904, 1.000]: verify
```

At belief 0.714 (the prior) the controller runs L3. After L3, belief
moves to 0.76 (pass) or 0.60 (fail). Both still fall inside the
"critic_L3" region, so the controller would like to run L3 *again*.

But running L3 a second time on the same patch is a no-op — L3 is
deterministic on a fixed patch, so you would get the same answer.
The simulation's used-critics fallback catches this and substitutes
verify/generate instead, but that is a band-aid, not a plan. The
Bellman never considered "L3 then L2" as a sequence because the
state it solves over is only `(belief, step)`, not
`(belief, critics_used_on_this_patch, step)`.

### What a true multi-critic policy would do

The ideal policy at `alpha = 0.5` would be something like:

```
Run L3. If L3 passes (belief 0.76), run L2 on the same patch
to combine independent evidence. If L2 also passes (belief climbs
near 0.9), verify. If L2 fails, belief drops — either give up or
generate and start over.
```

The joint posterior after both independent passes is

```
b_both_pass = P(Y=1) * P(L3=pass|Y=1) * P(L2=pass|Y=1)
            / [ P(Y=1) * P(L3=pass|Y=1) * P(L2=pass|Y=1)
              + P(Y=0) * P(L3=pass|Y=0) * P(L2=pass|Y=0) ]
            = 0.714 * 0.58 * 0.79
            / ( 0.714*0.58*0.79 + 0.286*0.31*0.50 )
            = 0.327 / (0.327 + 0.0443)
            ≈ 0.88
```

That is genuinely higher than either single-critic posterior (0.76
and 0.825), so verifying at that point has higher expected reward for
a small extra critic cost. This is the regime where multi-critic
planning should pay off.

### The next step (in code)

To realize the multi-critic policy we have to extend the state space
of the Bellman solver from `(belief, step)` to
`(belief, used_critics_mask, step)`. The mask is a small integer —
one bit per critic that tracks which critics have already been run on
the current patch. Generating a new patch resets the mask to zero.

- Memory: `O(grid_size × 2^n_critics × horizon)` — about 320,000 states
  for 1000 grid points, 5 critics, and horizon 10. Negligible.
- Bellman update: per (belief, mask, step) cell we only consider the
  actions whose critic bits are *not* set in the mask.

This is implemented in `controller/multi_critic_controller.py` (work
in progress). Once wired into `l2_noise_ablation.py` we can re-run the
noise sweep and check whether Bayesian pulls ahead of the best
single-critic threshold in the interesting alpha range.

---

## 9. Glossary

- **POMDP** — Partially Observable Markov Decision Process. The agent
  cannot see the hidden state directly; instead it gets noisy
  observations and must plan against a belief.
- **Belief** — `b = P(Y = 1 | observations)`. A number in `[0, 1]`.
- **Prior** — the belief before any observation, usually set to the
  base rate of correct patches in the calibration dataset.
- **Base rate** — the empirical fraction of correct patches across all
  calibration data. On LCB v2 this is `255 / 357 ≈ 0.714`.
- **TPR / FPR** — True Positive Rate and False Positive Rate of a
  critic.
- **Gap** — `TPR - FPR`. A scalar measure of critic informativeness.
- **Critic** — a cheap diagnostic (lint, public tests, LLM reviewer,
  type checker) that returns pass/fail.
- **Verifier** — the full hidden test suite. Expensive, terminal,
  reveals the true `Y`.
- **Q-value** — expected utility of taking action `a` at belief `b`
  and then behaving optimally afterward.
- **Value function `V_t(b)`** — the max Q-value at belief `b` with `t`
  steps remaining.
- **Bellman equation** — the recursive definition of `V_t` in terms of
  `V_{t+1}`.
- **Backward induction** — solving the Bellman equation starting from
  `V_T` and walking back to `V_0`.
- **Laplace smoothing** — adding a small pseudocount when estimating
  probabilities from frequencies so that nothing is ever exactly 0
  or exactly 1.
- **Threshold policy** — "run critic `k`, if pass verify, if fail
  regenerate". No belief tracking.
- **Fixed pipeline** — "always run L1 lint then verify". No belief
  tracking, no adaptation.
- **Utility** — `reward * resolved - total_cost` for a single episode.

---

## 10. Where to find each concept in the code

| Concept                         | File / function                                               |
|---------------------------------|---------------------------------------------------------------|
| Critic likelihood dataclass     | `controller/bayesian_controller.py: CriticLikelihood`         |
| Transition kernel dataclass     | `controller/bayesian_controller.py: TransitionKernel`         |
| Q-value formulas                | `controller/bayesian_controller.py: _q_verify/_q_generate/_q_critic` |
| Backward induction              | `controller/bayesian_controller.py: _solve_bellman`           |
| Bayes update (standalone)       | `controller/bayesian_controller.py: update_belief`            |
| Multi-critic Bellman            | `controller/multi_critic_controller.py`                       |
| Calibration estimation          | `calibration/compute_likelihoods.py`                          |
| Simulator core loop             | `evaluation/run_simulation.py: run_bayesian_policy`           |
| Threshold baseline              | `evaluation/run_simulation.py: run_threshold_policy`          |
| Fixed-pipeline baseline         | `evaluation/run_simulation.py: run_fixed_pipeline`            |
| Cost sensitivity sweep          | `evaluation/lcb_cost_ablation.py`                             |
| L2 noise ablation               | `evaluation/l2_noise_ablation.py`                             |
| LCB raw patches                 | `calibration/data/lcb_results_v2.jsonl`                       |
| LCB likelihood tables           | `calibration/data/lcb_likelihood_tables_v2.json`              |

---

## 11. A minimal mental model (if you only remember one thing)

**Belief is a probability. Bayes updates it on every observation. The
Bellman equation tells you which action maximizes expected future reward
at the current belief. The whole experiment is asking: does solving this
optimization give you a policy that beats "run the best-looking critic
and hope"? On LCB the answer so far is "they tie, because one critic is
so good that there's nothing to optimize". The ongoing work is checking
whether that changes when you force a regime where no single critic
dominates.**
