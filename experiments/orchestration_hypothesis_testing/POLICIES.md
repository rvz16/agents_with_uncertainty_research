# Policies — a per-strategy reference

A self-contained reference for every policy / strategy in the paper:
what it does, what data it depends on, whether it needs a train/eval
split, what to expect when it wins or loses, and where it lives in the
code. Sister doc to `PLAYBOOK.md` (extending the matrix),
`COLLEAGUE_RUNBOOK.md` (running the pipeline), `EXPERIMENTAL_LOG.md`
(historical audit log), and `PRE_REGISTRATION.md` (the headline
prediction).

---

## TL;DR — master table

| # | Policy | Family | Fitted? | Mode | Needs split? |
|---|---|---|---|---|---|
| 1 | `always_verify` | reference | no | replay | no |
| 2 | `best_of_3` | unfitted baseline | no | replay | no |
| 3 | `threshold_L0` | unfitted baseline | no | replay | no |
| 4 | `threshold_L2` | unfitted baseline | no | replay | no |
| 5 | `threshold_L3` | unfitted baseline | no | replay | no |
| 6 | `fixed_pipeline` | unfitted baseline | no | replay | no |
| 7 | `bayesian_greedy` (BG) | **Bayesian controller** | **yes** | replay | **yes** |
| 8 | `bayesian_DP` (BDP) | **Bayesian controller** | **yes** | replay | **yes** |
| 9 | `Self-Refine` (real impl) | trajectory baseline | no | live | no |
| 10 | `Reflexion` (real impl) | trajectory baseline | no | live | no |
| 11 | `Self-Refine` / `Reflexion` (replay) | replay variant of 9/10 | no | replay over iter trajectory | no |
| 12 | `greedy_fitted` (GrFt) | **Bayesian controller, live** | **yes** | live | **yes** |
| 13 | `dp_fitted` (DPFt) | **Bayesian controller, live** | **yes** | live | **yes** |

Bolded policies consume fitted parameters and therefore require
train/eval discipline. Unbolded policies have no learnable parameters
and need no split — the concept doesn't apply.

---

## 0. Problem setting (the POMDP)

For each (benchmark, generator) cell we evaluate every policy on a
common stream of code-generation episodes. One episode:

- A problem instance is presented to the generator.
- The generator produces a candidate patch (or three, depending on the policy).
- The latent state is **patch correctness** $Y \in \{0, 1\}$.
- The controller can take one of these actions:
  - **`L_k`** — run critic $k \in \{L_0, L_2, L_3\}$, observe a noisy verdict $z_k \in \{\text{pass}, \text{fail}\}$, pay $c_{L_k}$.
  - **`verify`** — run the full ground-truth verifier (hidden tests, Docker harness, assert tests), pay $c_\text{ver}$, learn $Y$, episode terminates.
  - **`regenerate`** — sample a fresh candidate patch, pay $c_\text{gen}$. In replay mode this consumes the next patch from a pre-collected 3-patch pool. In live mode it makes a fresh LLM call.
  - **`stop` / `give_up`** — episode terminates with no reward and no further cost.
- Realised **utility** = $R \cdot Y_\text{verified} - \sum (\text{costs paid})$, where $R$ is the reward for a correctly-verified patch.

Two reported cost regimes (paper §5.3):

| Regime | $c_\text{gen}$ | $c_{L_0}$ | $c_{L_2}$ | $c_{L_3}$ | $c_\text{ver}$ | $R$ |
|---|---|---|---|---|---|---|
| **Slow-oracle** (function-level synthesis, repo-level patches) | 10 | 1 | 2 | 5 | 30 | 100 |
| **Fast-oracle** (bug-fixing) | 10 | 1 | 1 | 1 | 5 | 100 |

The headline metric is $\Delta_\pi = \bar{U}_\pi - \bar{U}_{\texttt{always\_verify}}$, the
mean per-instance utility difference of policy $\pi$ against the
reference policy.

---

## 1. The shared data corpus

For each (benchmark, generator) cell, **two JSONL files** drive
everything:

### `critic_results.jsonl` (the single-shot calibration corpus)

One row per `(instance, patch_id)` tuple. For LCB-hard, $n_\text{inst} \times k = 102 \times 3 = 306$ rows. Each row carries:

| Field | Meaning |
|---|---|
| `instance_id` | the problem |
| `patch_id` | which of the 3 candidates |
| `L0_syntax` | `ast.parse` succeeded |
| `L1_lint` | `ruff` clean |
| `L2_public_tests` | public tests pass |
| `L3_llm_review` | LLM judge said PASS |
| `Y` | hidden-test / verifier result |
| `cost_usd` | API cost for generating this patch |

This is the **observation log**. Every critic outcome on every patch
is logged, even ones the policy never queries. Every $Y$ is logged,
even for patches the policy never verifies. The log is exhaustive over
the pre-sampled 3-patch pool.

Generated once per cell by `lcb_calibrate.py` /
`mbpp_calibrate.py` / `humaneval_calibrate.py` /
`spot_check_generators.py`. Cost: ~$1–15 per cell depending on
generator + benchmark.

### `iter_records.jsonl` (the iterative-refinement corpus)

One row per `(instance, step)`. For an iter run of 4 refinement steps
on 30 instances, $n_\text{inst} \times 4 = 120$ rows. Each row carries:

| Field | Meaning |
|---|---|
| `instance_id` | the problem |
| `step` | 0, 1, 2, 3, 4 |
| `Y` | ground truth at this step |
| `L0/L1/L2/L3` | critic verdicts at this step |
| `diff` | the patch itself |

Used to estimate the **measured transition kernel** $P(\text{fix}\mid\text{broken}), P(\text{break}\mid\text{correct})$ via consecutive (parent → child) Y pairs.

Generated once per cell by `iter_refine_lcb.py` /
`iter_refine_swebench.py` / `iter_refine_real_baselines.py` plus
harness backfill via `populate_iter_y_verified.py`. Cost: ~$2–10 per
cell.

### What policies consume what

| Policy | Reads from `critic_results.jsonl` | Reads from `iter_records.jsonl` |
|---|---|---|
| `always_verify` | only $Y$ | – |
| `best_of_3` | only $Y$ | – |
| `threshold_L_k` | $L_k$ + $Y$ | – |
| `fixed_pipeline` | $L_0, L_2, L_3, Y$ | – |
| `bayesian_greedy` | $L_0, L_2, L_3, Y$ | (via fitted likelihoods only) |
| `bayesian_DP` | $L_0, L_2, L_3, Y$ | via fitted likelihoods AND measured kernel |
| `Self-Refine` / `Reflexion` (real impl) | – (live) | – (live) |
| `Self-Refine` / `Reflexion` (replay) | – | walks the trajectory |
| `greedy_fitted` / `dp_fitted` | (live) | via fitted parameters |

---

## 2. Fitted parameters (used by BG, BDP, GrFt, DPFt)

All four fitted policies share the same three Beta(1,1)-Laplace
estimates, all closed-form:

### Prior $P(Y=1)$ — per (benchmark, generator) cell

$$\hat{P}(Y{=}1) = \frac{n_{Y=1} + 1}{n_\text{total} + 2}$$

Computed once from `critic_results.jsonl`. Typical values: LCB-hard
$\approx 0.18$, MBPP+ $\approx 0.75$, HumanEval+ $\approx 0.85$.

### Critic likelihoods $P(z_k = \text{pass} \mid Y=y)$ — per critic $k$ and label $y$

$$\hat{P}(z_k{=}\text{pass}\mid Y{=}y) = \frac{n^{z_k=\text{pass},\,Y=y} + 1}{n^{Y=y} + 2}$$

Eight numbers per cell ($k \in \{L_0, L_2, L_3\}$, $y \in \{0, 1\}$, plus their PASS/FAIL complements). Stored in `likelihood_tables.json`.

Concretely for LCB-hard / gpt5_mini:
- $\hat{P}(L_2{=}\text{pass}\mid Y{=}1) \approx 0.95$ — strong PASS signal on correct patches
- $\hat{P}(L_2{=}\text{pass}\mid Y{=}0) \approx 0.05$ — strong FAIL signal on broken patches
- gap = 0.90 → near-oracle informativeness

### Transition kernel $P(\text{fix}\mid\text{broken}), P(\text{break}\mid\text{correct})$ — per cell

From the iter corpus, count $(Y_t, Y_{t+1})$ pairs across consecutive
refinement steps:

$$\hat{P}_\text{fix} = \frac{n_{0 \to 1} + 1}{n_{0 \to \cdot} + 2}, \quad \hat{P}_\text{break} = \frac{n_{1 \to 0} + 1}{n_{1 \to \cdot} + 2}$$

Stored in `transition_kernel.json`. Typical values: LCB-hard
$\hat{P}_\text{fix} \approx 0.07$, MBPP+ $\hat{P}_\text{fix} \approx 0.06$ (refinement rarely helps on saturated benchmarks).

All three estimates are microseconds to compute and re-fit per LOO
fold or train/eval split.

---

## 3. Per-policy detailed reference

Each entry below documents one policy in self-contained form.

---

### 3.1 `always_verify` (the reference)

**One-line:** Skip every critic; run the verifier on every patch.

**Decision rule:** `verify` always.

**Walkthrough.** On the 3-patch corpus for an instance:
1. Verify patch 0 → pay $c_\text{ver}$, observe $Y_0$.
2. If $Y_0 = 1$ → realised utility $= R - c_\text{ver}$. Episode terminates.
3. If $Y_0 = 0$ → take patch 1, repeat.
4. Continue until $Y = 1$ or all 3 patches exhausted.

Total cost: up to $3 c_\text{ver}$. Solves the instance iff at least one of the 3 patches has $Y = 1$.

**Data needs.** Only $Y$ per patch.

**Fitted parameters.** None.

**Inference mode.** Replay (reads $Y$ from `critic_results.jsonl`).

**Train/eval split needed?** No — zero learnable parameters.

**When it wins.** Regime C: high prior ($P(Y=1) \gtrsim 0.7$) and cheap verifier ($c_\text{ver}/R \lesssim 0.15$). On MBPP+ and HumanEval+ with closed-API generators, `always_verify` is hard to beat.

**When it loses.** Regime A: low prior + expensive verifier. The fixed cost $3 c_\text{ver}$ wipes out the modest reward from low base rates.

**Implementation.** `scripts/run_baseline_vs_controller.py:policy_always_verify` (one-liner).

---

### 3.2 `best_of_3`

**One-line:** Generate 3, verify each, return the best.

**Decision rule:** verify all 3 patches, take the max $Y$. In replay this just reads all 3 $Y$ values from the log.

**Walkthrough.** $Y$-values for one instance: `[True, False, True]`. `best_of_3` returns $Y = 1$, having paid $3 c_\text{ver} + 3 c_\text{gen}$ (the 3 patches are already in the pool; the $3 c_\text{gen}$ models the generation cost paid earlier).

**Data needs.** $Y$ on every patch.

**Fitted parameters.** None.

**Inference mode.** Replay.

**Train/eval split needed?** No.

**When it wins.** Niche — when you want a strict upper bound on what verification can achieve. Empirically rarely the per-cell winner because the extra $2c_\text{gen}$ over `always_verify` doesn't help if patch 0 is already correct.

**When it loses.** Almost always, except on very high-variance generators where the 3-patch diversity helps cover instances no single patch solves. On the paper's panel: 0 wins.

**Implementation.** `scripts/run_baseline_vs_controller.py:policy_best_of_N` (`N=3`).

---

### 3.3 `threshold_L0`, `threshold_L2`, `threshold_L3`

**One-line:** Run critic $L_k$ on the current patch. Verify iff $L_k$ said PASS; else regenerate (consume next patch).

**Decision rule:**
```
if L_k(current_patch) == PASS:
    verify
else:
    regenerate  # move to next patch in the 3-pool
```

**Walkthrough** (`threshold_L2` on an instance with patches that have $L_2$ = `[False, True, False]`):
1. Run $L_2$ on patch 0 → FAIL. Pay $c_{L_2}$. Move to patch 1.
2. Run $L_2$ on patch 1 → PASS. Pay $c_{L_2}$. Verify. Pay $c_\text{ver}$. Read $Y_1 = 1$.
3. Realised utility = $R - c_{L_2} - c_\text{gen} - c_{L_2} - c_\text{ver}$.

**Data needs.** $L_k$ and $Y$ per patch.

**Fitted parameters.** None.

**Inference mode.** Replay.

**Train/eval split needed?** No.

**When it wins.** Regime B: mid prior + near-oracle $L_k$. `threshold_L2` dominates regime B in our panel — when $L_2$'s gap is ≥ 0.85, one cheap public-test call gives you almost all the signal a Bayesian controller would extract.

**When it loses.** When $L_k$ has a low gap. `threshold_L0` (syntax) is rarely informative on its own. `threshold_L3` (LLM judge) sometimes wins on regime-A cells but is dominated by Bayesian gating elsewhere.

**Implementation.** `scripts/run_baseline_vs_controller.py:policy_threshold_L0/L2/L3`.

---

### 3.4 `fixed_pipeline`

**One-line:** Run $L_0$ AND $L_2$ AND $L_3$; verify iff all three pass.

**Decision rule:**
```
if L_0(p) == PASS and L_2(p) == PASS and L_3(p) == PASS:
    verify
else:
    regenerate
```

**Walkthrough.** Patch 0 has $(L_0, L_2, L_3) = (\text{True}, \text{False}, \text{True})$. AND-gate fails on $L_2$. Pay $c_{L_0} + c_{L_2}$ (early exit on first FAIL), move to patch 1.

**Data needs.** $L_0, L_2, L_3, Y$ per patch.

**Fitted parameters.** None.

**Inference mode.** Replay.

**Train/eval split needed?** No.

**When it wins.** Rarely. The AND-gate multiplies critic false-negative rates: if each critic has 5% FN on $Y=1$ patches, the AND-gate misses 15% of true positives. Useful only when critics are essentially independent + near-oracle.

**When it loses.** Most cells. Empirically dominated by `threshold_L2` (one near-oracle critic ≫ three weaker AND-gate critics).

**Implementation.** `scripts/run_baseline_vs_controller.py:policy_fixed_pipeline`.

---

### 3.5 `bayesian_greedy` (BG)

**One-line:** 1-step Bellman lookahead: at every $(b, t)$ compute $Q$-values for each action, pick the argmax, observe outcome, recurse.

**Decision rule:** at state $(b, t)$ where $b = P(Y{=}1 \mid \text{evidence so far})$ and $t$ counts regenerations:

$$Q_\text{verify}(b) = R \cdot b - c_\text{ver}$$
$$Q_\text{give-up} = 0$$
$$Q_\text{generate}(b) = -c_\text{gen} + \underbrace{P(Y{=}1) \cdot R - c_\text{ver}}_{\text{treats post-regen as a fresh draw from the prior}}$$
$$Q_\text{critic-}L_k(b) = -c_{L_k} + P(z_{L_k}{=}\text{pass}\mid b)\,Q_\text{verify}(b_\text{pass}) + P(z_{L_k}{=}\text{fail}\mid b)\,Q_\text{verify}(b_\text{fail})$$

Belief update on observing $z_{L_k}$:

$$b' = \frac{P(z_{L_k}\mid Y{=}1)\,b}{P(z_{L_k}\mid Y{=}1)\,b + P(z_{L_k}\mid Y{=}0)\,(1-b)}$$

**Walkthrough** (LCB-hard / gpt5_mini, $b_0 = 0.18$):
1. State $(b=0.18, t=0)$. Compute Q-values. Suppose $Q_{L_2}$ wins → run $L_2$ on patch 0.
2. Pre-collected $L_2 =$ FAIL. Bayes update: $b' = \frac{0.05 \cdot 0.18}{0.05 \cdot 0.18 + 0.95 \cdot 0.82} \approx 0.011$. Pay $c_{L_2}=2$.
3. State $(b=0.011, t=0)$. $Q_\text{verify} \approx -29.9$, $Q_\text{generate} \approx -27$, $Q_\text{give-up}=0$ wins. Return `give_up`.
4. Realised utility = $0 - 2 = -2$.

Contrast with `always_verify` on the same instance: 3 verifies × $c_\text{ver} = 90$, but recovers $R$ if at least one patch is correct (probability $\approx 0.45$ given prior 0.18 and 3 i.i.d. patches). Expected utility $\approx 0.45 \cdot 100 - 90 = -45$. BG saves cost by giving up cleanly.

**Defining property vs BDP:** `Q_generate` treats the post-regen state as a fresh draw from the prior. BG can't value multi-step refinement chains.

**Data needs.** $L_0, L_2, L_3, Y$ per patch + fitted likelihoods.

**Fitted parameters.** Prior $P(Y=1)$, critic likelihoods $P(z_k\mid Y=y)$. NOT the transition kernel.

**Inference mode.** Replay.

**Train/eval split needed?** **YES.** Likelihoods are fit from data; running BG on the same data biases the result optimistically. Use 25/75 fit/eval split (recommended) or LOO.

**When it wins.** Regime A: low prior + informative critics. The headline LCB-hard / gpt5_mini result is $\Delta_\text{BG} = +12.55$, pre-registered out of sample.

**When it loses.** Regimes B (where threshold_L2 already extracts most of the signal) and C (where the prior is high enough that always_verify is near-optimal).

**Implementation.** `scripts/run_baseline_vs_controller.py:BayesianController` with effective horizon 1 + the prior-rather-than-kernel regen branch; wired via `make_bayesian_policy`.

---

### 3.6 `bayesian_DP` (BDP)

**One-line:** Full backward-induction dynamic programming over a discretised $(b, t)$ state space using the **measured** transition kernel.

**Decision rule:** Precompute $V_t(b)$ for $t = H{-}1, \ldots, 0$ on a 51-point belief grid, with terminal $V_H(b) = \max(R \cdot b - c_\text{ver}, 0)$. At each $(b, t)$:

$$V_t(b) = \max\Bigl\{ Q_\text{verify}(b),\; Q_\text{give-up},\; Q_\text{generate}^{(t)}(b),\; Q_\text{critic-}L_k^{(t)}(b) \Bigr\}$$

where now $Q_\text{generate}^{(t)}(b) = -c_\text{gen} + V_{t+1}(b_\text{after regen})$ and $b_\text{after regen} = b\,(1 - P_\text{break}) + (1-b)\,P_\text{fix}$ uses the **measured kernel**, and the critic Q-value recurses on $V_t$ rather than truncating to `verify`.

At inference, the precomputed policy $\pi_t(b)$ is a lookup table.

**Walkthrough** (high-$P_\text{fix}$ cell, e.g. gpt5_mini / LCB-easy with $P_\text{fix} = 0.53$):
1. State $(b=0.40, t=0)$. The DP planner knows that if it regenerates, $b_\text{after regen} = 0.4 \cdot 0.9 + 0.6 \cdot 0.53 = 0.678$, and at the new belief the planner has positive expected utility from another `verify` cycle. So `generate` has positive value at this state.
2. BG would treat post-regen belief as the prior 0.18 here — negative value, would give up. BDP sees the chain value and regenerates.
3. After regen + new critic observations, BDP either verifies the new candidate or chains further.

**Defining property vs BG:** uses the measured kernel; can plan multi-step refinement chains (`L_0 → L_2 → verify`, `verify → fail → regen → L_2 → verify`, etc.).

**Data needs.** Same as BG plus the iter corpus (for the kernel).

**Fitted parameters.** Prior + critic likelihoods + transition kernel.

**Inference mode.** Replay.

**Train/eval split needed?** **YES.** Same reason as BG; kernel also needs to be split-fit.

**When it wins.** Regime A cells with **high measured $P_\text{fix} \gtrsim 0.15$**. Empirically rare in our panel — only 3 out of 24 cells. The clearest win is gpt5_mini / LCB-easy ($\Delta_\text{BDP} = +18.0$ vs $\Delta_\text{BG} = +5.5$).

**When it loses.** Cells with $P_\text{fix} < 0.10$ — DP collapses to BG because regen has negative EV. On most LCB-hard and SWE cells, BDP and BG produce nearly identical decisions.

**Implementation.** Same `BayesianController` class as BG, with horizon $H=3$ or $H=5$, and the kernel-based regen branch.

---

### 3.7 `Self-Refine` (real implementation)

**One-line:** Generate → verify → if fail, prompt the generator to critique its own patch + retry, up to $K$ steps.

**Decision rule:**
```
patch = generate()
for step in 1 .. K:
    if verify(patch) == PASS: return SUCCESS
    feedback = generator.critique(patch)
    patch = generator.refine(patch, feedback)
return FAIL
```

**Walkthrough.** Step 0: generate patch. Pay $c_\text{gen}$. Verify. Pay $c_\text{ver}$. If $Y=1$, terminate with utility $R - c_\text{gen} - c_\text{ver}$. Else feed back the failure trace + ask the generator to fix; pay $c_\text{gen}$ for the refinement call. Repeat up to $K=4$ steps.

**Data needs.** None at fit time. At inference time, the generator is called live.

**Fitted parameters.** None. Self-Refine is a heuristic feedback loop — the generator's behaviour adapts via in-context feedback, not via fitted weights.

**Inference mode.** Live (real impl) OR replay (over existing iter trajectory via `compute_iter_replay_baselines.py`).

**Train/eval split needed?** No. No learnable parameters.

**When it wins.** Rare in our panel. The 2-cell exceptions: gpt5_mini and haiku-4.5 on LCB-easy, where high $P_\text{fix}$ means mechanical retry actually fixes things. There Reflexion + Self-Refine outperform Bayesian gating.

**When it loses.** 22 out of 24 LCB cells. The empirical claim of the paper: the Bayesian framework beats Self-Refine in mean Δ by ~24.5 utility units on its winning cells.

**Implementation.** `scripts/iter_refine_real_baselines.py --method selfrefine` (live impl) and `scripts/compute_iter_replay_baselines.py` (replay variant).

---

### 3.8 `Reflexion` (real implementation)

**One-line:** Self-Refine plus a verbal-memory buffer of past failures, fed into the regeneration prompt.

**Decision rule:** identical to Self-Refine, except the prompt at each refinement step includes a textual buffer of past failure traces.

**Walkthrough.** Same as Self-Refine, but step 2's prompt includes "previous attempt failed because: ⟨trace 0⟩"; step 3's prompt includes both traces; etc.

**Data needs.** None at fit time. Generator is called live.

**Fitted parameters.** None.

**Inference mode.** Live OR replay.

**Train/eval split needed?** No.

**When it wins.** The 2 LCB-easy cells where mechanical retry helps. On real-impl Reflexion specifically, gpt5_mini / LCB-easy reached $\Delta = +27.7$ vs BG $+5.5$ — one of only 2 cells where a non-Bayesian baseline beat the framework.

**When it loses.** Same as Self-Refine — 22/24 LCB cells.

**Implementation.** `scripts/iter_refine_real_baselines.py --method reflexion`.

---

### 3.9 `Self-Refine` / `Reflexion` (replay variants)

**One-line:** Apply the Self-Refine or Reflexion stopping rule to a pre-collected iterative-refinement trajectory.

**Decision rule:** walk `iter_records.jsonl` row by row; stop at the first step where the policy's stopping criterion fires.

**Walkthrough.** An instance has 4 logged steps with $Y$ = `[False, False, True, True]`. Self-Refine replay walks steps 0..3, stops at step 2 (first PASS), pays $c_\text{gen} \cdot 3$ + $c_\text{ver}$, gets $R$.

**Data needs.** `iter_records.jsonl` only.

**Fitted parameters.** None.

**Inference mode.** Replay.

**Train/eval split needed?** No.

**When it wins / loses.** Used as a cost-controlled lower bound on what live Self-Refine / Reflexion can achieve. In practice the live and replay variants agree closely on cells where the feedback signal is weak. They diverge on the few cells where the live agent's feedback prompt yields better refinements than naive "regenerate from the same context" would.

**Implementation.** `scripts/compute_iter_replay_baselines.py`.

---

### 3.10 `greedy_fitted` (GrFt)

**One-line:** Live end-to-end deployment of `bayesian_greedy`'s controller with **fitted** likelihoods. Baseline is `simple` (= 1 generation + 1 verification).

**Decision rule:** Same Bellman 1-step lookahead as BG, but the controller drives a **live** generator. When the planner picks `regenerate`, it calls the LLM to produce a fresh patch. When it picks a critic, it runs the critic live on the current patch. When it picks `verify`, it runs the verifier live.

**Walkthrough** (live):
1. Generator produces patch 0. Pay $c_\text{gen}$.
2. Controller computes $Q$-values at $(b=P(Y{=}1), t=0)$. Picks `L_2`.
3. L_2 runs live on patch 0 → outputs PASS or FAIL. Pay $c_{L_2}$.
4. Bayes update on observed verdict.
5. Repeat until `verify` or `give_up`.

**Data needs (offline, for fitting):** `critic_results.jsonl` for the fit set — to estimate critic likelihoods and prior.

**Fitted parameters.** Prior + critic likelihoods. (No kernel — that's GrFt's defining "greedy" property.)

**Inference mode.** **Live.** Each `regenerate` is a fresh LLM call; each critic is a live subprocess; each `verify` is a live Docker / pytest run.

**Train/eval split needed?** **YES** — and unlike BG, this one **requires re-running the live agent** if the fit set changes. Fitting likelihoods on a different subset → controller makes different decisions → emits different fresh patches → those patches aren't in any log. Concrete cost: a fresh 25/75 run costs ~$2–8 per (benchmark, generator) cell.

**Baseline convention.** Reported as $\Delta_\text{GrFt} = U_\text{GrFt} - U_\text{simple}$ where `simple` = always-generate-then-verify (1 gen + 1 verify). This is the deployment-relevant comparison for a bug-fixing agent.

**When it wins / loses.** Same regime structure as BG; the live numbers should match BG's replay numbers within sampling noise on cells where the generator's stochasticity doesn't dramatically change which patches get produced.

**Currently reported on.** HumanEvalFix + CodeContests rows of `tab:full_results` for the closed-API panel; a colleague is running MBPP+ × 4 generators currently (results pending under the proposed 25/75 discipline).

**Implementation.** `bayesian_optimization_for_code_testing/agent-bugfix-bayes/`:
`scripts/run_humaneval_full.py` (HumanEvalFix runner); the agent loop lives in `src/abbo/realworld/agents/bayes_agent.py:_run_greedy_loop`; `_fitted` variant uses `realworld/calibration/fit.py:FittedBayesModel` for likelihoods.

---

### 3.11 `dp_fitted` (DPFt)

**One-line:** Live end-to-end deployment of `bayesian_DP`'s controller with **fitted** likelihoods AND **measured** kernel.

**Decision rule:** Same backward-induction DP as BDP, executed live. Picks the action that maximises $V_t(b)$ at each state; when that action is `regenerate`, fires a live LLM call.

**Data needs.** `critic_results.jsonl` + `iter_records.jsonl` for the fit set.

**Fitted parameters.** Prior + critic likelihoods + transition kernel.

**Inference mode.** Live.

**Train/eval split needed?** Yes — same caveats as GrFt. Re-running with a different fit set requires a fresh live run.

**Baseline convention.** $\Delta_\text{DPFt} = U_\text{DPFt} - U_\text{simple}$.

**When it wins.** Same regime as BDP — high $P_\text{fix}$ cells where multi-step refinement chains have positive EV. Currently reported wins on CodeContests claude-haiku-4.5 ($\Delta_\text{DPFt} = +23.0$) and gpt-oss-20b ($+22.0$).

**Implementation.** Same codebase as GrFt; `bayes_agent.py:_run_dp_loop` + `FittedBayesModel`.

---

## 4. Comparison summary — which policy wins which regime

| Regime | Defining condition | Winner |
|---|---|---|
| **A** | low prior $\sim 0.07$–$0.30$ + informative $L_2$ (gap $\geq 0.70$) | `bayesian_greedy` (or `bayesian_DP` if $P_\text{fix} \gtrsim 0.15$) |
| **B** | mid prior $\sim 0.3$–$0.7$ + near-oracle $L_2$ (gap $\geq 0.85$) | `threshold_L2` (one cheap test contains nearly all the signal) |
| **C** | high prior $\geq 0.75$ + cheap verifier | `always_verify` |

The Bayesian framework dominates regime A. `threshold_L2` dominates
regime B by exploiting near-oracle public-test critics. `always_verify`
dominates regime C (where the prior is high enough that blind
verification has positive expected utility on its own). Self-Refine
and Reflexion win on a tiny subset of regime-A cells with high
$P_\text{fix}$.

GrFt / DPFt extend the regime analysis to the live-agent setting,
relevant for the bug-fixing benchmarks (HumanEvalFix + CodeContests)
where the deployment scenario is an actual bug-fix agent rather than a
patch-set verifier.

---

## 5. Train/eval discipline — at a glance

| Policy | Discipline | Compute cost |
|---|---|---|
| `always_verify` | full N | $0 |
| `best_of_3` | full N | $0 |
| `threshold_L0`/`L2`/`L3` | full N | $0 |
| `fixed_pipeline` | full N | $0 |
| `bayesian_greedy` | **25/75 fit/eval, seed=42, shared across gens** | $0 (replay, microseconds for Beta-Binomial refit) |
| `bayesian_DP` | **25/75 fit/eval** | $0 (replay) |
| `Self-Refine` (real) | full N (already collected) | $0 (no re-run) |
| `Reflexion` (real) | full N (already collected) | $0 |
| `SR/Rfx` (replay) | full N | $0 |
| `greedy_fitted` | **25/75 fit/eval** | live API cost (~$2–8/cell) |
| `dp_fitted` | **25/75 fit/eval** | live API cost (~$2–8/cell) |

**Why the bolded four need a split:** they consume fitted parameters
estimated from data. Same-data fitting + same-data evaluation = small
optimistic bias (paper's LCB LOO check measured drift ≤ 0.06 utility
units, but the principle still applies and reviewers will ask about
it).

**Why the rest don't:** they have zero learnable parameters. There is
no training step to overlap with eval. Asking for a train/test split
on `always_verify` would be methodologically meaningless.

**The 25/75 convention** is shared across generators per benchmark
(paired comparison preserved): the same fit-set of instances is used
for all generators on a given benchmark, so cross-generator Δ
comparisons stay anchored on the same eval-set.

**Robustness check (paper's claim):** on the 12 LCB cells where LOO
was run, LOO results matched in-sample results to ≤ 0.06 utility
units. The 25/75 split is expected to agree with both within
~0.1 utility on the same cells. The Appendix table will report this
explicitly.

For the full reasoning behind why some policies need this and others
don't, see PR
[#3](https://github.com/rvz16/agents_with_uncertainty_research/pull/3)
comment thread (comments
[#4512839225](https://github.com/rvz16/agents_with_uncertainty_research/pull/3#issuecomment-4512839225),
[#4512892709](https://github.com/rvz16/agents_with_uncertainty_research/pull/3#issuecomment-4512892709),
[#4512991979](https://github.com/rvz16/agents_with_uncertainty_research/pull/3#issuecomment-4512991979)).

---

## 6. Implementation pointers

### Replay policies (BG, BDP, all unfitted)

- **Single entry point:** `scripts/lcb_compare.py` (works for LCB, MBPP+, HumanEval+, SWE-Bench cells).
- **Core machinery:** `scripts/run_baseline_vs_controller.py`
  - `BayesianController` — the DP planner (used by both BG and BDP, with different horizon settings).
  - `simulate_policy` — the state-machine loop that walks the `critic_results.jsonl` log.
  - `policy_always_verify`, `policy_threshold_L0/L2/L3`, `policy_fixed_pipeline`, `policy_best_of_N` — the unfitted policies as small lambda-like functions.
  - `make_bayesian_policy` — wraps `BayesianController` into a policy function for `simulate_policy`.

### Trajectory baselines

- **Live impl:** `scripts/iter_refine_real_baselines.py --method {selfrefine,reflexion}` (LCB) and `_swe.py` (SWE-Bench).
- **Replay impl:** `scripts/compute_iter_replay_baselines.py`.

### Live fitted policies (GrFt, DPFt)

- **Sister codebase:** `bayesian_optimization_for_code_testing/agent-bugfix-bayes/`.
- **Top-level runner:** `scripts/run_humaneval_full.py` (also covers CodeContests, SWE-Bench via adapter switches).
- **Agent loop:** `src/abbo/realworld/agents/bayes_agent.py`.
- **Fitting:** `src/abbo/realworld/calibration/fit.py:FittedBayesModel`.

### Notebook integration (W&B replay)

- **Statistics recomputation:** `experiments/orchestration/wandb/analysis.ipynb`
  - Cell 6 — `STAT_PRIOR` (priors)
  - Cell 7 — `STAT_LIK` (critic likelihoods)
  - Cell 8 — `STAT_KERN` (transition kernel)
  - Cell 13 — `STAT_POLICY` (all 8 replay policies via `run_policies`)
  - Cell 18 — `pc` (wide pivot of STAT_POLICY, restored from commit `b6448de6`)

Three filter lines on cells 7/8/13 to add a 25/75 split for BG/BDP — see PR #3 comment thread.

---

## Open questions / pending work

- Update `analysis.ipynb` to implement the 25/75 split for BG/BDP (3-line patch; covered in PR #3 comment thread).
- Re-run colleague's MBPP+ GrFt/DPFt runs under the 25/75 split.
- Run LOO on one MBPP+ cell as the cross-benchmark robustness check (currently only LCB cells have LOO).
- Decide whether to switch the paper's BG/BDP numbers in `tab:full_results` to the 25/75 numbers, or keep in-sample headline + cite the 25/75 result as a robustness check.

---

*Last updated: 2026-05-22 (PR #3 review iteration).*
