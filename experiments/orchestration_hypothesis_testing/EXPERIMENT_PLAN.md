# Mapping "Orchestration as Hypothesis Testing" to Real Tasks

## 1. What the paper proposes (recap)

Theo's paper frames agentic code generation as a **POMDP with three worker agents**:

- **Generator** $a_\text{gen}$: LLM that refines the current candidate. Cost $C_\text{gen}$.
- **Critic** $a_\text{crit}$: cheap diagnostic sensor. Returns noisy observation $z \sim P(z \mid Y)$. Cost $C_\text{crit}$.
- **Verifier** $a_\text{ver}$: ground-truth oracle. Returns $Y_t = \mathcal{O}(c_t, x)$. Cost $C_\text{ver}$.

A **Bayesian controller** maintains belief $b_t = P(Y_t = 1 \mid \mathcal{H}_t)$ and at each step picks the action that maximizes the Bellman equation:

$$V(b_t) = \max\{Q_\text{gen}(b_t),\; Q_\text{crit}(b_t),\; Q_\text{ver}(b_t)\}$$

The core claim: this controller outperforms heuristic threshold-based orchestration because two states with the same $b_t$ can have different optimal actions depending on *why* the belief is low (epistemic uncertainty vs. confirmed defect).

The paper demonstrates this on a **synthetic benchmark** (3 bug classes, 3 diagnostic tests, closed-form utilities). Expected utility gap: **130.3 vs 89.3** (Bayesian vs heuristic).

---

## 2. Connection to existing SAGE codebase

`bayesian_dp.py` already adapts the paper's `example.py` to SAGE's question-selection framework. The mapping documented in `BAYESIAN_DP.md`:

| Paper (Theo)               | SAGE (current)                     |
|----------------------------|-------------------------------------|
| Hypotheses H = {A,B,C}    | Tool-call candidates               |
| Diagnostic tests T1,T2,T3 | Clarification questions             |
| P(fail\|H,T) matrix       | Domain-based resolution probability |
| Binary outcome             | "answer resolves" vs "doesn't"      |
| C_TEST, C_PATCH, C_VER, R | CostModel presets                   |

But the paper's **code generation framing** (generator/critic/verifier) maps to a *different* real task than SAGE's current tool-call disambiguation. The paper is about **iterative code refinement under verification cost constraints**, not about which tool to call.

---

## 3. Target task: SWE-Bench Pro (primary) + SWE-bench Lite/Verified + LiveCodeBench (secondary)

### 3.1 Why SWE-bench

SWE-bench Verified (500 instances, human-validated) and SWE-Bench Pro (731 public instances, contamination-resistant) are the strongest fit because:

1. **Verification is genuinely expensive.** Running the full test suite requires: clone repo → install dependencies → apply patch → execute tests. This takes 30–120s per instance and often fails due to environment issues. Cost $C_\text{ver}$ is real.

2. **Cheap diagnostics exist and are noisy.** Linting, type checking, syntax validation, running a single fast test — these take 1–5s and provide partial signal. A lint failure is strong evidence of incorrectness; a lint pass is weak evidence of correctness. This asymmetry is exactly the confusion matrix structure in the paper.

3. **Generation is stochastic with moderate cost.** Each LLM refinement call costs $0.01–0.10 (API) and 5–15s (latency). The probability that refinement improves the patch is unknown and varies by problem difficulty.

4. **The existing codebase already has SWE-bench infrastructure** (`different_agents/v4/langgraph_sage_agent_v4_swe.py`, `different_agents/evaluations/run_swebench_eval.py`).

### 3.2 Why SWE-Bench Pro (primary evaluation benchmark)

SWE-Bench Pro (Deng et al., 2025) is a harder, contamination-resistant benchmark:

1. **1,865 problems** from 41 actively maintained repos (731 public, 858 held-out, 276 commercial). Multi-language: Python, JavaScript, TypeScript, Go.
2. **Long-horizon tasks**: patches span mean 4.1 files and 107.4 lines of code (vs 4.5 median LOC in SWE-bench Lite). This makes verification genuinely expensive (Docker environments, full dependency resolution).
3. **Contamination-resistant**: GPL-licensed repos + commercial codebases from startups. Not in LLM training data.
4. **Human-augmented**: each task has problem statement + requirements + interface spec, reducing false negatives in test evaluation.
5. **Best models score <45%** (Claude Sonnet 4.5: 43.6%, GPT-5: 41.8%) — plenty of room for orchestration improvements.
6. **Rich failure taxonomy** (Table 4): wrong solution (35.9%), syntax error (31.3%), incorrect file (4.9%), tool-use errors (68% of non-submitted). These failure modes map directly to our critic levels.

**Calibration strategy**: calibrate on SWE-bench Lite first (cheaper, more Y=1 signal), then validate and optionally re-calibrate on SWE-Bench Pro.

Data: `ScaleAI/SWE-bench_Pro` on HuggingFace. Code: https://github.com/scaleapi/SWE-bench_Pro-os

### 3.4 Why LiveCodeBench (secondary)

LiveCodeBench (competitive programming, post-training-cutoff problems) provides a cleaner cost structure:

- Test cases are split into example tests (public, fast) and hidden tests (many, slow).
- Running 3 example tests = cheap critic. Running 200+ hidden tests = expensive verifier.
- The P(example tests pass | all tests pass) and P(example tests pass | some hidden test fails) can be estimated empirically — this gives the likelihood model $P(z \mid Y)$.
- Problems are self-contained (no repo context), so environment setup noise is eliminated.

### 3.5 Why NOT HumanEval/MBPP

Too simple. Verification is cheap (few test cases, no environment complexity). The cost gap $C_\text{ver} \gg C_\text{crit}$ barely exists. The Bayesian controller's advantage would be negligible — a single generate-and-test loop already achieves high pass rates.

---

## 4. Concrete agent mapping

### 4.1 For SWE-bench

| Paper concept          | Concrete implementation                                    | Cost (approx)    |
|------------------------|------------------------------------------------------------|------------------|
| **Generator** $a_\text{gen}$ | LLM patch refinement: takes (issue, repo context, previous patch, critic feedback) → new patch | $C_\text{gen}$ = $0.05 + 10s |
| **Critic** $a_\text{crit}$   | Tiered diagnostics (see §4.2)                             | $C_\text{crit}$ = $0.00–0.01 + 1–5s |
| **Verifier** $a_\text{ver}$  | Full `pytest` / test suite execution in Docker container   | $C_\text{ver}$ = $0.00 + 30–120s |
| **Belief** $b_t$       | $P(\text{current patch passes full suite} \mid \text{observations})$ | — |
| **Specification** $x$  | GitHub issue description + repo snapshot                    | — |
| **Candidate** $c_t$    | Current patch (unified diff)                                | — |

### 4.2 Tiered critic design

The paper uses a single critic. In practice, we have **multiple critics at different cost-informativeness tradeoffs**. This is an extension of the paper's model where the critic action space becomes $\{a_\text{crit}^{(1)}, a_\text{crit}^{(2)}, \ldots\}$:

| Critic level | What it does                              | Cost  | Informativeness |
|-------------|-------------------------------------------|-------|-----------------|
| L0: Syntax  | `python -c "import ast; ast.parse(patch)"` | ~0.1s | High TPR for syntax bugs, zero info on logic |
| L1: Lint    | `ruff check` / `flake8` on changed files  | ~1s   | Catches style + some logic issues |
| L2: Type    | `mypy --no-incremental` on changed files  | ~3s   | Catches type mismatches |
| L3: Fast test | Run only the specific test file related to the issue | ~5–15s | Moderate signal on functional correctness |
| L4: LLM review | Ask a cheaper/smaller LLM "does this patch address the issue?" | ~3s + $0.01 | Noisy but captures semantic intent |

Each critic level has its own likelihood $P(z^{(k)} \mid Y)$ that must be calibrated.

### 4.3 For LiveCodeBench

| Paper concept          | Concrete implementation                        | Cost         |
|------------------------|-------------------------------------------------|--------------|
| **Generator**          | LLM solution generation/refinement              | ~$0.03 + 5s  |
| **Critic**             | Run public example test cases (2–3 cases)       | ~1–3s        |
| **Verifier**           | Run full hidden test suite (50–200+ cases)      | ~10–60s      |
| **Belief**             | $P(\text{solution passes all hidden tests} \mid \text{example test results, LLM confidence})$ | — |

---

## 5. Likelihood model calibration

This is the hardest part. The paper assumes $P(z \mid Y)$ is known. In practice we must estimate it.

### 5.1 Offline calibration (Phase 1)

Use a held-out set of (patch, ground-truth-label) pairs:

1. Take SWE-bench train split (or the ~2000 non-Verified instances).
2. For each instance, generate N patches using the LLM.
3. Run the full test suite to get ground-truth $Y \in \{0, 1\}$.
4. Run each critic level on each patch.
5. Estimate $P(z^{(k)} = \text{pass} \mid Y = 1)$ and $P(z^{(k)} = \text{pass} \mid Y = 0)$ via frequency counting.

This gives us the confusion matrix per critic level.

**Expected structure** (hypothesis based on known properties of linters vs. tests):

|              | $Y=1$ (correct patch) | $Y=0$ (incorrect patch) |
|-------------|----------------------|------------------------|
| Lint passes | ~0.95                | ~0.70                  |
| Type check passes | ~0.90          | ~0.55                  |
| Fast test passes | ~0.85            | ~0.15                  |
| LLM review positive | ~0.80         | ~0.40                  |

Note: lint/type checks have **high false-positive rate** (pass even for wrong patches) but **moderate true-positive rate** (most correct patches pass them). Fast tests have **low false-positive rate** (wrong patches rarely pass) but **moderate false-negative rate** (correct patches sometimes fail the specific test file).

### 5.2 Online calibration (Phase 2, optional)

Initialize with Bayesian priors from Phase 1. After each verifier call reveals ground truth, update $P(z \mid Y)$ using the observed (critic signal, true label) pair. This addresses distribution shift when the generator improves over episodes.

### 5.3 Generator transition model

Estimate $P(Y_{t+1} = 1 \mid Y_t = 0, a_t = a_\text{gen})$ — the probability that a refinement step fixes an incorrect patch. And $P(Y_{t+1} = 0 \mid Y_t = 1, a_t = a_\text{gen})$ — the probability that refinement breaks a correct patch.

**Approach**: on the calibration set, for each (patch_v1, patch_v2) pair where v2 is a refinement of v1:
- Count transitions: 0→1 (fixed), 1→0 (broken), 0→0 (still broken), 1→1 (still correct).
- Estimate as transition kernel $\mathcal{T}$.

**Expected values** (rough):
- $P(\text{fix} \mid \text{broken}) \approx 0.15–0.30$ (depends on feedback quality)
- $P(\text{break} \mid \text{correct}) \approx 0.05–0.10$ (LLMs sometimes regress)

---

## 6. Bellman equation — concrete instantiation

### State
Belief $b_t \in [0, 1]$: posterior probability that current patch is correct.

### Actions and Q-values

$$Q_\text{gen}(b_t) = -C_\text{gen} + \mathbb{E}[V(b_{t+1}) \mid b_t, a_\text{gen}]$$

where $b_{t+1}$ is computed via the transition kernel:
$$b_{t+1} = \frac{b_t \cdot P(\text{stay correct}) + (1 - b_t) \cdot P(\text{get fixed})}{b_t \cdot P(\text{stay correct}) + (1 - b_t) \cdot P(\text{get fixed}) + b_t \cdot P(\text{break}) + (1 - b_t) \cdot P(\text{stay broken})}$$

$$Q_\text{crit}^{(k)}(b_t) = -C_\text{crit}^{(k)} + \mathbb{E}_z[V(b_{t+1}) \mid b_t, a_\text{crit}^{(k)}]$$

where $b_{t+1}$ is the Bayes update given critic observation $z$:
$$b_{t+1}(z) = \frac{b_t \cdot P(z \mid Y=1)}{b_t \cdot P(z \mid Y=1) + (1 - b_t) \cdot P(z \mid Y=0)}$$

$$Q_\text{ver}(b_t) = b_t \cdot \lambda - C_\text{ver}$$

### Value function (discretized)

Discretize $b_t$ on a grid of 100–1000 points in $[0, 1]$. Solve via backward induction with horizon $T$ (max steps per episode, e.g., $T = 10$).

---

## 7. Baselines

| Baseline | Description |
|----------|-------------|
| **Fixed pipeline** | Generate → lint → generate again if lint fails → run full tests. No belief tracking. |
| **Confidence threshold** | Generate → run cheapest critic → if P(pass) > τ, verify; else regenerate. Single threshold. |
| **SAGE v4 (current)** | The existing SWE-bench agent with SAUP + resampling + reflexion. Uses heuristic confidence thresholds. |
| **Bayesian controller (ours)** | Full Bellman equation with calibrated likelihoods and transition kernel. |

The comparison that matters most: **SAGE v4 vs. Bayesian controller**, because both have the same LLM backbone and tool access. The difference is purely in the orchestration policy.

### 7.1 Key metrics

- **Pass@1**: fraction of instances where the submitted patch passes the full test suite.
- **Total cost**: sum of all API calls + compute time per instance.
- **Verification calls**: number of full test suite executions (the expensive action).
- **Cost-adjusted pass rate**: pass@1 / total_cost — the actual objective the Bayesian controller optimizes.

---

## 8. Implementation plan

### Phase 1a: Calibration on SWE-bench Lite (Week 1–2)

Calibrate on SWE-bench Lite first: simpler tasks (median 4.5 LOC patches), higher
chance of generating correct patches → better class balance for confusion matrix estimation.

1. For each of 300 SWE-bench Lite instances, generate 3 patches using an LLM.
2. Run tiered critics (L0–L2) on each patch:
   - L0: syntax check (`ast.parse`)
   - L1: lint (`ruff check --select=E,F`)
   - L2: fast test (run only the test file from `test_patch`)
3. Run the full test suite (verifier) to get ground truth $Y \in \{0, 1\}$.
4. Store results incrementally as JSONL: `calibration/data/raw_results.jsonl`.
5. Estimate likelihood tables $P(z^{(k)} \mid Y)$ and transition kernel $\mathcal{T}$.

**Scripts**: `calibration/generate_calibration_data.py`, `calibration/compute_likelihoods.py`
**Output**: `calibration/data/likelihood_tables.json`

### Phase 1b: Validate calibration on SWE-Bench Pro (Week 2)

SWE-Bench Pro (Deng et al., 2025) provides harder, contamination-resistant instances
from 41 repos (Python, JS, TS, Go). 731 public instances, multi-file patches (mean 107 LOC),
Docker-based environments. The cost gap $C_\text{ver} \gg C_\text{crit}$ is more
pronounced here, which is where the Bayesian controller's advantage should be largest.

1. Run the same calibration pipeline on SWE-Bench Pro public set (731 instances).
2. Check whether likelihood tables transfer from Lite → Pro (cross-difficulty generalization).
3. If they don't transfer well, calibrate separate tables for Pro.
4. This also serves as the primary **evaluation benchmark** for Phase 3.

**Data**: `ScaleAI/SWE-bench_Pro` on HuggingFace
**Ref**: https://arxiv.org/abs/2509.16941

### Phase 2: Bayesian controller implementation (Week 2–3)

1. Implement `BayesianCodeGenController` in `sage_agent/core/`:
   - Belief state: scalar $b_t \in [0,1]$.
   - Action selection via discretized Bellman equation.
   - Belief update for critic observations (Bayes rule).
   - Belief transition for generator actions (transition kernel).
   - Pre-computed policy table (offline) or real-time DP (small horizon).

2. Integrate with LangGraph as a new node in the v4 SWE-bench graph:
   - Replace the heuristic `decide_next_action` node with `bayesian_controller` node.
   - Keep all other nodes (search, read, edit, test) unchanged.

3. Wire up critic levels as concrete tool calls in the graph.

**Output**: `sage_agent/core/codegen_controller.py`, updated `different_agents/v4/langgraph_sage_agent_v4_swe_bayesian.py`

### Phase 3: Experiments on SWE-Bench Pro + SWE-bench Verified (Week 3–4)

1. Run all baselines on SWE-Bench Pro public set (731 instances) and SWE-bench Verified (500 instances).
2. Run Bayesian controller with calibrated parameters.
3. Ablations:
   - Remove critic (only generator + verifier) → measures value of cheap diagnostics.
   - Remove generator (only critic + verifier) → measures value of iterative refinement.
   - Single critic level only (L1 only, L2 only) → measures value of tiered critics.
   - Myopic (horizon-1) controller vs. full DP → measures value of lookahead.
   - Cross-benchmark likelihood transfer: Lite-calibrated tables on Pro instances.
4. Sensitivity analysis on likelihood misspecification: perturb $P(z \mid Y)$ by ±10%, ±20%.

### Phase 4: LiveCodeBench validation (Week 4–5)

1. Repeat Phase 1–3 on LiveCodeBench (cleaner setup, no environment noise).
2. Compare whether the value-of-information gap is larger or smaller than on SWE-bench.

---

## 9. What assumptions become meaningful (and where they break)

### Assumptions that hold

| Assumption | Why it holds |
|---|---|
| $C_\text{ver} \gg C_\text{crit}$ | Full test suite (30–120s) vs lint (1s). Real 30–120× gap. |
| Critic is noisy but informative | Lint catches syntax issues but misses logic bugs. Exactly the partial-information structure the paper models. |
| Binary correctness | SWE-bench patches either pass or fail the test suite. No partial credit. |
| Generator is stochastic | LLM refinement sometimes helps, sometimes hurts, sometimes changes nothing. |
| Budget constraint | API cost + time limit per instance. Can't run infinite refinement loops. |

### Assumptions that need relaxation

| Assumption | Reality | Mitigation |
|---|---|---|
| Known likelihood $P(z \mid Y)$ | Must be estimated. Calibration error propagates to suboptimal policy. | Phase 1 calibration + sensitivity analysis. |
| Stationary generator transition $\mathcal{T}$ | Transition probability changes with number of refinement steps (diminishing returns). | Make $\mathcal{T}$ step-dependent: $\mathcal{T}(b_t, t)$. |
| Single candidate | Paper tracks one candidate at a time. In practice, maintaining multiple candidates (best-of-N) is common. | Extension: belief over N candidates, generator produces N parallel patches, controller picks which to evaluate. |
| Independent critic observations | Lint and type check failures may be correlated (both triggered by the same syntax error). | Use conditional likelihoods or group correlated critics. |

### Assumptions that break (and why it's OK for a paper)

| Assumption | Why it breaks | Why it's OK |
|---|---|---|
| Verifier is ground truth | SWE-bench tests are not perfect (some false positives/negatives). | SWE-bench Verified reduces this. Acknowledge in limitations. |
| Stationary environment | Repo context changes as the agent edits files. | Within a single patch attempt, the environment is static. Reset between attempts. |
| No partial correctness | A patch might fix the main issue but break something else. | Binary outcome is the SWE-bench evaluation protocol. We match it. |

---

## 10. Expected outcome

If the calibration is reasonable, we expect:

- **Bayesian controller uses 30–50% fewer verification calls** than the fixed pipeline, because it learns to skip verification when belief is low (better to regenerate) or high enough from cheap critics alone.

- **Pass@1 stays the same or improves slightly**, because the saved verification budget can be reinvested into more generation attempts.

- **The gap is largest on medium-difficulty instances**, where the belief trajectory is non-trivial. On easy instances (first attempt works), everyone succeeds. On impossible instances (no LLM can solve), everyone fails. The Bayesian controller's advantage lives in the middle.

- **The "why heuristics fail" argument (Section 3.4 of the paper) should be empirically visible**: two instances with the same initial confidence but different critic profiles should lead to different controller decisions.

---

## 11. File structure

```
experiments/orchestration_hypothesis_testing/
├── EXPERIMENT_PLAN.md          ← this file
├── calibration/
│   ├── generate_calibration_data.py
│   ├── compute_likelihoods.py
│   └── data/                   ← likelihood tables, transition kernels
├── controller/
│   ├── bayesian_codegen_controller.py
│   └── test_controller.py
├── baselines/
│   ├── fixed_pipeline.py
│   └── confidence_threshold.py
├── evaluation/
│   ├── run_swebench_experiment.py
│   ├── run_livecodebench_experiment.py
│   └── analyze_results.py
└── results/                    ← experiment outputs
```
