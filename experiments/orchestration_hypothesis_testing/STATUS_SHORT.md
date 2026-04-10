# Bayesian orchestration — short status

### What we are testing

We are empirically validating the paper's claim that a Bayesian
decision-theoretic controller (Section 3) outperforms fixed
single-critic heuristics on real code-generation benchmarks. Our
calibration pipeline runs a real LLM generator against benchmark
problems, executes five critics per patch (syntax, lint, public
tests, Haiku-4.5 LLM review, mypy), and records the ground-truth
verifier outcome. Those calibrated confusion matrices are loaded
into a Bellman solver that we wrote from scratch, and a simulator
replays the data as episodes so every policy is scored on the same
instances. We have calibrated two benchmarks so far — a SWE-bench
Lite sympy subset (69 instances) and LiveCodeBench LeetCode
hard+medium (119 instances).

### Main result on LiveCodeBench

On LiveCodeBench (n = 119, base rate 0.714), the Bayesian controller
scores +124.75 ± 8.23 utility per episode against +126.85 ± 7.79 for
the best single-critic threshold policy (L2 = public tests). The
paired per-instance difference is −2.10 ± 0.64 (t = −3.29). That is a
statistically consistent but 1.6% effect — the two policies are a
near-tie. The same pattern holds across a seven-point cost-model
sweep: Threshold(L2) wins by 1.6 to 2.1 utility everywhere,
regardless of how we move `c_l2` and `c_ver` around. We also tested
a multi-critic Bellman extension that makes the set of critics
already used on the current patch part of the state; on LiveCodeBench
it does not improve over the single-critic version either, because
L2 and L3 on this data are not conditionally independent given `Y`
and L3 (Haiku reviewer) rejects 42% of actually-correct patches,
which makes the L3-first sequence the Bellman's math prefers
empirically worse than simply running L2. The same near-tie pattern
also held on SWE-bench Lite sympy, where L2 was even more of a
near-oracle and threshold policies again dominated.

### Why this is not a failure of the method

The paper's synthetic +41 utility gap comes from a three-way latent
class `H ∈ {A, B, C}` with conditionally independent diagnostics and
a symmetric, high-contrast `P(Z | H, T)` matrix. On real benchmarks
the dominant public-test critic has a TPR/FPR gap of 0.54, which is
strong enough that a naive "run it, verify on pass" heuristic is
already close to optimal and leaves no room for value-of-information
planning to add anything. When we synthetically degrade that critic
down to a gap of 0.25, neither our critic becomes weak enough
*relative to other available information* — L3 only has gap 0.28 and
is correlated with L2 — to give a multi-critic sequence a real
advantage. In other words, the Bellman is correctly solving the
optimization we asked it to solve, but neither benchmark we have so
far actually satisfies the informational preconditions the theory
needs (conditionally independent critics of comparable strength at a
non-trivial prior).

### Next steps

Four things, in priority order. First, run a conditional-independence
diagnostic on LiveCodeBench that reports `P(L2 | L3, Y)` as a 2×2
table — this quantifies exactly how far our critics are from the
paper's assumption and takes half a day. Second, regenerate the
LiveCodeBench calibration set using Haiku as the generator so the
base rate drops from 0.71 to roughly 0.35–0.45, which pushes us into
the low-prior regime where the Bellman's multi-step planning actually
matters (a one-day run). Third, reproduce the paper's own Section 4
synthetic benchmark inside our simulator and verify we recover the
+41 gap — this is a clean sanity check that separates "is our solver
correct?" from "does the theory apply to this data?". Fourth, if
those three steps still do not surface a regime where Bayesian
strictly beats the best threshold, reframe the contribution as a
**characterisation**: derive analytically the conditions under which
Bayesian orchestration strictly improves over single-critic
thresholds (no single critic above a gap threshold `g*`, conditional
independence, reward/verify ratio above some bound), and show on
SWE-bench Lite, LiveCodeBench, and the paper's synthetic benchmark
that the empirical win correlates tightly with those conditions
being met simultaneously.
