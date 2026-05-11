# PAPER_RELEASE — supplementary artifacts

Aggregated artifacts for the paper "Agentic AI Orchestration as Sequential
Hypothesis Testing for Code Generation". Excludes bulky raw artifacts
(predictions, raw_responses, calibration logs, harness eval logs).

## Structure

- `PAPER_TABLE.{json,csv}` — main results table, 28 cells × kernel/reviewer variants
- `paper_figs/` — figures referenced in the paper (PNG + PDF, plus CSV companions)
- `per_cell/<benchmark>/<generator>/` — per-cell aggregates:
  - `likelihood_tables.json` — Beta(1,1)-smoothed P(z|Y) per critic
  - `policy_comparison.json` — 8-policy utility comparison under default IID kernel
  - `policy_comparison_kernel_iterative.json` — under measured iter kernel (where available)
  - `policy_comparison_l3_<reviewer>.json` — L3 reviewer-swap variants
  - `policy_comparison_loo.json` — leave-one-out cross-validation
  - `policy_comparison_iter_replay_baselines.json` — Self-Refine + Reflexion replays
  - `transition_kernel_iid_baseline.json` — IID baseline kernel
  - `sensitivity.{json,csv}` — Tier-D θ-perturbation results (LCB-hard only)
- `iter_kernels/<benchmark>_iter/<generator>/transition_kernel.json` — measured iterative kernels
- `methodology/` — methodology rigor outputs:
  - `critic_independence/` — chi-squared + G-squared independence test (28 cells)
  - `mde_power/` — minimum detectable effect at 80% power (28 cells)
  - `fdr_correction/` — Benjamini-Hochberg FDR adjustment of policy p-values
  - `prior_ci/` — Wilson 95% CI on prior_Y1 (28 cells)
  - `cluster_bootstrap/` — within-repo cluster bootstrap on SWE-bench
- `docs/` — pre-registration and experimental log (referenced in paper)

## Cube coverage

- 4 generators: gpt-5-mini, qwen3-coder, claude-haiku-4.5, claude-sonnet-4.5
- 7 benchmarks: LiveCodeBench (hard, medium, easy), MBPP+, HumanEval+, SWE-bench (Lite, Verified)
- 3 L3 reviewers: claude-haiku-4.5, gpt-4o-mini, claude-sonnet-4.5

## Headline result

bayesian_greedy controller beats always_verify by Δ utility +5.5 to +20.3 on
LCB cells (n=29-90 instances), confirmed via paired-bootstrap CI (B=1000),
leave-one-out CV, BH-FDR correction, ±20% θ-perturbation, c_ver sensitivity
sweep, and within-repo cluster bootstrap on SWE-bench cells.

## How to reproduce

See `docs/EXPERIMENTAL_LOG.md` for full reproduction instructions and
`docs/PRE_REGISTRATION.md` for the pre-experiment methodology commitment.
