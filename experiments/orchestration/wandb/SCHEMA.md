# wandb run schema

Project: [nlpresearch.group / orchestration-hypothesis-testing](https://wandb.ai/nlpresearch.group/orchestration-hypothesis-testing)

This doc describes the run convention used by `upload_runs.py` and consumed
by `analysis.ipynb`. Edit both scripts together if you change anything here.

## Tracks

Each run carries exactly one of these tags so the analysis notebook can
filter:

| Tag | Meaning | Source dir |
|---|---|---|
| `track:orchestration` | The main pipeline (5 closed-API generators across LCB, MBPP+, HumanEval+, SWE-Lite, SWE-Verified). | `experiments/orchestration_hypothesis_testing/data/` |
| `track:abbo` | The `agent-bugfix-bayes` codebase (gpt-oss-20b on HumanEvalFix, CodeContests, SWE-Lite). | `bayesian_optimization_for_code_testing/agent-bugfix-bayes/sim_results/` |

## Experiment types

Tagged `experiment:<type>`. One of:

| Type | Run unit | Summary fields |
|---|---|---|
| `calibration` | one per (benchmark, generator) | `prior_Y1`, `prior_Y1_ci_lo/hi`, `L0_gap`, `L1_gap`, `L2_gap`, `L3_gap`, `n_records`, `n_instances` |
| `iter` | one per (benchmark, generator, method) where method ∈ {selfrefine, reflexion, single_method} | `P_fix_given_broken`, `P_fix_ci_lo/hi`, `P_break_given_correct`, `P_break_ci_lo/hi`, `n_pairs` |
| `policy_comparison` | one per (benchmark, generator, kernel_source, l3_reviewer) | per-policy `mean_utility`, `pass_rate`, `delta_vs_always_verify`, `ci95_lo/hi`, `verify_rate` (8 policies) |
| `cver_sweep` | one per (benchmark, generator, c_ver_value) | same as policy_comparison |
| `theta_sweep` | one per (benchmark, generator, perturbation_kind) | same as policy_comparison |
| `r_sweep` | one per (benchmark, generator, R_value) | same as policy_comparison |
| `methodology` | one per analysis (fdr, mde_power, cluster_bootstrap, critic_independence, prior_ci, loo_cv) | analysis-specific |

## Run naming

`{exp_type}__{track}__{benchmark}__{generator}[__{extra}]`

Examples:
- `calibration__orchestration__lcb_hard__gpt5_mini`
- `iter__orchestration__swe_verified__sonnet45__selfrefine`
- `policy_comparison__orchestration__lcb_hard__qwen3_coder__measured`
- `cver_sweep__orchestration__lcb_hard__gpt5_mini__cver30`
- `calibration__abbo__codecontests__gpt_oss_20b`

## Config (set at run start, immutable)

```python
{
  # identification
  "track":             "orchestration",       # or "abbo"
  "experiment_type":   "calibration",         # see table above
  "benchmark":         "lcb_hard",            # lcb_hard, lcb_medium, lcb_easy,
                                              # mbpp, humaneval, swe_lite, swe_verified,
                                              # humanevalfix, codecontests
  "generator":         "gpt5_mini",           # gpt5_mini, qwen3_coder, haiku45,
                                              # sonnet45, qwen25_32b, gpt_oss_20b
  "method":            "selfrefine",          # iter-only: selfrefine | reflexion | single_method
  "kernel_source":     "measured",            # policy_comparison-only:
                                              # iid_baseline | measured | iterative | hand_tuned | default
  "l3_reviewer":       "haiku45",             # policy_comparison: which L3 judge model

  # sample
  "n_instances":       102,
  "k_patches":         3,
  "seed":              42,
  "pool_version":      "v1_to_v6",            # LCB only

  # cost vector
  "cost_R":            100,
  "cost_ver":          30,
  "cost_gen":          5,
  "cost_L0":           1,
  "cost_L2":           2,
  "cost_L3":           5,

  # sweep value (for *_sweep experiments)
  "sweep_axis":        "c_ver",               # c_ver | R | theta | b0 | c_critic
  "sweep_value":       30,                    # numeric

  # provenance
  "git_sha":           "a7846d8",
  "data_source":       "experiments/.../data/lcb_calibration_v2/gpt5_mini",
}
```

## Summary (set when run finishes)

### Calibration runs
```python
run.summary["prior_Y1"] = 0.214
run.summary["prior_Y1_ci_lo"] = 0.13
run.summary["prior_Y1_ci_hi"] = 0.31
run.summary["L0_gap"] = 0.28
run.summary["L1_gap"] = 0.03
run.summary["L2_gap"] = 0.71
run.summary["L3_gap"] = 0.34
run.summary["n_records"] = 87
run.summary["n_instances"] = 29   # or 102 at full pool
```

### Iter runs
```python
run.summary["P_fix_given_broken"]    = 0.22
run.summary["P_fix_ci_lo"]            = 0.11
run.summary["P_fix_ci_hi"]            = 0.35
run.summary["P_break_given_correct"] = 0.16
run.summary["P_break_ci_lo"]          = 0.08
run.summary["P_break_ci_hi"]          = 0.25
run.summary["n_pairs"]                = 112
run.summary["n_fix"]   = 9
run.summary["n_break"] = 10
run.summary["n_persist_broken"]  = 35
run.summary["n_persist_correct"] = 58
```

### Policy comparison runs

Flattened: one row of summary fields per policy.

```python
for policy in ["always_verify","best_of_3","threshold_L0","threshold_L2",
               "threshold_L3","fixed_pipeline","bayesian_greedy","bayesian_DP"]:
    run.summary[f"{policy}/mean_utility"]   = ...
    run.summary[f"{policy}/pass_rate"]      = ...
    run.summary[f"{policy}/delta_vs_av"]    = ...   # vs always_verify
    run.summary[f"{policy}/ci95_lo"]        = ...
    run.summary[f"{policy}/ci95_hi"]        = ...
    run.summary[f"{policy}/verify_rate"]    = ...
    run.summary[f"{policy}/mean_cost"]      = ...
```

Plus the regime classification:
```python
run.summary["regime"] = "A"                  # A | B | C
run.summary["best_policy"] = "bayesian_greedy"
```

## Artifacts

Logged as `wandb.Artifact(type="raw")` and attached to the run:

| Artifact | What it is |
|---|---|
| `predictions.jsonl` | per-patch LLM outputs (one row per (instance_id, patch_id)) |
| `critic_results.jsonl` | per-patch critic verdicts + Y |
| `iter_records.jsonl` | per-step iterative trajectory |
| `likelihood_tables.json` | calibrated $P(z|Y)$ |
| `transition_kernel.json` | measured $P(Y_{k+1}|Y_k)$ |
| `policy_comparison.json` | full per-policy results with bootstrap distribution |
| `sensitivity.json` | $\theta$-perturbation outputs |

## Tables (for interactive UI exploration)

For runs where it's useful, log a `wandb.Table` for browsing in the wandb UI
(separate from artifacts which require download):

- `per_patch` (calibration): columns `(instance_id, patch_id, L0, L1, L2, L3, Y, cost_usd)`
- `iter_trajectory` (iter): columns `(instance_id, step, diff_hash, L0, L2, L3, Y, stop_reason, step_cost_usd)`
- `policy_summary` (policy_comparison): columns `(policy, mean_utility, pass_rate, delta_vs_av, ci95_lo, ci95_hi, verify_rate)`

## Conventions

- All numeric values use **`numpy.float64`** (no `Decimal`, no `str`)
- `ci95_lo/hi` is always the 95% paired-bootstrap CI from B=1000 resamples
- `delta_vs_av` is `mean_utility - mean_utility_of_always_verify_on_same_cohort`
- For policies that produce a constant outcome (e.g. `always_verify` on regime C),
  `ci95_lo == ci95_hi == 0.0`
- Track tags use **kebab-case** values (`track:orchestration`, not `track:Orchestration`)
- `n_records` is always `n_instances * k_patches` for single-shot calibration runs
