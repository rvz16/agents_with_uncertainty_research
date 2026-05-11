# PLAYBOOK — adding new benchmarks, generators, critics

This document is the canonical recipe for extending the empirical matrix.
Each addition type has explicit step-by-step commands, with the
**kernel-computation step** marked at its correct position in the flow.

## Concepts: two kernels in the pipeline

| Kernel type | When used | Source |
|---|---|---|
| **IID-synthesized** | Calibration phase (before iter data exists) | `P(fix\|broken) = prior` |
| **Measured** | After iter trajectories collected | `compute_transition_kernel.py` from `iter_records.jsonl` |

Steps 1–6 of every addition flow use the IID kernel as a placeholder so we
can run a policy comparison BEFORE iter data exists. The measured kernel
replaces it once step 10 (or the equivalent in a generator/critic flow)
runs. The §F1 ablation (commits `f800340`, `ef21a0a`) shows the IID
assumption is wrong on every benchmark we measure.

---

## Adding a new benchmark

### Required artifacts

- HuggingFace dataset OR local jsonl with `{instance_id, problem_statement, oracle/ground_truth}`
- Oracle execution mechanism — one of:
  - inline asserts (MBPP+, HumanEval+)
  - public + hidden tests (LCB)
  - per-instance container (SWE-bench)
- For multi-file/repo-level: `fetch_oracle_files()` adapter

### Algorithm

```
1.  Register dataset in scg.sample_instances() (add HF name + filtering).
2.  Sample n=30 instances with seed=42 (cached locally).
    Validate: re-run + diff to confirm reproducibility.
3.  Calibration: generate predictions_p{0,1,2}.jsonl (single-shot).
       python3 scripts/spot_check_generators.py \
         --dataset <hf-name> --n-instances 30 --n-patches 3 \
         --generators gpt5_mini,qwen3_coder,haiku45,sonnet45,qwen25_32b \
         --output-dir data/<bench>_calibration
4.  Harness eval on calibration patches → Y per (instance, patch_id).
5.  Compute likelihoods + prior.
       python3 scripts/calibrate_from_spotcheck.py \
         --output-dir data/<bench>_calibration --generators <list>
       → likelihood_tables.json
6.  Compute policy comparison on calibration data (IID kernel).
       python3 scripts/lcb_compare.py \
         --output-dir data/<bench>_calibration --generators <list>
       → policy_comparison.json (IID-synthesized kernel)

──── ITER PHASE BEGINS ────
7.  iter_refine_real_baselines{,_swe}.py
       → predictions_iter_step1..4.jsonl + iter_records.jsonl (Y=null)
8.  Harness eval on iter predictions → eval/<run_id>.<run_id>.json reports.
9.  Backfill Y in iter_records.jsonl.
       python3 scripts/backfill_swe_iter_y.py
10. ★ KERNEL COMPUTATION ★
       python3 scripts/compute_transition_kernel.py \
         --output-dir data/<bench>_realbaselines --generators <list>
       → transition_kernel.json (measured P(fix|broken), P(break|correct),
         cluster-bootstrap CI, IID baseline comparison)
11. Re-run policy comparison with measured kernel.
       python3 scripts/run_baseline_vs_controller.py \
         --output-dir data/<bench>_realbaselines --kernel measured
       → policy_comparison.json (now using measured kernel)
12. Aggregate to PAPER_TABLE.
       python3 scripts/lcb_summarize_paper.py \
         --cells "...,<bench>=data/<bench>_calibration"
13. Refresh figures.
       python3 scripts/lcb_make_figures.py \
         --paper-table data/PAPER_TABLE.json --out-dir data/paper_figs
```

### Validation checklist

- [ ] Prior `P(Y=1) ∈ [0.05, 0.95]` (sanity)
- [ ] χ² independence test passes on all (gen, bench) cells (`p > 0.05`)
- [ ] Critic gaps reported; flag any `|Δ| < 0.05` as no-op critic
- [ ] LOO-CV stability: per-instance utility within ±1 of in-sample mean
- [ ] Cluster-bootstrap CI on Bayesian Δ excludes 0 (or honestly report borderline)
- [ ] Measured kernel ≠ IID baseline (confirms F1 ablation expected pattern)

### Where it lands

- New column block in `fig1_headline.png`
- New point family in `fig4_regime_map.png`
- New row in `PAPER_TABLE.{json,csv}`

---

## Adding a new generator

### Required artifacts

- Model accessible via OpenRouter OR local vLLM endpoint at `<base_url>`.
- For vLLM: pre-warmed model, port reachable from harness machine.

### Algorithm

```
1.  Register generator in TWO places:
       # scripts/spot_check_generators.py
       GENERATORS["new_gen"] = ("provider/model-name", base_url_or_None,
                                thinking_flag)
       DEFAULT_COST_CAPS_USD["new_gen"] = 5.0  # or 1000.0 for free vLLM

       # scripts/lcb_calibrate.py
       GENERATORS["new_gen"] = ("provider/model-name", "Display Name",
                                base_url_or_None)
2.  For vLLM-only models: ensure iter_refine_real_baselines{,_swe}.py uses
    the two-client pattern (gen_client = vLLM, reviewer_client = OpenRouter).
3.  Run calibration on every benchmark in the panel:
       for bench in lcb-{hard,medium,easy} mbpp humaneval swe-{lite,verified}; do
         python3 scripts/spot_check_generators.py \
           --generators new_gen --dataset $hf_name \
           --output-dir data/<bench>_calibration
       done
4.  Harness on calibration patches.
5.  Compute likelihoods.
       for bench ...; do
         python3 scripts/calibrate_from_spotcheck.py \
           --output-dir data/<bench>_calibration --generators new_gen
       done
6.  Compute policy comparison (IID kernel) for calibration data.
7.  Iter phase per (bench, method):
       for bench in lcb-* swe-*; do for method in selfrefine reflexion; do
         python3 scripts/iter_refine_real_baselines{,_swe}.py \
           --generators new_gen --method $method \
           --src-dir data/<bench>_calibration \
           --output-dir data/<bench>_realbaselines
       done; done
8.  Harness on iter predictions → eval reports.
9.  Backfill Y in iter_records.jsonl.
10. ★ KERNEL COMPUTATION ★ per (bench, method):
       for bench in ...; do for method in ...; do
         python3 scripts/compute_transition_kernel.py \
           --output-dir data/<bench>_realbaselines/new_gen/<method>
       done; done
       → ~6 kernels for the new generator
         (3 LCB diffs + 2 SWE benches) × 1-2 methods
11. Re-run policy comparison with measured kernel for each iter cell.
12. Reviewer-invariance sweep (3 reviewers, ~$0.50):
       python3 scripts/lcb_l3_sweep.py \
         --output-dir data/lcb_calibration_v2 --generators new_gen \
         --reviewers haiku45=...,gpt4omini=...,sonnet45=...
13. PAPER_TABLE refresh + figures refresh.
```

### Cost estimate

- Closed-API generator: ~$3-5 across full panel (~2000 LLM calls)
- Open-weight (vLLM): ~$1 (only L3 reviewer cost; generation is free)

### Validation checklist

- [ ] Prior in expected range per cell
- [ ] L3 gap > 0.10 on at least one regime (otherwise reviewer choice is critical)
- [ ] χ² test passes
- [ ] L3 reviewer-invariance check: same Bayesian Δ ranking across haiku45/gpt4omini/sonnet45 reviewers

---

## Adding a new critic

### Most invasive change

Touches calibration, likelihood schema, Bayesian update, policy comparison.
**The kernel is NOT re-computed.**

### Algorithm

```
1.  Define critic function (new file under scripts/critics/):
       def critic_LX(modified_files: dict[str, str], problem: dict
                    ) -> tuple[bool, float]:
           """Returns (PASS, cost_in_usd)."""
2.  Update calibration: calibrate_from_spotcheck.py adds an LX column to
    critic_results.jsonl.
3.  Update likelihood schema: likelihood_tables.json["critic_likelihoods"]
    ["LX"] with the standard fields {P_pass_given_Y1, P_pass_given_Y0, gap,
    TP, FP, TN, FN}.
4.  Update independence test: chi2_critic_independence.py adds new pairwise
    L0×LX, L3×LX tests.
5.  Update Bayesian controller: run_baseline_vs_controller.py:
    BayesianController adds the new likelihood factor to the posterior.
6.  Add threshold policy: policy_threshold_LX function.
7.  Update cost vector: c_LX in cost model.

    ★ NO KERNEL RECOMPUTATION ★
    The kernel describes Y-transitions, which depend only on the model's
    refinement ability, not on which critics are observed. Reuse existing
    transition_kernel.json.

8.  Re-run calibration (--skip-generate) on every existing cell:
       python3 scripts/calibrate_from_spotcheck.py \
         --output-dir data/<each_bench>_calibration --generators all
       → adds LX column without regenerating patches
9.  Re-run policy comparison (uses existing kernel + new likelihood).
10. PAPER_TABLE refresh + figures refresh.
```

### Cost estimate

- Free critic (lint, mypy, ast.parse): $0
- LLM-based critic: ~$10-50 across full ~2400-patch corpus

### Validation checklist

- [ ] Critic gap |Δ| > 0.05 on at least one cell (otherwise no-op like our `mypy` finding)
- [ ] χ² independence vs L0 and L3 passes (new critic doesn't break the product-likelihood assumption)
- [ ] Bayesian Δ doesn't degrade after adding the critic (sanity)

---

## Master validation table (applies to all 3 addition flows)

| Check | Threshold | Failure means |
|---|---|---|
| Prior `P(Y=1)` | $\in [0.05, 0.95]$ | Saturation or pathological cell |
| χ² independence | $p > 0.05$ | Critic-independence violated; product likelihood unsafe |
| Critic gap | $\|\Delta\| > 0.05$ on at least one regime | Critic is no-op, drop from panel |
| Reviewer invariance | Same ranking across 3 L3 reviewers | Reviewer-specific artifact, report explicitly |
| LOO-CV stability | per-instance utility within $\pm 1$ of in-sample mean | Likelihoods overfit, Beta(1,1) smoothing not enough |
| Bootstrap CI on Bayesian Δ | excludes 0 (or borderline reported honestly) | Effect not statistically distinguishable from noise |
| Measured kernel vs IID | differs significantly | F1 ablation pattern as expected |

---

## Quick-reference: which scripts touch which step

| Step | Script | Output |
|---|---|---|
| 3 | `spot_check_generators.py` | `predictions_p{0,1,2}.jsonl` |
| 4 | `swebench.harness.run_evaluation` (or LCB inline) | `eval/<run_id>.json` |
| 5 | `calibrate_from_spotcheck.py` | `critic_results.jsonl`, `likelihood_tables.json` |
| 6 | `lcb_compare.py` | `policy_comparison.json` (IID kernel) |
| 7 | `iter_refine_real_baselines{,_swe}.py` | `iter_records.jsonl`, `predictions_iter_step1..4.jsonl` |
| 8 | harness on iter | `eval/<run_id>.json` |
| 9 | `backfill_swe_iter_y.py` | updated `iter_records.jsonl` |
| **10** | **`compute_transition_kernel.py`** | **`transition_kernel.json`** |
| 11 | `run_baseline_vs_controller.py --kernel measured` | `policy_comparison.json` (measured) |
| 12 | `lcb_summarize_paper.py` | `PAPER_TABLE.json`, `PAPER_TABLE.csv` |
| 13 | `lcb_make_figures.py`, `lcb_regime_map.py`, `fig_*` | `paper_figs/*.png/.pdf` |

The kernel step (10) is the bridge between "we have data" and "we have a
defensible policy comparison." Without it the comparison silently assumes
IID, which the §F1 ablation shows is wrong on every benchmark.
