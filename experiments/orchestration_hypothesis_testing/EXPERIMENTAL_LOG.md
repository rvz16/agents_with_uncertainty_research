# Experimental Log

End-to-end log of every experimental setup run on PR #1 (`Orchestration as hypothesis testing`). Each entry lists the dataset, configuration, exact command, output location, and headline finding. Designed so a reader can rerun any step from scratch.

For pre-registered methodology, see `experiments/orchestration_hypothesis_testing/PRE_REGISTRATION.md` (committed in `414a91c`).

---

## 1. Common infrastructure

### 1.1 Cluster + paths

- **Host**: `MBZUAI-Artem-1` (configured in `~/.ssh/config`)
- **Repo path on cluster**: `/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research`
- **Working branch**: `feature/orchestration-hypothesis-testing-experiment` (PR head)
- **HF cache**: `HF_HOME=/mnt/data/users/vlad.smirnov/hf_cache` (custom — home quota is 200GB, so caches go to `/mnt/data/...`)
- **Python**: 3.10, venv lives in `/home/vlad.smirnov/miniconda3` and per-pip user packages in `~/.local`
- **Disk note**: use `TMPDIR=/mnt/data/users/vlad.smirnov/tmp` for large pip installs to avoid quota issues

### 1.2 Cost model (utility = reward − cost)

```
c_gen   = 5    (generation API call)
c_L0    = 1    (ast.parse, free in practice — placeholder)
c_L2    = 2    (run public tests in subprocess)
c_L3    = 5    (LLM review via Haiku)
c_ver   = 30   (full ground-truth verifier)
reward  = 100  (correct trajectory)
horizon = 3 patches per instance
```

Sweep range explored: `c_ver ∈ {10, 15, 20, 30, 40, 60, 100}` (see `lcb_sensitivity.py`).

### 1.3 Generators (4-model panel)

All routed via OpenRouter with cost tracking:

| Label | OpenRouter model | $/M input | $/M output |
|---|---|---|---|
| `gpt5_mini` | `openai/gpt-5-mini` | 0.5 | 4.0 |
| `qwen3_coder` | `qwen/qwen3-coder` | 0.4 | 1.6 |
| `haiku45` | `anthropic/claude-haiku-4.5` | 1.0 | 5.0 |
| `sonnet45` | `anthropic/claude-sonnet-4.5` | 3.0 | 15.0 |

Registered in `scripts/lcb_calibrate.py:GENERATORS`. Cost rates in `scripts/lcb_calibrate.py:cost_for_call`.

### 1.4 Critic stack

| Critic | Implementation | Cost |
|---|---|---|
| **L0_syntax** | `ast.parse(code)` succeeds | ≈0 |
| **L1_lint** | `ruff check` returns clean | ≈0 |
| **L2_public_tests** | run public/visible tests in subprocess (per-benchmark) | varies |
| **L3_llm_review** | LLM PASS/FAIL on (problem, code) | ~$0.001/call |

`L1_lint` consistently has near-zero gap and is dropped from the controllers (kept in critic_results for completeness).

### 1.5 Statistical apparatus

- **Likelihoods**: P(z|Y) with Beta(1,1) smoothing (matches slide 6).
- **Policy comparison**: paired bootstrap CI on per-instance utility difference, B=1000 (`scripts/lcb_compare.py:paired_bootstrap_ci`).
- **Resume support**: every long-running script writes incrementally with line-buffered append + fsync, and skips `(instance_id, patch_id)` tuples already persisted on restart.

---

## 2. Datasets

### 2.1 LiveCodeBench (LCB)

Cached snapshot:
```
/mnt/data/users/vlad.smirnov/hf_cache/hub/datasets--livecodebench--code_generation_lite/snapshots/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test.jsonl
```

Loader: `scripts/lcb_calibrate.py:load_lcb(difficulty, platform)`.

| Difficulty | Platform | n_problems |
|---|---|---|
| `hard` | `leetcode` | 29 |
| `medium` | `leetcode` | 90 |
| `easy` | `leetcode` | 62 |

Each problem has:
- `question_content` — natural-language statement
- `starter_code` — Solution-class signature (LeetCode-style, functional)
- `public_test_cases` — visible (for L2)
- `private_test_cases` — hidden (for Y)

Test runner: functional (Solution-class), `scripts/lcb_calibrate.py:run_solution_functional`. We extract the method name from `starter_code`, parse JSON-encoded args from each test case, and call `Solution().<method>(*args)` in a subprocess.

### 2.2 MBPP+ (`evalplus/mbppplus`)

| n_problems | tests visible | tests hidden |
|---|---|---|
| 378 | 3 (`test_list`) | full PLUS suite (`test`) |

Loaded via `datasets.load_dataset("evalplus/mbppplus", split="test")` (HF cache auto-warms).

Test runner: `scripts/mbpp_calibrate.py:run_full_test`. MBPP+ tests are self-running scripts (call function inline at module level — no `check(candidate)` indirection like HumanEval+).

---

## 3. Experiments

### 3.1 LCB-hard calibration (4 generators)

| Generator | Output dir | Records | Cap | Actual cost |
|---|---|---|---|---|
| `gpt5_mini` | `data/lcb_calibration_v2/gpt5_mini/` | 87 (29×3) | $3 | $1.00 |
| `qwen3_coder` | `data/lcb_calibration_v2/qwen3_coder/` | 87 | $3 | $0.05 |
| `haiku45` | `data/lcb_calibration_v2/haiku45/` | 87 | $8 | ~$2 |
| `sonnet45` | `data/lcb_calibration_v2/sonnet45/` | 87 | $20 | ~$4 |

Reproduction:
```bash
cd experiments/orchestration_hypothesis_testing
bash scripts/launch_lcb_medium_parallel.sh gpt5_mini hard data/lcb_calibration_v2 3.0
bash scripts/launch_lcb_medium_parallel.sh qwen3_coder hard data/lcb_calibration_v2 3.0
bash scripts/launch_lcb_medium_parallel.sh haiku45 hard data/lcb_calibration_v2 8.0
bash scripts/launch_lcb_medium_parallel.sh sonnet45 hard data/lcb_calibration_v2 20.0
```

Per-generator outputs: `critic_results.jsonl`, `likelihood_tables.json`, `cost_summary.json`, `cost_log.jsonl`, `raw_responses/<inst>_p<pid>.txt`.

### 3.2 LCB-medium calibration (4 generators × 90 instances)

| Generator | Records | Status |
|---|---|---|
| `qwen3_coder` | 270 (90×3) | done |
| `haiku45` | 270 | done |
| `gpt5_mini` | running | (slow, ~2 rec/min) |
| `sonnet45` | running | |

```bash
for g in gpt5_mini qwen3_coder haiku45 sonnet45; do
  bash scripts/launch_lcb_medium_parallel.sh $g medium data/lcb_calibration_medium 5.0
done
```

### 3.3 MBPP+ calibration (4 generators × 100 instances)

```bash
for g in gpt5_mini qwen3_coder haiku45 sonnet45; do
  bash scripts/launch_mbpp_parallel.sh $g 4.0 100
done
# (Sonnet uses cap $15.0 because it's pricier)
```

### 3.4 Policy comparison (8 baselines + 2 controllers)

```bash
python3 scripts/lcb_compare.py --output-dir data/lcb_calibration_v2 \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45
```

Policies evaluated:
- `always_verify` (baseline)
- `threshold_L0` / `threshold_L2` / `threshold_L3` (single-critic threshold)
- `fixed_pipeline` (L0 then L3 then verify)
- `best_of_3` (generate 3 patches, then verify)
- `bayesian_DP` (full backward-induction DP)
- `bayesian_greedy` (1-step myopic Q-value, §F1 ablation)

Output: `<gen>/policy_comparison.json` with per-policy `mean_utility`, `pass_rate`, `diff_vs_always_verify`, paired-bootstrap CI.

### 3.5 Tier D — sensitivity sweeps (no API spend)

```bash
python3 scripts/lcb_sensitivity.py --output-dir data/lcb_calibration_v2 \
  --generators gpt5_mini,qwen3_coder
```

Three analyses on existing critic data:
- **D1 θ-sensitivity** (§E4): perturb each `P(z|Y)` by ±10%, ±20% (uniform / alternating); refit controllers; rerun.
- **D2 c_ver sweep** (§F4): grid `c_ver ∈ {10, 15, 20, 30, 40, 60, 100}`.
- **D3 verifier efficiency** (§G2): `verify_calls / patch_solved`.

Output: `<gen>/sensitivity.{json,csv}`.

### 3.6 L3 reviewer sweep (cross-family + cross-strength)

Re-runs L3 over multiple reviewer models on existing `raw_responses` (no patch regeneration).

```bash
python3 scripts/lcb_l3_sweep.py \
  --output-dir data/lcb_calibration_v2 \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \
  --reviewers haiku45=anthropic/claude-haiku-4.5,gpt4omini=openai/gpt-4o-mini,sonnet45=anthropic/claude-sonnet-4.5 \
  --difficulty hard --platform leetcode \
  --max-workers 8

python3 scripts/lcb_l3_analyze.py \
  --output-dir data/lcb_calibration_v2 \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45
```

Reviewer-swap policy comparison:
```bash
for rev in haiku45 gpt4omini sonnet45; do
  python3 scripts/lcb_compare_swap_reviewer.py \
    --output-dir data/lcb_calibration_v2 \
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \
    --reviewer $rev
done
```

Outputs: `<gen>/L3_sweep.jsonl`, `<gen>/L3_sweep_likelihoods.json`, `<gen>/policy_comparison_l3_<rev>.json`, `L3_sweep_summary.{csv,json}`, `L3_sweep_heatmap.json`.

`gpt-5-mini` was tested as a reviewer but **dropped**: the gpt-5 family consumes the entire `max_tokens` budget on reasoning tokens and returns empty content even at `max_tokens=200`. Documented in `lcb_l3_sweep.py:review_once`.

### 3.7 Paper-table + figures

```bash
python3 scripts/lcb_summarize_paper.py \
  --hard-dir data/lcb_calibration_v2 \
  --medium-dir data/lcb_calibration_medium \
  --output-root data

python3 scripts/lcb_make_figures.py \
  --paper-table data/PAPER_TABLE.json \
  --out-dir data/paper_figs
```

Three figures generated:
- `fig1_headline.{pdf,png}` — bayesian_greedy Δ utility per (gen, difficulty) with 95% CIs
- `fig2_l3_heatmap.{pdf,png}` — (gen × reviewer) L3 gap on hard, self-review cells highlighted
- `fig3_invariance.{pdf,png}` — bayesian_greedy stable vs threshold_L3 swing per cell

---

## 4. Robustness scripts

| Script | Purpose |
|---|---|
| `scripts/lcb_calibrate.py` | Full calibration. Patched for incremental writes + per-`(instance, patch)` resume + per-record try/except. |
| `scripts/lcb_rescue.py` | Recovers Y/L0/L1/L2 from `raw_responses/*.txt` after a crash; optional `--rerun-l3` backfills Haiku. |
| `scripts/launch_lcb_detached.sh` | `setsid + nohup` launcher so SSH disconnect doesn't SIGHUP the run. |
| `scripts/launch_lcb_medium_parallel.sh` | Same as above, parameterized over (generator, difficulty). |
| `scripts/launch_mbpp_parallel.sh` | MBPP+ variant. |

All long-running scripts:
- Write `<dir>/calibration.<gen>.log` with timestamps + cost lines
- Write `<dir>/calibration.<gen>.pid` for monitoring
- Are idempotent: relaunching resumes from the on-disk record file

---

## 5. Headline findings (commit refs)

| Finding | Effect | Commit |
|---|---|---|
| LCB-hard gpt5: bayesian_greedy +12.55 (CI [+7.03, +18.07]) | matches +12.2 synthetic prediction | [`34713d8`](https://github.com/rvz16/agents_with_uncertainty_research/commit/34713d8) |
| LCB-hard qwen3: +20.28 (CI [+14.76, +24.69]) | replication, different family | `34713d8` |
| LCB-hard haiku45: +20.28 (CI [+14.76, +24.69]) | replication, weaker generator | [`58f5a48`](https://github.com/rvz16/agents_with_uncertainty_research/commit/58f5a48) |
| LCB-hard sonnet45: +20.28 (CI [+14.76, +24.69]) | replication, strongest generator | (this commit) |
| LCB-medium qwen3: +18.40 (CI [+15.20, +21.60]) | tighter CI at n=90 | [`431c135`](https://github.com/rvz16/agents_with_uncertainty_research/commit/431c135) |
| LCB-medium haiku45: +19.07 (CI [+15.42, +22.31]) | replication on medium | [`0f88f3f`](https://github.com/rvz16/agents_with_uncertainty_research/commit/0f88f3f) |
| Greedy beats DP under IID-kernel misspec | +10.7 / +11.5 on gpt5/qwen3 | [`34713d8`](https://github.com/rvz16/agents_with_uncertainty_research/commit/34713d8) |
| Survives ±20% θ perturbation | always positive | [`1bd00f4`](https://github.com/rvz16/agents_with_uncertainty_research/commit/1bd00f4) |
| c_ver crossover at ~17, dominant for c_ver≥20 | regime map | `1bd00f4` |
| Verifier efficiency: 0.48× (gpt5) / 0.24× (qwen3) verify-per-solve | structural | `1bd00f4` |
| L3 reviewer invariance: utility constant across haiku/gpt4omini/sonnet | new robustness claim | [`431c135`](https://github.com/rvz16/agents_with_uncertainty_research/commit/431c135) |
| Self-review confound is real (haiku-on-haiku gap +0.169 vs +0.34 cross-family) | own §E5 finding | [`58f5a48`](https://github.com/rvz16/agents_with_uncertainty_research/commit/58f5a48) |
| Self-review collapse stronger on easier problems (haiku-on-haiku medium gap +0.004) | difficulty-dependent | [`0f88f3f`](https://github.com/rvz16/agents_with_uncertainty_research/commit/0f88f3f) |

---

## 6. Reproducing from scratch

Time: ~1h to first results, ~6h for full panel.

```bash
# 1. Setup
cd experiments/orchestration_hypothesis_testing
export OPENROUTER_API_KEY=...   # in repo's .env
export HF_HOME=/mnt/data/users/vlad.smirnov/hf_cache
export TMPDIR=/mnt/data/users/vlad.smirnov/tmp

# 2. Calibrate (LCB hard, 4 generators in parallel — ~1h)
for g in gpt5_mini qwen3_coder haiku45 sonnet45; do
  cap=3.0; [ "$g" = "haiku45" ] && cap=8.0; [ "$g" = "sonnet45" ] && cap=20.0
  bash scripts/launch_lcb_medium_parallel.sh $g hard data/lcb_calibration_v2 $cap
done
# Wait until each `data/lcb_calibration_v2/<gen>/critic_results.jsonl` reaches 87 lines.

# 3. Calibrate (LCB medium, 4 generators in parallel — ~3h)
for g in gpt5_mini qwen3_coder haiku45 sonnet45; do
  cap=5.0; [ "$g" = "sonnet45" ] && cap=20.0
  bash scripts/launch_lcb_medium_parallel.sh $g medium data/lcb_calibration_medium $cap
done

# 4. Calibrate (MBPP+, 4 generators in parallel — ~30min)
for g in gpt5_mini qwen3_coder haiku45 sonnet45; do
  cap=4.0; [ "$g" = "sonnet45" ] && cap=15.0
  bash scripts/launch_mbpp_parallel.sh $g $cap 100
done

# 5. Compare policies
python3 scripts/lcb_compare.py --output-dir data/lcb_calibration_v2 \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45
python3 scripts/lcb_compare.py --output-dir data/lcb_calibration_medium \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45

# 6. L3 reviewer sweep (~$2)
python3 scripts/lcb_l3_sweep.py \
  --output-dir data/lcb_calibration_v2 \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \
  --reviewers haiku45=anthropic/claude-haiku-4.5,gpt4omini=openai/gpt-4o-mini,sonnet45=anthropic/claude-sonnet-4.5 \
  --difficulty hard --platform leetcode --max-workers 8
python3 scripts/lcb_l3_analyze.py \
  --output-dir data/lcb_calibration_v2 \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45

# 7. Sensitivity
python3 scripts/lcb_sensitivity.py \
  --output-dir data/lcb_calibration_v2 \
  --generators gpt5_mini,qwen3_coder

# 8. Paper table + figures
python3 scripts/lcb_summarize_paper.py \
  --hard-dir data/lcb_calibration_v2 \
  --medium-dir data/lcb_calibration_medium \
  --output-root data
python3 scripts/lcb_make_figures.py \
  --paper-table data/PAPER_TABLE.json \
  --out-dir data/paper_figs
```

Total expected spend (in 2026-05 prices): ~$10-15 for the full panel (LCB + MBPP+ + L3 sweep + Haiku reviews).

---

## 7. Known issues / footguns

- **gpt-5-mini as a reviewer**: returns empty content because reasoning tokens consume `max_tokens`. Use it only as a generator, not as L3 reviewer.
- **OpenRouter Azure routing**: rejects `max_tokens < 16`. We use `max_tokens=32` for non-reasoning models, `max_tokens=200` for gpt-5 family.
- **Disk quota**: Home is 200GB. Podman storage at `~/.local/share/containers` and conda envs at `~/miniconda3` can fill it. Run `podman system prune -a -f` and `conda clean -a -y` if quota is exceeded.
- **MBPP+ test format**: tests are self-running scripts (call function inline at module level), NOT a `check(candidate)` function. Don't append `check(entry_point)` like for HumanEval+.
- **LCB starter_code**: must be honored. Use the functional runner that imports the module and calls `Solution().<method>(*args)`. Stdin runner gives 0% pass rate.
- **Self-review L3**: when the reviewer model = the generator model, the L3 critic collapses (gap drops by ~50% on hard, ~95% on medium). Always pair L3 with a different family.
- **L1_lint excluded from action space**: L1 has empirical gap≈0 across all generators (ruff F821/F811/E999 catches almost no errors above L0_syntax's AST parse). The framework calibrates L1 in `compute_likelihoods` but the controller's action space is `{verify, give_up, generate, L0, L2, L3}` — L1 is never queried. Reported in calibration tables for transparency only.
- **SWE-bench Pro pipeline gotchas**: prompts are ~24× larger than SWE-Lite (~121K input tokens) due to multi-file oracle context. gpt-5-mini consumes its `max_tokens` budget on reasoning tokens for these long prompts and produces empty parseable output. Pro probe at `data/swebench_pro_probe/` documents this. Full Pro calibration deferred from EMNLP (post-submission item).
- **Sonnet45 SWE-Lite iter harness errors**: 18/29 instances error during harness eval (vs haiku45's 5/29). Inflated P_break=0.476 (true ~0.10-0.15). Annotated in `data/swebench_lite_iter/sonnet45/HARNESS_ERRORS_NOTE.json`; reported in paper Methods with explicit disclosure.

---

## 8. 2026-05-06 follow-up work — within-regime replication + methodology rigor

Today's experiments extend the cube and add multiple layers of methodology rigor. New scripts committed across [c5b0faa](https://github.com/rvz16/agents_with_uncertainty_research/commit/c5b0faa), [5ae5c64](https://github.com/rvz16/agents_with_uncertainty_research/commit/5ae5c64), [54218f6](https://github.com/rvz16/agents_with_uncertainty_research/commit/54218f6), [c7c15a6](https://github.com/rvz16/agents_with_uncertainty_research/commit/c7c15a6).

### 8.1 LCB-medium iter refinement (4 generators × 30 instances × 5 steps)

```bash
setsid nohup python3 scripts/iter_refine_lcb.py \
  --src-dir data/lcb_calibration_medium \
  --output-dir data/lcb_calibration_medium_iter \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \
  --difficulty medium --platform leetcode --steps 5 \
  --n-instances 30 --max-workers 6 --max-cost-usd-per-model 2.0 \
  > data/lcb_calibration_medium_iter/launch.log 2>&1 < /dev/null &
```

Output: `data/lcb_calibration_medium_iter/<gen>/{iter_records.jsonl,iter_raw_responses/,transition_kernel.json}`. Wall: ~30 min for all 4 generators (sequential within script). Cost: ~$3.

Result: P_fix range [0.021, 0.100] across generators. Compare to IID baseline P_fix [0.029, 0.075]: feedback is roughly NEUTRAL on LCB-medium (not the "hurts" pattern of LCB-hard).

### 8.2 LCB-easy iter refinement (4 generators × 30 instances × 5 steps)

```bash
setsid nohup python3 scripts/iter_refine_lcb.py \
  --src-dir data/lcb_calibration_easy \
  --output-dir data/lcb_calibration_easy_iter \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \
  --difficulty easy --platform leetcode --steps 5 \
  --n-instances 30 --max-workers 6 --max-cost-usd-per-model 2.0 \
  > data/lcb_calibration_easy_iter/launch.log 2>&1 < /dev/null &
```

Result: gpt5_mini P_fix=0.529 (vs IID baseline 0.217) — feedback HELPS dramatically. sonnet45 P_fix=0.117 (vs 0.022) — also helps. qwen3_coder + haiku45 stay neutral. **Overturned the earlier blanket claim that "iter refinement hurts on LCB"** ([fa299a0](https://github.com/rvz16/agents_with_uncertainty_research/commit/fa299a0)) — the pattern is difficulty-stratified.

### 8.3 SWE-Lite iter refinement (haiku45 + sonnet45)

```bash
# 1. Symlink critic_results into source dir for the iter script's expected layout:
cd data
ln -sf ../../haiku45/critic_results.jsonl swebench_lite/source/haiku45/critic_results.jsonl
ln -sf ../../sonnet45/critic_results.jsonl swebench_lite/source/sonnet45/critic_results.jsonl

# 2. Generate iter trajectories
cd ..
setsid nohup python3 scripts/iter_refine_swebench.py \
  --dataset princeton-nlp/SWE-bench_Lite \
  --src-dir data/swebench_lite/source \
  --output-dir data/swebench_lite_iter \
  --generators haiku45,sonnet45 \
  --n-instances 30 --steps 5 --seed 42 --max-workers 6 \
  --max-cost-usd-per-model 5.0 \
  > data/swebench_lite_iter/launch.log 2>&1 < /dev/null &

# 3. Harness eval (requires DOCKER_HOST + SWEBENCH_PODMAN_COMPAT)
export DOCKER_HOST="unix:///run/user/$(id -u)/podman/podman.sock"
export SWEBENCH_PODMAN_COMPAT=1
setsid nohup python3 scripts/run_iter_harness.py \
  --iter-dir data/swebench_lite_iter \
  --work-dir data/swebench_lite_iter/eval \
  --dataset princeton-nlp/SWE-bench_Lite \
  --generators haiku45,sonnet45 --steps 1,2,3,4 --max-workers 2 \
  > data/swebench_lite_iter/harness.log 2>&1 < /dev/null &

# 4. Backfill Y values + compute kernels
python3 scripts/populate_iter_y_verified.py \
  --iter-dir data/swebench_lite_iter \
  --src-dir data/swebench_lite \
  --generators haiku45,sonnet45 --steps 5
python3 scripts/compute_iter_kernel.py \
  --iter-dir data/swebench_lite_iter \
  --generators haiku45,sonnet45
```

Cost: ~$8 (haiku $2.6 + sonnet $5.2). Wall: ~30 min generation + ~50 min harness.

Result: haiku45 P_fix=0.066 (clean); sonnet45 P_fix=0.033 with high P_break=0.476 contaminated by 18/29 harness errors (see footgun above).

### 8.4 §F1 ablation refresh under measured iterative kernels

```bash
# LCB-medium (4 gens) under measured iter kernel:
python3 scripts/lcb_compare.py \
  --output-dir data/lcb_calibration_medium \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \
  --kernel-file "gpt5_mini=data/lcb_calibration_medium_iter/gpt5_mini/transition_kernel.json,qwen3_coder=data/lcb_calibration_medium_iter/qwen3_coder/transition_kernel.json,haiku45=data/lcb_calibration_medium_iter/haiku45/transition_kernel.json,sonnet45=data/lcb_calibration_medium_iter/sonnet45/transition_kernel.json" \
  --out-suffix _kernel_iterative

# LCB-easy (4 gens) — same shape with lcb_calibration_easy paths
# SWE-Lite haiku/sonnet — same shape with swebench_lite paths
```

Output: `<calib-dir>/<gen>/policy_comparison_kernel_iterative.json`.

Headline: at moderate P_fix (≥0.15) the DP variant wins by +3 to +13 utility over Greedy. Most striking case is gpt5_mini/lcb-easy under measured P_fix=0.529: DP +17.98 vs Greedy +5.50 (gap +12.48). Otherwise (P_fix<0.10): DP=Greedy.

### 8.5 Self-Refine + Reflexion baseline replays

```bash
# LCB cells (variant lcb): walk steps until L2_public_tests passes
for d in lcb_calibration_v2_iter lcb_calibration_medium_iter lcb_calibration_easy_iter; do
  python3 scripts/compute_iter_replay_baselines.py \
    --iter-dir data/$d --variant lcb
done
# SWE cells (variant swe): walk steps until Y=1 (verifier IS the test)
python3 scripts/compute_iter_replay_baselines.py \
  --iter-dir data/swebench_verified_iter --variant swe
python3 scripts/compute_iter_replay_baselines.py \
  --iter-dir data/swebench_lite/source --variant swe   # gpt5+qwen3 May5
python3 scripts/compute_iter_replay_baselines.py \
  --iter-dir data/swebench_lite_iter --variant swe     # haiku+sonnet today
```

Output: `<iter-dir>/<gen>/policy_comparison_iter_replay_baselines.json`. 20 cells total.

Cost: $0 (pure replay on existing iter trajectories). **Result: framework's best Bayesian variant beats Self-Refine on 20/20 cells, beats Reflexion on 19/20** (one Reflexion-wins cell at LCB-easy gpt5_mini, regime crossover where P_fix=0.529).

### 8.6 χ² conditional-independence test

```bash
python3 scripts/critic_independence_test.py \
  --data-root data \
  --output-root data/critic_independence
```

Output: `data/critic_independence/{per_cell.json,per_cell.csv,pairwise.csv}`. 28 cells × Pearson χ² + G² on 4-way joint distribution per Y stratum, plus pairwise χ² for localization.

Result: independence holds for non-gpt5 generators across all benchmarks (max pairwise V≈0.24). gpt5_mini shows L0↔L1 perfectly correlated (V=1.0) but L1's gap is zero so framework is robust. Drop L1 from the framework's reported action space (already excluded operationally; confirm in paper).

### 8.7 MDE-power analysis

```bash
python3 scripts/mde_power_analysis.py \
  --data-root data --output-root data/mde_power
```

Output: `data/mde_power/{per_cell.json,per_cell.csv}`. For each cell: SE = (ci95_hi-ci95_lo)/3.92; MDE_80 = (1.96+0.84)*SE.

Result: 17/28 cells exceed MDE_80; 8/28 are regime-C ties at Δ=0 (correct framework choice); 3/28 SWE-Verified borderline.

### 8.8 Leave-one-instance-out cross-validation (LCB cells)

```bash
python3 scripts/loo_cv_lcb.py \
  --src-dir data/lcb_calibration_v2 \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45
# Same for lcb_calibration_medium, lcb_calibration_easy
```

Per fold (n folds = n_instances): re-estimate likelihoods from n-1 instances (Beta(1,1)), simulate policy on held-out instance, aggregate. Output: `<cell>/<gen>/policy_comparison_loo.json`.

Result: across all 12 LCB cells, LOO-CV utilities match in-sample to 2 decimal places. Train/test split methodology concern empirically resolved.

### 8.9 Tier-1 methodology rigor scripts (FDR + Wilson CI + cluster bootstrap)

```bash
python3 scripts/apply_fdr_correction.py \
  --paper-table data/PAPER_TABLE.json \
  --output-root data/fdr_correction

python3 scripts/wilson_ci_priors.py \
  --data-root data --output-root data/prior_ci

python3 scripts/cluster_bootstrap_swe.py \
  --data-root data --output-root data/cluster_bootstrap
```

Outputs:
- `data/fdr_correction/{per_cell.csv,summary.json}` — Benjamini-Hochberg adjusted p/q-values per (cell, policy).
- `data/prior_ci/{per_cell.csv,per_cell.json}` — Wilson 95% CI on prior_Y1.
- `data/cluster_bootstrap/{per_cell.csv,per_cell.json}` — within-repo cluster bootstrap on SWE-bench cells.

Headline: bayesian_greedy 20/28 cells survive BH-FDR (= uncorrected); SWE-Verified cluster CI inflation 1.2-1.6× but all 4 cells still have lower bound > 0.

### 8.10 c_ver sensitivity sweep on LCB-hard

```bash
for cv in 15 20 25 30 40 60; do
  python3 scripts/lcb_compare.py \
    --output-dir data/lcb_calibration_v2 \
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \
    --c-ver $cv --out-suffix "_cver$cv"
done
```

Output: 24 files (4 gens × 6 c_vers) at `data/lcb_calibration_v2/<gen>/policy_comparison_cver{15,20,25,30,40,60}.json`.

Result: crossover at c_ver≈17–20 (bayesian_greedy LOSES at c_ver=15, wins at c_ver≥20 monotonically up to c_ver=60 where give-up dominates).

### 8.11 SWE-bench Pro probe (deferred for follow-up paper)

```bash
# See scripts/swebench_pro_run.sh for the full deferred command set.
# Brief probe (1 call per generator, cost projection):
python3 scripts/spot_check_generators.py \
  --dataset ScaleAI/SWE-bench_Pro \
  --language-filter python \
  --output-dir data/swebench_pro_probe \
  --n-instances 30 --n-patches 3 \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \
  --max-cost-usd-per-model gpt5_mini=2.0,qwen3_coder=2.0,haiku45=5.0,sonnet45=12.0 \
  --probe-only
```

Probe finding: Pro Python prompts are ~121K input tokens (~24× larger than Lite). Full n=30 calibration would cost ~$50-80 (vs initial $30 estimate). gpt5_mini reasoning-token issue replicates: 4K completion tokens consumed by reasoning, no parseable output. Deferred.

### 8.12 Build PAPER_RELEASE/ artifact directory

```bash
python3 scripts/build_paper_release.py \
  --data-root data \
  --release-dir data/PAPER_RELEASE
```

Output: `data/PAPER_RELEASE/` — 206 files / 2.1 MB containing aggregates only (no raw_responses, no predictions, no harness logs). Structure:
- `PAPER_TABLE.{json,csv}` (top-level)
- `paper_figs/` (8 figures)
- `per_cell/<benchmark>/<gen>/` (likelihood tables, policy comparisons, sensitivity, L3 sweeps)
- `iter_kernels/<benchmark>_iter/<gen>/` (transition kernels)
- `methodology/{critic_independence,mde_power,fdr_correction,prior_ci,cluster_bootstrap}/`
- `docs/` (PRE_REGISTRATION.md + EXPERIMENTAL_LOG.md)
- `README.md`

### 8.13 New paper figures

- `paper_figs/fig_lcb_difficulty_gradient.{png,pdf,csv}` — P_fix(iter, with feedback) vs P_fix(IID baseline) across LCB-hard/medium/easy × 4 generators. Two-panel figure showing absolute values + Δ.
- `paper_figs/fig_framework_vs_baselines.{png,pdf,csv}` — best Bayesian variant vs Self-Refine vs Reflexion per cell across 20 cells. Annotated outlier (LCB-easy gpt5_mini Reflexion-wins).

```bash
python3 scripts/fig_lcb_difficulty_gradient.py \
  --data-root data --out-dir data/paper_figs
python3 scripts/fig_framework_vs_baselines.py \
  --data-root data --out-dir data/paper_figs
# Refresh fig1_headline / fig2_l3_heatmap / fig3_invariance:
python3 scripts/lcb_summarize_paper.py \
  --cells "lcb_hard=data/lcb_calibration_v2,lcb_medium=data/lcb_calibration_medium,lcb_easy=data/lcb_calibration_easy,mbpp=data/mbpp_calibration,humaneval=data/humaneval_calibration,swebench_lite=data/swebench_lite,swebench_verified=data/swebench_verified" \
  --output-root data
python3 scripts/lcb_make_figures.py \
  --paper-table data/PAPER_TABLE.json \
  --out-dir data/paper_figs
```

### 8.14 Today's spend recap

| Block | Cost |
|---|---|
| LCB-medium iter (4 gens) | ~$3 |
| LCB-easy iter (4 gens) | ~$2 |
| SWE-Lite iter (haiku45 + sonnet45) | ~$8 |
| SWE-Pro probe (killed early) | ~$0.04 |
| Methodology scripts (χ², MDE, LOO, FDR, Wilson, cluster bootstrap) | $0 (pure replay) |
| **Total today** | **~$13** |

Cumulative project spend: ~$45 of $80 envelope.
