# Colleague Runbook — running experiments end-to-end

This document is for **someone running these experiments for the first time**.
For the architecture of the pipeline (how to extend it with a new
benchmark/generator/critic) see `PLAYBOOK.md`. For the full audit log of
what was run historically see `EXPERIMENTAL_LOG.md`.

Scope of this runbook: **function-level synthesis** (LCB, MBPP+, HumanEval+)
and **repository-level patch generation** (SWE-Bench Lite, SWE-Bench
Verified). Bug-fixing benchmarks (HumanEvalFix, CodeContests) use the
`agent-bugfix-bayes/` pipeline and are out of scope here.

All commands assume cwd = `experiments/orchestration_hypothesis_testing/`.

---

## 0. Prerequisites

### 0.1 Python + repo

```bash
git clone https://github.com/rvz16/agents_with_uncertainty_research.git
cd agents_with_uncertainty_research
git checkout main                # base branch with the runbook
cd experiments/orchestration_hypothesis_testing
pip install -r ../../requirements.txt    # if present; otherwise install ad-hoc:
pip install openai datasets evalplus python-dotenv numpy scipy matplotlib swebench
```

Python ≥ 3.10. Verified locally on 3.13.

### 0.2 Secrets — put these in `<repo-root>/.env`

```
OPENROUTER_API_KEY=<OPENROUTER_API_KEY>
```

All four entry scripts (`calibration/lcb.py`, `calibration/mbpp.py`,
`calibration/humaneval.py`, `scripts/spot_check_generators.py`) auto-load
`.env` walking up from the script. You do **not** need to `export`.

Optional caches (recommended on shared clusters where `~/.cache` is
small):

```bash
export HF_HOME=/path/to/big/disk/hf_cache
export TMPDIR=/path/to/big/disk/tmp
```

### 0.3 Docker / Podman — **only required for SWE-Bench**

Function-level synthesis benchmarks (LCB / MBPP+ / HumanEval+) **do not
need Docker**. They run public/hidden tests in a subprocess.

SWE-Bench Phase 2 (harness eval) requires Docker or Podman with
pre-built images for the dataset:

```bash
# Docker users — just have docker running.
# Podman users — pin the socket and turn on the compat shim:
export DOCKER_HOST="unix:///run/user/$(id -u)/podman/podman.sock"
export SWEBENCH_PODMAN_COMPAT=1
```

**Architecture gotcha:** SWE-Bench pre-built images are `x86_64` only. On
Apple Silicon (`arm64`) the harness will 404 every image. Run Phase 2 on
a Linux x86_64 box (e.g. the cluster). Phase 1 (generation) is portable.

### 0.4 Local vLLM endpoints — only for open-weight generators

| Generator key | Model | Port | Needed for |
|---|---|---|---|
| `qwen25_32b` | Qwen/Qwen2.5-Coder-32B-Instruct | `8003` | LCB / MBPP+ / HumanEval+ |
| `qwen25_7b` | Qwen/Qwen2.5-7B-Instruct | `8001` | SWE-Bench (registered in `spot_check_generators.py`) |
| `qwen3_8b`, `qwen3_8b_thinking` | Qwen/Qwen3-8B | `8002` | SWE-Bench (extra, optional) |

Closed-API generators (`gpt5_mini`, `qwen3_coder`, `haiku45`, `sonnet45`)
go through OpenRouter and **do not need a local server**.

Start a vLLM endpoint (one per port) e.g.:

```bash
# 32B on port 8003 (used by LCB/MBPP+/HumanEval+ via lcb_calibrate.GENERATORS)
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
  --host 127.0.0.1 --port 8003 \
  --max-model-len 16384 --gpu-memory-utilization 0.9
```

If you're using an SSH-tunneled remote vLLM (the Runpod-1 box), set up a
local forward so the script can reach `127.0.0.1:8003`:

```bash
ssh -N -L 8003:127.0.0.1:8003 <runpod1-host>
```

### 0.5 Generator panel (which keys map to which model)

Function-level synthesis (LCB / MBPP+ / HumanEval+) reads
`calibration/lcb.py:GENERATORS`:

| Key | Model | Type |
|---|---|---|
| `gpt5_mini` | `openai/gpt-5-mini` | OpenRouter |
| `qwen3_coder` | `qwen/qwen3-coder` | OpenRouter |
| `haiku45` | `anthropic/claude-haiku-4.5` | OpenRouter |
| `sonnet45` | `anthropic/claude-sonnet-4.5` | OpenRouter |
| `qwen25_32b` | `Qwen/Qwen2.5-Coder-32B-Instruct` | **Local vLLM @ 8003** |

SWE-Bench reads `scripts/spot_check_generators.py:GENERATORS`:

| Key | Model | Type |
|---|---|---|
| `gpt5_mini`, `qwen3_coder`, `haiku45`, `sonnet45` | (same) | OpenRouter |
| `qwen25_7b` | `Qwen/Qwen2.5-7B-Instruct` | **Local vLLM @ 8001** |
| `qwen3_8b`, `qwen3_8b_thinking` | `Qwen/Qwen3-8B` | **Local vLLM @ 8002** |

> **Note.** The paper's panel references `Qwen2.5-Coder-7B-Instruct` and
> `gpt-oss-20b`, which are not yet registered in either GENERATORS dict.
> If we want those rows in `tab:full_results`, add them to both dicts
> (model id + base_url + enable_thinking) and start vLLM endpoints with
> the matching ports.

### 0.6 W&B access (READ THIS BEFORE RUNNING ANYTHING)

The paper's 35 main-panel cells (5 generators × 7 benchmarks) are
**already in W&B**. Don't re-run them — fetch them.

Project: [`nlpresearch.group/orchestration-hypothesis-testing`](https://wandb.ai/nlpresearch.group/orchestration-hypothesis-testing).

```bash
# One-time auth
pip install wandb
wandb login          # paste your W&B API key when prompted

# Confirm access
python3 -c "import wandb; api=wandb.Api(); print(len(list(api.runs('nlpresearch.group/orchestration-hypothesis-testing'))), 'runs visible')"
```

What's there right now (track:orchestration), per `SCHEMA.md`:

| Experiment type | # of runs | Coverage |
|---|---|---|
| `calibration` | 35 | 5 gens × 7 benchmarks |
| `iter` | ~25 | LCB-{hard,medium,easy} + SWE-{Lite,Verified} × 5 gens × {single_method, selfrefine, reflexion} |
| `policy_comparison` | 140+ | each calibration × kernel variant (default / iid_baseline / measured / iterative / loo) |
| `cver_sweep`, `theta_sweep`, `r_sweep`, `methodology` | various | sensitivity + methodology batches |

Tracks:
- `track:orchestration` — `gpt5_mini`, `qwen3_coder`, `haiku45`, `sonnet45`, `qwen25_32b` across LCB / MBPP+ / HumanEval+ / SWE-Lite / SWE-Verified.
- `track:abbo` — `gpt_oss_20b` on HumanEvalFix, CodeContests, SWE-Lite (from the `bayesian_optimization_for_code_testing/agent-bugfix-bayes/` codebase).

What's NOT there (these are the empty cells you need to fill):
- `Qwen2.5-Coder-7B-Instruct` — every benchmark.
- `gpt-oss-20b` — every benchmark in `track:orchestration` (it only exists on `track:abbo` for bug-fixing). Function-level synthesis + SWE-Bench cells are missing.
- `Qwen2.5-Coder-32B` on SWE-Lite + SWE-Verified.
- `gpt-5-mini` HumanEvalFix calibration; `gpt-5-mini` CodeContests BoN/threshold/SR/Rfx/BG/BDP.
- GrFt/DPFt columns for all function-level synthesis + SWE rows.

### 0.7 Fetching + re-analysing existing results

Use the analysis notebook — it does the W&B fetch + cache for you:

```bash
cd experiments/orchestration/wandb
jupyter notebook analysis.ipynb
# Run all cells. First cells call wandb.Api(), pull runs into a DataFrame,
# download per-run artifacts (critic_results.jsonl, iter_records.jsonl,
# policy_comparison.json), cache under .cache/runs.parquet + .cache/raw/.
# Re-runs use the cache; pass force_refresh=True to fetch_runs() to refresh.
```

If you only need a specific cell's raw data without the full notebook:

```python
import wandb
api = wandb.Api()
runs = api.runs("nlpresearch.group/orchestration-hypothesis-testing",
                filters={"config.experiment_type": "calibration",
                         "config.benchmark": "lcb_hard",
                         "config.generator": "gpt5_mini"})
for run in runs:
    for art in run.logged_artifacts():
        if "critic_results" in art.name:
            art.download(root="/tmp/lcb_hard_gpt5_mini")
```

---

## 1. Function-level synthesis (LCB / MBPP+ / HumanEval+)

All three benchmarks share the same shape: single-shot generation +
4 critics (L0 syntax, L1 lint, L2 public tests, L3 LLM-judge) + verifier
(hidden tests). No Docker needed. Output layout per cell:

```
data/<bench>_calibration/<gen>/
  ├─ critic_results.jsonl       # one row per (instance, patch)
  ├─ likelihood_tables.json     # P(z|Y), prior, gaps
  ├─ cost_summary.json
  ├─ cost_log.jsonl
  └─ raw_responses/<inst>_p<pid>.txt
```

### 1.0 Canonical N (paper Table 1) and per-cell draw n

Per the paper (§5.4 *Calibration and refinement pipelines*): *"Per cell
we draw n = 30–102 instances × k = 3 patches, with seed=42 fixed across
generators for paired comparison."* The canonical N is the full
benchmark size; the per-cell `n` is the subset actually run. Match the
paper's draws when you fill new cells:

| Benchmark | Canonical N (Table 1) | Per-cell n (paper draw) | LCB flag |
|---|---|---|---|
| LCB-hard | 102 | 102 | `--lcb-version all` |
| LCB-medium | 207 | 207 | `--lcb-version all` |
| LCB-easy | 135 | 135 | `--lcb-version all` |
| MBPP+ | 378 | 100 | n/a |
| HumanEval+ | 164 | 100 | n/a |
| SWE-Bench Lite | 300 | 30 | n/a |
| SWE-Bench Verified | 500 | 30 | n/a |
| HumanEvalFix | 164 | 30 | (bug-fixing track) |
| CodeContests | 165 | 30 | (bug-fixing track) |

> **LCB pool note.** `--lcb-version v1` (the default) gives 29 / 90 / 62
> instances for hard / medium / easy. To reach Table 1's full N use
> `--lcb-version all` (v1+v2+...+v6 union → 102 / 207 / 135).

### 1.1 LiveCodeBench (LCB-hard / medium / easy)

Closed-API generators (no vLLM needed). Use `--lcb-version all` and the
Table-1 N values to match the paper exactly:

```bash
cd experiments/orchestration_hypothesis_testing

# LCB-hard, full closed-API panel + Qwen32B (start its vLLM first, see §0.4)
python3 -m calibration.lcb \
  --output-dir data/lcb_calibration_hard \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45,qwen25_32b \
  --n-instances 102 --n-patches 3 \
  --difficulty hard --platform leetcode \
  --lcb-version all \
  --seed 42 \
  --max-cost-usd-per-model gpt5_mini=3.0,qwen3_coder=3.0,haiku45=8.0,sonnet45=20.0,qwen25_32b=1000.0

# LCB-medium (N=207)
python3 -m calibration.lcb \
  --output-dir data/lcb_calibration_medium \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45,qwen25_32b \
  --n-instances 207 --n-patches 3 \
  --difficulty medium --platform leetcode \
  --lcb-version all --seed 42 \
  --max-cost-usd-per-model gpt5_mini=5.0,qwen3_coder=5.0,haiku45=15.0,sonnet45=40.0,qwen25_32b=1000.0

# LCB-easy (N=135)
python3 -m calibration.lcb \
  --output-dir data/lcb_calibration_easy \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45,qwen25_32b \
  --n-instances 135 --n-patches 3 \
  --difficulty easy --platform leetcode \
  --lcb-version all --seed 42 \
  --max-cost-usd-per-model gpt5_mini=4.0,qwen3_coder=4.0,haiku45=10.0,sonnet45=25.0,qwen25_32b=1000.0
```

To skip Qwen32B, drop `qwen25_32b` from `--generators`. The
`--max-cost-usd-per-model` cap is per-generator and the cost tracker
aborts cleanly if hit (you can resume — the script skips
`(instance, patch_id)` tuples already on disk).

### 1.2 MBPP+ (N=378 canonical; per-cell n=100 in paper)

```bash
python3 -m calibration.mbpp \
  --output-dir data/mbpp_calibration \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45,qwen25_32b \
  --n-instances 100 --n-patches 3 \
  --seed 42 \
  --max-cost-usd-per-model gpt5_mini=2.0,qwen3_coder=2.0,haiku45=4.0,sonnet45=15.0,qwen25_32b=1000.0
```

### 1.3 HumanEval+ (N=164 canonical; per-cell n=100 in paper)

```bash
python3 -m calibration.humaneval \
  --output-dir data/humaneval_calibration \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45,qwen25_32b \
  --n-instances 100 --n-patches 3 \
  --plus-input-cap 200 \
  --seed 42 \
  --max-cost-usd-per-model gpt5_mini=2.0,qwen3_coder=2.0,haiku45=4.0,sonnet45=15.0,qwen25_32b=1000.0
```

### 1.4 Sanity check after each cell

```bash
ls data/<bench>_calibration/<gen>/critic_results.jsonl   # should be N×n_patches lines
jq -s 'length' data/<bench>_calibration/<gen>/critic_results.jsonl
cat data/<bench>_calibration/<gen>/likelihood_tables.json | jq '.prior_Y1, .critic_likelihoods | keys'
```

Acceptance gates (from `PLAYBOOK.md` master validation table):

- `prior_Y1 ∈ [0.05, 0.95]` (sanity — not saturated either way).
- For each critic L0/L2/L3, |gap| > 0.05 on at least one cell.

### 1.5 Resume after a crash

All three calibrators are idempotent: relaunching the same command skips
`(instance_id, patch_id)` tuples already written to `critic_results.jsonl`.

---

## 2. Repository-level patch generation (SWE-Bench Lite / Verified)

Two-phase pipeline. Phase 1 (generation) is portable; Phase 2 (Docker
harness eval) needs x86_64 Linux + Docker/Podman.

The script is **`scripts/spot_check_generators.py`**. The
`--dataset` flag selects Lite vs Verified (and `--language-filter`
is available for SWE-Pro multi-language filtering, e.g. `python`).

### 2.1 Cost probe (no patches, $0.02 per generator)

Before any real run, project the cost:

```bash
python3 scripts/spot_check_generators.py \
  --dataset princeton-nlp/SWE-bench_Lite \
  --output-dir data/swebench_lite \
  --n-instances 30 --n-patches 3 \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \
  --probe-only
```

This runs 1 generation per (instance, generator) and projects the full
cost. Read `data/swebench_lite/probe_report.json` before continuing.

### 2.2 Full single-shot calibration

```bash
# SWE-Bench Lite
python3 scripts/spot_check_generators.py \
  --dataset princeton-nlp/SWE-bench_Lite \
  --output-dir data/swebench_lite \
  --n-instances 30 --n-patches 3 --seed 42 \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \
  --max-cost-usd-per-model gpt5_mini=2.0,qwen3_coder=2.0,haiku45=5.0,sonnet45=12.0 \
  --max-workers-gen 8 --max-workers-eval 4

# SWE-Bench Verified — same shape, different dataset + output dir
python3 scripts/spot_check_generators.py \
  --dataset princeton-nlp/SWE-bench_Verified \
  --output-dir data/swebench_verified \
  --n-instances 30 --n-patches 3 --seed 42 \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \
  --max-cost-usd-per-model gpt5_mini=2.0,qwen3_coder=2.0,haiku45=5.0,sonnet45=12.0 \
  --max-workers-gen 8 --max-workers-eval 4
```

The script does:
1. **Phase 1 — generation**: 30 instances × 3 patches × 4 generators = 360
   calls (~25 min over OpenRouter). Output: `<gen>/predictions.jsonl` +
   `predictions_p{0,1,2}.jsonl` + `raw_responses/`.
2. **Phase 2 — Docker harness eval**: builds/pulls per-instance images,
   runs hidden tests, writes `eval/<gen>_p<pid>.json` reports. **Needs
   Docker/Podman + x86_64.**
3. **Phase 3 — aggregation**: per-generator `cost_summary.json` and
   run-level `summary.json` with base-rate-vs-PRE_REGISTRATION verdict.

### 2.3 Skipping phases

- `--skip-eval` — run Phase 1 + Phase 3 (skip Docker). Useful if you'll
  run Phase 2 on a different machine.
- `--skip-generate` — run Phase 2 only (uses existing predictions JSONLs).
  Useful for the Mac-Phase1 → cluster-Phase2 hand-off below.

### 2.4 Mac-Phase1 → cluster-Phase2 hand-off

If you're on Apple Silicon, generate locally and eval on the cluster:

```bash
# Local (arm64): Phase 1 only
python3 scripts/spot_check_generators.py ... --skip-eval

# Push the predictions to the cluster
rsync -av data/swebench_lite/ MBZUAI-Artem-1:/path/to/repo/.../data/swebench_lite/

# Cluster (x86_64 + podman): Phase 2 only
ssh MBZUAI-Artem-1
cd /path/to/repo/.../experiments/orchestration_hypothesis_testing
export DOCKER_HOST="unix:///run/user/$(id -u)/podman/podman.sock"
export SWEBENCH_PODMAN_COMPAT=1
python3 scripts/spot_check_generators.py \
  --dataset princeton-nlp/SWE-bench_Lite \
  --output-dir data/swebench_lite \
  --n-instances 30 --n-patches 3 --seed 42 \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \
  --skip-generate
```

### 2.5 Critic computation on the SWE-Bench corpus

After Phase 2 completes, run `calibrate_from_spotcheck.py` to compute
L0/L1/L2/L3 critic results + likelihoods:

```bash
python3 -m calibration.from_spotcheck \
  --output-dir data/swebench_lite \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \
  --dataset princeton-nlp/SWE-bench_Lite
```

(Use `--dataset princeton-nlp/SWE-bench_Verified` for the verified split.)

---

## 3. Policy comparison + paper table

### 3.0 Two families of policies (read first)

The paper's policies split into two families, and only one of them
needs fresh LLM calls. Important to internalise this before launching
runs — it saves a lot of compute and clarifies which steps cost money.

**Family A — replay policies (no new LLM calls).** These are decision
rules evaluated over the *evidence the single-shot calibration already
produced* (`critic_results.jsonl`: L0, L1, L2, L3, Y for each
`(instance, patch_id)` tuple). All eight are computed by one local pass
through `analysis/lcb_compare.py` (or `run_baseline_vs_controller.py`):

| Policy | Decision rule | New API calls? |
|---|---|---|
| `always_verify` | Skip critics, call oracle on every patch | No |
| `best_of_3` | Generate 3, verify each, take best (uses 3 existing patches) | No |
| `threshold_L0` / `L2` / `L3` | Verify iff critic_k = PASS, else regen | No |
| `fixed_pipeline` | L0 → L2 → L3 AND-gate then verify | No |
| `bayesian_greedy` | 1-step Bellman Q-value argmax | No |
| `bayesian_DP` | Full backward induction over `(b, k)` | No |

This is why `policy_comparison.json` is cheap to recompute and why all
the kernel / c_ver / θ-sensitivity sweeps in §5 are free once
calibration exists.

**Family B — trajectory policies (need fresh LLM calls).** These are
real implementations that loop generator → critic → regenerate, so
each step is a new completion:

| Policy | Script | Generator calls per instance |
|---|---|---|
| `Self-Refine` | `python -m iter.refine --method selfrefine` | 1 + up to 4 refinements |
| `Reflexion` | `python -m iter.refine --method reflexion` | 1 + up to 4 with verbal-memory buffer |

These produce `iter_records.jsonl` (per-step trajectory + Y at each
step) that **doesn't exist** in the single-shot calibration. The same
iter trajectory is *also* the input to `compute_transition_kernel.py`
which produces the measured `P(fix|broken)` kernel that `bayesian_DP`
uses (see §4) — so iter refinement does double duty for `bayesian_DP`
and the SR / Rfx columns.

**Subtlety — once a trajectory exists, SR / Rfx can be replayed for
free.** `iter/replay_baselines.py` takes existing
`iter_records.jsonl` (from any source — your own iter run, an old
W&B trajectory, etc.) and applies the SR / Rfx policy *replay* on top.
No new API calls. This is what produced the 20-cell SR/Rfx comparison
in the paper's §5.5. But you still need at least one trajectory to
exist for the cell — see §4 for how to generate it.

### 3.1 Running the replay policy comparison

After calibration is done for a cell, compute the 8-policy comparison
(this produces the Δ-utility numbers in `tab:full_results`):

```bash
# Works for LCB, MBPP+, HumanEval+, and the SWE-Bench cells alike.
python3 -m analysis.lcb_compare \
  --output-dir data/<bench>_calibration \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45
```

Output per cell: `<gen>/policy_comparison.json` with `mean_utility`,
`diff_vs_always_verify`, and paired-bootstrap 95% CIs for each of:
`always_verify`, `best_of_3`, `threshold_L{0,2,3}`, `fixed_pipeline`,
`bayesian_greedy`, `bayesian_DP` (IID kernel at this point).

Then aggregate to the paper-table:

```bash
python3 -m analysis.lcb_summarize_paper \
  --hard-dir data/lcb_calibration_hard \
  --medium-dir data/lcb_calibration_medium \
  --easy-dir data/lcb_calibration_easy \
  --output-root data
python3 figures/lcb_make_figures.py \
  --paper-table data/PAPER_TABLE.json \
  --out-dir data/paper_figs
```

For a multi-benchmark refresh (recommended once everything is in):

```bash
python3 -m analysis.lcb_summarize_paper \
  --cells "lcb_hard=data/lcb_calibration_hard,lcb_medium=data/lcb_calibration_medium,lcb_easy=data/lcb_calibration_easy,mbpp=data/mbpp_calibration,humaneval=data/humaneval_calibration,swebench_lite=data/swebench_lite,swebench_verified=data/swebench_verified" \
  --output-root data
```

---

## 4. Iterative refinement (measured transition kernel)

The IID kernel used by `lcb_compare.py` is a placeholder. To replace it
with the **measured** kernel (the one in §F1 of the paper), run iter
refinement and `compute_transition_kernel.py`.

### 4.1 LCB iter

The `single_method` iter (one-arm regeneration; produces the
`(Y_t, Y_{t+1})` pairs that the measured kernel is fit from) lives in
the legacy iter dir. The current `iter/refine.py` handles SR/Rfx only;
`single_method` was not ported because the kernel computation was the
only consumer and that flow is well-established.

```bash
python3 -m iter._legacy.refine_lcb \
  --src-dir data/lcb_calibration_hard \
  --output-dir data/lcb_calibration_hard_iter \
  --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \
  --difficulty hard --platform leetcode \
  --n-instances 30 --steps 5 \
  --max-workers 6 --max-cost-usd-per-model 2.0
```

### 4.2 SWE-Bench iter (needs Docker for the harness backfill)

```bash
# Generation (legacy single-method iter — see §4.1 note)
python3 -m iter._legacy.refine_swebench \
  --dataset princeton-nlp/SWE-bench_Lite \
  --src-dir data/swebench_lite/source \
  --output-dir data/swebench_lite_iter \
  --generators haiku45,sonnet45 \
  --n-instances 30 --steps 5 --seed 42 --max-workers 6 \
  --max-cost-usd-per-model 5.0

# Harness eval over the new iter predictions
python3 -m iter.harness \
  --iter-dir data/swebench_lite_iter \
  --work-dir data/swebench_lite_iter/eval \
  --dataset princeton-nlp/SWE-bench_Lite \
  --generators haiku45,sonnet45 --steps 1,2,3,4 --max-workers 2

# Backfill Y into iter_records.jsonl (post-refactor: at iter/swe_backfill_y.py)
python3 -m iter.swe_backfill_y \
  --iter-dir data/swebench_lite_iter \
  --src-dir data/swebench_lite \
  --generators haiku45,sonnet45 --steps 5

# Measured kernel
python3 -m iter.kernel \
  --iter-dir data/swebench_lite_iter \
  --generators haiku45,sonnet45
```

### 4.3 Re-run policy comparison with the measured kernel

```bash
python3 -m analysis.lcb_compare \
  --output-dir data/<bench>_calibration \
  --generators <list> \
  --kernel-file "gpt5_mini=data/<bench>_iter/gpt5_mini/transition_kernel.json,..." \
  --out-suffix _kernel_iterative
```

This writes `<gen>/policy_comparison_kernel_iterative.json`. Re-run
`lcb_summarize_paper.py` to fold the measured-kernel rows into
`PAPER_TABLE.json`.

---

## 5. Two cost vectors (paper §Critic Stack / §Cost-vector choice)

Both vectors share `c_gen=10, R=100`. The paper uses:

| Vector | `c_ver` | `c_L0` | `c_L2` | `c_L3` | Used for |
|---|---|---|---|---|---|
| **Slow-oracle** ("slow oracle, fast critics") | 30 | 1 | 2 | 5 | LCB, MBPP+, HumanEval+, SWE-Bench Lite/Verified |
| **Fast-oracle** ("fast oracle, balanced critics") | 5 | 1 | 1 | 1 | HumanEvalFix, CodeContests |

`lcb_compare.py` uses the slow-oracle vector by default. Override via
`--c-ver`, `--c-l0`, `--c-l2`, `--c-l3` for the fast-oracle vector or for
the c_ver sensitivity sweep (`c_ver ∈ {15,20,25,30,40,60}`).

---

## 6. Recommended run order for filling `tab:full_results`

> **Step 0 — Don't re-run cells that are already in W&B.** Run
> `analysis.ipynb` first (it fetches all 35 existing main-panel cells
> from `nlpresearch.group/orchestration-hypothesis-testing`). Only the
> cells that print `--` in `tab:full_results` need experiments. See §0.6.

The empty cells (the only ones you should generate from scratch):

| Empty (benchmark, generator) cells | Why |
|---|---|
| `Qwen2.5-Coder-7B-Instruct` × every benchmark | Not yet in any GENERATORS dict; not in W&B. |
| `gpt-oss-20b` × LCB / MBPP+ / HumanEval+ / SWE-Lite / SWE-Verified | Exists on `track:abbo` for bug-fixing only; not in `track:orchestration`. |
| `Qwen2.5-Coder-32B` × SWE-Lite + SWE-Verified | Closed-API rows filled; open-weight 32B missing on the SWE side. |
| `gpt-5-mini` × HumanEvalFix | Whole row empty in the paper table. |
| `gpt-5-mini` × CodeContests (BoN…BDP) | Only GrFt/DPFt populated; BoN-through-BDP replay missing. |
| `GrFt / DPFt` × all function-level synthesis + SWE rows | End-to-end fitted-agent columns; from the `agent-bugfix-bayes` codebase, see its own pipeline. |

For each missing (benchmark, generator) cell, run in this order:

1. **Add the generator** to `lcb_calibrate.GENERATORS` and
   `spot_check_generators.GENERATORS` if it isn't there yet. Start the
   vLLM endpoint at the registered port if open-weight.
2. **Probe** (SWE only): `--probe-only` for cost projection.
3. **Calibrate** (`python -m calibration.lcb` / `calibration.mbpp` /
   `calibration.humaneval` / `python3 scripts/spot_check_generators.py`)
   → produces `critic_results.jsonl` and `likelihood_tables.json`.
4. **Policy compare** (`lcb_compare.py`) → produces
   `policy_comparison.json` (IID kernel).
5. **Iter refinement** + harness backfill + `compute_transition_kernel.py`
   → measured kernel.
6. **Re-run policy compare** with `--kernel-file ...` → measured-kernel
   `policy_comparison_kernel_iterative.json`.
7. **Aggregate** with `lcb_summarize_paper.py` and refresh figures.
8. **Upload to W&B** so the analysis notebook picks the new cells up:

   ```bash
   cd experiments/orchestration/wandb
   python3 upload_runs.py --benchmark <bench> --generator <gen>     # single cell
   # OR
   python3 upload_runs.py --experiment calibration                   # all new calibrations
   ```

   `upload_runs.py` is idempotent — re-running skips runs that already
   exist. Add `--force` to replace existing runs.

9. **Re-run** `analysis.ipynb` (cell 3 `fetch_runs(force_refresh=True)`)
   to fold the new cells into the paper table.

Use the runtime estimates from `EXPERIMENTAL_LOG.md` (e.g. LCB-hard
~1 h per generator at n=30, scales roughly linearly to n=102; MBPP+
~30 min, SWE-Bench ~30 min generate + ~45 min harness eval; total
~$10–15 for the closed-API panel at the paper's draws).

---

## 7. Smoke tests (n=5 per benchmark) — do this first

Confirm your setup works before launching a real run.

```bash
cd experiments/orchestration_hypothesis_testing

# LCB-hard: 5 instances × 1 patch × gpt5_mini only
python3 -m calibration.lcb \
  --output-dir /tmp/smoke_lcb_hard \
  --generators gpt5_mini \
  --n-instances 5 --n-patches 1 \
  --difficulty hard --platform leetcode \
  --max-cost-usd-per-model 0.50

# MBPP+: 5 × 1
python3 -m calibration.mbpp \
  --output-dir /tmp/smoke_mbpp \
  --generators gpt5_mini \
  --n-instances 5 --n-patches 1 \
  --max-cost-usd-per-model 0.50

# HumanEval+: 5 × 1
python3 -m calibration.humaneval \
  --output-dir /tmp/smoke_humaneval \
  --generators gpt5_mini \
  --n-instances 5 --n-patches 1 \
  --max-cost-usd-per-model 0.50

# SWE-Bench Lite cost probe (no patches generated, no Docker)
python3 scripts/spot_check_generators.py \
  --dataset princeton-nlp/SWE-bench_Lite \
  --output-dir /tmp/smoke_swe_lite \
  --n-instances 5 --n-patches 1 \
  --generators gpt5_mini --probe-only

# SWE-Bench Verified cost probe
python3 scripts/spot_check_generators.py \
  --dataset princeton-nlp/SWE-bench_Verified \
  --output-dir /tmp/smoke_swe_verified \
  --n-instances 5 --n-patches 1 \
  --generators gpt5_mini --probe-only
```

Each smoke run should finish in under 2 minutes and cost < $0.10. If any
of them fails, fix it before starting the real cells.

---

## 8. Common gotchas (read before launching)

- **`gpt5_mini` as L3 reviewer**: returns empty content because reasoning
  tokens consume the entire `max_tokens` budget. Use it as a generator,
  not as the L3 critic.
- **OpenRouter Azure routing** rejects `max_tokens < 16`. The code uses
  `max_tokens=32` for non-reasoning models and `max_tokens=200` for gpt-5.
- **MBPP+ test format**: tests are self-running scripts (call function
  inline at module level). Do not append `check(entry_point)`.
- **LCB starter_code** must be honored. The functional runner imports the
  module and calls `Solution().<method>(*args)`. A stdin runner gives 0%.
- **Self-review L3**: when the L3 reviewer == the generator, the L3 gap
  collapses (~50% on hard, ~95% on medium). Always pair L3 with a
  different model family.
- **SWE-Bench arm64**: pre-built images are x86_64 only. Run Phase 2 on
  the cluster (`MBZUAI-Artem-*`), not on an M-series Mac.
- **Cluster disk quota**: home is 200 GB. Send `HF_HOME` and `TMPDIR` to
  `/mnt/data/...`. Run `podman system prune -a -f` and `conda clean -a`
  if a run errors out with `ENOSPC`.
- **`qwen3_coder` SWE-Bench parser bug**: thinking-block parser failure
  causes ~100% empty patches on SWE. Known issue; fix before relying on
  qwen3 SWE numbers.
- **Cost caps**: every long-running script honours
  `--max-cost-usd-per-model`. The cost tracker writes a per-call audit
  log and aborts cleanly when the cap is hit. The script is resumable —
  re-running picks up where it left off.

---

## 9. Where everything lands

```
data/
  lcb_calibration_{hard,medium,easy}/<gen>/    # function-level synthesis
  mbpp_calibration/<gen>/
  humaneval_calibration/<gen>/
  swebench_{lite,verified}/<gen>/              # repo-level patch generation
  <bench>_{calibration,}_iter/<gen>/            # iter trajectories + kernels
  PAPER_TABLE.{json,csv}                       # final aggregate
  paper_figs/*.{png,pdf}                       # figures for the paper
```

For the paper notebook (final analysis + figures), see
`experiments/orchestration/wandb/analysis.ipynb`.

---

## 10. Filling the GrFt / DPFt columns (sister codebase)

`GrFt` (`greedy_fitted`) and `DPFt` (`dp_fitted`) in `tab:full_results`
are produced by a **different codebase**:
`bayesian_optimization_for_code_testing/agent-bugfix-bayes/`. They are
fitted variants of the Bayesian controllers (Beta-Binomial-fitted
critic likelihoods + measured kernel, instead of hand-tuned constants).

### 10.1 Two ways the cells are produced

| Path | Script | LLM calls? | Output | Used for |
|---|---|---|---|---|
| **Replay** | `pytest tests/test_humaneval_simulation.py` | No | `sim_results/humaneval_simulation_metrics.json` | DP-vs-Greedy / hand-vs-fitted comparison on existing patches |
| **Replay** | `pytest tests/test_swebench_simulation.py` | No (uses Docker for critic outcomes) | `sim_results/swebench_simulation_metrics.json` | SWE-Bench appendix sub-rows |
| **Live end-to-end** | `python scripts/run_humaneval_full.py` | **Yes** | `sim_results/humaneval_full_endtoend.json` | The HumanEvalFix + CodeContests GrFt/DPFt values in the main paper table |

The fit step itself (`src/abbo/realworld/calibration/`) never needs
new API calls — it's Beta-Binomial Laplace smoothing on existing
calibration / iter data. **It's the evaluation that does or doesn't
call the LLM, depending on which path you use.**

### 10.2 Empty cells in the paper table — what's needed

| Empty cells | What to do | New API calls? |
|---|---|---|
| **Function-level synthesis** (LCB-{hard,medium,easy}, MBPP+, HumanEval+) | No adapter exists. `realworld/agents/` has only `humaneval_fix.py`, `code_contests.py`, `swe_bench.py`. Would need a new adapter per benchmark mirroring those three. **Reasonable to leave these `--` for EMNLP.** | Yes (if added) |
| **SWE-Bench Lite + Verified main rows** for missing generators | Adapter exists (`swe_bench.py`). Run the replay simulation via pytest (`-k swebench_simulation`). Needs Docker for critic recollection on missing instances, not LLM. | No (replay) |
| **HumanEvalFix + CodeContests** for `gpt5_mini`, `Qwen2.5-Coder-7B`, `Qwen2.5-Coder-32B` | Adapter exists. Edit `LLM_MODEL` in `scripts/run_humaneval_full.py` (currently `"openai/gpt-oss-20b:free"`) and re-run per generator; output saves to a fresh `humaneval_full_endtoend_<gen>.json`. | Yes (live agent) |

### 10.3 How to run

```bash
cd bayesian_optimization_for_code_testing/agent-bugfix-bayes
bash scripts/setup_env.sh
source .venv/bin/activate

# (a) Replay-based: HumanEvalFix + SWE-Bench simulations (no LLM)
#     Uses existing calibration data; outputs sim_results/*_simulation_metrics.json
pytest -q -k "humaneval_simulation or swebench_simulation" \
       --alluredir allure-results --clean-alluredir

# (b) Live end-to-end on HumanEvalFix for a new generator.
#     Edit LLM_MODEL in scripts/run_humaneval_full.py (line ~48),
#     and either also edit RESULTS_PATH or set it per-generator before launch.
LLM_MODEL=anthropic/claude-haiku-4.5 \
RESULTS_PATH=sim_results/humaneval_full_endtoend_haiku45.json \
python scripts/run_humaneval_full.py
#     (the script reads both via module globals; set them in the file or
#      patch the script to take CLI flags — see existing pattern).

# Resume after a crash: re-running the same command skips
# (task_id, variant) pairs already in the output JSON.
```

Per-instance cost on HumanEvalFix end-to-end is roughly 2–5 generator
calls (1 initial fix + 0–2 regenerations + 1 verifier), so an n=124
held-out split lands at ~$2–8 per closed-API generator. Local-vLLM
generators are free at the API layer.

### 10.4 Uploading results back to W&B

After producing a new `sim_results/*.json`, fold it into the W&B
project so the analysis notebook picks it up:

```bash
cd experiments/orchestration/wandb
python3 upload_runs.py --track abbo                # walks sim_results/
# OR a single cell:
python3 upload_runs.py --track abbo \
    --benchmark humanevalfix --generator haiku45
```

These land under `track:abbo` (the agent-bugfix-bayes track), not
`track:orchestration` — that's the convention `analysis.ipynb` uses to
distinguish replay-policy comparisons (orchestration) from fitted
end-to-end agent runs (abbo).

### 10.5 Practical recommendation for EMNLP deadline

- **High value, cheap:** fill HumanEvalFix + CodeContests GrFt/DPFt for `gpt5_mini`, `qwen3_coder`, `haiku45`, `sonnet45`. ~$30 total.
- **Medium value, cheap:** SWE-Bench Lite/Verified replay simulations for the missing generators. Docker required but no API spend.
- **Defer:** function-level synthesis GrFt/DPFt — requires writing new benchmark adapters. Not blocking the headline three-regime story.

---

## 11. After everything is done

1. `cd experiments/orchestration/wandb && python3 upload_runs.py` —
   pushes any local cells that aren't on W&B yet (idempotent;
   `--force` overwrites).
2. Open `analysis.ipynb`, run all cells with `force_refresh=True` on
   the W&B fetch — the notebook recomputes every paper table and
   figure from the W&B runs (no local-only path required).
3. Regenerate any figures consumed by the paper LaTeX
   (`paper_figs/*.{png,pdf}`).
4. See `emnlp2026/SUBMISSION_TODO.md` (in the sister paper folder) for
   the prioritised checklist that takes us from "experiments done" to
   "submission ready".
