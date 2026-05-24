# `single_method` measurement campaign — runbook (Path A)

**Goal:** fill the 30 missing (benchmark, generator) cells with measured
refine transition kernels, restoring `bayesian_DP` coverage uniformly
across all 54 cells. See `SINGLE_METHOD_AUDIT.md` for the gap analysis.

**Owner:** TBD
**Estimated effort:** ~1-2 hr coding + 15-30 h wall-clock compute,
parallelisable.

## What we need

For each of the 30 missing (b, g) cells, an iter run with:
- `config.experiment_type = "iter"`
- `config.method = "single_method"`
- Artifact `iter_records.jsonl` containing per-step `Y_t` outcomes (one
  row per `(instance_id, step)`)
- `transition_kernel.json` summarising P_fix_given_broken / P_break_given_correct

The notebook's cell 13 builds `KERN_MEAS[(b, g)]` from these artifacts at
analysis time.

## Step 1 — add `--method single_method` to refine scripts

Both `iter/refine.py` and `iter/refine_swe.py` currently restrict
`--method` to `{selfrefine, reflexion}`. We add a third choice that runs
vanilla generate→refine→refine→… with no Stage-1 critique/reflect call.

### `iter/refine.py`

Change the argparse line:

```python
# Before (refine.py:1330):
parser.add_argument("--method", required=True, choices=["selfrefine", "reflexion"])

# After:
parser.add_argument("--method", required=True,
                    choices=["selfrefine", "reflexion", "single_method"])
```

In the main loop, add a third branch alongside the existing selfrefine /
reflexion stages. The vanilla refine branch:

- Skips Stage 1 entirely (no critique, no reflect)
- Stage 2 (refine) gets the base prompt unchanged — no critique injection
- `method_specific` records `method = "single_method"` and no extra fields
- Telemetry record uses `action_type="refine"` (not "reflect")

Concrete shape (pseudocode — adapt to the existing branch structure):

```python
if method == "single_method":
    # No Stage 1. Skip directly to the refine call below using base_prompt.
    method_specific = {"method": "single_method"}
    # Optional: telemetry stub so action_telemetry.jsonl stays uniform
    if tele is not None:
        tele.record(action_type="no_op",
                    runtime_s=0.0,
                    instance_id=inst, step=t, api_cost_usd=0.0,
                    extra={"purpose": "single_method_skip_stage1",
                           "benchmark": variant})
elif method == "selfrefine":
    # ... existing critique-then-refine code ...
elif method == "reflexion":
    # ... existing reflect-then-refine code ...

# Stage 2 (refine) runs unchanged for all three methods.
```

### `iter/refine_swe.py`

Identical pattern — change the argparse `choices` and add the
`single_method` branch that skips Stage 1.

### Tests

- Add a smoke test: `python -m iter.refine --method single_method
  --variant mbpp --src-dir <cal-cache> --n-instances 2 --steps 3 ...`
  should produce 2 trajectories × 2 refine steps each, no critique/reflect
  calls in `action_telemetry.jsonl`.
- Verify `transition_kernel.json` schema matches existing single_method
  runs (same `kernel_all` wrapper with `P_stay_*` fields).

## Step 2 — generator-by-generator campaign matrix

Each cell below = one `iter.refine --method single_method` invocation.

### 2a. mbpp + humaneval (12 cells across all 6 generators)

These are missing for every generator. Run in parallel across generators
where vLLM capacity allows.

```bash
for GEN in gpt5_mini qwen3_coder qwen25_32b haiku45 sonnet45 gpt_oss_20b; do
  for VARIANT in mbpp humaneval; do
    SRC=$HOME/runs_$(date +%Y%m%d)/cal_cache/${VARIANT}__${GEN}
    OUT=$HOME/runs_$(date +%Y%m%d)/iter/${VARIANT}__${GEN}__single_method
    python -m iter.refine \
        --method single_method --variant $VARIANT \
        --src-dir $SRC \
        --output-dir $OUT \
        --generators $GEN \
        --n-instances 0 --steps 5 --seed 42 \
        --max-cost-usd-per-model 3.0 \
        --max-workers 4
  done
done
```

(Fetch calibration cache first per the existing PR #9 runbook pattern.)

### 2b. Generator-wide gaps for qwen25_32b + gpt_oss_20b (16 cells)

These two have **zero** single_method coverage. Run for all 8
non-mbpp/humaneval benchmarks (mbpp/humaneval handled in 2a):

```bash
for GEN in qwen25_32b gpt_oss_20b; do
  for VARIANT in lcb_easy lcb_medium lcb_hard humanevalfix codecontests; do
    # function-level: use refine.py
    python -m iter.refine \
        --method single_method --variant $VARIANT \
        --src-dir $SRC --output-dir $OUT \
        --generators $GEN \
        --n-instances 0 --steps 5 --seed 42 \
        --max-cost-usd-per-model 3.0 --max-workers 4
  done
  for VARIANT in swe_lite swe_verified; do
    # SWE: use refine_swe.py
    python -m iter.refine_swe \
        --method single_method --dataset princeton-nlp/SWE-bench_$VARIANT \
        --src-dir $SRC --output-dir $OUT \
        --generators $GEN --n-instances 0 --steps 5 --seed 42
  done
done
```

### 2c. Individual partial gaps

| Generator | Missing benchmarks |
|---|---|
| gpt5_mini | humanevalfix, codecontests, swe_lite |
| qwen3_coder | swe_lite |

(mbpp + humaneval already covered in 2a; qwen25_32b + gpt_oss_20b already
covered in 2b.)

## Step 3 — upload to W&B

After all runs complete, single uniform upload:

```bash
cd experiments/orchestration/wandb

# Dry-run first:
python upload_runs.py --track orchestration --dry-run --verbose | grep single_method

# Then for real:
python upload_runs.py --track orchestration --verbose
```

`upload_runs.py` is idempotent — re-running skips existing runs.

**Important:** `upload_runs.py`'s legacy single_method block (lines 362-416)
expects the OLD directory layout — `iter_records.jsonl` and
`transition_kernel.json` directly under `<cell-dir>/`, NOT under a
`single_method/` subdirectory. Either:
- (a) Have the new `--method single_method` runs write to that layout
  (no method subdir), OR
- (b) Extend `upload_runs.py` to also handle `<cell-dir>/single_method/`
  for the new runs.

Option (a) keeps upload_runs.py simple; option (b) keeps directory layout
uniform with selfrefine/reflexion. Pick one when implementing Step 1.

## Step 4 — verify coverage

```python
import wandb
api = wandb.Api()
ALL_BENCH = ["lcb_easy", "lcb_medium", "lcb_hard", "mbpp", "humaneval",
             "humanevalfix", "codecontests", "swe_lite", "swe_verified"]
ALL_GEN = ["gpt5_mini", "qwen3_coder", "qwen25_32b",
           "haiku45", "sonnet45", "gpt_oss_20b"]
GEN_ALIAS = {"gpt_oss_20b_local": "gpt_oss_20b"}

runs = list(api.runs("nlpresearch.group/orchestration-hypothesis-testing",
                     filters={"config.experiment_type": "iter",
                              "config.method": "single_method"}, per_page=200))
have = {(r.config["benchmark"], GEN_ALIAS.get(r.config["generator"], r.config["generator"]))
        for r in runs if r.config.get("track") == "orchestration"}

missing = [(b, g) for b in ALL_BENCH for g in ALL_GEN if (b, g) not in have]
print(f"single_method coverage: {len(have)}/54 cells")
if missing:
    for b, g in missing:
        print(f"  missing: {b} / {g}")
else:
    print("COMPLETE — all 54 cells have single_method iter.")
```

## Step 5 — re-run notebook

After full coverage:

1. Re-run cells 5/13 (W&B fetch + STAT_POLICY) — KERN_MEAS now has all 54
   entries, no `BDP_NO_KERNEL_CELLS` warning at the end of cell 13.
2. Re-run cells 23 / 41 / 52 / 78 — no `ERROR:` per-cell prints.
3. Visual: bayesian_DP appears in every (b, g) panel of every figure.

## Cost / time estimates

| Phase | Cells | Per-cell time | Total | $-cost |
|---|---|---|---|---|
| 2a (mbpp + humaneval × 6 gens) | 12 | ~20-40 min | ~4-8 h | ~$3-5 |
| 2b (qwen25_32b + gpt_oss_20b × 7 non-mbpp/humaneval benches) | 16 | varies (30 min function / 60+ min SWE) | ~8-16 h | ~$8-12 |
| 2c (5 individual cells) | 5 | varies | ~2-4 h | ~$2-3 |
| **Total** | **33** | | **~15-30 h** | **~$15-20** |

(33 not 30 because qwen25_32b + gpt_oss_20b on mbpp + humaneval is counted
in both 2a and 2b; deduplicate in execution.)

Parallelism: each generator runs on its own vLLM or API endpoint, so
phases 2a/2b/2c can interleave across generators trivially. Wall-clock
on a single host with one vLLM at a time is the upper bound.

## Open questions

1. **Directory layout** (Step 3): legacy `<cell>/iter_records.jsonl` vs
   modern `<cell>/single_method/iter_records.jsonl`. Pick one.
2. **Telemetry shape**: should the `single_method` runs record a `no_op`
   action_type for the skipped Stage 1, or just omit the row entirely?
   Choice affects how `action_telemetry.jsonl` aggregation looks.
3. **mbpp / humaneval anomaly**: why were these never in the original
   campaign? If there's a known reason (e.g. trivial saturated kernel),
   document it before running — might save compute.
