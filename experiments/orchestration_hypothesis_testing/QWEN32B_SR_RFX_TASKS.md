# Qwen2.5-Coder-32B — Self-Refine / Reflexion tasks

**For:** Artem
**Goal:** fill the remaining SR/Rfx cells for `qwen25_32b` on W&B.
**Total:** 10 runs (~6–8 h wall-clock, ~$5–7 OpenRouter spend for L3 reviewer).
**Prerequisite:** the post-refactor codebase on `main` (`calibration/`, `iter/`, `_common/` packages).

## What we already have on W&B

Pulled `2026-05-23` from `nlpresearch.group/orchestration-hypothesis-testing`,
filtered to `config.generator=qwen25_32b`:

| Benchmark | Cal | Iter SR | Iter Rfx |
|---|---|---|---|
| `lcb_easy`, `lcb_medium`, `lcb_hard` | ✅ | ✅ | ✅ |
| `mbpp` | ✅ | ❌ | ❌ |
| `humaneval` | ✅ | ❌ | ❌ |
| `humanevalfix` | ❌ | ❌ | ❌ |
| `codecontests` | ❌ | ❌ | ❌ |
| `swe_lite`, `swe_verified` | ✅ | ✅ | ✅ |

## The 10 runs to launch

| # | Bench | What | Depends on |
|---|---|---|---|
| 1 | `humanevalfix` | calibration | — |
| 2 | `codecontests` | calibration | — |
| 3–4 | `mbpp` | iter SR + Rfx | existing cal |
| 5–6 | `humaneval` | iter SR + Rfx | existing cal |
| 7–8 | `humanevalfix` | iter SR + Rfx | #1 |
| 9–10 | `codecontests` | iter SR + Rfx | #2 |

## Step 0 — start vLLM for `qwen25_32b`

The script expects `Qwen/Qwen2.5-Coder-32B-Instruct` at `http://127.0.0.1:8003/v1`
(URL baked into `_common/generators.py`).

```bash
tmux new -s vllm_qwen32b
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --served-model-name Qwen/Qwen2.5-Coder-32B-Instruct \
    --port 8003 \
    --tensor-parallel-size 2     # adjust to your GPU layout
# verify: curl http://localhost:8003/v1/models
# detach: Ctrl-b d
```

If `/v1/models` doesn't return 200, fix that before continuing — everything below depends on it.

## Step 1 — fetch existing cal data for mbpp + humaneval

The iter step needs the calibration's `critic_results.jsonl` + `raw_responses/` directory as `--src-dir`. They're on W&B already; pull them locally:

```bash
cd ~/agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing
OUT=$HOME/qwen32b_runs_$(date +%Y%m%d)
mkdir -p $OUT/cal_cache

python3 -c "
import wandb
api = wandb.Api()
for bench in ('mbpp', 'humaneval'):
    runs = list(api.runs('nlpresearch.group/orchestration-hypothesis-testing',
                          filters={'config.experiment_type': 'calibration',
                                   'config.benchmark': bench,
                                   'config.generator': 'qwen25_32b'}))
    assert len(runs) == 1, f'expected 1 cal run for {bench}/qwen25_32b, got {len(runs)}'
    dst = '$OUT/cal_cache/' + bench + '__qwen25_32b'
    import os; os.makedirs(dst, exist_ok=True)
    for art in runs[0].logged_artifacts():
        if 'critic_results' in art.name or 'raw_responses' in art.name:
            art.download(root=dst)
    print(f'  fetched {bench}/qwen25_32b -> {dst}')
"
```

## Step 2 — runs 1–2: missing calibration

```bash
# 1) HumanEvalFix (N=164)
python -m calibration.humanevalfix \
    --output-dir $OUT/humanevalfix__qwen25_32b \
    --generators qwen25_32b \
    --n-instances 164 --n-patches 3 --seed 42 \
    --max-cost-usd-per-model 5.0   # generous L3-reviewer cap

# 2) CodeContests (N=165 in test parquet — uses parquet-only loader, ~63 MB download not 13 GB)
python -m calibration.codecontests \
    --output-dir $OUT/codecontests__qwen25_32b \
    --generators qwen25_32b \
    --n-instances 165 --n-patches 3 --seed 42 \
    --max-cost-usd-per-model 5.0
```

## Step 3 — runs 3–10: iter SR + Rfx

Loop over the 4 (benchmark, src-dir) pairs and run both methods for each. Each iter run takes ~30–90 min depending on the vLLM throughput.

```bash
# Map: (variant, --src-dir to read step-0 patches from)
# mbpp / humaneval use the cal_cache from step 1
# humanevalfix / codecontests use the cal output from step 2

declare -A SRC
SRC[mbpp]=$OUT/cal_cache/mbpp__qwen25_32b
SRC[humaneval]=$OUT/cal_cache/humaneval__qwen25_32b
SRC[humanevalfix]=$OUT/humanevalfix__qwen25_32b
SRC[codecontests]=$OUT/codecontests__qwen25_32b

for VARIANT in mbpp humaneval humanevalfix codecontests; do
  for METHOD in selfrefine reflexion; do
    python -m iter.refine \
        --method $METHOD --variant $VARIANT \
        --src-dir ${SRC[$VARIANT]} \
        --output-dir $OUT/iter/${VARIANT}__qwen25_32b__${METHOD} \
        --generators qwen25_32b \
        --n-instances 0 --steps 5 --seed 42 \
        --max-cost-usd-per-model 3.0 \
        --max-workers 4
  done
done
```

**Flag glossary:**
- `--n-instances 0` — sentinel for "all available instances in the
  calibration corpus" (the script has `... if n_instances > 0 else problems`).
  Set to a positive integer to cap.
- `--steps 5` — upper bound (exclusive) of the refinement-step loop:
  the script iterates `for t in range(1, steps)`, so `steps=5` gives
  4 actual refinement steps (t=1, 2, 3, 4) on top of the sunk step 0.
  Matches the script default and the existing LCB/SWE iter runs already
  on W&B for `qwen25_32b`.
- `--seed 42` — matches the 75/25 split discipline used everywhere
  else in the pipeline.
- `--max-workers 4` — parallel instance trajectories within one
  generator. Bump if vLLM has spare capacity.

## Step 4 — upload to W&B

```bash
cd ~/agents_with_uncertainty_research/experiments/orchestration/wandb

# Dry-run first to confirm what would upload:
python upload_runs.py --track orchestration --generator qwen25_32b --dry-run --verbose

# Then for real:
python upload_runs.py --track orchestration --generator qwen25_32b --verbose
```

`upload_runs.py` is idempotent — re-running skips runs that already exist on W&B.

## Step 5 — verify

Ping me when done. I'll re-run `analysis.ipynb` cell 21 (data inventory) and confirm the `qwen25_32b` row flips fully green.

Quick sanity-check Artem can run locally if curious:

```python
# In a Python shell with wandb available
import wandb
api = wandb.Api()
needed = [
    ("mbpp", "selfrefine"), ("mbpp", "reflexion"),
    ("humaneval", "selfrefine"), ("humaneval", "reflexion"),
    ("humanevalfix", "selfrefine"), ("humanevalfix", "reflexion"),
    ("codecontests", "selfrefine"), ("codecontests", "reflexion"),
]
for bench, method in needed:
    runs = list(api.runs("nlpresearch.group/orchestration-hypothesis-testing",
                          filters={"config.experiment_type": "iter",
                                   "config.benchmark": bench,
                                   "config.generator": "qwen25_32b",
                                   "config.method": method}))
    print(f"  {bench:>13} / {method:>10}: {'✅' if runs else '❌'}")
```

When all 8 print ✅, the iter side is done. Calibration runs (cal for HEFix + CC) will show up under `config.experiment_type=calibration` instead.

## Gotchas

- vLLM crashes mid-run → tmux session likely OOM'd. Lower `--max-model-len` or `--gpu-memory-utilization`, restart, re-run the affected `iter.refine` call (the script has `--extend-existing` resume but you have to add the flag).
- L3 reviewer calls go to OpenRouter (claude-haiku-4.5). If `OPENROUTER_API_KEY` isn't set, `iter.refine` auto-loads `.env` walking up to 5 parent dirs from the script. If you stored the key elsewhere, `export OPENROUTER_API_KEY=...` before launching.
- CodeContests test parquet download: ~20s the first time, cached thereafter under `$HF_HOME` (default `~/.cache/huggingface`).
- HumanEvalFix loads `bigcode/humanevalpack` (~30 MB) on first run.

Questions → ping Karim.
