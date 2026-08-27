# SWE-Bench Verified expansion (200 → 489)

Goal: bring the SWE-Bench **Verified** cells from the stratified 200-instance
subset up to the full **489-instance working pool** (500 total − 11 build-errored
instances that the harness can't build; see `verified_200_subset.json`
→ `errored_excluded`).

## Current state (verified from disk + W&B, 2026-08-22)

| Generator | Clean Verified done | Needs | Subset file |
|---|---|---|---|
| sonnet45 | 200 (= stratified subset) | **+289** | `verified_missing_289.json` |
| qwen3_coder | 200 (= stratified subset) | **+289** | `verified_missing_289.json` |
| haiku45 | 200 (= stratified subset) | **+289** | `verified_missing_289.json` |
| gpt5_mini | 0 clean (all 200-runs broken, ~12% resolve) | **full 489** | `verified_gpt5_mini_missing_489.json` |

sonnet45 / qwen3_coder / haiku45 share the **identical** 289-ID list (all three
used the same stratified subset). gpt5_mini has no usable Verified run, so it
runs the full 489 fresh. The open-weight models (gpt_oss_20b, qwen25_32b) are
already at 500 and are not part of this expansion.

## Who runs what

- **Vlad** (Artem-1): `qwen3_coder` (+289), `sonnet45` (+289)
- **Viktor**: `haiku45` (+289), `gpt5_mini` (full 489)

## Prerequisites (per host, one-time)

1. Checkout this branch: `git fetch && git checkout feat/swe-verified-500-expansion && git pull`
2. `.env` at repo root with a **funded** `OPENROUTER_API_KEY`
3. Rootless podman socket up: `systemctl --user enable --now podman.socket`
4. Patched swebench harness: `python experiments/orchestration_hypothesis_testing/scripts/patch_swebench_harness.py`
5. ≥ ~10 GB free `/mnt/data` quota headroom (`quota -s`)

## Run (one command per generator, detached)

From `experiments/orchestration_hypothesis_testing/`:

```bash
# Vlad
scripts_pipeline/run_verified_expand.sh sonnet45    data/swebench_verified_calibration_full/verified_missing_289.json
scripts_pipeline/run_verified_expand.sh qwen3_coder data/swebench_verified_calibration_full/verified_missing_289.json

# Viktor
scripts_pipeline/run_verified_expand.sh haiku45     data/swebench_verified_calibration_full/verified_missing_289.json
scripts_pipeline/run_verified_expand.sh gpt5_mini   data/swebench_verified_calibration_full/verified_gpt5_mini_missing_489.json
```

Detached with tmux (survives disconnect):

```bash
tmux new -d -s swe_ver_exp_<gen> \
  "bash scripts_pipeline/run_verified_expand.sh <gen> <subset.json> 2>&1 | tee -a ../../logs/swebench_verified_expand_<gen>/master.log"
```

The launcher runs, Verified-only: **Cal → critics → Self-Refine (8 workers) →
eval → backfill → Reflexion (8 workers) → eval → backfill**, scoped to the
subset, into fresh `swebench_verified_{calibration,realbaselines_selfrefine,realbaselines_reflexion}_exp/<gen>/`.

`MAX_REFINE_WORKERS=8` is on by default (see PR #12) — the refine stages run
~8× faster; the harness eval (podman builds) is the multi-hour long pole.

### Cost caps (ceilings; override via args or env)

| Generator | Cal | Refine (SR & Rfx each) | Critics |
|---|---|---|---|
| sonnet45 | $50 | $25 | $25 |
| haiku45 | $25 | $15 | $15 |
| qwen3_coder | $20 | $12 | $12 |
| gpt5_mini | $20 | $12 | $15 |

Override: `scripts_pipeline/run_verified_expand.sh haiku45 <subset> 30 18 18`
or `CAL_CAP=30 REFINE_CAP=18 ... run_verified_expand.sh ...`.

## Safety

The launcher writes only to fresh `*_exp` dirs, so the existing 200-instance
`*_full` cells are never touched. Back up the `*_full/<gen>` dirs first if you
want extra insurance.

## Monitor

```bash
tail -f logs/swebench_verified_expand_<gen>/master.log
```

## Merge (after a generator finishes)

- **sonnet45 / qwen3_coder / haiku45**: concatenate the 289 `_exp` cell into the
  existing 200 `_full` cell (disjoint instances → clean append):
  ```bash
  for f in predictions.jsonl predictions_p0.jsonl predictions_p1.jsonl predictions_p2.jsonl \
           critic_results.jsonl generation_records.jsonl; do
    cat data/swebench_verified_calibration_exp/<gen>/$f \
      >> data/swebench_verified_calibration_full/<gen>/$f
  done
  # same idea for the SR/Rfx iter_records.jsonl under realbaselines_*_{exp,full}/<gen>/<method>/
  ```
  Re-run `from_spotcheck.py` over the merged `_full` cell (no subset) to
  recompute `critic_results` + likelihood tables on the full 489, then
  re-eval / re-backfill so `iter_records` carry Y for all 489.
- **gpt5_mini**: the `_exp` cell IS the full clean 489 — promote it to `_full`
  (the old broken 200 is discarded), no merge needed.

Then re-run the analysis notebook / `upload_runs.py` to refresh the cube.
