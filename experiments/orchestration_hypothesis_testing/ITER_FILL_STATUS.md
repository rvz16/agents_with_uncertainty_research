# Iter (Self-Refine + Reflexion) fill — VM status (live)

**Snapshot**: 2026-05-25 23:13 +04 (Asia/Dubai) — **upload pass 1 complete**
**Branch**: `main` (HEAD `fd980ea2` — analysis.ipynb §16 + staleness fixes)
**OpenRouter key**: `sk-or-v1-55161e1c…061c470c` ($250 budget; **$57.54 used**, $192 remaining)

## ✅ Upload pass 1 — 17 W&B runs landed (2026-05-25 23:10–23:13 +04)

| # | Cell | Method | W&B run id | Source VM | Cleanup needed? |
|---|---|---|---|---|---|
| 1 | mbpp / qwen3-coder            | SR  | `0dmepyfw` | mbz1 | no |
| 2 | mbpp / qwen3-coder            | Rfx | `l9mwuub3` | mbz1 | no |
| 3 | HumanEval+ / qwen3-coder      | SR  | `dsap7vod` | mbz1 | no |
| 4 | HumanEval+ / qwen3-coder      | Rfx | `lvmzcd39` | mbz1 | no |
| 5 | codecontests / qwen3-coder    | SR  | `ezujpws8` | mbz1 | no |
| 6 | codecontests / qwen3-coder    | Rfx | `4gcy31li` | mbz1 | no |
| 7 | codecontests / sonnet-4.5     | Rfx | `k1lqui6a` | mbz1 | **YES** — 322 records / 99 inst (normalized from legacy underscore format) |
| 8 | LCB-hard / gpt-5-mini         | SR  | `wfmj971t` | mbz3 | no |
| 9 | LCB-hard / gpt-5-mini         | Rfx | (re-uploaded) | mbz3 | no |
| 10 | LCB-medium / gpt-5-mini      | SR  | (re-uploaded) | mbz3 | no |
| 11 | LCB-medium / gpt-5-mini      | Rfx | (re-uploaded) | mbz3 | no |
| 12 | HumanEval+ / gpt-5-mini      | SR  | (re-uploaded) | mbz3 | no |
| 13 | HumanEval+ / gpt-5-mini      | Rfx | (re-uploaded) | mbz3 | no |
| 14 | codecontests / gpt-5-mini    | SR  | (re-uploaded) | mbz3 | no |
| 15 | codecontests / gpt-5-mini    | Rfx | (re-uploaded) | mbz3 | no |
| 16 | codecontests / claude-haiku-4.5 | SR  | (re-uploaded) | mbz4 | **YES** — 387 records / 152 inst (de-overlapped from 666 raw) |
| 17 | codecontests / claude-haiku-4.5 | Rfx | (re-uploaded) | mbz4 | **YES** — 620 records / 152 inst (de-overlapped from 947 raw) |

## ⏳ Pending pass 2 (1 cell still running)

- **codecontests / claude-sonnet-4.5 / selfrefine** — on `mbz1.iter_b`, ~80% done as of 23:09 +04. Step 3 of `1575_C. Cyclic Sum`. ETA: ~30–60 min remaining.

**After it finishes**, re-run on mbz1:
```bash
ssh MBZUAI-Artem-1 'python3 /tmp/cleanup_iter_codecontests.py --gen sonnet45 --methods selfrefine --apply'
ssh MBZUAI-Artem-1 'cd /mnt/data/users/vlad.smirnov/agents_with_uncertainty_research &&
  source .env && export WANDB_DIR=/mnt/data/users/vlad.smirnov/wandb_cache &&
  python3 experiments/orchestration/wandb/upload_runs.py \
    --track orchestration --experiment iter \
    --benchmark codecontests --generator sonnet45 \
    --data-root experiments/orchestration_hypothesis_testing/data --force'
```

## Data quality findings (during this batch)

1. **`code_chars=0` edge cases on gpt5_mini** (4–19 records per LCB/CC cell). These are **legitimate refinement failures** — `step >= 1`, `L0_syntax=False`, `Y=0`, `stop_decision` present. Real signal for the transition kernel, not bugs.

2. **codecontests instance_id format collision** (haiku45 + sonnet45). Two distinct naming conventions on disk:
   - Current cal: `1575_A. Another Sorting Problem` (space + period)
   - Legacy iter: `1575_A._Another_Sorting_Problem` (all underscores)

   Fixed via `/tmp/cleanup_iter_codecontests.py`:
   - Normalizes legacy underscore IDs to cal format
   - Drops records whose normalized ID is not in cal cohort (`_AND_version` / `_OR_version` variants — Codeforces problems with multiple alt formulations that aren't in the canonical 165)
   - Dedupes overlapping trajectories (haiku45 had 99 of 152 problems run BOTH legacy + modern; modern records kept, legacy duplicates dropped to avoid spurious step pairing in kernel computation)
   - Recomputes `transition_kernel.json` from the cleaned records (Beta(1,1) via `_common.kernel.compute_transition_kernel_from_pairs`)

3. **LCB cells capped at calibration set, not canonical N** (gpt-5-mini):
   - lcb_medium: **183 / 207** (canonical N comes from `--lcb-version all`; calibration set has only 183 of those problems)
   - lcb_hard:   **77 / 102**
   - To close the gap: re-run calibration with `--lcb-version all` first, then iter. Not done in this batch.

## Patches landed

1. **`upload_runs.py`**: `ORCH_REALBASELINES_BENCHMARKS` extended with new `<bench>_iter/` subdir convention for {lcb_easy, lcb_medium, lcb_hard, mbpp, humaneval, humanevalfix, codecontests}. Existing legacy `<bench>_*_realbaselines` entries kept for back-compat. The per-method upload loop filters out empty paths via `tk.exists()`, so duplicate `(bench, gen)` keys with different subdirs are safe.

2. **`/tmp/cleanup_iter_codecontests.py`** (helper on mbz1 / mbz3 / mbz4): normalize + filter + dedupe + recompute kernel. Idempotent. Run per `(gen, [methods])`.

## ORIGINAL CONTEXT (before this batch — historical)

## Goal

Complete `selfrefine` (SR) + `reflexion` (Rfx) iter trajectories for the 9 (benchmark, generator) cells where iter coverage lagged behind calibration coverage, per the data_coverage table:

| # | Benchmark | Generator | Cal | SR (before) | Rfx (before) |
|---|---|---|---|---|---|
| 1 | HumanEval+ | qwen3-coder | 164/164 | 16/164 | 18/164 |
| 2 | HumanEval+ | gpt-5-mini | 164/164 | 151/164 | 151/164 |
| 3 | MBPP+ | qwen3-coder | 378/378 | 346/378 | 346/378 |
| 4 | LCB-medium | gpt-5-mini | 207/207 | 183/207 | 183/207 |
| 5 | LCB-hard | gpt-5-mini | 102/102 | 77/102 | 77/102 |
| 6 | CodeContests | gpt-5-mini | 165/165 | 90/165 | 90/165 |
| 7 | CodeContests | qwen3-coder | 165/165 | 112/165 | 112/165 |
| 8 | CodeContests | claude-haiku-4.5 | 165/165 | 112/165 | 112/165 |
| 9 | CodeContests | claude-sonnet-4.5 | 165/165 | 112/165 | 112/165 |

## VM workload split (5 tmux sessions, 3 VMs)

Generator convention follows previous calibration runs: gpt-5-mini → mbz3; qwen3-coder + sonnet-4.5 → mbz1; haiku-4.5 → mbz4.

| VM | Host (IP) | Tmux | Cells (in run order) |
|---|---|---|---|
| **mbz3** | 10.127.105.14 | `iter_main` | LCB-medium → LCB-hard → HumanEval+ → CodeContests (all gpt-5-mini) |
| **mbz3** | 10.127.105.14 | `iter_lcb` | LCB-medium + LCB-hard (gpt-5-mini) **with `--lcb-version all`** — see "LCB problem-set caveat" below |
| **mbz1** | 10.127.105.20 | `iter_a` | HumanEval+ / qwen3-coder (biggest cell; runs solo) |
| **mbz1** | 10.127.105.20 | `iter_b` | MBPP+ → CodeContests qwen3 → CodeContests sonnet45 |
| **mbz4** | (mbz4 IP) | `iter_main` | CodeContests / claude-haiku-4.5 |

## Per-cell pipeline (what each tmux loop does)

For each `(bench, gen, variant, N_canonical, extra_args)` tuple, the per-VM tmux loop runs:

1. `python3 prep_cell.py --bench $bench --gen $gen --base-dir <data>` (shipped to all VMs at `/mnt/.../agents_with_uncertainty_research/prep_cell.py`)
   - Downloads the **latest** cal `critic_results.jsonl` from W&B
   - **Merges `raw_responses` across ALL cal runs (union)** — the union recovery is essential because older runs (100-inst snapshots) and newer runs (extend-existing 64-inst increments) each upload disjoint subsets; without the union, refine.py's eligibility check fails on the missing `<inst>_p0.txt` files
   - Downloads the **latest** iter `iter_records.jsonl` for both SR and Rfx
2. `python refine.py --method <selfrefine|reflexion> --variant <variant> --src-dir <cal> --output-dir <iter> --generators <gen> --n-instances <N> --extend-existing --max-cost-usd-per-model 10.0 <extra>`
   - SR first, then Rfx
   - `--extend-existing` skips instance IDs already in `iter_records.jsonl`

### Variant mapping

| bench | --variant | extra args | canonical N |
|---|---|---|---|
| humaneval | humaneval | (none) | 164 |
| mbpp | mbpp | (none) | 378 |
| codecontests | codecontests | (none) | 165 |
| lcb_medium | lcb | `--difficulty medium --platform leetcode --lcb-version all` | 207 |
| lcb_hard | lcb | `--difficulty hard --platform leetcode --lcb-version all` | 102 |

## LCB problem-set caveat

`refine.py --variant lcb` defaults to `--lcb-version v1` which loads only `test.jsonl` (90 medium problems, ~37 hard problems). The canonical N=207/102 from the coverage table comes from `--lcb-version all` (union of test1..test6.jsonl). The first run of LCB cells on mbz3.iter_main used the v1 default and exited with `0 new instances to run` because every problem in the v1 subset was already covered. The `iter_lcb` tmux re-runs LCB cells with `--lcb-version all` to pick up the v2..v6 instances.

## Disk + cache notes

- **mbz4** had `/` at 100% capacity before launch. Cleaned 120G of legacy caches (`/home/.cache/{huggingface,wandb,pip,vllm}`) that were already redirected to `/mnt` via env vars. `/` now at 94% / 117G free. The redirected paths are:
  - `TMPDIR=/mnt/data/users/vlad.smirnov/tmp`
  - `HF_HOME=/mnt/data/users/vlad.smirnov/hf_home`
  - `WANDB_DIR=/mnt/data/users/vlad.smirnov/wandb_cache`
  - `XDG_CACHE_HOME=/mnt/data/users/vlad.smirnov/xdg_cache`
- **mbz1**: 51% / + 84% /mnt — fine.
- **mbz3**: 70% / + 86% /mnt — fine.

## Log file paths

| VM | Log dir |
|---|---|
| mbz1 | `~/cal_logs/iter_<bench>_<gen>_<method>.log` |
| mbz3 | `~/cal_logs/iter_<bench>_<gen>_<method>.log` (the LCB --all variant suffixes with `_all`) |
| mbz4 | `/mnt/data/users/vlad.smirnov/cal_logs/iter_<bench>_<gen>_<method>.log` |

Each cell also writes its actual data to `<repo>/experiments/orchestration_hypothesis_testing/data/<bench>_iter/<gen>/<method>/iter_records.jsonl` (the canonical refine.py output path).

## Status check command (re-runnable)

```bash
# From local Mac, on VPN:
for vm in MBZUAI-Artem-1 MBZUAI-Artem-3 MBZUAI-Artem-4; do
  echo "=== $vm ==="
  ssh -o ConnectTimeout=10 $vm 'tmux ls; echo;
    ls -lt ~/cal_logs/iter_*.log 2>/dev/null | head -5 || \
      ls -lt /mnt/data/users/vlad.smirnov/cal_logs/iter_*.log 2>/dev/null | head -5;'
done
```

## ETA (rough, at 2026-05-25 21:55)

| Cell | New SR | New Rfx | Where | Throughput | ETA |
|---|---|---|---|---|---|
| HumanEval+ / qwen3-coder | 148 | 146 | mbz1.iter_a | ~30 inst/min (qwen3 + parallel) | ~10 min total for both methods (DONE for SR at 21:54; Rfx running) |
| HumanEval+ / gpt-5-mini | 13 | 13 | mbz3.iter_main | mid throughput | ~15 min |
| MBPP+ / qwen3-coder | 32 | 32 | mbz1.iter_b | fast | DONE for SR at 21:51; Rfx running |
| LCB-medium / gpt-5-mini | up to ~120 | up to ~120 | mbz3.iter_lcb | mid throughput | ~30–45 min depending on `--lcb-version all` discovery |
| LCB-hard / gpt-5-mini | up to ~25 | up to ~25 | mbz3.iter_lcb | mid throughput | ~15 min |
| CodeContests / gpt-5-mini | 75 | 75 | mbz3.iter_main (after HE+) | slow (long CC contexts) | ~2 h |
| CodeContests / qwen3-coder | 53 | 53 | mbz1.iter_b (after MBPP+) | fast | ~30 min |
| CodeContests / claude-haiku-4.5 | 53 | 53 | mbz4.iter_main | mid | ~1 h |
| CodeContests / claude-sonnet-4.5 | 53 | 53 | mbz1.iter_b (last) | slow + expensive | ~1.5 h |

**Total wall time** (parallel across 3 VMs): **~2–3 h** for everything except mbz1.iter_a's reflexion, which may take another hour on top.

## Upload-to-W&B plan (after each VM completes)

The new iter data needs to be pushed to W&B so the notebook's cell 5 (with the staleness/dedupe patches from commit `fd980ea2`) can pick it up. Per-VM upload command pattern:

```bash
# Once a VM hits ALL_DONE_<VM>:
ssh <vm> 'cd /mnt/.../agents_with_uncertainty_research && \
  <activate-venv-or-export-env-vars> && \
  python experiments/orchestration/wandb/upload_runs.py \
    --track orchestration --experiment iter \
    --data-root experiments/orchestration_hypothesis_testing/data \
    --force -v'
```

⚠️ **`upload_runs.py` does NOT currently list mbpp/humaneval/humanevalfix/codecontests in `ORCH_ITER_BENCHMARKS`** (only LCB + SWE-Bench). For the new iter data on these benchmarks to upload, the list needs to be extended — same pattern as the calibration fix we did for `codecontests` in commit `fd980ea2`. Add the missing entries first:

```python
ORCH_ITER_BENCHMARKS = [
    ("lcb_hard",     "lcb_calibration_v2_iter"),
    ("lcb_medium",   "lcb_calibration_medium_iter"),
    ("lcb_easy",     "lcb_calibration_easy_iter"),
    ("swe_lite",     "swebench_lite_iter"),
    ("swe_verified", "swebench_verified_iter"),
    # ADD THESE:
    ("mbpp",         "mbpp_iter"),
    ("humaneval",    "humaneval_iter"),
    ("humanevalfix", "humanevalfix_iter"),
    ("codecontests", "codecontests_iter"),
]
```

And the local subdirs match what prep_cell.py + refine.py used (`<bench>_iter`).

## Recovery / continuation

If a tmux session dies mid-run:

```bash
# Re-run the SAME tmux launch command — extend-existing skips done instance_ids.
# The script picks up where it left off; only the instance(s) interrupted
# mid-step may be re-run (or partially logged with stale fields).
```

The only risk: an instance whose iter_records.jsonl entry was being appended at kill-time. refine.py's append-only design means at most one record is half-written. The eligibility check uses the `instance_id` set, so a partial record's instance is treated as done if it has any line in iter_records.jsonl — meaning that one instance gets skipped on resume and ends up with truncated step data. To redo it cleanly: delete its lines from iter_records.jsonl manually before resuming.

## Open items / risks

1. **`upload_runs.py` patch needed** before uploading iter for mbpp/humaneval/humanevalfix/codecontests (see "Upload-to-W&B plan" above).
2. **raw_responses union** is essential for prep_cell.py — if a new cell ever uploads cal data without raw_responses, that cell's iter coverage will be capped at the union count of all historical raw_responses.
3. **OpenRouter budget**: monitor via the `/auth/key` endpoint. Cap is $250; estimated this batch costs ~$50–150. Each per-model cap is $10 (passed via `--max-cost-usd-per-model 10.0`).
4. **HumanEval+/qwen3-coder** ran 144 new instances in 5 min for $0.10 — *much* cheaper than expected. Other cells should also come in well under budget.

## Quick sanity checks

```bash
# 1. Are all 5 tmux sessions alive?
for vm in MBZUAI-Artem-1 MBZUAI-Artem-3 MBZUAI-Artem-4; do
  ssh $vm "echo === $vm ===; tmux ls 2>/dev/null | grep iter"
done

# 2. Per-cell records growth (should increase monotonically until done):
for vm in MBZUAI-Artem-1 MBZUAI-Artem-3 MBZUAI-Artem-4; do
  ssh $vm "find /mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing/data -name iter_records.jsonl -exec wc -l {} \;"
done

# 3. OpenRouter spend so far:
python3 -c "
import urllib.request, json, os
k=os.environ['OPENROUTER_API_KEY']
r=urllib.request.Request('https://openrouter.ai/api/v1/auth/key', headers={'Authorization':f'Bearer {k}'})
print(json.dumps(json.load(urllib.request.urlopen(r,timeout=10))['data'], indent=2))
"
```
