# Calibration fill — VM status (live)

**Snapshot**: 2026-05-24 23:36 MSK (≈ 2026-05-25 00:36 +04 on the VMs)
**Branch**: `main` (HEAD `b006f825` — `--extend-existing` indent fix merged)
**OpenRouter key**: `sk-or-v1-55161e1c...061c470c` (sha256 prefix `95341720`) — $250 budget, fresh

## What's running where

| VM | Host | Benchmark | N_target | Gens (sequential) | tmux session | Status |
|---|---|---|---|---|---|---|
| **mbz3** | 10.127.105.14 (MBZUAI-Artem-3) | **MBPP+** (N=378) | 378 | gpt5_mini → qwen3_coder → haiku45 → sonnet45 | `cal_mbpp` | running |
| **mbz4** | 10.127.105.20 (MBZUAI-Artem-4) | **HumanEval+** (N=164) | 164 | gpt5_mini → qwen3_coder → haiku45 → sonnet45 | `cal_humaneval` | running |

SSH from local Mac: `ssh -i ~/.ssh/id_ed25519 vlad.smirnov@<host>` (both reachable; user `vlad.smirnov`).

## Snapshot of progress (last check at ~00:35 +04)

### mbz3 (MBPP+)
| Generator | patch records | unique instances | raw_responses | state |
|---|---|---|---|---|
| gpt5_mini | 335 | 112 / 378 | 35 | **RUNNING** (+12 new since baseline of 100) |
| qwen3_coder | 300 | 100 / 378 | 0 | queued |
| haiku45 | 300 | 100 / 378 | 0 | queued |
| sonnet45 | 300 | 100 / 378 | 0 | queued |

### mbz4 (HumanEval+)
| Generator | patch records | unique instances | raw_responses | state |
|---|---|---|---|---|
| gpt5_mini | 312 | 104 / 164 | 12 | **RUNNING** (+4 new since baseline of 100) |
| qwen3_coder | 300 | 100 / 164 | 0 | queued |
| haiku45 | 300 | 100 / 164 | 0 | queued |
| sonnet45 | 300 | 100 / 164 | 0 | queued |

## ETAs (rough, based on observed throughput ~4 inst/min for mbpp, ~2 inst/min for humaneval)

| VM | Remaining work | Per-gen ETA | Total ETA |
|---|---|---|---|
| mbz3 | 278 new × 4 gens × 3 patches ≈ 3300 patches | ~70 min/gen | **~5 h** wall (sequential) |
| mbz4 | 64 new × 4 gens × 3 patches ≈ 770 patches | ~30 min/gen | **~2 h** wall (sequential) |

## Paths & environment

### mbz3
- **Repo**: `/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research` (on `main`, HEAD `b006f825`)
- **venv (working)**: `/mnt/data/users/vlad.smirnov/cal_venv/bin/python` — has wandb v0.27.0, datasets, evalplus, openai, dotenv, tqdm
- **Logs**: `~/cal_logs/mbpp_<gen>.log` (one per gen)
- **Output**: `<repo>/experiments/orchestration_hypothesis_testing/data/mbpp_calibration/{<gen>/critic_results.jsonl, <gen>/raw_responses/, sample.json}`
- **No special env vars needed** — `/` has 516 GB free.

### mbz4 (everything off `/mnt` — root partition is 100% full)
- **Repo**: same path
- **Python**: `/usr/bin/python3` (system Python 3.10) — DO NOT use `python3` directly in tmux because the default shell has conda `(base)` active with a broken `pydantic_core`. Always invoke `/usr/bin/python3` explicitly.
- **Packages**: `pip install --target=/mnt/data/users/vlad.smirnov/cal_packages` — required env: `PYTHONPATH=/mnt/data/users/vlad.smirnov/cal_packages`
- **Required env vars for any run** (because `~` and `/tmp` are on the full `/`):
  ```bash
  export PYTHONPATH=/mnt/data/users/vlad.smirnov/cal_packages
  export TMPDIR=/mnt/data/users/vlad.smirnov/tmp
  export WANDB_DIR=/mnt/data/users/vlad.smirnov/wandb_cache
  export WANDB_CACHE_DIR=/mnt/data/users/vlad.smirnov/wandb_cache
  export WANDB_ARTIFACT_DIR=/mnt/data/users/vlad.smirnov/wandb_cache/artifacts
  export XDG_CACHE_HOME=/mnt/data/users/vlad.smirnov/cache
  export HF_HOME=/mnt/data/users/vlad.smirnov/cache/huggingface
  ```
- **Logs**: `/mnt/data/users/vlad.smirnov/cal_logs/humaneval_<gen>.log`
- **Output**: `<repo>/experiments/orchestration_hypothesis_testing/data/humaneval_calibration/{<gen>/critic_results.jsonl, <gen>/raw_responses/, sample.json}`
- **Always prefix tmux command with** `conda deactivate 2>/dev/null; conda deactivate 2>/dev/null; ` to escape the auto-activated conda env.

## How to monitor

```bash
# Tail the live log
ssh vlad.smirnov@10.127.105.14 'tail -f ~/cal_logs/mbpp_$(ls -t ~/cal_logs/mbpp_*.log | head -1 | xargs basename)'
ssh vlad.smirnov@10.127.105.20 'tail -f /mnt/data/users/vlad.smirnov/cal_logs/humaneval_$(ls -t /mnt/data/users/vlad.smirnov/cal_logs/humaneval_*.log | head -1 | xargs basename)'

# Attach to tmux session
ssh -t vlad.smirnov@10.127.105.14 'tmux attach -t cal_mbpp'
ssh -t vlad.smirnov@10.127.105.20 'tmux attach -t cal_humaneval'

# Snapshot of progress (unique instances per gen)
ssh vlad.smirnov@10.127.105.14 'cd /mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing && for g in gpt5_mini qwen3_coder haiku45 sonnet45; do f=data/mbpp_calibration/$g/critic_results.jsonl; [ -f $f ] && /usr/bin/python3 -c "import json; ids=set(); [ids.add(json.loads(l)[\"instance_id\"]) for l in open(\"$f\") if l.strip()]; print(\"$g:\", len(ids), \"unique inst\")"; done'
```

## How to resume after a crash

The calibration scripts have **two layers of resume**:
1. **`--extend-existing`** (added to `mbpp.py` / `humaneval.py` / `codecontests.py` / `lcb.py`) — keeps the existing instance set as the head of the sampled list while bumping `--n-instances`. Saves `sample.json` at output-dir level for use across re-runs.
2. **Per-record dedup** in `calibrate_one_generator` — reads existing `critic_results.jsonl` and skips done `(instance, patch)` pairs at runtime.

So if a tmux dies, just re-launch the same command and it'll pick up where it left off. The OpenRouter `--max-cost-usd-per-model gpt5_mini=X.X` cap is per-run, not cumulative.

## Re-launch commands (copy-paste reference)

### mbz3 (mbpp)
```bash
ssh -t vlad.smirnov@10.127.105.14
cd /mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing
set -a; source ../../.env; set +a
tmux new -s cal_mbpp
for GEN in gpt5_mini qwen3_coder haiku45 sonnet45; do
  echo "=== Starting $GEN at $(date) ===" | tee -a ~/cal_logs/mbpp_$GEN.log
  /mnt/data/users/vlad.smirnov/cal_venv/bin/python -m calibration.mbpp \
      --output-dir data/mbpp_calibration \
      --generators $GEN --n-instances 378 --n-patches 3 --seed 42 \
      --extend-existing --max-cost-usd-per-model ${GEN}=5.0 \
      2>&1 | tee -a ~/cal_logs/mbpp_$GEN.log
  echo "=== Finished $GEN at $(date) ===" | tee -a ~/cal_logs/mbpp_$GEN.log
done
echo ALL_GENS_DONE
```

### mbz4 (humaneval)
```bash
ssh -t vlad.smirnov@10.127.105.20
conda deactivate 2>/dev/null; conda deactivate 2>/dev/null
cd /mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing
export PYTHONPATH=/mnt/data/users/vlad.smirnov/cal_packages \
       TMPDIR=/mnt/data/users/vlad.smirnov/tmp \
       WANDB_DIR=/mnt/data/users/vlad.smirnov/wandb_cache \
       WANDB_CACHE_DIR=/mnt/data/users/vlad.smirnov/wandb_cache \
       XDG_CACHE_HOME=/mnt/data/users/vlad.smirnov/cache \
       HF_HOME=/mnt/data/users/vlad.smirnov/cache/huggingface
set -a; source ../../.env; set +a
tmux new -s cal_humaneval
for GEN in gpt5_mini qwen3_coder haiku45 sonnet45; do
  echo "=== Starting $GEN at $(date) ===" | tee -a /mnt/data/users/vlad.smirnov/cal_logs/humaneval_$GEN.log
  /usr/bin/python3 -m calibration.humaneval \
      --output-dir data/humaneval_calibration \
      --generators $GEN --n-instances 164 --n-patches 3 --seed 42 \
      --extend-existing --max-cost-usd-per-model ${GEN}=3.0 \
      2>&1 | tee -a /mnt/data/users/vlad.smirnov/cal_logs/humaneval_$GEN.log
  echo "=== Finished $GEN at $(date) ===" | tee -a /mnt/data/users/vlad.smirnov/cal_logs/humaneval_$GEN.log
done
echo ALL_GENS_DONE
```

## When all gens finish on a VM — upload to W&B

From the **local Mac** (W&B `upload_runs.py` is idempotent):

```bash
cd "/Users/karantonis/CLAUDE COWORK/PROJECTS/MBZUAI/agents_with_uncertainty_research"

# Sync the new critic_results.jsonl + raw_responses back from VMs
rsync -av --include='*/' --include='critic_results.jsonl' --include='likelihood_tables.json' --include='raw_responses/***' --include='sample.json' --exclude='*' \
  vlad.smirnov@10.127.105.14:/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing/data/mbpp_calibration/ \
  experiments/orchestration_hypothesis_testing/data/mbpp_calibration/

rsync -av --include='*/' --include='critic_results.jsonl' --include='likelihood_tables.json' --include='raw_responses/***' --include='sample.json' --exclude='*' \
  vlad.smirnov@10.127.105.20:/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing/data/humaneval_calibration/ \
  experiments/orchestration_hypothesis_testing/data/humaneval_calibration/

# Upload (force-overwrites the existing 100-instance runs with the new 378/164-instance ones)
cd experiments/orchestration/wandb
python upload_runs.py --track orchestration --benchmark mbpp     --force
python upload_runs.py --track orchestration --benchmark humaneval --force
```

Then re-run cell 28 in `analysis.ipynb` to confirm the coverage matrix flips to FULL (378/378 and 164/164) for all 4 API generators on these benchmarks.

## Known issues & gotchas (for future runs)

1. **mbz4 root partition is 100% full.** Every tool that writes to `~`, `/tmp`, or any path under `/` will fail. Always set the cache env vars to `/mnt/data/users/vlad.smirnov/...`. The fix list above is the minimum set.
2. **mbz4 default shell auto-activates conda `(base)`** which has its own broken `pydantic_core`. Always `conda deactivate` twice (it stacks) and use `/usr/bin/python3` explicitly. `python3` alone resolves to conda's python.
3. **OpenRouter key has a per-key spending cap** (`limit: 250` USD). Monitor via:
   ```bash
   curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/auth/key | python3 -m json.tool
   ```
   Check `limit_remaining` periodically. If it hits 0, calibration silently fails the new generation calls (403). Resume = same `--extend-existing` command after topping up.
4. **mbz3 has wandb-friendly `cal_venv`**; mbz4 has `cal_packages/` via `pip install --target`. Don't mix.
5. **Per-VM cell numbering of `analysis.ipynb` is irrelevant for the VMs** — they only run the calibration scripts (Python modules), not the notebook.

## Remaining cells not in this batch (for later)

Still missing from W&B (see `analysis.ipynb` cell 28 coverage matrix):
- SWE-Bench Lite / Verified — every cell at 5-16% (cluster-side Docker-eval job, not done here)
- CodeContests — 4 API cells at 63-68% (could be batched similarly to mbpp/humaneval on either VM after this finishes)
- HumanEvalFix — `gpt5_mini` missing entirely (could be added to mbz3 after MBPP+ wraps)
- gpt_oss_20b on LCB / SWE — Artyom (artiomvazh99) handles via his vLLM cluster
- `qwen3_coder` on HumanEval+ — the existing iter sr/rfx is 17 / 33 pairs (anomalously small); a re-run is needed but not in this batch

---

*This file is a snapshot. Re-run the snapshot commands above to refresh, or update this file when a campaign completes.*
