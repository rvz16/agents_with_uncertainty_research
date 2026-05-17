# wandb — Agentic AI Orchestration as Sequential Hypothesis Testing

Project: [nlpresearch.group / orchestration-hypothesis-testing](https://wandb.ai/nlpresearch.group/orchestration-hypothesis-testing)

## Files

| File | Purpose |
|---|---|
| `SCHEMA.md` | Run convention — tags, config fields, summary fields, artifact types. Read first. |
| `upload_runs.py` | Walks local data, creates wandb runs. Idempotent (skips runs that exist). |
| `analysis.ipynb` | Fetches every run via `wandb.Api()`, builds a DataFrame, reproduces every deck figure. Cached. |

## Quick start

```bash
# 0. (one-time) authenticate
wandb login

# 1. Sync data from cluster (if not already done)
bash ../scripts/sync_from_cluster.sh

# 2. Dry-run the upload so you see what would happen
python upload_runs.py --dry-run

# 3. Upload everything (idempotent — safe to re-run)
python upload_runs.py

# 4. Open the notebook and run all cells
jupyter notebook analysis.ipynb
```

## Useful upload subsets

```bash
# Just one track
python upload_runs.py --track orchestration
python upload_runs.py --track abbo

# Just one experiment type
python upload_runs.py --experiment calibration
python upload_runs.py --experiment iter
python upload_runs.py --experiment policy_comparison

# Debug a single cell
python upload_runs.py --benchmark lcb_hard --generator gpt5_mini

# Force re-upload (replaces existing runs)
python upload_runs.py --force
```

## Two tracks

- **`track:orchestration`** — main pipeline (`experiments/orchestration_hypothesis_testing/data/`); 5 closed-API generators across 7 benchmarks.
- **`track:abbo`** — colleague's `bayesian_optimization_for_code_testing/agent-bugfix-bayes/` (gpt-oss-20b on HumanEvalFix, CodeContests, SWE-Lite).

Tags `track:*` and `experiment:*` are filterable in the wandb UI and used by `analysis.ipynb` to slice the runs DataFrame.
