# Trajectory UQ Toolkit

Standalone, environment-independent collection and analysis code for
trajectory-level uncertainty experiments. Copy this directory into another
repository or create a source-only archive with `scripts/make_bundle.py`.

The toolkit intentionally contains no research runs, model responses, raw
token distributions, benchmark datasets, or credentials.

The optional [`code_uq`](docs/code_uq.md) package contains the source-only
LCB/MBPP/HumanEval/CodeContests/SWE runner extracted from `uq_exps`. New
environments should use the generic adapter API below.

[`alfworld_uq`](alfworld_uq/README.md) is the concrete ALFWorld reference
environment from `new_env_test`. It has its own environment dependencies and
runner, but can export compact rows for the common analyzer.

## Source branches

- `uq_exps` became the installable `code_uq` package and its compatibility
  analysis modules.
- `new_env_test` contributed the ALFWorld reference environment and the
  Continuous/Tempered Bayes methods now exposed by the common analyzer.

See [`docs/migration_manifest.md`](docs/migration_manifest.md) for the exact
source-only transfer boundary.

## What is included

Collection:

- a compact versioned JSONL schema;
- a resumable, parallel runner for pluggable environment adapters;
- online reduction of token log-probabilities into small numeric features;
- separate `errors.jsonl` and `run_config.json` files.

Analysis:

- trajectory aggregations: last, first, mean, min, max, median, std, range,
  EWMA, last-k mean, and CVaR;
- sequence probability, mean/sum log-probability, perplexity, token entropy,
  KL to uniform, self-certainty, token count, verbalized confidence, and
  arbitrary adapter-defined scalar signals;
- Platt-calibrated feature baselines;
- binary Bayes with `quantile`, `sep`, `lr_pos`, `lr_neg`, and `double`;
- continuous Gaussian Bayes and tempered continuous Bayes;
- episode-level and stepwise binary critic Bayes states;
- fusion of each UQ Bayes model with each requested critic state;
- AUROC, AUPRC, PRR@0.5, Brier, NLL, and ECE.

An LLM judge needs no special analyzer code. Store its binary verdict as the
episode critic `llm_judge_pass`; it is calibrated and fused like every other
critic.

## Install

```bash
cd trajectory_uq_toolkit
python -m venv .venv
.venv/bin/pip install -e '.[test]'
```

The core package depends only on NumPy. Environment-specific dependencies
belong in the destination repository, not in this toolkit.

## Collection API

An adapter is a Python module with two functions:

```python
def list_episodes(config: dict) -> list[str]: ...

def run_episode(episode_id: str, seed: int, config: dict) -> dict:
    return {
        "episode_id": episode_id,
        "environment": "my_environment",
        "success": 0,  # terminal ground-truth label
        "generations": [
            {
                "index": 0,
                "signals": {
                    "sum_logprob": -12.3,
                    "perplexity": 1.4,
                    "num_tokens": 42,
                },
                "critics": {
                    "format_valid": True,
                    "action_valid": True,
                },
            }
        ],
        "critics": {
            "all_formats_valid": True,
            "all_actions_valid": True,
            "llm_judge_pass": False,
        },
        "features": {"tool_success_rate": 0.75},
        "metadata": {"task_family": "navigation"},
    }
```

Use `summarize_token_logprobs(chosen_logprobs, top_logprobs)` inside the
adapter. It computes all standard logprob/distribution signals immediately;
the large token arrays can then be discarded.

Run the included end-to-end example:

```bash
trajectory-uq-collect \
  --adapter examples/toy_adapter.py \
  --config examples/toy_config.json \
  --output-dir runs/toy \
  --workers 4
```

The runner is append-only and resumes by `episode_id`.

## Analysis

```bash
trajectory-uq-analyze \
  --episodes runs/toy/episodes.jsonl \
  --output-dir runs/toy/analysis \
  --seed 0 \
  --tempered-lambda 0.25
```

Unknown signals are assumed to be confidence scores where larger is better.
Override that for uncertainty scores where larger is worse:

```bash
trajectory-uq-analyze ... \
  --signal-direction custom_uncertainty=uncertain
```

By default the analyzer evaluates every individual episode critic, all common
episode critics together, all critics without `llm_judge_pass`, and all common
stepwise critics. Define explicit groups when experiments need fixed ablations:

```bash
trajectory-uq-analyze ... \
  --critic-set mechanical=all_formats_valid,all_actions_valid,no_repeated_fallback \
  --critic-set with_judge=all_formats_valid,all_actions_valid,no_repeated_fallback,llm_judge_pass
```

Outputs are deliberately small: `metrics.csv`, `model_parameters.json`, and
`summary.md`.

## Compact an existing ALFWorld run

The source repository already has richer ALFWorld JSONL files. Convert them
without copying prompts, responses, observations, admissible actions, usage,
or token-level logprobs:

```bash
PYTHONPATH=src python scripts/compact_alfworld_export.py \
  --trajectories ../alfworld_uq/runs/alfworld_baseline_100/trajectories.jsonl \
  --episodes ../alfworld_uq/runs/alfworld_baseline_100/episodes.jsonl \
  --judge-scores ../alfworld_uq/runs/alfworld_baseline_100/llm_judge_scores.jsonl \
  --output /tmp/alfworld_uq_compact.jsonl
```

For a small approximately class-balanced reproducibility sample, add
`--limit 20 --seed 0`. The compact export is analysis-ready but cannot replay
the original agent interaction; use the environment adapter for new live runs.

## Moving to another repository

Preferred options:

1. Copy this directory and commit it as a normal package.
2. Keep it as a git subtree/submodule if both repositories should receive
   future fixes.
3. Create a source-only archive:

```bash
python scripts/make_bundle.py --output /tmp/trajectory_uq_toolkit.tar.gz
```

Do not transfer `runs/`, raw completions, raw `top_logprobs`, caches, virtual
environments, or API credentials. If colleagues need a reproducibility sample,
generate a small toy run or export a stratified, anonymized set of compact
episode rows rather than the original multi-megabyte logs.
