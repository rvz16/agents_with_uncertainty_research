# ALFWorld trajectory-level uncertainty

Minimal text-only ALFWorld experiment for collecting ReAct trajectories and
evaluating uncertainty post hoc. It is isolated from the other experiments in
the parent repository and does not use SAGE clarification logic.

## Setup

Python 3.12 is verified in the current environment.

```bash
cd alfworld_uq
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/alfworld-download
cp .env.example .env
```

Set these values in `.env`:

```dotenv
LLM_BASE_URI=https://your-openai-compatible-endpoint/v1
LLM_API_KEY=...
MODEL_NAME=openai/gpt-oss-20b
```

ALFWorld defaults to `~/.cache/alfworld`. Override it with `--data-root` or
`ALFWORLD_DATA`.

## Collect trajectories

Run the 10-episode pilot:

```bash
.venv/bin/python -m experiments.run_alfworld \
  --num-episodes 10 \
  --max-steps 30 \
  --output-dir runs/alfworld_baseline_10
```

The equivalent checked-in config is:

```bash
.venv/bin/python -m experiments.run_alfworld \
  --config configs/default.json
```

Use `--policy random` only to smoke-test the environment and storage pipeline
without an API call. It is not an experimental baseline.

Each run contains:

- `trajectories.jsonl`: one complete record per agent step;
- `episodes.jsonl`: success, stop reason, length, tokens, and wall time;
- `run_config.json`: non-secret run configuration.

Every step stores UQ separately for `thought`, `action`, and `combined`. The
methods are perplexity, sum log-probability, mean token log-probability,
sequence probability, and optional parsed verbalized confidence. The legacy
top-level `perplexity` and `seqprob` fields mirror only `combined`.

The client requests chat token logprobs. If an OpenAI-compatible endpoint
rejects that parameter, collection continues without them and records
`logprobs_available=false`. In that case post-hoc logprob methods are
unavailable; point `LLM_BASE_URI` at a local endpoint such as vLLM that exposes
chat completion token logprobs.

For OpenRouter, the runner also sends `provider.require_parameters=true`.
Use `--provider-order <provider>` with `--no-allow-provider-fallbacks` after
probing providers when routing still yields intermittent missing logprobs. The
served provider is stored on every step.

`max_generation_tokens=1024` is intentional: lower limits can be consumed by
the model's hidden reasoning before it emits the visible ReAct response. If a
response is still empty, the default `--empty-response-retries 1` repeats that
step once with a doubled limit and includes both attempts in token accounting.

## Analyze without LLM calls

```bash
.venv/bin/python -m experiments.analyze_trajectories \
  --trajectories runs/alfworld_baseline_10/trajectories.jsonl \
  --output-dir runs/alfworld_baseline_10/analysis
```

Episodes are deterministically split into calibration and test sets. Thresholds,
Platt calibration, class-conditional UQ distributions, and the success prior
are fitted on calibration only.

## Add an LLM-as-a-judge critic

Judge the saved trajectories once; this does not rerun ALFWorld. The prompt is
built only from the task and visible Thought/Action/Observation transcript. It
does not include `final_success`, `done`, reward, progress, or stop reason.

```bash
.venv/bin/python -m experiments.judge_trajectories \
  --trajectories runs/alfworld_baseline_100/trajectories.jsonl \
  --output runs/alfworld_baseline_100/llm_judge_scores.jsonl \
  --model anthropic/claude-haiku-4.5 \
  --workers 8
```

The output is an append-only, resumable JSONL cache. Fuse its binary PASS/FAIL
verdict as another calibrated critic by passing it to the analysis:

```bash
.venv/bin/python -m experiments.analyze_trajectories \
  --trajectories runs/alfworld_baseline_100/trajectories.jsonl \
  --judge-scores runs/alfworld_baseline_100/llm_judge_scores.jsonl \
  --output-dir runs/alfworld_baseline_100/analysis_judge/seed_0 \
  --seed 0
```

The analysis compares the prior; a critic `bayes_state` fitted from ReAct format
validity, action admissibility, and repeated-action fallback; all trajectory
aggregations (`last`, `mean`, `min`, `max`, `median`, EWMA, uncertain fraction,
last-k mean, and CVaR); binary Bayes; continuous Bayes; tempered continuous
Bayes; and each UQ Bayes update fused on top of the critic state. It also reports
a `stepwise_bayes_state` that calibrates and applies the three proxy critics at
every generation, matching the sequential update mechanics of `uq_exps` (but
without a transition kernel, since ALFWorld has no intermediate correctness
label). It writes:

- `metrics.csv` with AUROC, AUPRC, PRR@0.5, Brier, NLL, and ECE;
- `prefix_metrics.csv` for 25%, 50%, 75%, and 100% prefixes;
- `risk_coverage.csv` and `model_parameters.csv`;
- `critic_likelihoods.csv` with calibration-only critic likelihoods;
- `bayes_states.csv` with critic observations and the resulting state per episode;
- belief JSONL and four PNG diagnostic plots;
- `report.md` with run coverage and endpoint limitations.

After validating the 10-episode run, start the 100-episode pilot in a new output
directory:

```bash
.venv/bin/python -m experiments.run_alfworld \
  --num-episodes 100 \
  --max-steps 30 \
  --output-dir runs/alfworld_baseline_100
```

For the same deterministic episode set with four parallel workers:

```bash
.venv/bin/python -m experiments.run_alfworld_sharded \
  --num-episodes 100 \
  --workers 4 \
  --max-steps 30 \
  --provider-order Novita \
  --no-allow-provider-fallbacks \
  --output-dir runs/alfworld_baseline_100
```

Workers receive disjoint offsets into the seeded split. Their JSONL files and
logs remain under `shards/`; merged `trajectories.jsonl` and `episodes.jsonl`
are written only after every shard succeeds.

If provider load causes episode-level `api_error`, rerun only those deterministic
offsets sequentially and replace them atomically:

```bash
.venv/bin/python -m experiments.repair_api_errors \
  --run-dir runs/alfworld_baseline_100 \
  --overwrite-repairs
```

The pre-repair merged files are retained as `*.pre_repair.jsonl`.
Repair uses the exact saved `gamefile`, so it is independent of TextWorld's
internal reset ordering inside a shard.

## Tests

```bash
.venv/bin/pytest
```
