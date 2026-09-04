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

`alfworld-download` fetches two archives; the second one carries
`initial_state.pddl` and the pre-generated `game.tw-pddl` files. A connection
that drops mid-archive still exits leaving only `traj_data.json`, and the
environment then reports `0 supported games`. Check before running anything:

```bash
find ~/.cache/alfworld/json_2.1.1 -name game.tw-pddl | wc -l   # expect ~4000
```

Re-run `alfworld-download` if it is zero. Generating those files locally with
`alfworld-generate` also needs `~/.cache/alfworld/logic/alfred.{pddl,twl2}`,
which ships inside the installed package (`alfworld/data/`).

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

## smolagents policy

`--policy smolagents` swaps the runner-driven ReAct loop for a smolagents
`CodeAgent` that owns the episode: the model writes Python that calls
`take_action("<admissible action>")`, so one generation can take zero, one, or
several environment steps. The environment, the admissible-action resolution,
the repeated-action fallback and the trajectory schema are shared with the
ReAct policy, so both runs feed the same analysis.

```bash
.venv/bin/python -m experiments.run_alfworld \
  --policy smolagents \
  --num-episodes 10 \
  --max-steps 30 \
  --max-generation-tokens 2048 \
  --provider-order Novita --no-allow-provider-fallbacks \
  --output-dir runs/smol_baseline_10
```

`--max-steps` stays the environment budget. `--agent-max-steps` is the separate
generation budget for the framework loop (default: the same number). The
episode ends the moment the environment reports `done` or the step budget runs
out, so a solved episode never pays for extra generations.

Differences a reader of the trajectories should know:

- **Rows are per generation, not per environment step.** `env_actions` lists
  what that generation actually did; `env_action_count` is 0 for a generation
  that only reasoned, printed, or called `final_answer`.
- **UQ segments follow the code-action format.** `thought` is the text before
  the code block and `action` is the code itself; `combined` is the whole
  response, as before. When the close tag is a stop sequence the response ends
  without it, so end-of-text closes the block — the same thing smolagents' own
  parser does when it re-appends the missing tag.
- **`format_valid`** means the generation produced a code block (or a bare code
  blob that parses, which smolagents also accepts). **`action_valid`** means
  every environment action in the generation was admissible; a lone
  `final_answer` counts as a deliberate stop, anything else with no environment
  action is recorded as `no_env_action`.
- **Log-probabilities are not free here.** `OpenAIServerModel` drops them, so
  the model is subclassed to send `logprobs=True` and keep each raw response.
  It also repeats a generation with a doubled token limit when the endpoint
  returns empty content (`--empty-response-retries`).

Two settings exist because of what gpt-oss-20b does with the stock smolagents
configuration; both are worth re-checking for a new model:

- **Stop sequences.** smolagents stops generation on `["Observation:",
  "Calling tools:", <code close tag>]`. The first two are words a reasoning
  model uses while it thinks, and a hit inside hidden reasoning ends the turn
  with *empty content* — in a 2-episode pilot that silently wasted a quarter of
  all generations (empty response, no log-probabilities, counted as a malformed
  step). Only the framework's own code-tag rule is kept. Pass
  `--no-smol-stop-sequences` to drop stops entirely; that lets a model
  hallucinate a second code block, which the framework's parser concatenates
  and executes.
- **Action format.** `--smol-code-tags markdown` (default) uses ```` ```python ````
  fences instead of the framework's `<code>` tags: gpt-oss-20b frequently
  answered with a bare `Thought:` line and no `<code>` block at all. Use
  `--smol-code-tags xml` for the stock format.

`stop_reason` gains two smolagents-only values: `agent_stopped` (the framework
called `final_answer` or ran out of generations) and `agent_error` (the
framework raised after at least one generation).

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
