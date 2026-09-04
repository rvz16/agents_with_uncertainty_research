# Running ALFWorld on the GPU cluster via ClearML

Same runner and same flags as a local run; this only adds "get it onto a GPU
agent and serve the model locally". Serving locally is the point: hosted APIs
return clamped or missing token log-probabilities, and every logprob-based UQ
signal in the analysis depends on them.

## Enqueue

```bash
python3 -m venv /tmp/clearmlvenv && /tmp/clearmlvenv/bin/pip install -q clearml
cd alfworld_uq/clearml

# end-to-end check first (2 episodes, ~15 min once the model is loaded)
/tmp/clearmlvenv/bin/python create_task.py --smoke

# gpt-oss first, Qwen after it works
/tmp/clearmlvenv/bin/python create_task.py --num-episodes 100
/tmp/clearmlvenv/bin/python create_task.py \
    --model Qwen/Qwen3.6-35B-A3B --tensor-parallel-size 2 --num-episodes 100
```

Credentials come from `~/clearml.conf`. The agent clones the repo itself, so
**push the branch before enqueuing** — it runs what is on the remote, not what
is in the working tree. `--branch` must name that branch.

## Budgets

`--max-steps` is the environment action budget (30, as in the ReAct baseline).
`--agent-max-steps` is the separate generation budget for the smolagents loop
and defaults to 45: the framework spends generations on steps that take no
environment action, and with both budgets equal the episode ends on
`agent_stopped` before the environment budget is reached, which is not
comparable to the ReAct run's `max_steps`.

## Queues

| queue | workers | state |
|---|---|---|
| `high_q_2xA100_80` | aiagent02:gpu1,2 | works — the default here |
| `high_q_80` | aiagent01:gpu0/gpu1, aiagent02:gpu0 | aiagent01:gpu0 pulls tasks and then fails |
| `high_q`, `sience` | aiagent03:gpu0/gpu1 | unusable — the agent strips `--entrypoint=` |

Moving a queued task: `Task.dequeue(t)` then `Task.enqueue(t, queue_name=...)`.
A *failed* task needs `t.reset(force=True)` first.

## What the container does

`entry.py` exports the task parameters as environment variables and hands over
to `run_in_container.sh`, which:

1. installs `alfworld`, `textworld[pddl]`, `smolagents`, `openai` — **not**
   `requirements.txt`, whose numpy/matplotlib pins would fight the image's
   torch build (the analysis runs locally anyway);
2. runs `alfworld-download` and *verifies the game files*. The dataset arrives
   as two archives and a dropped connection still exits 0, leaving a tree the
   environment reports as `0 supported games`; the download is retried up to
   three times and the task fails fast if `game.tw-pddl` is still missing;
3. serves the model with the image's vLLM on `127.0.0.1:8010` and waits for
   `/health`;
4. runs `experiments/run_alfworld.py` (or `run_alfworld_sharded.py` when
   `--workers > 1`) against that endpoint.

## Retrieving results

The task uploads the run directory as the `run_root` artifact to the ClearML
**File Store** (`https://files.clearai.innopolis.university`), not the s3
bucket — that bucket is full and returns `XMinioStorageFull` at the very end of
a run, after the GPU time is already spent. Download it from the ARTIFACTS tab;
it holds `trajectories.jsonl`, `episodes.jsonl` and `run_config.json`.

All post-hoc analysis then runs locally on CPU:

```bash
cd alfworld_uq
.venv/bin/python -m experiments.analyze_trajectories \
  --trajectories <downloaded>/trajectories.jsonl \
  --output-dir <downloaded>/analysis
```
