# Running on the GPU cluster via ClearML

Same pipeline and same config as a local run — this only adds "get it onto a GPU
agent". Everything about *what* is run stays in
[`../../JOINT_RUN_CONFIG.md`](../../JOINT_RUN_CONFIG.md) and
`run_sage_uncertainty_experiments.sh`.

## Enqueue

```bash
python3 -m venv /tmp/clearmlvenv && /tmp/clearmlvenv/bin/pip install -q clearml
export OPENROUTER_API_KEY=...            # omit to skip the L3 critic

/tmp/clearmlvenv/bin/python create_task.py --benchmarks lcb_hard --smoke   # 6 instances
/tmp/clearmlvenv/bin/python create_task.py --benchmarks codecontests       # full run
```

Credentials come from `~/clearml.conf`. The agent clones the repo itself, so
**push the branch before enqueuing** — it runs what is on the remote, not what is
in your working tree.

## Queues

| queue | workers | state |
|---|---|---|
| `high_q_2xA100_80` | aiagent02:gpu1,2 | works — the default here |
| `high_q_80` | aiagent01:gpu0/gpu1, aiagent02:gpu0 | aiagent01:gpu0 pulls tasks and then fails |
| `high_q`, `sience` | aiagent03:gpu0/gpu1 | unusable, see below |

`aiagent01:gpu0` accepts the task, docker gets `--gpus "device=0"`, `nvidia-smi`
prints a card in the setup script — and vLLM then dies with `RuntimeError: No
CUDA GPUs are available`. Failed twice in a row there.

`aiagent03` runs an agent version that prints `ignoring docker argument(s)
--entrypoint` and strips it. The vLLM image's entrypoint is `vllm` itself, so the
agent's startup script lands in `vllm serve`'s argv and the task dies inside a
minute with `error: argument --compilation-config`. Using aiagent03 needs a base
image whose entrypoint is not `vllm`.

Moving a queued task: `Task.dequeue(t)` then `Task.enqueue(t, queue_name=...)`.
A *failed* task needs `t.reset(force=True)` first.

## Retrieving results

The task uploads `run_root` as an artifact to the ClearML **File Store**
(`https://files.clearai.innopolis.university`), not the s3 bucket — that bucket is
full and returns `XMinioStorageFull` at the very end of a run, after the GPU time
is already spent. Download it from the ARTIFACTS tab; it holds the trajectories,
logprob sidecars, verbalized outputs and the readable CSVs.

All post-hoc analysis then runs locally on CPU:

```bash
cd ../uq_analysis
python3 prr_table.py
python3 nested_compare.py --run-root <downloaded_run_root> --benchmark codecontests
```

## Timing

About 3–9 min per instance for gpt-oss-20b at 32k generation tokens, single
worker: lcb_hard (76 test instances) ≈ 16 h, codecontests (124) ≈ 16 h,
lcb_medium (155) ≈ 20 h. Prior calibration on the train split adds ~30–60 min.
Watch the console in the ClearML UI; `[test N/M]` lines carry the progress.
