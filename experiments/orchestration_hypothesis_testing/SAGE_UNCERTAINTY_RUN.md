# SAGE Uncertainty Run

Minimal note on how we ran the uncertainty experiments with the SAGE agent.

## Environment

From the repository root:

```bash
python -m pip install -e '.[langgraph,openrouter]'
python -m pip install -r experiments/orchestration_hypothesis_testing/scripts/requirements-sage-uncertainty.txt
```

Required runtime services:

- an OpenAI-compatible local endpoint for the target generator;
- `OPENROUTER_API_KEY` for the L3 LLM critic;
- Docker or Podman Docker-API compatibility for SWE-Bench.

No API keys are stored in the launch scripts.

## Reproducible Launch Scripts

Tracked scripts:

- `experiments/orchestration_hypothesis_testing/scripts/run_sage_uncertainty_experiments.sh`
  runs the raw agent plus analysis for a comma-separated benchmark list.
- `experiments/orchestration_hypothesis_testing/scripts/submit_sage_uncertainty_slurm.sh`
  submits the same runner to Slurm. It assumes the model endpoint is reachable
  from the allocated node; it does not hard-code or start a specific private
  vLLM deployment.
- `experiments/orchestration_hypothesis_testing/scripts/aggregate_uncertainty_metric_tables.py`
  builds the final per-model markdown/CSV tables from `metric_scores.csv`.

Example, GPT-OSS-20B on non-SWE benchmarks:

```bash
export OPENROUTER_API_KEY=...
export GENERATOR_KEY=gpt_oss_20b_local
export GPT_OSS_20B_BASE_URL=http://127.0.0.1:8000/v1
export BENCHMARKS=lcb_hard,lcb_medium,lcb_easy,mbpp,humaneval,humanevalfix,codecontests
export RUN_ROOT=/path/to/results/sage_uncertainty_gpt_oss_20b

bash experiments/orchestration_hypothesis_testing/scripts/run_sage_uncertainty_experiments.sh
```

Example Slurm submission:

```bash
export OPENROUTER_API_KEY=...
export GENERATOR_KEY=qwen25_32b
export QWEN25_32B_BASE_URL=http://127.0.0.1:8000/v1
export BENCHMARKS=lcb_hard,lcb_medium,lcb_easy,mbpp,humaneval,humanevalfix,codecontests
export RUN_ROOT=/path/to/results/sage_uncertainty_qwen25_32b

bash experiments/orchestration_hypothesis_testing/scripts/submit_sage_uncertainty_slurm.sh
```

SWE-Bench example:

```bash
export OPENROUTER_API_KEY=...
export GENERATOR_KEY=gpt_oss_20b_local
export GPT_OSS_20B_BASE_URL=http://127.0.0.1:8000/v1
export BENCHMARKS=swebench_lite,swebench_verified
export N_INSTANCES=0
export N_TRAIN=
export TRAIN_FRACTION=0.25
export SWE_HARNESS_WORKERS=1
export RUN_ROOT=/path/to/results/sage_uncertainty_swe_gpt_oss_20b

bash experiments/orchestration_hypothesis_testing/scripts/run_sage_uncertainty_experiments.sh
```

For a small Verified smoke run with 25 train and 75 test examples:

```bash
export BENCHMARKS=swebench_verified
export N_INSTANCES=100
export N_TRAIN=25
```

For exactly 100 held-out test examples, use:

```bash
export BENCHMARKS=swebench_verified
export N_INSTANCES=125
export N_TRAIN=25
```

## Run

Assume an OpenAI-compatible vLLM endpoint is already running locally:

```bash
export GPT_OSS_20B_BASE_URL=http://127.0.0.1:8000/v1
mkdir -p sim_results/sage_uq

python different_agents/v4/lcb_llm_tool_agent.py \
  --benchmark lcb_hard \
  --generator gpt_oss_20b_local \
  --n-instances 0 \
  --train-fraction 0.25 \
  --prior-patches 1 \
  --private-test-cap 0 \
  --max-tokens-decision 4096 \
  --max-verifications 0 \
  --max-steps 20 \
  --max-generations 5 \
  --agent-backend sage \
  --final-verify \
  --save-generation-logprobs \
  --require-generation-logprobs \
  --save-verbalized-2s \
  --verbalized-2s-max-tokens 1024 \
  --output sim_results/sage_uq/lcb_hard__gpt_oss_20b_local.jsonl \
  --logprobs-output sim_results/sage_uq/lcb_hard__gpt_oss_20b_local.generation_logprobs.jsonl \
  --verbalized-2s-output sim_results/sage_uq/lcb_hard__gpt_oss_20b_local.verbalized_2s.jsonl \
  --split-output sim_results/sage_uq/lcb_hard__gpt_oss_20b_local.split.json \
  --prior-calibration-output sim_results/sage_uq/lcb_hard__gpt_oss_20b_local.train_prior_calibration.jsonl \
  --prior-calibration-logprobs-output sim_results/sage_uq/lcb_hard__gpt_oss_20b_local.train_prior_calibration.generation_logprobs.jsonl \
  --actions-output sim_results/sage_uq/lcb_hard__gpt_oss_20b_local.actions.jsonl
```

Repeat with `--benchmark lcb_medium` for LCB-Medium.

Main files:

- `lcb_hard__gpt_oss_20b_local.jsonl`: final agent results.
- `lcb_hard__gpt_oss_20b_local.actions.jsonl`: all agent actions.
- `lcb_hard__gpt_oss_20b_local.generation_logprobs.jsonl`: token logprobs.
- `lcb_hard__gpt_oss_20b_local.verbalized_2s.jsonl`: final-candidate
  Verbalized2S confidence.
- `lcb_hard__gpt_oss_20b_local.train_prior_calibration.jsonl`: train split used
  to estimate the prior.

## Analysis

First compute per-instance tool success, excluding the terminal final verifier:

```bash
python experiments/orchestration_hypothesis_testing/scripts/summarize_tool_action_success.py \
  sim_results/sage_uq/lcb_hard__gpt_oss_20b_local.jsonl \
  --per-instance-csv sim_results/sage_uq/tool_success_by_instance.csv
```

Then convert raw logs into uncertainty tables:

```bash
python experiments/orchestration_hypothesis_testing/scripts/analyze_lcb_llm_tool_agent_logs.py \
  --run-root sim_results/sage_uq \
  --benchmark lcb_hard \
  --generator gpt_oss_20b_local \
  --tool-success-csv sim_results/sage_uq/tool_success_by_instance.csv \
  --output-dir sim_results/sage_uq/readable
```

This produces:

- `final_logprob_bayes_quality.csv`: final quality, logprob scores, Bayes state,
  tool success, and Verbalized2S confidence.
- `generation_trajectory_scores.csv`: per-generation Bayes/logprob trajectory.
- `final_logprob_bayes_quality.jsonl`: aligned metadata.
- `metric_scores.csv`: notebook-compatible PRR/PRR_05 scores. The PRR
  calculation matches `lm_polygraph.ue_metrics.PredictionRejectionArea`
  followed by `normalize_metric`; it is not a trapezoidal approximation.

## Final Tables

After each benchmark has produced `metric_scores.csv`, build the per-model
tables:

```bash
python experiments/orchestration_hypothesis_testing/scripts/aggregate_uncertainty_metric_tables.py --metric PRR_05
```

For newly reproduced runs, pass their run roots explicitly:

```bash
python experiments/orchestration_hypothesis_testing/scripts/aggregate_uncertainty_metric_tables.py \
  --metric PRR_05 \
  --gpt-root /path/to/results/sage_uncertainty_gpt_oss_20b \
  --qwen-root /path/to/results/sage_uncertainty_qwen25_32b
```

Outputs:

- `experiments/orchestration_hypothesis_testing/sim_results/uncertainty_table__gpt_oss_20b__PRR_05.md`
- `experiments/orchestration_hypothesis_testing/sim_results/uncertainty_table__qwen25_32b__PRR_05.md`
- matching `.csv` files.

## How SAGE Works Here

`AGENT_BACKEND=sage` uses SAGE as the external agent controller. The LLM sees
the task state and action history, then chooses one of our available actions:

```text
generate, think, critic_L0, critic_L2, critic_L3, verify, finish
```

SAGE chooses the next action; our tool executor actually runs the action:
generation calls vLLM, critics run local/LLM checks, and `verify` runs the
hidden/private verifier.

The Bayesian model does not control the SAGE agent. The prior and Bayes belief
state are computed after the run from the saved trajectory. This gives an
uncertainty score for an externally controlled agent before the final verifier
label is revealed.
