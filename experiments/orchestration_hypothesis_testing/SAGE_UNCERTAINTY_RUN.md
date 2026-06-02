# SAGE Uncertainty Run

Minimal note on how we ran the LCB uncertainty experiments with the SAGE
agent.

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
  --max-steps 20 \
  --max-generations 5 \
  --agent-backend sage \
  --final-verify \
  --save-generation-logprobs \
  --require-generation-logprobs \
  --output sim_results/sage_uq/lcb_hard__gpt_oss_20b_local.jsonl \
  --logprobs-output sim_results/sage_uq/lcb_hard__gpt_oss_20b_local.generation_logprobs.jsonl \
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
- `lcb_hard__gpt_oss_20b_local.train_prior_calibration.jsonl`: train split used
  to estimate the prior.

## Analysis

Convert raw logs into uncertainty tables:

```bash
python experiments/orchestration_hypothesis_testing/scripts/analyze_lcb_llm_tool_agent_logs.py \
  --run-root sim_results/sage_uq \
  --benchmark lcb_hard \
  --generator gpt_oss_20b_local \
  --output-dir sim_results/sage_uq/readable
```

This produces:

- `final_logprob_bayes_quality.csv`: final quality, logprob scores, Bayes state.
- `generation_trajectory_scores.csv`: per-generation Bayes/logprob trajectory.
- `final_logprob_bayes_quality.jsonl`: aligned metadata.

Tool/action success summary:

```bash
python experiments/orchestration_hypothesis_testing/scripts/summarize_tool_action_success.py \
  sim_results/sage_uq/lcb_hard__gpt_oss_20b_local.jsonl \
  --per-instance-csv sim_results/sage_uq/tool_success_by_instance.csv
```

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
