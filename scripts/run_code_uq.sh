#!/usr/bin/env bash
set -euo pipefail

GENERATOR_KEY="${GENERATOR_KEY:-gpt_oss_20b_local}"
BENCHMARKS="${BENCHMARKS:-lcb_hard,lcb_medium,lcb_easy,mbpp,humaneval,humanevalfix,codecontests}"
RUN_ROOT="${RUN_ROOT:-runs/code_uq/${GENERATOR_KEY}}"

N_INSTANCES="${N_INSTANCES:-0}"
N_TRAIN="${N_TRAIN:-}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.25}"
SPLIT_SEED="${SPLIT_SEED:-42}"
PRIOR_PATCHES="${PRIOR_PATCHES:-1}"
LCB_PLATFORM="${LCB_PLATFORM:-leetcode}"
LCB_VERSION="${LCB_VERSION:-all}"
PRIVATE_TEST_CAP="${PRIVATE_TEST_CAP:-12}"
PLUS_INPUT_CAP="${PLUS_INPUT_CAP:-200}"
SWE_HARNESS_WORKERS="${SWE_HARNESS_WORKERS:-1}"
MAX_STEPS="${MAX_STEPS:-20}"
MAX_GENERATIONS="${MAX_GENERATIONS:-5}"
MAX_VERIFICATIONS="${MAX_VERIFICATIONS:-2}"
FINAL_VERIFY="${FINAL_VERIFY:-1}"
TOP_LOGPROBS="${TOP_LOGPROBS:-20}"
SAVE_VERBALIZED_2S="${SAVE_VERBALIZED_2S:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
RESUME="${RESUME:-1}"

mkdir -p "${RUN_ROOT}"

run_one() {
  local bench="$1"
  local stem="${bench}__${GENERATOR_KEY}"
  local output="${RUN_ROOT}/${stem}.jsonl"
  local readable="${RUN_ROOT}/readable/${bench}"
  local train_args=(--train-fraction "${TRAIN_FRACTION}")
  local optional_args=()

  if [[ -n "${N_TRAIN}" ]]; then
    train_args=(--n-train "${N_TRAIN}")
  fi
  if [[ "${FINAL_VERIFY}" == "1" ]]; then
    optional_args+=(--final-verify)
  fi
  if [[ "${RESUME}" == "1" ]]; then
    optional_args+=(--resume)
  fi
  if [[ "${SAVE_VERBALIZED_2S}" == "1" ]]; then
    optional_args+=(
      --save-verbalized-2s
      --verbalized-2s-output "${RUN_ROOT}/${stem}.verbalized_2s.jsonl"
    )
  fi

  python -m code_uq.agents.lcb_llm_tool_agent \
    --benchmark "${bench}" \
    --generator "${GENERATOR_KEY}" \
    --n-instances "${N_INSTANCES}" \
    --split-seed "${SPLIT_SEED}" \
    "${train_args[@]}" \
    --prior-patches "${PRIOR_PATCHES}" \
    --platform "${LCB_PLATFORM}" \
    --lcb-version "${LCB_VERSION}" \
    --private-test-cap "${PRIVATE_TEST_CAP}" \
    --plus-input-cap "${PLUS_INPUT_CAP}" \
    --swe-harness-workers "${SWE_HARNESS_WORKERS}" \
    --max-steps "${MAX_STEPS}" \
    --max-generations "${MAX_GENERATIONS}" \
    --max-verifications "${MAX_VERIFICATIONS}" \
    --save-generation-logprobs \
    --require-generation-logprobs \
    --top-logprobs "${TOP_LOGPROBS}" \
    --output "${output}" \
    --logprobs-output "${RUN_ROOT}/${stem}.generation_logprobs.jsonl" \
    --split-output "${RUN_ROOT}/${stem}.split.json" \
    --prior-calibration-output "${RUN_ROOT}/${stem}.train_prior_calibration.jsonl" \
    --prior-calibration-logprobs-output "${RUN_ROOT}/${stem}.train_prior_calibration.generation_logprobs.jsonl" \
    --actions-output "${RUN_ROOT}/${stem}.actions.jsonl" \
    --print-each \
    "${optional_args[@]}"

  if [[ "${RUN_ANALYSIS}" == "1" ]]; then
    mkdir -p "${readable}"
    python -m code_uq.analysis.summarize_tool_action_success \
      "${output}" \
      --per-instance-csv "${readable}/tool_success_by_instance.csv"
    python -m code_uq.analysis.analyze_lcb_llm_tool_agent_logs \
      --run-root "${RUN_ROOT}" \
      --benchmark "${bench}" \
      --generator "${GENERATOR_KEY}" \
      --tool-success-csv "${readable}/tool_success_by_instance.csv" \
      --output-dir "${readable}" \
      --seed "${SPLIT_SEED}" \
      --platform "${LCB_PLATFORM}" \
      --lcb-version "${LCB_VERSION}" \
      --private-test-cap "${PRIVATE_TEST_CAP}" \
      --plus-input-cap "${PLUS_INPUT_CAP}" \
      --swe-harness-workers "${SWE_HARNESS_WORKERS}"
  fi
}

IFS=',' read -ra benchmark_array <<< "${BENCHMARKS}"
for benchmark in "${benchmark_array[@]}"; do
  run_one "${benchmark}"
done
