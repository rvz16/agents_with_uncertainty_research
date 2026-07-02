#!/usr/bin/env bash
set -euo pipefail

# Thin Slurm submitter for run_sage_uncertainty_experiments.sh.
# It does not start vLLM. Start an OpenAI-compatible endpoint separately, or
# submit this on the same node where the endpoint is already available.

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
ACCOUNT="${ACCOUNT:-a0142}"
TIME="${TIME:-12:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-256G}"
GRES="${GRES:-gpu:1}"
CONDA_ENV="${CONDA_ENV:-agents}"
CONDA_ROOT="${CONDA_ROOT:-/users/avazhentsev/miniconda3}"
JOB_NAME="${JOB_NAME:-sage-uq}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/experiments/orchestration_hypothesis_testing/logs/sage_uncertainty}"
SCRIPT="${REPO_DIR}/experiments/orchestration_hypothesis_testing/scripts/run_sage_uncertainty_experiments.sh"

mkdir -p "${LOG_DIR}"

sbatch \
  --account="${ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --chdir="${REPO_DIR}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --gres="${GRES}" \
  --output="${LOG_DIR}/%x-%j.out" \
  --error="${LOG_DIR}/%x-%j.err" \
  --export=ALL,CONDA_ENV="${CONDA_ENV}" \
  --wrap="set -euo pipefail; source \"${CONDA_ROOT}/etc/profile.d/conda.sh\"; conda activate \"${CONDA_ENV}\"; bash \"${SCRIPT}\""
