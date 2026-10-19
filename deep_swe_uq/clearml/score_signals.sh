#!/usr/bin/env bash
# Re-score recorded DeepSWE actions to recover the per-step uncertainty signals.
# Inference only: one teacher-forced pass per step, no environment and no agent
# loop. vLLM serves the model because that is the path this project has already
# run these three executors on; its prompt_logprobs is the teacher forcing.
set -uo pipefail
OUT_DIR="${OUT_DIR:-/tmp/signal_runs}"
COHORT="${COHORT:-DG}"
MODEL="${MODEL:-openai/gpt-oss-20b}"
TOP_K="${TOP_K:-20}"
WINDOW="${WINDOW:-10}"
OBS_CHARS="${OBS_CHARS:-400}"
MAX_LEN="${MAX_LEN:-32768}"
TP="${TP:-1}"
GPU_UTIL="${GPU_UTIL:-0.90}"
BATCH="${BATCH:-8}"
LIMIT="${LIMIT:-}"

mkdir -p "$OUT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[score] no nvidia-smi"

# the wheels carry their own CUDA runtime, so a plain python image is enough;
# an earlier attempt on pytorch/pytorch hit mkl-service against libgomp and then
# a transformers that declared the image's own torch unavailable
python -m pip install -q --no-cache-dir "vllm==${VLLM_VERSION:-0.28.0}" || {
    echo "[score] vllm install failed"; exit 20; }
python -c "import vllm, torch; print('[score] vllm', vllm.__version__, 'torch', torch.__version__)" || {
    echo "[score] vllm imports but does not load"; exit 21; }
# the sampler this project already runs these three models with
export VLLM_USE_FLASHINFER_SAMPLER=0
which nvcc >/dev/null 2>&1 || echo "[score] no nvcc: the engine must stay eager"

cd deep_swe_uq
python -m experiments.score_signals \
    --inputs "data/replay_inputs/${COHORT}.jsonl.gz" \
    --views "../alfworld_uq/data/judge_views/${COHORT}.jsonl" \
    --model "$MODEL" --out "${OUT_DIR}/compact_${COHORT}.jsonl" \
    --top-k "$TOP_K" --window "$WINDOW" --obs-chars "$OBS_CHARS" \
    --max-len "$MAX_LEN" --tensor-parallel-size "$TP" --gpu-util "$GPU_UTIL" \
    --batch "$BATCH" \
    ${LIMIT:+--limit "$LIMIT"}
rc=$?
echo "[score] done rc=$rc"; ls -la "$OUT_DIR"
exit $rc
