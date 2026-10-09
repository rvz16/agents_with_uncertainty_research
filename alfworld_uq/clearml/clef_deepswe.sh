#!/usr/bin/env bash
# DeepSWE states with command output, built on the agent from the run archives
# (they are gigabytes; nothing is downloaded to a laptop), then scored by Clef.
set -euo pipefail

MODELS="${MODELS:-Cloudflare/clef-flash Cloudflare/clef}"
OUT_DIR="${OUT_DIR:-/tmp/clef_runs}"
STATES_DIR="${STATES_DIR:-/tmp/clef_runs/deepswe_states}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
# cohort:task_id:run_name, the three archives of the report's DeepSWE cohorts
ARCHIVES="${ARCHIVES:-DG:861bd85b9c85467c973e767b952956ff:deepswe_gptoss_113_v5 DL:9c3e98678a1f46898d44eb73e6363e33:deepswe_gptoss120_113_rp DQ:d9f8e22f8b764bab9bae2854d37403ea:deepswe_qwen_113_v8}"

mkdir -p "$OUT_DIR" "$STATES_DIR"
nvidia-smi || echo "no nvidia-smi"
df -h /tmp | tail -1

python -m pip install -q --upgrade pip
python -m pip install -q torch torchvision "transformers>=5.10" huggingface_hub accelerate safetensors pillow clearml

for entry in $ARCHIVES; do
    cohort="${entry%%:*}"; rest="${entry#*:}"; task_id="${rest%%:*}"; run_name="${rest#*:}"
    echo "[deepswe] $cohort <- $task_id ($run_name)"
    archive=$(python - "$task_id" <<'PY'
import sys
from clearml import Task
task = Task.get_task(task_id=sys.argv[1])
name = "run_root" if "run_root" in task.artifacts else list(task.artifacts)[0]
print(task.artifacts[name].get_local_copy())
PY
)
    echo "[deepswe] archive at $archive"
    python deep_swe_uq/experiments/deepswe_decision_state.py "$archive" \
        --run "$run_name" --cohort "$cohort" --out "$STATES_DIR/$cohort.jsonl"
    rm -rf ~/.clearml/cache/storage_manager/global/* || true
done

cd alfworld_uq
for model in $MODELS; do
    tag="$(echo "$model" | tr '/' '_')"
    echo "[clef] $model on the DeepSWE states"
    python -m experiments.decision_model_clef \
        --states "$STATES_DIR"/*.jsonl \
        --out "$OUT_DIR/answers_deepswe_${tag}.jsonl" \
        --model "$model" --max-length "$MAX_LENGTH" || echo "[clef] $model failed, continuing"
done
echo "[clef] done"
ls -la "$OUT_DIR"
