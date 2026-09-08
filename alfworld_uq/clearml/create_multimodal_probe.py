#!/usr/bin/env python3
"""Queue the probe that decides whether a visual ALFWorld run is possible.

Two unknowns, both cheap to settle and both blocking: whether ai2thor renders
headless in our container, and whether our own vLLM takes an image and still
returns log-probabilities for the answer. The model itself turned out to need
nothing -- the checkpoint already in use is image-text-to-text.
"""
import argparse

from clearml import Task

REPO = "https://github.com/rvz16/agents_with_uncertainty_research.git"
FILE_STORE = "https://files.clearai.innopolis.university"
DOCKER_IMAGE = "python:3.12"
# ai2thor renders through the GPU; --network=host keeps the served endpoint
# reachable, and the cache mount stops a 70 GB re-download.
DOCKER_ARGS = (
    "--entrypoint= --network=host --shm-size=16g "
    "-v /root/.clearml/hf-cache:/root/.cache/huggingface"
)
SETUP = """
df -h /
rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true
apt-get update -qq -o Acquire::AllowInsecureRepositories=true || true
apt-get install -y -qq --no-install-recommends git curl libvulkan1 libgl1 || true
nvidia-smi || echo "no nvidia-smi"
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="agentic-uq")
    p.add_argument("--name", default="alfworld multimodal probe")
    p.add_argument("--queue", default="high_q_2xA100_80")
    p.add_argument("--branch", default="alfworld_smolagents")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--serve", action="store_true",
                   help="also stand the model up and send it an image; without "
                        "this only the simulator half is probed, which needs no GPU "
                        "memory and can run anywhere")
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--max-model-len", type=int, default=32768)
    a = p.parse_args()

    task = Task.create(
        project_name=a.project,
        task_name=a.name,
        repo=REPO,
        branch=a.branch,
        script="alfworld_uq/clearml/probe_multimodal.py",
        docker=f"{DOCKER_IMAGE} {DOCKER_ARGS}",
        docker_bash_setup_script=SETUP,
        packages=["clearml", "boto3", "pillow", "openai", "alfworld"],
    )
    task.output_uri = FILE_STORE
    params = {"Args/MODEL": a.model}
    if a.serve:
        params.update({
            "Args/SERVE": "1",
            "Args/TENSOR_PARALLEL_SIZE": str(a.tensor_parallel_size),
            "Args/MAX_MODEL_LEN": str(a.max_model_len),
        })
    task.set_parameters(params)
    print(f"Created task {task.id}")
    for key, value in params.items():
        print(f"  {key}={value}")
    Task.enqueue(task, queue_name=a.queue)
    print(f"Enqueued to {a.queue!r}")


if __name__ == "__main__":
    main()
