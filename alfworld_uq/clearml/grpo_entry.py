#!/usr/bin/env python3
"""ClearML entry for GRPO over the confidence head.

Two things travel as artifacts rather than through git: the cohort inputs, as
for the supervised run, and -- for the on-top-of-SFT arm -- the supervised run's
per-fold adapters, which are laid out here as the trainer expects to find them.
"""
import os
import subprocess
import sys
from pathlib import Path

from clearml import Task

FILE_STORE = "https://files.clearai.innopolis.university"


def fetch_data(task, repo: Path) -> None:
    data_task = os.environ.get("DATA_TASK", "").strip()
    if not data_task:
        return
    source = Task.get_task(task_id=data_task)
    for name, artifact in (source.artifacts or {}).items():
        if "__" not in name:
            continue
        directory, stem = name.split("__", 1)
        target = repo / "alfworld_uq" / "data" / directory / f"{stem}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(artifact.get_local_copy()).read_bytes())
        print(f"[entry] fetched {directory}/{stem}.jsonl ({target.stat().st_size} bytes)",
              flush=True)


def fetch_adapters(root: Path) -> str:
    """Lay the supervised run's weights_s0f* artifacts out as <root>/s0f*/.

    The trainer takes one adapter per fold and refuses to mix them, so the
    directory names have to survive the round trip through the artifact store.
    """
    sft_task = os.environ.get("SFT_TASK", "").strip()
    if not sft_task:
        return ""
    source = Task.get_task(task_id=sft_task)
    root.mkdir(parents=True, exist_ok=True)
    found = 0
    for name, artifact in (source.artifacts or {}).items():
        if not name.startswith("weights_"):
            continue
        local = Path(artifact.get_local_copy())
        # a directory artifact comes back either as the directory itself or as a
        # single wrapper holding it; the adapter config is the marker either way
        if not (local / "adapter_config.json").exists():
            inner = [p for p in local.rglob("adapter_config.json")]
            if not inner:
                print(f"[entry] {name}: no adapter_config.json, skipped", flush=True)
                continue
            local = inner[0].parent
        target = root / name[len("weights_"):]
        target.mkdir(parents=True, exist_ok=True)
        for item in local.iterdir():
            if item.is_file():
                (target / item.name).write_bytes(item.read_bytes())
        found += 1
        print(f"[entry] adapter {target.name}: {sorted(p.name for p in target.iterdir())}",
              flush=True)
    if not found:
        raise SystemExit(f"SFT_TASK {sft_task} has no weights_* artifacts: the supervised run "
                         f"must have finished at least one fold with SAVE_WEIGHTS=1")
    return str(root)


def main() -> int:
    task = Task.current_task() or Task.init(project_name="agentic-uq", task_name="grpo-decision")
    task.output_uri = FILE_STORE
    for key, value in (task.get_parameters_as_dict().get("Args", {}) or {}).items():
        if value not in (None, ""):
            os.environ[key] = str(value)
    repo = Path(__file__).resolve().parents[2]
    fetch_data(task, repo)
    out_dir = Path(os.environ.get("OUT_DIR", "/tmp/decision_runs")) / task.id
    os.environ["OUT_DIR"] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapters = fetch_adapters(out_dir / "sft_adapters")
    if adapters:
        os.environ["INIT_ADAPTER"] = adapters
    rc = subprocess.call(["bash", str(repo / "alfworld_uq" / "clearml" / "grpo.sh")],
                         cwd=str(repo))
    print(f"[entry] grpo rc={rc}", flush=True)
    for f in sorted(list(out_dir.glob("answers_*.jsonl")) + list(out_dir.glob("steps_*.jsonl"))
                    + list(out_dir.glob("summary_*.json"))):
        if f.stat().st_size:
            task.upload_artifact(f.stem, artifact_object=f, wait_on_upload=True)
            print(f"[entry] uploaded {f.name} ({f.stat().st_size} bytes)", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
