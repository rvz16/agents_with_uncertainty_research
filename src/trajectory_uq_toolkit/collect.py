from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from .schema import SCHEMA_VERSION, validate_episode


def _load_adapter(reference: str) -> ModuleType:
    path = Path(reference)
    if path.suffix == ".py" and path.exists():
        spec = importlib.util.spec_from_file_location("trajectory_uq_external_adapter", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import adapter from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(reference)


def _read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(str(json.loads(line)["episode_id"]))
    return completed


def collect(
    *,
    adapter_reference: str,
    config: dict[str, Any],
    output_dir: Path,
    limit: int | None,
    seed: int,
    workers: int,
) -> tuple[int, int]:
    adapter = _load_adapter(adapter_reference)
    if not hasattr(adapter, "list_episodes") or not hasattr(adapter, "run_episode"):
        raise ValueError("adapter must expose list_episodes(config) and run_episode(id, seed, config)")
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.jsonl"
    errors_path = output_dir / "errors.jsonl"
    completed = _read_completed(episodes_path)
    episode_ids = [str(value) for value in adapter.list_episodes(config)]
    if limit is not None:
        episode_ids = episode_ids[:limit]
    pending = [(position, episode_id) for position, episode_id in enumerate(episode_ids) if episode_id not in completed]

    run_config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "adapter": adapter_reference,
        "seed": seed,
        "workers": workers,
        "requested_episodes": len(episode_ids),
        "adapter_config": config,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    def execute(position: int, episode_id: str) -> dict[str, Any]:
        record = adapter.run_episode(episode_id, seed + position, config)
        normalized = validate_episode(record)
        if normalized["episode_id"] != episode_id:
            raise ValueError(f"adapter returned episode_id={normalized['episode_id']!r}, expected {episode_id!r}")
        return normalized

    succeeded = 0
    failed = 0
    with episodes_path.open("a", encoding="utf-8") as episode_handle, errors_path.open("a", encoding="utf-8") as error_handle:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(execute, position, episode_id): episode_id
                for position, episode_id in pending
            }
            for future in as_completed(futures):
                episode_id = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    failed += 1
                    error_handle.write(
                        json.dumps(
                            {"episode_id": episode_id, "error_type": type(exc).__name__, "error": str(exc)},
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                    error_handle.flush()
                    continue
                episode_handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
                episode_handle.flush()
                succeeded += 1
    return succeeded, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an environment adapter and save compact UQ episodes")
    parser.add_argument("--adapter", required=True, help="Import path or path/to/adapter.py")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    succeeded, failed = collect(
        adapter_reference=args.adapter,
        config=config,
        output_dir=args.output_dir,
        limit=args.limit,
        seed=args.seed,
        workers=args.workers,
    )
    print(f"collected={succeeded} failed={failed} output={args.output_dir}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
