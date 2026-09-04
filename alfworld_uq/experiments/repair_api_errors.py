from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# `api_error` comes from the ReAct loop, `agent_error` from the smolagents one.
REPAIRABLE_STOP_REASONS = frozenset({"api_error", "agent_error"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    temporary.replace(path)


def replace_episodes(
    episodes: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    replacements: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        by_episode[row["episode_id"]].append(row)
    result_episodes = []
    result_trajectories = []
    for index, original_summary in enumerate(episodes):
        if index in replacements:
            summary, rows = replacements[index]
        else:
            summary = original_summary
            rows = by_episode[original_summary["episode_id"]]
        result_episodes.append(summary)
        result_trajectories.extend(sorted(rows, key=lambda row: int(row["step"])))
    return result_episodes, result_trajectories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequentially rerun failed-endpoint episodes and replace them atomically."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-repair-attempts", type=int, default=3)
    parser.add_argument("--overwrite-repairs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = json.loads((args.run_dir / "run_config.json").read_text())
    episodes_path = args.run_dir / "episodes.jsonl"
    trajectories_path = args.run_dir / "trajectories.jsonl"
    episodes = _read_jsonl(episodes_path)
    trajectories = _read_jsonl(trajectories_path)
    failed_indices = [
        index
        for index, summary in enumerate(episodes)
        if summary.get("stop_reason") in REPAIRABLE_STOP_REASONS
    ]
    if not failed_indices:
        print("No failed-endpoint episodes to repair.")
        return

    repairs_root = args.run_dir / "repairs"
    repairs_root.mkdir(exist_ok=True)
    replacements: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    repair_metadata = []
    for number, episode_index in enumerate(failed_indices, 1):
        original = episodes[episode_index]
        repaired = False
        for attempt in range(1, args.max_repair_attempts + 1):
            attempt_dir = (
                repairs_root / f"episode_{episode_index:03d}" / f"attempt_{attempt}"
            )
            attempt_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "experiments.run_alfworld",
                "--num-episodes",
                "1",
                "--gamefile",
                original["gamefile"],
                "--max-steps",
                str(config["max_steps"]),
                "--output-dir",
                str(attempt_dir),
                "--split",
                config["split"],
                "--seed",
                str(config["seed"]),
                "--policy",
                config["policy"],
                "--max-generation-tokens",
                str(config["max_generation_tokens"]),
                "--empty-response-retries",
                str(config["empty_response_retries"]),
                "--agent-max-steps",
                str(config.get("agent_max_steps", 0)),
            ]
            provider_order = config.get("provider_order", "")
            if provider_order:
                command.extend(["--provider-order", provider_order])
            if not config.get("allow_provider_fallbacks", True):
                command.append("--no-allow-provider-fallbacks")
            if args.overwrite_repairs:
                command.append("--overwrite")
            log_path = attempt_dir / "repair.log"
            with log_path.open("w", encoding="utf-8") as log_handle:
                result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if result.returncode != 0:
                print(
                    f"[{number}/{len(failed_indices)}] offset={episode_index} "
                    f"attempt={attempt}: process rc={result.returncode}",
                    flush=True,
                )
                continue
            repaired_summary = _read_jsonl(attempt_dir / "episodes.jsonl")[0]
            print(
                f"[{number}/{len(failed_indices)}] offset={episode_index} "
                f"attempt={attempt}: {repaired_summary['stop_reason']}",
                flush=True,
            )
            if repaired_summary["stop_reason"] in REPAIRABLE_STOP_REASONS:
                continue
            repaired_rows = _read_jsonl(attempt_dir / "trajectories.jsonl")
            if repaired_summary["episode_id"] != original["episode_id"]:
                raise RuntimeError(
                    f"Offset {episode_index} produced {repaired_summary['episode_id']}, "
                    f"expected {original['episode_id']}"
                )
            replacements[episode_index] = (repaired_summary, repaired_rows)
            repair_metadata.append(
                {
                    "episode_index": episode_index,
                    "episode_id": original["episode_id"],
                    "attempts": attempt,
                    "original_steps": original["num_steps"],
                    "original_tokens": original["total_tokens"],
                    "repaired_steps": repaired_summary["num_steps"],
                    "repaired_tokens": repaired_summary["total_tokens"],
                    "repaired_stop_reason": repaired_summary["stop_reason"],
                }
            )
            repaired = True
            break
        if not repaired:
            raise RuntimeError(
                f"Could not repair offset {episode_index} after "
                f"{args.max_repair_attempts} attempts"
            )

    repaired_episodes, repaired_trajectories = replace_episodes(
        episodes, trajectories, replacements
    )
    if any(
        row.get("stop_reason") in REPAIRABLE_STOP_REASONS
        for row in repaired_episodes
    ):
        raise RuntimeError(
            "endpoint failure remains after repair; original files were not changed"
        )
    backup_episodes = args.run_dir / "episodes.pre_repair.jsonl"
    backup_trajectories = args.run_dir / "trajectories.pre_repair.jsonl"
    if not backup_episodes.exists():
        shutil.copy2(episodes_path, backup_episodes)
    if not backup_trajectories.exists():
        shutil.copy2(trajectories_path, backup_trajectories)
    _write_jsonl_atomic(episodes_path, repaired_episodes)
    _write_jsonl_atomic(trajectories_path, repaired_trajectories)
    (args.run_dir / "repair_metadata.json").write_text(
        json.dumps(repair_metadata, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "repaired_episodes": len(replacements),
                "episodes": len(repaired_episodes),
                "trajectory_rows": len(repaired_trajectories),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
