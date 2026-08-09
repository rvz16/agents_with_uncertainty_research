#!/usr/bin/env python3
"""Convert current ALFWorld logs to the compact portable episode schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from trajectory_uq_toolkit.schema import validate_episode


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_ids(
    episode_rows: list[dict[str, Any]], limit: int | None, seed: int
) -> list[str]:
    ordered = sorted(
        episode_rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['episode_id']}".encode()).hexdigest(),
    )
    if limit is None or limit >= len(ordered):
        return [str(row["episode_id"]) for row in ordered]
    by_label = {
        label: [row for row in ordered if int(bool(row["final_success"])) == label]
        for label in (0, 1)
    }
    selected = []
    target_success = min(len(by_label[1]), limit // 2)
    target_failure = min(len(by_label[0]), limit - target_success)
    selected.extend(by_label[1][:target_success])
    selected.extend(by_label[0][:target_failure])
    if len(selected) < limit:
        selected_ids = {row["episode_id"] for row in selected}
        remaining = [row for row in ordered if row["episode_id"] not in selected_ids]
        selected.extend(remaining[: limit - len(selected)])
    return [str(row["episode_id"]) for row in selected]


def compact(
    *,
    trajectories_path: Path,
    episodes_path: Path,
    output_path: Path,
    judge_path: Path | None = None,
    target: str = "combined",
    limit: int | None = None,
    seed: int = 0,
) -> int:
    episodes = read_jsonl(episodes_path)
    selected = set(select_ids(episodes, limit, seed))
    episode_metadata = {str(row["episode_id"]): row for row in episodes if str(row["episode_id"]) in selected}
    judges = {}
    if judge_path:
        judges = {
            str(row["episode_id"]): bool(row["judge_pass"])
            for row in read_jsonl(judge_path)
            if row.get("status") == "ok" and row.get("judge_pass") is not None
        }
    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with trajectories_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = str(row["episode_id"])
            if episode_id in selected:
                trajectories[episode_id].append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for episode_id in sorted(selected):
            rows = sorted(trajectories.get(episode_id, []), key=lambda row: int(row.get("step", 0)))
            if not rows or episode_id not in episode_metadata:
                continue
            generations = []
            for index, row in enumerate(rows):
                source_signals = ((row.get("uq") or {}).get(target) or {})
                signals = {
                    name: value
                    for name, value in source_signals.items()
                    if value is not None and isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                generations.append(
                    {
                        "index": index,
                        "signals": signals,
                        "critics": {
                            "format_valid": row.get("format_valid") is True,
                            "action_valid": row.get("action_valid") is True,
                            "no_repeated_fallback": row.get("fallback_reason") != "repeated_action",
                        },
                    }
                )
            critics = {
                "all_formats_valid": all(row["critics"]["format_valid"] for row in generations),
                "all_actions_valid": all(row["critics"]["action_valid"] for row in generations),
                "no_repeated_fallback": all(
                    row["critics"]["no_repeated_fallback"] for row in generations
                ),
            }
            if episode_id in judges:
                critics["llm_judge_pass"] = judges[episode_id]
            metadata = episode_metadata[episode_id]
            record = validate_episode(
                {
                    "episode_id": episode_id,
                    "environment": "alfworld",
                    "success": int(bool(metadata["final_success"])),
                    "generations": generations,
                    "critics": critics,
                    "metadata": {
                        "task_type": metadata.get("task_type"),
                        "num_steps": metadata.get("num_steps"),
                    },
                }
            )
            handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--judge-scores", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=["thought", "action", "combined"], default="combined")
    parser.add_argument("--limit", type=int, help="Deterministic approximately balanced sample")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    written = compact(
        trajectories_path=args.trajectories,
        episodes_path=args.episodes,
        judge_path=args.judge_scores,
        output_path=args.output,
        target=args.target,
        limit=args.limit,
        seed=args.seed,
    )
    print(f"wrote {written} compact episodes to {args.output}")


if __name__ == "__main__":
    main()
