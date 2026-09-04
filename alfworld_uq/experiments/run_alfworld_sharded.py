from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def partition(total: int, workers: int) -> list[tuple[int, int]]:
    if total < 1 or workers < 1:
        raise ValueError("total and workers must be positive")
    workers = min(total, workers)
    base, remainder = divmod(total, workers)
    result = []
    offset = 0
    for worker in range(workers):
        count = base + int(worker < remainder)
        result.append((offset, count))
        offset += count
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run disjoint ALFWorld shards and merge their JSONL outputs."
    )
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["train", "valid_seen", "valid_unseen"],
        default="valid_seen",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy", choices=["llm", "random", "smolagents"], default="llm"
    )
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument(
        "--smol-code-tags",
        choices=["markdown", "xml"],
        default="markdown",
        help="Action format for --policy smolagents.",
    )
    parser.add_argument(
        "--agent-max-steps",
        type=int,
        default=0,
        help="Generation budget for --policy smolagents; 0 uses --max-steps.",
    )
    parser.add_argument("--max-generation-tokens", type=int, default=1024)
    parser.add_argument("--empty-response-retries", type=int, default=1)
    parser.add_argument("--provider-order", default="")
    parser.add_argument(
        "--allow-provider-fallbacks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _merge_jsonl(
    output_path: Path, shard_dirs: list[Path], filename: str
) -> int:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    rows = 0
    with temporary.open("w", encoding="utf-8") as destination:
        for shard_dir in shard_dirs:
            source_path = shard_dir / filename
            with source_path.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        destination.write(line)
                        rows += 1
    temporary.replace(output_path)
    return rows


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards_root = args.output_dir / "shards"
    shards_root.mkdir(exist_ok=True)
    layout = partition(args.num_episodes, args.workers)
    processes: list[tuple[subprocess.Popen[str], object, Path, int]] = []
    shard_dirs = []

    try:
        for shard_index, (offset, count) in enumerate(layout):
            shard_dir = shards_root / f"shard_{shard_index:02d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_dirs.append(shard_dir)
            log_handle = (shard_dir / "runner.log").open(
                "w" if args.overwrite else "a", encoding="utf-8"
            )
            command = [
                sys.executable,
                "-m",
                "experiments.run_alfworld",
                "--num-episodes",
                str(count),
                "--episode-offset",
                str(offset),
                "--max-steps",
                str(args.max_steps),
                "--output-dir",
                str(shard_dir),
                "--split",
                args.split,
                "--seed",
                str(args.seed),
                "--policy",
                args.policy,
                "--max-generation-tokens",
                str(args.max_generation_tokens),
                "--empty-response-retries",
                str(args.empty_response_retries),
                "--agent-max-steps",
                str(args.agent_max_steps),
                "--smol-code-tags",
                args.smol_code_tags,
                "--api-timeout",
                str(args.api_timeout),
            ]
            if args.provider_order:
                command.extend(["--provider-order", args.provider_order])
            if not args.allow_provider_fallbacks:
                command.append("--no-allow-provider-fallbacks")
            if args.overwrite:
                command.append("--overwrite")
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((process, log_handle, shard_dir, count))
            print(
                f"started shard {shard_index}: offset={offset}, "
                f"episodes={count}, pid={process.pid}",
                flush=True,
            )

        pending = set(range(len(processes)))
        while pending:
            time.sleep(10)
            for index in list(pending):
                process, _, shard_dir, expected = processes[index]
                return_code = process.poll()
                if return_code is None:
                    continue
                pending.remove(index)
                episodes_path = shard_dir / "episodes.jsonl"
                completed = (
                    sum(1 for line in episodes_path.open() if line.strip())
                    if episodes_path.exists()
                    else 0
                )
                print(
                    f"finished shard {index}: rc={return_code}, "
                    f"episodes={completed}/{expected}",
                    flush=True,
                )
                if return_code != 0 or completed != expected:
                    raise RuntimeError(
                        f"shard {index} failed; inspect {shard_dir / 'runner.log'}"
                    )
    except KeyboardInterrupt:
        for process, _, _, _ in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for _, log_handle, _, _ in processes:
            log_handle.close()

    episode_rows = _merge_jsonl(
        args.output_dir / "episodes.jsonl", shard_dirs, "episodes.jsonl"
    )
    trajectory_rows = _merge_jsonl(
        args.output_dir / "trajectories.jsonl", shard_dirs, "trajectories.jsonl"
    )
    if episode_rows != args.num_episodes:
        raise RuntimeError(
            f"Merged {episode_rows} episodes, expected {args.num_episodes}"
        )
    metadata = {
        "num_episodes": args.num_episodes,
        "workers": len(layout),
        "max_steps": args.max_steps,
        "split": args.split,
        "seed": args.seed,
        "policy": args.policy,
        "agent_max_steps": args.agent_max_steps,
        "smol_code_tags": args.smol_code_tags,
        "max_generation_tokens": args.max_generation_tokens,
        "empty_response_retries": args.empty_response_retries,
        "provider_order": args.provider_order,
        "allow_provider_fallbacks": args.allow_provider_fallbacks,
        "shards": [
            {"offset": offset, "num_episodes": count}
            for offset, count in layout
        ],
        "trajectory_rows": trajectory_rows,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "episodes": episode_rows,
                "trajectory_rows": trajectory_rows,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
