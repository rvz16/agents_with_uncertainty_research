"""Repeat the trajectory analysis over many calibration/test splits.

A single 50/50 split of 100 episodes leaves ~20 positive test outcomes, so one
run of `analyze_trajectories` reports numbers that move by a lot when the split
changes. This runs the same analysis for several deterministic split seeds and
reports the mean and spread per (target, uq_method, model), which is what a
result table should carry.

    python -m experiments.aggregate_splits \\
      --trajectories runs/alfworld_baseline_140/trajectories.jsonl \\
      --output-dir runs/alfworld_baseline_140/analysis_splits \\
      --seeds 20 --jobs 4

Split seeds are 0..seeds-1, so seed 0 reproduces the single-split analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


METRICS = ["auroc", "auprc", "prr_at_0_5", "brier", "nll", "ece"]
KEY_FIELDS = ["target", "uq_method", "model", "prefix"]

# The rows the ALFWorld report leads with; summarised first in summary.md.
HEADLINE = [
    ("combined", "perplexity", "bayes_state"),
    ("thought", "sum_logprob", "bayes_state"),
    ("thought", "sum_logprob", "bayes_state_plus_binary"),
    ("thought", "sum_logprob", "bayes_state_plus_sep"),
    ("thought", "sum_logprob", "bayes_state_plus_tempered"),
    ("combined", "sum_logprob", "bayes_state_plus_sep"),
]


def _run_one(seed: int, args: argparse.Namespace, passthrough: list[str]) -> Path:
    out = args.output_dir / f"seed_{seed:02d}"
    command = [
        sys.executable,
        "-m",
        "experiments.analyze_trajectories",
        "--trajectories",
        str(args.trajectories),
        "--output-dir",
        str(out),
        "--seed",
        str(seed),
        "--calibration-fraction",
        str(args.calibration_fraction),
        *passthrough,
    ]
    if args.episodes:
        command.extend(["--episodes", str(args.episodes)])
    if args.judge_scores:
        command.extend(["--judge-scores", str(args.judge_scores)])
    log = out / "analysis.log"
    out.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise SystemExit(f"seed {seed} failed; see {log}")
    return out / "metrics.csv"


def _read_test_rows(path: Path) -> dict[tuple[str, ...], dict[str, float | None]]:
    rows: dict[tuple[str, ...], dict[str, float | None]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "test":
                continue
            key = tuple(row[field] for field in KEY_FIELDS)
            rows[key] = {
                metric: (float(row[metric]) if row[metric] else None)
                for metric in METRICS
            }
    return rows


def _summarise(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "max": ordered[-1],
        "p10": ordered[max(0, round(0.10 * (len(ordered) - 1)))],
        "p90": ordered[max(0, round(0.90 * (len(ordered) - 1)))],
    }


def _markdown(summary: dict[tuple[str, ...], dict[str, dict[str, float]]],
              seeds: int, episodes: int, positives: int) -> str:
    lines = [
        "# ALFWorld trajectory UQ over repeated splits",
        "",
        f"- Splits: {seeds} deterministic calibration/test partitions (seeds 0-{seeds - 1})",
        f"- Episodes: {episodes} ({positives} successes)",
        "- Every cell is the mean over splits; +- is the standard deviation, "
        "and the range is the 10th-90th percentile across splits.",
        "",
        "## Headline models",
        "",
        "| Target | Method | Model | AUROC | PRR@0.5 | Brier |",
        "|---|---|---|---|---|---|",
    ]

    def cell(stats: dict[str, float] | None) -> str:
        if not stats:
            return "n/a"
        return f"{stats['mean']:.3f} ± {stats['std']:.3f} [{stats['p10']:.3f}, {stats['p90']:.3f}]"

    for target, method, model in HEADLINE:
        key = (target, method, model, "")
        stats = summary.get(key)
        if not stats:
            continue
        lines.append(
            f"| {target} | {method} | {model} | {cell(stats.get('auroc'))} | "
            f"{cell(stats.get('prr_at_0_5'))} | {cell(stats.get('brier'))} |"
        )

    lines += ["", "## Most split-sensitive cells (AUROC)", "",
              "| Target | Method | Model | mean | std | min | max |",
              "|---|---|---|---|---|---|---|"]
    ranked = sorted(
        (
            (stats["auroc"]["std"], key, stats["auroc"])
            for key, stats in summary.items()
            if "auroc" in stats
        ),
        reverse=True,
    )
    for _, key, stats in ranked[:10]:
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {stats['mean']:.3f} | "
            f"{stats['std']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--episodes", type=Path)
    parser.add_argument("--judge-scores", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--jobs", type=int, default=4)
    return parser


def main() -> None:
    parser = build_parser()
    args, passthrough = parser.parse_known_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        metric_paths = list(
            pool.map(lambda seed: _run_one(seed, args, passthrough), range(args.seeds))
        )

    collected: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in metric_paths:
        for key, metrics in _read_test_rows(path).items():
            for metric, value in metrics.items():
                if value is not None:
                    collected[key][metric].append(value)

    summary = {
        key: {metric: _summarise(values) for metric, values in metrics.items() if values}
        for key, metrics in collected.items()
    }

    fields = ["target", "uq_method", "model", "prefix", "metric",
              "n_splits", "mean", "std", "min", "max", "p10", "p90"]
    rows = [
        {
            **dict(zip(KEY_FIELDS, key)),
            "metric": metric,
            "n_splits": stats["n"],
            **{name: round(stats[name], 6) for name in ("mean", "std", "min", "max", "p10", "p90")},
        }
        for key, metrics in sorted(summary.items())
        for metric, stats in metrics.items()
    ]
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    episodes_path = args.episodes or args.trajectories.with_name("episodes.jsonl")
    episode_rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    positives = sum(1 for row in episode_rows if row.get("final_success"))
    (args.output_dir / "summary.md").write_text(
        _markdown(summary, args.seeds, len(episode_rows), positives), encoding="utf-8"
    )
    print(json.dumps({
        "splits": args.seeds,
        "episodes": len(episode_rows),
        "successes": positives,
        "summary_rows": len(rows),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
