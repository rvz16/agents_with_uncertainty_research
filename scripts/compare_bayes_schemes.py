#!/usr/bin/env python3
"""Published belief against the corrected one, on the same episodes.

Why this is a separate script. The published scheme (`bayes_state` in
`analysis/analyze_lcb_llm_tool_agent_logs.py`) pushes the belief through a
transition kernel `belief*(1-p_break) + (1-belief)*p_fix` on every generation,
using constants marked "initial uninformative" in the sources. They were meant
to be calibrated, but no run ever passed `--kernel`, so the defaults were always
in force. One of the two cannot be measured at all: there are no observed
transitions out of the correct state, because the episode ends there.

The corrected scheme (`analysis/bayes_trajectory.py`) has no kernel: evidence is
summed in log-odds, each piece decays with generation index, and three scalars
are calibrated per benchmark. A regeneration on its own does not move the
belief; only observations do.

The script prints PRR for both schemes over the intersection of episodes.
Comparing two numbers on one dataset is the only way to show that replacing the
scheme changes anything; numbers from different runs are not comparable.

The corrected scheme is scored by cross-fitting over episodes: calibrated on the
training folds, evaluated on the held-out ones.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from code_uq.analysis.bayes_trajectory import belief_logit, calibrate  # noqa: E402
from code_uq.analysis.uq_features import (  # noqa: E402
    load_cell,
    per_generation_metrics,
)
from trajectory_uq_toolkit.metrics import prr  # noqa: E402

#: Per-generation features used in the step updates. Kept here rather than in
#: the module: this is an experimental choice, not a property of the scheme.
STEP_FEATURES = ["ntok", "sum", "answer_sum"]


def read_jsonl(path: Path) -> list[dict]:
    """Line by line, not `read_text().splitlines()`.

    `splitlines()` also breaks on U+2028, vertical tab and form feed, and
    the records hold raw model text.
    """
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def step_series(run_root: Path, stem: str, episodes: list[dict]) -> list[dict[str, list[float]]]:
    """Per episode, the series of scores over its generations, in order."""
    steps = per_generation_metrics(run_root, stem)
    out = []
    for episode in episodes:
        instance = str(episode["instance_id"])
        series: dict[str, list[float]] = {}
        index = 0
        for step in episode.get("trajectory") or []:
            if step.get("action") == "generate" and not step.get("skipped"):
                for key, value in steps.get((instance, index), {}).items():
                    if key in STEP_FEATURES:
                        series.setdefault(key, []).append(value)
                index += 1
        out.append(series)
    return out


def folds(instances: list[str], n_folds: int, seed: int) -> list[np.ndarray]:
    """Split by episode, not by step: steps within an episode are dependent."""
    order = np.random.RandomState(seed).permutation(len(instances))
    return [order[i::n_folds] for i in range(n_folds)]


def cross_fitted_belief(rows: list[dict], labels: np.ndarray,
                        series: list[dict], instances: list[str],
                        n_folds: int, seed: int) -> np.ndarray:
    out = np.full(len(labels), np.nan)
    parts = folds(instances, n_folds, seed)
    for held_index, held in enumerate(parts):
        if len(held) == 0:
            continue
        train = np.concatenate([p for j, p in enumerate(parts) if j != held_index])
        cal = calibrate([rows[int(i)] for i in train], labels[train],
                        [series[int(i)] for i in train], STEP_FEATURES)
        for i in held:
            out[i] = belief_logit(cal, rows[int(i)], series[int(i)], STEP_FEATURES)
    return out


#: Where the analyser writes its tables. `--output-dir` defaults to the run
#: root, while wrappers usually pass `readable/<benchmark>`; look in both.
READABLE_SUBDIRS = ("readable/{benchmark}", "readable", ".")


def published_table(run_root: Path, benchmark: str) -> Path | None:
    for pattern in READABLE_SUBDIRS:
        path = (run_root / pattern.format(benchmark=benchmark)
                / "final_logprob_bayes_quality.csv")
        if path.exists():
            return path
    return None


def published_belief(path: Path | None) -> dict[str, tuple[float, int]]:
    """`bayes_state` and the label per instance, from the analyser table."""
    if path is None:
        return {}
    out: dict[str, tuple[float, int]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            raw = row.get("bayes_state")
            if raw in (None, "", "None"):
                continue
            out[str(row["instance_id"])] = (float(raw), int(float(row["quality"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", type=Path, required=True,
                    help="run directory: <benchmark>__<generator>.jsonl and readable/")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3,
                    help="average over split seeds: on a hundred episodes the "
                         "spread of a single split rivals the effect")
    args = ap.parse_args()

    root: Path = args.run_root
    stem = f"{args.benchmark}__{args.generator}"
    pack = load_cell(str(root.parent), root.name, args.benchmark, args.generator,
                     label=root.name)
    if not pack:
        raise SystemExit(f"no episodes in {root / (stem + '.jsonl')}")

    episodes = read_jsonl(root / f"{stem}.jsonl")
    series = step_series(root, stem, episodes)
    labels, rows, instances = pack["y"], pack["features"], pack["instances"]

    table = published_table(root, args.benchmark)
    published = published_belief(table)
    if not published:
        print(f"no final_logprob_bayes_quality.csv under {root}: "
              "run analysis/analyze_lcb_llm_tool_agent_logs.py first")

    # Compare on the intersection only: an episode missing from one scheme would
    # shift its number through a different sample, not through the scheme.
    keep = [i for i, inst in enumerate(instances) if not published or inst in published]
    if len(set(labels[keep].tolist())) < 2:
        raise SystemExit("label is degenerate on this sample: PRR is uninformative")

    per_seed = []
    for seed in range(args.seeds):
        scores = cross_fitted_belief(rows, labels, series, instances, args.folds, seed)
        ok = [i for i in keep if np.isfinite(scores[i])]
        value = prr([float(scores[i]) for i in ok], [int(labels[i]) for i in ok])
        if value is not None:
            per_seed.append(value)

    print(f"episodes: {len(keep)}  success rate: {float(labels[keep].mean()):.3f}")
    if published:
        pairs = [published[instances[i]] for i in keep]
        value = prr([p[0] for p in pairs], [p[1] for p in pairs])
        print(f"PRR bayes_state (published scheme, transition kernel): "
              f"{'—' if value is None else f'{value:+.3f}'}")
    if per_seed:
        print(f"PRR belief_logit (corrected, no kernel, cross-fitted): "
              f"{float(np.mean(per_seed)):+.3f}  "
              f"(over {len(per_seed)} splits, spread {float(np.std(per_seed)):.3f})")
    else:
        print("PRR belief_logit: not computed (not enough data to calibrate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
