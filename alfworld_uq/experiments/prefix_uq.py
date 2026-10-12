"""Our own uncertainty methods against a label at every step.

Everywhere else an episode carries one label, its final outcome, and every
prefix of it inherits that label. This scores prefixes against what was
actually true at the step they end on, which is a different question -- how
much of the task is solved *now* rather than whether it will be solved -- and
turns one episode into as many scored units as it has verified points.

The labels come from a file rather than from here, because producing them means
grading a patch in the task's own container: ``{"id": ..., "step": k, "score":
c}`` with the verifier's partial credit at that point. Between two graded points
the repository has not changed, so the label is carried forward, and a prefix
before the first graded point is dropped rather than guessed.

Folds are cut by episode and then expanded to that episode's prefixes, never by
prefix: a prefix and its continuation share almost all their evidence, so a
split that separates them reports a model's ability to interpolate inside one
trajectory. For the same reason the effective sample size stays close to the
number of episodes, and the intervals should be read against that rather than
against the row count.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.prr_report_v2 import (FOLDS, SEEDS, SIGNALS, load_deepswe, method_table, prr,
                                       with_labels)

RUNS = {"DG": "deepswe_gptoss_113_v5", "DL": "deepswe_gptoss120_113_rp", "DQ": "deepswe_qwen_113_v8"}


def read_labels(path: Path) -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for line in open(path):
        if line.strip():
            row = json.loads(line)
            out[row["id"]].append((int(row["step"]), float(row["score"])))
    return {k: sorted(v) for k, v in out.items()}


def prefixes(episodes: dict[str, dict], labels: dict[str, list[tuple[int, float]]],
             carry: bool, stride: int) -> dict[str, dict]:
    """One scored unit per label point, with every sequence cut at that step."""
    out: dict[str, dict] = {}
    for eid, episode in episodes.items():
        points = labels.get(eid)
        if not points:
            continue
        graded = dict(points)
        first = min(graded)
        current = None
        for k in range(episode["n_steps"]):
            if k in graded:
                current = graded[k]
            if current is None or k < first:
                continue  # before the first verified point the label is unknown, not zero
            if not carry and k not in graded:
                continue
            if stride > 1 and k % stride and k not in graded:
                continue
            cut = k + 1
            record = dict(episode["record"])
            record["generations"] = record["generations"][:cut]
            out[f"{eid}@{k}"] = {
                "id": f"{eid}@{k}", "episode": eid, "harness": episode["harness"],
                "score": float(current), "label": int(current > 0.5),
                "signals": {name: list(seq)[:cut] for name, seq in episode["signals"].items()},
                "critics": episode["critics"][:cut],
                "episode_critics": episode["episode_critics"],
                "judge_steps": episode["judge_steps"][:cut],
                "mte_steps": episode["mte_steps"][:cut], "saup": episode["saup"][:cut],
                "uprop": episode.get("uprop"),
                "cross_critics": episode.get("cross_critics"),
                "htc_tokens": tuple(side[:cut] for side in episode["htc_tokens"]),
                "n_steps": cut, "tool_rate": episode["tool_rate"], "record": record,
            }
    return out


def folds_by_episode(units: dict[str, dict], seed: int) -> list[list[str]]:
    """Split episodes, then expand each side to its prefixes."""
    by_episode: dict[str, list[str]] = defaultdict(list)
    for key, unit in units.items():
        by_episode[unit["episode"]].append(key)
    episodes = sorted(by_episode)
    strata = {e: int(st.fmean(units[k]["score"] for k in by_episode[e]) > 0.5) for e in episodes}
    rng = np.random.RandomState(seed)
    folds: list[list[str]] = [[] for _ in range(FOLDS)]
    for value in (0, 1):
        members = [e for e in episodes if strata[e] == value]
        rng.shuffle(members)
        for position, episode in enumerate(members):
            folds[position % FOLDS] += by_episode[episode]
    return folds


def oof_by_episode(units: dict[str, dict], fn, seed: int) -> float | None:
    pred: dict[str, float] = {}
    for fold in folds_by_episode(units, seed):
        held = set(fold)
        train = with_labels([units[i] for i in units if i not in held], True)
        for key, value in zip(fold, fn(train, [units[i] for i in fold])):
            if value is None:
                return None
            pred[key] = float(value)
    keys = list(units)
    return prr([units[k]["score"] for k in keys], [pred[k] for k in keys])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True, help="directory of labels_<cohort>.jsonl")
    p.add_argument("--toolkit", required=True)
    p.add_argument("--cohorts", nargs="+", default=list(RUNS))
    p.add_argument("--carry", action="store_true",
                   help="carry a verified label forward to the steps that did not change the "
                        "repository, which is exact rather than an approximation")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--methods", nargs="+", default=None, help="restrict to these method labels")
    a = p.parse_args()

    methods = method_table(a.toolkit)
    wanted = [k for k in methods if a.methods is None or any(m in str(k) for m in a.methods)]
    results: dict[tuple, dict[str, float]] = defaultdict(dict)
    for cohort in a.cohorts:
        episodes = load_deepswe(a.runs / RUNS[cohort] / "compact.jsonl", "mini-swe-agent")
        labels = read_labels(a.labels / f"labels_{cohort}.jsonl")
        units = prefixes(episodes, labels, a.carry, a.stride)
        covered = len({u["episode"] for u in units.values()})
        print(f"[prefix] {cohort}: {len(units)} scored prefixes over {covered} episodes "
              f"(of {len(episodes)}), score mean {st.fmean(u['score'] for u in units.values()):.2f}",
              flush=True)
        if len(units) < 20:
            continue
        for key in wanted:
            values = [v for s in SEEDS if (v := oof_by_episode(units, methods[key], s)) is not None]
            if values:
                results[key][cohort] = st.fmean(values)

    print(f"\n{'method':62} " + " ".join(f"{c:>7}" for c in a.cohorts) + f" {'Avg':>7}")
    print("-" * 96)
    for key in sorted(results, key=lambda k: -st.fmean(results[k].values())):
        row = results[key]
        if len(row) < len(a.cohorts):
            continue
        name = f"{key[1]} / {key[2]}"[:62]
        print(f"{name:62} " + " ".join(f"{row[c]:+7.2f}" for c in a.cohorts)
              + f" {st.fmean(row.values()):+7.2f}")


if __name__ == "__main__":
    main()
