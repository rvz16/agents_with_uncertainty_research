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

from experiments.deepswe_table import _TEST_CMD
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


def _cut(seq, cut):
    """Slice a per-step sequence, keeping a missing one missing.

    DeepSWE runs carry no judge or HTC sidecar, so those arrive as None. A
    method that reads them sees None here exactly as it would for a whole
    episode, rather than the prefix cutter failing on the first one.
    """
    return None if seq is None else seq[:cut]


def step_primitives(path: Path) -> dict[str, list[tuple[str | None, int | None, bool]]]:
    """Per-step command / returncode / format error, sliced as load_deepswe slices.

    The critics have to be rebuilt from the prefix's own steps, and the episode
    dict does not keep the raw steps, so they are read again here. The last step
    is dropped exactly as there, or the indices would not line up with the cut.
    """
    out: dict[str, list[tuple[str | None, int | None, bool]]] = {}
    for line in open(path):
        row = json.loads(line)
        steps = row["steps"]
        if len(steps) > 1:
            steps = steps[:-1]  # pre-terminal, as in load_deepswe
        out[row["id"]] = [(s["command"], s["returncode"], bool(s["format_error"])) for s in steps]
    return out


def critics_at(steps: list[tuple[str | None, int | None, bool]]) -> dict[str, bool]:
    """The five tool critics over these steps only.

    load_deepswe computes them over the whole episode and stamps the same dict on
    every step, so a prefix that keeps them reads its own future: the last test of
    the episode, a commit that has not happened yet. Same definitions, prefix scope.
    """
    commands = [c for c, _, _ in steps if c]
    test_rcs = [rc for c, rc, _ in steps if c and _TEST_CMD.search(c) and rc is not None]
    first_fail = next((k for k, rc in enumerate(test_rcs) if rc != 0), None)
    return {
        "ran_tests": len(test_rcs) > 0,
        "ran_tests_twice": len(test_rcs) >= 2,
        "last_test_passed": bool(test_rcs) and test_rcs[-1] == 0,
        "test_flipped": first_fail is not None and any(rc == 0 for rc in test_rcs[first_fail:]),
        "committed_twice": sum("git commit" in c for c in commands) >= 2,
        "no_format_errors": not any(fe for _, _, fe in steps),
    }


def tool_rate_at(steps: list[tuple[str | None, int | None, bool]]) -> float:
    """Return-code-0 rate over these steps only, not over the episode."""
    observed = [rc == 0 for _, rc, _ in steps if rc is not None]
    return st.fmean(observed) if observed else 0.0


def prefixes(episodes: dict[str, dict], labels: dict[str, list[tuple[int, float]]],
             carry: bool, stride: int,
             primitives: dict[str, list[tuple[str | None, int | None, bool]]] | None = None,
             uncut_critics: bool = False) -> dict[str, dict]:
    """One scored unit per label point, with every sequence cut at that step.

    ``primitives`` lets the tool critics and the tool-success rate be rebuilt from
    the prefix's own steps. Without it (or with ``uncut_critics``) they keep the
    whole-episode values load_deepswe produced, which is hindsight: that is what
    the 2026-11-04 correction is about, and the flag exists only to reproduce the
    earlier numbers.
    """
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
            prefix_steps = None if primitives is None else primitives.get(eid, [])[:cut]
            honest = prefix_steps is not None and not uncut_critics
            crit = critics_at(prefix_steps) if honest else None
            record = dict(episode["record"])
            generations = record["generations"][:cut]
            if honest:
                # new dicts: the generations are shared with every other prefix
                generations = [{**g, "critics": dict(crit)} for g in generations]
            record["generations"] = generations
            out[f"{eid}@{k}"] = {
                "id": f"{eid}@{k}", "episode": eid, "harness": episode["harness"],
                "score": float(current), "label": int(current > 0.5),
                "signals": {name: list(seq)[:cut] for name, seq in episode["signals"].items()},
                "critics": ([dict(crit) for _ in range(cut)] if honest
                            else _cut(episode["critics"], cut)),
                "episode_critics": dict(crit) if honest else episode["episode_critics"],
                "judge_steps": _cut(episode["judge_steps"], cut),
                "mte_steps": _cut(episode["mte_steps"], cut), "saup": _cut(episode["saup"], cut),
                # an episode-level decomposition cannot be cut, so it is dropped
                # rather than carried into a prefix as a constant
                "uprop": None if honest else episode.get("uprop"),
                "cross_critics": _cut(episode.get("cross_critics"), cut),
                "htc_tokens": (None if episode["htc_tokens"] is None
                               else tuple(side[:cut] for side in episode["htc_tokens"])),
                "n_steps": cut,
                "tool_rate": tool_rate_at(prefix_steps) if honest else episode["tool_rate"],
                "record": record,
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
    p.add_argument("--uncut-critics", action="store_true",
                   help="keep load_deepswe's whole-episode critics and tool rate on every "
                        "prefix, as the tables before 2026-11-04 did; hindsight, for "
                        "reproducing those numbers only")
    a = p.parse_args()

    methods = method_table(a.toolkit)
    wanted = [k for k in methods if a.methods is None or any(m in str(k) for m in a.methods)]
    results: dict[tuple, dict[str, float]] = defaultdict(dict)
    for cohort in a.cohorts:
        episodes = load_deepswe(a.runs / RUNS[cohort] / "compact.jsonl", "mini-swe-agent")
        labels = read_labels(a.labels / f"labels_{cohort}.jsonl")
        primitives = step_primitives(a.runs / RUNS[cohort] / "compact.jsonl")
        units = prefixes(episodes, labels, a.carry, a.stride, primitives, a.uncut_critics)
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
