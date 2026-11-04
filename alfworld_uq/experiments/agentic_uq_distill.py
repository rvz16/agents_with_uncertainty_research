"""Training inputs for the prefix student from the colleague's five code benchmarks.

Writes the two files `train_decision.py` reads, one cohort per benchmark:
`judge_views/<CODE>.jsonl` (the text the student sees) and
`distill_targets/targets_<CODE>.jsonl` (the teacher's posterior at every prefix).

Three things about this release shape the build, and all three are limits worth
stating next to any number that comes out of it:

* **There is no trajectory text.** Prompts, tool observations and judge feedback
  were never exported; the only text is the candidate code. So `action` is the
  candidate, `observation` is synthesized from the critic verdicts -- the same
  device the OSWorld plan needs, where the real observation is a screenshot --
  and `task` is empty, because the prompt is recoverable only from the upstream
  public benchmark by `instance_id`.
* **31 991 episodes are 1 310 tasks.** Arms pool into one cohort for training,
  but the fold key is the task, never the arm or the episode, or the 24 copies of
  a task would straddle the split.
* **Critic levels are an escalation**, so each enters as `seen`/`pass`; a gap is
  not a failed verdict.

Usage:
  agentic_uq_distill.py --data <snapshot> --toolkit <agentic-uq/src> --benchmark bcb
                        --views data/judge_views --targets data/distill_targets
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.agentic_uq_table import LEVELS, arms, critic_pair, load_arm
from experiments.distill_targets import TEACHERS
from experiments.prr_report_v2 import FOLDS, SEEDS, is_continuous, method_table, with_labels

#: one cohort code per benchmark, to add to train_decision.COHORTS
CODES = {"bcb": "CB", "lcb": "CL", "cc": "CC", "swe": "CS", "poly": "CP"}


def prefix_records(episode: dict, cut: int) -> dict:
    """The episode after ``cut`` steps. None-safe, and the critics are cut too."""
    record = dict(episode["record"])
    record["generations"] = record["generations"][:cut]
    crit = episode["critics"][:cut]
    return dict(episode, record=record, n_steps=cut,
                signals={k: list(v)[:cut] for k, v in episode["signals"].items()},
                critics=crit, episode_critics=dict(crit[-1]) if crit else {},
                judge_steps=episode["judge_steps"][:cut],
                mte_steps=episode["mte_steps"][:cut], saup=episode["saup"][:cut],
                htc_tokens=None)


def folds_by_task(episodes: dict[str, dict], seed: int) -> list[list[str]]:
    """Split on the task, then expand to every arm that ran it."""
    by_task: dict[str, list[str]] = defaultdict(list)
    for key, episode in episodes.items():
        by_task[episode["task_key"]].append(key)
    tasks = sorted(by_task)
    strata = {t: int(st.fmean(episodes[k]["score"] for k in by_task[t]) > 0.5) for t in tasks}
    rng = np.random.RandomState(seed)
    folds: list[list[str]] = [[] for _ in range(FOLDS)]
    for value in (0, 1):
        members = [t for t in tasks if strata[t] == value]
        rng.shuffle(members)
        for position, task in enumerate(members):
            folds[position % FOLDS] += by_task[task]
    return folds


def teacher_curves(episodes: dict[str, dict], key, methods, seed: int) -> dict[str, list[float]]:
    """The teacher at every prefix, out of fold, with the task-level split."""
    fn = methods[key]
    continuous = is_continuous(episodes)
    out: dict[str, list[float]] = {}
    for fold in folds_by_task(episodes, seed):
        held = set(fold)
        train = with_labels([episodes[i] for i in episodes if i not in held], continuous)
        for i in fold:
            episode = episodes[i]
            cuts = [prefix_records(episode, k + 1) for k in range(episode["n_steps"])]
            values = fn(train, cuts)
            out[i] = [float(v) if v is not None else float("nan") for v in values]
    return out


def observation(critics: dict[str, bool]) -> str:
    """The right-hand side of the text template, built from the critic verdicts.

    The release exports no tool output, so this is what the student can be told
    about what happened: which checks ran and how they came back.
    """
    parts = []
    for level in LEVELS:
        name = level.replace("critic_", "")
        if critics.get(f"{level}_seen"):
            parts.append(f"{name}=pass" if critics.get(f"{level}_pass") else f"{name}=FAIL")
        else:
            parts.append(f"{name}=not run")
    return "checks: " + ", ".join(parts)


def code_text(data: Path, bench: str) -> dict[tuple[str, str], list[str]]:
    """(arm tag, instance) -> candidate code per generation, from episodes.jsonl."""
    out: dict[tuple[str, str], list[str]] = {}
    for path in sorted(data.glob(f"benchmarks/{bench}/arms/*/*/*/episodes.jsonl")):
        tag = str(path).split("/")[-2]
        for line in open(path):
            if not line.strip():
                continue
            row = json.loads(line)
            out[(tag, row["instance_id"])] = [
                (g.get("code") or "").strip() for g in (row.get("generations") or [])]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--toolkit", required=True)
    p.add_argument("--benchmark", required=True, choices=sorted(CODES))
    p.add_argument("--views", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--teacher", default="b4", choices=sorted(TEACHERS))
    p.add_argument("--act-chars", type=int, default=600)
    a = p.parse_args()

    episodes: dict[str, dict] = {}
    for bench, model, harness, tag, path in arms(a.data, [a.benchmark]):
        for instance, episode in load_arm(path, harness).items():
            episode["task_key"] = instance
            episode["arm"] = tag
            episodes[f"{tag}::{instance}"] = episode
    tasks = len({e["task_key"] for e in episodes.values()})
    print(f"[{a.benchmark}] {len(episodes)} episodes over {tasks} tasks", flush=True)
    if len({e["score"] for e in episodes.values()}) < 2:
        raise SystemExit("one outcome only: nothing to fit")

    code = code_text(a.data, a.benchmark)
    cohort = CODES[a.benchmark]
    a.views.mkdir(parents=True, exist_ok=True)
    a.targets.mkdir(parents=True, exist_ok=True)

    view_path = a.views / f"{cohort}.jsonl"
    missing = 0
    with open(view_path, "w") as handle:
        for key, episode in episodes.items():
            candidates = code.get((episode["arm"], episode["task_key"]), [])
            if not candidates:
                missing += 1
            steps = [{"action": (candidates[k] if k < len(candidates) else "")[:a.act_chars],
                      "observation": observation(episode["critics"][k])}
                     for k in range(episode["n_steps"])]
            handle.write(json.dumps({"id": key, "task": "", "steps": steps}) + "\n")
    print(f"[{a.benchmark}] views -> {view_path} ({missing} episodes with no candidate text)",
          flush=True)

    methods = method_table(a.toolkit)
    per_seed = [teacher_curves(episodes, TEACHERS[a.teacher], methods, seed) for seed in SEEDS]
    target_path = a.targets / f"targets_{cohort}.jsonl"
    rows = 0
    with open(target_path, "w") as handle:
        for key, episode in episodes.items():
            curves = [s[key] for s in per_seed if key in s]
            if not curves:
                continue
            for k in range(episode["n_steps"]):
                handle.write(json.dumps({
                    "id": key, "cohort": cohort, "step": k,
                    "posterior": st.fmean(c[k] for c in curves if k < len(c)),
                    "label": int(episode["score"]), "score": episode["score"],
                    "repeat_share": None, "stale_share": None}) + "\n")
                rows += 1
    print(f"[{a.benchmark}] {rows} prefix targets -> {target_path}", flush=True)


if __name__ == "__main__":
    main()
