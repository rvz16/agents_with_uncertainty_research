"""Build the OQ cohort's two training files: the text view and the distilled targets.

Views mirror alfworld_uq/data/judge_views/<cohort>.jsonl -- {id, task, steps:
[{action, observation}]} -- and targets mirror data/distill_targets/targets_<cohort>.jsonl.
Nothing about the trainer changes; OSWorld just has to arrive in the same shape.

Three decisions are made here, and each is a place a reviewer will look:

* **The observation is synthesized.** OSWorld's reply to an action is a PNG, so
  the right-hand side of our template is rendered from what the collector already
  measured per step: whether the action executed, whether the screen changed and
  by how much, and the repetition flags. Text in, text out -- the modality stays
  identical to the other seven cohorts, so a transfer failure is about the domain
  and not about pixels.
* **Steps are deduped by ``step_num``.** traj.jsonl re-emits a step when the
  harness retried it; the last row is the one that ran. After dedupe the
  trajectory matches uq_episode.json exactly on 375 of 384 episodes and is short
  by one or two on the rest, which the trainer already tolerates.
* **``stale_share`` is left empty.** ALFWorld's stale label means "resting on a
  check the agent has since invalidated", and no GUI analogue is honest here;
  inventing one would quietly redefine what head 2 predicts. ``repeat_share``
  does transfer: the collector flags repeated actions and action cycles.
"""
from __future__ import annotations

import argparse, os, json, statistics as st, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLKIT = Path(os.environ.get("UQ_TOOLKIT", ""))
if TOOLKIT:
    sys.path.insert(0, str(TOOLKIT))

from experiments.distill_targets import TEACHERS, teacher_curves   # noqa: E402
from experiments.prr_report_v2 import SEEDS, method_table          # noqa: E402
from experiments.osworld.load_osworld import load_osworld                              # noqa: E402

RUNS = ["full-2gpu-max100-20260911", "full-recovery-infra-20260912",
        "affected-recovery-20260915.IFxOYC"]


def episode_dirs(raw: Path) -> dict[str, Path]:
    """task id -> its directory, later runs winning, matching load_osworld's dedupe."""
    out: dict[str, Path] = {}
    for run in RUNS:
        for traj in (raw / run).rglob("traj.jsonl"):
            out[traj.parent.name] = traj.parent
    return out


def steps_of(directory: Path) -> list[dict]:
    rows = [json.loads(l) for l in open(directory / "traj.jsonl") if l.strip()]
    last: dict[int, dict] = {}
    for r in rows:
        last[int(r["step_num"])] = r          # a retried step keeps the row that ran
    return [last[k] for k in sorted(last)]


def action_text(row: dict) -> str:
    """The agent's stated intent plus the code it actually ran."""
    response = (row.get("response") or "").strip()
    intent = next((l.strip()[len("Action:"):].strip() for l in response.splitlines()
                   if l.strip().startswith("Action:")), "")
    code = (row.get("action") or "").strip().replace("\n", " ")
    if intent and code:
        return f"{intent} -> {code}"
    return intent or code or "(no action)"


def observation_text(gen: dict) -> str:
    critics = gen.get("critics") or {}
    meta = gen.get("metadata") or {}
    delta = (gen.get("signals") or {}).get("state_change_score")
    parts = ["executed" if critics.get("ui_execution_success") else "did not execute"]
    if critics.get("state_change"):
        parts.append(f"screen changed (delta {delta:.3f})" if delta is not None else "screen changed")
    else:
        parts.append(f"screen unchanged (delta {delta:.3f})" if delta is not None else "screen unchanged")
    if not critics.get("action_format_valid"):
        parts.append("malformed action")
    if not critics.get("no_stagnation"):
        parts.append("stagnating")
    if meta.get("repeated_action"):
        parts.append("repeats an earlier action")
    if meta.get("action_cycle"):
        parts.append(f"in an action cycle of length {meta.get('cycle_length')}")
    if meta.get("no_effect_loop"):
        parts.append("in a loop with no effect")
    return "; ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path(os.environ.get("OSWORLD_RAW", HERE / "raw")))
    p.add_argument("--out-views", type=Path, required=True)
    p.add_argument("--out-targets", type=Path, required=True)
    p.add_argument("--teacher", default="b4")
    a = p.parse_args()

    episodes = load_osworld([a.raw / r / "osworld-qwen-100" for r in RUNS])
    instructions = json.loads((HERE / "osworld_instructions.json").read_text())
    dirs = episode_dirs(a.raw)
    raw_gens = {}
    for run in RUNS:
        for line in open(a.raw / run / "osworld-qwen-100" / "uq" / "episodes.jsonl"):
            if line.strip():
                row = json.loads(line)
                raw_gens[row["episode_id"]] = row.get("generations") or []

    a.out_views.parent.mkdir(parents=True, exist_ok=True)
    a.out_targets.parent.mkdir(parents=True, exist_ok=True)
    kept, dropped, lengths = {}, 0, []
    with open(a.out_views, "w") as handle:
        for eid, episode in episodes.items():
            directory, task = dirs.get(eid), instructions.get(eid, "")
            if not directory or not task or task.startswith("__error__"):
                dropped += 1
                continue
            rows, gens = steps_of(directory), raw_gens[eid]
            n = min(len(rows), len(gens))
            if not n:
                dropped += 1
                continue
            steps = [{"action": action_text(rows[k]), "observation": observation_text(gens[k])}
                     for k in range(n)]
            handle.write(json.dumps({"id": eid, "task": task, "steps": steps}) + "\n")
            kept[eid] = n
            lengths.append(n)
    print(f"[views] {len(kept)} episodes, {sum(lengths)} steps, median {st.median(lengths):.0f} "
          f"({dropped} dropped)  -> {a.out_views}")

    episodes = {k: v for k, v in episodes.items() if k in kept}
    methods = method_table(str(TOOLKIT))
    per_seed = [teacher_curves(episodes, TEACHERS[a.teacher], methods, seed) for seed in SEEDS]
    written = 0
    with open(a.out_targets, "w") as handle:
        for eid, episode in episodes.items():
            curves = [s[eid] for s in per_seed if eid in s]
            if not curves:
                continue
            repeats = [r or c for r, c in zip(episode["repeated"], episode["cycles"])]
            for k in range(min(kept[eid], episode["n_steps"])):
                values = [c[k] for c in curves if k < len(c)]
                if not values:
                    continue
                handle.write(json.dumps({
                    "id": eid, "cohort": "OQ", "step": k,
                    "posterior": st.fmean(values),
                    "label": episode["label"], "score": episode["score"],
                    "repeat_share": st.fmean(float(x) for x in repeats[: k + 1]),
                    "stale_share": None}) + "\n")
                written += 1
    print(f"[targets] {written} prefix targets from {len(episodes)} episodes -> {a.out_targets}")


if __name__ == "__main__":
    main()
