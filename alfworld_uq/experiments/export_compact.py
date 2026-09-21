"""Export our ALFWorld runs and DeepSWE cohorts in the toolkit's portable episode schema.

One JSONL per run, every record passed through ``trajectory_uq_toolkit.schema.validate_episode``:

* ``episode_id``, ``environment``, ``success`` (ALFWorld: the environment's verdict; DeepSWE:
  the cohort-median split of the verifier's partial score, the score itself is in ``features``);
* ``generations[k] = {index, signals, critics}`` with every numeric per-step UQ value we use
  (ALFWorld: the ``combined`` segment of the response plus, when present, the action-segment
  self-certainty, the prefix LLM judge, the SAUP distances; DeepSWE: mean_logprob, perplexity,
  mean_entropy, self_certainty, verbalized confidence from the side-query replay, judge,
  SAUP distances) and the step critics (ALFWorld: format_valid, action_valid,
  no_repeated_fallback, tool_success, state_changed; DeepSWE: the task-level critics repeated);
* episode-level ``critics`` (ALFWorld: the three all-steps flags; DeepSWE: ran_tests, ...),
  ``features`` (DeepSWE scores) and ``metadata`` (task type, stop reason, step count, cohort).

    python -m experiments.export_compact --toolkit agentic-uq/src --out exports/ \\
        --alfworld react_gptoss_140_giveup_sc runs/react_gptoss_140_giveup_sc ... \\
        --deepswe DG deep_swe_uq/runs/deepswe_gptoss_113_v5/compact.jsonl deep_swe_uq/runs/deepswe_gptoss_113_v5/verb.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from experiments.analyze_trajectories import STEP_CRITIC_NAMES, _critic_observations, _step_critic_observation
from experiments.prr_report_v2 import load_deepswe, read_judge, read_saup


def alfworld_records(run: Path, cohort: str) -> list[dict]:
    rows_by = defaultdict(list)
    for line in open(run / "trajectories.jsonl"):
        if line.strip():
            r = json.loads(line); rows_by[r["episode_id"]].append(r)
    judge = read_judge(run / "judge.jsonl"); saup = read_saup(run / "saup_dist.jsonl")
    out = []
    for line in open(run / "episodes.jsonl"):
        if not line.strip():
            continue
        e = json.loads(line); rows = sorted(rows_by.get(e["episode_id"], []), key=lambda r: int(r.get("step", 0)))
        if not rows:
            continue
        gens = []
        for k, r in enumerate(rows):
            uq = r.get("uq") or {}
            sig = {n: float(v) for n, v in (uq.get("combined") or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))}
            a = (uq.get("action") or {}).get("self_certainty")
            if a is not None and math.isfinite(float(a)):
                sig["self_certainty_action"] = float(a)
            if (e["episode_id"], k) in judge:
                sig["llm_judge"] = judge[(e["episode_id"], k)]
            if (e["episode_id"], k) in saup:
                sig["inquiry_drift"], sig["inference_gap"] = saup[(e["episode_id"], k)]
            crit = {c: bool(_step_critic_observation(r)[c]) for c in STEP_CRITIC_NAMES}
            gens.append({"index": k, "signals": sig, "critics": crit})
        out.append({
            "episode_id": e["episode_id"], "environment": "alfworld", "success": int(bool(e["final_success"])),
            "generations": gens, "critics": {k: bool(v) for k, v in _critic_observations(rows).items()},
            "metadata": {"task_type": e.get("task_type"), "task": e.get("task"), "num_steps": len(rows),
                         "stop_reason": e.get("stop_reason"), "cohort": cohort, "run": run.name},
        })
    return out


def deepswe_records(key: str, compact: Path, verb: Path | None) -> list[dict]:
    eps = load_deepswe(compact, key, verb)
    med = st.median(e["score"] for e in eps.values())
    raw = {json.loads(l)["id"]: json.loads(l) for l in open(compact) if l.strip()}
    out = []
    for eid, e in eps.items():
        gens = []
        for k, g in enumerate(e["record"]["generations"]):
            sig = dict(g["signals"])
            if e.get("judge_steps") and k < len(e["judge_steps"]) and e["judge_steps"][k] is not None:
                sig["llm_judge"] = float(e["judge_steps"][k])
            if e.get("saup") and k < len(e["saup"]) and e["saup"][k] is not None:
                sig["inquiry_drift"], sig["inference_gap"] = e["saup"][k]
            gens.append({"index": k, "signals": sig, "critics": {c: bool(v) for c, v in g["critics"].items()}})
        rw = raw[eid]["rewards"]
        out.append({
            "episode_id": eid, "environment": "deepswe", "success": int(e["score"] > med),
            "generations": gens, "critics": {c: bool(v) for c, v in e["episode_critics"].items()},
            "features": {"deepswe_partial": float(rw.get("partial", 0.0)), "deepswe_f2p": float(rw.get("f2p", 0.0)),
                         "deepswe_reward": float(rw.get("reward", 0.0))},
            "metadata": {"exit_status": raw[eid]["exit_status"], "num_steps": e["n_steps"], "cohort": key,
                         "success_rule": f"partial > cohort median ({med:.3f}); pre-terminal (submit command dropped)"},
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--toolkit", required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--alfworld", nargs=2, action="append", metavar=("NAME", "RUN_DIR"), default=[])
    p.add_argument("--deepswe", nargs=3, action="append", metavar=("KEY", "COMPACT", "VERB"), default=[])
    a = p.parse_args()
    sys.path.insert(0, a.toolkit)
    from trajectory_uq_toolkit.schema import validate_episode
    a.out.mkdir(parents=True, exist_ok=True)
    for name, run in a.alfworld:
        recs = [validate_episode(r) for r in alfworld_records(Path(run), name)]
        with open(a.out / f"alfworld_{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r, separators=(",", ":")) + "\n" for r in recs)
        print(f"alfworld_{name}.jsonl: {len(recs)} episodes, {sum(len(r['generations']) for r in recs)} generations")
    for key, compact, verb in a.deepswe:
        recs = [validate_episode(r) for r in deepswe_records(key, Path(compact), Path(verb) if verb != "-" else None)]
        with open(a.out / f"deepswe_{key}.jsonl", "w") as f:
            f.writelines(json.dumps(r, separators=(",", ":")) + "\n" for r in recs)
        print(f"deepswe_{key}.jsonl: {len(recs)} episodes, {sum(len(r['generations']) for r in recs)} generations")


if __name__ == "__main__":
    main()
