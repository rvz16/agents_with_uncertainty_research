"""The method table on the colleague's five code benchmarks, one arm at a time.

`Alexander4127/agentic_uq_trajectories` ships `temporal_belief_episodes.jsonl` in
the toolkit's own schema, so the methods need a loader rather than a conversion.
Two properties of that release decide the protocol:

* 31 991 episodes are only 1 310 unique tasks -- 24.4 arms per task on average.
  The card is explicit: "the same task can occur in many arms. Do not treat those
  repetitions as independent tasks." Scoring **per arm** keeps every occurrence of
  a task inside one partition, because `instance_id` is unique within an arm, so
  folds by episode are folds by task. Pooling arms would both leak and, in the
  card's words, "change the experiment".
* The critic levels are an escalation, not a fixed panel: 27% of generations carry
  no critic at all and the present set differs between two generations of the same
  episode. A missing level means "not evaluated", which is not the same as
  "failed", so each level enters as a pair -- `L{i}_seen` and `L{i}_pass` -- giving
  every record the same critic keys without reading a gap as a verdict.

The label is the terminal binary outcome, so `is_continuous` is false and the
trained methods take the score itself rather than a median split; there are no
prefixes here and therefore none of the prefix hindsight the 2026-11-04
correction is about.

Usage:
  agentic_uq_table.py --data <snapshot> --toolkit <agentic-uq/src> [--benchmarks bcb cc]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from experiments.prr_report_v2 import SEEDS, SIGNALS, method_table, oof

LEVELS = ("critic_L0", "critic_L1", "critic_L2", "critic_L3")


def critic_pair(critics: dict | None) -> dict[str, bool]:
    """One `seen` / `pass` pair per level, so every record has the same keys."""
    critics = critics or {}
    out: dict[str, bool] = {}
    for level in LEVELS:
        seen = level in critics
        out[f"{level}_seen"] = seen
        out[f"{level}_pass"] = bool(critics[level]) if seen else False
    return out


def load_arm(path: Path, harness: str) -> dict[str, dict]:
    """One arm's finished cohort, in the shape the methods already expect."""
    episodes: dict[str, dict] = {}
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        gens = row.get("generations") or []
        if not gens or row.get("success") is None:
            continue
        per_step = [dict(g.get("signals") or {}) for g in gens]
        crit = [critic_pair(g.get("critics")) for g in gens]

        def seq(key: str) -> list[float]:
            return [float(s[key]) for s in per_step if s.get(key) is not None]

        seqs = {name: [] for name in SIGNALS}
        seqs["Logprob"] = seq("mean_token_logprob")
        seqs["Perplexity"] = seq("perplexity")
        seqs["MTE"] = seq("mean_token_entropy")
        seqs["Self-certainty"] = seq("self_certainty")
        if not seqs["Logprob"]:
            continue
        record_gens = [
            {"index": g.get("index", k), "critics": dict(crit[k]),
             "signals": {name: float(value) for name, value in (
                 ("mean_logprob", s.get("mean_token_logprob")),
                 ("perplexity", s.get("perplexity")),
                 ("mean_entropy", s.get("mean_token_entropy")),
                 ("self_certainty", s.get("self_certainty")),
             ) if value is not None}}
            for k, (g, s) in enumerate(zip(gens, per_step))
        ]
        passed = [c[f"{level}_pass"] for c in crit for level in LEVELS if c[f"{level}_seen"]]
        episodes[row["episode_id"]] = {
            "id": row["episode_id"], "harness": harness,
            "score": float(bool(row["success"])), "label": 0,
            "signals": seqs, "critics": crit, "episode_critics": dict(crit[-1]),
            "cross_critics": None, "judge_steps": [None] * len(gens),
            "mte_steps": seqs["MTE"] or [None] * len(gens),
            "saup": [None] * len(gens), "uprop": None, "htc_tokens": None,
            "n_steps": len(gens),
            "tool_rate": st.fmean(passed) if passed else 0.0,
            "record": {"episode_id": row["episode_id"], "environment": "code",
                       "success": 0, "generations": record_gens},
        }
    return episodes


def arms(data: Path, benchmarks: list[str] | None):
    for path in sorted(data.glob("benchmarks/*/arms/*/*/*/temporal_belief_episodes.jsonl")):
        parts = str(path).split("/benchmarks/")[1].split("/")
        bench, model, harness, tag = parts[0], parts[2], parts[3], parts[4]
        if benchmarks and bench not in benchmarks:
            continue
        yield bench, model, harness, tag, path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--toolkit", required=True)
    p.add_argument("--benchmarks", nargs="+", default=None)
    p.add_argument("--min-episodes", type=int, default=30)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=None)
    a = p.parse_args()

    methods = method_table(a.toolkit)
    wanted = [k for k in methods if a.methods is None or any(m in str(k) for m in a.methods)]
    results: dict[str, dict] = {}
    for bench, model, harness, tag, path in arms(a.data, a.benchmarks):
        eps = load_arm(path, harness)
        if len(eps) < a.min_episodes or len({e["score"] for e in eps.values()}) < 2:
            print(f"[skip] {bench}/{tag}: {len(eps)} episodes", flush=True)
            continue
        row = {}
        for key in wanted:
            values = [v for s in SEEDS if (v := oof(eps, methods[key], s)) is not None]
            if values:
                row[" / ".join(str(part) for part in key[1:])] = round(st.fmean(values), 4)
        results[f"{bench}/{model}/{harness}/{tag}"] = {
            "benchmark": bench, "model": model, "harness": harness, "tag": tag,
            "episodes": len(eps), "success_rate": round(st.fmean(e["score"] for e in eps.values()), 4),
            "prr": row,
        }
        best = max(row.items(), key=lambda kv: kv[1]) if row else ("-", 0.0)
        print(f"[done] {bench}/{tag}: {len(eps)} eps, {len(row)} methods, "
              f"best {best[0]} {best[1]:+.3f}", flush=True)
        a.out.write_text(json.dumps(results, indent=1))
    a.out.write_text(json.dumps(results, indent=1))
    print(f"\n{len(results)} arms -> {a.out}")


if __name__ == "__main__":
    main()
