"""The DeepSWE columns of the results table, built the way the ALFWorld ones are.

Same estimators -- CriticBayesState, ContinuousBayesUQ, BinaryBayesUQ(sep),
PRR@0.5 over 20 calibration/test splits -- on a different question. A binary
"solved" label is degenerate here (1 of 113 for gpt-oss, 0 for Qwen), so each
model is scored on the outcome its runs actually vary in:

  gpt-oss  "did the patch leave the repository working?"  (p2p > 0)
           among non-empty patches that passed no F2P test: 31 vs 22
  Qwen     "did the patch make any progress?"              (f2p > 0)
           among non-empty patches: 49 vs 11

Critics are the DeepSWE analogues of ours: no format errors, submitted,
ran tests, committed, no command repeated three times. Tool success rate is
the fraction of bash commands returning 0. MTE and verbalised confidence
were not collected on DeepSWE and are reported as absent, not as zero.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics as st
from pathlib import Path
from typing import Any

from belief.binary_bayes import BinaryBayesUQ
from belief.continuous_bayes import ContinuousBayesUQ
from belief.critic_bayes import CriticBayesState
from experiments.analyze_trajectories import (
    _predict_from_belief,
    _split_ids,
    metric_values,
    prediction_rejection_area,
    _prr_references,
)


def _commands(tr: dict[str, Any]) -> list[str]:
    out = []
    for m in tr.get("messages", []):
        msg = ((m.get("extra") or {}).get("response", {}) or {}).get("choices", [{}])[0].get("message", {}) or {}
        for tc in msg.get("tool_calls") or []:
            try:
                c = json.loads(tc["function"]["arguments"]).get("command", "")
                out.append(c if isinstance(c, str) else " ".join(map(str, c)))
            except Exception:  # noqa: BLE001
                pass
    return out


def load(run: Path, keep) -> list[dict[str, Any]]:
    rows = []
    for d in sorted(p for p in run.glob("*") if p.is_dir()):
        res, tp = d / "result.json", d / "agent/mini-swe-agent.trajectory.json"
        if not res.exists() or not tp.exists():
            continue
        vr = (json.load(open(res)).get("verifier_result") or {}).get("rewards") or {}
        if not vr:
            continue
        patch = d / "artifacts/model.patch"
        size = patch.stat().st_size if patch.exists() else 0
        tr = json.load(open(tp)); info = tr.get("info", {}); msgs = tr.get("messages", [])
        steps, rcs = [], []
        for m in msgs:
            ch = ((m.get("extra") or {}).get("response", {}) or {}).get("choices", [{}])[0]
            v = [t["logprob"] for t in ((ch.get("logprobs") or {}).get("content") or []) if t.get("logprob") is not None]
            if v:
                steps.append(sum(v) / len(v))
            if m.get("role") == "tool":
                try:
                    rcs.append(int(json.loads(m.get("content") or "{}").get("returncode", 1)) == 0)
                except Exception:  # noqa: BLE001
                    pass
        if not steps:
            continue
        cmds = _commands(tr); counts = collections.Counter(cmds)
        item = {
            "id": d.name, "size": size,
            "f2p": vr.get("f2p_passed", 0) / max(vr.get("f2p_total", 1), 1),
            "p2p": vr.get("p2p_passed", 0) / max(vr.get("p2p_total", 1), 1),
            "steps": steps,
            "tool_success": (sum(rcs) / len(rcs)) if rcs else 0.0,
            "critics": {
                "no_format_errors": not any("Tool call error" in str(m.get("content") or "") for m in msgs),
                "submitted": info.get("exit_status") == "Submitted",
                "ran_tests": any(("test" in c or "pytest" in c) for c in cmds),
                "committed": any("git commit" in c for c in cmds),
                "no_repeated_command": all(v < 3 for v in counts.values()) if counts else False,
            },
        }
        if keep(item):
            rows.append(item)
    return rows


def load_compact(path: Path, keep) -> list[dict[str, Any]]:
    """Records written by deep_swe_uq/experiments/compact_jobs.py (top_logprobs runs)."""
    rows = []
    for line in open(path):
        r = json.loads(line); rw = r["rewards"]; steps = r["steps"]
        lp = [s["mean_logprob"] for s in steps if s["mean_logprob"] is not None]
        if not rw or not lp:
            continue
        ent = [s["mean_entropy"] for s in steps if s["mean_entropy"] is not None]
        rcs = [s["returncode"] == 0 for s in steps if s["returncode"] is not None]
        cmds = [s["command"] for s in steps if s["command"]]; counts = collections.Counter(cmds)
        item = {
            "id": r["id"], "size": r["patch_bytes"],
            "f2p": rw.get("f2p_passed", 0) / max(rw.get("f2p_total", 1), 1),
            "p2p": rw.get("p2p_passed", 0) / max(rw.get("p2p_total", 1), 1),
            "steps": lp, "entropies": ent or None,
            "tool_success": (sum(rcs) / len(rcs)) if rcs else 0.0,
            "critics": {
                "no_format_errors": not any(s["format_error"] for s in steps),
                "submitted": r["exit_status"] == "Submitted",
                "ran_tests": any(("test" in c or "pytest" in c) for c in cmds),
                "committed": any("git commit" in c for c in cmds),
                "no_repeated_command": all(v < 3 for v in counts.values()) if counts else False,
            },
        }
        if keep(item):
            rows.append(item)
    return rows


def prr(y, conf):
    a = prediction_rejection_area([-c for c in conf], y, 0.5); o, r = _prr_references(tuple(y), 0.5)
    return None if None in (a, o, r) or abs(o - r) < 1e-9 else (a - r) / (o - r)


def column(rows: list[dict[str, Any]], label_fn, seeds: int) -> dict[str, float | None]:
    by = {r["id"]: r for r in rows}; ids = list(by); y = {i: label_fn(by[i]) for i in ids}
    raw = {
        "Logprob (mean)": lambda r: st.fmean(r["steps"]),
        "Perplexity (max)": lambda r: -max(math.exp(-s) for s in r["steps"]),
        "MTE (max)": lambda r: -max(r["entropies"]) if r.get("entropies") else None,
        "Tool success rate": lambda r: r["tool_success"],
        "Length-only baseline": lambda r: -float(len(r["steps"])),
    }
    out: dict[str, float | None] = {"MTE (max)": None, "Verbalized UQ (final)": None}
    acc = collections.defaultdict(list)
    for seed in range(seeds):
        cal, test = _split_ids(ids, 0.5, seed)
        yc = [y[i] for i in cal]; yt = [y[i] for i in test]
        crit = CriticBayesState.fit([by[i]["critics"] for i in cal], yc)
        cont = ContinuousBayesUQ.fit([by[i]["steps"] for i in cal], yc, lambda_=1.0)
        sep = BinaryBayesUQ.fit([by[i]["steps"] for i in cal], yc, threshold_mode="sep", higher_is_uncertain=False)
        for k, f in raw.items():  # same test halves as the Bayes rows
            conf = [f(by[i]) for i in test]
            v = None if any(c is None for c in conf) else prr(yt, conf)
            if v is not None:
                acc[k].append(v)
        belief = {i: crit.predict(by[i]["critics"]) for i in test}
        variants = {
            "Bayes tool-only": [belief[i] for i in test],
            "Bayes UQ-only (cont.)": [cont.predict(by[i]["steps"]) for i in test],
            "Bayes Fused (cont.)": [_predict_from_belief(cont, by[i]["steps"], belief[i]) for i in test],
            "Bayes Fused (SEP)": [_predict_from_belief(sep, by[i]["steps"], belief[i]) for i in test],
        }
        for k, p in variants.items():
            v = metric_values(yt, p).get("prr_at_0_5")
            if v is not None:
                acc[k].append(v)
    for k, v in acc.items():
        out[k] = st.fmean(v)
    out["n"] = len(ids); out["positives"] = sum(y.values())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gptoss", type=Path, required=True)
    p.add_argument("--qwen", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--unified", action="store_true", help="score both models on F2P>0 among non-empty patches")
    a = p.parse_args()
    def read(path: Path, keep):
        return load_compact(path, keep) if path.suffix == ".jsonl" else load(path, keep)

    nonempty = lambda r: r["size"] > 0  # noqa: E731
    progress = lambda r: int(r["f2p"] > 0)  # noqa: E731
    if a.unified:
        # Both models scored on the same question: did a non-empty patch pass any F2P test?
        cols = {
            "gpt-oss (any progress)": column(read(a.gptoss, nonempty), progress, a.seeds),
            "Qwen (any progress)": column(read(a.qwen, nonempty), progress, a.seeds),
        }
    else:
        cols = {
            "gpt-oss (repo intact)": column(read(a.gptoss, lambda r: r["size"] > 0 and r["f2p"] == 0), lambda r: int(r["p2p"] > 0), a.seeds),
            "Qwen (any progress)": column(read(a.qwen, nonempty), progress, a.seeds),
        }
    order = ["Logprob (mean)", "Perplexity (max)", "MTE (max)", "Verbalized UQ (final)", "Tool success rate",
             "Bayes tool-only", "Bayes UQ-only (cont.)", "Bayes Fused (cont.)", "Bayes Fused (SEP)", "Length-only baseline"]
    print(f"{'method':24s}" + "".join(f"{c:>24s}" for c in cols))
    for k in order:
        print(f"{k:24s}" + "".join((f"{cols[c][k]:+24.3f}" if cols[c].get(k) is not None else f"{'NA':>24s}") for c in cols))
    print(f"{'tasks (positives)':24s}" + "".join(f"{str(cols[c]['n'])+' ('+str(cols[c]['positives'])+')':>24s}" for c in cols))


if __name__ == "__main__":
    main()
