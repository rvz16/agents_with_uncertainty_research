"""The DeepSWE columns of the results table, built the way the ALFWorld ones are.

Same estimators -- CriticBayesState, ContinuousBayesUQ, BinaryBayesUQ(sep),
PRR@0.5 over 20 calibration/test splits -- on a different question. A binary
"solved" label is degenerate here (1 of 113 for gpt-oss, 0 for Qwen), so each
model is scored on the outcome its runs actually vary in:

  gpt-oss  "did the patch leave the repository working?"  (p2p > 0)
           among non-empty patches that passed no F2P test: 31 vs 22
  Qwen     "did the patch make any progress?"              (f2p > 0)
           among non-empty patches: 49 vs 11

Critics are read from the agent's own test runs and commits (see
load_compact): ran the test suite, ran it at least twice, last run passed, a
run that failed then passed, committed at least twice, no format errors. The
process critics we started with (submitted, committed, "ran tests" matched by
the substring "test") were saturated in both classes and carried nothing. Tool success rate is
the fraction of bash commands returning 0. MTE and verbalised confidence
were not collected on DeepSWE and are reported as absent, not as zero.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
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


def load_compact(path: Path, keep, drop_last: bool = False) -> list[dict[str, Any]]:
    """Records written by deep_swe_uq/experiments/compact_jobs.py (top_logprobs runs).

    `drop_last` scores the trajectory before its terminal command, as the
    ALFWorld tables do: on a submitted episode the last command is the submit
    marker, on a context death it is whatever the model was doing.
    """
    rows = []
    for line in open(path):
        r = json.loads(line); rw = r["rewards"]; steps = r["steps"]
        if drop_last and len(steps) > 1:
            steps = steps[:-1]
        lp = [s["mean_logprob"] for s in steps if s["mean_logprob"] is not None]
        if not rw or not lp:
            continue
        ent = [s["mean_entropy"] for s in steps if s["mean_entropy"] is not None]
        conf = [s["confidence"] for s in steps if s.get("confidence") is not None]
        rcs = [s["returncode"] == 0 for s in steps if s["returncode"] is not None]
        cmds = [s["command"] for s in steps if s["command"]]; counts = collections.Counter(cmds)
        # Task-level evidence, from the agent's own test runs (public tests it
        # chose to execute, not the hidden grading tests): did a test command
        # fail and later pass, and did the last one pass?
        test_rcs = [s["returncode"] for s in steps
                    if s["command"] and _TEST_CMD.search(s["command"]) and s["returncode"] is not None]
        flipped = any(rc != 0 for rc in test_rcs) and any(
            rc == 0 for rc in test_rcs[next(i for i, rc in enumerate(test_rcs) if rc != 0):]) if any(rc != 0 for rc in test_rcs) else False
        item = {
            "id": r["id"], "size": r["patch_bytes"],
            "f2p": rw.get("f2p_passed", 0) / max(rw.get("f2p_total", 1), 1),
            "p2p": rw.get("p2p_passed", 0) / max(rw.get("p2p_total", 1), 1),
            "partial": float(rw.get("partial", 0.0)),
            "submitted": r["exit_status"] == "Submitted",
            "steps": lp, "entropies": ent or None,
            "verbalized_final": conf[-1] if conf else None,
            "verbalized_mean": st.fmean(conf) if conf else None,
            "tool_success": (sum(rcs) / len(rcs)) if rcs else 0.0,
            # Critics about the task, not the process. The first set we used
            # (no format errors, submitted, committed, "ran tests" by the
            # substring "test", no repeated command) was saturated at ~100% in
            # both classes: `ls tests/` counted as a test run, and the harness
            # makes everyone commit and submit. What separates outcomes is
            # whether the agent actually ran the test suite, how often, and
            # whether it passed at the end (gpt-oss: 58%/20%, 47%/13%, 38%/9%
            # between the upper and lower half by partial score), plus whether
            # work was committed more than once. Note what these can and
            # cannot see: the agent runs the repository's existing suite, the
            # verifier's F2P tests are new, so these critics measure "kept the
            # repository working" (P2P), not "built the feature".
            "critics": {
                "ran_tests": len(test_rcs) > 0,
                "ran_tests_twice": len(test_rcs) >= 2,
                "last_test_passed": bool(test_rcs) and test_rcs[-1] == 0,
                "test_flipped": bool(flipped),
                "committed_twice": sum("git commit" in c for c in cmds) >= 2,
                "no_format_errors": not any(s["format_error"] for s in steps),
            },
        }
        if keep(item):
            rows.append(item)
    return rows


_TEST_CMD = re.compile(r"\b(pytest|go test|npm test|npx jest|npx vitest|cargo test|yarn test|pnpm test|python -m pytest|python -m unittest|make test)\b")


def prr(y, conf):
    if any(isinstance(v, float) and not float(v).is_integer() for v in y):
        return prr_continuous(y, conf)
    a = prediction_rejection_area([-c for c in conf], y, 0.5); o, r = _prr_references(tuple(y), 0.5)
    return None if None in (a, o, r) or abs(o - r) < 1e-9 else (a - r) / (o - r)


def _area(conf, score, max_rejection=0.5):
    """Mean retained quality while rejecting the least confident half, on a
    continuous score in [0, 1] (the repository PRR truncates labels to int)."""
    import numpy as np
    conf = np.asarray(conf, float); score = np.asarray(score, float)
    ranked = score[np.argsort(-conf)]  # most confident first
    rejected = int(max_rejection * len(ranked))
    if rejected <= 0:
        return None
    cumulative = np.cumsum(ranked); retained = np.arange(1, len(ranked) + 1)
    kept = (cumulative / retained)[len(ranked) - rejected:]
    return float(kept.mean())


def prr_continuous(score, conf):
    """PRR@0.5 against native partial scores, as the OSWorld/WebArena report
    does: no binarisation. Oracle ranks by the true score; the random
    reference is the mean score (its expectation at every rejection level)."""
    import statistics as st
    a = _area(conf, score); o = _area(score, score); r = st.fmean(score)
    return None if a is None or o is None or abs(o - r) < 1e-9 else (a - r) / (o - r)


def column(rows: list[dict[str, Any]], label_fn, seeds: int) -> dict[str, float | None]:
    by = {r["id"]: r for r in rows}; ids = list(by); y = {i: label_fn(by[i]) for i in ids}
    continuous = any(isinstance(v, float) and not float(v).is_integer() for v in y.values())
    # Bayes models need a binary training label; with a continuous score the
    # calibration half is split at its median (the split is refitted per seed).
    def train_labels(cal):
        if not continuous:
            return [int(y[i]) for i in cal]
        med = st.median(y[i] for i in cal)
        return [int(y[i] > med) for i in cal]
    raw = {
        "Logprob (mean)": lambda r: st.fmean(r["steps"]),
        "Perplexity (max)": lambda r: -max(math.exp(-s) for s in r["steps"]),
        "MTE (max)": lambda r: -max(r["entropies"]) if r.get("entropies") else None,
        "Verbalized UQ (final)": lambda r: r.get("verbalized_final"),
        "Verbalized UQ (mean)": lambda r: r.get("verbalized_mean"),
        "Tool success rate": lambda r: r["tool_success"],
        "Length-only baseline": lambda r: -float(len(r["steps"])),
    }
    out: dict[str, float | None] = {"MTE (max)": None, "Verbalized UQ (final)": None, "Verbalized UQ (mean)": None}
    acc = collections.defaultdict(list)
    for seed in range(seeds):
        cal, test = _split_ids(ids, 0.5, seed)
        yc = train_labels(cal); yt = [y[i] for i in test]
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
            v = prr(yt, p) if continuous else metric_values(yt, p).get("prr_at_0_5")
            if v is not None:
                acc[k].append(v)
    for k, v in acc.items():
        out[k] = st.fmean(v)
    out["n"] = len(ids); out["positives"] = (f"mean {st.fmean(y.values()):.2f}" if continuous else sum(y.values()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gptoss", type=Path, required=True)
    p.add_argument("--qwen", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--unified", action="store_true", help="score both models on F2P>0 among non-empty patches")
    p.add_argument("--finished-only", action="store_true", help="keep only episodes the agent submitted itself")
    p.add_argument("--drop-last", action="store_true", help="score the trajectory before its terminal command")
    p.add_argument("--score", choices=["progress", "partial", "f2p"], default="progress",
                   help="progress: binary F2P>0; partial: the verifier's native partial score (all passed / all tests); f2p: fraction of F2P tests passed; the last two without binarisation")
    a = p.parse_args()
    def read(path: Path, keep):
        return load_compact(path, keep, drop_last=a.drop_last) if path.suffix == ".jsonl" else load(path, keep)

    nonempty = (lambda r: r["size"] > 0 and r["submitted"]) if a.finished_only else (lambda r: r["size"] > 0)  # noqa: E731
    progress = {"partial": lambda r: r["partial"], "f2p": lambda r: float(r["f2p"]), "progress": lambda r: int(r["f2p"] > 0)}[a.score]
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
    order = ["Logprob (mean)", "Perplexity (max)", "MTE (max)", "Verbalized UQ (final)", "Verbalized UQ (mean)", "Tool success rate",
             "Bayes tool-only", "Bayes UQ-only (cont.)", "Bayes Fused (cont.)", "Bayes Fused (SEP)", "Length-only baseline"]
    print(f"{'method':24s}" + "".join(f"{c:>24s}" for c in cols))
    for k in order:
        print(f"{k:24s}" + "".join((f"{cols[c][k]:+24.3f}" if cols[c].get(k) is not None else f"{'NA':>24s}") for c in cols))
    print(f"{'tasks (positives)':24s}" + "".join(f"{str(cols[c]['n'])+' ('+str(cols[c]['positives'])+')':>24s}" for c in cols))
    if a.score != "progress":
        print(f"label: {a.score} score (continuous PRR; Bayes trained on a per-seed median split of the calibration half)")


if __name__ == "__main__":
    main()
