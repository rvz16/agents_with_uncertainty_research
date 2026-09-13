"""Table cells where every method chooses its own segment on the calibration half.

The earlier tables let the feature rows take the best of four response
segments (thought / action / reasoning / combined), chosen on the test half,
while the Bayes rows were fixed to `combined`. On smolagents the segments
differ a lot -- gpt-oss's `thought` log-probability separates outcomes at
PRR .94 where `combined` gives .63 -- so the two kinds of row were not
comparable. Here every method has the same freedom and the same discipline:
for each split seed, among its candidate (segment, signal) cells, the one
with the best PRR on the calibration half is chosen and its test-half PRR is
reported; the cell is the mean over seeds. Length-only and tool-success-rate
baselines have no choice to make and are computed from the episodes.

    python -m experiments.select_on_calibration RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from experiments.analyze_trajectories import _prr_references, prediction_rejection_area

SEGMENTS = ("thought", "action", "reasoning", "combined")
UQ_SIGNALS = ("mean_token_logprob", "perplexity", "mean_token_entropy", "sequence_probability", "sum_logprob", "verbalized_confidence")
# label -> (models, methods)  -- candidates are every segment x listed methods
ROWS = {
    "Logprob (mean)": (("feature_mean",), ("mean_token_logprob",)),
    "Perplexity (max)": (("feature_max",), ("perplexity",)),
    "MTE (max)": (("feature_max",), ("mean_token_entropy",)),
    "Verbalized UQ (final)": (("feature_last",), ("verbalized_confidence",)),
    "Bayes tool-only": (("bayes_state",), UQ_SIGNALS),  # critic-only belief, identical across segments
    "Bayes UQ-only (cont.)": (("continuous_bayes",), UQ_SIGNALS),
    "Bayes Fused (cont.)": (("bayes_state_plus_continuous",), UQ_SIGNALS),
    "Bayes Fused (SEP)": (("bayes_state_plus_sep",), UQ_SIGNALS),
    "Bayes tool-only (tempered)": (("stepwise_bayes_state_tempered",), UQ_SIGNALS),
    "Bayes Fused (cont., tempered)": (("stepwise_tempered_plus_continuous",), UQ_SIGNALS),
    "Bayes Fused (SEP, tempered)": (("stepwise_tempered_plus_sep",), UQ_SIGNALS),
}
METRIC = "prr_at_0_5"


def _prr(y, conf):
    a = prediction_rejection_area([-c for c in conf], y, 0.5)
    o, r = _prr_references(tuple(y), 0.5)
    return None if None in (a, o, r) or abs(o - r) < 1e-9 else (a - r) / (o - r)


def cell(run: Path, models, methods, full_coverage: bool = True) -> tuple[float | None, dict[str, int]]:
    """A segment that only some episodes have (gpt-oss writes a `thought`
    prose block mostly when it is about to give up: 38 of 70 test episodes,
    5 of them successes) is scored on a different, easier cohort. Unless
    told otherwise, only cells evaluated on every test episode compete."""
    per_seed = []; chosen = defaultdict(int)
    for seed_dir in sorted(run.glob("analysis_splits/seed_*")):
        cal, test, n_of = {}, {}, {}
        rows = [r for r in csv.DictReader(open(seed_dir / "metrics.csv")) if not r["prefix"] and r[METRIC]]
        n_full = max(int(r["n"]) for r in rows if r["split"] == "test")
        for r in rows:
            if r["model"] not in models or r["uq_method"] not in methods:
                continue
            key = (r["target"], r["uq_method"], r["model"])
            (cal if r["split"] == "calibration" else test)[key] = float(r[METRIC])
            if r["split"] == "test":
                n_of[key] = int(r["n"])
        usable = [k for k in cal if k in test and (not full_coverage or n_of[k] == n_full)]
        if not usable:
            continue
        best = max(usable, key=lambda k: cal[k])
        per_seed.append(test[best]); chosen[f"{best[0]}/{best[1]}"] += 1
    return (st.fmean(per_seed) if per_seed else None), dict(chosen)


def baselines(run: Path) -> dict[str, float | None]:
    eps = [json.loads(l) for l in open(run / "episodes.jsonl") if l.strip()]
    y = [int(bool(e["final_success"])) for e in eps]
    return {
        "Tool success rate": _prr(y, [e.get("tool_success_rate") or 0.0 for e in eps]),
        "Length-only baseline": _prr(y, [-float(e["num_steps"]) for e in eps]),
        "n": len(y), "positives": sum(y),
    }


def main() -> None:
    runs = [Path(p) for p in sys.argv[1:]]
    print(f"{'method':24s}" + "".join(f"{r.name[-22:]:>24s}" for r in runs))
    choices = {}
    for label, (models, methods) in ROWS.items():
        vals = []
        for run in runs:
            v, ch = cell(run, models, methods, full_coverage=not label.startswith("Verbalized"))
            vals.append(v); choices[(label, run.name)] = ch
        print(f"{label:24s}" + "".join((f"{v:+24.3f}" if v is not None else f"{'NA':>24s}") for v in vals))
    base = [baselines(r) for r in runs]
    for k in ("Tool success rate", "Length-only baseline"):
        print(f"{k:24s}" + "".join((f"{b[k]:+24.3f}" if b[k] is not None else f"{'NA':>24s}") for b in base))
    print(f"{'episodes (successes)':24s}" + "".join(f"{str(b['n'])+' ('+str(b['positives'])+')':>24s}" for b in base))
    print("\nchosen (segment/signal: seeds):")
    for (label, run), ch in choices.items():
        top = sorted(ch.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {label:24s} {run[-22:]:>22s}  " + ", ".join(f"{k}:{n}" for k, n in top))


if __name__ == "__main__":
    main()
