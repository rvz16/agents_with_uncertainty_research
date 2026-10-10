"""Is the decision model right about the present state?

p(success) is a forecast of a latent terminal outcome -- nothing any of these
models was trained or calibrated on. Two of the four questions we asked are
instead judgements about the state in front of the model, which is what a
System One model is for, and for those ALFWorld gives mechanical ground truth:
a repeated (action, observation) pair is a repetition, and a mutating action
after the most recent inspection is stale evidence.

So this scores the two on-label answers against that ground truth, per prefix,
and reports AUROC, ECE and the base rate -- the same measurements we applied to
the forecast, but on a question the model is actually sold for.
"""
import argparse
import json
import statistics as st
from pathlib import Path

COHORTS = ("RG", "SG", "RQ", "SQ")

CHECK = ("look", "examine", "inventory", "go to")
MUTATE = ("take ", "put ", "open ", "close ", "toggle ", "heat ", "cool ", "clean ", "slice ", "use ")


def norm(s): return " ".join((s or "").lower().split())


def labels_for(steps):
    """per step: (repetition, stale_evidence); None where the question does not apply."""
    out = []
    seen = set()
    last_check, mutated_since = -1, False
    for k, s in enumerate(steps):
        a, o = norm(s.get("action")), norm(s.get("observation"))
        repeat = int(((a, o) in seen) or "nothing happens" in o)
        seen.add((a, o))
        stale = None if last_check < 0 else int(mutated_since)
        if any(a.startswith(c) for c in CHECK):
            last_check, mutated_since = k, False
        elif any(a.startswith(m) for m in MUTATE):
            mutated_since = True
        out.append((repeat, stale))
    return out


def auroc(y, p):

    pos = [q for t, q in zip(y, p) if t]
    neg = [q for t, q in zip(y, p) if not t]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def ece(y, p, bins=10):
    total, err = len(y), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = [(t, q) for t, q in zip(y, p) if (q > lo or b == 0) and q <= hi]
        if m:
            err += len(m) / total * abs(st.fmean(q for _, q in m) - st.fmean(t for t, _ in m))
    return err


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--views", type=Path, default=Path("data/judge_views"))
    p.add_argument("--judge-dir", type=Path, required=True)
    p.add_argument("--model", default="Cloudflare_clef-flash")
    a = p.parse_args()

    rows = {}
    for cohort in COHORTS:
        steps_of = {}
        for line in open(a.views / f"{cohort}.jsonl"):
            if line.strip():
                v = json.loads(line)
                steps_of[v["id"]] = labels_for(v["steps"])
        data = {"stuck": ([], []), "stale": ([], [])}
        skipped = 0
        for line in open(a.judge_dir / f"judge_{a.model}_{cohort}.jsonl"):
            if not line.strip():
                continue
            r = json.loads(line)
            truth = steps_of.get(r["id"])
            if truth is None or r["step"] >= len(truth):
                skipped += 1
                continue
            repeat, stale = truth[r["step"]]
            if r.get("stuck") is not None:
                data["stuck"][0].append(repeat); data["stuck"][1].append(float(r["stuck"]))
            if stale is not None and r.get("stale") is not None:
                data["stale"][0].append(stale); data["stale"][1].append(float(r["stale"]))
        rows[cohort] = (data, skipped)

    print(f"{'question':8} {'cohort':7} {'n':>6} {'base':>6} {'mean p':>7} {'AUROC':>7} {'ECE':>7}")
    print("-" * 54)
    for q in ("stuck", "stale"):
        aurocs = []
        for cohort in COHORTS:
            y, p = rows[cohort][0][q]
            if not y:
                continue
            a = auroc(y, p)
            aurocs.append(a)
            print(f"{q:8} {cohort:7} {len(y):6d} {st.fmean(y):6.2f} {st.fmean(p):7.2f} {a:7.3f} {ece(y, p):7.3f}")
        print(f"{q:8} {'avg':7} {'':6} {'':6} {'':7} {st.fmean(aurocs):7.3f}")
        print()
    print("skipped prefixes (no matching view step):", {c: rows[c][1] for c in COHORTS})


if __name__ == "__main__":
    main()
