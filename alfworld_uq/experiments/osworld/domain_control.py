"""Is the OSWorld signal about this run, or about which task was drawn?

P1 showed -N steps at +0.68. Two confounds sit underneath it: the 100-step cap,
and task difficulty -- OSWorld's multi_apps split is a quarter of the cohort,
scores 0.27 and runs twice as long as anything else, so "long" and "failed"
partly just mean "multi_apps". Scoring inside a domain removes that.

Also merges the three runs. The recovery run re-ran 23 episodes of the main
run; --dedupe picks which copy wins.
"""
import argparse, os, statistics as st, sys
from collections import Counter
from pathlib import Path

TOOLKIT = Path(os.environ.get("UQ_TOOLKIT", ""))
HERE = Path(__file__).resolve().parent
if TOOLKIT:
    sys.path.insert(0, str(TOOLKIT))
from experiments import prr_report_v2 as R          # noqa: E402
from experiments.osworld.load_osworld import load_osworld               # noqa: E402

RAW = HERE / "raw"
RUNS = ["full-2gpu-max100-20260911", "full-recovery-infra-20260912",
        "affected-recovery-20260915.IFxOYC"]
PICK = {"-N steps": ("Reference", "N steps", "\\ensuremath{-}N"),
        "B4": ("Reference", "TemporalBelief (B4)", "final checkpoint"),
        "Logprob SEP": ("Logprob", "Logprob \\ensuremath{-} Bayes UQ-only", "SEP"),
        "Verb SEP": ("Verb actions", "Verb actions \\ensuremath{-} Bayes UQ-only", "SEP")}


def binarize(d):
    return {k: dict(v, score=float(v["score"] > 0.5), label=int(v["score"] > 0.5))
            for k, v in d.items()}


def scored(eps, methods, names):
    out = {}
    for name in names:
        vals = [v for s in R.SEEDS if (v := R.oof(eps, methods[PICK[name]], s)) is not None]
        out[name] = st.fmean(vals) if len(vals) == len(R.SEEDS) else None
    return out


def show(title, eps, methods, names=tuple(PICK)):
    eps = binarize(eps)
    if len(eps) < 25 or len({e["score"] for e in eps.values()}) < 2:
        print(f"{title:36} n={len(eps):3}  too small / single class"); return
    row = scored(eps, methods, names)
    cells = "  ".join(f"{n} {row[n]:+.3f}" if row[n] is not None else f"{n}   n/a" for n in names)
    print(f"{title:36} n={len(eps):3}  base {st.fmean(e['score'] for e in eps.values()):.2f}  {cells}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dedupe", choices=("latest", "main"), default="latest")
    a = p.parse_args()
    runs = [RAW / r / "osworld-qwen-100" for r in RUNS]
    if a.dedupe == "main":
        runs = runs[:1] + runs[1:2]          # main + the one genuinely new episode
    eps = load_osworld(runs)
    print(f"dedupe={a.dedupe}: {len(eps)} episodes, mean score "
          f"{st.fmean(e['score'] for e in eps.values()):.3f}, "
          f"{sum(e['n_steps'] >= 100 for e in eps.values())} at the step cap\n")
    methods = R.method_table(str(TOOLKIT))
    show("whole cohort", eps, methods)
    show("capped episodes dropped", {k: v for k, v in eps.items() if v["n_steps"] < 100}, methods)
    show("10-40 steps", {k: v for k, v in eps.items() if 10 <= v["n_steps"] <= 40}, methods)
    print()
    per = {}
    for domain, n in Counter(e["app"] for e in eps.values()).most_common():
        sub = {k: v for k, v in eps.items() if v["app"] == domain}
        show(f"domain: {domain}", sub, methods)
        if n >= 40:
            b = binarize(sub)
            if len({e["score"] for e in b.values()}) > 1:
                per[domain] = scored(b, methods, tuple(PICK))
    if per:
        print("\nmean over the four domains with n>=40:")
        for name in PICK:
            vals = [v[name] for v in per.values() if v[name] is not None]
            print(f"   {name:12} {st.fmean(vals):+.3f}" if vals else f"   {name:12} n/a")


if __name__ == "__main__":
    main()
