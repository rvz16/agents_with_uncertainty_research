"""How much of the OSWorld signal is just "the harness ran out of steps"?

Three readings: the step cap as a one-bit predictor, PRR of -N and of the best
genuine signal inside length-homogeneous strata, and the same with capped
episodes removed.
"""
import statistics as st, sys
from collections import Counter
from pathlib import Path
TOOLKIT = Path(os.environ.get("UQ_TOOLKIT", ""))
if TOOLKIT:
    sys.path.insert(0, str(TOOLKIT))
from experiments import prr_report_v2 as R
from experiments.osworld.load_osworld import load_osworld

eps = load_osworld([Path(sys.argv[1])])
cap = max(e["n_steps"] for e in eps.values())
hit = [e for e in eps.values() if e["n_steps"] >= cap]
print(f"cap = {cap} steps | episodes at the cap: {len(hit)} | their mean score {st.fmean(e['score'] for e in hit):.3f}"
      f" | solved among them {sum(e['score'] > .5 for e in hit)}")
y = [e["score"] for e in eps.values()]
print(f"PRR of the one-bit 'did not hit the cap': {R.prr(y, [float(e['n_steps'] < cap) for e in eps.values()]):+.3f}")

PICK = [("Reference", "N steps", "\\ensuremath{-}N"),
        ("Verb actions", "Verb actions \\ensuremath{-} Bayes UQ-only", "SEP"),
        ("Logprob", "Logprob \\ensuremath{-} Bayes UQ-only", "Mean only"),
        ("Self-certainty", "Self-certainty \\ensuremath{-} Bayes UQ-only", "Tempered"),
        ("Reference", "TemporalBelief (B4)", "final checkpoint")]
methods = R.method_table(str(TOOLKIT))

def binarize(d):
    """The median split that with_labels() uses for a continuous score collapses
    on these subsets: once the capped failures are gone the median score is 1.0,
    every training label becomes 0, and the Bayes fits degenerate to a constant
    (the tell was four different methods returning one identical PRR). The
    control therefore scores the binary target, 0.5 threshold, as ALFWorld does."""
    return {k: dict(v, score=float(v["score"] > 0.5), label=int(v["score"] > 0.5)) for k, v in d.items()}


def run(subset, title):
    subset = binarize(subset)
    print(f"\n{title}: {len(subset)} episodes, mean score {st.fmean(e['score'] for e in subset.values()):.3f}")
    for key in PICK:
        vals = [v for s in R.SEEDS if (v := R.oof(subset, methods[key], s)) is not None]
        print(f"   {key[1][:44]:46} {st.fmean(vals):+.3f}" if vals else f"   {key[1][:44]:46}  n/a")

run(eps, "all episodes")
run({k: v for k, v in eps.items() if v["n_steps"] < cap}, "capped episodes dropped")
mid = {k: v for k, v in eps.items() if 10 <= v["n_steps"] <= 40}
run(mid, "length-homogeneous band, 10-40 steps")
