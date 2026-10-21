"""P1: the paper's signal table on the OSWorld cohort, before any training.

Nothing here is new machinery -- it is experiments.prr_report_v2's own method
table and its own cross-fitting (3 seeds x 5 stratified folds, PRR@0.5 on the
native continuous score), pointed at the new cohort. The point of running it
first is that it costs no GPU and it says whether the teacher posterior is
worth distilling here at all.

Read every row against "Reference / N steps". OSWorld failures run into the
100-step cap while successes finish in 10-35 steps, so step count alone ranks
episodes well for reasons that have nothing to do with uncertainty.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLKIT = Path(os.environ.get("UQ_TOOLKIT", ""))
if TOOLKIT:
    sys.path.insert(0, str(TOOLKIT))

from experiments import prr_report_v2 as R  # noqa: E402
from experiments.osworld.load_osworld import load_osworld  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="run roots holding uq/episodes.jsonl")
    p.add_argument("--top", type=int, default=25)
    a = p.parse_args()

    eps = load_osworld([Path(r) for r in a.runs])
    scores = [e["score"] for e in eps.values()]
    steps = [e["n_steps"] for e in eps.values()]
    print(f"{len(eps)} episodes | mean score {st.fmean(scores):.3f} | "
          f"solved {sum(s > 0.5 for s in scores)} | partial {sum(0 < s < 1 for s in scores)} | "
          f"continuous={R.is_continuous(eps)}")
    print(f"steps: median {st.median(steps):.0f}, mean {st.fmean(steps):.1f}, max {max(steps)}")
    print("apps:", dict(Counter(e["app"] for e in eps.values()).most_common()))
    solved = [e["n_steps"] for e in eps.values() if e["score"] > 0.5]
    failed = [e["n_steps"] for e in eps.values() if e["score"] <= 0.5]
    print(f"steps by outcome: solved median {st.median(solved):.0f}, failed median {st.median(failed):.0f}")

    methods = R.method_table(str(TOOLKIT))
    rows = []
    for key, fn in methods.items():
        vals = [v for s in R.SEEDS if (v := R.oof(eps, fn, s)) is not None]
        if len(vals) == len(R.SEEDS):
            rows.append((st.fmean(vals), st.stdev(vals) if len(vals) > 1 else 0.0, key))
    rows.sort(reverse=True)
    print(f"\n{len(rows)} methods scored (of {len(methods)} in the table)\n")
    print(f"{'PRR':>14}  method")
    for mean, sd, key in rows[:a.top]:
        print(f"{mean:+7.3f} ±{sd:5.3f}  {key[1]} / {key[2]}"[:110])
    print("  ...")
    for mean, sd, key in rows[-3:]:
        print(f"{mean:+7.3f} ±{sd:5.3f}  {key[1]} / {key[2]}"[:110])
    for want in (("Reference", "N steps", "\\ensuremath{-}N"),
                 ("Reference", "Tool success rate", "mean of observed tool critics")):
        hit = next((r for r in rows if r[2] == want), None)
        if hit:
            rank = rows.index(hit) + 1
            print(f"\n{want[1]:20} {hit[0]:+.3f} ±{hit[1]:.3f}   rank {rank}/{len(rows)}")


if __name__ == "__main__":
    main()
