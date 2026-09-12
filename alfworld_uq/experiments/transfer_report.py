"""Out-of-domain transfer: fit the Bayesian models on one run, score another.

Every number in the main table fits its parameters on a calibration half of
the very run it is scored on. That leaves a question a reviewer will ask:
are the critic likelihoods and the UQ thresholds properties of the agent, of
the model, or of neither -- and how much is lost when one changes?

This fits on a random half of the source run and scores the target run
whole, repeated over seeds -- the same amount of calibration data the
in-domain protocol gets, so that a difference is the cost of transfer and
not of sample size. The same three Bayesian variants the main table reports.

    python -m experiments.transfer_report \\
        --pair runs/A runs/B --pair runs/B runs/A ...
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

from belief.binary_bayes import BinaryBayesUQ
from belief.continuous_bayes import ContinuousBayesUQ
from belief.critic_bayes import CriticBayesState
from experiments.analyze_trajectories import (
    _critic_observations,
    _predict_from_belief,
    _sequence,
    _split_ids,
    metric_values,
)

TARGET = "combined"
METHOD = "mean_token_logprob"   # the fused rows in the main table use this signal


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(run: Path) -> dict[str, Any]:
    episodes = _read(run / "episodes.jsonl")
    rows: dict[str, list] = defaultdict(list)
    for row in _read(run / "trajectories.jsonl"):
        rows[row["episode_id"]].append(row)
    ids = [e["episode_id"] for e in episodes if e["episode_id"] in rows]
    return {
        "ids": ids,
        "labels": {e["episode_id"]: int(bool(e["final_success"])) for e in episodes},
        "critics": {i: _critic_observations(rows[i]) for i in ids},
        "sequences": {i: _sequence(sorted(rows[i], key=lambda r: int(r["step"])), TARGET, METHOD) for i in ids},
    }


def fit(source: dict[str, Any], ids: list[str]) -> dict[str, Any]:
    labels = [source["labels"][i] for i in ids]
    critics = CriticBayesState.fit([source["critics"][i] for i in ids], labels)
    usable = [i for i in ids if source["sequences"][i]]
    seqs = [source["sequences"][i] for i in usable]
    seq_labels = [source["labels"][i] for i in usable]
    return {
        "critics": critics,
        "continuous": ContinuousBayesUQ.fit(seqs, seq_labels, lambda_=1.0),
        "sep": BinaryBayesUQ.fit(seqs, seq_labels, threshold_mode="sep", higher_is_uncertain=False),
    }


def score(models: dict[str, Any], target: dict[str, Any], ids: list[str]) -> dict[str, float | None]:
    usable = [i for i in ids if target["sequences"][i]]
    labels = [target["labels"][i] for i in usable]
    belief = {i: models["critics"].predict(target["critics"][i]) for i in usable}
    out = {}
    out["Bayes tool-only"] = metric_values(labels, [belief[i] for i in usable])
    out["Bayes Fused (cont.)"] = metric_values(
        labels, [_predict_from_belief(models["continuous"], target["sequences"][i], belief[i]) for i in usable]
    )
    out["Bayes Fused (SEP)"] = metric_values(
        labels, [_predict_from_belief(models["sep"], target["sequences"][i], belief[i]) for i in usable]
    )
    return {k: v.get("prr_at_0_5") for k, v in out.items()}


def in_domain(run: dict[str, Any], seeds: int) -> dict[str, float]:
    """The main table's protocol: fit on half, score the other half, average."""
    acc: dict[str, list[float]] = defaultdict(list)
    for seed in range(seeds):
        cal, test = _split_ids(run["ids"], 0.5, seed)
        for k, v in score(fit(run, cal), run, test).items():
            if v is not None:
                acc[k].append(v)
    return {k: st.fmean(v) for k, v in acc.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", nargs=2, action="append", metavar=("SOURCE", "TARGET"), required=True)
    p.add_argument("--seeds", type=int, default=20)
    a = p.parse_args()
    cache: dict[str, dict[str, Any]] = {}
    ref: dict[str, dict[str, float]] = {}
    for src, tgt in a.pair:
        for r in (src, tgt):
            if r not in cache:
                cache[r] = load(Path(r))
                ref[r] = in_domain(cache[r], a.seeds)
        # Fit on a random half of the source, score the whole target, repeat.
        # Fitting on all of the source would give the transferred model twice
        # the data the in-domain one gets, and that alone read as a "gain" of
        # up to +.18 in the first pass. Same amount of data on both sides now.
        acc: dict[str, list[float]] = defaultdict(list)
        for seed in range(a.seeds):
            half, _ = _split_ids(cache[src]["ids"], 0.5, seed)
            for k, v in score(fit(cache[src], half), cache[tgt], cache[tgt]["ids"]).items():
                if v is not None:
                    acc[k].append(v)
        print(f"\n{Path(src).name}  ->  {Path(tgt).name}")
        for k in ("Bayes tool-only", "Bayes Fused (cont.)", "Bayes Fused (SEP)"):
            i = ref[tgt].get(k)
            if not acc.get(k) or i is None:
                print(f"   {k:22s} —"); continue
            t = st.fmean(acc[k]); sd = st.stdev(acc[k]) if len(acc[k]) > 1 else 0.0
            print(f"   {k:22s} transfer {t:+.3f} ± {sd:.3f} | in-domain {i:+.3f} | drop {t - i:+.3f}")


if __name__ == "__main__":
    main()
