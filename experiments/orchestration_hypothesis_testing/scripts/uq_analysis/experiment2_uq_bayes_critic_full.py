#!/usr/bin/env python3
"""Experiment 2: fuse a logprob UQ signal into the Bayesian belief as a critic.

`bayes_state` is a posterior P(correct) built from critic/verifier outcomes.
Here we treat a logprob UQ score (default: llm_log_seq_prob of the final answer,
the strongest single signal from Experiment 1) as additional evidence. Binary
modes turn it into one or two critics; continuous modes fit class-conditional
Gaussian likelihoods and use the full score:

    uq_passed   = 1[ score >= threshold ]                    # "model is confident"
    theta_uq    = { P(uq_passed | Y=1), P(uq_passed | Y=0) } # learned from data
    belief_uq   = bayes_update(bayes_state, theta_uq, uq_passed)

Direction/strength are learned via theta (a near-uninformative theta => the
fusion barely moves belief => UQ adds nothing beyond the tools). Threshold and
theta are fit with honest k-fold CV over instances (no instance is scored by a
model that saw it), then we compare PRR of:
  - bayes_state (baseline)
  - the UQ feature alone
  - the fused belief
plus a PAIRED bootstrap CI of (fused - bayes) to test significance.

Inputs come from <readable>/final_logprob_bayes_quality.csv and, for trajectory
aggregation, generation_trajectory_scores.jsonl. No LLM re-run is needed.

Example:
  python scripts/experiment2_uq_bayes_critic.py \
      --readable-dir .../readable/lcb_medium --feature llm_log_seq_prob --k 5
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import prediction_rejection_area, prr

# raw-score direction: higher raw value == more likely correct?
HIGHER_BETTER = {
    "llm_log_seq_prob": True,   # sum logprob
    "llm_perplexity": True,     # mean logprob (higher==closer to 0==confident)
    "perplexity": False,        # exp(-mean logprob): higher==worse
}
EPS = 1e-6
AGGREGATORS = {
    "last": lambda values: values[-1],
    "mean": statistics.fmean,
    "min": min,
    "max": max,
}


def load_final(
    path: Path,
    feature: str,
    feature_overrides: dict[str, float] | None = None,
) -> list[dict]:
    out = []
    for r in csv.DictReader(path.open()):
        try:
            iid = str(r["instance_id"])
            feature_value = (
                feature_overrides[iid]
                if feature_overrides is not None and iid in feature_overrides
                else float(r[feature])
            )
            if feature_overrides is not None and iid not in feature_overrides:
                continue
            out.append({
                "iid": iid,
                "bayes": float(r["bayes_state"]),
                "quality": int(r["quality"]),
                "feat_raw": feature_value,
            })
        except (KeyError, TypeError, ValueError):
            continue
    return out


def load_trajectory_feature(
    path: Path, feature: str, aggregation: str
) -> dict[str, float]:
    if aggregation not in AGGREGATORS:
        raise ValueError(f"unsupported aggregation: {aggregation}")
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("logprobs_supported") is False:
                continue
            try:
                value = float(row[feature])
                order = int(row.get("patch_idx", row.get("action_step", 0)))
                iid = str(row["instance_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                grouped[iid].append((order, value))
    aggregate = AGGREGATORS[aggregation]
    return {
        iid: float(aggregate([value for _, value in sorted(values)]))
        for iid, values in grouped.items()
        if values
    }


def bayes_update(belief: float, p_pass_y1: float, p_pass_y0: float, passed: bool) -> float:
    p = p_pass_y1 if passed else (1.0 - p_pass_y1)
    q = p_pass_y0 if passed else (1.0 - p_pass_y0)
    num = belief * p
    den = num + (1.0 - belief) * q
    return num / den if den > 0 else belief


def _theta_for_threshold(feats, quality, thr, higher_better):
    """θ = (p1, p0) for the critic uq_passed = feat>=thr (or <= if lower-better)."""
    n1 = sum(quality) or 1
    n0 = (len(quality) - sum(quality)) or 1
    passed = [(f >= thr) if higher_better else (f <= thr) for f in feats]
    p1 = (sum(p for p, y in zip(passed, quality) if y == 1) + 1) / (n1 + 2)  # Beta(1,1)
    p0 = (sum(p for p, y in zip(passed, quality) if y == 0) + 1) / (n0 + 2)
    return p1, p0


def fit_threshold_theta(feats, quality, higher_better, *, mode="sep"):
    """Pick a threshold over a quantile grid, by one of several criteria:

      sep     — max |p1 - p0|            (balanced separation; the original)
      lr_pos  — max p1/p0                (decisive-correct: 'pass' ⇒ likely Y=1)
      lr_neg  — min (1-p1)/(1-p0)        (decisive-error:   'fail' ⇒ likely Y=0)

    Returns (thr, p1, p0). Beta(1,1) smoothing keeps p0>0 / p1<1 so updates stay
    finite even at an 'ideal' threshold.
    """
    srt = sorted(feats)
    best = None
    for qtl in [i / 20 for i in range(1, 20)]:
        thr = srt[min(len(srt) - 1, int(qtl * len(srt)))]
        p1, p0 = _theta_for_threshold(feats, quality, thr, higher_better)
        if mode == "sep":
            score = abs(p1 - p0)
        elif mode == "lr_pos":
            score = p1 / p0                      # want ≫ 1
        elif mode == "lr_neg":
            score = -((1 - p1) / (1 - p0))       # want (1-p1)/(1-p0) ≪ 1
        else:
            raise ValueError(mode)
        if best is None or score > best[0]:
            best = (score, thr, p1, p0)
    return best[1], best[2], best[3]


def kfold_fuse(rows, higher_better, *, k, seed, mode="sep"):
    """Fuse UQ critic(s) into bayes via honest k-fold.

    mode in {sep, lr_pos, lr_neg}: one critic with that threshold criterion.
    mode == 'double': fit BOTH lr_pos and lr_neg on train, apply both critics
    sequentially to the belief (two pieces of evidence from one score).
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    folds = [order[i::k] for i in range(k)]
    fused = {}
    modes = ["lr_pos", "lr_neg"] if mode == "double" else [mode]
    for f in range(k):
        test_idx = set(folds[f])
        train = [rows[i] for i in order if i not in test_idx]
        tf = [r["feat_raw"] for r in train]
        tq = [r["quality"] for r in train]
        fitted = [(m, *fit_threshold_theta(tf, tq, higher_better, mode=m)) for m in modes]
        for i in test_idx:
            r = rows[i]
            belief = r["bayes"]
            for _m, thr, p1, p0 in fitted:
                passed = (r["feat_raw"] >= thr) if higher_better else (r["feat_raw"] <= thr)
                belief = bayes_update(belief, p1, p0, passed)
            fused[r["iid"]] = belief
    return fused


def _clip_probability(value: float) -> float:
    return min(1.0 - EPS, max(EPS, value))


def _logit(value: float) -> float:
    value = _clip_probability(value)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _gaussian_logpdf(value: float, mean: float, std: float) -> float:
    z = (value - mean) / std
    return -0.5 * z * z - math.log(std) - 0.5 * math.log(2.0 * math.pi)


def fit_class_gaussians(feats, quality, *, min_std=1e-3):
    pooled_mean = statistics.fmean(feats)
    pooled_std = max(statistics.pstdev(feats), min_std)

    def parameters(label):
        values = [feat for feat, outcome in zip(feats, quality) if outcome == label]
        if not values:
            return pooled_mean, pooled_std
        return statistics.fmean(values), max(statistics.pstdev(values), min_std)

    failure_mean, failure_std = parameters(0)
    success_mean, success_std = parameters(1)
    return success_mean, success_std, failure_mean, failure_std


def continuous_bayes_update(belief, value, parameters, *, lambda_=1.0):
    success_mean, success_std, failure_mean, failure_std = parameters
    llr = _gaussian_logpdf(value, success_mean, success_std) - _gaussian_logpdf(
        value, failure_mean, failure_std
    )
    return _clip_probability(_sigmoid(_logit(belief) + lambda_ * llr))


def kfold_continuous_fuse(rows, *, k, seed, lambda_=1.0):
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")
    import numpy as np

    rng = np.random.RandomState(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    folds = [order[i::k] for i in range(k)]
    fused = {}
    for fold in folds:
        test_idx = set(fold)
        train = [rows[i] for i in order if i not in test_idx]
        parameters = fit_class_gaussians(
            [row["feat_raw"] for row in train],
            [row["quality"] for row in train],
        )
        for i in test_idx:
            row = rows[i]
            fused[row["iid"]] = continuous_bayes_update(
                row["bayes"], row["feat_raw"], parameters, lambda_=lambda_
            )
    return fused


def auroc(labels, scores):
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return None
    return statistics.fmean(
        float(positive > negative) + 0.5 * float(positive == negative)
        for positive in positives
        for negative in negatives
    )


def probability_metrics(labels, probabilities, *, bins=10):
    clipped = [_clip_probability(value) for value in probabilities]
    n = len(labels)
    brier = statistics.fmean(
        (probability - label) ** 2
        for probability, label in zip(clipped, labels)
    )
    nll = -statistics.fmean(
        label * math.log(probability) + (1 - label) * math.log(1 - probability)
        for probability, label in zip(clipped, labels)
    )
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            i
            for i, probability in enumerate(clipped)
            if lower <= probability <= upper
            if index == bins - 1 or probability < upper
        ]
        if selected:
            confidence = statistics.fmean(clipped[i] for i in selected)
            accuracy = statistics.fmean(labels[i] for i in selected)
            ece += len(selected) / n * abs(confidence - accuracy)
    return {"auroc": auroc(labels, clipped), "brier": brier, "nll": nll, "ece": ece}


def paired_diff_ci(a_conf, b_conf, quality, *, n_boot, max_rej=0.5, seed=0):
    """CI of PRR(a) - PRR(b) at rejection budget max_rej, paired resampling."""
    import numpy as np
    n = len(quality)
    oracle = prediction_rejection_area([-float(q) for q in quality], quality, max_rej)
    rng = np.random.RandomState(seed)
    arr = np.arange(n)
    rnd = []
    for _ in range(200):
        rng.shuffle(arr)
        s = prediction_rejection_area(arr.tolist(), quality, max_rej)
        if s is not None:
            rnd.append(s)
    random = float(np.mean(rnd)) if rnd else None
    if oracle is None or random is None or abs(oracle - random) < 1e-12:
        return None

    diffs = []
    idx = np.arange(n)
    for _ in range(n_boot):
        take = rng.choice(idx, size=n, replace=True)
        q = [quality[i] for i in take]
        if len(set(q)) < 2:
            continue
        pa = prediction_rejection_area([-a_conf[i] for i in take], q, max_rej)
        pb = prediction_rejection_area([-b_conf[i] for i in take], q, max_rej)
        if pa is not None and pb is not None:
            diffs.append(((pa - random) / (oracle - random))
                         - ((pb - random) / (oracle - random)))
    if len(diffs) < max(10, n_boot // 4):
        return None
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readable-dir", type=Path, required=True)
    p.add_argument("--feature", choices=list(HIGHER_BETTER), default="llm_log_seq_prob")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--max-rej", type=float, default=0.5,
                   help="PRR rejection budget; 0.5 = PRR@0.5 (matches the main tables)")
    p.add_argument(
        "--mode",
        choices=[
            "sep",
            "lr_pos",
            "lr_neg",
            "double",
            "continuous",
            "tempered",
            "all",
        ],
        default="sep",
                   help="threshold criterion: sep=|p1-p0| (orig); lr_pos=decisive-correct; "
                        "lr_neg=decisive-error; double=fuse both lr critics; "
                        "continuous/tempered=Gaussian likelihood-ratio fusion")
    p.add_argument("--tempered-lambda", type=float, default=0.25)
    p.add_argument("--trajectory", type=Path)
    p.add_argument(
        "--aggregation", choices=["final", *AGGREGATORS], default="final"
    )
    p.add_argument("--output-csv", type=Path)
    args = p.parse_args()

    hb = HIGHER_BETTER[args.feature]
    overrides = None
    if args.aggregation != "final":
        trajectory_path = args.trajectory or (
            args.readable_dir / "generation_trajectory_scores.jsonl"
        )
        overrides = load_trajectory_feature(
            trajectory_path, args.feature, args.aggregation
        )
    rows = load_final(
        args.readable_dir / "final_logprob_bayes_quality.csv",
        args.feature,
        overrides,
    )
    q = [r["quality"] for r in rows]
    if len(set(q)) < 2:
        print("degenerate (single class), abort"); return

    ids = [r["iid"] for r in rows]
    quality = [r["quality"] for r in rows]
    conf_bayes = [r["bayes"] for r in rows]
    conf_feat = [(r["feat_raw"] if hb else -r["feat_raw"]) for r in rows]
    mr = args.max_rej
    prr_bayes = prr(conf_bayes, quality, mr)
    prr_feat = prr(conf_feat, quality, mr)
    tag = f"PRR@{mr:g}"
    print(f"dir={args.readable_dir}  feature={args.feature}  n={len(ids)}  "
          f"pass@1={sum(quality)/len(quality):.2f}  k={args.k}  metric={tag}  "
          f"aggregation={args.aggregation}")
    print(
        f"{'signal':<30}{tag:>8}{'delta':>9}{'AUROC':>9}"
        f"{'Brier':>9}{'NLL':>9}{'ECE':>9}"
    )
    baseline_metrics = probability_metrics(quality, conf_bayes)
    print(
        f"{'bayes_state (baseline)':<30}{prr_bayes:>8.3f}{0.0:>+9.3f}"
        f"{baseline_metrics['auroc']:>9.3f}{baseline_metrics['brier']:>9.3f}"
        f"{baseline_metrics['nll']:>9.3f}{baseline_metrics['ece']:>9.3f}"
    )
    print(f"{args.feature + ' alone':<30}{prr_feat:>8.3f}")

    modes = (
        ["sep", "lr_pos", "lr_neg", "double", "continuous", "tempered"]
        if args.mode == "all"
        else [args.mode]
    )
    output_rows = [
        {
            "feature": args.feature,
            "aggregation": args.aggregation,
            "method": "bayes_state",
            "n": len(rows),
            "pass_at_1": sum(quality) / len(quality),
            "prr": prr_bayes,
            "prr_at_0_5": prr_bayes if mr == 0.5 else None,
            "delta_prr_vs_bayes": 0.0,
            "delta_prr_at_0_5_vs_bayes": 0.0 if mr == 0.5 else None,
            "ci_low": None,
            "ci_high": None,
            **baseline_metrics,
        }
    ]
    for mode in modes:
        if mode == "continuous":
            fused = kfold_continuous_fuse(rows, k=args.k, seed=args.seed, lambda_=1.0)
        elif mode == "tempered":
            fused = kfold_continuous_fuse(
                rows, k=args.k, seed=args.seed, lambda_=args.tempered_lambda
            )
        else:
            fused = kfold_fuse(rows, hb, k=args.k, seed=args.seed, mode=mode)
        probabilities = [fused[iid] for iid in ids]
        method_prr = prr(probabilities, quality, mr)
        delta = method_prr - prr_bayes
        ci = paired_diff_ci(
            probabilities,
            conf_bayes,
            quality,
            n_boot=args.n_boot,
            max_rej=mr,
            seed=args.seed,
        )
        metrics = probability_metrics(quality, probabilities)
        ci_low, ci_high = ci if ci is not None else (None, None)
        output_rows.append(
            {
                "feature": args.feature,
                "aggregation": args.aggregation,
                "method": mode,
                "n": len(rows),
                "pass_at_1": sum(quality) / len(quality),
                "prr": method_prr,
                "prr_at_0_5": method_prr if mr == 0.5 else None,
                "delta_prr_vs_bayes": delta,
                "delta_prr_at_0_5_vs_bayes": delta if mr == 0.5 else None,
                "ci_low": ci_low,
                "ci_high": ci_high,
                **metrics,
            }
        )
        print(
            f"{('bayes + ' + mode):<30}{method_prr:>8.3f}{delta:>+9.3f}"
            f"{metrics['auroc']:>9.3f}{metrics['brier']:>9.3f}"
            f"{metrics['nll']:>9.3f}{metrics['ece']:>9.3f}"
        )
        if ci is not None:
            significant = "YES" if ci[0] > 0 or ci[1] < 0 else "no"
            print(
                f"  delta 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], "
                f"significant={significant}"
            )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)


if __name__ == "__main__":
    main()
