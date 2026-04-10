#!/usr/bin/env python3
"""L2 noise ablation on LiveCodeBench.

Question: when L2 (public test execution) is no longer near-oracle, does the
Bayesian controller — which can combine L2 with L3 (LLM reviewer) in the
Bellman equation — beat any fixed single-critic threshold policy?

At alpha=0 the raw LCB L2 has gap ~0.54 and the single-critic Threshold(L2)
dominates. As alpha grows we replace each L2 outcome with a coin flip with
probability alpha, which drives L2's TPR and FPR toward 0.5 and shrinks its
gap linearly. L3 (Haiku-4.5 reviewer) stays at gap ~0.28. Somewhere around
alpha=0.5 the L2 gap falls below L3's, and neither critic dominates.

For each alpha we:
  1. Deterministically noise every patch's L2 outcome (seeded by
     instance+patch_id so results are reproducible across runs).
  2. Recompute likelihood tables from the noised data.
  3. Build a Bayesian controller with the new tables and the iid transition
     kernel (fresh patch per generate, appropriate for our iid-sampled data).
  4. Run Bayesian + Threshold(L1/L2/L3) + Fixed on the noised episodes.
  5. Report marginal means, SE, paired Bayesian - best_single_threshold
     differences with t-statistics.

Usage:
    python l2_noise_ablation.py
    python l2_noise_ablation.py --alphas 0.0 0.3 0.5 0.7 0.9 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controller.bayesian_controller import (
    BayesianController,
    CostModel,
    CriticLikelihood,
    TransitionKernel,
)
from controller.multi_critic_controller import MultiCriticBayesianController
from evaluation.run_simulation import (
    EpisodeResult,
    run_bayesian_policy,
    run_fixed_pipeline,
    run_multi_critic_policy,
    run_threshold_policy,
)

DEFAULT_DATA = (
    Path(__file__).resolve().parents[1]
    / "calibration" / "data" / "lcb_results_v2.jsonl"
)

CRITIC_LEVELS = ["L0_syntax", "L1_lint", "L2_fast_test", "L3_llm_review", "L4_mypy"]


def load_episodes_lcb(path: Path) -> dict[str, list[dict]]:
    """Load LCB jsonl and group by question_id/instance_id."""
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        key = r.get("instance_id") or r.get("question_id") or "?"
        by_instance[key].append(r)
    for patches in by_instance.values():
        patches.sort(key=lambda r: r.get("patch_id", r.get("step", 0)))
    return dict(by_instance)


def noise_l2(
    episodes: dict[str, list[dict]],
    alpha: float,
    seed: int,
) -> dict[str, list[dict]]:
    """Replace each patch's L2 verdict with a fresh coin flip w/ prob alpha.

    Per-patch RNG is seeded by (seed, instance_id, patch_id) so the output is
    deterministic across runs and only a function of the inputs.
    """
    noised: dict[str, list[dict]] = {}
    for inst, patches in episodes.items():
        new_patches = []
        for p in patches:
            p_copy = deepcopy(p)
            l2 = p_copy["critic_results"].get("L2_fast_test")
            if l2 is not None:
                pid = p.get("patch_id", p.get("step", 0))
                rng = random.Random(f"{seed}|{inst}|{pid}")
                if rng.random() < alpha:
                    l2["passed"] = rng.random() < 0.5
                    l2["detail"] = f"[NOISE alpha={alpha}] resampled"
            new_patches.append(p_copy)
        noised[inst] = new_patches
    return noised


def compute_likelihoods_from_episodes(
    episodes: dict[str, list[dict]],
    smoothing: float = 1.0,
) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    )
    for patches in episodes.values():
        for p in patches:
            y = p["ground_truth"]
            for level in CRITIC_LEVELS:
                c = p["critic_results"].get(level)
                if c is None:
                    continue
                passed = bool(c.get("passed", False))
                if passed and y == 1:
                    counts[level]["tp"] += 1
                elif passed and y == 0:
                    counts[level]["fp"] += 1
                elif (not passed) and y == 1:
                    counts[level]["fn"] += 1
                else:
                    counts[level]["tn"] += 1

    likelihoods: dict[str, dict[str, float]] = {}
    for level in CRITIC_LEVELS:
        c = counts[level]
        n_correct = c["tp"] + c["fn"]
        n_incorrect = c["fp"] + c["tn"]
        if n_correct + n_incorrect == 0:
            continue
        p_pass_correct = (c["tp"] + smoothing) / (n_correct + 2 * smoothing)
        p_pass_incorrect = (c["fp"] + smoothing) / (n_incorrect + 2 * smoothing)
        likelihoods[level] = {
            "p_pass_given_correct": round(p_pass_correct, 4),
            "p_pass_given_incorrect": round(p_pass_incorrect, 4),
        }
    return likelihoods


def build_controller(
    likelihoods: dict[str, dict[str, float]],
    base_rate: float,
    costs: CostModel,
    horizon: int = 10,
) -> BayesianController:
    critics = {
        level: CriticLikelihood(
            p_pass_given_correct=lk["p_pass_given_correct"],
            p_pass_given_incorrect=lk["p_pass_given_incorrect"],
        )
        for level, lk in likelihoods.items()
    }
    transition = TransitionKernel(
        p_fix=base_rate,
        p_break=1.0 - base_rate,
    )
    return BayesianController(
        critic_likelihoods=critics,
        transition=transition,
        costs=costs,
        horizon=horizon,
        grid_size=1000,
    )


def build_multi_critic_controller(
    likelihoods: dict[str, dict[str, float]],
    base_rate: float,
    costs: CostModel,
    horizon: int = 10,
) -> MultiCriticBayesianController:
    critics = {
        level: CriticLikelihood(
            p_pass_given_correct=lk["p_pass_given_correct"],
            p_pass_given_incorrect=lk["p_pass_given_incorrect"],
        )
        for level, lk in likelihoods.items()
    }
    transition = TransitionKernel(
        p_fix=base_rate,
        p_break=1.0 - base_rate,
    )
    return MultiCriticBayesianController(
        critic_likelihoods=critics,
        transition=transition,
        costs=costs,
        horizon=horizon,
        grid_size=500,
    )


def _utils(results: list[EpisodeResult], reward: float) -> np.ndarray:
    return np.array(
        [(reward if r.resolved else 0) - r.total_cost for r in results],
        dtype=float,
    )


def _mean_se(arr: np.ndarray) -> tuple[float, float]:
    if len(arr) == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))


def _paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    diff = a - b
    mean, se = _mean_se(diff)
    t = mean / se if se > 0 else 0.0
    return mean, se, t


def run_single_alpha(
    episodes: dict[str, list[dict]],
    alpha: float,
    seed: int,
    costs: CostModel,
    horizon: int,
) -> dict:
    noised = noise_l2(episodes, alpha, seed)
    likelihoods = compute_likelihoods_from_episodes(noised)

    all_patches = [p for patches in noised.values() for p in patches]
    base_rate = (
        sum(1 for p in all_patches if p["ground_truth"] == 1) / len(all_patches)
    )
    prior = max(base_rate, 0.05)

    ctrl = build_controller(likelihoods, base_rate, costs, horizon)
    ctrl_mc = build_multi_critic_controller(likelihoods, base_rate, costs, horizon)

    b_res, mc_res, f_res, t1, t2, t3 = [], [], [], [], [], []
    for patches in noised.values():
        b_res.append(run_bayesian_policy(ctrl, patches, costs, prior))
        mc_res.append(run_multi_critic_policy(ctrl_mc, patches, costs, prior))
        f_res.append(run_fixed_pipeline(patches, costs))
        t1.append(run_threshold_policy(patches, costs, "L1_lint"))
        t2.append(run_threshold_policy(patches, costs, "L2_fast_test"))
        t3.append(run_threshold_policy(patches, costs, "L3_llm_review"))

    b_u = _utils(b_res, costs.reward)
    mc_u = _utils(mc_res, costs.reward)
    f_u = _utils(f_res, costs.reward)
    t1_u = _utils(t1, costs.reward)
    t2_u = _utils(t2, costs.reward)
    t3_u = _utils(t3, costs.reward)

    # Best single-threshold on marginal mean
    threshold_means = {
        "L1": t1_u.mean(),
        "L2": t2_u.mean(),
        "L3": t3_u.mean(),
    }
    best_thr_name = max(threshold_means, key=threshold_means.get)
    best_thr_u = {"L1": t1_u, "L2": t2_u, "L3": t3_u}[best_thr_name]

    return {
        "alpha": alpha,
        "prior": prior,
        "likelihoods": likelihoods,
        "l2_gap": (
            likelihoods["L2_fast_test"]["p_pass_given_correct"]
            - likelihoods["L2_fast_test"]["p_pass_given_incorrect"]
        ),
        "l3_gap": (
            likelihoods["L3_llm_review"]["p_pass_given_correct"]
            - likelihoods["L3_llm_review"]["p_pass_given_incorrect"]
        ),
        "Bayesian": {"mean": _mean_se(b_u), "utils": b_u},
        "BayesianMC": {"mean": _mean_se(mc_u), "utils": mc_u},
        "Fixed": {"mean": _mean_se(f_u), "utils": f_u},
        "Threshold(L1)": {"mean": _mean_se(t1_u), "utils": t1_u},
        "Threshold(L2)": {"mean": _mean_se(t2_u), "utils": t2_u},
        "Threshold(L3)": {"mean": _mean_se(t3_u), "utils": t3_u},
        "best_threshold": best_thr_name,
        "paired_vs_L2": _paired(b_u, t2_u),
        "paired_vs_L3": _paired(b_u, t3_u),
        "paired_vs_best": _paired(b_u, best_thr_u),
        "mc_paired_vs_L2": _paired(mc_u, t2_u),
        "mc_paired_vs_L3": _paired(mc_u, t3_u),
        "mc_paired_vs_best": _paired(mc_u, best_thr_u),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--out", default=None, help="Save results JSON here")
    args = parser.parse_args()

    episodes = load_episodes_lcb(Path(args.data))
    all_patches = [p for ps in episodes.values() for p in ps]
    base_rate = (
        sum(1 for p in all_patches if p["ground_truth"] == 1) / len(all_patches)
    )
    print(
        f"Loaded {len(episodes)} instances, {len(all_patches)} patches, "
        f"base_rate={base_rate:.3f}"
    )

    costs = CostModel()

    header = (
        f"\n{'alpha':>5} {'L2g':>5} {'L3g':>5} | "
        f"{'Bay(single)':>13} {'Bay(multi)':>13} "
        f"{'Thr(L2)':>13} {'Thr(L3)':>13} | "
        f"{'MC-best':>13} {'best':>5} {'verdict':>10}"
    )
    print(header)
    print("-" * len(header))

    all_results = []
    for alpha in args.alphas:
        r = run_single_alpha(episodes, alpha, args.seed, costs, args.horizon)
        all_results.append(r)

        b_m, b_se = r["Bayesian"]["mean"]
        mc_m, mc_se = r["BayesianMC"]["mean"]
        t2_m, t2_se = r["Threshold(L2)"]["mean"]
        t3_m, t3_se = r["Threshold(L3)"]["mean"]
        mc_d, mc_dse, mc_t = r["mc_paired_vs_best"]

        if abs(mc_t) < 2.0:
            verdict = "tie"
        elif mc_d > 0:
            verdict = "BayesianMC"
        else:
            verdict = f"Thr({r['best_threshold']})"

        print(
            f"{alpha:>5.2f} {r['l2_gap']:>5.2f} {r['l3_gap']:>5.2f} | "
            f"{b_m:>+8.1f}±{b_se:>3.1f} "
            f"{mc_m:>+8.1f}±{mc_se:>3.1f} "
            f"{t2_m:>+8.1f}±{t2_se:>3.1f} "
            f"{t3_m:>+8.1f}±{t3_se:>3.1f} | "
            f"{mc_d:>+7.2f}±{mc_dse:>3.2f} {r['best_threshold']:>5} "
            f"{verdict:>10}"
        )

    if args.out:
        serializable = []
        for r in all_results:
            rr = {k: v for k, v in r.items() if k != "likelihoods"}
            rr["likelihoods"] = r["likelihoods"]
            for pol in [
                "Bayesian",
                "BayesianMC",
                "Fixed",
                "Threshold(L1)",
                "Threshold(L2)",
                "Threshold(L3)",
            ]:
                rr[pol] = {
                    "mean": list(rr[pol]["mean"]),
                    "utils": rr[pol]["utils"].tolist(),
                }
            for key in [
                "paired_vs_L2",
                "paired_vs_L3",
                "paired_vs_best",
                "mc_paired_vs_L2",
                "mc_paired_vs_L3",
                "mc_paired_vs_best",
            ]:
                rr[key] = list(rr[key])
            serializable.append(rr)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"alphas": args.alphas, "seed": args.seed, "results": serializable}, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
