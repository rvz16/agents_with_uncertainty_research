#!/usr/bin/env python
"""Analyze K-shot sweep + bail-risk UQ.

Inputs (CC / gpt5_mini n=20):
- cc_live_kshot_Kall_offline.json: offline K=all log, used to:
  * derive offline+K-shot for K ∈ {0,2,5,10,all} by truncating the first K
    forced refines per cell, then recomputing utility
  * extract (Y_t, Y_{t+1}) pairs for bail-risk UQ
- cc_live_kshot_K{K}_online.json for K ∈ {2,5,10,all}: per-cell online runs
  (used directly for online+K-shot utility)
- cc_live_measured_gpt5mini_n20.json: baseline `dp_fitted` (≡ K=0)

For each K we want:
- offline+K-shot: utility derived from offline K=all log (truncated to first K)
- online+K-shot: utility from the online K={K} run
- paired Δ_Ū (online - offline) per K with bootstrap CI

For bail-risk UQ:
- From offline K=all obs_log, fit Beta posterior on p_catch
- Compare to threshold c_retry/R = (c_gen + c_ver)/R = 30/100 = 0.30
- Report mean, 95% credible interval, P(p_catch > 0.30)

Writes a markdown report to stdout (and saves a JSON summary).
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path

R = 100
C_GEN = 10                # AgentCostConfig.c_llm_call
C_VER = 5                 # AgentCostConfig.c_full_test
C_CRITIC = 1              # AgentCostConfig.c_critic_test
C_RETRY = C_GEN + C_VER   # one (generate, verify) = 15
THRESHOLD = C_RETRY / R   # 0.15

BASE = Path("bayesian_optimization_for_code_testing/agent-bugfix-bayes/sim_results")


def U(rec: dict) -> float:
    return R * int(rec["fixed"]) - rec["total_cost"]


def load(fn: str) -> dict:
    return json.load(open(BASE / fn))


def variant_records(d: dict, variant: str) -> dict:
    return {k.rsplit("|", 1)[0]: rec
            for k, rec in d["results"].items()
            if rec.get("variant") == variant}


def derive_offline_at_K(offline_all: dict, K: int) -> dict:
    """Derive offline+K-shot from the offline K=all log.

    Mechanic: walk each instance's action trace; the *first* forced refine
    pair is kept iff cumulative forced count so far ≤ K. If a forced refine
    is dropped, the episode is treated as if the planner had bailed directly
    at that point (no fix from the forced refine, no extra cost).

    Returns: dict[instance_id -> derived rec with updated fixed + total_cost]
    """
    out = {}
    # Walk in test-task order from the original "results" insertion order,
    # which matches the cell's per-instance iteration order.
    n_forced_so_far = 0
    for k, rec in offline_all["results"].items():
        tid, var = k.rsplit("|", 1)
        if var != f"kshot_K999_offline":
            continue
        new_rec = dict(rec)
        actions = rec.get("actions", [])
        # Find forced (generate_on_bail, verify_on_bail) pair indices.
        # In our runner they come together as a pair at the end of the trace
        # (or near it, depending on flow). Find both.
        i_gen = i_ver = None
        for i, a in enumerate(actions):
            if a.get("action") == "generate_on_bail":
                i_gen = i
            elif a.get("action") == "verify_on_bail":
                i_ver = i
        had_forced = i_gen is not None and i_ver is not None
        if had_forced:
            if n_forced_so_far < K:
                # Keep the forced refine — record stays as is.
                n_forced_so_far += 1
                out[tid] = new_rec
                continue
            # Drop the forced refine: subtract its cost; if it fixed, undo.
            new_cost = rec["total_cost"] - C_GEN - C_VER
            new_fixed = rec["fixed"] and rec.get("final_action") != "verify_on_bail_pass"
            new_final = "bail"  # would have bailed instead of forcing
            new_actions = [a for j, a in enumerate(actions)
                           if j != i_gen and j != i_ver]
            new_rec.update({
                "fixed": bool(new_fixed),
                "total_cost": float(new_cost),
                "final_action": new_final,
                "actions": new_actions,
                "derived_from_offline_Kall": True,
                "derived_K": K,
            })
            out[tid] = new_rec
        else:
            out[tid] = new_rec
    return out


def boot_paired(diffs: list, B: int = 4000, seed: int = 0):
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0)
    means = []
    for _ in range(B):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return means[int(B * 0.025)], means[int(B * 0.975)]


def beta_cdf(p: float, alpha: float, beta: float) -> float:
    """Regularized incomplete beta I_p(α, β). Uses math.lgamma."""
    # Use scipy if available, else fall back to a series.
    try:
        from scipy.special import betainc
        return float(betainc(alpha, beta, p))
    except Exception:
        pass
    # Continued fraction (Lentz) for I_p(α, β). Good enough for our use.
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    if p > (alpha + 1) / (alpha + beta + 2):
        return 1.0 - beta_cdf(1.0 - p, beta, alpha)
    # Power series at p=0
    eps = 1e-12
    log_bt = (math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
              + alpha * math.log(p) + beta * math.log(1 - p))
    bt = math.exp(log_bt)
    qab = alpha + beta
    qap = alpha + 1.0
    qam = alpha - 1.0
    c = 1.0
    d = 1.0 - qab * p / qap
    if abs(d) < eps:
        d = eps
    d = 1.0 / d
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (beta - m) * p / ((qam + m2) * (alpha + m2))
        d = 1.0 + aa * d
        if abs(d) < eps: d = eps
        c = 1.0 + aa / c
        if abs(c) < eps: c = eps
        d = 1.0 / d
        h *= d * c
        aa = -(alpha + m) * (qab + m) * p / ((alpha + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < eps: d = eps
        c = 1.0 + aa / c
        if abs(c) < eps: c = eps
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return bt * h / alpha


def beta_ci(alpha: float, beta: float, lo_q: float = 0.025, hi_q: float = 0.975) -> tuple:
    """95% credible interval via root-finding on beta_cdf."""
    def find_q(q):
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if beta_cdf(mid, alpha, beta) < q:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    return find_q(lo_q), find_q(hi_q)


def main():
    # ---- Load all cells we have ----
    try:
        offline_all = load("cc_live_kshot_Kall_offline.json")
    except FileNotFoundError:
        print("ERROR: cc_live_kshot_Kall_offline.json not found — has the offline run finished?",
              file=sys.stderr)
        sys.exit(1)

    online_at_K = {}
    for K in (2, 5, 10, 999):
        fn = f"cc_live_kshot_K{'all' if K == 999 else K}_online.json"
        try:
            online_at_K[K] = load(fn)
        except FileNotFoundError:
            print(f"WARN: {fn} not found, skipping K={K} online cell", file=sys.stderr)

    # Baseline: dp_fitted from measured run
    try:
        baseline_d = load("cc_live_measured_gpt5mini_n20.json")
        baseline = variant_records(baseline_d, "dp_fitted")
    except FileNotFoundError:
        baseline = {}
        print("WARN: cc_live_measured_gpt5mini_n20.json not found; K=0 baseline unavailable",
              file=sys.stderr)

    # ---- K-shot sweep ----
    print("# K-shot active calibration on CC / gpt5_mini (n=20)\n")
    print("Mechanic: when DP picks `bail_out` and the cell has < K forced refines so far,")
    print("force one (generate, verify) before bailing. Log (Y_t=0, Y_{t+1}). Online mode")
    print("updates a Beta posterior on p_fix_broken; offline mode keeps the kernel frozen.\n")

    K_list = [0, 2, 5, 10, 999]
    table = []
    for K in K_list:
        # offline = derive from offline-Kall log truncated to first K forced refines
        offline_at = derive_offline_at_K(offline_all, K)
        # online = per-K run, or = offline at K=0 (no forced refines → no updates)
        if K == 0:
            online_at = offline_at
        elif K in online_at_K:
            online_at = variant_records(online_at_K[K], f"kshot_K{K}_online")
        else:
            online_at = {}

        # paired Δ on matched IDs
        matched = sorted(set(offline_at) & set(online_at))
        diffs = [U(online_at[t]) - U(offline_at[t]) for t in matched]
        off_U = [U(offline_at[t]) for t in matched]
        on_U = [U(online_at[t]) for t in matched]
        n = len(matched)
        if n == 0:
            table.append({"K": K, "n": 0})
            continue
        off_mean = statistics.mean(off_U)
        on_mean = statistics.mean(on_U)
        delta = statistics.mean(diffs)
        lo, hi = boot_paired(diffs)
        off_fix = sum(int(offline_at[t]["fixed"]) for t in matched) / n
        on_fix = sum(int(online_at[t]["fixed"]) for t in matched) / n
        off_cost = sum(offline_at[t]["total_cost"] for t in matched) / n
        on_cost = sum(online_at[t]["total_cost"] for t in matched) / n

        table.append({
            "K": K, "n": n,
            "offline_U": off_mean, "online_U": on_mean,
            "offline_fix": off_fix, "online_fix": on_fix,
            "offline_cost": off_cost, "online_cost": on_cost,
            "delta": delta, "ci_lo": lo, "ci_hi": hi,
        })

    # Format table
    print("| K | n | offline Ū | online Ū | offline fix% | online fix% | offline cost | online cost | Δ (on−off) | 95% CI |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in table:
        if row.get("n", 0) == 0:
            print(f"| {row['K']} | 0 | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        sig = " ✓" if (row["ci_lo"] > 0 or row["ci_hi"] < 0) else ""
        klabel = "all" if row["K"] == 999 else str(row["K"])
        print(f"| {klabel} | {row['n']} | {row['offline_U']:+.2f} | {row['online_U']:+.2f} "
              f"| {row['offline_fix']*100:.0f}% | {row['online_fix']*100:.0f}% "
              f"| {row['offline_cost']:.2f} | {row['online_cost']:.2f} "
              f"| {row['delta']:+.2f} | [{row['ci_lo']:+.2f}, {row['ci_hi']:+.2f}]{sig} |")

    # ---- Bail-risk UQ ----
    print("\n# Bail-risk UQ from forced bail audits\n")
    obs_log = offline_all.get("kshot_state", {}).get("obs_log", [])
    if not obs_log:
        print("(No obs_log available in offline-Kall run; UQ skipped.)")
    else:
        n_obs = len(obs_log)
        n_catch = sum(o["y_t1"] for o in obs_log)
        n_miss = n_obs - n_catch
        # Beta(1+n_catch, 1+n_miss) — uninformative prior so posterior reflects audits only
        alpha_post = 1.0 + n_catch
        beta_post = 1.0 + n_miss
        mean = alpha_post / (alpha_post + beta_post)
        ci_lo, ci_hi = beta_ci(alpha_post, beta_post)
        p_unsafe = 1.0 - beta_cdf(THRESHOLD, alpha_post, beta_post)

        print(f"Threshold C_retry/R = ({C_GEN}+{C_VER})/{R} = {THRESHOLD:.2f}\n")
        print(f"Forced bail audits: n={n_obs} ({n_catch} catches, {n_miss} misses)\n")
        print(f"Posterior on p_catch: Beta({alpha_post:.0f}, {beta_post:.0f})")
        print(f"  mean = {mean:.3f}")
        print(f"  95% credible interval = [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"  **P(p_catch > {THRESHOLD:.2f}) = {p_unsafe:.3f}**\n")
        if p_unsafe > 0.5:
            print(f"Interpretation: posterior probability that bailing is *unsafe* (i.e., expected reward of one refine exceeds its cost) is {p_unsafe:.0%}. Bail is risky on this cell.")
        else:
            print(f"Interpretation: posterior probability that bailing is unsafe is {p_unsafe:.0%}. Bail is well-justified on this cell.")

    # Save JSON summary
    out = {
        "kshot_table": table,
        "bail_risk_uq": {
            "threshold": THRESHOLD,
            "n_obs": n_obs if obs_log else 0,
            "n_catch": n_catch if obs_log else 0,
            "n_miss": n_miss if obs_log else 0,
            "alpha_post": alpha_post if obs_log else None,
            "beta_post": beta_post if obs_log else None,
            "mean": mean if obs_log else None,
            "ci_lo": ci_lo if obs_log else None,
            "ci_hi": ci_hi if obs_log else None,
            "P_unsafe": p_unsafe if obs_log else None,
        },
    }
    out_path = BASE / "cc_live_kshot_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved JSON summary: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
