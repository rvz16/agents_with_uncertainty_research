"""Cost-vector balance search.

Goal: find c_ver values (within measurement-justified ranges) that produce
**balanced policy comparison histograms** — i.e., where multiple policies
beat `always_verify` AND multiple policies lose to it, rather than the
degenerate case where AV dominates everything.

Why this matters: if all eight policies sit below the AV baseline (the
`lcb_hard/gpt5_mini` pathology in our cell 13 output), the comparison
isn't informative — the cost vector is in a regime where verification is
so cheap that nothing beats just-verify-everything. The interesting story
sits at the inflection point where critic-based and Bayesian policies
can outperform AV but also where naive policies (best_of_N, fixed_pipeline)
can do worse. That's the regime where uncertainty-aware orchestration
matters.

Two measurement-derived mode regimes (Table tab:action_latency):

  FAST mode  (function-level + bug-fix benchmarks)
    Measured c_ver / c_gen ratios: 0.009 - 0.059
    Current §2.5: c_ver=5, c_gen=10, critic=1/1/1   (ratio 0.5)
    Sweep range: c_ver in [1, 100] (keep c_gen=10 fixed)

  SLOW mode  (SWE-Bench Lite + Verified)
    Measured: pooled median c_ver/c_gen = 0.19; heavy-suite p90 ≈ 68
    Current §2.5: c_ver=30, c_gen=10, critic=1/2/5  (ratio 3.0 after unification; was 6.0)
    Sweep range: c_ver in [5, 200] (keep c_gen=10 fixed; ratio 3.0)

The current §2.5 c_ver values sit ~10x higher than measured medians for
the FAST mode and within bimodal range for SLOW. The measured medians
likely under-represent the perceived cost (they ignore API spend,
batch-effect latencies, multi-tenant retry cost, etc.) — but our cost
vectors are abstract utility units that already bake in those factors.
This module's job is to find c_ver values in each mode that make the
policy comparison *informative*, not to pin a single "true" c_ver.

The balance metric (C: effective-competitor count):

  For each (benchmark, generator) cell, given policy deltas Δ_p vs AV:
    pos = #{p : Δ_p > +ε}     (policies meaningfully above AV)
    neg = #{p : Δ_p < -ε}     (policies meaningfully below AV)
    balance = min(pos, neg)   (both sides must have multiple non-trivial
                              members for the cell to count as "balanced")
  Default ε = 2.0 utility points. Ties broken by max(pos + neg).

Aggregation:
  Per mode (FAST/SLOW), recommend the c_ver that maximizes the average
  balance score across all cells in that mode.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

# Allow direct import from package root.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


# ---------------------------------------------------------------------------
# Mode definitions — measurement-derived sweep ranges (Table tab:action_latency)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CostMode:
    name: str
    c_gen: float       # generation cost (held fixed during the sweep)
    c_critic_l0: float
    c_critic_l2: float
    c_critic_l3: float
    c_ver_range: tuple[float, float]   # sweep range for verification cost
    c_ver_current: float               # current §2.5 value (annotation)
    benchmarks: tuple[str, ...]        # which benchmarks fall in this mode


# c_gen is unified across both modes (was c_gen=10 FAST, c_gen=5 SLOW in
# the original §2.5 vectors). The original asymmetry had no measurement
# basis — Table tab:action_latency actually shows SWE-Bench has the
# *highest* measured a_gen (10.0s vs 1.3s for MBPP+), so assigning SLOW
# the *lowest* c_gen was double-counting the FAST/SLOW asymmetry. Now the
# only structural difference between modes is in c_critic (Docker-level
# critics on SWE are measurably slower than function-level) and in the
# c_ver sweep range.
#
# c_ver sweep upper bound: capped at 0.9 · reward = 90. The previous sweep
# upper-bounds (FAST=100, SLOW=200) violated the methodology constraint
# c_ver < reward: when c_ver > reward, AV's utility is strictly negative
# even on correct patches (reward · Y − c_ver < 0 for any Y), so AV
# becomes trivially worst. The "balance" metric then saturates artificially
# at the high-c_ver end — picking c_vers from this regime as "balance-
# optimal" is meaningless because the comparison is degenerate-in-disguise.
# The 0.9 buffer keeps AV competitive on Y=1 patches across plausible
# priors. A stricter per-benchmark bound would be c_ver < reward · prior_Y1
# (so AV has positive expected utility), but priors vary per cell — the
# fixed 0.9·reward is a defensible mode-level cap.

FAST_MODE = CostMode(
    name="FAST",
    c_gen=10.0, c_critic_l0=1.0, c_critic_l2=1.0, c_critic_l3=1.0,
    c_ver_range=(1.0, 90.0), c_ver_current=5.0,
    benchmarks=("lcb_easy", "lcb_medium", "lcb_hard",
                "mbpp", "humaneval", "humanevalfix", "codecontests"),
)

SLOW_MODE = CostMode(
    name="SLOW",
    c_gen=10.0, c_critic_l0=1.0, c_critic_l2=2.0, c_critic_l3=5.0,
    c_ver_range=(5.0, 90.0), c_ver_current=30.0,
    benchmarks=("swe_lite", "swe_verified"),
)

MODES = (FAST_MODE, SLOW_MODE)


def mode_for_benchmark(benchmark: str) -> CostMode | None:
    for m in MODES:
        if benchmark in m.benchmarks:
            return m
    return None


# ---------------------------------------------------------------------------
# Balance metric — effective-competitor count
# ---------------------------------------------------------------------------

def balance_score(
    policy_deltas: Mapping[str, float],
    *,
    epsilon: float = 2.0,
    exclude_av: bool = True,
) -> dict:
    """Compute the balance score for one (cell, c_ver) configuration.

    Parameters
    ----------
    policy_deltas : {policy_name → Δ_vs_av}.  The Δ value for the
        `always_verify` policy itself is 0 by definition.
    epsilon : minimum |Δ| for a policy to count as "meaningfully" above or
        below AV. Default 2.0 utility points.
    exclude_av : if True (default), drop the `always_verify` entry from
        the count (it sits exactly at 0 by definition).

    Returns a dict with:
        n_above : # policies with Δ > +ε
        n_below : # policies with Δ < -ε
        n_neutral : # policies with |Δ| ≤ ε
        n_total : # policies counted (excludes AV iff exclude_av=True)
        balance : min(n_above, n_below) — the headline metric
        spread  : std(Δ) across counted policies (tiebreaker)

    Higher `balance` = better. A balance of 0 means at least one side is
    empty (either all winners or all losers). A balance of 3 means at
    least 3 winners AND at least 3 losers — informative comparison.
    """
    deltas = dict(policy_deltas)
    if exclude_av:
        deltas.pop("always_verify", None)
    vals = list(deltas.values())
    n_above = sum(1 for v in vals if v > epsilon)
    n_below = sum(1 for v in vals if v < -epsilon)
    n_neutral = sum(1 for v in vals if abs(v) <= epsilon)
    return {
        "n_above": n_above,
        "n_below": n_below,
        "n_neutral": n_neutral,
        "n_total": len(vals),
        "balance": min(n_above, n_below),
        "spread": float(np.std(vals)) if vals else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-cell c_ver sweep
# ---------------------------------------------------------------------------

def make_cost_model(mode: CostMode, c_ver: float):
    """Build a CostModel for the given mode at the given c_ver. The CostModel
    is imported lazily to avoid a circular import when this module is loaded
    early in the notebook (before analysis.controller is on the path)."""
    from analysis.controller import CostModel
    return CostModel(
        c_gen=mode.c_gen,
        c_L0=mode.c_critic_l0,
        c_L2=mode.c_critic_l2,
        c_L3=mode.c_critic_l3,
        c_ver=c_ver,
        reward=100.0,
    )


def sweep_c_ver_one_cell(
    cell_traj: Mapping[str, list[dict]],
    likes: Mapping,
    prior: float,
    mode: CostMode,
    *,
    c_ver_values: Sequence[float] | None = None,
    epsilon: float = 2.0,
    n_boot: int = 100,
) -> list[dict]:
    """Sweep c_ver over the mode's range for one (benchmark, generator) cell.

    Returns a list of dicts, one per c_ver value:
        {c_ver, balance, n_above, n_below, n_neutral, spread, policy_deltas}

    `policy_deltas` is the full per-policy dict so downstream can plot or
    re-compute custom metrics.
    """
    from analysis.lcb_sensitivity import run_policies

    if c_ver_values is None:
        lo, hi = mode.c_ver_range
        c_ver_values = list(np.linspace(lo, hi, 20))

    out = []
    for c_ver in c_ver_values:
        cost = make_cost_model(mode, c_ver)
        res = run_policies(cell_traj, likes, prior, cost, n_boot=n_boot)
        # Extract Δ vs AV per policy
        deltas = {p: (r.get("diff_vs_baseline") or 0.0)
                  for p, r in res.items()}
        bs = balance_score(deltas, epsilon=epsilon)
        out.append({
            "c_ver": float(c_ver),
            **bs,
            "policy_deltas": deltas,
        })
    return out


# ---------------------------------------------------------------------------
# Per-mode aggregation: balance-optimal c_ver across cells in a mode
# ---------------------------------------------------------------------------

def aggregate_mode_balance(
    per_cell_sweeps: Mapping[tuple[str, str], list[dict]],
    mode: CostMode,
) -> list[dict]:
    """Aggregate per-cell sweep results to find the c_ver value that
    maximizes the *mean* balance score across all cells in the mode.

    per_cell_sweeps : {(benchmark, generator) → sweep result}
        where each sweep result is the list returned by sweep_c_ver_one_cell

    Returns a list of dicts, one per c_ver value:
        {c_ver, mean_balance, mean_spread, n_cells, per_cell_balances}

    The returned list is sorted by c_ver ascending. The recommended c_ver
    is the one with the highest mean_balance (ties broken by mean_spread).
    """
    # Collect all c_ver values seen (should be the same grid per cell, but
    # be defensive about float comparisons).
    all_c_vers: list[float] = []
    for sweep in per_cell_sweeps.values():
        for row in sweep:
            if not any(abs(row["c_ver"] - c) < 1e-9 for c in all_c_vers):
                all_c_vers.append(row["c_ver"])
    all_c_vers.sort()

    out = []
    for c_ver in all_c_vers:
        balances, spreads, per_cell = [], [], {}
        for (b, g), sweep in per_cell_sweeps.items():
            if b not in mode.benchmarks:
                continue
            # Find the matching c_ver row
            row = next((r for r in sweep if abs(r["c_ver"] - c_ver) < 1e-9),
                       None)
            if row is None:
                continue
            balances.append(row["balance"])
            spreads.append(row["spread"])
            per_cell[f"{b}/{g}"] = row["balance"]
        if not balances:
            continue
        out.append({
            "c_ver": float(c_ver),
            "mean_balance": float(np.mean(balances)),
            "median_balance": float(np.median(balances)),
            "mean_spread": float(np.mean(spreads)),
            "n_cells": len(balances),
            "per_cell_balances": per_cell,
        })
    return out


def recommend_c_ver_for_mode(
    aggregated: Sequence[dict],
    *,
    min_balance: float = 1.0,
) -> dict | None:
    """Pick the c_ver that maximizes mean_balance across cells, ties broken
    by mean_spread. Returns None if no c_ver achieves min_balance.

    The min_balance threshold filters out trivial recommendations — if the
    best c_ver only achieves balance=0.5 (most cells degenerate), there's
    no point pretending to recommend it.
    """
    if not aggregated:
        return None
    eligible = [a for a in aggregated if a["mean_balance"] >= min_balance]
    if not eligible:
        return None
    # max by (mean_balance, mean_spread) tuple
    return max(eligible, key=lambda a: (a["mean_balance"], a["mean_spread"]))


__all__ = [
    "CostMode", "FAST_MODE", "SLOW_MODE", "MODES",
    "mode_for_benchmark",
    "balance_score",
    "make_cost_model",
    "sweep_c_ver_one_cell",
    "aggregate_mode_balance",
    "recommend_c_ver_for_mode",
]
