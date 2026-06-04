"""Diagnostic table per supervisor request.

For each (live cell, method) report:
  - % instances where first action is verify or bail (no refinement happens)
  - mean # refinements (generate actions) per instance
  - mean # online kernel updates (verify-following-prior-verify) per instance
  - # instances where dp_fitted action sequence differs from offline measured (flips)
  - Ū delta vs offline measured
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Any

R = 100  # reward

# ---------------------------------------------------------------------------
# Result file inventory
# ---------------------------------------------------------------------------

BAYES_DIR = Path("/Users/victor/Documents/vs_files/research/article_implementation/"
                  "agents_with_uncertainty_research/bayesian_optimization_for_code_testing/"
                  "agent-bugfix-bayes/sim_results")
LCB_DIR = Path("/Users/victor/Documents/vs_files/research/article_implementation/"
                "agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing/"
                "data/online_vs_offline/eps_explore")

# (cell, method) -> result_path. method names match supervisor's table.
RESULTS = {
    # LCB-hard / haiku45 (uses run_synthesis_live.py — different action schema)
    ("LCB-hard / haiku45", "measured"):           LCB_DIR / "lcb_hard_haiku45_thompson_n20.json",  # placeholder — see note
    ("LCB-hard / haiku45", "online"):             LCB_DIR / "lcb_hard_haiku45_online_eps020_n20.json",
    ("LCB-hard / haiku45", "conditional"):        LCB_DIR / "lcb_hard_haiku45_conditional_n20.json",
    ("LCB-hard / haiku45", "cond_online"):        LCB_DIR / "lcb_hard_haiku45_conditional_online_n20.json",
    ("LCB-hard / haiku45", "thompson"):           LCB_DIR / "lcb_hard_haiku45_thompson_n20.json",
    # CC / haiku45
    ("CC / haiku45", "measured"):                 None,  # use thompson file's simple variant utilities for ref
    ("CC / haiku45", "thompson"):                 BAYES_DIR / "cc_live_thompson_haiku45_n20.json",
    ("CC / haiku45", "conditional"):              BAYES_DIR / "cc_live_conditional_haiku45_n20.json",
    # CC / gpt5_mini
    ("CC / gpt5_mini", "measured"):               BAYES_DIR / "cc_live_measured_gpt5mini_n20.json",
    ("CC / gpt5_mini", "online"):                 BAYES_DIR / "cc_live_online_gpt5mini_n20.json",
    ("CC / gpt5_mini", "conditional"):            BAYES_DIR / "cc_live_conditional_gpt5mini_n20.json",
    ("CC / gpt5_mini", "thompson"):               BAYES_DIR / "cc_live_thompson_gpt5mini_n20.json",
    ("CC / gpt5_mini", "thompson_cond"):          BAYES_DIR / "cc_live_thompson_conditional_gpt5mini_n20.json",
    ("CC / gpt5_mini", "online+refine_bail"):     BAYES_DIR / "cc_live_online_rob_gpt5mini_n20.json",
    # CC / sonnet45
    ("CC / sonnet45", "measured"):                BAYES_DIR / "cc_live_measured_sonnet45_n20.json",
    ("CC / sonnet45", "online"):                  BAYES_DIR / "cc_live_online_sonnet45_n20.json",
    ("CC / sonnet45", "conditional"):             BAYES_DIR / "cc_live_conditional_sonnet45_n20.json",
    ("CC / sonnet45", "thompson"):                BAYES_DIR / "cc_live_thompson_sonnet45_n20.json",
    # HumanEvalFix / haiku45
    ("HumanEvalFix / haiku45", "measured"):       BAYES_DIR / "humanevalfix_live_measured_haiku45_n20.json",
    ("HumanEvalFix / haiku45", "online"):         BAYES_DIR / "humanevalfix_live_online_haiku45_n20.json",
    ("HumanEvalFix / haiku45", "conditional"):    BAYES_DIR / "humanevalfix_live_conditional_haiku45_n20.json",
    ("HumanEvalFix / haiku45", "thompson"):       BAYES_DIR / "humanevalfix_live_thompson_haiku45_n20.json",
}

# Modes that use a running online estimator (= online_kernel.update is called)
ONLINE_MODES = {"online", "thompson", "cond_online", "online+refine_bail"}
THOMPSON_LIKE = {"thompson", "thompson_cond"}


# ---------------------------------------------------------------------------
# Per-instance diagnostics
# ---------------------------------------------------------------------------

def get_actions(r: dict) -> list[dict]:
    return r.get("actions", [])

def first_action_type(actions: list[dict]) -> str:
    if not actions:
        return "none"
    a = actions[0]
    raw = a.get("action") or ""
    if raw.startswith("generate"):
        return "generate"
    if raw.startswith("critic"):
        return "critic"
    if raw == "verify":
        return "verify"
    if raw in ("bail", "bail_out"):
        return "bail"
    # Synthesis_live action schema is slightly different
    if "verify_pass" in a:
        return "verify"
    return raw or "unknown"

def n_refinements(actions: list[dict]) -> int:
    """Number of generate actions (= multi-step refinements taken).

    Includes refine-on-bail forced generate (action='generate_on_bail').
    """
    n = 0
    for a in actions:
        raw = a.get("action") or ""
        if raw.startswith("generate") or raw == "generate_on_bail":
            n += 1
    return n

def n_verifies(actions: list[dict]) -> int:
    return sum(1 for a in actions
               if (a.get("action") or "") in ("verify", "verify_on_bail")
               or "verify_pass" in a)

def n_online_updates(actions: list[dict], runner_kind: str) -> int:
    """Number of (Y_t, Y_{t+1}) updates the online estimator received.

    CC/synthesis runner: prev_Y starts None → 1st verify is no-update,
    each subsequent verify triggers update. So updates = max(0, n_verifies - 1).
    HumanEvalFix runner: prev_Y starts at 0 by construction → every verify triggers
    an update. So updates = n_verifies.
    """
    n_v = n_verifies(actions)
    if runner_kind == "humanevalfix":
        return n_v
    return max(0, n_v - 1)

def utility(r: dict) -> float:
    return (R if r.get("fixed", False) else 0) - r.get("total_cost", 0)

def parse_results(path: Path) -> dict[str, dict]:
    """Return {task_id: {variant: result_dict}}."""
    if path is None or not path.exists():
        return {}
    d = json.load(open(path))
    out: dict[str, dict] = {}
    for key, rec in d.get("results", {}).items():
        tid, var = key.split("|")
        out.setdefault(tid, {})[var] = rec
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def runner_kind_for_cell(cell: str) -> str:
    if cell.startswith("HumanEvalFix"):
        return "humanevalfix"
    if cell.startswith("LCB-hard"):
        return "synthesis"
    return "cc"

def diagnose(cell: str, method: str, path: Path,
              offline_results: dict | None) -> dict:
    data = parse_results(path)
    if not data:
        return {}
    runner_kind = runner_kind_for_cell(cell)
    rows = []
    for tid, by_var in data.items():
        r = by_var.get("dp_fitted")
        if r is None:
            continue
        acts = get_actions(r)
        rows.append({
            "tid": tid,
            "first": first_action_type(acts),
            "n_ref": n_refinements(acts),
            "n_upd": n_online_updates(acts, runner_kind) if method in ONLINE_MODES else 0,
            "u": utility(r),
            "fixed": int(r.get("fixed", False)),
            "actions": tuple((a.get("action") or "") for a in acts),
        })
    if not rows:
        return {}

    n = len(rows)
    pct_verify_or_bail = sum(1 for r in rows
                              if r["first"] in ("verify", "bail")) / n * 100
    mean_ref = sum(r["n_ref"] for r in rows) / n
    mean_upd = sum(r["n_upd"] for r in rows) / n
    mean_u = sum(r["u"] for r in rows) / n
    n_fixed = sum(r["fixed"] for r in rows)

    # Action flips vs offline measured (action-sequence-different)
    n_flips = 0
    if offline_results and method != "measured":
        for r in rows:
            tid = r["tid"]
            off = offline_results.get(tid, {}).get("dp_fitted")
            if off is None: continue
            off_acts = tuple((a.get("action") or "") for a in get_actions(off))
            if off_acts != r["actions"]:
                n_flips += 1

    # Δ vs offline measured
    delta_u = None
    if offline_results and method != "measured":
        offline_us = []
        method_us = []
        for r in rows:
            tid = r["tid"]
            off = offline_results.get(tid, {}).get("dp_fitted")
            if off is None: continue
            offline_us.append(utility(off))
            method_us.append(r["u"])
        if offline_us:
            import statistics
            delta_u = statistics.mean(method_us) - statistics.mean(offline_us)

    return {
        "n": n,
        "pct_verify_or_bail": pct_verify_or_bail,
        "mean_refinements": mean_ref,
        "mean_online_updates": mean_upd,
        "n_flips_vs_offline": n_flips,
        "Ubar": mean_u,
        "n_fixed": n_fixed,
        "delta_u_vs_offline": delta_u,
    }


def main():
    # Group by cell, gather offline reference
    by_cell: dict[str, dict[str, Path]] = defaultdict(dict)
    for (cell, method), path in RESULTS.items():
        by_cell[cell][method] = path

    rows_out = []
    for cell, methods in by_cell.items():
        # Choose offline reference (the file used for "measured" or fallback)
        offline_path = methods.get("measured")
        # LCB-hard uses synthesis runner where measured wasn't tested → use thompson sim as proxy
        if offline_path is None and cell == "CC / haiku45":
            offline_path = methods.get("thompson")  # measured = thompson on this cell
        offline_results = parse_results(offline_path) if offline_path else {}

        for method, path in methods.items():
            diag = diagnose(cell, method, path, offline_results)
            if not diag:
                continue
            rows_out.append({"cell": cell, "method": method, **diag})

    # Pretty print as markdown
    print("| Cell | Method | n | % first=verify/bail | mean refinements | "
          "**mean online updates** | flips vs offline | $\\bar U$ | Δ vs offline |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows_out:
        du = "—" if r["delta_u_vs_offline"] is None else f"{r['delta_u_vs_offline']:+.2f}"
        print(f"| {r['cell']} | {r['method']} | {r['n']} | "
              f"{r['pct_verify_or_bail']:.0f}% | "
              f"{r['mean_refinements']:.2f} | "
              f"**{r['mean_online_updates']:.2f}** | "
              f"{r['n_flips_vs_offline']} | "
              f"{r['Ubar']:+.2f} | {du} |")


if __name__ == "__main__":
    main()
