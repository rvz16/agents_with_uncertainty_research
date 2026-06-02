#!/usr/bin/env python3
"""Upload experiment artifacts to wandb.

Walks the local data tree (synced from cluster via sync_from_cluster.sh)
and creates one wandb run per (experiment_type, benchmark, generator, ...)
cell per the SCHEMA.md convention.

Idempotent: skips runs that already exist in the target project (matched
by run name). Use --force to overwrite.

Usage:
  # Upload everything (orchestration + abbo tracks):
  python upload_runs.py

  # Upload only one track:
  python upload_runs.py --track orchestration
  python upload_runs.py --track abbo

  # Upload only one experiment type:
  python upload_runs.py --experiment calibration
  python upload_runs.py --experiment iter
  python upload_runs.py --experiment policy_comparison

  # Dry-run (list what would be uploaded, do nothing):
  python upload_runs.py --dry-run

  # Re-upload (replaces existing runs):
  python upload_runs.py --force

  # Limit to specific cells (debugging):
  python upload_runs.py --benchmark lcb_hard --generator gpt5_mini
"""

import argparse
import json
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Iterable

import wandb

ENTITY = "nlpresearch.group"
PROJECT = "orchestration-hypothesis-testing"

# Anchored at this file's grandparent (the experiment root)
EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_ROOT.parent.parent

# Authoritative results store. The working-tree `experiments/orchestration/data/`
# only retains raw `critic_results.jsonl` after the emnlp2026 reorg stripped the
# derived artifacts; the COMPLETE per-cell results (likelihood_tables.json,
# transition_kernel*.json, policy_comparison*.json, sensitivity.{json,csv}, ...)
# live in the local server backup `data_archive/artem1_repo/...`. Default to that;
# override with --data-root. (qwen2.5-32b SWE cells: the harness-backfilled Y
# labels are canonical only in data_archive/runpod1_repo — point a second
# `--data-root ... --benchmark swe_*` run there for those two cells.)
_ARCHIVE_DATA = (REPO_ROOT / "data_archive" / "artem1_repo" / "experiments"
                 / "orchestration_hypothesis_testing" / "data")
_LOCAL_DATA = EXPERIMENT_ROOT / "data"
DATA_ROOT = _ARCHIVE_DATA if _ARCHIVE_DATA.exists() else _LOCAL_DATA
ABBO_RESULTS = REPO_ROOT / "bayesian_optimization_for_code_testing" / "agent-bugfix-bayes" / "sim_results"

GENERATORS = [
    "gpt5_mini", "qwen3_coder", "haiku45", "sonnet45",
    "qwen25_32b", "gpt_oss_20b_local",
]
ABBO_GENERATORS = ["gpt_oss_20b"]

# (benchmark_key, data_subdir)
ORCH_CALIBRATION_BENCHMARKS = [
    ("lcb_hard",     "lcb_calibration_v2"),
    ("lcb_medium",   "lcb_calibration_medium"),
    ("lcb_easy",     "lcb_calibration_easy"),
    ("mbpp",         "mbpp_calibration"),
    ("humaneval",    "humaneval_calibration"),
    ("humanevalfix", "humanevalfix_calibration"),
    ("codecontests", "codecontests_calibration"),
    ("swe_lite",     "swebench_lite"),
    ("swe_verified", "swebench_verified"),
]

ORCH_ITER_BENCHMARKS = [
    ("lcb_hard",     "lcb_calibration_v2_iter"),
    ("lcb_medium",   "lcb_calibration_medium_iter"),
    ("lcb_easy",     "lcb_calibration_easy_iter"),
    ("swe_lite",     "swebench_lite_iter"),
    ("swe_verified", "swebench_verified_iter"),
]

ORCH_REALBASELINES_BENCHMARKS = [
    ("lcb_hard",     "lcb_calibration_hard_realbaselines"),
    ("lcb_medium",   "lcb_calibration_medium_realbaselines"),
    ("lcb_easy",     "lcb_calibration_easy_realbaselines"),
    ("mbpp",         "mbpp_realbaselines"),
    ("humaneval",    "humaneval_realbaselines"),
    ("humanevalfix", "humanevalfix_realbaselines"),
    ("codecontests", "codecontests_realbaselines"),
    ("swe_lite",     "swebench_lite_realbaselines"),
    ("swe_verified", "swebench_verified_realbaselines"),
]

POLICY_NAMES = [
    "always_verify", "best_of_3", "threshold_L0", "threshold_L2",
    "threshold_L3", "fixed_pipeline", "bayesian_greedy", "bayesian_DP",
]

DEFAULT_COSTS = {"cost_R": 100, "cost_ver": 30, "cost_gen": 5,
                 "cost_L0": 1, "cost_L2": 2, "cost_L3": 5}

log = logging.getLogger("upload")

def rel(path):
    """Repo-relative provenance string; falls back to abs path for external --data-root."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(path).resolve())



# ---------------------------------------------------------------------------
# Helpers

def load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        log.warning(f"Bad JSON in {p}: {e}")
        return None


def line_count(p: Path) -> int:
    if not p.exists():
        return 0
    with p.open("rb") as f:
        return sum(1 for _ in f)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def existing_run_names(api: "wandb.Api") -> set[str]:
    """Set of names of runs that already exist in the project."""
    try:
        runs = api.runs(f"{ENTITY}/{PROJECT}")
        return {r.name for r in runs}
    except Exception as e:
        log.warning(f"Could not list existing runs: {e}")
        return set()


def safe_init(name: str, config: dict, tags: list[str],
              existing: set[str], force: bool, dry_run: bool):
    if dry_run:
        log.info(f"  [DRY-RUN] would create run: {name}")
        return None
    if name in existing and not force:
        log.info(f"  skip (exists): {name}")
        return None
    if name in existing and force:
        log.info(f"  overwriting: {name}")
        # wandb auto-versions runs with the same name; the old one stays but
        # the new one becomes the latest. We don't delete the old to preserve
        # history.
    log.info(f"  uploading: {name}")
    return wandb.init(
        entity=ENTITY, project=PROJECT, name=name,
        config=config, tags=tags, reinit=True,
    )


def maybe_log_artifact(run, path: Path, artifact_type: str, name: str | None = None):
    """Attach a local file to the run as a wandb Artifact."""
    if run is None or not path.exists():
        return
    art = wandb.Artifact(name=name or path.stem, type=artifact_type)
    art.add_file(str(path))
    run.log_artifact(art)


def maybe_log_dir_artifact(run, path: Path, artifact_type: str, name: str):
    """Attach an entire directory as a single wandb Artifact (one art entry per file)."""
    if run is None or not path.exists() or not path.is_dir():
        return
    files = [p for p in path.rglob("*") if p.is_file()]
    if not files:
        return
    art = wandb.Artifact(name=name, type=artifact_type)
    art.add_dir(str(path))
    run.log_artifact(art)
    return len(files)


def compute_likelihoods_from_critic_results(path: Path) -> dict | None:
    """Beta(1,1)-smoothed prior + per-critic gaps from critic_results.jsonl.

    Used as fallback when the per-cell likelihood_tables.json was not synced
    locally. Matches the schema produced by calibrate_from_spotcheck.py.
    """
    if not path.exists():
        return None
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        return None

    def _beta_smooth(k: int, n: int) -> float:
        # Beta(1,1) Laplace smoothing
        return (k + 1) / (n + 2)

    Y_vals = [bool(r.get("Y")) for r in rows]
    n_records = len(rows)
    n_y1 = sum(Y_vals)
    n_y0 = n_records - n_y1
    prior = _beta_smooth(n_y1, n_records)

    critics = ["L0_syntax", "L1_lint", "L2_public_tests", "L3_llm_review"]
    cl: dict[str, dict[str, float]] = {}
    for c in critics:
        verdicts = [bool(r.get(c)) for r in rows if c in r and r.get(c) is not None]
        if len(verdicts) != n_records:
            cl[c] = {"P_pass_given_Y1": None, "P_pass_given_Y0": None, "gap": None}
            continue
        tp = sum(1 for v, y in zip(verdicts, Y_vals) if v and y)
        fn = sum(1 for v, y in zip(verdicts, Y_vals) if not v and y)
        fp = sum(1 for v, y in zip(verdicts, Y_vals) if v and not y)
        tn = sum(1 for v, y in zip(verdicts, Y_vals) if not v and not y)
        p1 = _beta_smooth(tp, tp + fn) if (tp + fn) > 0 else None
        p0 = _beta_smooth(fp, fp + tn) if (fp + tn) > 0 else None
        gap = (p1 - p0) if (p1 is not None and p0 is not None) else None
        cl[c] = {"P_pass_given_Y1": p1, "P_pass_given_Y0": p0, "gap": gap,
                 "TP": tp, "FN": fn, "FP": fp, "TN": tn}

    return {
        "generator": rows[0].get("generator"),
        "n_records": n_records,
        "n_resolved": n_y1,
        "prior_Y1": prior,
        "critic_likelihoods": cl,
        "smoothing": "Beta(1,1) [computed inline from critic_results.jsonl]",
    }


# ---------------------------------------------------------------------------
# CALIBRATION (single-shot) — one run per (benchmark, generator)

def l3_per_reviewer_summary(cell: Path) -> dict:
    """Per-reviewer L3 informativeness gap from L3_sweep_likelihoods.json
    (LCB cells only; absent elsewhere -> {}). Adds W&B summary fields
    L3_gap__<rev>, L3_p_pass_y1__<rev>, L3_p_pass_y0__<rev>, L3_n_y1/__y0
    so the L3-reviewer-invariance analysis is W&B-native."""
    d = load_json(cell / "L3_sweep_likelihoods.json")
    if not d or not isinstance(d.get("per_reviewer"), dict):
        return {}
    out: dict = {}
    for rev, st in d["per_reviewer"].items():
        if not isinstance(st, dict):
            continue
        for src, dst in (("gap", "L3_gap__%s"),
                         ("P_pass_given_Y1", "L3_p_pass_y1__%s"),
                         ("P_pass_given_Y0", "L3_p_pass_y0__%s"),
                         ("n_y1", "L3_n_y1__%s"),
                         ("n_y0", "L3_n_y0__%s")):
            if st.get(src) is not None:
                out[dst % rev] = st[src]
    return out


def upload_calibration_orch(args, existing):
    log.info("=== Calibration runs (track:orchestration) ===")
    for bench, subdir in ORCH_CALIBRATION_BENCHMARKS:
        if args.benchmark and args.benchmark != bench:
            continue
        for gen in GENERATORS:
            if args.generator and args.generator != gen:
                continue
            cell = DATA_ROOT / subdir / gen
            lt = cell / "likelihood_tables.json"
            tk = cell / "transition_kernel_iid_baseline.json"
            cr = cell / "critic_results.jsonl"

            ll = load_json(lt)
            ll_source = "likelihood_tables.json"
            if ll is None:
                # Fall back to computing inline from critic_results.jsonl
                ll = compute_likelihoods_from_critic_results(cr)
                ll_source = "computed_from_critic_results"
                if ll is None:
                    log.debug(f"  miss: neither {lt} nor {cr} usable")
                    continue
                log.info(f"  {bench}/{gen}: computed likelihoods inline from critic_results.jsonl (n={ll['n_records']})")
            tk_data = load_json(tk) or {}
            cl = ll.get("critic_likelihoods", {})

            def gap(critic: str) -> float | None:
                v = cl.get(critic, {})
                if not isinstance(v, dict):
                    return None
                return v.get("gap")

            name = f"calibration__orchestration__{bench}__{gen}"
            config = {
                "track": "orchestration",
                "experiment_type": "calibration",
                "benchmark": bench, "generator": gen,
                "n_instances": ll.get("n_instances", ll.get("n_records", 0) // 3),
                "k_patches": 3, "seed": 42,
                "likelihoods_source": ll_source,
                "git_sha": git_sha(),
                "data_source": rel(cell),
                **DEFAULT_COSTS,
            }
            tags = ["track:orchestration", "experiment:calibration",
                    f"benchmark:{bench}", f"generator:{gen}"]
            run = safe_init(name, config, tags, existing, args.force, args.dry_run)
            if run is None:
                continue

            run.summary["prior_Y1"]       = ll.get("prior_Y1")
            run.summary["L0_gap"]         = gap("L0_syntax")
            run.summary["L1_gap"]         = gap("L1_lint")
            run.summary["L2_gap"]         = gap("L2_public_tests") or gap("L2_fast_test")
            run.summary["L3_gap"]         = gap("L3_llm_review")
            run.summary["n_records"]      = ll.get("n_records", 0)
            run.summary["n_resolved"]     = ll.get("n_resolved", 0)
            run.summary["iid_baseline_P_fix"]   = tk_data.get("kernel_all", {}).get("P_fix_given_broken")
            run.summary["iid_baseline_P_break"] = tk_data.get("kernel_all", {}).get("P_break_given_correct")
            for _k, _v in l3_per_reviewer_summary(cell).items():
                run.summary[_k] = _v

            # Attach the aggregate likelihoods AND the raw per-instance
            # critic_results.jsonl. The aggregate alone cannot drive a policy
            # re-simulation at arbitrary cost vectors (e.g. the R x c_ver grid);
            # the raw per-(instance, patch) verdicts + Y are required. SCHEMA.md
            # lists both as intended per-run calibration artifacts.
            if lt.exists():
                maybe_log_artifact(run, lt, "calibration", f"{bench}_{gen}_likelihood_tables")
            if cr.exists():
                maybe_log_artifact(run, cr, "calibration", f"{bench}_{gen}_critic_results")
            if tk.exists():
                maybe_log_artifact(run, tk, "calibration", f"{bench}_{gen}_iid_kernel")
            # Raw LLM responses (per-patch text) — uploaded as a separate
            # artifact so it's downloadable without pulling the whole project.
            raw_dir = cell / "raw_responses"
            n_raw = maybe_log_dir_artifact(run, raw_dir, "raw_responses",
                                           f"{bench}_{gen}_raw_responses")
            if n_raw:
                run.summary["n_raw_responses_uploaded"] = n_raw
            run.finish()


# ---------------------------------------------------------------------------
# ITER (single-method or per-method) — one run per (benchmark, generator, method)

def upload_iter_orch(args, existing):
    log.info("=== Iter runs (track:orchestration) ===")

    # Single-method iter (old layout: one transition_kernel.json per cell)
    for bench, subdir in ORCH_ITER_BENCHMARKS:
        if args.benchmark and args.benchmark != bench:
            continue
        for gen in GENERATORS:
            if args.generator and args.generator != gen:
                continue
            cell = DATA_ROOT / subdir / gen
            tk = cell / "transition_kernel.json"
            ir = cell / "iter_records.jsonl"
            if not tk.exists():
                continue
            tkd = load_json(tk) or {}
            ka = tkd.get("kernel_all", {})
            n_pairs = ka.get("n_pairs") or sum((ka.get("raw_counts") or {}).values())

            name = f"iter__orchestration__{bench}__{gen}__single_method"
            config = {
                "track": "orchestration",
                "experiment_type": "iter",
                "benchmark": bench, "generator": gen,
                "method": "single_method",
                "git_sha": git_sha(),
                "data_source": rel(cell),
            }
            tags = ["track:orchestration", "experiment:iter",
                    f"benchmark:{bench}", f"generator:{gen}",
                    "method:single_method"]
            run = safe_init(name, config, tags, existing, args.force, args.dry_run)
            if run is None:
                continue

            run.summary["P_fix_given_broken"]    = ka.get("P_fix_given_broken")
            run.summary["P_break_given_correct"] = ka.get("P_break_given_correct")
            run.summary["n_pairs"] = n_pairs
            counts = ka.get("raw_counts", {})
            run.summary["n_fix"]              = counts.get("0->1", 0)
            run.summary["n_persist_broken"]   = counts.get("0->0", 0)
            run.summary["n_break"]            = counts.get("1->0", 0)
            run.summary["n_persist_correct"]  = counts.get("1->1", 0)
            # transition_kernel.json is the *derived* stat; iter_records.jsonl
            # is the *raw evidence* the notebook recomputes the kernel from —
            # it must always be attached. A missing raw file is logged loudly,
            # never silently skipped.
            maybe_log_artifact(run, tk, "iter", f"{bench}_{gen}_single_method_kernel")
            if ir.exists():
                maybe_log_artifact(run, ir, "iter", f"{bench}_{gen}_single_method_records")
            else:
                log.warning(f"  {bench}/{gen}/single_method: iter_records.jsonl "
                            f"MISSING at {ir} — raw evidence NOT uploaded")
            run.finish()

    # Per-method realbaselines (newer layout: selfrefine/ + reflexion/ subdirs)
    for bench, subdir in ORCH_REALBASELINES_BENCHMARKS:
        if args.benchmark and args.benchmark != bench:
            continue
        for gen in GENERATORS:
            if args.generator and args.generator != gen:
                continue
            for method in ["selfrefine", "reflexion"]:
                cell = DATA_ROOT / subdir / gen / method
                tk = cell / "transition_kernel.json"
                ir = cell / "iter_records.jsonl"
                if not tk.exists():
                    continue
                tkd = load_json(tk) or {}
                ka = tkd.get("kernel_all", {})
                n_pairs = ka.get("n_pairs") or sum((ka.get("raw_counts") or {}).values())

                name = f"iter__orchestration__{bench}__{gen}__{method}"
                config = {
                    "track": "orchestration",
                    "experiment_type": "iter",
                    "benchmark": bench, "generator": gen, "method": method,
                    "git_sha": git_sha(),
                    "data_source": rel(cell),
                }
                tags = ["track:orchestration", "experiment:iter",
                        f"benchmark:{bench}", f"generator:{gen}",
                        f"method:{method}"]
                run = safe_init(name, config, tags, existing, args.force, args.dry_run)
                if run is None:
                    continue
                run.summary["P_fix_given_broken"]    = ka.get("P_fix_given_broken")
                run.summary["P_break_given_correct"] = ka.get("P_break_given_correct")
                run.summary["n_pairs"] = n_pairs
                counts = ka.get("raw_counts", {})
                run.summary["n_fix"]              = counts.get("0->1", 0)
                run.summary["n_persist_broken"]   = counts.get("0->0", 0)
                run.summary["n_break"]            = counts.get("1->0", 0)
                run.summary["n_persist_correct"]  = counts.get("1->1", 0)
                maybe_log_artifact(run, tk, "iter", f"{bench}_{gen}_{method}_kernel")
                if ir.exists():
                    maybe_log_artifact(run, ir, "iter", f"{bench}_{gen}_{method}_records")
                else:
                    log.warning(f"  {bench}/{gen}/{method}: iter_records.jsonl "
                                f"MISSING at {ir} — raw evidence NOT uploaded")
                run.finish()


# ---------------------------------------------------------------------------
# POLICY_COMPARISON — one run per (benchmark, generator, kernel_source)

def upload_policy_comparison_orch(args, existing):
    log.info("=== Policy comparison runs (track:orchestration) ===")

    # Variants of policy_comparison*.json that ship per cell
    KERNEL_VARIANTS = {
        "policy_comparison.json":                        "default",
        "policy_comparison_kernel_iid_baseline.json":   "iid_baseline",
        "policy_comparison_kernel_measured.json":       "measured",
        "policy_comparison_kernel_iterative.json":      "iterative",
        "policy_comparison_loo.json":                    "loo",
    }

    for bench, subdir in ORCH_CALIBRATION_BENCHMARKS:
        if args.benchmark and args.benchmark != bench:
            continue
        for gen in GENERATORS:
            if args.generator and args.generator != gen:
                continue
            cell = DATA_ROOT / subdir / gen
            for fname, kernel_src in KERNEL_VARIANTS.items():
                pcf = cell / fname
                if not pcf.exists():
                    continue
                pc = load_json(pcf) or {}
                policies = pc.get("policies") or pc  # support both schemas

                name = f"policy_comparison__orchestration__{bench}__{gen}__{kernel_src}"
                config = {
                    "track": "orchestration",
                    "experiment_type": "policy_comparison",
                    "benchmark": bench, "generator": gen,
                    "kernel_source": kernel_src,
                    "git_sha": git_sha(),
                    "data_source": rel(pcf),
                    **DEFAULT_COSTS,
                }
                tags = ["track:orchestration", "experiment:policy_comparison",
                        f"benchmark:{bench}", f"generator:{gen}",
                        f"kernel_source:{kernel_src}"]
                run = safe_init(name, config, tags, existing, args.force, args.dry_run)
                if run is None:
                    continue

                for policy in POLICY_NAMES:
                    info = policies.get(policy)
                    if not isinstance(info, dict):
                        continue
                    run.summary[f"{policy}/mean_utility"] = info.get("mean_utility")
                    run.summary[f"{policy}/pass_rate"]    = info.get("pass_rate")
                    run.summary[f"{policy}/delta_vs_av"]  = info.get("diff_vs_always_verify")
                    run.summary[f"{policy}/ci95_lo"]      = info.get("ci95_lo")
                    run.summary[f"{policy}/ci95_hi"]      = info.get("ci95_hi")

                # Best policy + regime classification (derived)
                deltas = {p: (policies.get(p) or {}).get("diff_vs_always_verify")
                          for p in POLICY_NAMES if (policies.get(p) or {}).get("diff_vs_always_verify") is not None}
                if deltas:
                    best = max(deltas, key=deltas.get)
                    run.summary["best_policy"] = best
                    run.summary["best_policy_delta"] = deltas[best]

                maybe_log_artifact(run, pcf, "policy_comparison", f"{bench}_{gen}_{kernel_src}_pc")
                run.finish()


# ---------------------------------------------------------------------------
# COST-VECTOR SWEEPS — one run per (benchmark, generator, c_ver_value)

def upload_cver_sweep_orch(args, existing):
    log.info("=== c_ver sweep runs (track:orchestration) ===")
    for bench, subdir in ORCH_CALIBRATION_BENCHMARKS:
        if args.benchmark and args.benchmark != bench:
            continue
        for gen in GENERATORS:
            if args.generator and args.generator != gen:
                continue
            cell = DATA_ROOT / subdir / gen
            # files named policy_comparison_cver{NN}.json
            for pcf in cell.glob("policy_comparison_cver*.json"):
                try:
                    cver = int(pcf.stem.split("cver")[-1])
                except ValueError:
                    continue
                pc = load_json(pcf) or {}
                policies = pc.get("policies") or pc
                name = f"cver_sweep__orchestration__{bench}__{gen}__cver{cver}"
                cost = {**DEFAULT_COSTS, "cost_ver": cver}
                config = {
                    "track": "orchestration", "experiment_type": "cver_sweep",
                    "benchmark": bench, "generator": gen,
                    "sweep_axis": "c_ver", "sweep_value": cver,
                    "git_sha": git_sha(),
                    "data_source": rel(pcf),
                    **cost,
                }
                tags = ["track:orchestration", "experiment:cver_sweep",
                        f"benchmark:{bench}", f"generator:{gen}",
                        f"cver:{cver}"]
                run = safe_init(name, config, tags, existing, args.force, args.dry_run)
                if run is None:
                    continue
                for policy in POLICY_NAMES:
                    info = policies.get(policy)
                    if not isinstance(info, dict):
                        continue
                    run.summary[f"{policy}/mean_utility"] = info.get("mean_utility")
                    run.summary[f"{policy}/delta_vs_av"]  = info.get("diff_vs_always_verify")
                    run.summary[f"{policy}/ci95_lo"]      = info.get("ci95_lo")
                    run.summary[f"{policy}/ci95_hi"]      = info.get("ci95_hi")
                run.finish()


# ---------------------------------------------------------------------------
# THETA SENSITIVITY — one run per (benchmark, generator, perturbation_kind)

def upload_theta_sweep_orch(args, existing):
    log.info("=== Theta-sensitivity runs (track:orchestration) ===")
    for bench, subdir in ORCH_CALIBRATION_BENCHMARKS:
        if args.benchmark and args.benchmark != bench:
            continue
        for gen in GENERATORS:
            if args.generator and args.generator != gen:
                continue
            cell = DATA_ROOT / subdir / gen
            sf = cell / "sensitivity.json"
            if not sf.exists():
                continue
            sd = load_json(sf) or {}
            # sensitivity.json contains multiple perturbation variants
            for variant, payload in (sd.get("variants") or {}).items():
                if not isinstance(payload, dict):
                    continue
                name = f"theta_sweep__orchestration__{bench}__{gen}__{variant}"
                config = {
                    "track": "orchestration", "experiment_type": "theta_sweep",
                    "benchmark": bench, "generator": gen,
                    "sweep_axis": "theta", "sweep_value": variant,
                    "git_sha": git_sha(),
                    "data_source": rel(sf),
                    **DEFAULT_COSTS,
                }
                tags = ["track:orchestration", "experiment:theta_sweep",
                        f"benchmark:{bench}", f"generator:{gen}",
                        f"perturbation:{variant}"]
                run = safe_init(name, config, tags, existing, args.force, args.dry_run)
                if run is None:
                    continue
                for policy in POLICY_NAMES:
                    info = (payload.get("policies") or {}).get(policy)
                    if not isinstance(info, dict):
                        continue
                    run.summary[f"{policy}/delta_vs_av"] = info.get("diff_vs_always_verify")
                    run.summary[f"{policy}/ci95_lo"]     = info.get("ci95_lo")
                    run.summary[f"{policy}/ci95_hi"]     = info.get("ci95_hi")
                run.finish()


# ---------------------------------------------------------------------------
# METHODOLOGY summaries — one run per analysis type

def upload_methodology(args, existing):
    log.info("=== Methodology runs (track:orchestration) ===")
    methodology_files = {
        "fdr":                    (DATA_ROOT / "fdr_correction" / "per_cell.csv",
                                   DATA_ROOT / "fdr_correction" / "summary.json"),
        "mde_power":              (DATA_ROOT / "mde_power" / "per_cell.csv", None),
        "cluster_bootstrap":      (DATA_ROOT / "cluster_bootstrap" / "per_cell.csv", None),
        "critic_independence":    (DATA_ROOT / "critic_independence" / "per_cell.csv",
                                   DATA_ROOT / "critic_independence" / "pairwise.csv"),
        "prior_ci":               (DATA_ROOT / "prior_ci" / "per_cell.csv", None),
    }
    for analysis, (primary, extra) in methodology_files.items():
        if not primary.exists():
            continue
        name = f"methodology__orchestration__{analysis}"
        config = {
            "track": "orchestration", "experiment_type": "methodology",
            "analysis": analysis,
            "git_sha": git_sha(),
            "data_source": rel(primary),
        }
        tags = ["track:orchestration", "experiment:methodology",
                f"analysis:{analysis}"]
        run = safe_init(name, config, tags, existing, args.force, args.dry_run)
        if run is None:
            continue
        run.summary["n_rows"] = line_count(primary) - 1
        if extra and extra.exists():
            run.summary["extra_file"] = str(extra.name)
        maybe_log_artifact(run, primary, "methodology", f"{analysis}_per_cell")
        if extra:
            maybe_log_artifact(run, extra, "methodology", f"{analysis}_{extra.stem}")
        # FDR summary.json gets parsed inline
        if analysis == "fdr" and extra and extra.exists():
            s = load_json(extra) or {}
            run.summary["alpha"] = s.get("alpha")
            run.summary["n_total_tests"] = s.get("n_total_tests")
            for policy, pd in (s.get("per_policy") or {}).items():
                run.summary[f"fdr/{policy}/n_cells"] = pd.get("n_cells")
                run.summary[f"fdr/{policy}/n_sig_uncorrected"] = pd.get("n_significant_uncorrected_p05")
                run.summary[f"fdr/{policy}/n_sig_bh"] = pd.get("n_significant_after_BH_FDR_per_policy")
        run.finish()


# ---------------------------------------------------------------------------
# PAPER_TABLE — single canonical aggregate as its own run

def upload_paper_table(args, existing):
    log.info("=== PAPER_TABLE aggregate (track:orchestration) ===")
    pt_json = DATA_ROOT / "PAPER_TABLE.json"
    pt_csv  = DATA_ROOT / "PAPER_TABLE.csv"
    if not pt_json.exists():
        log.warning("PAPER_TABLE.json missing — skipping aggregate")
        return
    name = "aggregate__orchestration__paper_table"
    config = {
        "track": "orchestration", "experiment_type": "aggregate",
        "analysis": "paper_table",
        "git_sha": git_sha(),
        "data_source": rel(pt_json),
    }
    tags = ["track:orchestration", "experiment:aggregate", "analysis:paper_table"]
    run = safe_init(name, config, tags, existing, args.force, args.dry_run)
    if run is None:
        return
    pt = load_json(pt_json) or {}
    run.summary["n_cells"] = sum(len(v) for v in pt.values() if isinstance(v, dict))
    run.summary["n_benchmarks"] = len(pt)
    maybe_log_artifact(run, pt_json, "aggregate", "paper_table_json")
    if pt_csv.exists():
        maybe_log_artifact(run, pt_csv, "aggregate", "paper_table_csv")
    run.finish()


# ---------------------------------------------------------------------------
# ABBO track — colleague's codebase

def upload_abbo(args, existing):
    log.info("=== ABBO runs (track:abbo) ===")
    files = {
        "humanevalfix":  ABBO_RESULTS / "humaneval_full_endtoend.json",
        "codecontests":  ABBO_RESULTS / "codecontests_simulation_metrics.json",
        "swe_lite":      ABBO_RESULTS / "swebench_simulation_metrics.json",
        "humaneval_smoke": ABBO_RESULTS / "humaneval_smoke_results.json",
    }
    for bench, path in files.items():
        if args.benchmark and args.benchmark != bench:
            continue
        if not path.exists():
            log.debug(f"  miss: {path}")
            continue
        data = load_json(path) or {}
        # The abbo schema has top-level "summaries": [{policy, n_episodes, mean_utility, ...}, ...]
        summaries = data.get("summaries") or []
        if not summaries:
            log.warning(f"  no 'summaries' field in {path}")
            continue
        name = f"policy_comparison__abbo__{bench}__gpt_oss_20b"
        config = {
            "track": "abbo",
            "experiment_type": "policy_comparison",
            "benchmark": bench, "generator": "gpt_oss_20b",
            "n_instances": data.get("n_instances") or max((s.get("n_episodes", 0) for s in summaries), default=0),
            "git_sha": git_sha(),
            "data_source": rel(path),
            # abbo default cost vector (per colleague's deck)
            "cost_R": 100, "cost_ver": 5, "cost_gen": 10,
            "cost_L0": 1, "cost_L2": 1, "cost_L3": 1,
        }
        tags = ["track:abbo", "experiment:policy_comparison",
                f"benchmark:{bench}", "generator:gpt_oss_20b"]
        run = safe_init(name, config, tags, existing, args.force, args.dry_run)
        if run is None:
            continue
        for s in summaries:
            policy = s.get("policy", "unknown").replace(" ", "_").lower()
            for k in ("mean_utility", "pass_rate", "delta_vs_av",
                      "fix_rate", "mean_cost", "n_episodes"):
                if k in s:
                    run.summary[f"{policy}/{k}"] = s[k]
        maybe_log_artifact(run, path, "policy_comparison", f"abbo_{bench}_results")
        run.finish()


# ---------------------------------------------------------------------------
# CLI

def main():
    global DATA_ROOT, ABBO_RESULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=["orchestration", "abbo", "all"], default="all")
    ap.add_argument("--experiment", choices=[
        "calibration", "iter", "policy_comparison", "cver_sweep",
        "theta_sweep", "methodology", "paper_table", "all",
    ], default="all")
    ap.add_argument("--benchmark", default=None, help="Filter by benchmark (debug)")
    ap.add_argument("--generator", default=None, help="Filter by generator (debug)")
    ap.add_argument("--force", action="store_true",
                    help="Re-upload runs that already exist (creates a new version)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be uploaded; do not actually call wandb")
    ap.add_argument("--data-root", default=None,
                    help=f"Override orchestration results dir (default: {DATA_ROOT})")
    ap.add_argument("--abbo-root", default=None,
                    help=f"Override abbo sim_results dir (default: {ABBO_RESULTS})")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        format="%(asctime)s  %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
        datefmt="%H:%M:%S",
    )

    if args.data_root:
        DATA_ROOT = Path(args.data_root).expanduser().resolve()
    if args.abbo_root:
        ABBO_RESULTS = Path(args.abbo_root).expanduser().resolve()
    if not DATA_ROOT.exists():
        log.error(f"DATA_ROOT does not exist: {DATA_ROOT}. Pass --data-root.")
        sys.exit(1)
    log.info(f"DATA_ROOT = {DATA_ROOT}")
    log.info(f"ABBO_RESULTS = {ABBO_RESULTS}")

    # Sanity: is wandb authed?
    if not args.dry_run:
        try:
            api = wandb.Api()
            api.viewer  # probes auth
        except Exception as e:
            log.error(f"wandb not authed ({e}). Run `wandb login` first.")
            sys.exit(1)
        existing = existing_run_names(api)
        log.info(f"Project has {len(existing)} existing runs.")
    else:
        existing = set()

    want_orch = args.track in ("orchestration", "all")
    want_abbo = args.track in ("abbo", "all")

    if want_orch:
        if args.experiment in ("calibration", "all"):
            upload_calibration_orch(args, existing)
        if args.experiment in ("iter", "all"):
            upload_iter_orch(args, existing)
        if args.experiment in ("policy_comparison", "all"):
            upload_policy_comparison_orch(args, existing)
        if args.experiment in ("cver_sweep", "all"):
            upload_cver_sweep_orch(args, existing)
        if args.experiment in ("theta_sweep", "all"):
            upload_theta_sweep_orch(args, existing)
        if args.experiment in ("methodology", "all"):
            upload_methodology(args, existing)
        if args.experiment in ("paper_table", "all"):
            upload_paper_table(args, existing)

    if want_abbo:
        if args.experiment in ("policy_comparison", "all"):
            upload_abbo(args, existing)

    log.info("Done.")


if __name__ == "__main__":
    main()
