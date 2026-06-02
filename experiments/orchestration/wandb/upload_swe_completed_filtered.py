#!/usr/bin/env python3
"""Upload completed-only SWE calibration and iter runs to W&B.

This is intentionally separate from upload_runs.py: it creates clean runs with
the same high-level schema, plus score_scope=completed_filtered, without
overwriting the original noisy SWE runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import wandb

ENTITY = "nlpresearch.group"
PROJECT = "orchestration-hypothesis-testing"
DEFAULT_DATA_ROOT = Path("/capstor/store/cscs/swissai/a0142/agents_uq/sr_rfx_local")
DEFAULT_ARTIFACT_ROOT = Path(
    "/capstor/store/cscs/swissai/a0142/agents_uq/wandb_completed_filtered"
)

REPO_ROOT = Path(__file__).resolve().parents[3]
HYP_ROOT = REPO_ROOT / "experiments" / "orchestration_hypothesis_testing"
sys.path.insert(0, str(HYP_ROOT))

from _common.kernel import compute_transition_kernel_from_pairs  # noqa: E402

log = logging.getLogger("upload_swe_completed_filtered")

CELLS = {
    ("swe_lite", "gpt_oss_20b_local"): {
        "run_root": "swe_lite_slurm_2365574",
        "calib_subdir": "swebench_lite",
        "iter_subdir": "swebench_lite_realbaselines",
    },
    ("swe_lite", "qwen25_32b"): {
        "run_root": "swe_lite_slurm_2365576",
        "calib_subdir": "swebench_lite",
        "iter_subdir": "swebench_lite_realbaselines",
    },
    ("swe_verified", "gpt_oss_20b_local"): {
        "run_root": "swe_verified_slurm_2365575",
        "calib_subdir": "swebench_verified",
        "iter_subdir": "swebench_verified_realbaselines",
    },
    ("swe_verified", "qwen25_32b"): {
        "run_root": "swe_verified_slurm_2365577",
        "calib_subdir": "swebench_verified",
        "iter_subdir": "swebench_verified_realbaselines",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def beta_smooth(k: int, n: int) -> float:
    return (k + 1) / (n + 2)


def critic_likelihoods(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_vals = [int(row["Y"]) for row in rows if row.get("Y") in (0, 1, True, False)]
    n = len(y_vals)
    n_y1 = sum(y_vals)
    out: dict[str, Any] = {
        "n_records": n,
        "n_resolved": n_y1,
        "prior_Y1": beta_smooth(n_y1, n) if n else None,
        "critic_likelihoods": {},
        "smoothing": "Beta(1,1)",
        "score_scope": "completed_filtered",
    }
    critics = ["L0_syntax", "L1_lint", "L2_public_tests", "L2_fast_test", "L3_llm_review"]
    for critic in critics:
        usable = [row for row in rows if row.get(critic) is not None]
        if not usable:
            out["critic_likelihoods"][critic] = {
                "P_pass_given_Y1": None,
                "P_pass_given_Y0": None,
                "gap": None,
            }
            continue
        y1_rows = [row for row in usable if int(row["Y"]) == 1]
        y0_rows = [row for row in usable if int(row["Y"]) == 0]
        pass_y1 = sum(1 for row in y1_rows if bool(row.get(critic)))
        pass_y0 = sum(1 for row in y0_rows if bool(row.get(critic)))
        p1 = beta_smooth(pass_y1, len(y1_rows)) if y1_rows else None
        p0 = beta_smooth(pass_y0, len(y0_rows)) if y0_rows else None
        out["critic_likelihoods"][critic] = {
            "P_pass_given_Y1": p1,
            "P_pass_given_Y0": p0,
            "gap": (p1 - p0) if p1 is not None and p0 is not None else None,
            "n_y1": len(y1_rows),
            "n_y0": len(y0_rows),
        }
    return out


def eval_file(eval_dir: Path, gen: str, patch_id: int) -> Path:
    return eval_dir / f"{gen}__p{patch_id}.{gen}_p{patch_id}.json"


def load_eval_sets(eval_dir: Path, gen: str, patch_id: int) -> dict[str, Any]:
    path = eval_file(eval_dir, gen, patch_id)
    data = read_json(path)
    resolved = set(data.get("resolved_ids", []))
    unresolved = set(data.get("unresolved_ids", []))
    completed = set(data.get("completed_ids", [])) or (resolved | unresolved)
    return {
        "path": path,
        "data": data,
        "resolved": resolved,
        "unresolved": unresolved,
        "completed": completed,
    }


def completed_score(data: dict[str, Any]) -> float | None:
    denom = int(data.get("completed_instances") or 0)
    if denom <= 0:
        return None
    return float(data.get("resolved_instances") or 0) / denom


def calibration_dirs(data_root: Path, bench: str, gen: str) -> tuple[Path, Path]:
    cell = CELLS[(bench, gen)]
    root = data_root / cell["run_root"] / cell["calib_subdir"]
    return root / gen, root / "eval"


def iter_dir(data_root: Path, bench: str, gen: str, method: str) -> Path:
    cell = CELLS[(bench, gen)]
    return data_root / cell["run_root"] / cell["iter_subdir"] / gen / method


def raw_counts(kernel: dict[str, Any]) -> dict[str, int]:
    return dict(kernel.get("raw_counts") or {})


def transition_kernel_json(generator: str, pairs: list[tuple[int, int]]) -> dict[str, Any]:
    k = compute_transition_kernel_from_pairs(pairs)
    return {
        "generator": generator,
        "score_scope": "completed_filtered",
        "kernel_all": {
            "P_fix_given_broken": k["P_fix_given_broken"],
            "P_stay_broken": k["P_stay_broken"],
            "P_break_given_correct": k["P_break_given_correct"],
            "P_stay_correct": k["P_stay_correct"],
            "raw_counts": k["raw_counts"],
            "n_pairs": k["n_pairs"],
            "smoothing": k["smoothing"],
        },
    }


def pairs_from_rows(rows: list[dict[str, Any]], step_key: str) -> list[tuple[int, int]]:
    by_inst: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("Y") in (0, 1, True, False):
            by_inst[str(row["instance_id"])].append(row)
    pairs: list[tuple[int, int]] = []
    for traj in by_inst.values():
        traj.sort(key=lambda row: int(row.get(step_key, -1)))
        for a, b in zip(traj, traj[1:]):
            # Avoid fabricating transitions across missing steps.
            if int(b[step_key]) != int(a[step_key]) + 1:
                continue
            pairs.append((int(a["Y"]), int(b["Y"])))
    return pairs


def build_clean_calibration(
    data_root: Path, artifact_root: Path, bench: str, gen: str
) -> dict[str, Any]:
    calib_dir, eval_dir = calibration_dirs(data_root, bench, gen)
    source_rows = read_jsonl(calib_dir / "critic_results.jsonl")
    eval_by_patch = {pid: load_eval_sets(eval_dir, gen, pid) for pid in [0, 1, 2]}
    clean_rows: list[dict[str, Any]] = []

    for row in source_rows:
        patch_id = int(row.get("patch_id", -1))
        if patch_id not in eval_by_patch:
            continue
        inst = str(row.get("instance_id"))
        sets = eval_by_patch[patch_id]
        if inst not in sets["completed"]:
            continue
        new_row = dict(row)
        new_row["Y"] = 1 if inst in sets["resolved"] else 0
        new_row["score_scope"] = "completed_filtered"
        new_row["harness_completed"] = True
        clean_rows.append(new_row)

    pairs = pairs_from_rows(clean_rows, "patch_id")
    kernel = transition_kernel_json(gen, pairs)
    likelihoods = critic_likelihoods(clean_rows)
    likelihoods["generator"] = gen
    likelihoods["benchmark"] = bench

    per_patch = {}
    for pid, sets in eval_by_patch.items():
        data = sets["data"]
        per_patch[f"p{pid}"] = {
            "total_instances": data.get("total_instances"),
            "submitted_instances": data.get("submitted_instances"),
            "completed_instances": data.get("completed_instances"),
            "resolved_instances": data.get("resolved_instances"),
            "unresolved_instances": data.get("unresolved_instances"),
            "empty_patch_instances": data.get("empty_patch_instances"),
            "error_instances": data.get("error_instances"),
            "score_completed_filtered": completed_score(data),
        }

    summary = {
        "score_scope": "completed_filtered",
        "benchmark": bench,
        "generator": gen,
        "n_records": len(clean_rows),
        "n_resolved": sum(int(row["Y"]) for row in clean_rows),
        "n_unique_instances": len({row["instance_id"] for row in clean_rows}),
        "per_patch": per_patch,
        "likelihoods": likelihoods,
        "transition_kernel": kernel,
        "source_calibration_dir": str(calib_dir),
        "source_eval_dir": str(eval_dir),
    }

    out_dir = artifact_root / "calibration" / bench / gen
    paths = {
        "critic_results": out_dir / "critic_results.completed_filtered.jsonl",
        "likelihoods": out_dir / "likelihood_tables.completed_filtered.json",
        "kernel": out_dir / "transition_kernel_iid_baseline.completed_filtered.json",
        "summary": out_dir / "summary.completed_filtered.json",
    }
    write_jsonl(paths["critic_results"], clean_rows)
    write_json(paths["likelihoods"], likelihoods)
    write_json(paths["kernel"], kernel)
    write_json(paths["summary"], summary)
    return {"summary": summary, "paths": paths}


def latest_rows_by_instance_step(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], tuple[int, dict[str, Any]]] = {}
    for idx, row in enumerate(rows):
        key = (str(row.get("instance_id")), int(row.get("step", -1)))
        latest[key] = (idx, row)
    return [row for _, row in sorted(latest.values(), key=lambda item: item[0])]


def build_clean_iter(
    data_root: Path, artifact_root: Path, bench: str, gen: str, method: str
) -> dict[str, Any]:
    source_dir = iter_dir(data_root, bench, gen, method)
    source_rows = latest_rows_by_instance_step(read_jsonl(source_dir / "iter_records.jsonl"))
    _, eval_dir = calibration_dirs(data_root, bench, gen)
    p0 = load_eval_sets(eval_dir, gen, 0)

    clean_rows: list[dict[str, Any]] = []
    step_stats: dict[int, dict[str, Any]] = defaultdict(lambda: {"completed": 0, "resolved": 0})
    for row in source_rows:
        step = int(row.get("step", -1))
        inst = str(row.get("instance_id"))
        new_row = dict(row)
        if step == 0:
            if inst not in p0["completed"]:
                continue
            new_row["Y"] = 1 if inst in p0["resolved"] else 0
        elif row.get("Y") in (0, 1, True, False):
            new_row["Y"] = int(row["Y"])
        else:
            continue
        new_row["score_scope"] = "completed_filtered"
        new_row["harness_completed"] = True
        clean_rows.append(new_row)
        step_stats[step]["completed"] += 1
        step_stats[step]["resolved"] += int(new_row["Y"])

    pairs = pairs_from_rows(clean_rows, "step")
    kernel = transition_kernel_json(method, pairs)
    summary = {
        "score_scope": "completed_filtered",
        "benchmark": bench,
        "generator": gen,
        "method": method,
        "n_source_rows": len(source_rows),
        "n_clean_records": len(clean_rows),
        "n_clean_instances": len({row["instance_id"] for row in clean_rows}),
        "n_pairs": kernel["kernel_all"]["n_pairs"],
        "step_stats": {},
        "transition_kernel": kernel,
        "source_iter_dir": str(source_dir),
        "source_p0_eval": str(p0["path"]),
    }
    for step, stats in sorted(step_stats.items()):
        completed = stats["completed"]
        resolved = stats["resolved"]
        summary["step_stats"][str(step)] = {
            "completed": completed,
            "resolved": resolved,
            "score_completed_filtered": resolved / completed if completed else None,
        }

    out_dir = artifact_root / "iter" / bench / gen / method
    paths = {
        "iter_records": out_dir / "iter_records.completed_filtered.jsonl",
        "kernel": out_dir / "transition_kernel.completed_filtered.json",
        "summary": out_dir / "summary.completed_filtered.json",
    }
    write_jsonl(paths["iter_records"], clean_rows)
    write_json(paths["kernel"], kernel)
    write_json(paths["summary"], summary)
    return {"summary": summary, "paths": paths}


def existing_run_names() -> set[str]:
    try:
        api = wandb.Api()
        return {run.name for run in api.runs(f"{ENTITY}/{PROJECT}")}
    except Exception as exc:
        log.warning("Could not list existing W&B runs: %s", exc)
        return set()


def add_artifact(run: Any, path: Path, artifact_type: str, name: str) -> None:
    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_file(str(path))
    run.log_artifact(artifact)


def upload_calibration(
    bench: str,
    gen: str,
    built: dict[str, Any],
    existing: set[str],
    force: bool,
    dry_run: bool,
) -> None:
    summary = built["summary"]
    paths = built["paths"]
    name = f"calibration__orchestration__{bench}__{gen}__completed_filtered"
    if name in existing and not force:
        log.info("skip existing: %s", name)
        return
    log.info("%s%s", "[DRY-RUN] " if dry_run else "uploading: ", name)
    if dry_run:
        return
    config = {
        "track": "orchestration",
        "experiment_type": "calibration",
        "benchmark": bench,
        "generator": gen,
        "score_scope": "completed_filtered",
        "n_instances": summary["n_unique_instances"],
        "k_patches": 3,
        "git_sha": git_sha(),
        "data_source": str(paths["summary"].parent),
    }
    tags = [
        "track:orchestration",
        "experiment:calibration",
        f"benchmark:{bench}",
        f"generator:{gen}",
        "score_scope:completed_filtered",
    ]
    run = wandb.init(entity=ENTITY, project=PROJECT, name=name, config=config, tags=tags, reinit=True)
    ll = summary["likelihoods"]
    cl = ll.get("critic_likelihoods", {})
    run.summary["score_scope"] = "completed_filtered"
    run.summary["prior_Y1"] = ll.get("prior_Y1")
    run.summary["L0_gap"] = (cl.get("L0_syntax") or {}).get("gap")
    run.summary["L1_gap"] = (cl.get("L1_lint") or {}).get("gap")
    run.summary["L2_gap"] = (cl.get("L2_public_tests") or cl.get("L2_fast_test") or {}).get("gap")
    run.summary["L3_gap"] = (cl.get("L3_llm_review") or {}).get("gap")
    run.summary["n_records"] = summary["n_records"]
    run.summary["n_resolved"] = summary["n_resolved"]
    run.summary["n_unique_instances"] = summary["n_unique_instances"]
    ka = summary["transition_kernel"]["kernel_all"]
    run.summary["iid_baseline_P_fix"] = ka.get("P_fix_given_broken")
    run.summary["iid_baseline_P_break"] = ka.get("P_break_given_correct")
    for patch_id, stats in summary["per_patch"].items():
        run.summary[f"{patch_id}/completed_instances"] = stats["completed_instances"]
        run.summary[f"{patch_id}/resolved_instances"] = stats["resolved_instances"]
        run.summary[f"{patch_id}/score_completed_filtered"] = stats["score_completed_filtered"]
        run.summary[f"{patch_id}/error_instances_excluded"] = stats["error_instances"]
        run.summary[f"{patch_id}/empty_patch_instances_excluded"] = stats["empty_patch_instances"]
    add_artifact(run, paths["critic_results"], "calibration", f"{bench}_{gen}_critic_results_completed_filtered")
    add_artifact(run, paths["likelihoods"], "calibration", f"{bench}_{gen}_likelihood_tables_completed_filtered")
    add_artifact(run, paths["kernel"], "calibration", f"{bench}_{gen}_iid_kernel_completed_filtered")
    add_artifact(run, paths["summary"], "calibration", f"{bench}_{gen}_summary_completed_filtered")
    run.finish()


def upload_iter(
    bench: str,
    gen: str,
    method: str,
    built: dict[str, Any],
    existing: set[str],
    force: bool,
    dry_run: bool,
) -> None:
    summary = built["summary"]
    paths = built["paths"]
    name = f"iter__orchestration__{bench}__{gen}__{method}__completed_filtered"
    if name in existing and not force:
        log.info("skip existing: %s", name)
        return
    log.info("%s%s", "[DRY-RUN] " if dry_run else "uploading: ", name)
    if dry_run:
        return
    config = {
        "track": "orchestration",
        "experiment_type": "iter",
        "benchmark": bench,
        "generator": gen,
        "method": method,
        "score_scope": "completed_filtered",
        "git_sha": git_sha(),
        "data_source": str(paths["summary"].parent),
    }
    tags = [
        "track:orchestration",
        "experiment:iter",
        f"benchmark:{bench}",
        f"generator:{gen}",
        f"method:{method}",
        "score_scope:completed_filtered",
    ]
    run = wandb.init(entity=ENTITY, project=PROJECT, name=name, config=config, tags=tags, reinit=True)
    ka = summary["transition_kernel"]["kernel_all"]
    counts = raw_counts(ka)
    run.summary["score_scope"] = "completed_filtered"
    run.summary["P_fix_given_broken"] = ka.get("P_fix_given_broken")
    run.summary["P_break_given_correct"] = ka.get("P_break_given_correct")
    run.summary["n_pairs"] = ka.get("n_pairs")
    run.summary["n_fix"] = counts.get("0->1", 0)
    run.summary["n_persist_broken"] = counts.get("0->0", 0)
    run.summary["n_break"] = counts.get("1->0", 0)
    run.summary["n_persist_correct"] = counts.get("1->1", 0)
    run.summary["n_clean_records"] = summary["n_clean_records"]
    run.summary["n_clean_instances"] = summary["n_clean_instances"]
    for step, stats in summary["step_stats"].items():
        run.summary[f"step{step}/completed"] = stats["completed"]
        run.summary[f"step{step}/resolved"] = stats["resolved"]
        run.summary[f"step{step}/score_completed_filtered"] = stats["score_completed_filtered"]
    add_artifact(run, paths["kernel"], "iter", f"{bench}_{gen}_{method}_kernel_completed_filtered")
    add_artifact(run, paths["iter_records"], "iter", f"{bench}_{gen}_{method}_records_completed_filtered")
    add_artifact(run, paths["summary"], "iter", f"{bench}_{gen}_{method}_summary_completed_filtered")
    run.finish()


def parse_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--benchmarks", default="swe_lite,swe_verified")
    parser.add_argument("--generators", default="gpt_oss_20b_local,qwen25_32b")
    parser.add_argument("--methods", default="selfrefine,reflexion")
    parser.add_argument("--only", choices=["all", "calibration", "iter"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="write clean artifacts but do not upload")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if not args.dry_run and not args.prepare_only and not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is required unless --dry-run or --prepare-only is used")

    benchmarks = parse_csv(args.benchmarks, ["swe_lite", "swe_verified"])
    generators = parse_csv(args.generators, ["gpt_oss_20b_local", "qwen25_32b"])
    methods = parse_csv(args.methods, ["selfrefine", "reflexion"])

    existing = set() if args.dry_run or args.prepare_only else existing_run_names()
    for bench in benchmarks:
        for gen in generators:
            if (bench, gen) not in CELLS:
                log.warning("unknown cell, skipping: %s/%s", bench, gen)
                continue
            if args.only in ("all", "calibration"):
                built = build_clean_calibration(args.data_root, args.artifact_root, bench, gen)
                upload_calibration(bench, gen, built, existing, args.force, args.dry_run or args.prepare_only)
            if args.only in ("all", "iter"):
                for method in methods:
                    source_dir = iter_dir(args.data_root, bench, gen, method)
                    if not (source_dir / "iter_records.jsonl").exists():
                        log.warning("missing iter_records, skipping: %s", source_dir)
                        continue
                    built = build_clean_iter(args.data_root, args.artifact_root, bench, gen, method)
                    upload_iter(bench, gen, method, built, existing, args.force, args.dry_run or args.prepare_only)

    log.info("done; clean artifacts under %s", args.artifact_root)


if __name__ == "__main__":
    main()
