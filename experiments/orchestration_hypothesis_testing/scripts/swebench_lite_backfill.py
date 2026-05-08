"""Backfill SWE-bench Lite critics into LCB schema using existing spot_check_n50 data.

Reads the May spot_check_n50 outputs (raw_responses, harness eval reports,
existing partial critic_results) and writes a clean
<gen>/critic_results.jsonl + likelihood_tables.json under data/swebench_lite/
in LCB format, so lcb_compare.py can run today's L2-aware controller.

For each (generator, instance, patch_id) it produces:
  Y               = report['resolved']               (full SWE-bench eval)
  L0_syntax       = ast.parse(extracted code)        (free)
  L1_lint         = ruff check on extracted code     (free)
  L2_public_tests = len(FAIL_TO_PASS.failure) == 0   (free, from harness report)
  L3_llm_review   = Haiku PASS/FAIL                  (cached if exists, else API)

Generators run in parallel via ThreadPoolExecutor.

Usage:
  python3 swebench_lite_backfill.py \\
    --src-root /mnt/.../.claude/worktrees/.../experiments/orchestration_hypothesis_testing/data/spot_check_n50 \\
    --dst-root data/swebench_lite \\
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \\
    --max-workers-l3 4
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lcb_calibrate import critic_L3_review  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("swe_lite_bf")


# ---------- Critic helpers ----------

def _critic_L0(code: str) -> bool:
    if not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _critic_L1_diff(diff: str) -> bool:
    """Lint pass = ruff finds no issues on the diff's added code.

    SWE-bench patches are unified diffs — we extract +lines and ruff them.
    """
    if not diff.strip():
        return False
    added_lines = []
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added_lines.append(line[1:])
    if not added_lines:
        return False
    code = "\n".join(added_lines)
    try:
        r = subprocess.run(["ruff", "check", "--select=E,F", "-", "--no-cache",
                             "--exit-zero"],
                            input=code, text=True, capture_output=True, timeout=15)
        return "warning" not in r.stdout.lower() and "error" not in r.stdout.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------- Data extraction ----------

def find_reports(src: Path, gen: str) -> dict[tuple[str, int], Path]:
    """Map (instance_id, patch_id) → path to the per-instance report.json.

    Scans both run_evaluation (gpt5_mini) and run_evaluation_v1
    (haiku/qwen3/sonnet) — different runs landed under different parent dirs.
    """
    out: dict[tuple[str, int], Path] = {}
    bases = [src / "eval" / "logs" / "run_evaluation",
             src / "eval" / "logs" / "run_evaluation_v1"]
    for base in bases:
        if not base.exists():
            continue
        for pid in (0, 1, 2):
            pdir = base / f"{gen}_p{pid}" / f"{gen}__p{pid}"
            if not pdir.exists():
                continue
            for inst_dir in pdir.iterdir():
                rep = inst_dir / "report.json"
                if rep.exists():
                    out[(inst_dir.name, pid)] = rep
    return out


def find_predictions(src: Path, gen: str) -> dict[tuple[str, int], dict]:
    """Map (instance_id, patch_id) → prediction record (has model_patch + raw response)."""
    out: dict[tuple[str, int], dict] = {}
    for pid in (0, 1, 2):
        pred = src / gen / f"predictions_p{pid}.jsonl"
        if not pred.exists():
            continue
        for line in open(pred):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "instance_id" not in r:
                continue
            out[(r["instance_id"], pid)] = r
    return out


def find_existing_l3(src: Path, gen: str) -> dict[tuple[str, int], bool]:
    """Pull L3_llm_review from existing critic_results.jsonl if present."""
    out: dict[tuple[str, int], bool] = {}
    crit = src / gen / "critic_results.jsonl"
    if not crit.exists():
        return out
    for line in open(crit):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        l3 = r.get("L3_llm_review")
        if l3 is None:
            continue
        out[(r["instance_id"], r["patch_id"])] = bool(l3)
    return out


def extract_diff_from_prediction(rec: dict) -> str:
    """SWE-bench predictions store the diff under 'model_patch'."""
    return rec.get("model_patch") or rec.get("patch") or ""


# ---------- Per-generator pipeline ----------

def process_generator(gen: str, src: Path, dst_root: Path, client,
                       max_workers_l3: int, force_l3: bool) -> None:
    log.info("[%s] starting", gen)
    reports = find_reports(src, gen)
    preds = find_predictions(src, gen)
    existing_l3 = find_existing_l3(src, gen)
    log.info("[%s] %d reports, %d predictions, %d cached L3",
             gen, len(reports), len(preds), len(existing_l3))

    keys = sorted(set(reports) & set(preds))
    if not keys:
        log.warning("[%s] no overlap between reports and predictions", gen)
        return

    # SWE-bench instances → problem statement (for L3 review). Use the dataset.
    log.info("[%s] loading SWE-bench Lite for problem statements...", gen)
    os.environ.setdefault("HF_HOME", "/mnt/data/users/vlad.smirnov/hf_cache")
    from datasets import load_dataset
    swe = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    by_inst = {r["instance_id"]: r for r in swe}

    # First pass: synchronous L0/L1/L2/Y extraction (no API)
    rows: list[dict] = []
    need_l3: list[tuple] = []
    for (inst, pid) in keys:
        rep = json.loads(reports[(inst, pid)].read_text())
        rep_inner = rep.get(inst, {})
        ts = rep_inner.get("tests_status", {}) or {}
        f2p = ts.get("FAIL_TO_PASS", {}) or {}
        Y = bool(rep_inner.get("resolved"))
        l2 = (len(f2p.get("failure", []) or []) == 0
              and len(f2p.get("success", []) or []) > 0)
        diff = extract_diff_from_prediction(preds[(inst, pid)])
        # L0 on the added code only
        added = "\n".join(line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
        l0 = _critic_L0(added)
        l1 = _critic_L1_diff(diff)
        l3 = existing_l3.get((inst, pid))
        rec = {
            "generator": gen, "instance_id": inst, "patch_id": pid,
            "Y": int(Y),
            "L0_syntax": bool(l0),
            "L1_lint": bool(l1),
            "L2_public_tests": bool(l2),
            "L3_llm_review": l3,
            "diff_chars": len(diff),
        }
        rows.append(rec)
        if force_l3 or l3 is None:
            problem = by_inst.get(inst, {}).get("problem_statement", "")[:3000]
            need_l3.append((inst, pid, problem, diff))

    log.info("[%s] %d records assembled, %d need L3 backfill", gen, len(rows), len(need_l3))

    # Second pass: parallel L3 calls
    if need_l3 and client is not None:
        idx_by_key = {(r["instance_id"], r["patch_id"]): i for i, r in enumerate(rows)}
        n_calls = 0
        cost_total = 0.0
        with ThreadPoolExecutor(max_workers=max_workers_l3) as ex:
            futs = {}
            for inst, pid, problem, diff in need_l3:
                futs[ex.submit(critic_L3_review, problem, diff, client)] = (inst, pid)
            for fut in as_completed(futs):
                inst, pid = futs[fut]
                try:
                    passed, cost = fut.result()
                except Exception as e:
                    log.warning("[%s] L3 failed for %s_p%d: %s", gen, inst, pid, e)
                    continue
                rows[idx_by_key[(inst, pid)]]["L3_llm_review"] = bool(passed)
                cost_total += cost
                n_calls += 1
                if n_calls % 50 == 0:
                    log.info("[%s] L3 progress: %d/%d (cost so far $%.4f)",
                             gen, n_calls, len(need_l3), cost_total)
        log.info("[%s] L3 backfill: %d calls, total $%.4f", gen, n_calls, cost_total)

    # Write
    gen_dst = dst_root / gen
    gen_dst.mkdir(parents=True, exist_ok=True)
    out_path = gen_dst / "critic_results.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log.info("[%s] wrote %s (%d rows)", gen, out_path, len(rows))

    # Likelihood tables
    n = len(rows)
    n_y1 = sum(1 for r in rows if r["Y"] == 1)
    n_y0 = n - n_y1
    def likes_for(field: str) -> dict:
        TP = sum(1 for r in rows if r["Y"] == 1 and r.get(field))
        FP = sum(1 for r in rows if r["Y"] == 0 and r.get(field))
        FN = n_y1 - TP
        TN = n_y0 - FP
        p1 = (TP + 1) / (n_y1 + 2)
        p0 = (FP + 1) / (n_y0 + 2)
        return {"P_pass_given_Y1": p1, "P_pass_given_Y0": p0,
                "gap": p1 - p0, "TP": TP, "FN": FN, "FP": FP, "TN": TN}
    likes = {
        "generator": gen,
        "n_records": n, "n_evaluated": n, "n_resolved": n_y1,
        "prior_Y1": (n_y1 + 1) / (n + 2),
        "critic_likelihoods": {
            "L0_syntax": likes_for("L0_syntax"),
            "L1_lint": likes_for("L1_lint"),
            "L2_public_tests": likes_for("L2_public_tests"),
            "L3_llm_review": likes_for("L3_llm_review"),
        },
        "smoothing": "Beta(1,1)",
    }
    (gen_dst / "likelihood_tables.json").write_text(json.dumps(likes, indent=2))
    cl = likes["critic_likelihoods"]
    print(f"\n=== {gen} (n={n}) ===")
    print(f"  prior_Y1 = {likes['prior_Y1']:.3f}  ({n_y1}/{n} resolved)")
    for k, v in cl.items():
        print(f"  {k:<20} P(z|Y=1)={v['P_pass_given_Y1']:.3f}  P(z|Y=0)={v['P_pass_given_Y0']:.3f}  gap={v['gap']:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--dst-root", type=Path, required=True)
    parser.add_argument("--generators", required=True)
    parser.add_argument("--max-workers-gen", type=int, default=4,
                        help="parallel processing across generators")
    parser.add_argument("--max-workers-l3", type=int, default=4,
                        help="parallel L3 API calls within a generator")
    parser.add_argument("--force-l3", action="store_true",
                        help="re-call L3 even if cached value exists")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    client = None
    if api_key:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    else:
        log.warning("OPENROUTER_API_KEY not set, L3 backfill disabled")

    args.dst_root.mkdir(parents=True, exist_ok=True)
    gens = [g.strip() for g in args.generators.split(",") if g.strip()]

    # Process each generator in its own thread (each has its own L3 ThreadPool)
    with ThreadPoolExecutor(max_workers=args.max_workers_gen) as ex:
        futs = {ex.submit(process_generator, gen, args.src_root, args.dst_root,
                           client, args.max_workers_l3, args.force_l3): gen
                for gen in gens}
        for fut in as_completed(futs):
            gen = futs[fut]
            try:
                fut.result()
            except Exception as e:
                log.error("[%s] crashed: %s", gen, e)


if __name__ == "__main__":
    main()
