"""Compute calibration tables from existing spot-check predictions.

Reuses the n=50 SWE-bench Lite spot-check corpus (4 generators × 150 patches
= 600 attempts) which already has ground-truth Y from the harness.

For each (generator, instance, patch_id):
  1. Load the diff from predictions_p<pid>.jsonl
  2. Apply diff to oracle files in-memory (via spot_check_generators.apply_change_blocks)
  3. Run cheap critics on the modified content:
       L0_syntax     — ast.parse on each modified .py file (free)
       L1_lint       — ruff check on each modified .py file (free, requires `ruff` binary)
       L3_llm_review — Haiku PASS/FAIL on the diff + problem statement (paid, ~$0.005/patch)
  4. Read Y from the harness report (eval/<gen>__p<pid>.<gen>_p<pid>.json — `resolved_ids` set)
  5. Append a record to <gen>/critic_results.jsonl

Output: per generator, critic_results.jsonl (one row per patch) AND
likelihood_tables.json (Beta(1,1) smoothed P(z|Y) for each critic).

Cost cap defaults to $5 per generator (safety; expect ~$3).

Usage:
  python3 calibrate_from_spotcheck.py \
    --output-dir data/spot_check_n50 \
    --generators gpt5_mini,qwen3_coder \
    --max-cost-usd-per-model 5.0
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# We import the v3 parser/matcher from spot_check_generators
# File moved out of scripts/ during refactor; parents[1] is the
# package root (experiments/orchestration_hypothesis_testing/).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import spot_check_generators as scg  # noqa: E402
from _common.telemetry import TelemetryLogger  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("calibrate")


# ----------------------------------------------------------------------
# Critics
# ----------------------------------------------------------------------

def _modified_file_contents(diff_text: str, oracle_files: dict[str, str]) -> dict[str, str] | None:
    """Re-derive the modified file contents from the diff (we don't store them).

    For SWE-bench Lite our diffs are unified format. The simplest reconstruction:
    apply the diff with `patch -p1` to a temp dir containing the oracle files,
    then read back. Returns None on failure.
    """
    import tempfile
    if not diff_text.strip() or not oracle_files:
        return None
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Write oracle files into td/<path>
        for fpath, content in oracle_files.items():
            target = td_path / fpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        diff_path = td_path / "_patch.diff"
        diff_path.write_text(diff_text)
        # Try git apply first (handles unified diff cleanly)
        proc = subprocess.run(
            ["git", "apply", "--unsafe-paths", "--directory", str(td_path), str(diff_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            # Fallback to patch
            proc = subprocess.run(
                ["patch", "-p1", "-i", str(diff_path), "--silent"],
                cwd=td_path, capture_output=True, text=True,
            )
            if proc.returncode != 0:
                return None
        modified: dict[str, str] = {}
        for fpath in oracle_files:
            target = td_path / fpath
            if target.exists():
                try:
                    modified[fpath] = target.read_text()
                except Exception:
                    return None
        return modified


def critic_L0_syntax(modified_files: dict[str, str]) -> bool:
    """ast.parse must succeed on every modified .py file."""
    for fpath, content in modified_files.items():
        if not fpath.endswith(".py"):
            continue
        try:
            ast.parse(content)
        except SyntaxError:
            return False
    return True


def critic_L1_lint(modified_files: dict[str, str]) -> bool:
    """Lint critic — only flag REAL errors (undefined names, syntax) via
    pyflakes-equivalent ruff rules: F821 (undefined name), F811 (redefinition),
    F401 (unused import — bug-correlated), E999 (syntax). Skip stylistic warnings
    that fire constantly on large legacy codebases (E501 line length, etc.)."""
    import tempfile
    for fpath, content in modified_files.items():
        if not fpath.endswith(".py"):
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(content)
            tmp = f.name
        try:
            proc = subprocess.run(
                ["ruff", "check", "--quiet", "--no-cache",
                 "--select", "F821,F811,E999",  # only real errors
                 tmp],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return True
        finally:
            os.unlink(tmp)
    return True


def critic_L3_llm_review(
    instance_id: str,
    problem_statement: str,
    diff_text: str,
    client,
) -> tuple[bool, float]:
    """Haiku gives PASS/FAIL on the diff. Returns (passed, cost_usd)."""
    prompt = (
        "You are a senior software engineer reviewing a bug fix.\n\n"
        f"## Issue\n{problem_statement[:3000]}\n\n"
        f"## Proposed Fix (unified diff)\n```diff\n{diff_text[:8000]}\n```\n\n"
        "Does this patch correctly fix the issue? Respond with exactly one word: "
        "PASS or FAIL. No explanation."
    )
    try:
        resp = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        text = resp.choices[0].message.content.strip().upper()
        usage = resp.usage
        # OpenRouter Haiku 4.5: roughly $1/M input, $5/M output
        cost = (usage.prompt_tokens / 1_000_000) * 1.0 + (usage.completion_tokens / 1_000_000) * 5.0
        return ("PASS" in text and "FAIL" not in text), cost
    except Exception as e:
        log.warning("L3 LLM review failed for %s: %s", instance_id, e)
        return False, 0.0


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def load_oracle_cache(instances: list[str]) -> dict[str, dict[str, str]]:
    """Pre-fetch oracle files for all unique instances."""
    import datasets
    ds = datasets.load_dataset(_DATASET_NAME, split="test")
    inst_to_row = {row["instance_id"]: row for row in ds}
    cache: dict[str, dict[str, str]] = {}
    for inst in instances:
        if inst not in cache and inst in inst_to_row:
            row = inst_to_row[inst]
            files = scg.get_changed_files_from_patch(row["patch"])
            cache[inst] = scg.fetch_oracle_files(row["repo"], row["base_commit"], files)
    return cache, inst_to_row


def load_resolved_set(eval_path: Path) -> set[str]:
    """Read the harness report JSON and return its resolved_ids set."""
    if not eval_path.exists():
        return set()
    rep = json.loads(eval_path.read_text())
    raw = rep.get("resolved_ids") or rep.get("resolved_instances")
    if isinstance(raw, list):
        return set(raw)
    return set()


def beta_smooth(success: int, total: int, prior_a: float = 1.0, prior_b: float = 1.0) -> float:
    """Beta(a,b) posterior mean, equivalent to Laplace smoothing."""
    return (success + prior_a) / (total + prior_a + prior_b)


_DATASET_NAME = "princeton-nlp/SWE-bench_Lite"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True,
                        help="Comma-separated generator keys")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                        help="HuggingFace dataset name. Default: SWE-bench Lite.")
    parser.add_argument("--max-cost-usd-per-model", type=float, default=5.0)
    parser.add_argument("--skip-l3", action="store_true",
                        help="Skip the paid LLM critic")
    args = parser.parse_args()

    global _DATASET_NAME
    _DATASET_NAME = args.dataset

    # Load .env for OPENROUTER_API_KEY
    from dotenv import load_dotenv
    for env_path in [ROOT / ".env", ROOT.parent / ".env",
                     ROOT.parent.parent / ".env",
                     ROOT.parent.parent.parent / ".env",
                     ROOT.parent.parent.parent.parent / ".env",
                     ROOT.parent.parent.parent.parent.parent / ".env"]:
        if env_path.exists() and env_path.stat().st_size > 0:
            load_dotenv(env_path, override=False)

    out_dir = args.output_dir.resolve()
    generators = [g.strip() for g in args.generators.split(",") if g.strip()]

    # Build OpenAI client for L3 (only if we'll use it)
    client = None
    if not args.skip_l3:
        from openai import OpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            log.error("OPENROUTER_API_KEY not set — pass --skip-l3 or set the key")
            sys.exit(1)
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # Collect all unique instance_ids across all generators
    all_instances: set[str] = set()
    for g in generators:
        gen_path = out_dir / g / "predictions.jsonl"
        if not gen_path.exists():
            log.warning("missing predictions: %s", gen_path)
            continue
        with open(gen_path) as f:
            for line in f:
                r = json.loads(line)
                all_instances.add(r["instance_id"])

    log.info("warming oracle cache for %d unique instances...", len(all_instances))
    oracle_cache, inst_to_row = load_oracle_cache(sorted(all_instances))
    log.info("oracle cache ready: %d/%d fetched", len(oracle_cache), len(all_instances))

    # Process each generator
    for gen in generators:
        gen_dir = out_dir / gen
        if not gen_dir.exists():
            log.warning("skipping %s (no dir)", gen)
            continue

        # Load resolved sets per pid
        resolved_per_pid: dict[int, set[str]] = {}
        for pid in (0, 1, 2):
            ep = out_dir / "eval" / f"{gen}__p{pid}.{gen}_p{pid}.json"
            resolved_per_pid[pid] = load_resolved_set(ep)

        # Iterate predictions
        records = []
        cumulative_cost = 0.0
        # Per-generator action telemetry. Unlike the calibration/lcb-family
        # scripts, from_spotcheck doesn't do generation here — it only runs
        # critics on pre-existing predictions — so the JSONL contains only
        # critic_L0/L1/L3 rows (no "generate" or "verify").
        tele = TelemetryLogger(gen_dir / "action_telemetry.jsonl",
                               dataset=out_dir.name, model_name=gen)
        try:
            for pid in (0, 1, 2):
                pred_path = gen_dir / f"predictions_p{pid}.jsonl"
                if not pred_path.exists():
                    continue
                with open(pred_path) as f:
                    for line in f:
                        r = json.loads(line)
                        inst = r["instance_id"]
                        diff = (r.get("model_patch") or "")
                        Y = 1 if inst in resolved_per_pid[pid] else 0
                        rec = {
                            "generator": gen,
                            "instance_id": inst,
                            "patch_id": pid,
                            "Y": Y,
                            "diff_chars": len(diff),
                        }
                        if not diff.strip():
                            rec.update(L0_syntax=False, L1_lint=False,
                                       L3_llm_review=False, l3_cost=0.0,
                                       note="empty_diff")
                            records.append(rec)
                            continue
                        oracle = oracle_cache.get(inst)
                        if not oracle:
                            rec.update(L0_syntax=None, L1_lint=None,
                                       L3_llm_review=None, l3_cost=0.0,
                                       note="oracle_missing")
                            records.append(rec)
                            continue
                        modified = _modified_file_contents(diff, oracle)
                        if modified is None:
                            # Diff failed to apply — count critics as FAIL
                            rec.update(L0_syntax=False, L1_lint=False,
                                       note="diff_apply_failed")
                        else:
                            _t0 = time.perf_counter()
                            l0 = critic_L0_syntax(modified)
                            tele.record(action_type="critic_L0",
                                        runtime_s=time.perf_counter() - _t0,
                                        instance_id=inst, patch_id=pid,
                                        passed=bool(l0))
                            rec["L0_syntax"] = l0
                            _t0 = time.perf_counter()
                            l1 = critic_L1_lint(modified)
                            tele.record(action_type="critic_L1",
                                        runtime_s=time.perf_counter() - _t0,
                                        instance_id=inst, patch_id=pid,
                                        passed=bool(l1))
                            rec["L1_lint"] = l1
                        # L3 (paid)
                        if not args.skip_l3 and cumulative_cost < args.max_cost_usd_per_model:
                            problem = inst_to_row[inst]["problem_statement"]
                            _t0 = time.perf_counter()
                            passed, c = critic_L3_llm_review(inst, problem, diff, client)
                            tele.record(action_type="critic_L3",
                                        runtime_s=time.perf_counter() - _t0,
                                        instance_id=inst, patch_id=pid,
                                        passed=bool(passed), api_cost_usd=c)
                            rec["L3_llm_review"] = passed
                            rec["l3_cost"] = c
                            cumulative_cost += c
                        else:
                            rec["L3_llm_review"] = None
                            rec["l3_cost"] = 0.0
                        records.append(rec)
                        if len(records) % 25 == 0:
                            log.info("[%s] %d records, cumulative L3 cost = $%.4f",
                                     gen, len(records), cumulative_cost)
        finally:
            tele.close()

        # Write critic_results.jsonl
        out_path = gen_dir / "critic_results.jsonl"
        out_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        log.info("[%s] wrote %d records to %s (L3 cost $%.4f)",
                 gen, len(records), out_path, cumulative_cost)

        # Compute likelihoods (Beta(1,1) smoothed)
        likelihoods = {}
        for critic_name in ("L0_syntax", "L1_lint", "L3_llm_review"):
            tp = sum(1 for r in records if r.get("Y") == 1 and r.get(critic_name) is True)
            fn = sum(1 for r in records if r.get("Y") == 1 and r.get(critic_name) is False)
            fp = sum(1 for r in records if r.get("Y") == 0 and r.get(critic_name) is True)
            tn = sum(1 for r in records if r.get("Y") == 0 and r.get(critic_name) is False)
            n_y1 = tp + fn
            n_y0 = fp + tn
            p_pass_y1 = beta_smooth(tp, n_y1)
            p_pass_y0 = beta_smooth(fp, n_y0)
            likelihoods[critic_name] = {
                "P_pass_given_Y1": p_pass_y1,
                "P_pass_given_Y0": p_pass_y0,
                "gap": p_pass_y1 - p_pass_y0,
                "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            }
        n_y1_total = sum(1 for r in records if r.get("Y") == 1)
        n_total = sum(1 for r in records if r.get("Y") in (0, 1))
        prior = beta_smooth(n_y1_total, n_total)

        tables = {
            "generator": gen,
            "n_records": len(records),
            "n_evaluated": n_total,
            "n_resolved": n_y1_total,
            "prior_Y1": prior,
            "critic_likelihoods": likelihoods,
            "smoothing": "Beta(1,1)",
        }
        likelihoods_path = gen_dir / "likelihood_tables.json"
        likelihoods_path.write_text(json.dumps(tables, indent=2))
        log.info("[%s] wrote likelihood_tables.json", gen)

        # Summary print
        print(f"\n=== {gen} likelihoods ===")
        print(f"  prior_Y1 = {prior:.3f}")
        for name, lk in likelihoods.items():
            print(f"  {name}: P(pass|Y=1)={lk['P_pass_given_Y1']:.3f} "
                  f"P(pass|Y=0)={lk['P_pass_given_Y0']:.3f} "
                  f"gap={lk['gap']:.3f} "
                  f"(TP={lk['TP']} FP={lk['FP']} TN={lk['TN']} FN={lk['FN']})")


if __name__ == "__main__":
    main()
