"""Streamlined LCB calibration for the orchestration controller.

Mirrors calibrate_from_spotcheck.py but for LiveCodeBench:
  - Loads LCB hard problems
  - Generates K solutions per problem with chosen OpenRouter model
  - Extracts code from response (no SEARCH/REPLACE — LCB models output raw code)
  - Runs PUBLIC tests as L2 critic
  - Runs PRIVATE tests as ground-truth verifier (Y)
  - Runs L0 (ast.parse), L3 (Haiku review) on each solution
  - Saves critic_results.jsonl + likelihood_tables.json (Beta(1,1))

Per-generator output goes under <output-dir>/<gen>/ matching our
SWE-bench Lite layout.

Cost cap per model + per-call cost tracking via the same cost_tracker
the spot-check uses.

Usage:
  python3 lcb_calibrate.py \\
    --output-dir data/lcb_calibration \\
    --generators gpt5_mini,qwen3_coder \\
    --n-instances 50 \\
    --n-patches 3 \\
    --max-cost-usd-per-model gpt5_mini=3.0,qwen3_coder=3.0
"""
from __future__ import annotations

import argparse
import ast
import base64
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "scripts" in str(Path(__file__).resolve()) else Path("/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/.claude/worktrees/reverent-vaughan-017bf5/experiments/orchestration_hypothesis_testing")
sys.path.insert(0, str(ROOT / "scripts"))
from cost_tracker import CostTracker  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("lcb_cal")


# ---------- LCB dataset loader ----------

CACHED_LCB = Path("/mnt/data/users/vlad.smirnov/hf_cache/hub/datasets--livecodebench--code_generation_lite/snapshots/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test.jsonl")


def load_lcb(difficulty: str = "hard", platform: str = "leetcode") -> list[dict]:
    """Load LiveCodeBench code-generation problems.

    Prefers the locally-cached HF snapshot (from earlier sympy/LCB work)
    to avoid the broken Xet downloader. Falls back to hf_hub_download if
    cache is missing.
    """
    if CACHED_LCB.exists():
        path = CACHED_LCB
        log.info("using cached LCB at %s", path)
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            "livecodebench/code_generation_lite",
            "test.jsonl",
            repo_type="dataset",
        )
    raw = []
    with open(path) as f:
        for line in f:
            raw.append(json.loads(line))
    out = []
    for r in raw:
        if difficulty != "all" and r.get("difficulty") != difficulty:
            continue
        if platform != "all" and r.get("platform") != platform:
            continue
        out.append(r)
    log.info("loaded %d LCB problems (%s/%s) of %d total", len(out), platform, difficulty, len(raw))
    return out


def decode_private_tests(encoded: str) -> list[dict]:
    """LCB encodes private_test_cases as zlib-compressed pickled JSON."""
    if not encoded:
        return []
    try:
        decoded = json.loads(pickle.loads(zlib.decompress(base64.b64decode(encoded.encode("utf-8")))))
        return decoded
    except Exception as e:
        log.warning("decode_private_tests failed: %s", e)
        return []


# ---------- Code extraction (LCB outputs raw code, simpler than SEARCH/REPLACE) ----------

def extract_code(response: str) -> str:
    """Extract Python code from a model response. Tries fenced blocks first."""
    if not response:
        return ""
    m = re.search(r"```(?:python)?\s*\n([\s\S]+?)```", response)
    if m:
        return m.group(1).strip()
    return response.strip()


# ---------- Test runners ----------

TEST_PYTHON = sys.executable
TIMEOUT_S = 5
MAX_PRIVATE_TESTS = 12


def run_solution_stdin(code: str, stdin: str) -> tuple[bool, str]:
    """Run code as standalone script with stdin input. Returns (passed, output)."""
    if not code:
        return False, ""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        script = f.name
    try:
        proc = subprocess.run(
            [TEST_PYTHON, script], input=stdin, capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        return proc.returncode == 0, proc.stdout
    except subprocess.TimeoutExpired:
        return False, ""
    except Exception:
        return False, ""
    finally:
        os.unlink(script)


def check_tests(code: str, tests: list[dict]) -> tuple[int, int]:
    """Run all tests, return (n_passed, n_total)."""
    if not tests:
        return 0, 0
    n_pass = 0
    for t in tests:
        stdin = t.get("input", "")
        expected = t.get("output", "").strip()
        ok, out = run_solution_stdin(code, stdin)
        if ok and out.strip() == expected:
            n_pass += 1
    return n_pass, len(tests)


# ---------- Critics ----------

def critic_L0_syntax(code: str) -> bool:
    if not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def critic_L1_lint(code: str) -> bool:
    """Same conservative ruff ruleset as SWE-bench Lite (only F821, F811, E999)."""
    if not code.strip():
        return False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    try:
        proc = subprocess.run(
            ["ruff", "check", "--quiet", "--no-cache", "--select", "F821,F811,E999", tmp],
            capture_output=True, text=True, timeout=15,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return True
    finally:
        os.unlink(tmp)


def critic_L3_review(problem: str, code: str, client) -> tuple[bool, float]:
    """Haiku PASS/FAIL on (problem, code). Returns (passed, cost_usd)."""
    prompt = (
        "You are a senior software engineer reviewing a code submission.\n\n"
        f"## Problem\n{problem[:3000]}\n\n"
        f"## Submitted code\n```python\n{code[:6000]}\n```\n\n"
        "Does this code correctly solve the problem? Respond with exactly one word: "
        "PASS or FAIL. No explanation."
    )
    try:
        resp = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=10,
        )
        text = resp.choices[0].message.content.strip().upper()
        usage = resp.usage
        cost = (usage.prompt_tokens / 1_000_000) * 1.0 + (usage.completion_tokens / 1_000_000) * 5.0
        return ("PASS" in text and "FAIL" not in text), cost
    except Exception as e:
        log.warning("L3 failed: %s", e)
        return False, 0.0


# ---------- Generators ----------

GENERATORS = {
    "gpt5_mini":   ("openai/gpt-5-mini",   "OpenAI gpt-5-mini"),
    "qwen3_coder": ("qwen/qwen3-coder",    "Qwen3 Coder"),
}


def build_prompt(problem: dict) -> str:
    title = problem.get("question_title", "")
    statement = problem.get("question_content", "")
    starter = problem.get("starter_code", "") or ""
    starter_block = ""
    if starter.strip():
        starter_block = "Starter code:\n```python\n" + starter + "\n```\n\n"
    return (
        f"# Problem: {title}\n\n"
        f"{statement}\n\n"
        f"{starter_block}"
        "Write a complete Python solution. Output ONLY the code in a single ```python```\n"
        "code block. Read input from stdin and print results to stdout."
    )


# ---------- Main pipeline ----------

def cost_for_call(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Approximate per-call cost (USD)."""
    if "gpt-5-mini" in model_id:
        return (prompt_tokens / 1_000_000) * 0.5 + (completion_tokens / 1_000_000) * 4.0
    if "qwen3-coder" in model_id:
        return (prompt_tokens / 1_000_000) * 0.4 + (completion_tokens / 1_000_000) * 1.6
    return (prompt_tokens / 1_000_000) * 1.0 + (completion_tokens / 1_000_000) * 5.0


def calibrate_one_generator(
    gen_key: str, problems: list[dict], n_patches: int, out_dir: Path,
    max_cost_usd: float, client,
) -> None:
    model_id, label = GENERATORS[gen_key]
    log.info("=== %s (%s) — cap $%.2f ===", gen_key, model_id, max_cost_usd)
    gen_dir = out_dir / gen_key
    gen_dir.mkdir(parents=True, exist_ok=True)
    cost = CostTracker(name=gen_key, cap_usd=max_cost_usd, log_path=gen_dir / "cost_log.jsonl")
    records = []
    raw_path = gen_dir / "raw_responses"
    raw_path.mkdir(exist_ok=True)
    for inst in problems:
        if cost.capped:
            log.warning("[%s] cost cap reached, stopping", gen_key)
            break
        public_tests = inst.get("public_test_cases") or []
        if isinstance(public_tests, str):
            try:
                public_tests = json.loads(public_tests)
            except Exception:
                public_tests = []
        private_tests = decode_private_tests(inst.get("private_test_cases", "") or "")
        for pid in range(n_patches):
            if cost.capped:
                break
            inst_id = inst["question_id"]
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": build_prompt(inst)}],
                    temperature=0.7, max_tokens=4000,
                )
                text = resp.choices[0].message.content or ""
                usage = resp.usage
                c = cost_for_call(model_id, usage.prompt_tokens, usage.completion_tokens)
                cost.record(c, prompt_tokens=usage.prompt_tokens,
                            completion_tokens=usage.completion_tokens,
                            instance_id=inst_id, patch_id=pid)
            except Exception as e:
                log.warning("[%s] gen failed for %s: %s", gen_key, inst_id, e)
                continue
            (raw_path / f"{inst_id}_p{pid}.txt").write_text(text)
            code = extract_code(text)
            # L2: public tests
            l2_pass, l2_total = check_tests(code, public_tests)
            l2_ok = (l2_pass == l2_total) and l2_total > 0
            # Y: private tests (capped for speed; 12 is enough to estimate Y reliably)
            y_pass, y_total = check_tests(code, private_tests[:MAX_PRIVATE_TESTS])
            Y = 1 if (y_pass == y_total) and y_total > 0 else 0
            # L0/L1
            l0 = critic_L0_syntax(code)
            l1 = critic_L1_lint(code)
            # L3 review (paid, only if budget allows)
            l3 = None
            if not cost.capped:
                l3_pass, l3_cost = critic_L3_review(inst.get("question_content", "")[:3000], code, client)
                cost.record(l3_cost, prompt_tokens=0, completion_tokens=0,
                            instance_id=inst_id, patch_id=pid,
                            extra={"kind": "L3_review"})
                l3 = l3_pass
            records.append({
                "generator": gen_key,
                "instance_id": inst_id,
                "patch_id": pid,
                "Y": Y,
                "L0_syntax": l0,
                "L1_lint": l1,
                "L2_public_tests": l2_ok,
                "L3_llm_review": l3,
                "diff_chars": len(code),
                "y_pass_rate": (y_pass / max(y_total, 1)),
                "l2_pass_rate": (l2_pass / max(l2_total, 1)),
            })
            if len(records) % 10 == 0:
                log.info("[%s] %d records, cost $%.4f", gen_key, len(records), cost.total_usd)
    # Save
    (gen_dir / "critic_results.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    summary = {
        "model": gen_key, "cap_usd": cost.cap_usd, "total_usd": cost.total_usd,
        "n_calls": cost.n_calls, "remaining_usd": cost.remaining,
        "capped": cost.capped,
    }
    (gen_dir / "cost_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("[%s] done: %d records, total cost $%.4f", gen_key, len(records), cost.total_usd)
    return records


def compute_likelihoods(records: list[dict], gen: str, out_dir: Path) -> None:
    from collections import Counter
    rows = [r for r in records if r["Y"] in (0, 1)]
    n_y1 = sum(1 for r in rows if r["Y"] == 1)
    n_total = len(rows)
    prior = (n_y1 + 1) / (n_total + 2)
    likelihoods = {}
    for k in ("L0_syntax", "L1_lint", "L2_public_tests", "L3_llm_review"):
        tp = sum(1 for r in rows if r["Y"] == 1 and r.get(k) is True)
        fn = sum(1 for r in rows if r["Y"] == 1 and r.get(k) is False)
        fp = sum(1 for r in rows if r["Y"] == 0 and r.get(k) is True)
        tn = sum(1 for r in rows if r["Y"] == 0 and r.get(k) is False)
        n_y1_with = tp + fn
        n_y0_with = fp + tn
        p_y1 = (tp + 1) / (n_y1_with + 2) if n_y1_with > 0 else None
        p_y0 = (fp + 1) / (n_y0_with + 2) if n_y0_with > 0 else None
        gap = (p_y1 - p_y0) if (p_y1 is not None and p_y0 is not None) else None
        likelihoods[k] = {
            "P_pass_given_Y1": p_y1, "P_pass_given_Y0": p_y0, "gap": gap,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        }
    tables = {
        "generator": gen, "n_records": len(records), "n_evaluated": n_total,
        "n_resolved": n_y1, "prior_Y1": prior,
        "critic_likelihoods": likelihoods, "smoothing": "Beta(1,1)",
    }
    (out_dir / gen / "likelihood_tables.json").write_text(json.dumps(tables, indent=2))
    print(f"\n=== {gen} likelihoods ===")
    print(f"  prior_Y1 = {prior:.3f}")
    for k, v in likelihoods.items():
        if v["gap"] is not None:
            print(f"  {k}: P(pass|Y=1)={v['P_pass_given_Y1']:.3f} P(pass|Y=0)={v['P_pass_given_Y0']:.3f} gap={v['gap']:.3f} (TP={v['TP']} FP={v['FP']} TN={v['TN']} FN={v['FN']})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--generators", required=True)
    p.add_argument("--n-instances", type=int, default=50)
    p.add_argument("--n-patches", type=int, default=3)
    p.add_argument("--difficulty", default="hard", choices=["easy", "medium", "hard", "all"])
    p.add_argument("--platform", default="leetcode", choices=["leetcode", "atcoder", "codeforces", "all"])
    p.add_argument("--max-cost-usd-per-model", default="3.0",
                   help="Single float or 'key=val,...' per model")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from dotenv import load_dotenv
    for env_path in [ROOT / ".env", ROOT.parent / ".env",
                     ROOT.parent.parent / ".env", ROOT.parent.parent.parent / ".env",
                     ROOT.parent.parent.parent.parent / ".env",
                     ROOT.parent.parent.parent.parent.parent / ".env"]:
        if env_path.exists() and env_path.stat().st_size > 0:
            load_dotenv(env_path, override=False)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    generators = [g.strip() for g in args.generators.split(",") if g.strip()]

    # Parse cost caps
    caps: dict[str, float] = {}
    if "=" in args.max_cost_usd_per_model:
        for pair in args.max_cost_usd_per_model.split(","):
            k, v = pair.strip().split("=")
            caps[k.strip()] = float(v)
    else:
        flat = float(args.max_cost_usd_per_model)
        for g in generators:
            caps[g] = flat

    # Load LCB
    problems = load_lcb(difficulty=args.difficulty, platform=args.platform)
    import random
    random.seed(args.seed)
    random.shuffle(problems)
    problems = problems[: args.n_instances]
    log.info("sampled %d problems", len(problems))

    # Save sample manifest
    (out_dir / "sample.json").write_text(json.dumps([
        {"question_id": p["question_id"], "platform": p.get("platform"), "difficulty": p.get("difficulty")}
        for p in problems
    ], indent=2))

    # OpenAI client
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")

    # Calibrate each generator
    for g in generators:
        records = calibrate_one_generator(
            g, problems, args.n_patches, out_dir, caps.get(g, 3.0), client,
        )
        compute_likelihoods(records, g, out_dir)


if __name__ == "__main__":
    main()
