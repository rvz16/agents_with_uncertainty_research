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


def load_lcb(difficulty: str = "hard", platform: str = "leetcode",
             lcb_version: str = "v1") -> list[dict]:
    """Load LiveCodeBench code-generation problems.

    lcb_version:
      "v1"  → only the original test.jsonl (400 problems, May 2023 - Mar 2024)
      "all" → union of test.jsonl + test2.jsonl + ... + test6.jsonl
              (1055 problems through v6). Each testN.jsonl is the *delta*
              for that release, so the union is cumulative.

    Prefers the locally-cached HF snapshot to avoid the broken Xet
    downloader. Falls back to hf_hub_download if cache is missing.
    """
    if lcb_version == "v1":
        files = ["test.jsonl"]
    elif lcb_version == "all":
        files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl",
                 "test5.jsonl", "test6.jsonl"]
    else:
        raise ValueError(f"unknown lcb_version: {lcb_version!r}")

    raw = []
    seen_ids = set()
    for fname in files:
        cached_path = CACHED_LCB.parent / fname
        if cached_path.exists():
            path = cached_path
            log.info("using cached LCB %s", path)
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                "livecodebench/code_generation_lite",
                fname,
                repo_type="dataset",
            )
            log.info("downloaded LCB %s", path)
        n_in_file = 0
        for line in open(path):
            r = json.loads(line)
            if r["question_id"] in seen_ids:
                continue
            seen_ids.add(r["question_id"])
            raw.append(r)
            n_in_file += 1
        log.info("  +%d new problems from %s (cumulative pool: %d)",
                 n_in_file, fname, len(raw))

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


def _parse_lcb_test_input(input_str: str) -> list:
    """Parse LCB functional test inputs.

    LCB encodes args as one-per-line, each value JSON-encoded:
      "1"        -> str "1"
      "12"       -> str "12"
      1          -> int 1
      [1,2,3]    -> list [1,2,3]
    Returns the args as Python objects.
    """
    args = []
    for line in input_str.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            args.append(json.loads(line))
        except json.JSONDecodeError:
            args.append(line)  # fall back to raw string
    return args


def _parse_lcb_test_output(output_str: str):
    """Parse expected output. JSON-decoded if possible."""
    s = output_str.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def run_solution_functional(code: str, input_str: str, expected_output: str,
                             starter_code: str) -> bool:
    """Run a functional LeetCode-style test.

    Wraps the model's `class Solution` with a runner that calls the method
    declared in `starter_code` with parsed args, and compares the return
    value to the expected output (JSON-decoded).
    """
    if not code or not starter_code:
        return False
    # Extract method name from starter_code: `def <name>(self, ...)`
    m = re.search(r"def\s+(\w+)\s*\(\s*self", starter_code)
    if not m:
        return False
    method_name = m.group(1)
    try:
        args = _parse_lcb_test_input(input_str)
        expected = _parse_lcb_test_output(expected_output)
    except Exception:
        return False

    # Build a small driver script that imports the model code, calls the method, prints JSON.
    driver = f"""
import sys, json
{code}

try:
    sol = Solution()
    args = json.loads(sys.argv[1])
    result = sol.{method_name}(*args)
    print(json.dumps(result))
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(driver)
        script = f.name
    try:
        proc = subprocess.run(
            [TEST_PYTHON, script, json.dumps(args)],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        if proc.returncode != 0:
            return False
        try:
            actual = json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            actual = proc.stdout.strip()
        return actual == expected
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        os.unlink(script)


def check_tests(code: str, tests: list[dict], starter_code: str = "") -> tuple[int, int]:
    """Run all tests. Detects functional vs stdin tests automatically.

    Returns (n_passed, n_total). For functional tests, requires `starter_code`
    to extract the method name.
    """
    if not tests:
        return 0, 0
    n_pass = 0
    for t in tests:
        testtype = t.get("testtype", "stdin")
        if testtype == "functional" and starter_code.strip():
            ok = run_solution_functional(code, t.get("input", ""),
                                         t.get("output", ""), starter_code)
        else:
            stdin = t.get("input", "")
            expected = t.get("output", "").strip()
            ok, out = run_solution_stdin(code, stdin)
            ok = ok and out.strip() == expected
        if ok:
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
    "gpt5_mini":   ("openai/gpt-5-mini",          "OpenAI gpt-5-mini",       None),
    "qwen3_coder": ("qwen/qwen3-coder",           "Qwen3 Coder",             None),
    "haiku45":     ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5",        None),
    "sonnet45":    ("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5",      None),
    "qwen25_32b":  ("Qwen/Qwen2.5-Coder-32B-Instruct", "Qwen2.5-Coder-32B (open-weight, local vLLM)", "http://127.0.0.1:8003/v1"),
}



def _make_client(generator_key: str | None = None):
    """Build an OpenAI-compatible client.

    For qwen25_32b: local vLLM at base_url stored in GENERATORS tuple slot 2.
    Otherwise: OpenRouter.
    """
    from openai import OpenAI
    base_url = None
    if generator_key and generator_key in GENERATORS and len(GENERATORS[generator_key]) >= 3:
        base_url = GENERATORS[generator_key][2]
    if base_url:
        return OpenAI(api_key="EMPTY", base_url=base_url)
    return OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")


def build_prompt(problem: dict) -> str:
    title = problem.get("question_title", "")
    statement = problem.get("question_content", "")
    starter = problem.get("starter_code", "") or ""
    if starter.strip():
        # Functional (LeetCode-style) — model fills in the method
        return (
            f"# Problem: {title}\n\n"
            f"{statement}\n\n"
            f"Starter code (you MUST complete this exact class and method signature):\n"
            f"```python\n{starter}\n```\n\n"
            "Output ONLY a complete `class Solution:` definition in a single "
            "```python``` code block. The class must contain the method exactly "
            "as declared in the starter code. Do NOT add a `__main__` block or "
            "stdin parsing — your code will be imported and the method called "
            "directly with parsed arguments."
        )
    # Stdin-based (atcoder, codeforces) — model writes a script
    return (
        f"# Problem: {title}\n\n"
        f"{statement}\n\n"
        "Write a complete Python solution that reads input from stdin and "
        "writes results to stdout. Output ONLY the code in a single ```python``` "
        "code block."
    )


# ---------- Main pipeline ----------

def cost_for_call(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Approximate per-call cost (USD)."""
    if "gpt-5-mini" in model_id:
        return (prompt_tokens / 1_000_000) * 0.5 + (completion_tokens / 1_000_000) * 4.0
    if "qwen3-coder" in model_id:
        return (prompt_tokens / 1_000_000) * 0.4 + (completion_tokens / 1_000_000) * 1.6
    if "claude-haiku" in model_id:
        return (prompt_tokens / 1_000_000) * 1.0 + (completion_tokens / 1_000_000) * 5.0
    if "claude-sonnet" in model_id:
        return (prompt_tokens / 1_000_000) * 3.0 + (completion_tokens / 1_000_000) * 15.0
    return (prompt_tokens / 1_000_000) * 1.0 + (completion_tokens / 1_000_000) * 5.0


def calibrate_one_generator(
    gen_key: str, problems: list[dict], n_patches: int, out_dir: Path,
    max_cost_usd: float, client,
) -> None:
    # Per-generator clients: gen_client for the generation call (may be vLLM),
    # reviewer_client for L3 review (always OpenRouter; reviewer is Haiku-4.5).
    gen_client = _make_client(gen_key)
    reviewer_client = _make_client(None)
    model_id, label, _base_url = GENERATORS[gen_key]
    log.info("=== %s (%s) — cap $%.2f ===", gen_key, model_id, max_cost_usd)
    gen_dir = out_dir / gen_key
    gen_dir.mkdir(parents=True, exist_ok=True)
    cost = CostTracker(name=gen_key, cap_usd=max_cost_usd, log_path=gen_dir / "cost_log.jsonl")
    raw_path = gen_dir / "raw_responses"
    raw_path.mkdir(exist_ok=True)
    results_path = gen_dir / "critic_results.jsonl"

    # --- resume support: skip (inst_id, pid) tuples already persisted ---
    done: set[tuple[str, int]] = set()
    records: list[dict] = []
    if results_path.exists():
        with open(results_path) as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                done.add((str(r["instance_id"]), int(r["patch_id"])))
                records.append(r)
        log.info("[%s] resuming with %d records already persisted", gen_key, len(records))

    out_fp = open(results_path, "a", buffering=1)  # line-buffered append

    try:
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
                if (str(inst_id), pid) in done:
                    continue
                try:
                    resp = gen_client.chat.completions.create(
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
                starter = inst.get("starter_code", "") or ""
                # Critic eval — wrap each so a single bad subprocess doesn't tank the run
                try:
                    l2_pass, l2_total = check_tests(code, public_tests, starter_code=starter)
                    l2_ok = (l2_pass == l2_total) and l2_total > 0
                    y_pass, y_total = check_tests(code, private_tests[:MAX_PRIVATE_TESTS], starter_code=starter)
                    Y = 1 if (y_pass == y_total) and y_total > 0 else 0
                    l0 = critic_L0_syntax(code)
                    l1 = critic_L1_lint(code)
                except Exception as e:
                    log.warning("[%s] critic eval failed for %s_p%d: %s", gen_key, inst_id, pid, e)
                    continue
                l3 = None
                if not cost.capped:
                    try:
                        l3_pass, l3_cost = critic_L3_review(inst.get("question_content", "")[:3000], code, reviewer_client)
                        cost.record(l3_cost, prompt_tokens=0, completion_tokens=0,
                                    instance_id=inst_id, patch_id=pid,
                                    extra={"kind": "L3_review"})
                        l3 = l3_pass
                    except Exception as e:
                        log.warning("[%s] L3 failed for %s_p%d: %s", gen_key, inst_id, pid, e)
                rec = {
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
                }
                records.append(rec)
                out_fp.write(json.dumps(rec) + "\n")
                out_fp.flush()
                os.fsync(out_fp.fileno())
                done.add((str(inst_id), pid))
                if len(records) % 10 == 0:
                    log.info("[%s] %d records, cost $%.4f", gen_key, len(records), cost.total_usd)
    finally:
        out_fp.close()
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
    p.add_argument("--lcb-version", default="v1", choices=["v1", "all"],
                   help="v1 = original test.jsonl (400 problems); all = union of v1..v6 (1055)")
    p.add_argument("--extend-existing", action="store_true",
                   help="If <output-dir>/sample.json already exists, keep its instance_ids "
                        "as the head of the sampled list, then append new instances "
                        "from the larger pool to reach --n-instances. Resume logic "
                        "in calibrate_one_generator skips already-completed records.")
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
    problems = load_lcb(difficulty=args.difficulty, platform=args.platform,
                        lcb_version=args.lcb_version)
    import random
    random.seed(args.seed)
    random.shuffle(problems)

    sample_path = out_dir / "sample.json"
    if args.extend_existing and sample_path.exists():
        # Preserve continuity: keep the existing sampled IDs as the FRONT of the
        # list (in their original order), then append shuffled new candidates
        # until we reach n_instances.
        existing = json.loads(sample_path.read_text())
        existing_ids = [s["question_id"] for s in existing]
        existing_set = set(existing_ids)
        problems_by_id = {p["question_id"]: p for p in problems}
        head = [problems_by_id[i] for i in existing_ids if i in problems_by_id]
        if len(head) != len(existing_ids):
            missing = [i for i in existing_ids if i not in problems_by_id]
            log.warning("%d existing IDs not found in current pool (%s/%s); "
                        "they may have been dropped between LCB releases. "
                        "Examples: %s", len(missing), args.platform,
                        args.difficulty, missing[:3])
        tail = [p for p in problems if p["question_id"] not in existing_set]
        problems = head + tail
        log.info("extend-existing: %d existing + %d new candidates available "
                 "(target n=%d)", len(head), len(tail), args.n_instances)
    problems = problems[: args.n_instances]
    log.info("sampled %d problems", len(problems))

    # Save sample manifest
    (out_dir / "sample.json").write_text(json.dumps([
        {"question_id": p["question_id"], "platform": p.get("platform"), "difficulty": p.get("difficulty")}
        for p in problems
    ], indent=2))

    # OpenAI client
    from openai import OpenAI
    client = _make_client()

    # Calibrate each generator
    for g in generators:
        records = calibrate_one_generator(
            g, problems, args.n_patches, out_dir, caps.get(g, 3.0), client,
        )
        compute_likelihoods(records, g, out_dir)


if __name__ == "__main__":
    main()
