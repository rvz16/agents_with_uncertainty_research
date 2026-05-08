"""Iterative refinement for LiveCodeBench.

Reuses lcb_calibrate.py helpers (build_prompt, extract_code, run_inline_test_inputs,
critic_L0_syntax, critic_L1_lint, critic_L3_review). Each instance's full
N-step trajectory runs in its own ThreadPool worker (steps within instance
must be sequential — step k uses step k-1 critic feedback). Multiple
instances run in parallel.

Step 0 = the existing independent-sample patch (patch_id=0) from
<src-dir>/<gen>/raw_responses/<inst>_p0.txt. Steps 1..N-1 refined with
critic feedback.

Output:
  <out-dir>/<gen>/iter_records.jsonl     — per-step trajectory rows
  <out-dir>/<gen>/iter_raw_responses/    — raw model output per step
  <out-dir>/<gen>/transition_kernel.json — measured (Y_t, Y_{t+1}) kernel

Usage:
  python3 iter_refine_lcb.py \\
    --src-dir data/lcb_calibration_v2 \\
    --output-dir data/lcb_calibration_v2_iter \\
    --generators gpt5_mini \\
    --difficulty hard --platform leetcode \\
    --steps 5 --max-workers 6 --max-cost-usd-per-model 1.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1] if "scripts" in str(Path(__file__).resolve()) else Path("/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/experiments/orchestration_hypothesis_testing")
sys.path.insert(0, str(ROOT / "scripts"))

from lcb_calibrate import (  # noqa: E402
    GENERATORS, build_prompt, extract_code,
    critic_L0_syntax, critic_L1_lint, critic_L3_review,
    cost_for_call, check_tests, decode_private_tests,
    load_lcb, MAX_PRIVATE_TESTS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("iter_refine_lcb")


REFINEMENT_FEEDBACK = """\

## Refinement feedback from your previous attempt (step {prev_step})

Your previous patch was evaluated. The diagnostic critics reported:

{feedback}

Public tests {pt_status} ({pt_pass}/{pt_total} passed).

If a critic FAILED or public tests failed, your previous attempt likely
had the corresponding issue. If everything PASSED but you're being asked
to refine, the patch was syntactically clean and passed visible tests
but did not handle hidden edge cases — reconsider corner cases, large
inputs, empty inputs, etc.

Generate a NEW solution (in the same `class Solution:` format) that
addresses the diagnostic feedback.

Your previous code was:
```python
{prev_code}
```
"""


def _critic_feedback(critics: dict[str, Any]) -> str:
    lines = []
    for k in ("L0_syntax", "L3_llm_review"):
        v = critics.get(k)
        if v is True:
            lines.append(f"- {k}: PASS")
        elif v is False:
            lines.append(f"- {k}: FAIL")
        else:
            lines.append(f"- {k}: skipped")
    return "\n".join(lines) if lines else "- no diagnostics available"


def run_one_instance(
    inst: dict, step0_code: str, step0_record: dict,
    model_id: str, steps: int, temperature: float, client,
    gen_name: str, raw_dir: Path, cost_lock: threading.Lock,
    cost_counter: dict, cap_usd: float,
) -> list[dict]:
    """Run one instance's N-step trajectory. Returns trajectory rows."""
    inst_id = str(inst["question_id"])
    public_tests = inst.get("public_test_cases") or []
    if isinstance(public_tests, str):
        try:
            public_tests = json.loads(public_tests)
        except Exception:
            public_tests = []
    private_tests = decode_private_tests(inst.get("private_test_cases", "") or "")
    starter = inst.get("starter_code", "") or ""
    base_prompt = build_prompt(inst)

    traj = [{
        "step": 0, "instance_id": inst_id,
        "code_chars": len(step0_code),
        "Y": step0_record.get("Y"),
        "L0_syntax": step0_record.get("L0_syntax"),
        "L1_lint": step0_record.get("L1_lint"),
        "L2_public_tests": step0_record.get("L2_public_tests"),
        "L3_llm_review": step0_record.get("L3_llm_review"),
        "cost_usd": 0.0,
    }]

    prev_code = step0_code
    prev_l2_pass, prev_l2_total = check_tests(prev_code, public_tests, starter_code=starter)
    for t in range(1, steps):
        with cost_lock:
            if cost_counter["v"] >= cap_usd:
                log.warning("[%s/%s] cap hit at step %d", gen_name, inst_id, t)
                break

        prev = traj[t - 1]
        prompt = base_prompt + REFINEMENT_FEEDBACK.format(
            prev_step=t - 1,
            feedback=_critic_feedback(prev),
            pt_status="PASS" if prev_l2_pass == prev_l2_total and prev_l2_total > 0 else "FAIL",
            pt_pass=prev_l2_pass, pt_total=prev_l2_total,
            prev_code=prev_code[:3000],
        )
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=4000,
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            cost = cost_for_call(model_id, usage.prompt_tokens, usage.completion_tokens)
        except Exception as e:
            log.warning("[%s/%s] step %d gen failed: %s", gen_name, inst_id, t, e)
            break

        with cost_lock:
            cost_counter["v"] += cost
        (raw_dir / f"{inst_id}_step{t}.txt").write_text(text)

        code = extract_code(text)
        # Inline critic evaluation
        try:
            l2_pass, l2_total = check_tests(code, public_tests, starter_code=starter)
            l2_ok = (l2_pass == l2_total) and l2_total > 0
            y_pass, y_total = check_tests(code, private_tests[:MAX_PRIVATE_TESTS], starter_code=starter)
            Y = 1 if (y_pass == y_total) and y_total > 0 else 0
            l0 = critic_L0_syntax(code)
            l1 = critic_L1_lint(code)
        except Exception as e:
            log.warning("[%s/%s] step %d critic failed: %s", gen_name, inst_id, t, e)
            break

        l3 = None
        with cost_lock:
            cap_ok = cost_counter["v"] < cap_usd
        if cap_ok:
            try:
                l3_pass, l3_cost = critic_L3_review(
                    inst.get("question_content", "")[:3000], code, client,
                )
                l3 = bool(l3_pass)
                with cost_lock:
                    cost_counter["v"] += l3_cost
            except Exception as e:
                log.warning("[%s/%s] step %d L3 failed: %s", gen_name, inst_id, t, e)

        traj.append({
            "step": t, "instance_id": inst_id,
            "code_chars": len(code),
            "Y": int(Y),
            "L0_syntax": bool(l0), "L1_lint": bool(l1),
            "L2_public_tests": bool(l2_ok),
            "L3_llm_review": l3,
            "cost_usd": cost,
        })
        log.info("[%s/%s] step %d: Y=%d L0=%s L2=%s L3=%s",
                 gen_name, inst_id, t, Y, l0, l2_ok, l3)

        prev_code = code
        prev_l2_pass, prev_l2_total = l2_pass, l2_total

    return traj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--platform", default="leetcode")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--n-instances", type=int, default=0,
                        help="0 = all available")
    parser.add_argument("--max-cost-usd-per-model", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args()

    src_dir = args.src_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", "/mnt/data/users/vlad.smirnov/hf_cache")
    problems = load_lcb(difficulty=args.difficulty, platform=args.platform)
    by_qid = {str(p["question_id"]): p for p in problems}
    log.info("loaded %d %s/%s problems", len(problems), args.difficulty, args.platform)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY not set"); sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    generators = [g.strip() for g in args.generators.split(",") if g.strip()]
    for gen in generators:
        if gen not in GENERATORS:
            log.error("unknown generator: %s", gen); continue
        model_id, _ = GENERATORS[gen]
        log.info("=== %s (%s) ===", gen, model_id)

        # Step 0 from src critic_results + raw_responses
        gen_src = src_dir / gen
        crit_path = gen_src / "critic_results.jsonl"
        raw_src = gen_src / "raw_responses"
        if not crit_path.exists() or not raw_src.exists():
            log.warning("[%s] missing src data, skipping", gen); continue

        step0_records: dict[str, dict] = {}
        for line in open(crit_path):
            if not line.strip(): continue
            r = json.loads(line)
            if r.get("patch_id") == 0:
                step0_records[str(r["instance_id"])] = r

        # Pick instances: must have step-0 record + raw response + dataset row
        candidates = []
        for inst_id, rec in step0_records.items():
            raw_path = raw_src / f"{inst_id}_p0.txt"
            if not raw_path.exists() or inst_id not in by_qid:
                continue
            candidates.append((inst_id, rec, raw_path))
        if args.n_instances and args.n_instances < len(candidates):
            candidates = candidates[:args.n_instances]
        log.info("[%s] %d eligible instances", gen, len(candidates))

        gen_out = out_dir / gen
        gen_out.mkdir(parents=True, exist_ok=True)
        raw_dir = gen_out / "iter_raw_responses"
        raw_dir.mkdir(exist_ok=True)

        cost_lock = threading.Lock()
        cost_counter = {"v": 0.0}

        all_traj: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {}
            for inst_id, rec, raw_path in candidates:
                step0_text = raw_path.read_text()
                step0_code = extract_code(step0_text)
                problem = by_qid[inst_id]
                fut = ex.submit(
                    run_one_instance,
                    problem, step0_code, rec, model_id, args.steps,
                    args.temperature, client, gen, raw_dir,
                    cost_lock, cost_counter, args.max_cost_usd_per_model,
                )
                futures[fut] = inst_id
            for fut in as_completed(futures):
                inst_id = futures[fut]
                try:
                    all_traj.extend(fut.result())
                except Exception as e:
                    log.error("[%s/%s] worker crashed: %s", gen, inst_id, e)

        out_path = gen_out / "iter_records.jsonl"
        with open(out_path, "w") as f:
            for r in all_traj:
                f.write(json.dumps(r) + "\n")
        log.info("[%s] wrote %d records to %s, total cost $%.4f",
                 gen, len(all_traj), out_path, cost_counter["v"])

        # Compute transition kernel
        by_inst: dict[str, list[dict]] = {}
        for r in all_traj:
            by_inst.setdefault(r["instance_id"], []).append(r)
        for inst in by_inst:
            by_inst[inst].sort(key=lambda r: r["step"])
        counts = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
        for inst, traj in by_inst.items():
            for i in range(len(traj) - 1):
                yt, yt1 = traj[i].get("Y"), traj[i + 1].get("Y")
                if yt is None or yt1 is None: continue
                counts[f"{yt}->{yt1}"] += 1
        n_broken = counts["0->0"] + counts["0->1"]
        n_correct = counts["1->0"] + counts["1->1"]
        if n_broken + n_correct > 0:
            P_fix = (counts["0->1"] + 1) / (n_broken + 2)
            P_break = (counts["1->0"] + 1) / (n_correct + 2)
            kernel = {
                "generator": gen, "source": "iterative_refinement_with_feedback",
                "kernel_all": {
                    "P_fix_given_broken": P_fix,
                    "P_break_given_correct": P_break,
                    "raw_counts": counts,
                    "n_pairs": n_broken + n_correct,
                    "smoothing": "Beta(1,1)",
                },
            }
            (gen_out / "transition_kernel.json").write_text(json.dumps(kernel, indent=2))
            log.info("[%s] kernel: P_fix=%.3f, P_break=%.3f (n=%d pairs)",
                     gen, P_fix, P_break, n_broken + n_correct)


if __name__ == "__main__":
    main()
