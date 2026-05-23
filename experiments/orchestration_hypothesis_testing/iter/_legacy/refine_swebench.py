"""Iterative refinement for SWE-bench Lite / Verified / Pro.

Reuses spot_check_generators helpers (make_prompt, parse_change_blocks,
apply_change_blocks, build_diff). Runs N-step trajectory per instance with
critic feedback as refinement context.

Parallelism design:
  - Each instance's full N-step trajectory runs in its own worker
    (steps within an instance must be sequential — step k uses step k-1's
    critic feedback).
  - Multiple instances run in parallel via ThreadPoolExecutor(--max-workers).
  - Multiple generators launch as separate detached processes (one per
    `--generators gen1` invocation) for true cross-generator parallelism.

Output per (output_dir, generator):
  iter_raw_responses/<inst>_step<t>.txt  — raw model output
  iter_records.jsonl                     — per-step trajectory rows
  predictions_iter_step<t>.jsonl         — per-step harness-input format

Usage (one generator, parallel within):
  python3 iter_refine_swebench.py \\
    --dataset princeton-nlp/SWE-bench_Verified \\
    --src-dir data/swebench_verified_n30 \\
    --output-dir data/swebench_verified_iter \\
    --generators gpt5_mini \\
    --n-instances 30 --steps 5 --max-workers 6 \\
    --max-cost-usd-per-model 3.0
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
# Package root (parents[1]) on sys.path so imports like `from calibration.X import Y`,
# `from iter.X import Y`, etc. resolve to the new refactored layout.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import spot_check_generators as scg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("iter_refine")


REFINEMENT_FEEDBACK_SUFFIX = """\

## Refinement feedback from your previous attempt (step {prev_step})

Your previous attempt was evaluated. The diagnostic critics reported:

{feedback}

If a critic FAILED, your previous attempt likely had the corresponding issue.
If all critics PASSED but you're being asked to refine, the patch was
syntactically clean but did not actually fix the underlying bug — reconsider
the semantic meaning of the issue and try a different approach.

Generate a NEW patch (in the same SEARCH/REPLACE format described above) that
addresses the diagnostic feedback.
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


def _cost_for(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    if "gpt-5-mini" in model_id:
        return (prompt_tokens / 1_000_000) * 0.5 + (completion_tokens / 1_000_000) * 4.0
    if "qwen3-coder" in model_id:
        return (prompt_tokens / 1_000_000) * 0.4 + (completion_tokens / 1_000_000) * 1.6
    if "claude-sonnet" in model_id:
        return (prompt_tokens / 1_000_000) * 3.0 + (completion_tokens / 1_000_000) * 15.0
    if "claude-haiku" in model_id:
        return (prompt_tokens / 1_000_000) * 1.0 + (completion_tokens / 1_000_000) * 5.0
    return (prompt_tokens / 1_000_000) * 1.0 + (completion_tokens / 1_000_000) * 5.0


def run_one_instance(
    inst: str, row: dict, oracle: dict[str, str],
    step0_record: dict, step0_diff: str, model_id: str, steps: int,
    temperature: float, client, gen_name: str, raw_dir: Path,
    cost_lock: threading.Lock, cost_counter: dict, cap_usd: float,
) -> list[dict]:
    """Run one instance's N-step refinement trajectory. Returns trajectory rows."""
    traj = [{
        "step": 0, "instance_id": inst, "diff": step0_diff,
        "Y": step0_record.get("Y"),
        "L0_syntax": step0_record.get("L0_syntax"),
        "L1_lint": step0_record.get("L1_lint"),
        "L3_llm_review": step0_record.get("L3_llm_review"),
        "cost_usd": 0.0,
    }]

    instance = {
        "repo": row["repo"],
        "problem_statement": row["problem_statement"],
        "hints_text": row.get("hints_text", "") or "",
        "instance_id": inst,
    }
    base_prompt = scg.make_prompt(instance, oracle)

    for t in range(1, steps):
        with cost_lock:
            if cost_counter["v"] >= cap_usd:
                log.warning("[%s/%s] cap hit at step %d", gen_name, inst, t)
                break
        prev = traj[t - 1]
        prompt = base_prompt + REFINEMENT_FEEDBACK_SUFFIX.format(
            prev_step=t - 1, feedback=_critic_feedback(prev),
        )
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=4000,
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            cost = _cost_for(model_id, usage.prompt_tokens, usage.completion_tokens)
        except Exception as e:
            log.warning("[%s/%s] step %d gen failed: %s", gen_name, inst, t, e)
            break

        with cost_lock:
            cost_counter["v"] += cost

        (raw_dir / f"{inst}_step{t}.txt").write_text(text)

        blocks = scg.parse_change_blocks(text)
        modified = scg.apply_change_blocks(oracle, blocks) if blocks else {}
        diff = scg.build_diff(oracle, modified) if modified else ""

        from calibration.from_spotcheck import (
            _modified_file_contents, critic_L0_syntax,
            critic_L1_lint, critic_L3_llm_review,
        )
        l0 = l1 = l3 = None
        if diff.strip():
            mod_files = _modified_file_contents(diff, oracle)
            if mod_files is not None:
                l0 = critic_L0_syntax(mod_files)
                l1 = critic_L1_lint(mod_files)
            else:
                l0 = l1 = False
            with cost_lock:
                cap_ok = cost_counter["v"] < cap_usd
            if cap_ok:
                try:
                    l3_pass, l3_cost = critic_L3_llm_review(
                        inst, row["problem_statement"], diff, client,
                    )
                    l3 = l3_pass
                    with cost_lock:
                        cost_counter["v"] += l3_cost
                except Exception as e:
                    log.warning("[%s/%s] step %d L3 failed: %s", gen_name, inst, t, e)
        else:
            l0 = l1 = l3 = False

        traj.append({
            "step": t, "instance_id": inst, "diff": diff, "Y": None,
            "L0_syntax": l0, "L1_lint": l1, "L3_llm_review": l3,
            "cost_usd": cost,
        })
        log.info("[%s/%s] step %d: diff=%d L0=%s L3=%s",
                 gen_name, inst, t, len(diff), l0, l3)

    return traj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--src-dir", required=True, type=Path,
                        help="Source dir containing <gen>/predictions_p0.jsonl + critic_results.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Where to write iter_records.jsonl + iter_raw_responses/")
    parser.add_argument("--generators", required=True)
    parser.add_argument("--n-instances", type=int, default=30)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cost-usd-per-model", type=float, default=8.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-workers", type=int, default=6,
                        help="Parallel instance trajectories within one generator")
    args = parser.parse_args()

    src_dir = args.src_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", "/mnt/data/users/vlad.smirnov/hf_cache")
    import datasets
    ds = datasets.load_dataset(args.dataset, split="test")
    inst_to_row = {row["instance_id"]: row for row in ds}
    log.info("loaded %d instances from %s", len(inst_to_row), args.dataset)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY not set")
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    generators = [g.strip() for g in args.generators.split(",") if g.strip()]
    for gen in generators:
        if gen not in scg.GENERATORS:
            log.error("unknown generator: %s", gen); continue
        model_id, base_url, _ = scg.GENERATORS[gen]
        if base_url is not None:
            log.warning("[%s] vLLM endpoint not supported here; skipping", gen)
            continue
        log.info("=== %s (%s) ===", gen, model_id)

        # Step-0 critics + diff per instance from src_dir
        gen_src = src_dir / gen
        crit_path = gen_src / "critic_results.jsonl"
        crit_by_key: dict[tuple, dict] = {}
        if crit_path.exists():
            for line in open(crit_path):
                if not line.strip(): continue
                r = json.loads(line)
                crit_by_key[(r["instance_id"], r["patch_id"])] = r
        diff_by_inst: dict[str, str] = {}
        pred_path = gen_src / "predictions_p0.jsonl"
        if pred_path.exists():
            for line in open(pred_path):
                if not line.strip(): continue
                r = json.loads(line)
                diff_by_inst[r["instance_id"]] = r.get("model_patch", "") or ""

        # Pick instances that have step-0 records and matched dataset rows
        candidate = [k[0] for k in crit_by_key if k[1] == 0]
        candidate = [c for c in candidate if c in inst_to_row and c in diff_by_inst]
        candidate = candidate[: args.n_instances]
        if not candidate:
            log.warning("[%s] no eligible instances", gen); continue
        log.info("[%s] %d eligible instances", gen, len(candidate))

        # Pre-fetch oracles
        oracle_cache: dict[str, dict] = {}
        for inst in candidate:
            row = inst_to_row[inst]
            files = scg.get_changed_files_from_patch(row["patch"])
            oracle_cache[inst] = scg.fetch_oracle_files(row["repo"], row["base_commit"], files)
        log.info("[%s] oracle cache ready: %d/%d", gen, len(oracle_cache), len(candidate))

        gen_out = out_dir / gen
        gen_out.mkdir(parents=True, exist_ok=True)
        raw_dir = gen_out / "iter_raw_responses"
        raw_dir.mkdir(exist_ok=True)

        cost_lock = threading.Lock()
        cost_counter = {"v": 0.0}

        # Parallel: one worker per instance, full trajectory per worker.
        # Steps within an instance are sequential (each step uses prev critic
        # feedback); instances are independent so we parallelize across them.
        all_traj: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {}
            for inst in candidate:
                step0_record = crit_by_key.get((inst, 0), {})
                step0_diff = diff_by_inst.get(inst, "")
                fut = ex.submit(
                    run_one_instance,
                    inst, inst_to_row[inst], oracle_cache[inst],
                    step0_record, step0_diff, model_id, args.steps,
                    args.temperature, client, gen, raw_dir,
                    cost_lock, cost_counter, args.max_cost_usd_per_model,
                )
                futures[fut] = inst
            for fut in as_completed(futures):
                inst = futures[fut]
                try:
                    traj = fut.result()
                    all_traj.extend(traj)
                except Exception as e:
                    log.error("[%s/%s] worker crashed: %s", gen, inst, e)

        out_path = gen_out / "iter_records.jsonl"
        with open(out_path, "w") as f:
            for r in all_traj:
                f.write(json.dumps(r) + "\n")
        log.info("[%s] wrote %d records to %s, total cost $%.4f",
                 gen, len(all_traj), out_path, cost_counter["v"])

        # Per-step prediction files for harness eval
        max_step = max(r["step"] for r in all_traj) if all_traj else 0
        for t in range(1, max_step + 1):
            preds_path = gen_out / f"predictions_iter_step{t}.jsonl"
            with open(preds_path, "w") as f:
                for r in all_traj:
                    if r["step"] != t:
                        continue
                    f.write(json.dumps({
                        "instance_id": r["instance_id"],
                        "model_name_or_path": f"{gen}_iter_step{t}",
                        "model_patch": r["diff"],
                    }) + "\n")
            log.info("[%s] wrote %s", gen, preds_path.name)


if __name__ == "__main__":
    main()
