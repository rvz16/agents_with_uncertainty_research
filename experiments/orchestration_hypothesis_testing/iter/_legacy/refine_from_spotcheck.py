"""Iterative refinement to measure the generator transition kernel
P(Y_{t+1} | Y_t).

Step 0 = the existing spot-check patches (we already have Y_0 from harness).
Steps 1..N-1: regenerate with a refinement prompt that includes the prior
patch + critic feedback, then critic-score and harness-evaluate each step.

Output: <generator>/iter_records.jsonl with full trajectories,
        <generator>/transition_kernel.json with smoothed P(fix|broken),
        P(break|correct) estimates.

Usage:
  python3 iterative_refine_from_spotcheck.py \
    --output-dir data/spot_check_n50 \
    --generators gpt5_mini,qwen3_coder \
    --n-instances 30 --steps 5 \
    --max-cost-usd-per-model 8.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/mnt/data/users/vlad.smirnov/agents_with_uncertainty_research/.claude/worktrees/reverent-vaughan-017bf5/experiments/orchestration_hypothesis_testing")
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
    """Format critic outcomes as a brief feedback string for the model.

    Note: L1_lint is intentionally OMITTED from feedback even though we
    record it. On SWE-bench Lite ruff fires on every single-file patch
    (because imports can't resolve without project context), making L1
    always FAIL. Including it in the prompt confuses the model rather
    than helping. We keep L1 in the records for offline analysis.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    parser.add_argument("--n-instances", type=int, default=30)
    parser.add_argument("--steps", type=int, default=5,
                        help="Total trajectory length (step 0 reused from spot-check)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cost-usd-per-model", type=float, default=8.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

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

    # Load SWE-bench dataset
    import datasets
    ds = datasets.load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    inst_to_row = {row["instance_id"]: row for row in ds}

    # Pick instance subset (deterministic from existing sample)
    sample = json.loads((out_dir / "sample.json").read_text())
    sample_ids = [s["instance_id"] for s in sample][: args.n_instances]
    log.info("running iterative refinement on %d instances", len(sample_ids))

    # Pre-fetch oracles
    oracle_cache = {}
    for inst in sample_ids:
        row = inst_to_row[inst]
        files = scg.get_changed_files_from_patch(row["patch"])
        oracle_cache[inst] = scg.fetch_oracle_files(row["repo"], row["base_commit"], files)
    log.info("oracle cache ready: %d/%d", len(oracle_cache), len(sample_ids))

    # OpenAI/OpenRouter client
    from openai import OpenAI
    api_key = os.environ["OPENROUTER_API_KEY"]
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # Reuse generator config from spot_check_generators.GENERATORS
    for gen in generators:
        if gen not in scg.GENERATORS:
            log.error("unknown generator: %s", gen)
            continue
        model_id, base_url, _ = scg.GENERATORS[gen]
        log.info("=== %s (%s) ===", gen, model_id)

        # Load step-0 critic outcomes
        crit_path = out_dir / gen / "critic_results.jsonl"
        crit_by_key = {}
        if crit_path.exists():
            with open(crit_path) as f:
                for line in f:
                    r = json.loads(line)
                    crit_by_key[(r["instance_id"], r["patch_id"])] = r

        # Use patch_id=0 as step 0
        records: list[dict] = []
        cumulative_cost = 0.0
        for inst in sample_ids:
            key0 = (inst, 0)
            if key0 not in crit_by_key:
                log.warning("no step-0 critic record for %s", inst)
                continue
            step0 = crit_by_key[key0]
            traj = [{
                "step": 0,
                "instance_id": inst,
                "diff": "",  # we'll fill from predictions
                "Y": step0["Y"],
                "L0_syntax": step0.get("L0_syntax"),
                "L1_lint": step0.get("L1_lint"),
                "L3_llm_review": step0.get("L3_llm_review"),
                "cost_usd": 0.0,
            }]
            # Pull step-0 diff from predictions_p0.jsonl
            with open(out_dir / gen / "predictions_p0.jsonl") as f:
                for line in f:
                    r = json.loads(line)
                    if r["instance_id"] == inst:
                        traj[0]["diff"] = r["model_patch"]
                        break

            # Refinement loop
            for t in range(1, args.steps):
                if cumulative_cost >= args.max_cost_usd_per_model:
                    log.warning("[%s] hit cost cap at step %d, instance %s", gen, t, inst)
                    break
                prev = traj[t - 1]
                feedback = _critic_feedback(prev)
                row = inst_to_row[inst]
                # Build instance dict in the same shape spot_check_generators
                # expects, then reuse make_prompt for byte-identical Phase-1
                # prompt formatting (96% extraction rate). Append a refinement
                # feedback section at the end.
                instance = {
                    "repo": row["repo"],
                    "problem_statement": row["problem_statement"],
                    "hints_text": row.get("hints_text", "") or "",
                    "instance_id": inst,
                }
                base_prompt = scg.make_prompt(instance, oracle_cache[inst])
                prompt = base_prompt + REFINEMENT_FEEDBACK_SUFFIX.format(
                    prev_step=t - 1,
                    feedback=feedback,
                )
                try:
                    create_kwargs: dict[str, Any] = {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": args.temperature,
                        "max_tokens": 4000,
                    }
                    if base_url is not None:
                        # Local vLLM — different client; skip for now (we only iterate OpenRouter)
                        log.warning("vLLM endpoint not supported in this script; skipping %s", gen)
                        break
                    resp = client.chat.completions.create(**create_kwargs)
                    text = resp.choices[0].message.content or ""
                    usage = resp.usage
                    # Rough cost: gpt5_mini ~ $0.5/M in, $4/M out; qwen3-coder ~ $0.4/M in, $1.6/M out
                    if "gpt-5-mini" in model_id:
                        cost = (usage.prompt_tokens / 1_000_000) * 0.5 + (usage.completion_tokens / 1_000_000) * 4.0
                    elif "qwen3-coder" in model_id:
                        cost = (usage.prompt_tokens / 1_000_000) * 0.4 + (usage.completion_tokens / 1_000_000) * 1.6
                    else:
                        cost = (usage.prompt_tokens / 1_000_000) * 1.0 + (usage.completion_tokens / 1_000_000) * 5.0
                    cumulative_cost += cost
                except Exception as e:
                    log.warning("[%s] step %d %s: %s", gen, t, inst, e)
                    break
                # Save raw response for diagnostic (always, not only on failure)
                raw_dir = out_dir / gen / "iter_raw_responses"
                raw_dir.mkdir(exist_ok=True)
                (raw_dir / f"{inst}_step{t}.txt").write_text(text)
                # Parse & apply
                blocks = scg.parse_change_blocks(text)
                modified = scg.apply_change_blocks(oracle_cache[inst], blocks) if blocks else {}
                diff = scg.build_diff(oracle_cache[inst], modified) if modified else ""

                # Inline critic scoring so step t+1 gets real feedback (not None).
                # L0/L1 are free; L3 is paid (~$0.001/call).
                from calibration.from_spotcheck import (
                    _modified_file_contents, critic_L0_syntax,
                    critic_L1_lint, critic_L3_llm_review,
                )
                l0 = l1 = l3 = None
                if diff.strip():
                    mod_files = _modified_file_contents(diff, oracle_cache[inst])
                    if mod_files is not None:
                        l0 = critic_L0_syntax(mod_files)
                        l1 = critic_L1_lint(mod_files)
                    else:
                        l0 = False
                        l1 = False
                    if cumulative_cost < args.max_cost_usd_per_model:
                        l3_pass, l3_cost_inc = critic_L3_llm_review(
                            inst, row["problem_statement"], diff, client,
                        )
                        l3 = l3_pass
                        cumulative_cost += l3_cost_inc
                else:
                    l0 = False
                    l1 = False
                    l3 = False

                traj.append({
                    "step": t,
                    "instance_id": inst,
                    "diff": diff,
                    "Y": None,  # will fill from harness eval
                    "L0_syntax": l0,
                    "L1_lint": l1,
                    "L3_llm_review": l3,
                    "cost_usd": cost,
                })
                log.info("[%s] %s step %d: diff=%d chars L0=%s L1=%s L3=%s cum=$%.4f",
                         gen, inst, t, len(diff), l0, l1, l3, cumulative_cost)
            for r in traj:
                records.append(r)

        # Save
        out_path = out_dir / gen / "iter_records.jsonl"
        out_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        log.info("[%s] wrote %d records to %s, cost $%.4f",
                 gen, len(records), out_path, cumulative_cost)


if __name__ == "__main__":
    main()
