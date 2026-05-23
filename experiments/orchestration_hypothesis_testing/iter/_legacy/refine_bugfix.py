"""Iterative refinement trajectories for bug-fixing Table 4 rows.

Reads step-0 patches from `bugfix_calibrate.py` output and rolls them forward
for a fixed number of refinement steps, recording `(instance, step, L0, L2,
L3, Y)` tuples compatible with:
  - `compute_iter_replay_baselines.py`
  - `lcb_compare.py --kernel-file ...`

Example:
  python3 scripts/iter_refine_bugfix.py \\
    --benchmark humanevalfix \\
    --src-dir data/humanevalfix_calibration \\
    --output-dir data/humanevalfix_iter \\
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45,qwen25_7b,qwen25_32b,gpt_oss_20b \\
    --steps 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Package root (parents[1]) on sys.path so imports like `from calibration.X import Y`,
# `from iter.X import Y`, etc. resolve to the new refactored layout.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bugfix_table4_common import (  # noqa: E402
    build_initial_prompt,
    evaluate_candidate,
    extract_code,
    get_failure_output,
    get_initial_source,
    safe_stem,
)
from _common.cost import CostTracker  # noqa: E402
from calibration.lcb import (  # noqa: E402
    GENERATORS,
    OPENROUTER_KEY_NAMES,
    _make_client,
    canonical_generator_key,
    cost_for_call,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("iter_refine_bugfix")

DEFAULT_GENERATORS = ",".join([
    "gpt5_mini",
    "qwen3_coder",
    "haiku45",
    "sonnet45",
    "qwen25_7b",
    "qwen25_32b",
    "gpt_oss_20b",
])


def load_env_chain() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        load_dotenv = None

    for env_path in [
        ROOT / ".env",
        ROOT.parent / ".env",
        ROOT.parent.parent / ".env",
        ROOT.parent.parent.parent / ".env",
        ROOT.parent.parent.parent.parent / ".env",
    ]:
        if env_path.exists() and env_path.stat().st_size > 0:
            if load_dotenv is not None:
                load_dotenv(env_path, override=False)
                continue
            for raw_line in env_path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key or key in os.environ:
                    continue
                if value and value[0] in {"'", '"'}:
                    try:
                        parsed = shlex.split(f"dummy={value}", posix=True)
                    except ValueError:
                        parsed = [f"dummy={value.strip('\"')}"]
                    value = parsed[0].split("=", 1)[1] if parsed else value
                else:
                    value = value.split(" #", 1)[0].strip()
                os.environ[key] = value


def validate_provider_env(generators: list[str]) -> None:
    needs_openrouter = any(GENERATORS[gen][2] is None for gen in generators)
    if needs_openrouter and not any(os.environ.get(name, "").strip() for name in OPENROUTER_KEY_NAMES):
        raise SystemExit(
            "OpenRouter API key is not set. Expected one of: "
            "OPENROUTER_API_KEY, OPEN_ROUTER_API_KEY, OPEN_ROUTER."
        )


REFINEMENT_SUFFIX = """

Additional diagnostic feedback from your previous attempt:
- L0 syntax critic: {l0_status}
- L2 early-tests critic: {l2_status}
- L3 mid-tests critic: {l3_status}
- Full oracle label on the previous attempt: {y_status}

Oracle excerpt from the previous attempt:
```
{oracle_detail}
```

Produce a NEW complete corrected solution. Keep the same interface. Output only code.
"""


def parse_cap_map(raw: str, generators: list[str]) -> dict[str, float]:
    if "=" not in raw:
        flat = float(raw)
        return {gen: flat for gen in generators}
    out: dict[str, float] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, val = pair.split("=", 1)
        out[canonical_generator_key(key)] = float(val)
    for gen in generators:
        out.setdefault(gen, 5.0)
    return out


def load_sample_ids(src_dir: Path) -> list[str] | None:
    sample_path = src_dir / "sample.json"
    if not sample_path.exists():
        return None
    raw = json.loads(sample_path.read_text())
    if raw and isinstance(raw[0], dict):
        return [str(row["instance_id"]) for row in raw]
    return [str(x) for x in raw]


def load_step0_records(gen_src: Path) -> dict[str, dict]:
    rec_path = gen_src / "critic_results.jsonl"
    out: dict[str, dict] = {}
    if not rec_path.exists():
        return out
    for line in open(rec_path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if int(rec.get("patch_id", -1)) == 0:
            out[str(rec["instance_id"])] = rec
    return out


def load_completed_trajectories(path: Path, steps: int) -> tuple[dict[str, list[dict]], set[str]]:
    by_inst: dict[str, list[dict]] = {}
    if path.exists():
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_inst.setdefault(str(rec["instance_id"]), []).append(rec)
    for inst_id in by_inst:
        by_inst[inst_id].sort(key=lambda r: int(r["step"]))
    completed = {
        inst_id for inst_id, rows in by_inst.items()
        if rows and max(int(r["step"]) for r in rows) >= steps - 1
    }
    return by_inst, completed


def compute_transition_kernel(records: list[dict], gen_key: str, benchmark: str) -> dict:
    by_inst: dict[str, list[dict]] = {}
    for rec in records:
        by_inst.setdefault(str(rec["instance_id"]), []).append(rec)
    for inst_id in by_inst:
        by_inst[inst_id].sort(key=lambda r: int(r["step"]))

    counts = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
    for rows in by_inst.values():
        for idx in range(len(rows) - 1):
            y0 = rows[idx].get("Y")
            y1 = rows[idx + 1].get("Y")
            if y0 is None or y1 is None:
                continue
            counts[f"{int(y0)}->{int(y1)}"] += 1

    n_broken = counts["0->0"] + counts["0->1"]
    n_correct = counts["1->0"] + counts["1->1"]
    return {
        "generator": gen_key,
        "benchmark": benchmark,
        "source": "iter_refine_bugfix",
        "kernel_all": {
            "P_fix_given_broken": (counts["0->1"] + 1) / (n_broken + 2) if n_broken or counts["0->1"] else 0.5,
            "P_break_given_correct": (counts["1->0"] + 1) / (n_correct + 2) if n_correct or counts["1->0"] else 0.5,
            "raw_counts": counts,
            "n_pairs": n_broken + n_correct,
            "smoothing": "Beta(1,1)",
        },
    }


def run_one_instance(
    *,
    benchmark: str,
    task_id: str,
    model_id: str,
    client,
    initial_source: str,
    step0_code: str,
    step0_record: dict,
    steps: int,
    temperature: float,
    max_tokens: int,
    raw_dir: Path,
    cost: CostTracker,
) -> list[dict]:
    stem = safe_stem(task_id)
    traj = [{
        "benchmark": benchmark,
        "instance_id": task_id,
        "step": 0,
        "Y": step0_record.get("Y"),
        "L0_syntax": step0_record.get("L0_syntax"),
        "L1_lint": step0_record.get("L1_lint"),
        "L2_public_tests": step0_record.get("L2_public_tests"),
        "L3_llm_review": step0_record.get("L3_llm_review"),
        "oracle_detail": step0_record.get("oracle_detail", ""),
        "code_chars": len(step0_code),
        "code": step0_code,
        "step_cost_usd": 0.0,
    }]

    current_code = step0_code
    for step in range(1, steps):
        if cost.capped:
            break
        prev = traj[-1]
        test_output = get_failure_output(benchmark, task_id, current_code)
        prompt = build_initial_prompt(benchmark, task_id, current_code, test_output)
        prompt += REFINEMENT_SUFFIX.format(
            l0_status="PASS" if prev.get("L0_syntax") else "FAIL",
            l2_status="PASS" if prev.get("L2_public_tests") else "FAIL",
            l3_status="PASS" if prev.get("L3_llm_review") else "FAIL",
            y_status="PASS" if prev.get("Y") else "FAIL",
            oracle_detail=(prev.get("oracle_detail") or "")[:1200],
        )
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            call_cost = cost_for_call(
                model_id,
                usage.prompt_tokens,
                usage.completion_tokens,
            )
            cost.record(
                call_cost,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                instance_id=task_id,
                patch_id=step,
                extra={"kind": "iter_refine"},
            )
        except Exception as exc:
            log.warning("[%s/%s] refinement failed at step %d: %s", benchmark, task_id, step, exc)
            break

        (raw_dir / f"{stem}_step{step}.txt").write_text(text)
        current_code = extract_code(text, current_code or initial_source)
        eval_rec = evaluate_candidate(benchmark, task_id, current_code)
        traj.append({
            "benchmark": benchmark,
            "instance_id": task_id,
            "step": step,
            "Y": eval_rec["Y"],
            "L0_syntax": eval_rec["L0_syntax"],
            "L1_lint": eval_rec["L1_lint"],
            "L2_public_tests": eval_rec["L2_public_tests"],
            "L3_llm_review": eval_rec["L3_llm_review"],
            "oracle_detail": eval_rec["oracle_detail"],
            "code_chars": len(current_code),
            "code": current_code,
            "step_cost_usd": call_cost,
        })
    return traj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=["humanevalfix", "codecontests"])
    parser.add_argument("--src-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", default=DEFAULT_GENERATORS)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--n-instances", type=int, default=0, help="0 = use all step-0 instances")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-cost-usd-per-model", default="5.0")
    args = parser.parse_args()

    src_dir = args.src_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    generators = [canonical_generator_key(g) for g in args.generators.split(",") if g.strip()]
    load_env_chain()
    validate_provider_env(generators)
    caps = parse_cap_map(args.max_cost_usd_per_model, generators)
    sample_ids = load_sample_ids(src_dir)

    for gen_key in generators:
        model_id, _label, _base_url = GENERATORS[gen_key]
        client = _make_client(gen_key)

        gen_src = src_dir / gen_key
        raw_src = gen_src / "raw_responses"
        step0_records = load_step0_records(gen_src)
        if not step0_records or not raw_src.exists():
            log.warning("[%s/%s] missing step-0 data, skipping", args.benchmark, gen_key)
            continue

        ordered_ids = sample_ids or list(step0_records.keys())
        candidate_ids = [task_id for task_id in ordered_ids if task_id in step0_records]
        if args.n_instances > 0:
            candidate_ids = candidate_ids[:args.n_instances]

        gen_out = out_dir / gen_key
        gen_out.mkdir(parents=True, exist_ok=True)
        raw_dir = gen_out / "iter_raw_responses"
        raw_dir.mkdir(exist_ok=True)
        iter_path = gen_out / "iter_records.jsonl"
        existing_by_inst, completed = load_completed_trajectories(iter_path, args.steps)
        cost = CostTracker(name=gen_key, cap_usd=caps[gen_key], log_path=gen_out / "cost_log.jsonl")

        all_records = []
        for task_id, rows in existing_by_inst.items():
            if task_id in completed:
                all_records.extend(rows)

        for idx, task_id in enumerate(candidate_ids, 1):
            if cost.capped:
                log.warning("[%s/%s] cost cap reached", args.benchmark, gen_key)
                break
            if task_id in completed:
                continue
            step0 = step0_records[task_id]
            initial_source = get_initial_source(args.benchmark, task_id)
            raw_path = raw_src / f"{safe_stem(task_id)}_p0.txt"
            if raw_path.exists():
                step0_code = extract_code(raw_path.read_text(), initial_source)
            else:
                step0_code = initial_source
            traj = run_one_instance(
                benchmark=args.benchmark,
                task_id=task_id,
                model_id=model_id,
                client=client,
                initial_source=initial_source,
                step0_code=step0_code,
                step0_record=step0,
                steps=args.steps,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                raw_dir=raw_dir,
                cost=cost,
            )
            all_records.extend(traj)
            if idx % 10 == 0:
                log.info("[%s/%s] %d/%d instances processed",
                         args.benchmark, gen_key, idx, len(candidate_ids))

        all_records.sort(key=lambda r: (str(r["instance_id"]), int(r["step"])))
        with open(iter_path, "w") as fp:
            for rec in all_records:
                fp.write(json.dumps(rec) + "\n")
        kernel = compute_transition_kernel(all_records, gen_key, args.benchmark)
        (gen_out / "transition_kernel.json").write_text(json.dumps(kernel, indent=2))
        (gen_out / "cost_summary.json").write_text(json.dumps(cost.snapshot(), indent=2))
        log.info("[%s/%s] wrote %d iter records", args.benchmark, gen_key, len(all_records))


if __name__ == "__main__":
    main()
