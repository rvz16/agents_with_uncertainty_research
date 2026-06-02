"""Real Self-Refine + Reflexion implementations for SWE-bench cells.

Companion to iter_refine_real_baselines.py (LCB version). Mirrors
iter_refine_swebench.py's flow but replaces the structured-critic feedback
with model-driven critique (Self-Refine) or reflection (Reflexion).

Method semantics:
  Self-Refine: model generates self-critique → if "CRITIQUE_OK", stop;
               otherwise refine. No external test signal.
  Reflexion (SWE approximation): uses L0_syntax AND L3_llm_review from
               previous step as approximate "external test pass" signal.
               Documented as approximation in paper since true Reflexion
               would require harness-in-loop (cost-prohibitive on SWE).
               If approximation says pass, stop. Else write reflection
               and refine using accumulated reflections.

LOGGING SPEC (designed so re-runs are never needed):
  Output: <output-dir>/<gen>/<method>/
    - iter_records.jsonl       per-step trajectory (Y null until harness)
    - raw_calls/<inst>_step<k>_<purpose>.json    full prompts + responses
    - cost_log.jsonl           per-API-call audit
    - cost_summary.json
    - predictions_iter_step{1..N}.jsonl   harness-input per step
    - RUN_CONFIG.json
  After harness eval (separate step):
    - iter_records.jsonl populated with Y per step
    - transition_kernel.json
    - policy_comparison.json

Usage:
  python3 scripts/iter_refine_real_baselines_swe.py \\
      --method selfrefine --dataset princeton-nlp/SWE-bench_Verified \\
      --src-dir data/swebench_verified_n30 \\
      --output-dir data/swebench_verified_realbaselines \\
      --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \\
      --n-instances 30 --steps 5 \\
      --max-cost-usd-per-model gpt5_mini=5,qwen3_coder=2,haiku45=4,sonnet45=12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Package root (parents[1]) on sys.path so imports like `from calibration.X import Y`,
# `from iter.X import Y`, etc. resolve to the new refactored layout.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import spot_check_generators as scg  # noqa: E402
from _common.logprobs import (  # noqa: E402
    extract_completion_logprobs,
    logprob_summary_fields,
    make_chat_completion_kwargs,
)

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("real_baselines_swe")


# ============================================================================
# Method-specific prompt templates
# ============================================================================

SELFREFINE_CRITIQUE_PROMPT = """\
You are reviewing a code patch you previously generated for a SWE-bench
issue. Carefully examine the patch and identify any problems.

## Issue
{issue_text}

## Your previous patch (diff format)
```
{prev_diff}
```

If the patch correctly addresses the issue and handles edge cases, respond
with exactly:
CRITIQUE_OK

Otherwise, list specific issues (one per line). Focus on:
- Bugs in the changed logic
- Edge cases the patch doesn't handle
- Wrong files modified or wrong locations
- Inconsistencies with the rest of the codebase

Do NOT include a refined patch in this response — only the critique.
"""

SELFREFINE_REFINE_SUFFIX = """

## Self-critique of your previous attempt
{critique}

Now provide a NEW patch that addresses the critique. Use the same
<<<CHANGE path>>>...<<<CHANGE>>> format as your original attempt.
"""

REFLEXION_REFLECT_PROMPT = """\
Your previous attempt at solving this SWE-bench issue did not pass internal
quality checks (syntax or LLM review).

## Issue
{issue_text}

## Your previous patch
```
{prev_diff}
```

## Internal feedback
{feedback}

In 2-3 sentences, reflect on what went wrong with your previous attempt and
what strategy you should try next. Be specific and actionable, not vague.
This reflection will guide your next attempt.
"""

REFLEXION_REFINE_SUFFIX = """

## Reflections from past attempts (most recent last)
{reflections_section}

Now provide a NEW patch addressing the issues identified in the reflections.
Use the same <<<CHANGE path>>>...<<<CHANGE>>> format as your original attempt.
"""


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping malformed jsonl row in %s", path)
    return rows


def group_records_by_instance(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        inst = row.get("instance_id")
        if inst is not None:
            grouped[str(inst)].append(row)
    for inst_rows in grouped.values():
        inst_rows.sort(key=lambda row: int(row.get("step") or 0))
    return grouped


def instance_is_complete(rows: list[dict], steps: int) -> bool:
    if not rows:
        return False
    if any(row.get("stop_decision") is True for row in rows):
        return True
    return max(int(row.get("step") or 0) for row in rows) >= steps - 1


def rewrite_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=ROOT.parent, text=True).strip()
    except Exception:
        return "unknown"


# ============================================================================
# CallLogger
# ============================================================================

class CallLogger:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.raw_dir = out_dir / "raw_calls"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cost_log_path = out_dir / "cost_log.jsonl"
        self.cumulative_cost = 0.0
        self.lock = threading.Lock()

    def log_call(self, *, instance_id: str, step: int, purpose: str,
                  model: str, prompt: str, response: str,
                  prompt_tokens: int, completion_tokens: int,
                  cost_usd: float,
                  extra: dict | None = None) -> None:
        with self.lock:
            self.cumulative_cost += cost_usd
            cumulative = self.cumulative_cost
        raw_path = self.raw_dir / f"{instance_id}_step{step}_{purpose}.json"
        write_json(raw_path, {
            "ts": now_iso(),
            "instance_id": instance_id, "step": step, "purpose": purpose,
            "model": model, "prompt": prompt, "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "extra": extra or {},
        })
        cost_row = {
            "ts": now_iso(),
            "instance_id": instance_id, "step": step, "purpose": purpose,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "cumulative_usd": cumulative,
            "prompt_hash": sha256_str(prompt),
            "response_hash": sha256_str(response),
        }
        if extra:
            cost_row.update(logprob_summary_fields(
                extra.get("generation_logprobs") or extra.get("logprobs")
            ))
        append_jsonl(self.cost_log_path, cost_row)


# ============================================================================
# Cost computation (matches iter_refine_swebench.py)
# ============================================================================

def _cost_for(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:
    if "Qwen/Qwen2.5" in model_id or "gpt-oss" in model_id:
        return 0.0
    if "gpt-5-mini" in model_id:
        return (prompt_tokens / 1_000_000) * 0.25 + (completion_tokens / 1_000_000) * 2.0
    if "qwen" in model_id.lower():
        return (prompt_tokens / 1_000_000) * 0.4 + (completion_tokens / 1_000_000) * 1.6
    if "claude-sonnet" in model_id:
        return (prompt_tokens / 1_000_000) * 3.0 + (completion_tokens / 1_000_000) * 15.0
    if "claude-haiku" in model_id:
        return (prompt_tokens / 1_000_000) * 1.0 + (completion_tokens / 1_000_000) * 5.0
    return (prompt_tokens / 1_000_000) * 1.0 + (completion_tokens / 1_000_000) * 5.0


# ============================================================================
# Per-instance trajectory loop
# ============================================================================

def run_one_instance(*, inst: str, row: dict, oracle: dict[str, str],
                      step0_record: dict, step0_diff: str,
                      method: str, model_id: str, gen_name: str,
                      steps: int, temperature: float, client,
                      call_logger: CallLogger,
                      cost_lock: threading.Lock, cost_counter: dict,
                      cap_usd: float, gen_client=None,
                      request_logprobs: bool = False,
                      top_logprobs: int | None = None) -> dict:
    if gen_client is None:
        gen_client = client
    """Run one SWE instance's N-step trajectory under Self-Refine or
    Reflexion. Returns trajectory rows + stop info. Y is null until harness."""
    issue_text = (row["problem_statement"] or "")[:4000]

    # Step 0 (sunk)
    traj = [{
        "step": 0, "instance_id": inst, "method": method, "diff": step0_diff,
        "Y": step0_record.get("Y"),
        "L0_syntax": step0_record.get("L0_syntax"),
        "L1_lint": step0_record.get("L1_lint"),
        "L3_llm_review": step0_record.get("L3_llm_review"),
        "step_cost_usd": 0.0,
        "method_specific": {},
    }]

    instance_for_prompt = {
        "repo": row["repo"],
        "problem_statement": row["problem_statement"],
        "hints_text": row.get("hints_text", "") or "",
        "instance_id": inst,
    }
    base_prompt = scg.make_prompt(instance_for_prompt, oracle)

    reflections: list[str] = []
    prev_diff = step0_diff
    stop_step = None
    stop_reason = None

    from calibration.from_spotcheck import (
        _modified_file_contents, critic_L0_syntax,
        critic_L1_lint, critic_L3_llm_review,
    )

    for t in range(1, steps):
        with cost_lock:
            if cost_counter["v"] >= cap_usd:
                stop_reason = "cost_cap"
                stop_step = t
                break

        method_specific: dict = {}
        step_cost = 0.0
        prev = traj[t - 1]

        # ----- Stage 1: method-specific pre-refine call -----
        if method == "selfrefine":
            critique_prompt = SELFREFINE_CRITIQUE_PROMPT.format(
                issue_text=issue_text, prev_diff=prev_diff[:3000])
            try:
                resp = gen_client.chat.completions.create(**make_chat_completion_kwargs(
                    model=model_id,
                    messages=[{"role": "user", "content": critique_prompt}],
                    temperature=temperature, max_tokens=1500,
                    request_logprobs=request_logprobs,
                    top_logprobs=top_logprobs,
                ))
                critique_text = resp.choices[0].message.content or ""
                generation_logprobs = extract_completion_logprobs(
                    resp, requested=request_logprobs)
                u = resp.usage
                cost = _cost_for(model_id, u.prompt_tokens, u.completion_tokens)
            except Exception as e:
                log.warning("[%s/%s/%s] step %d critique failed: %s",
                             method, gen_name, inst, t, e)
                stop_reason = "critique_api_error"
                stop_step = t
                break
            with cost_lock:
                cost_counter["v"] += cost
            step_cost += cost
            call_logger.log_call(
                instance_id=inst, step=t, purpose="critique",
                model=model_id, prompt=critique_prompt, response=critique_text,
                prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                cost_usd=cost,
                extra={"generation_logprobs": generation_logprobs}
                if request_logprobs else None)
            method_specific["critique_text"] = critique_text
            if "CRITIQUE_OK" in critique_text.upper():
                stop_reason = "selfrefine_ok"
                stop_step = t
                traj.append({
                    "step": t, "instance_id": inst, "method": method,
                    "diff": prev_diff,
                    "Y": prev.get("Y"),
                    "L0_syntax": prev.get("L0_syntax"),
                    "L1_lint": prev.get("L1_lint"),
                    "L3_llm_review": prev.get("L3_llm_review"),
                    "step_cost_usd": step_cost,
                    "method_specific": method_specific,
                    "stop_decision": True,
                })
                break

        elif method == "reflexion":
            # Approximate "external test" = L0_syntax AND L3_llm_review on prev
            ext_pass = bool(prev.get("L0_syntax")) and bool(prev.get("L3_llm_review"))
            method_specific["external_test_approximate_pass"] = ext_pass
            method_specific["external_test_signal"] = "L0_syntax AND L3_llm_review (proxy for harness)"
            if ext_pass:
                stop_reason = "reflexion_test_pass"
                stop_step = t
                traj.append({
                    "step": t, "instance_id": inst, "method": method,
                    "diff": prev_diff,
                    "Y": prev.get("Y"),
                    "L0_syntax": prev.get("L0_syntax"),
                    "L1_lint": prev.get("L1_lint"),
                    "L3_llm_review": prev.get("L3_llm_review"),
                    "step_cost_usd": 0.0,
                    "method_specific": method_specific,
                    "stop_decision": True,
                })
                break
            feedback_text = (
                f"L0_syntax: {prev.get('L0_syntax')}, "
                f"L3_llm_review: {prev.get('L3_llm_review')}. "
                "These cheap proxies failed; the patch likely has problems "
                "with syntax, structure, or correctness."
            )
            reflect_prompt = REFLEXION_REFLECT_PROMPT.format(
                issue_text=issue_text, prev_diff=prev_diff[:3000],
                feedback=feedback_text)
            try:
                resp = gen_client.chat.completions.create(**make_chat_completion_kwargs(
                    model=model_id,
                    messages=[{"role": "user", "content": reflect_prompt}],
                    temperature=temperature, max_tokens=500,
                    request_logprobs=request_logprobs,
                    top_logprobs=top_logprobs,
                ))
                reflection_text = resp.choices[0].message.content or ""
                generation_logprobs = extract_completion_logprobs(
                    resp, requested=request_logprobs)
                u = resp.usage
                cost = _cost_for(model_id, u.prompt_tokens, u.completion_tokens)
            except Exception as e:
                log.warning("[%s/%s/%s] step %d reflect failed: %s",
                             method, gen_name, inst, t, e)
                stop_reason = "reflect_api_error"
                stop_step = t
                break
            with cost_lock:
                cost_counter["v"] += cost
            step_cost += cost
            call_logger.log_call(
                instance_id=inst, step=t, purpose="reflect",
                model=model_id, prompt=reflect_prompt, response=reflection_text,
                prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                cost_usd=cost,
                extra={"generation_logprobs": generation_logprobs}
                if request_logprobs else None)
            reflections.append(reflection_text)
            method_specific["reflection_text"] = reflection_text
            method_specific["memory_size"] = len(reflections)
        else:
            raise ValueError(f"unknown method: {method}")

        # ----- Stage 2: refine call -----
        if method == "selfrefine":
            refine_prompt = base_prompt + SELFREFINE_REFINE_SUFFIX.format(
                critique=method_specific["critique_text"][:1500])
        else:
            refl_section = "\n\n".join(
                f"Reflection {i+1}: {r}" for i, r in enumerate(reflections))
            refine_prompt = base_prompt + REFLEXION_REFINE_SUFFIX.format(
                reflections_section=refl_section[:3000])

        try:
            resp = gen_client.chat.completions.create(**make_chat_completion_kwargs(
                model=model_id,
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=temperature, max_tokens=4000,
                request_logprobs=request_logprobs,
                top_logprobs=top_logprobs,
            ))
            text = resp.choices[0].message.content or ""
            generation_logprobs = extract_completion_logprobs(
                resp, requested=request_logprobs)
            u = resp.usage
            cost = _cost_for(model_id, u.prompt_tokens, u.completion_tokens)
        except Exception as e:
            log.warning("[%s/%s/%s] step %d refine failed: %s",
                         method, gen_name, inst, t, e)
            stop_reason = "refine_api_error"
            stop_step = t
            break
        with cost_lock:
            cost_counter["v"] += cost
        step_cost += cost
        call_logger.log_call(
            instance_id=inst, step=t, purpose="refine",
            model=model_id, prompt=refine_prompt, response=text,
            prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
            cost_usd=cost,
            extra={"generation_logprobs": generation_logprobs}
            if request_logprobs else None)

        # Parse new diff
        blocks = scg.parse_change_blocks(text)
        modified = scg.apply_change_blocks(oracle, blocks) if blocks else {}
        new_diff = scg.build_diff(oracle, modified) if modified else ""

        # Inline cheap critics
        l0 = l1 = l3 = None
        if new_diff.strip():
            mod_files = _modified_file_contents(new_diff, oracle)
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
                        inst, row["problem_statement"], new_diff, client)
                    l3 = bool(l3_pass)
                    with cost_lock:
                        cost_counter["v"] += l3_cost
                    step_cost += l3_cost
                    append_jsonl(call_logger.cost_log_path, {
                        "ts": now_iso(),
                        "instance_id": inst, "step": t, "purpose": "L3_review",
                        "model": "L3_reviewer",
                        "prompt_tokens": -1, "completion_tokens": -1,
                        "cost_usd": l3_cost,
                        "cumulative_usd": cost_counter["v"],
                    })
                except Exception as e:
                    log.warning("[%s/%s/%s] step %d L3 failed: %s",
                                 method, gen_name, inst, t, e)
        else:
            l0 = l1 = l3 = False

        traj.append({
            "step": t, "instance_id": inst, "method": method, "diff": new_diff,
            "Y": None,  # populated by harness later
            "L0_syntax": l0, "L1_lint": l1, "L3_llm_review": l3,
            "step_cost_usd": step_cost,
            "method_specific": method_specific,
            "stop_decision": False,
        })
        log.info("[%s/%s/%s] step %d: diff=%d L0=%s L3=%s",
                 method, gen_name, inst, t, len(new_diff), l0, l3)
        prev_diff = new_diff

    return {
        "instance_id": inst,
        "trajectory": traj,
        "stop_step": stop_step,
        "stop_reason": stop_reason,
        "n_reflections": len(reflections),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["selfrefine", "reflexion"])
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--src-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    parser.add_argument("--n-instances", type=int, default=30)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cost-usd-per-model", default="5.0",
                        help="float OR key=val,key=val,...")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--request-logprobs", action="store_true",
                        help="Request and persist token logprobs for generation calls.")
    parser.add_argument("--top-logprobs", type=int, default=0,
                        help="Optional top_logprobs value when --request-logprobs is set.")
    parser.add_argument("--extend-existing", action="store_true",
                        help=("Skip completed instance_ids already present in "
                              "iter_records.jsonl. Partial existing trajectories "
                              "are dropped and rerun from step 0."))
    args = parser.parse_args()

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.error("OPENROUTER_API_KEY not set"); sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    os.environ.setdefault("HF_HOME", "/mnt/data/users/vlad.smirnov/hf_cache")
    import datasets
    ds = datasets.load_dataset(args.dataset, split="test")
    inst_to_row = {row["instance_id"]: row for row in ds}
    log.info("loaded %d instances from %s", len(inst_to_row), args.dataset)

    # Parse caps
    def _parse_caps(s: str) -> dict:
        try:
            return {None: float(s)}
        except ValueError:
            d = {}
            for kv in s.split(","):
                if "=" not in kv: continue
                k, v = kv.split("=", 1)
                d[k.strip()] = float(v.strip())
            return d
    caps_dict = _parse_caps(args.max_cost_usd_per_model)

    gens = [g.strip() for g in args.generators.split(",") if g.strip()]
    summary_per_gen: dict[str, dict] = {}

    for gen in gens:
        if gen not in scg.GENERATORS:
            log.error("unknown generator: %s", gen); continue
        model_id, base_url, _ = scg.GENERATORS[gen]
        # Per-generator generation client: vLLM if base_url is set, else OpenRouter.
        if base_url is not None:
            from openai import OpenAI as _OAI
            gen_client = _OAI(api_key="EMPTY", base_url=base_url)
            log.info("[%s] using vLLM endpoint %s", gen, base_url)
        else:
            gen_client = client
        cap_usd = caps_dict.get(gen, caps_dict.get(None, 5.0))

        gen_out = out_root / gen / args.method
        gen_out.mkdir(parents=True, exist_ok=True)
        records_path = gen_out / "iter_records.jsonl"

        # Run config
        write_json(gen_out / "RUN_CONFIG.json", {
            "ts": now_iso(), "git_sha": get_git_sha(),
            "method": args.method, "variant": "swe",
            "generator": gen, "model_id": model_id,
            "n_instances": args.n_instances, "steps": args.steps,
            "seed": args.seed, "temperature": args.temperature,
            "max_workers": args.max_workers,
            "request_logprobs": bool(args.request_logprobs),
            "top_logprobs": int(args.top_logprobs or 0),
            "cap_usd": cap_usd,
            "src_dir": str(args.src_dir),
            "output_dir": str(out_root),
            "dataset": args.dataset,
            "prompt_template_hashes": {
                "selfrefine_critique": sha256_str(SELFREFINE_CRITIQUE_PROMPT),
                "selfrefine_refine_suffix": sha256_str(SELFREFINE_REFINE_SUFFIX),
                "reflexion_reflect": sha256_str(REFLEXION_REFLECT_PROMPT),
                "reflexion_refine_suffix": sha256_str(REFLEXION_REFINE_SUFFIX),
            },
            "argv": sys.argv,
            "reflexion_external_test_proxy": "L0_syntax AND L3_llm_review",
        })

        # Step 0 critic_results
        crit_path = args.src_dir / gen / "critic_results.jsonl"
        if not crit_path.exists():
            log.error("[%s] no critic_results at %s", gen, crit_path); continue
        crit_by_inst = {}
        for line in open(crit_path):
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("patch_id") == 0:
                crit_by_inst[r["instance_id"]] = r

        # Step 0 diffs from predictions_p0
        pred_path = args.src_dir / gen / "predictions_p0.jsonl"
        diff_by_inst = {}
        if pred_path.exists():
            for line in open(pred_path):
                line = line.strip()
                if not line: continue
                r = json.loads(line)
                diff_by_inst[r["instance_id"]] = r.get("model_patch", "") or ""

        candidate = [k for k in crit_by_inst if k in inst_to_row and k in diff_by_inst]
        if args.n_instances and args.n_instances > 0:
            candidate = candidate[:args.n_instances]

        existing_rows: list[dict] = []
        completed_existing: set[str] = set()
        if args.extend_existing and records_path.exists():
            existing_rows = load_jsonl(records_path)
            existing_by_inst = group_records_by_instance(existing_rows)
            completed_existing = {
                inst for inst, rows in existing_by_inst.items()
                if instance_is_complete(rows, args.steps)
            }
            partial_existing = set(existing_by_inst) - completed_existing
            if partial_existing:
                backup_path = records_path.with_suffix(".jsonl.pre_extend_existing.bak")
                if not backup_path.exists():
                    backup_path.write_text(records_path.read_text())
                existing_rows = [
                    row for row in existing_rows
                    if str(row.get("instance_id")) in completed_existing
                ]
                rewrite_jsonl(records_path, existing_rows)
                log.info("[%s/%s] --extend-existing: dropped %d partial "
                         "instance trajectories for rerun",
                         gen, args.method, len(partial_existing))
            before = len(candidate)
            candidate = [inst for inst in candidate if inst not in completed_existing]
            log.info("[%s/%s] --extend-existing: %d completed existing, "
                     "%d -> %d new instances",
                     gen, args.method, len(completed_existing),
                     before, len(candidate))

        log.info("[%s/%s] %d eligible instances (cap $%.1f)",
                 gen, args.method, len(candidate), cap_usd)

        # Pre-fetch oracles
        oracle_cache: dict[str, dict] = {}
        for inst in candidate:
            row = inst_to_row[inst]
            files = scg.get_changed_files_from_patch(row["patch"])
            oracle_cache[inst] = scg.fetch_oracle_files(row["repo"], row["base_commit"], files)

        cost_lock = threading.Lock()
        existing_cost = sum(float(row.get("step_cost_usd") or 0.0)
                            for row in existing_rows)
        cost_counter = {"v": existing_cost}
        call_logger = CallLogger(gen_out)
        call_logger.cumulative_cost = existing_cost

        all_results: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {}
            for inst in candidate:
                row = inst_to_row[inst]
                fut = ex.submit(
                    run_one_instance,
                    inst=inst, row=row, oracle=oracle_cache[inst],
                    step0_record=crit_by_inst[inst],
                    step0_diff=diff_by_inst[inst],
                    method=args.method, model_id=model_id, gen_name=gen,
                    steps=args.steps, temperature=args.temperature,
                    client=client, gen_client=gen_client, call_logger=call_logger,
                    cost_lock=cost_lock, cost_counter=cost_counter,
                    cap_usd=cap_usd,
                    request_logprobs=args.request_logprobs,
                    top_logprobs=args.top_logprobs)
                futures[fut] = inst
            for fut in as_completed(futures):
                inst = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    log.error("[%s/%s/%s] failed: %s", args.method, gen, inst, e)
                    continue
                all_results.append(result)
                for row in result["trajectory"]:
                    append_jsonl(records_path, row)

        final_rows = load_jsonl(records_path)
        final_by_inst = group_records_by_instance(final_rows)
        completed_final_by_inst = {
            inst: rows for inst, rows in final_by_inst.items()
            if instance_is_complete(rows, args.steps)
        }
        new_result_meta = {r["instance_id"]: r for r in all_results}

        def stop_distribution_entry(inst: str, rows: list[dict]) -> dict:
            meta = new_result_meta.get(inst)
            if meta is not None:
                return {
                    "instance_id": inst,
                    "stop_step": meta["stop_step"],
                    "stop_reason": meta["stop_reason"],
                    "n_reflections": meta["n_reflections"],
                }
            return {
                "instance_id": inst,
                "stop_step": next(
                    (int(row.get("step") or 0) for row in rows
                     if row.get("stop_decision") is True),
                    max(int(row.get("step") or 0) for row in rows),
                ),
                "stop_reason": (
                    "stop_decision"
                    if any(row.get("stop_decision") is True for row in rows)
                    else "max_steps"
                ),
                "n_reflections": max(int(row.get("step") or 0) for row in rows),
            }

        # Stop distribution
        write_json(gen_out / "stop_distribution.json", {
            "n_instances": len(completed_final_by_inst),
            "instances": [
                stop_distribution_entry(inst, rows)
                for inst, rows in completed_final_by_inst.items()
            ],
        })

        # Cost summary
        total_cost = sum(float(row.get("step_cost_usd") or 0.0)
                         for row in final_rows)
        write_json(gen_out / "cost_summary.json", {
            "generator": gen, "method": args.method,
            "n_instances_completed": len(completed_final_by_inst),
            "total_cost_usd": total_cost,
            "cap_usd": cap_usd, "cap_hit": total_cost >= cap_usd,
        })

        # Write per-step predictions for harness
        for step in range(1, args.steps):
            pred_step_path = gen_out / f"predictions_iter_step{step}.jsonl"
            with open(pred_step_path, "w") as f:
                for inst, rows in completed_final_by_inst.items():
                    step_rows = [
                        row for row in rows
                        if int(row.get("step") or 0) == step and row.get("diff")
                    ]
                    if not step_rows:
                        continue
                    row = step_rows[-1]
                    f.write(json.dumps({
                        "instance_id": inst,
                        "model_patch": row["diff"],
                        "model_name_or_path": f"{gen}_{args.method}_iter_step{step}",
                    }) + "\n")

        log.info("[%s/%s] done: %d new instances, %d total completed, "
                 "$%.3f spent (cap $%.2f)",
                 gen, args.method, len(all_results),
                 len(completed_final_by_inst), total_cost, cap_usd)
        summary_per_gen[gen] = {
            "n_instances": len(completed_final_by_inst),
            "total_cost_usd": total_cost,
            "cap_hit": total_cost >= cap_usd,
        }

    write_json(out_root / f"SUMMARY_{args.method}.json", summary_per_gen)
    log.info("Method %s done. Per-gen summary at %s/SUMMARY_%s.json",
             args.method, out_root, args.method)


if __name__ == "__main__":
    main()
