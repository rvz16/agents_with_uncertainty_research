#!/usr/bin/env python3
"""Strict multi-step train calibration followed by frozen LCB test replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ORCH = REPO / "experiments" / "orchestration_hypothesis_testing"
sys.path[:0] = [str(REPO), str(ORCH), str(ORCH / "scripts")]

from different_agents.v4 import lcb_llm_tool_agent as agent  # noqa: E402
from analyze_lcb_llm_tool_agent_logs import prr, spearman  # noqa: E402
from fitted_live.common import Candidate, safe_stem  # noqa: E402

CRITICS = {
    "L0_syntax": "L0",
    "L1_lint": "L1",
    "L2_public_tests": "L2",
    "L3_llm_review": "L3",
}
ACTION_CRITIC = {
    "critic_L0": "L0_syntax",
    "critic_L1": "L1_lint",
    "critic_L2": "L2_public_tests",
    "critic_L3": "L3_llm_review",
}


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["lcb_medium", "lcb_hard"], required=True)
    p.add_argument("--generator", default="gpt_oss_20b_local")
    p.add_argument("--source-run-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--max-generations", type=int, default=5)
    p.add_argument("--max-verifications", type=int, default=1)
    p.add_argument("--max-tokens-decision", type=int, default=4096)
    p.add_argument("--max-tokens-generation", type=int, default=32768)
    p.add_argument("--top-logprobs", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--l3-workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["score", "higher_is_better", "spearman", "PRR", "PRR_05"]
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def source_paths(a: argparse.Namespace) -> dict[str, Path]:
    stem = f"{a.benchmark}__{agent.canonical_generator_key(a.generator)}"
    return {
        "split": a.source_run_root / f"{stem}.split.json",
        "results": a.source_run_root / f"{stem}.jsonl",
        "analysis": a.source_run_root / "readable" / a.benchmark / "analysis_summary.json",
        "metrics": a.source_run_root / "readable" / a.benchmark / "metric_scores.csv",
    }


def output_paths(a: argparse.Namespace) -> dict[str, Path]:
    stem = f"{a.benchmark}__{agent.canonical_generator_key(a.generator)}.strict"
    return {
        "trajectories": a.output_dir / f"{stem}.trajectories.jsonl",
        "actions": a.output_dir / f"{stem}.actions.jsonl",
        "logprobs": a.output_dir / f"{stem}.generation_logprobs.jsonl",
        "evaluations": a.output_dir / f"{stem}.candidate_evaluations.jsonl",
        "params": a.output_dir / "frozen_params.json",
        "replay": a.output_dir / "test_replay_scores.jsonl",
        "metrics": a.output_dir / "metric_scores.csv",
        "report": a.output_dir / "REPORT.md",
    }


def make_adapter(a: argparse.Namespace):
    return agent.make_function_adapter(
        benchmark=a.benchmark,
        n_instances=0,
        seed=a.seed,
        lcb_version="all",
        plus_input_cap=200,
        lcb_private_test_cap=0,
        platform="leetcode",
    )


def make_deps(a: argparse.Namespace, adapter: Any, logprobs: Path) -> agent.AgentDeps:
    generator = agent.canonical_generator_key(a.generator)
    llm = agent._make_client(generator)
    reviewer_url = os.environ.get("REVIEWER_BASE_URL", "").strip()
    if reviewer_url:
        from openai import OpenAI

        reviewer = OpenAI(api_key="EMPTY", base_url=reviewer_url)
    else:
        try:
            reviewer = agent._make_client(None)
        except SystemExit as exc:
            raise RuntimeError("OPENROUTER_API_KEY is required") from exc
    return agent.AgentDeps(
        adapter=adapter,
        llm_client=llm,
        reviewer_client=reviewer,
        model_id=agent.GENERATORS[generator][0],
        decision_temperature=0.2,
        generation_temperature=0.7,
        max_tokens_decision=a.max_tokens_decision,
        max_tokens_generation=a.max_tokens_generation,
        max_code_chars=131072,
        save_generation_logprobs=True,
        require_generation_logprobs=True,
        top_logprobs=a.top_logprobs,
        logprobs_output=logprobs,
    )


def complete_trajectory(row: dict[str, Any]) -> bool:
    return (
        int(row.get("n_generations") or 0) > 0
        and not row.get("error")
        and not str(row.get("final_action") or "").startswith("exception:")
    )


def latest_by_instance(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["instance_id"]): row for row in read_jsonl(path) if row.get("instance_id")}


def run_train(
    a: argparse.Namespace,
    src: dict[str, Path],
    out: dict[str, Path],
    adapter: Any,
    deps: agent.AgentDeps,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    split = json.loads(src["split"].read_text())
    train_ids = [str(x) for x in split["train_ids"]]
    instances = {adapter.instance_id(x): x for x in adapter.load_instances()}
    if missing := [x for x in train_ids if x not in instances]:
        raise RuntimeError(f"train IDs missing from current LCB data: {missing}")
    if not a.resume:
        for name in ("trajectories", "actions", "logprobs", "evaluations"):
            out[name].unlink(missing_ok=True)
    prior = json.loads(src["analysis"].read_text()).get("prior") or {}
    latest = latest_by_instance(out["trajectories"])
    done = {key for key, row in latest.items() if complete_trajectory(row)}
    print(f"train={len(train_ids)} remaining={len(set(train_ids) - done)}", flush=True)
    for index, instance_id in enumerate(train_ids, 1):
        if instance_id in done:
            continue
        started = time.perf_counter()
        state = agent.initial_state(
            instance=instances[instance_id],
            instance_id=instance_id,
            benchmark=a.benchmark,
            max_steps=a.max_steps,
            max_generations=a.max_generations,
            max_verifications=a.max_verifications,
            prior_summary=prior,
        )
        try:
            final = agent.run_sage_agent_episode(state, deps)
            row = agent.result_record(
                final,
                deps,
                time.perf_counter() - started,
                split_summary=split,
                prior_summary=prior,
            )
            row.update(split="train_strict_calibration", agent_backend="sage")
        except Exception as exc:
            row = {
                "benchmark": a.benchmark,
                "split": "train_strict_calibration",
                "instance_id": instance_id,
                "final_action": f"exception:{type(exc).__name__}",
                "error": str(exc),
                "trajectory": [],
            }
        agent.append_jsonl(out["trajectories"], row)
        agent.append_actions(
            out["actions"],
            split="train_strict_calibration",
            benchmark=a.benchmark,
            instance_id=instance_id,
            model_id=deps.model_id,
            actions=row.get("trajectory", []),
            extra={"agent_backend": "sage"},
        )
        print(
            f"[{index}/{len(train_ids)}] {instance_id} "
            f"g={row.get('n_generations', 0)} final={row.get('final_action')}",
            flush=True,
        )
    latest = latest_by_instance(out["trajectories"])
    failed = [x for x in train_ids if not complete_trajectory(latest.get(x, {}))]
    if failed:
        raise RuntimeError(f"incomplete train trajectories: {failed}")
    rows = [latest[x] for x in train_ids]
    agent.write_jsonl_atomic(out["trajectories"], rows)
    actions = []
    for row in rows:
        for action in row.get("trajectory", []):
            actions.append(
                {
                    "split": "train_strict_calibration",
                    "benchmark": a.benchmark,
                    "instance_id": row["instance_id"],
                    "model_id": deps.model_id,
                    **action,
                }
            )
    agent.write_jsonl_atomic(out["actions"], actions)
    return rows, instances


def load_candidates(path: Path) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    marker = ', "logprobs": '
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            prefix, sep, _ = line.rpartition(marker)
            raw = json.loads(prefix + "}") if sep else json.loads(line)
            instance_id = str(raw["instance_id"])
            generation_index = int(raw.get("generation_index", 0))
            code = str(raw.get("code") or "")
            latest[(instance_id, generation_index)] = {
                "benchmark": raw.get("benchmark"),
                "instance_id": instance_id,
                "generation_index": generation_index,
                "step": int(raw.get("step", 0)),
                "code": code,
                "code_sha256": digest(code),
                "code_chars": len(code),
                "source_line": line_no,
            }
    return sorted(latest.values(), key=lambda x: (x["instance_id"], x["generation_index"]))


def validate_candidates(trajectories: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    keys = {(x["instance_id"], x["generation_index"]) for x in rows}
    expected = {
        (str(t["instance_id"]), index)
        for t in trajectories
        for index in range(int(t["n_generations"]))
    }
    if missing := expected - keys:
        raise RuntimeError(f"missing candidate sidecars: {sorted(missing)[:10]}")


def eval_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["instance_id"]), int(row["generation_index"]), str(row["code_sha256"])


def complete_eval(row: dict[str, Any]) -> bool:
    return all(row.get(x) in (True, False) for x in ["Y", *CRITICS])


def eval_local(adapter: Any, instance: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    candidate = Candidate(row["code"], row["code"], "code")
    run_id = safe_stem(f"strict__{row['instance_id']}__g{row['generation_index']}", 180)
    started = time.perf_counter()
    verify = adapter.verify(instance, candidate, run_id)
    result = {
        **{k: v for k, v in row.items() if k != "code"},
        "Y": bool(verify.passed),
        "private_test_detail": verify.detail,
    }
    for field, critic_name in CRITICS.items():
        if critic_name == "L3":
            continue
        critic = adapter.run_critic(critic_name, instance, candidate, None)
        result[field] = critic.passed
        result[f"{field}_detail"] = critic.detail
    result["local_wall_clock_s"] = round(time.perf_counter() - started, 4)
    return result


def eval_l3(
    adapter: Any,
    reviewer: Any,
    instance: dict[str, Any],
    candidate_row: dict[str, Any],
    base: dict[str, Any],
) -> dict[str, Any]:
    candidate = Candidate(candidate_row["code"], candidate_row["code"], "code")
    started = time.perf_counter()
    result = None
    for attempt in range(3):
        result = adapter.run_critic("L3", instance, candidate, reviewer)
        if result.passed in (True, False):
            break
        time.sleep(2**attempt)
    assert result is not None
    return {
        **base,
        "L3_llm_review": result.passed,
        "L3_llm_review_detail": result.detail,
        "L3_llm_review_raw_response": result.raw_response,
        "L3_llm_review_api_cost_usd": result.api_cost_usd,
        "L3_llm_review_prompt_tokens": result.prompt_tokens,
        "L3_llm_review_completion_tokens": result.completion_tokens,
        "L3_wall_clock_s": round(time.perf_counter() - started, 4),
    }


def evaluate_candidates(
    a: argparse.Namespace,
    out: dict[str, Path],
    adapter: Any,
    deps: agent.AgentDeps,
    instances: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cache = {eval_key(x): x for x in read_jsonl(out["evaluations"])}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[(row["instance_id"], row["code_sha256"])].append(row)
    resolved: dict[tuple[str, str], dict[str, Any]] = {}
    pending = {}
    for group, members in groups.items():
        hit = next((cache[eval_key(x)] for x in members if complete_eval(cache.get(eval_key(x), {}))), None)
        if hit:
            resolved[group] = hit
        else:
            pending[group] = members[0]
    if pending:
        print(f"all private tests + L0/L1/L2: {len(pending)} unique", flush=True)
        local: dict[tuple[str, str], dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = {
                pool.submit(eval_local, adapter, instances[row["instance_id"]], row): group
                for group, row in pending.items()
            }
            for count, future in enumerate(as_completed(futures), 1):
                group = futures[future]
                local[group] = future.result()
                agent.append_jsonl(out["evaluations"], local[group])
                if count % 20 == 0 or count == len(futures):
                    print(f"  local {count}/{len(futures)}", flush=True)
        print(f"L3: {len(local)} unique", flush=True)
        with ThreadPoolExecutor(max_workers=a.l3_workers) as pool:
            futures = {
                pool.submit(
                    eval_l3,
                    adapter,
                    deps.reviewer_client,
                    instances[row["instance_id"]],
                    pending[group],
                    row,
                ): group
                for group, row in local.items()
            }
            for count, future in enumerate(as_completed(futures), 1):
                group = futures[future]
                resolved[group] = future.result()
                agent.append_jsonl(out["evaluations"], resolved[group])
                if count % 20 == 0 or count == len(futures):
                    print(f"  L3 {count}/{len(futures)}", flush=True)
    rows = []
    for candidate in candidates:
        source = resolved[(candidate["instance_id"], candidate["code_sha256"])]
        rows.append({**source, **{k: v for k, v in candidate.items() if k != "code"}})
    rows.sort(key=lambda x: (x["instance_id"], x["generation_index"]))
    if incomplete := [eval_key(x) for x in rows if not complete_eval(x)]:
        raise RuntimeError(f"incomplete exhaustive evaluations: {incomplete[:10]}")
    agent.write_jsonl_atomic(out["evaluations"], rows)
    return rows


def beta(n_pass: int, total: int) -> float:
    return (n_pass + 1.0) / (total + 2.0)


def trans_counts(sequences: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    counts = Counter({"n00": 0, "n01": 0, "n10": 0, "n11": 0})
    for rows in sequences.values():
        for before, after in zip(rows, rows[1:]):
            counts[f"n{int(before['Y'])}{int(after['Y'])}"] += 1
    return dict(counts)


def fit_kernel(counts: dict[str, int]) -> dict[str, Any]:
    broken = counts["n00"] + counts["n01"]
    correct = counts["n10"] + counts["n11"]
    return {
        "p_fix_broken": beta(counts["n01"], broken),
        "p_break_correct": beta(counts["n10"], correct),
        **counts,
        "n_transitions": broken + correct,
        "smoothing": "Beta(1,1)",
    }


def fit_critic(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    usable = [x for x in rows if x.get("Y") in (True, False) and x.get(field) in (True, False)]
    y1 = [x for x in usable if x["Y"] is True]
    y0 = [x for x in usable if x["Y"] is False]
    tp = sum(x[field] is True for x in y1)
    fp = sum(x[field] is True for x in y0)
    p1, p0 = beta(tp, len(y1)), beta(fp, len(y0))
    return {
        "p_pass_y1": p1,
        "p_pass_y0": p0,
        "gap": p1 - p0,
        "TP": tp,
        "FN": len(y1) - tp,
        "FP": fp,
        "TN": len(y0) - fp,
        "n": len(usable),
        "smoothing": "Beta(1,1)",
    }


def repo_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, text=True, capture_output=True, check=False
        ).stdout.strip()

    diff = git("diff", "--binary")
    return {
        "commit": git("rev-parse", "HEAD"),
        "status": git("status", "--short").splitlines(),
        "tracked_diff_sha256": digest(diff),
    }


def fit_params(
    a: argparse.Namespace,
    src: dict[str, Path],
    trajectories: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sequences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sequences[row["instance_id"]].append(row)
    for sequence in sequences.values():
        sequence.sort(key=lambda x: x["generation_index"])
    first = [x[0] for x in sequences.values()]
    correct = sum(x["Y"] is True for x in first)
    max_g = max(map(len, sequences.values()))
    by_position = {}
    for position in range(max_g - 1):
        slices = {
            key: value[position : position + 2]
            for key, value in sequences.items()
            if len(value) > position + 1
        }
        by_position[f"{position + 1}_to_{position + 2}"] = fit_kernel(trans_counts(slices))
    by_generation = {}
    generation_accuracy = {}
    for index in range(max_g):
        generation_rows = [x for x in rows if x["generation_index"] == index]
        if generation_rows:
            n_correct = sum(x["Y"] is True for x in generation_rows)
            generation_accuracy[str(index + 1)] = {
                "n": len(generation_rows),
                "correct": n_correct,
                "accuracy": n_correct / len(generation_rows),
            }
            by_generation[str(index + 1)] = {
                field: fit_critic(generation_rows, field) for field in CRITICS
            }
    split_bytes = src["split"].read_bytes()
    return {
        "protocol": "exact train IDs -> multi-step trajectories -> exhaustive labels -> frozen test replay",
        "benchmark": a.benchmark,
        "source_run_root": str(a.source_run_root),
        "source_split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "n_train_instances": len(sequences),
        "n_train_trajectories": len(trajectories),
        "n_train_candidates": len(rows),
        "prior": {
            "prior_Y1": beta(correct, len(first)),
            "n": len(first),
            "correct": correct,
            "smoothing": "Beta(1,1)",
        },
        "transition_kernel": {
            "aggregate": fit_kernel(trans_counts(sequences)),
            "by_position": by_position,
            "application": "before generations 2+ only",
        },
        "critic_likelihoods": {field: fit_critic(rows, field) for field in CRITICS},
        "critic_likelihoods_by_generation": by_generation,
        "generation_accuracy": generation_accuracy,
        "repository": repo_state(),
        "config": {
            "max_steps": a.max_steps,
            "max_generations": a.max_generations,
            "max_tokens_decision": a.max_tokens_decision,
            "max_tokens_generation": a.max_tokens_generation,
            "private_test_cap": 0,
            "generator": agent.canonical_generator_key(a.generator),
        },
    }


def bayes_update(belief: float, theta: dict[str, Any], passed: bool) -> float:
    p1, p0 = float(theta["p_pass_y1"]), float(theta["p_pass_y0"])
    l1, l0 = (p1, p0) if passed else (1 - p1, 1 - p0)
    denominator = l1 * belief + l0 * (1 - belief)
    return belief if denominator <= 1e-12 else l1 * belief / denominator


def kernel_update(belief: float, kernel: dict[str, Any]) -> float:
    return belief * (1 - kernel["p_break_correct"]) + (1 - belief) * kernel["p_fix_broken"]


def replay_one(
    result: dict[str, Any],
    prior: float,
    theta: dict[str, Any],
    aggregate: dict[str, Any] | None,
    by_position: dict[str, Any] | None = None,
) -> float:
    belief, generation_count, before_verify = prior, 0, None
    trajectory = sorted(result.get("trajectory") or [], key=lambda x: int(x.get("step", 0)))
    for row in trajectory:
        action = str(row.get("action") or "")
        if action == "generate":
            if row.get("skipped") is True:
                continue
            if aggregate is not None and generation_count > 0:
                kernel = aggregate
                if by_position is not None and generation_count > 0:
                    key = f"{generation_count}_to_{generation_count + 1}"
                    kernel = by_position.get(key, aggregate)
                belief = kernel_update(belief, kernel)
            generation_count += 1
        elif action in ACTION_CRITIC and row.get("passed") in (True, False):
            belief = bayes_update(belief, theta[ACTION_CRITIC[action]], bool(row["passed"]))
        elif action in {"verify", "final_verify"} and row.get("passed") in (True, False):
            before_verify = belief
    return belief if before_verify is None else before_verify


def replay(src: dict[str, Path], params: dict[str, Any]):
    results = read_jsonl(src["results"])
    prior = params["prior"]["prior_Y1"]
    theta = params["critic_likelihoods"]
    aggregate = params["transition_kernel"]["aggregate"]
    positions = params["transition_kernel"]["by_position"]
    variants = {
        "bayes_strict_critics_no_transition": (None, None),
        "bayes_strict_aggregate_transition": (aggregate, None),
        "bayes_strict_position_transition": (aggregate, positions),
    }
    replay_rows = []
    for result in results:
        row = {
            "benchmark": result.get("benchmark"),
            "instance_id": str(result["instance_id"]),
            "quality": int(bool(result.get("fixed"))),
            "n_generations": int(result.get("n_generations") or 0),
        }
        for name, spec in variants.items():
            row[name] = replay_one(result, prior, theta, *spec)
        replay_rows.append(row)
    quality = [x["quality"] for x in replay_rows]
    metric_rows = []
    for name in variants:
        scores = [x[name] for x in replay_rows]
        metric_rows.append(
            {
                "score": name,
                "higher_is_better": True,
                "spearman": spearman(scores, quality),
                "PRR": prr(scores, quality, 1.0),
                "PRR_05": prr(scores, quality, 0.5),
            }
        )
    return replay_rows, metric_rows


def read_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: Any) -> str:
    return "NA" if value in (None, "") else f"{float(value):.3f}"


def write_report(path: Path, params: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    kernel = params["transition_kernel"]["aggregate"]
    lines = [
        f"# Strict calibration: {params['benchmark']}",
        "",
        "All train candidates use all private tests and exhaustive L0/L1/L2/L3 labels. "
        "No test label was used while fitting.",
        "",
        "## PRR",
        "",
        "| Method | PRR@0.5 | PRR@1.0 | Spearman |",
        "|---|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['score']} | {fmt(row.get('PRR_05'))} | "
            f"{fmt(row.get('PRR'))} | {fmt(row.get('spearman'))} |"
        )
    lines += [
        "",
        "## Transition",
        "",
        f"- p_fix_broken: {kernel['p_fix_broken']:.6f}",
        f"- p_break_correct: {kernel['p_break_correct']:.6f}",
        f"- transitions: {kernel['n_transitions']}",
        f"- counts: 00={kernel['n00']}, 01={kernel['n01']}, "
        f"10={kernel['n10']}, 11={kernel['n11']}",
        "",
        "## Critics",
        "",
        "| Critic | P(pass|correct) | P(pass|wrong) | Gap | TP | FN | FP | TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in params["critic_likelihoods"].items():
        lines.append(
            f"| {name} | {row['p_pass_y1']:.3f} | {row['p_pass_y0']:.3f} | "
            f"{row['gap']:.3f} | {row['TP']} | {row['FN']} | "
            f"{row['FP']} | {row['TN']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    a = args_parser()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    src, out = source_paths(a), output_paths(a)
    if missing := [str(x) for x in src.values() if not x.exists()]:
        raise FileNotFoundError(f"missing source files: {missing}")
    adapter = make_adapter(a)
    deps = make_deps(a, adapter, out["logprobs"])
    trajectories, instances = run_train(a, src, out, adapter, deps)
    candidates = load_candidates(out["logprobs"])
    validate_candidates(trajectories, candidates)
    evaluations = evaluate_candidates(a, out, adapter, deps, instances, candidates)
    params = fit_params(a, src, trajectories, evaluations)
    atomic_json(out["params"], params)
    replay_rows, strict_metrics = replay(src, params)
    agent.write_jsonl_atomic(out["replay"], replay_rows)
    metrics = read_metrics(src["metrics"]) + strict_metrics
    atomic_csv(out["metrics"], metrics)
    write_report(out["report"], params, metrics)
    print(f"complete: {out['report']}", flush=True)


if __name__ == "__main__":
    main()
