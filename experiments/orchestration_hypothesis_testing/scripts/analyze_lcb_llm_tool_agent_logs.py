#!/usr/bin/env python3
"""Build notebook-style logprob / Bayesian-quality tables for LCB tool-agent runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CRITIC_MAP = {
    "critic_L0": "L0_syntax",
    "critic_L1": "L1_lint",
    "critic_L2": "L2_public_tests",
    "critic_L3": "L3_llm_review",
}
CRITIC_SHORT = {
    "L0_syntax": "L0",
    "L1_lint": "L1",
    "L2_public_tests": "L2",
    "L3_llm_review": "L3",
}
DEFAULT_KERNEL = {"p_fix_broken": 0.50, "p_break_correct": 0.05}


def configure_hf_cache() -> None:
    root = Path(
        os.environ.get(
            "ORCH_HF_CACHE_DIR",
            "/capstor/store/cscs/swissai/a0142/hf_cache",
        )
    )
    for name, path in {
        "HF_HOME": root,
        "HF_HUB_CACHE": root / "hub",
        "HUGGINGFACE_HUB_CACHE": root / "hub",
        "HF_DATASETS_CACHE": root / "datasets",
        "XDG_CACHE_HOME": root / "xdg",
    }.items():
        os.environ.setdefault(name, str(path))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_or_none(x: Any) -> float | None:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def logprob_stats(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "llm_perplexity": None,
            "perplexity": None,
            "llm_log_seq_prob": None,
            "n_logprob_tokens": 0,
            "logprobs_supported": False,
        }
    vals = []
    for item in ((row.get("logprobs") or {}).get("content") or []):
        lp = finite_or_none(item.get("logprob") if isinstance(item, dict) else None)
        if lp is not None and lp > -9998:
            vals.append(lp)
    seq = sum(vals) if vals else None
    mean = (seq / len(vals)) if vals else None
    return {
        # Historical name used in analysis.ipynb: this is mean token logprob.
        "llm_perplexity": mean,
        "perplexity": (math.exp(-mean) if mean is not None else None),
        "llm_log_seq_prob": seq,
        "n_logprob_tokens": len(vals),
        "logprobs_supported": bool(vals),
    }


def beta_rate(n_pass: int, n_total: int) -> float:
    return (n_pass + 1.0) / (n_total + 2.0)


def prior_from_train(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if r.get("passed") in (True, False)]
    correct = sum(1 for r in usable if r.get("passed") is True)
    return {
        "prior_Y1": beta_rate(correct, len(usable)) if usable else 0.5,
        "prior_calibration_n": len(usable),
        "prior_calibration_correct": correct,
        "prior_smoothing": "Beta(1,1)",
    }


def normalize_theta(j: dict[str, Any]) -> dict[str, dict[str, float]]:
    theta = {}
    for name, row in (j.get("critic_likelihoods") or {}).items():
        p1 = row.get("p_pass_y1", row.get("P_pass_given_Y1"))
        p0 = row.get("p_pass_y0", row.get("P_pass_given_Y0"))
        if p1 is not None and p0 is not None:
            theta[name] = {"p_pass_y1": float(p1), "p_pass_y0": float(p0)}
    return theta


def fit_likelihood_table(records: list[dict[str, Any]], generator: str) -> dict[str, Any]:
    rows = [r for r in records if r.get("Y") in (0, 1)]
    n_y1 = sum(1 for r in rows if r["Y"] == 1)
    likelihoods = {}
    for critic in CRITIC_MAP.values():
        tp = sum(1 for r in rows if r["Y"] == 1 and r.get(critic) is True)
        fn = sum(1 for r in rows if r["Y"] == 1 and r.get(critic) is False)
        fp = sum(1 for r in rows if r["Y"] == 0 and r.get(critic) is True)
        tn = sum(1 for r in rows if r["Y"] == 0 and r.get(critic) is False)
        n_pos = tp + fn
        n_neg = fp + tn
        p1 = beta_rate(tp, n_pos) if n_pos else None
        p0 = beta_rate(fp, n_neg) if n_neg else None
        likelihoods[critic] = {
            "P_pass_given_Y1": p1,
            "P_pass_given_Y0": p0,
            "gap": (p1 - p0) if p1 is not None and p0 is not None else None,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
        }
    return {
        "generator": generator,
        "source": "train_prior_calibration candidates; critics recomputed on train only",
        "n_records": len(records),
        "n_evaluated": len(rows),
        "n_resolved": n_y1,
        "prior_Y1": beta_rate(n_y1, len(rows)) if rows else 0.5,
        "critic_likelihoods": likelihoods,
        "smoothing": "Beta(1,1)",
    }


def fit_train_likelihoods(
    *,
    benchmark: str,
    generator: str,
    split_path: Path,
    train_rows: list[dict[str, Any]],
    output_path: Path,
    private_test_cap: int,
    seed: int,
    platform: str,
    lcb_version: str,
    include_l3: bool,
) -> dict[str, Any]:
    from calibration import lcb as lcb_calibrate

    sys.modules.setdefault("lcb_calibrate", lcb_calibrate)
    from fitted_live.common import Candidate
    from fitted_live.function_adapters import LCBAdapter

    split = json.loads(split_path.read_text())
    train_ids = {str(x) for x in split.get("train_ids", [])}
    by_train_row = {
        str(r.get("instance_id")): r
        for r in train_rows
        if str(r.get("instance_id")) in train_ids and r.get("passed") in (True, False)
    }
    difficulty = {"med": "medium"}.get(benchmark.split("_", 1)[1], benchmark.split("_", 1)[1])
    adapter = LCBAdapter(
        benchmark=benchmark,
        difficulty=difficulty,
        n_instances=0,
        seed=seed,
        platform=platform,
        lcb_version=lcb_version,
        private_test_cap=private_test_cap,
    )
    instances = {adapter.instance_id(inst): inst for inst in adapter.load_instances()}
    records = []
    for iid in sorted(train_ids):
        row = by_train_row.get(iid)
        inst = instances.get(iid)
        if row is None or inst is None:
            continue
        rec: dict[str, Any] = {
            "instance_id": iid,
            "patch_id": int(row.get("patch_id", 0) or 0),
            "Y": int(bool(row.get("passed"))),
        }
        candidate = Candidate(
            payload=str(row.get("code") or ""),
            raw_text=str(row.get("code") or ""),
            kind="code",
        )
        for critic in ("L0_syntax", "L1_lint", "L2_public_tests"):
            try:
                rec[critic] = bool(adapter.run_critic(CRITIC_SHORT[critic], inst, candidate, None).passed)
            except Exception as exc:
                rec[critic] = None
                rec[f"{critic}_error"] = f"{type(exc).__name__}: {exc}"
        if include_l3:
            rec["L3_llm_review"] = None
        records.append(rec)

    table = fit_likelihood_table(records, generator)
    output_path.write_text(json.dumps(table, indent=2))
    records_path = output_path.with_name("train_critic_results.jsonl")
    write_jsonl(records_path, records)
    return table


def bayes_update(belief: float, theta: dict[str, float], passed: bool) -> float:
    p1 = theta["p_pass_y1"] if passed else 1.0 - theta["p_pass_y1"]
    p0 = theta["p_pass_y0"] if passed else 1.0 - theta["p_pass_y0"]
    den = p1 * belief + p0 * (1.0 - belief)
    return belief if den <= 1e-12 else (p1 * belief / den)


def kernel_update(belief: float, kernel: dict[str, float]) -> float:
    p_fix = float(kernel.get("p_fix_broken", kernel.get("P_fix_given_broken", 0.50)))
    p_break = float(kernel.get("p_break_correct", kernel.get("P_break_given_correct", 0.05)))
    return belief * (1.0 - p_break) + (1.0 - belief) * p_fix


def action_token(row: dict[str, Any]) -> str:
    action = str(row.get("action"))
    if row.get("skipped") is True:
        if action == "generate":
            return "generate_skipped"
        if action in CRITIC_MAP:
            return f"{CRITIC_MAP[action]}_skipped"
        if action in {"verify", "final_verify"}:
            return f"{action}_skipped"
        return f"{action}_skipped"
    if action == "generate":
        return "generate"
    if action in CRITIC_MAP:
        return f"{CRITIC_MAP[action]}{'+' if bool(row.get('passed')) else '-'}"
    if action in {"verify", "final_verify"}:
        return f"{action}{'+' if bool(row.get('passed')) else '-'}"
    if action == "think" and row.get("reasoning") == "decision parse failed":
        return "think_parse_failed"
    return action


def compact_path(tokens: list[str]) -> str:
    out = []
    for token in tokens:
        if out and out[-1][0] == token:
            out[-1][1] += 1
        else:
            out.append([token, 1])
    return " -> ".join(t if n == 1 else f"{t}x{n}" for t, n in out)


def spearman(xs: list[float], ys: list[int]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and math.isfinite(x)]
    if len(pairs) < 2:
        return None
    x, y = zip(*pairs)
    try:
        import pandas as pd

        return float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    except Exception:
        return None


def rejection_area(conf: list[float], quality: list[int], max_rejection: float) -> float | None:
    pairs = [
        (float(c), int(q))
        for c, q in zip(conf, quality)
        if c is not None and math.isfinite(float(c))
    ]
    if len(pairs) < 2:
        return None
    pairs.sort(key=lambda x: x[0])  # reject low-confidence first
    q = [v for _, v in pairs]
    n = len(q)
    max_k = min(n - 1, max(1, int(math.floor(max_rejection * n))))
    xs, ys = [], []
    for k in range(max_k + 1):
        xs.append(k / n)
        ys.append(sum(q[k:]) / (n - k))
    return sum((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2 for i in range(1, len(xs)))


def prr(conf: list[float], quality: list[int], max_rejection: float) -> float | None:
    area = rejection_area(conf, quality, max_rejection)
    oracle = rejection_area([float(q) for q in quality], quality, max_rejection)
    random = (sum(quality) / len(quality)) * (math.floor(max_rejection * len(quality)) / len(quality))
    if area is None or oracle is None or abs(oracle - random) < 1e-12:
        return None
    return (area - random) / (oracle - random)


def load_logprob_stats(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("instance_id")), int(row.get("generation_index", 0)))
            stat = logprob_stats(row)
            stat.update(
                {
                    "prompt_tokens": int(row.get("prompt_tokens") or 0),
                    "completion_tokens": int(row.get("completion_tokens") or 0),
                    "api_cost_usd": float(row.get("api_cost_usd") or 0.0),
                    "candidate_chars": len(row.get("code") or ""),
                    "raw_response_chars": len(row.get("raw_text") or ""),
                }
            )
            out[key] = stat
    return out


def simulate_instance(
    result: dict[str, Any],
    actions: list[dict[str, Any]],
    logprobs: dict[tuple[str, int], dict[str, Any]],
    *,
    generator: str,
    prior: float,
    theta: dict[str, dict[str, float]],
    kernel: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    iid = str(result["instance_id"])
    belief = float(prior)
    gen_idx = 0
    generation_trace = []
    before_final_label = None
    after_final_label = None
    tokens = []

    for row in sorted(actions, key=lambda r: int(r.get("step", 0))):
        action = str(row.get("action"))
        tokens.append(action_token(row))
        if action == "generate":
            if row.get("skipped") is True:
                continue
            before = belief
            belief = kernel_update(belief, kernel)
            stat = logprobs.get((iid, gen_idx), logprob_stats(None))
            generation_trace.append(
                {
                    "benchmark": result.get("benchmark"),
                    "split": "test",
                    "generator": generator,
                    "model_id": result.get("model_id"),
                    "policy": "llm_tool_agent",
                    "instance_id": iid,
                    "patch_idx": gen_idx,
                    "action_step": int(row.get("step", 0)),
                    "bayes_state_at_generation": before,
                    "bayes_state_after_generation": belief,
                    **stat,
                    "is_final_candidate": False,
                    "final_quality": None,
                    "final_bayes_state": None,
                    "final_action": "",
                }
            )
            gen_idx += 1
            continue

        critic = CRITIC_MAP.get(action)
        if critic and row.get("passed") in (True, False) and critic in theta:
            belief = bayes_update(belief, theta[critic], bool(row["passed"]))
            continue

        if action in {"verify", "final_verify"} and row.get("passed") in (True, False):
            before_final_label = belief
            if row.get("passed") is False and action == "verify":
                belief = 0.05
            after_final_label = belief

    if before_final_label is None:
        before_final_label = belief
    if after_final_label is None:
        after_final_label = belief

    if generation_trace:
        generation_trace[-1]["is_final_candidate"] = True
        generation_trace[-1]["final_quality"] = int(bool(result.get("fixed")))
        generation_trace[-1]["final_bayes_state"] = before_final_label
        generation_trace[-1]["final_action"] = result.get("final_action", "")
        final_stats = generation_trace[-1]
    else:
        final_stats = logprob_stats(None)

    final = {
        "benchmark": result.get("benchmark"),
        "split": "test",
        "generator": generator,
        "model_id": result.get("model_id"),
        "policy": "llm_tool_agent",
        "instance_id": iid,
        "llm_perplexity": final_stats.get("llm_perplexity"),
        "perplexity": final_stats.get("perplexity"),
        "llm_log_seq_prob": final_stats.get("llm_log_seq_prob"),
        "bayes_state": before_final_label,
        "bayes_state_before_final_label": before_final_label,
        "bayes_state_after_final_label": after_final_label,
        "bayes_state_after_generation": (
            generation_trace[-1]["bayes_state_after_generation"] if generation_trace else None
        ),
        "quality": int(bool(result.get("fixed"))),
        "final_action": result.get("final_action", ""),
        "label_source": result.get("final_action", ""),
        "n_logprob_tokens": final_stats.get("n_logprob_tokens", 0),
        "logprobs_supported": bool(final_stats.get("logprobs_supported")),
        "prompt_tokens": int(result.get("prompt_tokens") or 0),
        "completion_tokens": int(result.get("completion_tokens") or 0),
        "n_llm_calls": int(result.get("n_generations") or 0),
        "n_critic_runs": int(result.get("n_critic_runs") or 0),
        "n_full_tests": int(result.get("n_verifications") or 0),
        "wall_clock": float(result.get("wall_clock_s") or 0.0),
        "prior_Y1": prior,
        "actions": actions,
        "generation_trace": generation_trace,
    }
    return generation_trace, final, compact_path(tokens)


def build_metrics(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quality = [int(r["quality"]) for r in final_rows]
    specs = [
        ("llm_perplexity", True),
        ("perplexity", False),
        ("llm_log_seq_prob", True),
        ("bayes_state", True),
        ("bayes_state_after_generation", True),
    ]
    rows = []
    for name, higher_is_better in specs:
        raw = [finite_or_none(r.get(name)) for r in final_rows]
        conf = [x if (x is None or higher_is_better) else -x for x in raw]
        rows.append(
            {
                "score": name,
                "higher_is_better": higher_is_better,
                "spearman": spearman(conf, quality),
                "PRR": prr(conf, quality, 1.0),
                "PRR_05": prr(conf, quality, 0.5),
            }
        )
    return rows


def main() -> None:
    configure_hf_cache()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--benchmark", default="lcb_hard")
    p.add_argument("--generator", default="gpt_oss_120b_local")
    p.add_argument("--likelihoods", type=Path, default=None)
    p.add_argument("--kernel", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--platform", default="leetcode")
    p.add_argument("--lcb-version", default="all")
    p.add_argument("--private-test-cap", type=int, default=12)
    p.add_argument("--include-l3-train-likelihood", action="store_true")
    args = p.parse_args()

    stem = f"{args.benchmark}__{args.generator}"
    root = args.run_root
    out_dir = args.output_dir or root
    out_dir.mkdir(parents=True, exist_ok=True)

    results = read_jsonl(root / f"{stem}.jsonl")
    actions = [r for r in read_jsonl(root / f"{stem}.actions.jsonl") if r.get("split") == "test"]
    prior_rows = read_jsonl(root / f"{stem}.train_prior_calibration.jsonl")
    logprobs = load_logprob_stats(root / f"{stem}.generation_logprobs.jsonl")
    prior = prior_from_train(prior_rows)

    if args.likelihoods:
        table = json.loads(args.likelihoods.read_text())
        theta_source = str(args.likelihoods)
    else:
        likelihood_path = out_dir / "likelihood_tables.json"
        table = fit_train_likelihoods(
            benchmark=args.benchmark,
            generator=args.generator,
            split_path=root / f"{stem}.split.json",
            train_rows=prior_rows,
            output_path=likelihood_path,
            private_test_cap=args.private_test_cap,
            seed=args.seed,
            platform=args.platform,
            lcb_version=args.lcb_version,
            include_l3=args.include_l3_train_likelihood,
        )
        theta_source = str(likelihood_path)
    theta = normalize_theta(table)

    kernel = dict(DEFAULT_KERNEL)
    if args.kernel:
        raw_kernel = json.loads(args.kernel.read_text())
        kernel = raw_kernel.get("kernel_all", raw_kernel)

    actions_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in actions:
        actions_by_id[str(row.get("instance_id"))].append(row)

    final_rows, traj_rows, path_rows = [], [], []
    for result in results:
        iid = str(result["instance_id"])
        trace, final, path = simulate_instance(
            result,
            actions_by_id.get(iid, []),
            logprobs,
            generator=args.generator,
            prior=float(prior["prior_Y1"]),
            theta=theta,
            kernel=kernel,
        )
        final_rows.append(final)
        traj_rows.extend(trace)
        path_rows.append(
            {
                "instance_id": iid,
                "policy": "llm_tool_agent",
                "quality": final["quality"],
                "final_action": final["final_action"],
                "bayes_state": final["bayes_state"],
                "bayes_state_after_generation": final["bayes_state_after_generation"],
                "action_path": path,
            }
        )

    compact_rows = [
        {
            "benchmark": r["benchmark"],
            "split": "test",
            "generator": r["generator"],
            "model_id": r["model_id"],
            "policy": r["policy"],
            "instance_id": r["instance_id"],
            "llm_perplexity": r["llm_perplexity"],
            "llm_log_seq_prob": r["llm_log_seq_prob"],
            "bayes_state": r["bayes_state"],
            "quality": r["quality"],
            "bayes_state_after_generation": r["bayes_state_after_generation"],
            "perplexity": r["perplexity"],
            "final_action": r["final_action"],
        }
        for r in final_rows
    ]
    write_csv(out_dir / "final_logprob_bayes_quality.csv", compact_rows)
    write_jsonl(out_dir / "final_logprob_bayes_quality.jsonl", final_rows)
    write_csv(out_dir / "generation_trajectory_scores.csv", traj_rows)
    write_jsonl(out_dir / "generation_trajectory_scores.jsonl", traj_rows)

    grouped = defaultdict(list)
    for row in path_rows:
        grouped[row["action_path"]].append(row)
    path_table = []
    for i, (path, rows) in enumerate(
        sorted(grouped.items(), key=lambda kv: (-len(kv[1]), -sum(r["quality"] for r in kv[1]) / len(kv[1]))),
        start=1,
    ):
        path_table.append(
            {
                "path_id": f"P{i:02d}",
                "action_path": path,
                "n": len(rows),
                "quality_mean": sum(r["quality"] for r in rows) / len(rows),
                "final_actions": ", ".join(f"{k}:{v}" for k, v in Counter(r["final_action"] for r in rows).items()),
                "bayes_states": ", ".join(f"{x:.6f}" for x in sorted({r["bayes_state"] for r in rows if r["bayes_state"] is not None}, reverse=True)),
                "bayes_states_after_generation": ", ".join(f"{x:.6f}" for x in sorted({r["bayes_state_after_generation"] for r in rows if r["bayes_state_after_generation"] is not None}, reverse=True)),
                "examples": ", ".join(str(r["instance_id"]) for r in rows[:10]),
            }
        )
    write_csv(out_dir / "action_path_analysis.csv", path_table)

    metric_rows = build_metrics(final_rows)
    write_csv(out_dir / "metric_scores.csv", metric_rows)

    summary = {
        "run_root": str(root),
        "output_dir": str(out_dir),
        "n": len(final_rows),
        "quality_mean": sum(r["quality"] for r in final_rows) / len(final_rows) if final_rows else None,
        "final_actions": dict(Counter(r["final_action"] for r in final_rows)),
        "prior": prior,
        "theta_source": theta_source,
        "theta": theta,
        "kernel": kernel,
        "action_counts": dict(Counter(r.get("action") for r in actions)),
        "outputs": [
            "final_logprob_bayes_quality.csv",
            "final_logprob_bayes_quality.jsonl",
            "generation_trajectory_scores.csv",
            "generation_trajectory_scores.jsonl",
            "action_path_analysis.csv",
            "metric_scores.csv",
            "analysis_summary.json",
        ],
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
