#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from types import SimpleNamespace
from typing import Dict, List, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs enhanced (and optional LangGraph) on benchmarks."
    )
    parser.add_argument(
        "--benchmark",
        choices=["humaneval", "mbpp", "gsm8k", "hotpotqa", "all"],
        default="all",
        help="Which benchmark(s) to run.",
    )
    parser.add_argument("--limit", type=int, default=30, help="Limit examples per benchmark.")
    parser.add_argument("--print-each", action="store_true", help="Print each example.")
    parser.add_argument("--model", default="qwen/qwen-2.5-7b-instruct", help="Model name.")
    parser.add_argument("--service-url", default="http://localhost:8001/v1", help="TTS service URL.")
    parser.add_argument("--tts-budget", type=int, default=8, help="TTS budget.")
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama instead of TTS.")
    parser.add_argument("--ollama-model", default="qwen2.5:7b-instruct", help="Ollama model name.")
    parser.add_argument("--uncertainty-threshold", type=float, default=0.5, help="Uncertainty threshold.")
    parser.add_argument("--include-langgraph", action="store_true", help="Also run LangGraph code pipeline.")
    return parser.parse_args()


def _build_eval_args(base: argparse.Namespace, **overrides) -> argparse.Namespace:
    data = asdict(base) if hasattr(base, "__dataclass_fields__") else vars(base).copy()
    data.update(overrides)
    return SimpleNamespace(**data)


def _format_row(name: str, result) -> List[str]:
    return [
        name,
        f"{result.accuracy:.4f}",
        f"{result.avg_uncertainty:.4f}",
        f"{result.confident_accuracy:.4f}",
        f"{result.abstention_rate:.4f}",
        f"{result.ece:.4f}",
    ]


def _print_table(title: str, rows: List[List[str]]) -> None:
    headers = ["Run", "Acc", "AvgUnc", "ConfAcc", "Abstain", "ECE"]
    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    print("\n" + title)
    print(" | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))))


def main() -> None:
    args = _parse_args()

    import run_multi_benchmark_eval as base_eval
    import run_multi_benchmark_langgraph_eval as lg_eval

    benchmarks: Dict[str, Tuple] = {
        "humaneval": (base_eval.evaluate_humaneval, lg_eval.evaluate_humaneval),
        "mbpp": (base_eval.evaluate_mbpp, lg_eval.evaluate_mbpp),
        "gsm8k": (base_eval.evaluate_gsm8k, lg_eval.evaluate_gsm8k),
        "hotpotqa": (base_eval.evaluate_hotpotqa, lg_eval.evaluate_hotpotqa),
    }

    selected = benchmarks.keys() if args.benchmark == "all" else [args.benchmark]

    base_args = _build_eval_args(
        args,
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        service_url=args.service_url,
        tts_budget=args.tts_budget,
        uncertainty_threshold=args.uncertainty_threshold,
        print_each=args.print_each,
        saup_samples=5,
        reflexion_max_attempts=2,
    )

    # Baseline: disable all enhancements (closest to initial)
    baseline_args = _build_eval_args(
        base_args,
        disable_saup=True,
        disable_cot=True,
        disable_reflexion=True,
        disable_propagation=True,
    )

    # Enhanced: default (all enabled)
    enhanced_args = _build_eval_args(
        base_args,
        disable_saup=False,
        disable_cot=False,
        disable_reflexion=False,
        disable_propagation=False,
    )

    llm = base_eval.create_llm_client(base_args)
    if args.include_langgraph:
        lg_llm = lg_eval.create_llm_client(base_args)
    else:
        lg_llm = None

    for name in selected:
        base_fn, lg_fn = benchmarks[name]
        rows = []

        baseline_result = base_fn(baseline_args, llm)
        rows.append(_format_row("baseline", baseline_result))

        enhanced_result = base_fn(enhanced_args, llm)
        rows.append(_format_row("v2_enhanced", enhanced_result))

        if lg_llm is not None:
            lg_result = lg_fn(base_args, lg_llm)
            rows.append(_format_row("langgraph", lg_result))

        _print_table(f"{name.upper()} Results", rows)


if __name__ == "__main__":
    main()
