#!/usr/bin/env python3
"""Run all binary and continuous UQ fusion methods over local readable exports."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import experiment2_uq_bayes_critic as fusion
from analyze_lcb_llm_tool_agent_logs import prr


METHODS = ("sep", "lr_pos", "lr_neg", "double", "continuous", "tempered")


def _fused_probabilities(rows, method, *, higher_better, k, seed, lambda_):
    if method == "continuous":
        predictions = fusion.kfold_continuous_fuse(
            rows, k=k, seed=seed, lambda_=1.0
        )
    elif method == "tempered":
        predictions = fusion.kfold_continuous_fuse(
            rows, k=k, seed=seed, lambda_=lambda_
        )
    else:
        predictions = fusion.kfold_fuse(
            rows, higher_better, k=k, seed=seed, mode=method
        )
    return [predictions[row["iid"]] for row in rows]


def evaluate_directory(
    readable_dir: Path,
    *,
    feature: str,
    aggregations: list[str],
    k: int,
    seed: int,
    n_boot: int,
    max_rej: float,
    lambda_: float,
) -> list[dict]:
    final_path = readable_dir / "final_logprob_bayes_quality.csv"
    trajectory_path = readable_dir / "generation_trajectory_scores.jsonl"
    with final_path.open() as handle:
        first = next(csv.DictReader(handle), {})
    dataset = str(first.get("benchmark") or readable_dir.name)
    generator = str(first.get("generator") or "unknown")
    run = readable_dir.parents[1].name
    output = []

    for aggregation in aggregations:
        if aggregation == "final":
            overrides = None
        else:
            if not trajectory_path.exists():
                continue
            overrides = fusion.load_trajectory_feature(
                trajectory_path, feature, aggregation
            )
        rows = fusion.load_final(final_path, feature, overrides)
        quality = [row["quality"] for row in rows]
        if len(rows) < 4 or len(set(quality)) < 2:
            continue
        ids = [row["iid"] for row in rows]
        baseline = [row["bayes"] for row in rows]
        baseline_prr = prr(baseline, quality, max_rej)
        higher_better = fusion.HIGHER_BETTER[feature]
        raw_confidence = [
            row["feat_raw"] if higher_better else -row["feat_raw"] for row in rows
        ]
        feature_prr = prr(raw_confidence, quality, max_rej)
        baseline_metrics = fusion.probability_metrics(quality, baseline)
        output.append(
            {
                "dataset": dataset,
                "generator": generator,
                "run": run,
                "source": str(readable_dir),
                "feature": feature,
                "aggregation": aggregation,
                "method": "bayes_state",
                "n": len(rows),
                "n_success": sum(quality),
                "feature_prr": feature_prr,
                "feature_prr_at_0_5": feature_prr,
                "prr": baseline_prr,
                "prr_at_0_5": baseline_prr,
                "delta_prr_vs_bayes": 0.0,
                "delta_prr_at_0_5_vs_bayes": 0.0,
                "ci_low": None,
                "ci_high": None,
                **baseline_metrics,
            }
        )
        for method in METHODS:
            predictions = _fused_probabilities(
                rows,
                method,
                higher_better=higher_better,
                k=k,
                seed=seed,
                lambda_=lambda_,
            )
            method_prr = prr(predictions, quality, max_rej)
            ci = fusion.paired_diff_ci(
                predictions,
                baseline,
                quality,
                n_boot=n_boot,
                max_rej=max_rej,
                seed=seed,
            )
            metrics = fusion.probability_metrics(quality, predictions)
            output.append(
                {
                    "dataset": dataset,
                    "generator": generator,
                    "run": run,
                    "source": str(readable_dir),
                    "feature": feature,
                    "aggregation": aggregation,
                    "method": method,
                    "n": len(rows),
                    "n_success": sum(quality),
                    "feature_prr": feature_prr,
                    "feature_prr_at_0_5": feature_prr,
                    "prr": method_prr,
                    "prr_at_0_5": method_prr,
                    "delta_prr_vs_bayes": method_prr - baseline_prr,
                    "delta_prr_at_0_5_vs_bayes": method_prr - baseline_prr,
                    "ci_low": ci[0] if ci else None,
                    "ci_high": ci[1] if ci else None,
                    **metrics,
                }
            )
    return output


def write_report(rows: list[dict], path: Path) -> None:
    method_rows = [row for row in rows if row["method"] != "bayes_state"]
    baselines = {
        (row["source"], row["aggregation"]): row
        for row in rows
        if row["method"] == "bayes_state"
    }
    summaries = []
    for method in METHODS:
        selected = [row for row in method_rows if row["method"] == method]
        if selected:
            summaries.append(
                (
                    method,
                    statistics.fmean(row["prr_at_0_5"] for row in selected),
                    statistics.fmean(row["delta_prr_vs_bayes"] for row in selected),
                    sum(
                        row["ci_low"] is not None and row["ci_low"] > 0
                        for row in selected
                    ),
                    sum(
                        row["ci_high"] is not None and row["ci_high"] < 0
                        for row in selected
                    ),
                    statistics.fmean(
                        row["brier"]
                        - baselines[(row["source"], row["aggregation"])]["brier"]
                        for row in selected
                    ),
                    statistics.fmean(
                        row["nll"]
                        - baselines[(row["source"], row["aggregation"])]["nll"]
                        for row in selected
                    ),
                )
            )

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in method_rows:
        key = (row["source"], row["dataset"], row["aggregation"])
        groups.setdefault(key, []).append(row)

    lines = [
        "# UQ Bayes fusion transfer grid",
        "",
        "All fused predictions are out-of-fold. Delta is PRR@0.5 minus the existing "
        "`bayes_state` on the same instances.",
        "",
        "## Method averages",
        "",
        "| Method | Mean PRR@0.5 | Mean delta PRR@0.5 | Mean delta Brier | Mean delta NLL | Significant wins | Significant losses |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, mean_prr, delta, wins, losses, delta_brier, delta_nll in summaries:
        lines.append(
            f"| {method} | {mean_prr:.3f} | {delta:+.3f} | {delta_brier:+.3f} | "
            f"{delta_nll:+.3f} | {wins} | {losses} |"
        )

    lines.extend(
        [
            "",
            "## Best method per dataset and aggregation",
            "",
            "| Run | Dataset | Generator | Aggregation | n | Baseline PRR@0.5 | Best method | Best PRR@0.5 | Delta PRR@0.5 | 95% CI |",
            "|---|---|---|---|---:|---:|---|---:|---:|---|",
        ]
    )
    for (_, dataset, aggregation), candidates in sorted(groups.items()):
        best = max(candidates, key=lambda row: row["delta_prr_vs_bayes"])
        baseline = next(
            row
            for row in rows
            if row["source"] == best["source"]
            and row["aggregation"] == aggregation
            and row["method"] == "bayes_state"
        )
        ci = (
            "-"
            if best["ci_low"] is None
            else f"[{best['ci_low']:+.3f}, {best['ci_high']:+.3f}]"
        )
        lines.append(
            f"| {best['run']} | {dataset} | {best['generator']} | {aggregation} | {best['n']} | "
            f"{baseline['prr']:.3f} | {best['method']} | "
            f"{best['prr_at_0_5']:.3f} | {best['delta_prr_vs_bayes']:+.3f} | {ci} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature", choices=list(fusion.HIGHER_BETTER), default="llm_log_seq_prob")
    parser.add_argument(
        "--aggregations",
        nargs="+",
        choices=["final", *fusion.AGGREGATORS],
        default=["final", "mean", "min", "max"],
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--max-rej", type=float, default=0.5)
    parser.add_argument("--tempered-lambda", type=float, default=0.25)
    parser.add_argument("--min-class-count", type=int, default=3)
    args = parser.parse_args()

    rows = []
    for final_path in sorted(args.root.rglob("final_logprob_bayes_quality.csv")):
        with final_path.open() as handle:
            quality = []
            for row in csv.DictReader(handle):
                try:
                    quality.append(int(row["quality"]))
                except (KeyError, TypeError, ValueError):
                    continue
        if not quality or min(sum(quality), len(quality) - sum(quality)) < args.min_class_count:
            continue
        rows.extend(
            evaluate_directory(
                final_path.parent,
                feature=args.feature,
                aggregations=args.aggregations,
                k=args.k,
                seed=args.seed,
                n_boot=args.n_boot,
                max_rej=args.max_rej,
                lambda_=args.tempered_lambda,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("no eligible datasets found")
    csv_path = args.output_dir / "fusion_grid.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(rows, args.output_dir / "report.md")
    print(f"wrote rows={len(rows)} datasets={len(set(row['source'] for row in rows))}")


if __name__ == "__main__":
    main()
