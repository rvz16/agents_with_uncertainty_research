"""Confidence intervals over episodes, and over the contrasts we actually claim.

Every table in this work reports the spread over three seeds, which is the
narrowest honest number available: it says how much the fold assignment moves
the estimate, not how much another 97 episodes would. With 15 failures in the
ReAct Qwen cohort and 12 in the smolagents Qwen cohort, the second question is
the one a reader will ask, so this answers it by resampling episodes.

Two design choices matter. The out-of-fold prediction for each episode is
computed once, averaged over the seeds, and then held fixed while the episode
set is resampled; the interval is therefore about the sample of episodes, which
is what it claims to be. And every row is resampled on the *same* draws, within
cohort, so a difference between two rows can be given an interval too -- which
is the only way to ask whether the backbone really beats its own decision model
rather than beating it on this particular 86 episodes.

Rows come from two places: an episode-level answer file (a decision model, a
backbone, the fine-tuned encoder) or our own estimators through the method
table. Both are reduced to one confidence per episode before any resampling, so
they are treated identically from there on.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.decision_model_score import ece, extract
from experiments.prr_report_v2 import (SEEDS, SIGNALS, is_continuous, load_alfworld, method_table,
                                       prr, stratified_folds, with_labels)

RUNS = {"RG": "react_gptoss_140_giveup_sc_finished_nolast", "SG": "smol_gptoss_140_sc_finished_nolast",
        "RQ": "react_qwen_140_giveup_sc_finished_nolast", "SQ": "smol_qwen_140_sc_finished_nolast"}
COHORTS = tuple(RUNS)
#: per-step verdicts injected as signals, for the fusion rows
FUSED = {"Decision p(success)": ("judge_Cloudflare_clef-flash", "p", False),
         "Decision progress": ("judge_Cloudflare_clef-flash", "progress", False)}


def oof_pred(eps: dict, fn, seed: int) -> dict[str, float] | None:
    """The out-of-fold confidence per episode -- ``oof`` without the metric."""
    continuous = is_continuous(eps)
    pred: dict[str, float] = {}
    for fold in stratified_folds(eps, seed):
        held = set(fold)
        train = with_labels([eps[i] for i in eps if i not in held], continuous)
        for i, c in zip(fold, fn(train, [eps[i] for i in fold])):
            if c is None:
                return None
            pred[i] = float(c)
    return pred


def read_answers(path: Path, question: str = "success") -> dict[str, dict[str, tuple[float, float]]]:
    """cohort -> id -> (score, probability)."""
    out: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        value = extract(row["answer"], question)
        if value is not None:
            out[row["cohort"]][row["id"]] = (float(row["score"]), float(value))
    return out


def read_steps(path: Path, field: str) -> dict[str, list[float]]:
    out: dict[str, list[tuple[int, float]]] = defaultdict(list)
    if not path.exists():
        return {}
    for line in open(path):
        if line.strip():
            row = json.loads(line)
            if row.get(field) is not None:
                out[row["id"]].append((int(row["step"]), float(row[field])))
    return {k: [v for _, v in sorted(pairs)] for k, pairs in out.items()}


def interval(values: list[float], lo: float = 2.5, hi: float = 97.5) -> tuple[float, float, float]:
    clean = [v for v in values if v == v]
    if not clean:
        return (float("nan"),) * 3
    return st.fmean(clean), float(np.percentile(clean, lo)), float(np.percentile(clean, hi))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--answers-dir", type=Path, required=True)
    p.add_argument("--judge-dir", type=Path, required=True)
    p.add_argument("--encoder-dir", type=Path, required=True)
    p.add_argument("--toolkit", required=True)
    p.add_argument("--step-rule", default="giveup+acted")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--metric", choices=["prr", "ece"], default="prr")
    a = p.parse_args()

    episodes = {}
    for cohort, run in RUNS.items():
        eps = load_alfworld(a.runs / run, "react" if "react" in run else "smolagents", a.step_rule)
        episodes[cohort] = {k: v for k, v in eps.items() if v["n_steps"]}

    for label, (_, _, hiu) in FUSED.items():
        SIGNALS[label] = (label, None, hiu)
    methods = method_table(a.toolkit)

    # one confidence per episode, per row
    rows: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(dict)

    def add_answers(name: str, filename: str, question: str = "success") -> None:
        candidates = [a.answers_dir / filename, *sorted(a.answers_dir.glob(f"*/{filename}"))]
        path = next((c for c in candidates if c.exists()), candidates[0])
        if not path.exists():
            print(f"[ci] missing {path.name}, skipped", flush=True)
            return
        data = read_answers(path, question)
        for cohort in COHORTS:
            if data.get(cohort):  # a model's answer does not depend on the fold seed
                rows[name][cohort] = [{i: v for i, (_, v) in data[cohort].items()}]

    def add_encoder(name: str) -> None:
        for cohort in COHORTS:
            path = a.encoder_dir / f"answers_modernbert_last_{cohort}.jsonl"
            if path.exists():
                data = read_answers(path)
                rows[name][cohort] = [{i: v for i, (_, v) in data.get(cohort, {}).items()}]

    def add_ours(name: str, key) -> None:
        for cohort in COHORTS:
            fn = methods.get(key)
            if fn is None:
                print(f"[ci] no method {key}", flush=True)
                return
            per_seed = [p for s in SEEDS if (p := oof_pred(episodes[cohort], fn, s)) is not None]
            if not per_seed:
                continue
            rows[name][cohort] = per_seed
        print(f"[ci] {name} done", flush=True)

    add_answers("Clef 27B, p(success)", "answers_Cloudflare_clef.jsonl")
    add_answers("Clef-flash 9B, p(success)", "answers_Cloudflare_clef-flash.jsonl")
    add_answers("Jeeves 9B, p(success)", "answers_think.jsonl")
    add_answers("Qwen3.5-9B, P(yes)", "answers_Qwen_Qwen3_5-9B_token.jsonl")
    add_answers("Qwen3.8-27B, P(yes)", "answers_Qwen_Qwen3_8-27B_token.jsonl")
    add_encoder("ModernBERT 149M")
    add_answers("Logistic on frozen Qwen3.5-9B", "answers_schema_logreg_Qwen3.5-9B.jsonl")
    add_answers("Our head on frozen Qwen3.5-9B", "answers_schema_schema_Qwen3.5-9B.jsonl")
    add_answers("Control: permuted labels", "answers_schema_schema_shuffled_Qwen3.5-9B.jsonl")
    # Decision 3.0 against the Qwen each release is built on -- the pairing is exact
    for label, filename in (("d3-lite 0.85B", "answers_d3_vllm-sr_d3-lite_clean.jsonl"),
                            ("d3-nano 2.2B", "answers_d3_vllm-sr_d3-nano_clean.jsonl"),
                            ("d3-mini 4.5B", "answers_d3_vllm-sr_d3-mini_clean.jsonl"),
                            ("d3-flash 8.4B", "answers_d3_vllm-sr_d3-flash_clean.jsonl"),
                            ("d3 26B", "answers_d3_vllm-sr_d3_clean.jsonl"),
                            ("Qwen3.5-0.8B", "answers_Qwen_Qwen3_5-0_8B_token.jsonl"),
                            ("Qwen3.5-2B", "answers_Qwen_Qwen3_5-2B_token.jsonl"),
                            ("Qwen3.5-4B", "answers_Qwen_Qwen3_5-4B_token.jsonl")):
        add_answers(label, filename)

    if a.metric == "prr":
        for cohort in COHORTS:
            for label, (stem, field, _) in FUSED.items():
                sequences = read_steps(a.judge_dir / f"{stem}_{cohort}.jsonl", field)
                for key, e in episodes[cohort].items():
                    e["signals"][label] = (sequences.get(key) or [])[: e["n_steps"]]
        add_ours("Temporal belief B4 (ours)", ("Reference", "TemporalBelief (B4)", "final checkpoint"))
        add_ours("Tool-only, tempered (ours)", ("Reference", "Bayes tool-only", "Step critics, tempered"))
        add_ours("Judge fusion (ours)",
                 ("LLM judge", "LLM judge \\ensuremath{-} Bayes UQ + tools", "Last only"))
        add_ours("N steps (reference)", ("Reference", "N steps", "\\ensuremath{-}N"))
        add_ours("Fused: Decision p + tools (ours)",
                 ("Decision p(success)", "Decision p(success) \\ensuremath{-} Bayes UQ + tools", "Last only"))

    present = [name for name in rows if len(rows[name]) == len(COHORTS)]
    print(f"\n[ci] rows with all four cohorts: {len(present)}")
    shared = {}
    for cohort in COHORTS:
        ids = set.intersection(*(set(p) for name in present for p in rows[name][cohort]))
        shared[cohort] = sorted(ids)
        print(f"[ci] {cohort}: {len(ids)} episodes common to every row "
              f"(of {len(episodes[cohort])})")

    scores = {c: np.array([episodes[c][i]["score"] for i in shared[c]]) for c in COHORTS}
    conf = {name: {c: [np.array([p[i] for i in shared[c]]) for p in rows[name][c]] for c in COHORTS}
            for name in present}

    def metric(y: np.ndarray, seeds: list[np.ndarray]) -> float:
        """Averaged over the seeds, the way every table in this work reports it."""
        values = []
        for x in seeds:
            if a.metric == "ece":
                values.append(ece(y.tolist(), x.tolist()))
                continue
            if len(set(y.tolist())) < 2:
                return float("nan")
            value = prr(y.tolist(), x.tolist())
            if value is None:
                return float("nan")
            values.append(value)
        return st.fmean(values) if values else float("nan")

    rng = np.random.RandomState(0)
    draws = {name: {c: [] for c in COHORTS} for name in present}
    avg = {name: [] for name in present}
    for _ in range(a.draws):
        picks = {c: rng.randint(0, len(shared[c]), len(shared[c])) for c in COHORTS}
        for name in present:
            per_cohort = []
            for c in COHORTS:
                v = metric(scores[c][picks[c]], [x[picks[c]] for x in conf[name][c]])
                draws[name][c].append(v)
                per_cohort.append(v)
            avg[name].append(st.fmean(per_cohort) if all(v == v for v in per_cohort) else float("nan"))

    title = "PRR@0.5" if a.metric == "prr" else "ECE"
    print(f"\n{title}, percentile bootstrap over episodes, {a.draws} draws\n")
    print(f"{'row':34} " + " ".join(f"{c:>18}" for c in COHORTS) + f" {'Avg':>18}")
    print("-" * 132)
    for name in present:
        cells = []
        for c in COHORTS:
            m, lo, hi = interval(draws[name][c])
            cells.append(f"{m:+5.2f} [{lo:+5.2f},{hi:+5.2f}]")
        m, lo, hi = interval(avg[name])
        print(f"{name:34} " + " ".join(f"{s:>18}" for s in cells)
              + f" {f'{m:+5.2f} [{lo:+5.2f},{hi:+5.2f}]':>18}")

    contrasts = [("d3-lite 0.85B", "Qwen3.5-0.8B"),
                 ("d3-nano 2.2B", "Qwen3.5-2B"),
                 ("d3-mini 4.5B", "Qwen3.5-4B"),
                 ("d3-flash 8.4B", "Qwen3.5-9B, P(yes)"),
                 ("d3 26B", "Qwen3.8-27B, P(yes)"),
                 ("d3 26B", "Clef 27B, p(success)"),
                 ("Our head on frozen Qwen3.5-9B", "Clef-flash 9B, p(success)"),
                 ("Our head on frozen Qwen3.5-9B", "Qwen3.5-9B, P(yes)"),
                 ("Our head on frozen Qwen3.5-9B", "ModernBERT 149M"),
                 ("Our head on frozen Qwen3.5-9B", "Control: permuted labels"),
                 ("Logistic on frozen Qwen3.5-9B", "Our head on frozen Qwen3.5-9B"),
                 ("Qwen3.5-9B, P(yes)", "Clef-flash 9B, p(success)"),
                 ("Qwen3.8-27B, P(yes)", "Clef 27B, p(success)"),
                 ("ModernBERT 149M", "Clef-flash 9B, p(success)"),
                 ("ModernBERT 149M", "Clef 27B, p(success)"),
                 ("Fused: Decision p + tools (ours)", "Temporal belief B4 (ours)"),
                 ("Fused: Decision p + tools (ours)", "Judge fusion (ours)"),
                 ("Clef-flash 9B, p(success)", "N steps (reference)"),
                 ("Temporal belief B4 (ours)", "N steps (reference)"),
                 ("Fused: Decision p + tools (ours)", "Clef-flash 9B, p(success)")]
    print(f"\nPaired differences on the same draws (positive favours the first row)\n")
    print(f"{'contrast':62} {'Avg difference':>26} {'excludes 0':>11}")
    print("-" * 102)
    for left, right in contrasts:
        if left not in avg or right not in avg:
            continue
        diff = [x - y for x, y in zip(avg[left], avg[right]) if x == x and y == y]
        m, lo, hi = interval(diff)
        print(f"{left + '  -  ' + right:62} {f'{m:+5.2f} [{lo:+5.2f}, {hi:+5.2f}]':>26} "
              f"{'yes' if lo > 0 or hi < 0 else 'NO':>11}")


if __name__ == "__main__":
    main()
