#!/usr/bin/env python3
"""Cross-fitted belief_logit vs our fusion variants and honest baselines.

Extends scripts/compare_bayes_schemes.py (student branch) in three ways:
  * PRR is the analyser's (MC random baseline) at max_rejection 0.5 AND 1.0,
    so the numbers line up with readable/<bench>/metric_scores.csv;
  * our experiment2 fusion (DoubleBinaryBayes etc.) is cross-fitted over the
    SAME folds/seeds, into the honest bayes_state_after_generation prior;
  * the 3.5 GB logprob sidecar is parsed once and cached to a pickle.
"""
from __future__ import annotations

import _compat  # noqa: F401  # регистрирует code_uq.* / trajectory_uq_toolkit.*

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent / "askutsakov"
sys.path.insert(0, str(REPO / "src"))

from code_uq.analysis.bayes_trajectory import belief_logit, calibrate  # noqa: E402
from code_uq.analysis import uq_features as UF  # noqa: E402
from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import prr  # noqa: E402
from trajectory_uq_toolkit.bayes import BinaryBayes, DoubleBinaryBayes, logit, sigmoid  # noqa: E402

STEP_FEATURES = ["ntok", "sum", "answer_sum"]


def read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def cached_steps(root: Path, stem: str, cache_dir: Path) -> dict:
    cache = cache_dir / f"steps__{root.name}__{stem}.pkl"
    if cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)
    steps = UF.per_generation_metrics(root, stem)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as fh:
        pickle.dump(steps, fh)
    return steps


def episode_series(ep: dict, steps: dict) -> dict[str, list[float]]:
    """Per-generation series for one episode, in trajectory order."""
    inst = str(ep["instance_id"])
    series: dict[str, list[float]] = {}
    index = 0
    for step in ep.get("trajectory") or []:
        if step.get("action") == "generate" and not step.get("skipped"):
            for key, value in steps.get((inst, index), {}).items():
                series.setdefault(key, []).append(value)
            index += 1
    return series


def folds(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    order = np.random.RandomState(seed).permutation(n)
    return [order[i::n_folds] for i in range(n_folds)]


def cross_fitted(fit_predict, n: int, n_folds: int, seed: int) -> np.ndarray:
    """fit_predict(train_idx, held_idx) -> scores for held_idx."""
    out = np.full(n, np.nan)
    parts = folds(n, n_folds, seed)
    for k, held in enumerate(parts):
        if len(held) == 0:
            continue
        train = np.concatenate([p for j, p in enumerate(parts) if j != k])
        out[held] = fit_predict(train, held)
    return out


def published(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[str(row["instance_id"])] = row
    return out


def col(rows: dict[str, dict], instances: list[str], name: str) -> np.ndarray:
    vals = []
    for inst in instances:
        raw = (rows.get(inst) or {}).get(name)
        try:
            vals.append(float(raw))
        except (TypeError, ValueError):
            vals.append(np.nan)
    return np.asarray(vals, dtype=float)


def score(values, labels, keep) -> tuple[float | None, float | None]:
    ok = [i for i in keep if np.isfinite(values[i])]
    if len(ok) < 2:
        return None, None
    conf = [float(values[i]) for i in ok]
    qual = [int(labels[i]) for i in ok]
    return prr(conf, qual, 0.5), prr(conf, qual, 1.0)


def fmt(pair) -> str:
    return "   —  /   —  " if pair[0] is None else f"{pair[0]:+.3f} / {pair[1]:+.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(__file__).resolve().parent / "cache")
    args = ap.parse_args()

    root, stem = args.run_root, f"{args.benchmark}__{args.generator}"
    episodes = read_jsonl(root / f"{stem}.jsonl")
    if not episodes:
        raise SystemExit(f"no episodes in {root / (stem + '.jsonl')}")

    steps = cached_steps(root, stem, args.cache_dir)
    cap = max((int(e.get("n_steps") or 0) for e in episodes), default=0)
    rows = [UF.episode_features(e, steps, cap) for e in episodes]
    series = [episode_series(e, steps) for e in episodes]
    labels = np.array([int(bool(e["fixed"])) for e in episodes])
    instances = [str(e["instance_id"]) for e in episodes]
    n = len(labels)

    table = published(root / "readable" / args.benchmark / "final_logprob_bayes_quality.csv")
    keep = [i for i, inst in enumerate(instances) if inst in table] or list(range(n))
    if len(set(labels[keep].tolist())) < 2:
        raise SystemExit("degenerate label")

    print(f"run: {root.name}  benchmark: {args.benchmark}  "
          f"episodes: {len(keep)}  pass@1: {float(labels[keep].mean()):.3f}  "
          f"errors: {int((1 - labels[keep]).sum())}")
    print(f"cross-fit: {args.folds} folds x {args.seeds} seeds\n")
    print(f"{'signal':<44} {'PRR@0.5 / PRR@1.0':>18}")
    print("-" * 64)

    # --- published, per-instance, no fitting -----------------------------
    for name in ("bayes_state", "bayes_state_after_generation", "tool_success",
                 "verbalized_2s_confidence", "llm_log_seq_prob", "llm_perplexity"):
        print(f"{name:<44} {fmt(score(col(table, instances, name), labels, keep)):>18}")

    # --- honest per-generation aggregations ------------------------------
    for feat in ("sum:last", "sum:mean", "sum:min", "ntok:mean", "answer_sum:min"):
        values = np.array([r.get(feat, np.nan) for r in rows], dtype=float)
        print(f"{feat:<44} {fmt(score(values, labels, keep)):>18}")

    print("-" * 64)

    # --- cross-fitted belief_logit (student's corrected scheme) ----------
    def student(train, held):
        cal = calibrate([rows[int(i)] for i in train], labels[train],
                        [series[int(i)] for i in train], STEP_FEATURES)
        return [belief_logit(cal, rows[int(i)], series[int(i)], STEP_FEATURES)
                for i in held]

    # --- cross-fitted fusion into the honest belief ----------------------
    prior_col = col(table, instances, "bayes_state_after_generation")

    def fusion(mode: str, feature: str):
        def fit_predict(train, held):
            seqs = [series[int(i)].get(feature, []) for i in train]
            labs = [int(labels[int(i)]) for i in train]
            usable = [(s, l) for s, l in zip(seqs, labs) if s]
            if len({l for _, l in usable}) < 2:
                return [np.nan] * len(held)
            cls = DoubleBinaryBayes if mode == "double" else BinaryBayes
            kwargs = {} if mode == "double" else {"mode": mode}
            model = cls.fit([s for s, _ in usable], [l for _, l in usable],
                            higher_is_uncertain=False, **kwargs)
            out = []
            for i in held:
                start = prior_col[int(i)]
                seq = series[int(i)].get(feature, [])
                if not seq or not np.isfinite(start):
                    out.append(np.nan)
                else:
                    out.append(model.predict(seq, start=float(np.clip(start, 1e-6, 1 - 1e-6))))
            return out
        return fit_predict

    runs: dict[str, list[tuple[float, float]]] = {}
    for seed in range(args.seeds):
        for name, fp in (("belief_logit (cross-fit)", student),
                         ("fusion double -> honest belief", fusion("double", "sum")),
                         ("fusion lr_neg -> honest belief", fusion("lr_neg", "sum")),
                         ("fusion sep    -> honest belief", fusion("sep", "sum"))):
            values = cross_fitted(fp, n, args.folds, seed)
            pair = score(values, labels, keep)
            if pair[0] is not None:
                runs.setdefault(name, []).append(pair)

    for name, pairs in runs.items():
        a = float(np.mean([p[0] for p in pairs]))
        b = float(np.mean([p[1] for p in pairs]))
        sa = float(np.std([p[0] for p in pairs]))
        sb = float(np.std([p[1] for p in pairs]))
        print(f"{name:<44} {a:+.3f} / {b:+.3f}   (sd {sa:.3f} / {sb:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
