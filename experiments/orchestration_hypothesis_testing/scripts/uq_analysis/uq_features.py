"""Episode features shared by every benchmark.

Collects everything known **before the oracle is called**: critic verdicts,
per-generation scores under several aggregations, trajectory shape and the
verbalised confidence. One function for all benchmarks, because otherwise the
benchmarks cannot be compared against each other.

Sign convention: larger means more confident. Length and step counts enter
negated, as does entropy.

Deliberately absent: anything computed from the label or from the oracle call.
`verify` and `final_verify` never become features.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

CRITICS = ("critic_L0", "critic_L1", "critic_L2", "critic_L3")

#: Per-step metrics and aggregations kept after the aggregation sweep: `first`
#: and `ewma 0.9` rank best on average, `min` is best for the answer-channel
#: logprob sum, `mean` is a stable middle.
STEP_METRICS = ("answer_sum", "sum", "ntok", "answer_mean", "answer_entropy",
                "answer_self_cert", "reasoning_sum", "reasoning_mean", "entropy",
                "self_cert", "min")
STEP_AGGS = ("first", "min", "mean", "ewma0.9", "last")


def _read_jsonl(path: Path) -> list[dict]:
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


def _last_verdicts(ep: dict) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for step in ep.get("trajectory") or []:
        if step.get("action") in CRITICS and not step.get("skipped") \
                and isinstance(step.get("passed"), bool):
            out[str(step["action"])] = bool(step["passed"])
    return out


def _all_verdicts(ep: dict) -> list[tuple[str, bool]]:
    return [(str(s["action"]), bool(s["passed"])) for s in ep.get("trajectory") or []
            if s.get("action") in CRITICS and not s.get("skipped")
            and isinstance(s.get("passed"), bool)]


def per_generation_metrics(run_root: Path, stem: str, with_topk: bool = False
                           ) -> dict[tuple[str, int], dict]:
    """Per-generation scores read from the logprob sidecar.

    `with_topk` also computes entropy and self-certainty. They are derived from
    twenty top-logprobs per token, which costs hours over the whole pool, and
    they never reached the top of the aggregation sweep -- hence off by default.
    """
    from code_uq.analysis.entropy_kl_trajectory import (
        token_entropy_kl, token_self_certainty)
    from code_uq.common.logprobs import split_channels

    out: dict[tuple[str, int], dict[str, float]] = {}
    for rec in _read_jsonl(run_root / f"{stem}.generation_logprobs.jsonl"):
        content = ((rec.get("logprobs") or {}) or {}).get("content")
        if not content:
            continue
        feats: dict[str, float] = {}
        for span, tokens in split_channels(content).items():
            vals, ents, certs = [], [], []
            for tok in tokens:
                if not isinstance(tok, dict):
                    continue
                lp = tok.get("logprob")
                if isinstance(lp, (int, float)) and lp > -9998:
                    vals.append(float(lp))
                if with_topk:
                    pair = token_entropy_kl(tok.get("top_logprobs") or [])
                    if pair:
                        ents.append(pair[0])
                    cert = token_self_certainty(tok.get("top_logprobs") or [])
                    if cert is not None:
                        certs.append(cert)
            if not vals:
                continue
            prefix = "" if span == "all" else f"{span}_"
            arr = np.asarray(vals)
            feats[f"{prefix}sum"] = float(arr.sum())
            feats[f"{prefix}mean"] = float(arr.mean())
            feats[f"{prefix}min"] = float(arr.min())
            feats[f"{prefix}ntok"] = -float(len(arr))
            if ents:
                feats[f"{prefix}entropy"] = -float(np.mean(ents))
            if certs:
                feats[f"{prefix}self_cert"] = float(np.mean(certs))
        if feats:
            out[(str(rec.get("instance_id")), int(rec.get("generation_index", 0)))] = feats
    return out


def _aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    out = {
        "first": float(arr[0]), "last": float(arr[-1]), "mean": float(arr.mean()),
        "min": float(arr.min()), "max": float(arr.max()),
    }
    acc = arr[0]
    for v in arr[1:]:
        acc = 0.9 * acc + 0.1 * v
    out["ewma0.9"] = float(acc)
    return out


def episode_features(ep: dict, steps: dict[tuple[str, int], dict],
                     step_cap: int) -> dict[str, float]:
    """Features of one episode. Keys are identical across benchmarks."""
    inst = str(ep["instance_id"])
    feats: dict[str, float] = {}

    # --- critics
    last = _last_verdicts(ep)
    for critic in CRITICS:
        if critic in last:
            feats[f"critic_{critic[-2:]}"] = float(last[critic])
    every = _all_verdicts(ep)
    if last:
        feats["critic_frac_last"] = float(np.mean([float(v) for v in last.values()]))
    if every:
        feats["critic_frac_all"] = float(np.mean([float(v) for _, v in every]))
        feats["critic_n_fail"] = -float(sum(1 for _, v in every if not v))

    # --- trajectory shape
    n_gen = int(ep.get("n_generations") or 0)
    n_steps = int(ep.get("n_steps") or 0)
    feats["n_generations"] = -float(n_gen)
    feats["n_steps"] = -float(n_steps)
    feats["hit_cap"] = -float(n_steps >= step_cap)

    # --- verbalised confidence
    verb = ep.get("verbalized_2s_confidence")
    if isinstance(verb, (int, float)) and math.isfinite(float(verb)):
        feats["verbalized"] = float(verb)

    # --- per-generation scores, aggregated
    series: dict[str, list[float]] = {}
    idx = 0
    for step in ep.get("trajectory") or []:
        if step.get("action") == "generate" and not step.get("skipped"):
            for key, value in steps.get((inst, idx), {}).items():
                series.setdefault(key, []).append(value)
            idx += 1
    for metric, values in series.items():
        if metric not in STEP_METRICS or not values:
            continue
        agg = _aggregate(values)
        for name in STEP_AGGS:
            if name in agg:
                feats[f"{metric}:{name}"] = agg[name]
    return feats


def load_cell(base: str, cell: str, benchmark: str, generator: str,
              label: str, with_topk: bool = False) -> dict[str, Any]:
    """Features, labels and bookkeeping for one cell."""
    root = Path(base) / cell
    stem = f"{benchmark}__{generator}"
    episodes = _read_jsonl(root / f"{stem}.jsonl")
    if not episodes:
        return {}
    steps = per_generation_metrics(root, stem, with_topk=with_topk)
    cap = max((int(e.get("n_steps") or 0) for e in episodes), default=0)
    rows = [episode_features(e, steps, cap) for e in episodes]
    return {
        "label": label,
        "benchmark": benchmark,
        "y": np.array([int(bool(e["fixed"])) for e in episodes]),
        "instances": [str(e["instance_id"]) for e in episodes],
        "features": rows,
        "n_generations": np.array([int(e.get("n_generations") or 0) for e in episodes]),
        "step_cap": cap,
    }


def matrix(rows: list[dict[str, float]], names: list[str]) -> np.ndarray:
    """Dense matrix; gaps are filled with the column median."""
    m = np.full((len(rows), len(names)), np.nan)
    for i, row in enumerate(rows):
        for j, name in enumerate(names):
            value = row.get(name)
            if value is not None and math.isfinite(float(value)):
                m[i, j] = float(value)
    for j in range(m.shape[1]):
        column = m[:, j]
        if np.isnan(column).all():
            m[:, j] = 0.0
        else:
            m[np.isnan(column), j] = float(np.nanmedian(column))
    return m


def common_names(cells: list[dict], min_share: float = 0.6) -> list[str]:
    """Features present in a large enough share of cells.

    Full intersection is too strict: tau2 has no logprobs and some cells lack
    individual critics. Union is too loose: a feature living in one cell
    generalises to nothing.
    """
    counts: dict[str, int] = {}
    for cell in cells:
        present = {k for row in cell["features"] for k in row}
        for name in present:
            counts[name] = counts.get(name, 0) + 1
    need = max(1, int(min_share * len(cells)))
    return sorted(n for n, c in counts.items() if c >= need)
