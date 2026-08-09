#!/usr/bin/env python3
"""Experiment 1 (verbalized): per-generation Verbalized-2S over the trajectory.

The pipeline computes Verbalized-2S confidence only for the FINAL candidate.
This script computes it *post-hoc for every generation* of each instance, then
aggregates (mean/max/min/last/std) and scores each aggregator with PRR — the
same trajectory-aggregation question as ``aggregate_trajectory_uq.py`` but for
the verbalized signal (which needs no logprobs, so no provider artifact).

Aggregating verbalized confidence over the agent's multiple generations is
conceptually the "multiple guesses" trick from Tian et al. 2023
(Just Ask for Calibration), which improved calibration.

Inputs (per benchmark, from a completed run):
  <run_root>/<bench>__<gen>.generation_logprobs.jsonl   # per-generation code
  <run_root>/readable/<bench>/final_logprob_bayes_quality.csv   # quality label

Verbalized-2S is re-elicited against an OpenAI-compatible endpoint (local vLLM),
reconstructing the 2-stage dialogue: problem -> the candidate answer -> ask for
P(correct).

Example:
  python scripts/verbalized_trajectory.py \
      --run-root .../sim_results/sage_uq_gpt_oss_20b_local_v2 \
      --benchmark lcb_hard --generator gpt_oss_20b_local \
      --base-url http://127.0.0.1:8010/v1 --model openai/gpt-oss-20b \
      --out .../readable/lcb_hard/metric_scores_verb_trajectory.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from code_uq.analysis.analyze_lcb_llm_tool_agent_logs import prediction_rejection_area, prr


def bootstrap_prr_ci(conf, quality, *, n_boot, seed=0):
    """Percentile bootstrap 95% CI for PRR (cheap: normalisers fit once)."""
    import numpy as np
    pairs = [(c, q) for c, q in zip(conf, quality) if c is not None]
    if n_boot <= 0 or len(pairs) < 3:
        return (None, None)
    cf = [c for c, _ in pairs]
    qf = [q for _, q in pairs]
    oracle = prediction_rejection_area([-float(q) for q in qf], qf, 1.0)
    rng = np.random.RandomState(seed)
    arr = np.arange(len(qf))
    rnd = []
    for _ in range(200):
        rng.shuffle(arr)
        s = prediction_rejection_area(arr.tolist(), qf, 1.0)
        if s is not None:
            rnd.append(s)
    random = float(np.mean(rnd)) if rnd else None
    if oracle is None or random is None or abs(oracle - random) < 1e-12:
        return (None, None)
    idx = np.arange(len(pairs))
    scores = []
    for _ in range(n_boot):
        take = rng.choice(idx, size=len(idx), replace=True)
        bc = [cf[i] for i in take]
        bq = [qf[i] for i in take]
        if len(set(bq)) < 2:
            continue
        area = prediction_rejection_area([-c for c in bc], bq, 1.0)
        if area is not None:
            scores.append((area - random) / (oracle - random))
    if len(scores) < max(10, n_boot // 4):
        return (None, None)
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))

VERBALIZED_2S_CONFIDENCE_PROMPT = (
    "Provide the probability that your guess is correct. Give ONLY the probability, "
    "no other words or explanation.\n\nFor example:\n\nProbability: <the probability "
    "between 0.0 and 1.0 that your guess is correct, without any extra commentary "
    "whatsoever; just the probability!>"
)


def parse_confidence(text: str) -> float | None:
    patterns = [
        r"probability\s*[:=]\s*((?:[0-9]+(?:\.[0-9]+)?)|(?:\.[0-9]+))\s*(%)?",
        r"confidence\s*[:=]\s*((?:[0-9]+(?:\.[0-9]+)?)|(?:\.[0-9]+))\s*(%)?",
        r"\b((?:[0-9]+(?:\.[0-9]+)?)|(?:\.[0-9]+))\s*%",
        r"((?:0(?:\.[0-9]+)?)|(?:1(?:\.0+)?)|(?:\.[0-9]+))\s*$",
    ]
    for pat in patterns:
        m = None
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            pass  # keep the last match (reasoning models emit the answer last)
        if not m:
            continue
        try:
            val = float(m.group(1))
        except (TypeError, ValueError):
            continue
        if (len(m.groups()) >= 2 and m.group(2)) or val > 1.0:
            val /= 100.0
        return max(0.0, min(1.0, val))
    return None


def load_problems(benchmark: str, *, seed: int, lcb_version: str,
                  platform: str, plus_input_cap: int) -> dict[str, str]:
    from code_uq.environments.fitted_live.function_adapters import make_function_adapter

    bench = "lcb_medium" if benchmark == "lcb_med" else benchmark
    adapter = make_function_adapter(
        benchmark=bench, n_instances=0, seed=seed,
        lcb_version=lcb_version, plus_input_cap=plus_input_cap, platform=platform,
    )
    out: dict[str, str] = {}
    for inst in adapter.load_instances():
        iid = str(adapter.instance_id(inst))
        try:
            out[iid] = adapter.build_prompt(inst, None, [])
        except Exception:
            out[iid] = ""
    return out


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def load_quality(csv_path: Path) -> dict[str, int]:
    out = {}
    for r in csv.DictReader(csv_path.open()):
        try:
            out[str(r["instance_id"])] = int(r["quality"])
        except (KeyError, ValueError):
            pass
    return out


def elicit(client, model: str, problem: str, answer: str, max_tokens: int) -> float | None:
    if problem:
        messages = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": VERBALIZED_2S_CONFIDENCE_PROMPT},
        ]
    else:
        # code-only: no problem statement available (e.g. SWE diffs). Judge the
        # candidate patch/solution on its own. Weaker (the model can't see what
        # to solve), but lets us cover SWE where the export has no task text.
        messages = [
            {"role": "user",
             "content": "A coding agent produced the following patch/solution:\n\n"
                        f"{answer}\n\n{VERBALIZED_2S_CONFIDENCE_PROMPT}"},
        ]
    import time as _time
    last_exc = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=0.0, max_tokens=max_tokens,
            )
            return parse_confidence(resp.choices[0].message.content or "")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            name = type(exc).__name__
            transient = name in {"APIConnectionError", "APITimeoutError",
                                  "RateLimitError", "InternalServerError"} \
                or getattr(exc, "status_code", None) in (429, 500, 502, 503, 529)
            if not transient or attempt == 4:
                raise
            _time.sleep(min(2 ** attempt, 20))
    raise last_exc


AGGS = {
    "last": lambda v: v[-1],
    "first": lambda v: v[0],
    "mean": lambda v: statistics.fmean(v),
    "max": lambda v: max(v),
    "min": lambda v: min(v),
    "std": lambda v: statistics.pstdev(v) if len(v) > 1 else 0.0,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, default=None)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--generator", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lcb-version", default="all")
    p.add_argument("--platform", default="leetcode")
    p.add_argument("--plus-input-cap", type=int, default=200)
    p.add_argument("--limit", type=int, default=0, help="cap instances (debug)")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="bootstrap resamples for a 95%% PRR CI; 0 = off")
    p.add_argument("--cache", type=Path, default=None,
                   help="JSON to cache/reuse per-instance verbalized values "
                        "(skips vLLM re-elicitation)")
    p.add_argument("--code-only", action="store_true",
                   help="no problem statement: judge the candidate patch/solution "
                        "alone (needed for SWE — diffs, no task text in export)")
    p.add_argument("--flat-dir", type=Path, default=None,
                   help="colleague export layout: <flat-dir> holds "
                        "generation_logprobs.jsonl + final_logprob_bayes_quality.csv")
    p.add_argument("--api-key", default=None,
                   help="OpenRouter/OpenAI key (default EMPTY for local vLLM); "
                        "or set OPENROUTER_API_KEY")
    args = p.parse_args()

    import os
    from openai import OpenAI
    key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or "EMPTY"
    client = OpenAI(api_key=key, base_url=args.base_url)

    if args.flat_dir:
        gens = read_jsonl(args.flat_dir / "generation_logprobs.jsonl")
        quality = load_quality(args.flat_dir / "final_logprob_bayes_quality.csv")
    else:
        stem = f"{args.benchmark}__{args.generator}"
        gens = read_jsonl(args.run_root / f"{stem}.generation_logprobs.jsonl")
        quality = load_quality(
            args.run_root / "readable" / args.benchmark / "final_logprob_bayes_quality.csv"
        )
    if args.code_only:
        problems = {}   # elicit judges the candidate patch on its own
    else:
        problems = load_problems(
            args.benchmark, seed=args.seed, lcb_version=args.lcb_version,
            platform=args.platform, plus_input_cap=args.plus_input_cap,
        )

    # group generations per instance, ordered
    by_inst: dict[str, list[dict]] = defaultdict(list)
    for g in gens:
        by_inst[str(g["instance_id"])].append(g)
    for iid in by_inst:
        by_inst[iid].sort(key=lambda g: int(g.get("generation_index", g.get("step", 0))))

    if args.code_only:
        ids = [i for i in by_inst if i in quality]
    else:
        ids = [i for i in by_inst if i in quality and i in problems and problems[i]]
    if args.limit:
        ids = ids[: args.limit]

    # Incremental cache: load whatever is already done, elicit only the missing
    # instances, and persist as we go so a network drop can be resumed.
    verbs: dict[str, list[float]] = {}
    if args.cache and args.cache.exists():
        verbs = {k: v for k, v in json.loads(args.cache.read_text()).items() if k in quality}
        print(f"loaded {len(verbs)} cached instances from {args.cache}", file=sys.stderr)
    todo = [i for i in ids if i not in verbs]
    print(f"{len(verbs)} cached, {len(todo)} to elicit", file=sys.stderr)

    def _save():
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            args.cache.write_text(json.dumps(verbs))

    for n, iid in enumerate(todo, 1):
        prob = "" if args.code_only else problems[iid]
        vals = []
        for g in by_inst[iid]:
            answer = g.get("raw_text") or g.get("code") or ""
            if not answer.strip():
                continue
            try:
                c = elicit(client, args.model, prob, answer, args.max_tokens)
            except Exception as exc:
                print(f"  {iid}: elicit error {type(exc).__name__}: {exc}", file=sys.stderr)
                c = None
            if c is not None:
                vals.append(c)
        if vals:
            verbs[iid] = vals
        if n % 10 == 0:
            print(f"[{n}/{len(todo)}] elicited", file=sys.stderr, flush=True)
            _save()   # checkpoint
    _save()

    scored = [i for i in ids if i in verbs]
    q = [quality[i] for i in scored]
    rows = []
    for name, fn in AGGS.items():
        conf = [fn(verbs[i]) for i in scored]
        ci_lo, ci_hi = bootstrap_prr_ci(conf, q, n_boot=args.bootstrap)
        rows.append({
            "score": f"verbalized_{name}",
            "aggregator": name,
            "n": len(scored),
            "PRR": prr(conf, q, 1.0),
            "PRR_ci_lo": ci_lo,
            "PRR_ci_hi": ci_hi,
            "PRR_05": prr(conf, q, 0.5),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nn_scored={len(scored)}  pass@1={sum(q)/len(q):.2f}" if q else "no data")
    for r in rows:
        pr = r["PRR"]
        ci = (f" [{r['PRR_ci_lo']:+.3f}, {r['PRR_ci_hi']:+.3f}]"
              if r["PRR_ci_lo"] is not None else "")
        print(f"  {r['score']:<22} PRR={pr:+.3f}{ci}" if pr is not None
              else f"  {r['score']:<22} PRR=.")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
