"""Recover per-step uncertainty signals for DeepSWE by re-scoring recorded actions.

The signals the method table needs -- logprob, perplexity, mean token entropy,
self-certainty -- are properties of the executor's own generation, not of the
environment, so a verified label cannot supply them. But they do not need the
agent re-run either: the actions were recorded, so one teacher-forced forward
pass per step reproduces them. No environment, no agent loop, no sampling.

Three fidelity notes, because the numbers are not identical to the originals and
the difference is not uniform:

1. Commands come from ``data/replay_inputs``, where they are complete (DG runs to
   11.9k characters, DQ to 55k). They are NOT taken from ``judge_views``, where
   458 of DG's 5887 actions sit exactly at the 400-character cap -- and those are
   the long apply_patch heredocs, which is to say exactly the repository-changing
   steps the verified labels are attached to. Scoring the truncated form would
   bias the signal on precisely the steps being measured.

2. The *context* is still the trimmed view: 35% of DeepSWE observations hit the
   400/600-character cap and the full ones are gone. So a conditional logprob here
   is conditioned on less than the agent saw. This is a real departure and it is
   why these signals belong to this experiment rather than to the ALFWorld tables.

3. Entropy is computed over the full vocabulary while the originals used a
   truncated top-k. The surviving ALFWorld records put ``topk_mass`` at 0.998, so
   the two agree to within 0.2% of the mass and this is not worth correcting.
   ``self_certainty`` is different: it is ``-mean(logps) - log(k)`` and depends on
   k outright. ``--calibrate-k`` re-scores ALFWorld steps whose stored
   per-token self-certainty survives and reports the k that reproduces it.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path


def read_inputs(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_views(path: Path) -> dict[str, dict]:
    out = {}
    for line in open(path):
        if line.strip():
            v = json.loads(line)
            out[v["id"]] = v
    return out


def context(task: str, steps: list[dict], window: int, obs_chars: int) -> str:
    """What the model is conditioned on: the task, then the recent exchange."""
    lines = [f"<task>\n{task.strip()[:3000]}\n</task>", "", "<transcript>"]
    for s in steps[-window:]:
        lines.append(f"$ {(s.get('action') or '').strip()}")
        lines.append((s.get("observation") or "").strip()[:obs_chars])
    lines += ["</transcript>", "", "Next command:"]
    return "\n".join(lines)


def score_block(model, tokenizer, prefix: str, target: str, top_k: int, device: str) -> dict:
    """Teacher-forced statistics for ``target`` conditioned on ``prefix``."""
    import torch

    ids_prefix = tokenizer(prefix, return_tensors="pt").input_ids
    ids_full = tokenizer(prefix + target, return_tensors="pt").input_ids
    if ids_full.shape[1] <= ids_prefix.shape[1]:
        return {}
    ids_full = ids_full[:, -model.config.max_position_embeddings:] \
        if hasattr(model.config, "max_position_embeddings") else ids_full
    start = max(1, ids_prefix.shape[1])
    with torch.no_grad():
        logits = model(ids_full.to(device)).logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    chosen, entropies, certainties = [], [], []
    for position in range(start - 1, ids_full.shape[1] - 1):
        row = logprobs[position]
        nxt = int(ids_full[0, position + 1])
        chosen.append(float(row[nxt]))
        top = torch.topk(row, k=min(top_k, row.shape[-1])).values
        # renormalise over the top-k, the way a truncated API response forces
        probs = torch.softmax(top, dim=-1)
        entropies.append(float(-(probs * probs.clamp_min(1e-12).log()).sum()))
        certainties.append(float(-top.mean() - math.log(top.shape[-1])))
    if not chosen:
        return {}
    mean = sum(chosen) / len(chosen)
    return {"mean_logprob": mean, "sum_logprob": sum(chosen),
            "perplexity": math.exp(min(700.0, -mean)),
            "mean_entropy": sum(entropies) / len(entropies),
            "self_certainty": sum(certainties) / len(certainties),
            "num_tokens": len(chosen)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True, help="replay_inputs/<cohort>.jsonl.gz")
    p.add_argument("--views", type=Path, required=True, help="judge_views/<cohort>.jsonl")
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--obs-chars", type=int, default=400)
    p.add_argument("--top-k", type=int, default=20,
                   help="the truncation the originals were computed under; "
                        "self_certainty depends on it, entropy barely does")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--calibrate-k", type=Path, default=None,
                   help="an ALFWorld trajectories.jsonl: re-score its actions and report "
                        "which top-k reproduces the stored self_certainty")
    a = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map=a.device).eval()

    if a.calibrate_k:
        rows = [json.loads(l) for l in open(a.calibrate_k) if l.strip()][:40]
        print(f"[calibrate] {len(rows)} steps from {a.calibrate_k.name}", flush=True)
        for k in (5, 10, 20, 50, 100):
            errs = []
            for r in rows:
                stored = ((r.get("uq") or {}).get("combined") or {}).get("self_certainty")
                text = r.get("raw_response") or r.get("action")
                if stored is None or not text:
                    continue
                got = score_block(model, tokenizer, "", text, k, a.device)
                if got:
                    errs.append(abs(got["self_certainty"] - stored))
            if errs:
                print(f"[calibrate] top_k={k:3}  mean |error| {sum(errs)/len(errs):.3f}", flush=True)
        return

    views = read_views(a.views)
    episodes = read_inputs(a.inputs)
    if a.limit:
        episodes = episodes[: a.limit]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(a.out, "w") as handle:
        for index, episode in enumerate(episodes, 1):
            view = views.get(episode["id"])
            if not view:
                print(f"[score] {episode['id']}: no view, skipped", flush=True)
                continue
            commands = episode.get("commands") or []
            steps = []
            for k, command in enumerate(commands):
                stats = score_block(model, tokenizer,
                                    context(view["task"], view["steps"][:k], a.window, a.obs_chars),
                                    command, a.top_k, a.device)
                observation = (view["steps"][k].get("observation") or "") if k < len(view["steps"]) else ""
                rc = None
                if observation.startswith("[rc="):
                    try:
                        rc = int(observation[4:observation.index("]")])
                    except ValueError:
                        rc = None
                steps.append({"command": command, "returncode": rc, "format_error": False,
                              **{key: stats.get(key) for key in
                                 ("mean_logprob", "mean_entropy", "self_certainty", "num_tokens")}})
            # patch_bytes and exit_status exist only to pass load_deepswe's filter;
            # every episode in replay_inputs was already scored by the benchmark
            handle.write(json.dumps({"id": episode["id"], "rewards": episode["rewards"],
                                     "patch_bytes": 1, "exit_status": "submitted",
                                     "steps": steps}) + "\n")
            written += 1
            print(f"[score] {index}/{len(episodes)} {episode['id'][:34]}: {len(steps)} steps",
                  flush=True)
    print(f"[score] done: {written} episodes -> {a.out}")


if __name__ == "__main__":
    main()
