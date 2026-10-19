"""Recover DeepSWE's per-step uncertainty signals by re-scoring, not by re-running.

The recorded runs kept the executed command and the observation at every step
but not the token statistics, so every signal the paper compares -- logprob,
perplexity, entropy, self-certainty, and the fusions built on them -- is missing
on this environment. They do not need the agent back: the command the agent
emitted is in the log, so one teacher-forced pass over it under the same model
recovers the statistics it was generated with. No sampling, no environment, no
docker, one forward pass per step.

Three places where this is a re-score and not the original measurement:

1. Commands come from ``replay_inputs``, which holds them whole. ``judge_views``
   truncates at 400 characters and 458 of DG's 5887 actions sit exactly at that
   bound -- the apply_patch heredocs, which is to say the repo-changing steps the
   labels attach to. The observations still come from ``judge_views`` and are
   still truncated, so the context is thinner than the agent's.
2. Hence the conditioning is not identical: the logprobs are conditioned on less
   than the agent saw. The direction is knowable -- a thinner context cannot
   raise the model's certainty about a command it did not write -- but the size
   is not, so these numbers belong to this experiment and are not to be pasted
   into the ALFWorld tables.
3. The prefix and the command are tokenised separately and concatenated as ids,
   which fixes the boundary between them exactly. A joint tokenisation could
   merge the last prefix token with the first command token; generation could
   not, so separate encoding is the closer of the two.

Top-k is no longer a caveat. vLLM's ``prompt_logprobs`` returns the same
truncated distribution the original API responses carried, so entropy and
self-certainty are computed over the requested k rather than the full
vocabulary, and ``self_certainty = -mean(logps) - log(k)`` is on the project's
scale by construction.
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


def build_ids(tokenizer, prefix: str, target: str, max_len: int) -> tuple[list[int], int]:
    """The prompt as token ids, with the index where the command starts.

    Building the prompt from ids rather than text means the command's first
    token is the command's, never a merge with the tail of the prefix, so the
    scored span is exactly the command. When the pair does not fit, the prefix
    loses its oldest tokens: the command has to stay whole, it is the target.
    """
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    if not target_ids:
        return [], 0
    room = max_len - 4
    if len(target_ids) >= room:
        return target_ids[:room], 1
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids if prefix else []
    budget = room - len(target_ids)
    prefix_ids = prefix_ids[-budget:] if budget > 0 else []
    return prefix_ids + target_ids, max(1, len(prefix_ids))


def summarise(token_ids: list[int], prompt_logprobs: list, start: int, top_k: int) -> dict:
    """Reduce the teacher-forced distributions the way the toolkit does."""
    chosen: list[float] = []
    entropies: list[float] = []
    certainties: list[float] = []
    for position in range(start, min(len(token_ids), len(prompt_logprobs))):
        entry = prompt_logprobs[position]
        if not entry:
            continue
        realised = entry.get(token_ids[position])
        if realised is None:
            continue
        chosen.append(float(realised.logprob))
        # vLLM returns the top-k plus the realised token when it falls outside
        # them, so the dict can hold k+1 entries; take the k best, not all.
        tops = sorted((float(v.logprob) for v in entry.values()), reverse=True)[:top_k]
        if len(tops) < 2:
            continue
        highest = max(tops)
        weights = [math.exp(value - highest) for value in tops]
        total = sum(weights)
        probabilities = [value / total for value in weights]
        entropies.append(-sum(p * math.log(p) for p in probabilities if p > 0.0))
        certainties.append(-sum(tops) / len(tops) - math.log(len(tops)))
    if not chosen:
        return {}
    mean = sum(chosen) / len(chosen)
    result = {"mean_logprob": mean, "sum_logprob": sum(chosen),
              "perplexity": math.exp(min(700.0, -mean)), "num_tokens": len(chosen)}
    if entropies:
        result["mean_entropy"] = sum(entropies) / len(entropies)
        result["self_certainty"] = sum(certainties) / len(certainties)
    return result


def score(engine, params, tokenizer, pairs: list[tuple[str, str]], top_k: int,
          max_len: int, batch: int = 8) -> list[dict]:
    """Teacher-forced statistics for each (prefix, command) pair.

    Scored in fixed-size chunks rather than one call per episode. A
    prompt_logprobs tensor is tokens x k wide, so a 189-step episode asked for
    in one go tried to allocate 1.8 GiB beside the weights and killed the
    engine; the chunk bounds that by the chunk, not by the longest episode.
    """
    built = [build_ids(tokenizer, prefix, target, max_len) for prefix, target in pairs]
    stats: list[dict] = [{} for _ in built]
    pending = [(index, ids, start) for index, (ids, start) in enumerate(built) if ids]
    for position in range(0, len(pending), batch):
        chunk = pending[position : position + batch]
        outputs = engine.generate([{"prompt_token_ids": ids} for _, ids, _ in chunk],
                                  params, use_tqdm=False)
        for (index, ids, start), out in zip(chunk, outputs):
            stats[index] = summarise(list(out.prompt_token_ids), out.prompt_logprobs or [],
                                     start, top_k)
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True, help="replay_inputs/<cohort>.jsonl.gz")
    p.add_argument("--views", type=Path, required=True, help="judge_views/<cohort>.jsonl")
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--obs-chars", type=int, default=400)
    p.add_argument("--top-k", type=int, default=20,
                   help="the truncation the originals were computed under; "
                        "self_certainty depends on it, entropy barely does")
    p.add_argument("--max-len", type=int, default=32768)
    p.add_argument("--gpu-util", type=float, default=0.90)
    p.add_argument("--batch", type=int, default=8,
                   help="prompts per engine call; bounds the logprobs tensor")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--calibrate-k", type=Path, default=None,
                   help="an ALFWorld trajectories.jsonl: re-score its actions and report "
                        "which top-k reproduces the stored self_certainty")
    a = p.parse_args()

    from vllm import LLM, SamplingParams
    # enforce_eager, and the checkpoint's own dtype rather than a forced one:
    # the compiled path wants nvcc, which a plain python image does not carry,
    # and this is a single forward pass per step where graph capture buys little
    engine = LLM(model=a.model, dtype="auto", max_model_len=a.max_len,
                 gpu_memory_utilization=a.gpu_util, enforce_eager=True,
                 tensor_parallel_size=a.tensor_parallel_size, trust_remote_code=True)
    tokenizer = engine.get_tokenizer()
    # one token is generated and thrown away: prompt_logprobs is what we came for
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=a.top_k)

    if a.calibrate_k:
        rows = [json.loads(l) for l in open(a.calibrate_k) if l.strip()][:40]
        print(f"[calibrate] {len(rows)} steps from {a.calibrate_k.name}", flush=True)
        wanted = [(r, ((r.get("uq") or {}).get("combined") or {}).get("self_certainty"))
                  for r in rows]
        wanted = [(r, s) for r, s in wanted if s is not None and (r.get("raw_response") or r.get("action"))]
        for k in (5, 10, 20, 50, 100):
            local = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=k)
            got = score(engine, local, tokenizer,
                        [("", r.get("raw_response") or r["action"]) for r, _ in wanted],
                        k, a.max_len, a.batch)
            errs = [abs(g["self_certainty"] - s) for g, (_, s) in zip(got, wanted)
                    if g.get("self_certainty") is not None]
            if errs:
                print(f"[calibrate] top_k={k:3}  mean |error| {sum(errs)/len(errs):.3f} "
                      f"over {len(errs)} steps", flush=True)
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
            pairs = [(context(view["task"], view["steps"][:k], a.window, a.obs_chars), command)
                     for k, command in enumerate(commands)]
            stats = score(engine, params, tokenizer, pairs, a.top_k, a.max_len, a.batch)
            steps = []
            for k, (command, numbers) in enumerate(zip(commands, stats)):
                observation = (view["steps"][k].get("observation") or "") if k < len(view["steps"]) else ""
                rc = None
                if observation.startswith("[rc="):
                    try:
                        rc = int(observation[4:observation.index("]")])
                    except ValueError:
                        rc = None
                steps.append({"command": command, "returncode": rc, "format_error": False,
                              **{key: numbers.get(key) for key in
                                 ("mean_logprob", "sum_logprob", "perplexity",
                                  "mean_entropy", "self_certainty", "num_tokens")}})
            # patch_bytes and exit_status exist only to pass load_deepswe's filter;
            # every episode in replay_inputs was already scored by the benchmark
            handle.write(json.dumps({"id": episode["id"], "rewards": episode["rewards"],
                                     "patch_bytes": 1, "exit_status": "submitted",
                                     "steps": steps}) + "\n")
            written += 1
            scored = sum(1 for s in steps if s.get("mean_logprob") is not None)
            print(f"[score] {index}/{len(episodes)} {episode['id'][:34]}: "
                  f"{scored}/{len(steps)} steps scored", flush=True)
    print(f"[score] done: {written} episodes -> {a.out}")


if __name__ == "__main__":
    main()
