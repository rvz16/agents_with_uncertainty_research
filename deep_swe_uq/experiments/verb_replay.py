"""Verbalised confidence for DeepSWE as a side query, after the fact.

Both models ignored a confidence instruction inside the tool-calling loop
(Qwen 0 of 1806 commands, gpt-oss 49 of 6365). The OSWorld/WebArena tables
get "Verb actions" from a separate side query per step; this does the same
on the recorded trajectories: for every assistant step, replay the
conversation up to and including that step's tool result and ask the model,
without tools, for a 0-100 probability of finishing the task. The trajectory
was generated without these queries, so nothing feeds back into it.

    python verb_replay.py --jobs JOBS_ZIP_OR_DIR --run deepswe_qwen_113_v8 \\
        --base-url http://host:port/v1 --model Qwen/Qwen3.6-35B-A3B --out verb.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

QUESTION = (
    "Pause for a moment. Before you continue, estimate the probability that you "
    "will complete this whole task successfully, as an integer from 0 to 100. "
    "Reply with the integer only."
)
NUM = re.compile(r"(\d{1,3})")
FINAL = re.compile(r"<\|channel\|>final<\|message\|>\s*(\d{1,3})")


def parse_confidence(text: str) -> int | None:
    """gpt-oss served without a reasoning parser returns raw harmony markup and
    spends most of a short budget in the analysis channel (4708 of 6365 empty
    answers at max_tokens=16). Prefer the number in the final channel, else the
    last integer in the text."""
    m = FINAL.search(text)
    if m:
        return min(100, int(m.group(1)))
    nums = NUM.findall(re.sub(r"<\|[^|]*\|>", " ", text))
    return min(100, int(nums[-1])) if nums else None


def trajectories(jobs: str, run: str):
    if jobs.endswith(".zip"):
        z = zipfile.ZipFile(jobs)
        for n in sorted(z.namelist()):
            if n.startswith(run + "/") and n.endswith("mini-swe-agent.trajectory.json"):
                yield n.split("/")[1], json.loads(z.read(n))
    else:
        for p in sorted(Path(jobs, run).glob("*/agent/mini-swe-agent.trajectory.json")):
            yield p.parents[1].name, json.loads(p.read_text())


def clean(messages: list[dict]) -> list[dict]:
    """The conversation as the API saw it: roles, text, tool calls, tool results."""
    out = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            msg = ((m.get("extra") or {}).get("response") or {}).get("choices", [{}])[0].get("message") or {}
            entry = {"role": "assistant", "content": m.get("content") or ""}
            calls = msg.get("tool_calls") or m.get("tool_calls")
            if calls:
                entry["tool_calls"] = [{"id": c.get("id") or f"call_{k}", "type": "function",
                                        "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}
                                       for k, c in enumerate(calls)]
            out.append(entry)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id") or "call_0", "content": str(m.get("content") or "")[:4000]})
        elif role in ("system", "user"):
            out.append({"role": role, "content": str(m.get("content") or "")})
    return out


def ask(client, model: str, prefix: list[dict], extra: dict, max_tokens: int) -> tuple[int | None, str]:
    for attempt in range(4):
        try:
            r = client.chat.completions.create(model=model, messages=prefix + [{"role": "user", "content": QUESTION}],
                                               max_tokens=max_tokens, temperature=0.0, **extra)
            msg = r.choices[0].message
            text = (msg.content or "").strip()
            if not text:  # a reasoning parser may have routed everything to reasoning_content
                text = (getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or "").strip()
            if not text and getattr(msg, "tool_calls", None):
                text = "tool_call"
            return parse_confidence(text), text
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            if "context" in err.lower() or "maximum" in err.lower():
                return None, "context_overflow"
            time.sleep(2 * (attempt + 1))
    return None, err


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", required=True); p.add_argument("--run", required=True)
    p.add_argument("--base-url", required=True); p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--reasoning-effort", default=None, help="gpt-oss: low; Qwen: enable_thinking=false is sent instead")
    a = p.parse_args()
    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key="local", timeout=600)
    extra: dict = {}
    if a.reasoning_effort:
        extra["reasoning_effort"] = a.reasoning_effort
        max_tokens = 512  # the analysis channel comes first and is not free
    else:
        # Qwen answers most prefixes with a <tool_call> that the server's tool
        # parser strips out of ``content``; a regex constraint leaves it no
        # option but the integer.
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False},
                               "structured_outputs": {"regex": "(100|[1-9]?[0-9])"}}
        max_tokens = 8
    done = set()
    if a.out.exists():  # resume, but only from rows that actually got an answer (the host /tmp survives between tasks)
        rows = [json.loads(l) for l in open(a.out) if l.strip()]
        rows = [r for r in rows if r.get("confidence") is not None]
        with open(a.out, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        done = {(r["id"], r["step"]) for r in rows}
        print(f"[verb] resuming: {len(done)} answered rows kept from {a.out}", flush=True)
    out = open(a.out, "a"); total = asked = 0
    for tid, tr in trajectories(a.jobs, a.run):
        msgs = clean(tr.get("messages", []))
        # step k = the k-th assistant message; the prefix ends after its tool result
        cuts = []
        k = 0
        for j, m in enumerate(msgs):
            if m["role"] == "assistant":
                end = j + 1
                if end < len(msgs) and msgs[end]["role"] == "tool":
                    end += 1
                cuts.append((k, end)); k += 1
        jobs = [(k, end) for k, end in cuts if (tid, k) not in done]
        total += len(cuts)
        with ThreadPoolExecutor(a.workers) as pool:
            for (k, end), (conf, text) in zip(jobs, pool.map(lambda ke: ask(client, a.model, msgs[:ke[1]], extra, max_tokens), jobs)):
                out.write(json.dumps({"id": tid, "step": k, "confidence": conf, "raw": text[-200:]}) + "\n"); asked += 1
        out.flush()
        print(f"[verb] {tid}: {len(cuts)} steps ({len(jobs)} asked)", flush=True)
    print(f"[verb] done: {asked} queries, {total} steps in total -> {a.out}")


if __name__ == "__main__":
    main()
