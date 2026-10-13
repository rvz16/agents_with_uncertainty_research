"""A decision model as the per-step judge, in place of the generative one.

At step k the model sees the task and the last ``--window`` steps up to and
including k, with no step counter and nothing after it, and answers the schema
of ``decision_state.QUESTIONS``. The prefix and its truncation are the ones
``prefix_judge`` uses, so the new signal is comparable with the judge already
in the tables.

Four sequences come out per episode rather than one: the probability of
success, whether the agent is stuck, whether its last check is stale, and how
far it has come. They enter the report as ordinary per-step signals, which is
the point -- the structural questions are what an additive log-likelihood ratio
over token statistics cannot express.

Two back ends, both typed, so nothing has to be parsed out of prose:
``liquid`` for a LiquidAI d1 release (``AutoModel.system_one``) and ``clef``
for a Cloudflare release (``joint_schema_model.systemone``).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.decision_model_d3 import to_our_shape
from experiments.decision_state import GATE_QUESTIONS, QUESTIONS
from experiments.prefix_judge import alfworld_view, prompt

ANSWER_KEYS = ("success", "stuck", "stale_evidence", "progress")


def load_liquid(model_id: str, device: str):
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16)
    model = model.to(device).eval()
    return model


SCHEMAS = {"outcome": QUESTIONS, "gate": GATE_QUESTIONS}
SCHEMA = QUESTIONS  # set from --schema before any backend call


def ask_liquid(model, states: list[str]) -> list[dict]:
    """The release batches on a sequence of (state, questions) tuples; passing
    dicts there indexes them with r[1] and raises KeyError(1)."""
    if hasattr(model, "system_one_batch"):
        return list(model.system_one_batch([(state, SCHEMA) for state in states]))
    return [model.system_one(state, SCHEMA) for state in states]


def load_d3(model_id: str, device: str):
    """Decision 3.0: its own released remote code, one call per state.

    The loader rejects a device without an index -- plain "cuda" raises -- which
    cost a whole sweep the first time.
    """
    import torch
    from transformers import AutoModel
    return AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16,
                                     device_map=device).eval()


def ask_d3(model, states: list[str]) -> list[dict]:
    import torch
    out = []
    for state in states:
        try:
            with torch.no_grad():
                out.append(model.system_one(state=state, questions=SCHEMA))
        except Exception as exc:  # noqa: BLE001 - one bad state must not end the sweep
            out.append({"error": repr(exc)[:200]})
    return out


def load_clef(model_id: str, device: str):
    import sys

    from huggingface_hub import snapshot_download

    path = snapshot_download(model_id)
    sys.path.insert(0, path)
    import joint_schema_model as jsm  # noqa: E402 - importable once downloaded

    model, processor = jsm.load_release_model(path, device=device)
    return jsm, model, processor


def ask_clef(bundle, states: list[str], model_id: str, max_length: int) -> list[dict]:
    jsm, model, processor = bundle
    return [jsm.systemone(model, processor, {"model": model_id, "state": s, "questions": SCHEMA},
                          max_length=max_length) for s in states]


def value_of(answer: dict, key: str) -> float | None:
    node = ((answer or {}).get("answers") or {}).get(key) or (answer or {}).get(key)
    if not isinstance(node, dict):
        return float(node) if isinstance(node, (int, float)) else None
    if node.get("type") == "score":
        levels = node.get("legend") or node.get("criteria") or {}
        top = max(len(levels) - 1, 1)
        return float(node["score"]) / top if isinstance(node.get("score"), (int, float)) else None
    for field in ("noul", "value", "probability", "p", "true"):
        if isinstance(node.get(field), (int, float)):
            return float(node[field])
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alfworld", type=Path, default=None, help="run directory with trajectories.jsonl")
    p.add_argument("--view", type=Path, default=None, help="judge view jsonl (DeepSWE)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=["liquid", "clef", "d3"], default="liquid")
    p.add_argument("--schema", choices=list(SCHEMAS), default="outcome",
                   help="outcome asks the four questions scored everywhere else; gate asks the one "
                        "choice question these models are sold for -- continue, retry, ask or stop")
    p.add_argument("--model", default="LiquidAI/d1-omni-600M")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--obs-chars", type=int, default=400)
    p.add_argument("--act-chars", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--limit", type=int, default=0, help="episodes (debug)")
    a = p.parse_args()

    global SCHEMA
    SCHEMA = SCHEMAS[a.schema]

    if a.alfworld:
        view = alfworld_view(a.alfworld)
    else:
        view = [json.loads(line) for line in open(a.view) if line.strip()]
    view = view[: a.limit or None]

    jobs: list[tuple[str, int, str]] = []
    for episode in view:
        steps = episode["steps"]
        for k in range(len(steps)):
            jobs.append((episode["id"], k,
                         prompt(episode["task"], steps[: k + 1], a.window, a.obs_chars, a.act_chars)))

    done: set[tuple[str, int]] = set()
    if a.out.exists():
        field = "choice" if a.schema == "gate" else "p"
        kept = [line for line in open(a.out)
                if line.strip() and json.loads(line).get(field) is not None]
        with open(a.out, "w") as handle:
            handle.writelines(kept)
        done = {(json.loads(line)["id"], json.loads(line)["step"]) for line in kept}
        print(f"[decision-judge] resuming: {len(done)} usable verdicts kept", flush=True)
    asked = len(jobs)
    jobs = [j for j in jobs if (j[0], j[1]) not in done]
    print(f"[decision-judge] {len(jobs)} prefixes to score with {a.model}", flush=True)
    if done and not jobs:
        print("[decision-judge] WARNING: every prefix was already answered in this file. "
              "If this run meant to use a different model, its verdicts are not in it.", flush=True)

    backend = {"liquid": load_liquid, "clef": load_clef, "d3": load_d3}[a.backend](a.model, a.device)
    out = open(a.out, "a")
    started = time.time()
    for index in range(0, len(jobs), a.batch):
        chunk = jobs[index: index + a.batch]
        states = [text for _, _, text in chunk]
        try:
            if a.backend == "liquid":
                answers = ask_liquid(backend, states)
            elif a.backend == "d3":
                answers = ask_d3(backend, states)
            else:
                answers = ask_clef(backend, states, a.model, a.max_length)
        except Exception as exc:  # noqa: BLE001 - a bad batch must not end the sweep
            answers = [{"error": repr(exc)} for _ in chunk]
        if index == 0:
            print("[decision-judge] first answer:", json.dumps(answers[0])[:500], flush=True)
        for (episode_id, step, _), answer in zip(chunk, answers):
            if a.schema == "gate":
                node = (answer.get("answers") or {}).get("action") or {}
                shares = node.get("probabilities") or {}
                row = {"id": episode_id, "step": step, "choice": node.get("choice"),
                       **{f"p_{k}": float(v) for k, v in shares.items()}}
                if row["choice"] is None:
                    row["raw"] = json.dumps(answer)[:300]
            else:
                node = (answer.get("answers") or {}).get("progress")
                if node is not None:
                    answer = dict(answer, answers=dict(answer["answers"],
                                  progress=to_our_shape("progress", node, QUESTIONS["progress"])))
                row = {"id": episode_id, "step": step,
                       "p": value_of(answer, "success"),
                       "stuck": value_of(answer, "stuck"),
                       "stale": value_of(answer, "stale_evidence"),
                       "progress": value_of(answer, "progress")}
                if row["p"] is None:
                    row["raw"] = json.dumps(answer)[:300]
            out.write(json.dumps(row) + "\n")
        if (index // a.batch) % 20 == 0:
            out.flush()
            rate = (index + len(chunk)) / max(time.time() - started, 1e-6)
            print(f"[decision-judge] {index + len(chunk)}/{len(jobs)}  {rate:.1f}/s", flush=True)
    out.close()
    print(f"[decision-judge] done -> {a.out}")


if __name__ == "__main__":
    main()
