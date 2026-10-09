"""Ask Cloudflare's Clef about every recorded trajectory, in process.

Clef ships weights and `joint_schema_model.py` on the Hub but no server, so this
loads the release model directly and writes the same answer file as
``decision_model_query``: one JSON object per trajectory, with the probabilities
under ``answer.answers``, so ``decision_model_score`` reads either unchanged.

The conversion to a SystemOne answer body is the release's own `systemone`
helper, so a noul's positive column is the one the release names `true` rather
than a column this script guessed at.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from experiments.decision_state import QUESTIONS


def load(model_id: str, device: str):
    from huggingface_hub import snapshot_download

    path = snapshot_download(model_id)
    sys.path.insert(0, path)
    import joint_schema_model as jsm  # noqa: E402 - only importable once downloaded

    model, processor = jsm.load_release_model(path, device=device)
    model.eval()
    return jsm, model, processor


def answer_for(jsm, model, processor, state: str, model_id: str, max_length: int) -> dict:
    """The release ships the /v1/systemone conversion itself; use it rather than
    reading the logits by hand. A noul's options are named, and `true` is one of
    them, so nothing here has to guess which column is the positive one."""
    request = {"model": model_id, "state": state, "questions": QUESTIONS}
    return jsm.systemone(model, processor, request, max_length=max_length)


def _resume(path: Path, tag: str) -> set[tuple[str, str]]:
    """Keep the answers that carry a probability and re-ask the rest.

    Keyed by (cohort, id): the four ALFWorld cohorts run the same 140 tasks, so
    an episode id alone names four different trajectories, and a sweep keyed by
    id drops three quarters of its work. An errored row is not an answer either:
    treating it as one is how a whole sweep came back empty.
    """
    if not path.exists():
        return set()
    kept = [line for line in open(path) if line.strip() and "error" not in json.loads(line)["answer"]]
    with open(path, "w") as handle:
        handle.writelines(kept)
    keys = {(json.loads(line)["cohort"], json.loads(line)["id"]) for line in kept}
    print(f"[{tag}] resuming: {len(keys)} usable answers kept", flush=True)
    return keys


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Cloudflare/clef-flash")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    jsm, model, processor = load(a.model, a.device)
    done = _resume(a.out, "clef")
    rows = []
    for path in a.states:
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                if (row["cohort"], row["id"]) not in done:
                    rows.append(row)
    rows = rows[: a.limit or None]
    print(f"[clef] {len(rows)} trajectories to ask about with {a.model}", flush=True)

    out = open(a.out, "a")
    for index, row in enumerate(rows, 1):
        started = time.time()
        try:
            answer = answer_for(jsm, model, processor, row["state"], a.model, a.max_length)
            answer["latency_ms"] = (time.time() - started) * 1000
        except Exception as exc:  # noqa: BLE001 - one bad record must not end the sweep
            answer = {"error": repr(exc)}
        if index == 1:
            print("[clef] first answer:", json.dumps(answer)[:600], flush=True)
        out.write(json.dumps({"id": row["id"], "cohort": row["cohort"], "label": row["label"],
                              "score": row["score"], "answer": answer}) + "\n")
        if index % 50 == 0:
            out.flush(); print(f"[clef] {index}/{len(rows)}", flush=True)
    out.close()
    print(f"[clef] done -> {a.out}")


if __name__ == "__main__":
    main()
