"""Ask a System One decision model for a belief about every recorded trajectory.

Runs against any server that speaks the TypeSafe `/v1/systemone` shape: PostHog
Jeeves (``python -m inference.serve``), Laya's router, Cloudflare Clef on
Workers AI, or the TypeSafe API itself. The states come from
``experiments.decision_state``; the schema is its QUESTIONS.

Only the answers are written here. Scoring stays with the report, so the same
folds and seeds apply to this estimator as to every other one.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.decision_state import QUESTIONS


def ask(session, url: str, state: str, questions: dict, options: dict, token: str | None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = {"state": state, "questions": questions, "options": options}
    for attempt in range(4):
        try:
            response = session.post(url, json=payload, headers=headers, timeout=180)
            if response.status_code == 200:
                return response.json()
            error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:  # noqa: BLE001 - the server may still be warming up
            error = repr(exc)
        time.sleep(2 * (attempt + 1))
    return {"error": error}


def main() -> None:
    import requests

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", type=Path, nargs="+", required=True, help="files from decision_state")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--url", default="http://localhost:8009/v1/systemone")
    p.add_argument("--model", default=None, help="sent as `model` when the endpoint needs one")
    p.add_argument("--token-env", default=None, help="env var holding a bearer token")
    p.add_argument("--think", action="store_true", help="let the model reason before it answers")
    p.add_argument("--max-think", type=int, default=768)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    options = {"think": bool(a.think), "max_think": a.max_think, "return_reasoning": bool(a.think)}
    questions = dict(QUESTIONS)
    token = os.environ.get(a.token_env) if a.token_env else None

    from experiments.decision_model_clef import _resume

    done = _resume(a.out, "decision")
    out = open(a.out, "a")

    rows = []
    for path in a.states:
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                if (row["cohort"], row["id"]) not in done and row["id"] not in done:
                    rows.append(row)
    rows = rows[: a.limit or None]
    print(f"[decision] {len(rows)} trajectories to ask about", flush=True)

    session = requests.Session()
    with ThreadPoolExecutor(a.workers) as pool:
        def one(row):
            payload = dict(questions)
            body = {"state": row["state"]}
            if a.model:
                body["model"] = a.model
            answer = ask(session, a.url, row["state"], payload, options, token)
            return row, answer

        for index, (row, answer) in enumerate(pool.map(one, rows), 1):
            out.write(json.dumps({"id": row["id"], "cohort": row["cohort"],
                                  "label": row["label"], "score": row["score"],
                                  "answer": answer}) + "\n")
            if index % 25 == 0:
                out.flush(); print(f"[decision] {index}/{len(rows)}", flush=True)
    out.close()
    print(f"[decision] done -> {a.out}")


if __name__ == "__main__":
    main()
