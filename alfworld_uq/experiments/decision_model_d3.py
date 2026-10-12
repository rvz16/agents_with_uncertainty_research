"""Decision 3.0 over our trajectory states, at every released size.

The family is six models on five Qwen backbones -- 26B on Qwen3.8-27B, 8.4B on
Qwen3.5-9B, then 4.5B, 2.2B and two under a billion -- which is what the
earlier comparison lacked. Against Clef we could say only that the 9B head
loses to its own backbone and the 27B breaks even; with five sizes the question
becomes whether that is a trend in scale or a property of one release. Two of
these sit on exactly the backbones Clef uses, so they are also a second vendor
on the same foundation.

The call is the family's own ``system_one``, read from its released module
rather than reinvented: ``model.system_one(state=..., questions=...)`` returns
one probability per option with no text generated. A noul already arrives in
the shape our scorer reads. A score question arrives choice-shaped, with a
probability per level, so it is reduced here to the expected level over the
legend -- which is the quantity the question asks for, and keeps the ordinal
information a top-1 pick would throw away.

Resume is keyed on (cohort, id) and the answer file carries both the model and
the state set, because the four ALFWorld cohorts run the same 140 task ids and
a file keyed on the id alone silently reports a different run's numbers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.decision_state import QUESTIONS


def to_our_shape(name: str, answer: dict, question: dict) -> dict:
    """A score answer arrives choice-shaped: reduce it to the expected level."""
    if question.get("type") != "score" or not isinstance(answer, dict):
        return answer
    probabilities = answer.get("probabilities")
    legend = list(question.get("criteria") or [])
    if not isinstance(probabilities, dict) or not legend:
        return answer

    def share(index: int, key: str) -> float:
        """The release keys these by position -- "0", "1" -- not by the legend text."""
        for candidate in (str(index), index, key):
            if candidate in probabilities:
                return float(probabilities[candidate])
        return 0.0

    weights = [share(index, key) for index, key in enumerate(legend)]
    total = sum(weights)
    if total <= 0:
        return answer
    expected = sum(index * weight for index, weight in enumerate(weights)) / total
    return {"type": "score", "score": expected, "legend": legend,
            "probabilities": probabilities}


def resume(path: Path) -> set[tuple[str, str]]:
    """Keep only the rows that carry a probability, so an errored row is retried."""
    if not path.exists():
        return set()
    kept = []
    for line in open(path):
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" not in (row.get("answer") or {"error": 1}):
            kept.append(line)
    with open(path, "w") as handle:
        handle.writelines(kept)
    done = {(json.loads(line)["cohort"], json.loads(line)["id"]) for line in kept}
    print(f"[d3] resuming {path.name}: {len(done)} usable answers kept", flush=True)
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="vllm-sr/d3-flash")
    p.add_argument("--tag", default="", help="names the state set in the output file")
    p.add_argument("--device", default="cuda:0",
                   help="the family's loader rejects a device without an index: 'cuda' raises "
                        "ValueError, 'cuda:0' is what it wants")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    import torch
    from transformers import AutoModel

    a.out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or a.states[0].parent.name
    answers_path = a.out / f"answers_d3_{a.model.replace('/', '_')}_{tag}.jsonl"
    done = resume(answers_path)

    rows = []
    for path in a.states:
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                if (row["cohort"], row["id"]) not in done:
                    rows.append(row)
    rows = rows[: a.limit or None]
    print(f"[d3] {len(rows)} states to score with {a.model}", flush=True)
    if not rows:
        print("[d3] nothing to ask -- every state already has an answer. "
              "Check the model and tag if this is unexpected.", flush=True)
        return

    model = AutoModel.from_pretrained(a.model, trust_remote_code=True,
                                      dtype=torch.bfloat16, device_map=a.device).eval()
    out = open(answers_path, "a")
    started = time.time()
    for index, row in enumerate(rows, 1):
        try:
            with torch.no_grad():
                result = model.system_one(state=row["state"], questions=QUESTIONS)
            answers = {name: to_our_shape(name, value, QUESTIONS[name])
                       for name, value in (result.get("answers") or {}).items()}
            answer = {"model": a.model, "answers": answers}
            if not answers:
                answer = {"error": "no answers", **answer}
        except Exception as exc:  # noqa: BLE001
            answer = {"error": repr(exc)[:300]}
        if index == 1:
            print("[d3] first answer:", json.dumps(answer)[:400], flush=True)
        out.write(json.dumps({"id": row["id"], "cohort": row["cohort"], "label": row["label"],
                              "score": row["score"], "answer": answer}) + "\n")
        if index % 50 == 0:
            out.flush()
            print(f"[d3] {index}/{len(rows)}  {index / max(time.time() - started, 1e-6):.2f}/s", flush=True)
    out.close()
    print(f"[d3] done -> {answers_path}")


if __name__ == "__main__":
    main()
