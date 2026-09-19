"""SAUP surrogate distances per step (Zhao et al., ACL 2025, "Uncertainty Propagation on LLM Agent").

Two situational distances for every step of a trajectory, from sentence
embeddings (all-MiniLM-L6-v2, cosine distance):

* inquiry drift  D_a = dist(question, thought + action + observation of the step)
* inference gap  D_o = dist(observation, thought + action of the step)

The paper computes "plain distances" with a RoBERTa fine-tuned on SQuAD v2 and
does not release code; a sentence encoder is the closest reproducible stand-in.
Output: one row per step, ``{"id", "step", "da", "do"}``, indexed exactly as the
cohort's steps (ALFWorld: trajectories.jsonl rows; DeepSWE: judge_view steps).

    python -m experiments.saup_distances --alfworld runs/<run> --out runs/<run>/saup_dist.jsonl
    python -m experiments.saup_distances --view deep_swe_uq/runs/<run>/judge_view.jsonl --out .../saup_dist.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHARS = 1500  # the encoder's context is short; keep the head of long shell outputs


def alfworld_episodes(run: Path) -> list[dict]:
    eps: dict[str, dict] = {}
    for line in open(run / "trajectories.jsonl"):
        if line.strip():
            r = json.loads(line)
            e = eps.setdefault(r["episode_id"], {"id": r["episode_id"], "task": r["task"], "steps": []})
            e["steps"].append({"thought": str(r.get("thought") or ""), "action": str(r.get("action") or ""),
                               "observation": str(r.get("observation") or "")})
    return list(eps.values())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alfworld", type=Path, default=None)
    p.add_argument("--view", type=Path, default=None, help="DeepSWE judge_view.jsonl (task, action, observation)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=128)
    a = p.parse_args()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    episodes = alfworld_episodes(a.alfworld) if a.alfworld else [json.loads(l) for l in open(a.view) if l.strip()]
    texts, index = [], []
    for e in episodes:
        for k, s in enumerate(e["steps"]):
            thought, action, obs = s.get("thought", ""), s.get("action", ""), s.get("observation", "")
            ta = (thought + "\n" + action).strip()[:CHARS]
            index.append((e["id"], k, len(texts)))
            texts += [e["task"][:CHARS], (ta + "\n" + obs).strip()[:CHARS], obs.strip()[:CHARS] or " ", ta or " "]
    emb = model.encode(texts, batch_size=a.batch, normalize_embeddings=True, show_progress_bar=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for eid, k, i in index:
            q, tao, obs, ta = emb[i], emb[i + 1], emb[i + 2], emb[i + 3]
            f.write(json.dumps({"id": eid, "step": k, "da": float(1.0 - q @ tao), "do": float(1.0 - obs @ ta)}) + "\n")
    print(f"wrote {len(index)} steps of {len(episodes)} episodes to {a.out}")


if __name__ == "__main__":
    main()
