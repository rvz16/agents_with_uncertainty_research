"""Per-token confidences of a DeepSWE run for HTC (Zhang et al., ICML 2026).

One row per assistant step, indexed as in compact_jobs.py: ``{"id", "step",
"top1": [p_i], "topk": [mean top-k p_i]}`` with p_i = exp(logprob of the
generated token) and the top-k mean from the recorded top_logprobs (k = 20).
Reads the jobs.zip archive or an extracted directory.

    python -m experiments.deepswe_htc_tokens jobs.zip --run deepswe_gptoss_113_v5 --out runs/deepswe_gptoss_113_v5/htc_tokens.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path


def rows(tid: str, tr: dict) -> list[dict]:
    out, k = [], 0
    for m in tr.get("messages", []):
        if m.get("role") != "assistant":
            continue
        resp = ((m.get("extra") or {}).get("response") or {})
        content = ((resp.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
        top1, topk = [], []
        for t in content:
            lp = t.get("logprob")
            if lp is None:
                continue
            top1.append(round(math.exp(float(lp)), 6))
            alts = [a.get("logprob") for a in (t.get("top_logprobs") or []) if a.get("logprob") is not None]
            topk.append(round(sum(math.exp(float(a)) for a in alts) / len(alts), 6) if alts else round(math.exp(float(lp)), 6))
        out.append({"id": tid, "step": k, "top1": top1, "topk": topk}); k += 1
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zip", type=Path); p.add_argument("--run", required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    prefix = a.run.rstrip("/") + "/"
    if a.zip.is_dir():
        tasks = sorted(d.name for d in (a.zip / a.run).iterdir() if d.is_dir())
        read = lambda member: (a.zip / member).read_bytes()  # noqa: E731
    else:
        z = zipfile.ZipFile(a.zip); names = z.namelist()
        if not any(n.startswith(prefix) for n in names):
            prefix = ""  # a per-run archive has no run directory
        tasks = sorted({n[len(prefix):].split("/")[0] for n in names if n.startswith(prefix) and "/" in n[len(prefix):]})
        read = z.read
    n = 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as out:
        for t in tasks:
            try:
                tr = json.loads(read(f"{prefix}{t}/agent/mini-swe-agent.trajectory.json"))
            except (KeyError, FileNotFoundError):
                continue
            for r in rows(t, tr):
                out.write(json.dumps(r) + "\n"); n += 1
    print(f"wrote {n} steps to {a.out}")


if __name__ == "__main__":
    main()
