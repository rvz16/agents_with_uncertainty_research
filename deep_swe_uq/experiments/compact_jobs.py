"""Reduce a pier jobs archive to one compact record per task.

With top_logprobs=20 a single DeepSWE trajectory is 250-330 MB (mini-swe-agent
stores the full server response for every step), which is more than a laptop
can hold for 113 tasks. The table needs only per-step summaries: mean token
log-probability, mean token entropy over the renormalised top-k head, token
count, the command issued and its return code. This streams the zip and keeps
exactly that, plus result.json, the patch size and the agent's exit status.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import zipfile
from pathlib import Path
from typing import Any


def head_entropy(top: list[dict[str, Any]]) -> float | None:
    vals = [float(t["logprob"]) for t in top or [] if t.get("logprob") is not None]
    if not vals:
        return None
    probs = [math.exp(v) for v in vals]
    mass = sum(probs)
    if mass <= 0:
        return None
    return float(-sum((p / mass) * math.log(p / mass) for p in probs if p > 0))


def compact(tr: dict[str, Any]) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for m in tr.get("messages", []):
        role = m.get("role")
        if role == "assistant":
            resp = ((m.get("extra") or {}).get("response") or {})
            choice = (resp.get("choices") or [{}])[0]
            content = ((choice.get("logprobs") or {}).get("content")) or []
            lps = [float(t["logprob"]) for t in content if t.get("logprob") is not None]
            ents = [e for e in (head_entropy(t.get("top_logprobs")) for t in content) if e is not None]
            cmd = None
            for tc in (choice.get("message") or {}).get("tool_calls") or []:
                try:
                    c = json.loads(tc["function"]["arguments"]).get("command", "")
                    cmd = c if isinstance(c, str) else " ".join(map(str, c))
                except Exception:  # noqa: BLE001
                    cmd = None
            pending = {
                "num_tokens": len(lps),
                "mean_logprob": st.fmean(lps) if lps else None,
                "min_logprob": min(lps) if lps else None,
                "mean_entropy": st.fmean(ents) if ents else None,
                "max_entropy": max(ents) if ents else None,
                "command": cmd,
                "returncode": None,
                "format_error": False,
            }
            steps.append(pending)
        elif role in ("tool", "user") and pending is not None:
            text = str(m.get("content") or "")
            if role == "tool":
                try:
                    pending["returncode"] = int(json.loads(text).get("returncode"))
                except Exception:  # noqa: BLE001
                    pending["returncode"] = None
            if "Tool call error" in text or "format error" in text.lower():
                pending["format_error"] = True
            pending = None
    info = tr.get("info", {}) or {}
    return {"exit_status": info.get("exit_status"), "steps": steps}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zip", type=Path)
    p.add_argument("--run", required=True, help="top-level directory inside the archive, e.g. deepswe_gptoss_113_v2")
    p.add_argument("--out", type=Path, required=True, help="jsonl, one record per task")
    a = p.parse_args()
    z = zipfile.ZipFile(a.zip)
    prefix = a.run.rstrip("/") + "/"
    tasks = sorted({n[len(prefix):].split("/")[0] for n in z.namelist() if n.startswith(prefix) and "/" in n[len(prefix):]})
    written = 0
    with open(a.out, "w") as out:
        for t in tasks:
            base = f"{prefix}{t}/"
            try:
                result = json.loads(z.read(base + "result.json"))
            except KeyError:
                continue
            try:
                patch_size = z.getinfo(base + "artifacts/model.patch").file_size
            except KeyError:
                patch_size = 0
            try:
                tr = json.loads(z.read(base + "agent/mini-swe-agent.trajectory.json"))
            except KeyError:
                tr = {}
            rec = {"id": t, "patch_bytes": patch_size,
                   "rewards": (result.get("verifier_result") or {}).get("rewards") or {},
                   **compact(tr)}
            out.write(json.dumps(rec) + "\n")
            written += 1
            print(f"[{written:3d}/{len(tasks)}] {t}: {len(rec['steps'])} steps, {rec['exit_status']}, patch {patch_size} B", flush=True)
    print(f"wrote {written} records to {a.out}")


if __name__ == "__main__":
    main()
