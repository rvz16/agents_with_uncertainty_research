"""OSWorld episodes in the shape experiments.prr_report_v2 expects.

The colleague's dataset (ArtemVazhentsev21/osworld-qwen-trajectories) already
writes ``trajectory-uq/v1``: ``uq/episodes.jsonl`` holds one record per episode
with per-step ``signals``, per-step ``critics`` and per-step ``metadata``. So
this is a rename, not a reconstruction -- the only real decisions are:

* the target is ``features.osworld_score`` (continuous), not the ``success``
  flag: the two disagree on the 10 episodes with partial credit, and the
  continuous score is what the DeepSWE half of the paper already uses;
* signal keys are renamed to the ALFWorld spelling (``sum_logprob``,
  ``mean_token_entropy``, ...) so one method table covers every cohort;
* the four per-step critics are kept under their own names -- the Bayes critic
  model reads whatever keys the episode carries, so inventing a mapping onto
  the ALFWorld five would only double-count ``action_format_valid``.
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path
from typing import Any

#: our label -> the key the colleague's collector writes per step
SIGNAL_KEYS = {
    "Logprob": ("sum_logprob", "sum_logprob"),
    "Perplexity": ("perplexity", "perplexity"),
    "MTE": ("mean_token_entropy", "mean_token_entropy"),
    "Self-certainty": ("self_certainty", "self_certainty"),
    "Verb actions": ("verbalized_confidence", "verb_action"),
}

#: in the shared SIGNALS table but never collected here; the methods that need
#: them drop out on coverage rather than crashing on a missing key
ABSENT = ("LLM judge", "Self-certainty (action)", "SAUP-PD (MTE)", "UProp H_t")

APPS = {"libreoffice calc": "libreoffice_calc", "calc": "libreoffice_calc",
        "vscode": "vs_code", "OS": "os"}

#: OSWorld's own domain per task id (evaluation_examples/test_all.json). The
#: dataset's metadata.related_apps is NOT this: it names one application even
#: for the 92 episodes OSWorld files under multi_apps, so folds stratified on
#: related_apps would silently mix the hardest split into every other domain.
_INDEX = Path(__file__).resolve().parent / "osworld_index.json"


def _domains() -> dict[str, str]:
    if not _INDEX.exists():
        return {}
    idx = json.loads(_INDEX.read_text())
    return {task: domain for domain, tasks in idx.items() for task in tasks}


DOMAIN = _domains()


def _app(record: dict) -> str:
    canonical = DOMAIN.get(record.get("episode_id"))
    if canonical:
        return canonical
    apps = (record.get("metadata") or {}).get("related_apps") or ["unknown"]
    name = str(apps[0]).strip()
    return APPS.get(name, name.replace(" ", "_").lower())


def load_osworld(runs: list[Path], harness: str = "OQ") -> dict[str, dict[str, Any]]:
    """Later runs win: the recovery runs re-ran tasks the main run lost."""
    episodes: dict[str, dict[str, Any]] = {}
    for run in runs:
        path = run / "uq" / "episodes.jsonl"
        for line in open(path):
            if not line.strip():
                continue
            raw = json.loads(line)
            gens = raw.get("generations") or []
            if not gens:
                continue
            score = (raw.get("features") or {}).get("osworld_score")
            if score is None:
                continue
            seqs: dict[str, list[float]] = {}
            for name, (_, src) in SIGNAL_KEYS.items():
                vals = [(g.get("signals") or {}).get(src) for g in gens]
                seqs[name] = [float(v) for v in vals
                              if v is not None and math.isfinite(float(v))]
            for name in ABSENT:
                seqs[name] = []
            crit = [{k: bool(v) for k, v in (g.get("critics") or {}).items()} for g in gens]
            ep_crit = {k: bool(v) for k, v in (raw.get("critics") or {}).items() if v is not None}
            mte = [(g.get("signals") or {}).get("mean_token_entropy") for g in gens]
            mean_lp = [(g.get("signals") or {}).get("mean_token_logprob") for g in gens]
            episodes[raw["episode_id"]] = {
                "id": raw["episode_id"], "harness": harness, "app": _app(raw),
                "score": float(score), "label": int(float(score) > 0.5),
                "signals": seqs, "critics": crit, "episode_critics": ep_crit,
                "n_steps": len(gens),
                "tool_rate": st.fmean([v for c in crit for v in c.values()] or [0.0]),
                "cross_logprob": [float(v) for v in mean_lp
                                  if v is not None and math.isfinite(float(v))],
                "mte_steps": mte, "saup": [None] * len(gens),
                "judge_steps": [None] * len(gens), "uprop": None,
                "htc_tokens": ([], []),
                "record": _record(raw, gens, crit),
                "state_change": [(g.get("signals") or {}).get("state_change_score") for g in gens],
                "repeated": [bool((g.get("metadata") or {}).get("repeated_action")) for g in gens],
                "cycles": [bool((g.get("metadata") or {}).get("action_cycle")) for g in gens],
                "no_effect": [bool((g.get("metadata") or {}).get("no_effect_loop")) for g in gens],
            }
    return episodes


def _record(raw: dict, gens: list[dict], crit: list[dict]) -> dict:
    out = []
    for k, (g, c) in enumerate(zip(gens, crit)):
        sig = {}
        for _, (ours, src) in SIGNAL_KEYS.items():
            v = (g.get("signals") or {}).get(src)
            if v is not None and math.isfinite(float(v)):
                sig[ours] = float(v)
        out.append({"index": k, "signals": sig, "critics": dict(c)})
    score = (raw.get("features") or {}).get("osworld_score")
    return {"episode_id": raw["episode_id"], "environment": "osworld",
            "success": int(float(score) > 0.5), "generations": out}
