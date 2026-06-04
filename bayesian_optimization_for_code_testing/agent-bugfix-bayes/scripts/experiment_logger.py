"""ExperimentLogger — live console + JSONL events stream for benchmark runs.

Drop-in helper for any experiment runner. Two outputs:

1. **Console** — pretty per-instance summary with running aggregates
   (fix%, Ū, $cost, ETA), compact action traces, and ★-flagged
   "interesting" events (forced refines that catch, large posterior
   shifts, exceptions).

2. **JSONL events file** — every state-changing event is recorded as one
   JSON line with a timestamp. Auto-named by experiment name + model +
   start time, e.g. `kshot_K2_online__gpt-5-mini__2026-06-04_153012.events.jsonl`.

Usage:
    log = ExperimentLogger(
        name="kshot_K2_online", model="openai/gpt-5-mini",
        output_dir=Path("logs"), n_total=20,
    )
    log.boot({"K": 2, "mode": "online", ...})
    for i, tid in enumerate(test_ids):
        log.instance_start(i, tid, meta={"cf_rating": 1200})
        # ... run ...
        log.action(step=0, action="generate", belief_before=0.5, belief_after=0.52)
        log.action(step=1, action="verify", ok=False)
        log.forced_refine(catch=False, belief_at_bail=0.05,
                          alpha=1.2, beta=3.8)  # ★ flagged
        log.instance_done(rec)
    log.cell_done(final_stats)

Event schema (JSONL):
    {"t": "<iso>", "kind": "boot",          "data": {...}}
    {"t": "<iso>", "kind": "instance_start","instance": "X", "i": 5, "n": 20, "meta": {...}}
    {"t": "<iso>", "kind": "action",        "instance": "X", "step": 0, "action": "generate",
                                            "belief_before": ..., "belief_after": ...}
    {"t": "<iso>", "kind": "forced_refine", "instance": "X", "catch": false,
                                            "belief_at_bail": ..., "posterior": {"alpha": .., "beta": ..}}
    {"t": "<iso>", "kind": "kernel_update", "delta_mean": ..., "alpha": .., "beta": ..}
    {"t": "<iso>", "kind": "instance_done", "instance": "X", "fixed": false, "cost": 39.0,
                                            "running": {"fix_pct": .., "U_mean": .., "dollars": ..}}
    {"t": "<iso>", "kind": "exception",     "instance": "X", "error": "<str>"}
    {"t": "<iso>", "kind": "cell_done",     "summary": {...}}
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------- pricing
# Output cost only — OpenRouter input prices are negligible for these jobs.
# Values per million tokens (USD). Used for the live $ estimate.
# Add new models as needed; missing models display "?" instead of $.
_MODEL_PRICE_PER_MTOK = {
    "openai/gpt-5-mini":         {"in": 0.250,  "out": 2.000},
    "openai/gpt-oss-20b:free":   {"in": 0.000,  "out": 0.000},
    "anthropic/claude-haiku-4.5":{"in": 0.800,  "out": 4.000},
    "anthropic/claude-sonnet-4.5":{"in": 3.000, "out": 15.000},
    "qwen/qwen3-coder":          {"in": 0.300,  "out": 1.200},
}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _short_action(action: str) -> str:
    """gen / crit:L2_public_tests / ver / bail / fgen / fver / think / finish."""
    if action == "generate":         return "gen"
    if action == "generate_on_bail": return "★fgen"
    if action == "verify":           return "ver"
    if action == "verify_on_bail":   return "★fver"
    if action == "bail_out" or action == "bail": return "bail"
    if action.startswith("generate:"):
        return f"gen({action.split(':', 1)[1][:4]})"
    if action.startswith("critic:"):
        return f"crit:{action.split(':', 1)[1][:8]}"
    return action[:10]


# --------------------------------------------------------------------- main class


class ExperimentLogger:
    """One logger per experiment cell. Writes JSONL stream + pretty console."""

    def __init__(
        self,
        name: str,
        model: str,
        output_dir: Path,
        n_total: int,
        verbose: bool = True,
    ) -> None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        model_slug = _slug(model.split("/")[-1])  # strip provider prefix
        self.name = name
        self.model = model
        self.n_total = n_total
        self.verbose = verbose
        output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = output_dir / f"{_slug(name)}__{model_slug}__{ts}.events.jsonl"
        self._fh = open(self.events_path, "w", buffering=1)  # line-buffered
        self._t0 = time.time()

        # running aggregates
        self._n_done = 0
        self._n_fixed = 0
        self._sum_cost = 0.0
        self._sum_U = 0.0
        self._dollars = 0.0
        self._tokens_in = 0
        self._tokens_out = 0

        # per-instance state
        self._cur_actions: list[str] = []
        self._cur_instance: str | None = None
        self._cur_start: float = 0.0

    # ------------------------------------------------------------------ JSONL
    def _emit(self, kind: str, **data: Any) -> None:
        self._fh.write(json.dumps({"t": _now_iso(), "kind": kind, **data},
                                  default=str) + "\n")

    # ------------------------------------------------------------------ helpers
    def _running(self) -> dict:
        return {
            "fix_pct": (100.0 * self._n_fixed / self._n_done) if self._n_done else 0.0,
            "U_mean": (self._sum_U / self._n_done) if self._n_done else 0.0,
            "cost_mean": (self._sum_cost / self._n_done) if self._n_done else 0.0,
            "dollars": self._dollars,
            "tokens_in": self._tokens_in,
            "tokens_out": self._tokens_out,
        }

    def _estimate_dollars(self, tokens_in: int, tokens_out: int) -> float:
        prices = _MODEL_PRICE_PER_MTOK.get(self.model)
        if not prices:
            return 0.0
        return (tokens_in / 1e6) * prices["in"] + (tokens_out / 1e6) * prices["out"]

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------ public API
    def boot(self, config: dict) -> None:
        self._emit("boot", name=self.name, model=self.model, n_total=self.n_total,
                   config=config, events_path=str(self.events_path))
        self._say(f"=== {self.name} | {self.model} | n={self.n_total} ===")
        self._say(f"events → {self.events_path}")
        kv = "  ".join(f"{k}={v}" for k, v in config.items())
        if kv:
            self._say(f"config: {kv}")

    def instance_start(self, i: int, instance_id: str, meta: dict | None = None) -> None:
        self._cur_instance = instance_id
        self._cur_actions = []
        self._cur_start = time.time()
        self._emit("instance_start", instance=instance_id, i=i, n=self.n_total,
                   meta=meta or {})
        elapsed_min = (time.time() - self._t0) / 60
        rate = max(1, self._n_done)
        eta_min = (self.n_total - i) * (elapsed_min / rate) if self._n_done else 0
        meta_str = " ".join(f"{k}={v}" for k, v in (meta or {}).items())
        self._say(f"\n[{i+1}/{self.n_total}] {instance_id}  {meta_str}  "
                  f"elapsed={elapsed_min:.1f}m  ETA={eta_min:.1f}m")

    def action(self, step: int, action: str, **deltas: Any) -> None:
        """Log one in-episode action with belief/likelihood deltas.

        deltas: any of belief_before, belief_after, ok, passed, posterior, ...
        """
        self._emit("action", instance=self._cur_instance, step=step,
                   action=action, **deltas)
        # Pretty trace token for end-of-instance summary
        tok = _short_action(action)
        if "ok" in deltas:
            tok += "(✓)" if deltas["ok"] else "(✗)"
        elif "passed" in deltas:
            tok += "(✓)" if deltas["passed"] else "(✗)"
        if "belief_after" in deltas:
            tok += f"[b={deltas['belief_after']:.2f}]"
        self._cur_actions.append(tok)

    def forced_refine(self, catch: bool, **state: Any) -> None:
        """Log a forced-refine outcome (Y_t=0, Y_{t+1}=catch). ★ when catch=True."""
        self._emit("forced_refine", instance=self._cur_instance, catch=catch, **state)
        flag = "★ CAUGHT" if catch else "miss"
        self._say(f"  forced refine: {flag}  state={state}")

    def kernel_update(self, **posterior: Any) -> None:
        self._emit("kernel_update", instance=self._cur_instance, **posterior)
        # Don't print every kernel update; consolidated in instance_done.

    def llm_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Tally one LLM call for live $-estimate."""
        self._tokens_in += prompt_tokens
        self._tokens_out += completion_tokens
        self._dollars = self._estimate_dollars(self._tokens_in, self._tokens_out)
        self._emit("llm_usage", instance=self._cur_instance,
                   prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                   running_dollars=self._dollars)

    def exception(self, error: str) -> None:
        self._emit("exception", instance=self._cur_instance, error=error)
        self._say(f"  ⚠ EXCEPTION: {error}")

    def instance_done(self, rec: dict, R: float = 100.0) -> None:
        """rec must have: fixed, total_cost (BDP units), final_action, actions."""
        fixed = bool(rec.get("fixed"))
        cost = float(rec.get("total_cost", 0.0))
        U = R * int(fixed) - cost
        wall = time.time() - self._cur_start

        self._n_done += 1
        if fixed: self._n_fixed += 1
        self._sum_cost += cost
        self._sum_U += U

        run = self._running()
        proj_dollars = (self._dollars / self._n_done) * self.n_total if self._n_done else 0

        self._emit("instance_done", instance=self._cur_instance,
                   fixed=fixed, total_cost=cost, U=U,
                   final_action=rec.get("final_action"),
                   wall_clock=wall,
                   running=run)

        # Compact action trace, then per-instance + running line
        trace = " → ".join(self._cur_actions) if self._cur_actions else "(no actions logged)"
        tag = "✓" if fixed else "✗"
        self._say(f"  {tag} fix={fixed!s:<5}  cost={cost:5.1f}  U={U:+6.1f}  "
                  f"wc={wall:.1f}s  final={rec.get('final_action')}")
        self._say(f"  trace: {trace}")
        self._say(f"  running: fix={run['fix_pct']:.0f}%  "
                  f"Ū={run['U_mean']:+.2f}  "
                  f"$={self._dollars:.3f} (proj ${proj_dollars:.2f})  "
                  f"tok={self._tokens_in/1000:.1f}k/{self._tokens_out/1000:.1f}k")

    def cell_done(self, extras: dict | None = None) -> None:
        run = self._running()
        elapsed_min = (time.time() - self._t0) / 60
        self._emit("cell_done", summary={**run, "elapsed_min": elapsed_min,
                                         **(extras or {})})
        self._say("\n=== Final aggregate ===")
        self._say(f"  n={self._n_done}  fix={run['fix_pct']:.0f}%  "
                  f"Ū={run['U_mean']:+.2f}  cost={run['cost_mean']:.2f}")
        self._say(f"  total $={self._dollars:.3f}  "
                  f"tokens in/out={self._tokens_in/1000:.1f}k/{self._tokens_out/1000:.1f}k")
        self._say(f"  wall={elapsed_min:.1f} min")
        if extras:
            for k, v in extras.items():
                self._say(f"  {k}: {v}")
        self._say(f"\nevents log: {self.events_path}")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args
        self.close()
        return False
