"""Shared transition-kernel utilities — post-hoc estimation + online updates.

The transition kernel describes Markovian dynamics of the per-instance
correctness signal Y across refinement steps:
  P(fix|broken)   = P(Y_{t+1}=1 | Y_t=0)
  P(break|correct) = P(Y_{t+1}=0 | Y_t=1)

This module exposes:

1) `compute_transition_kernel_from_pairs(pairs)` — Beta(α,β)-smoothed point
   estimate from a list of (Y_t, Y_{t+1}) pairs. Used by every "compute
   kernel after the run completes" script in the pipeline. Replaces the
   inline duplicates that previously lived in iter/refine.py, iter/kernel.py,
   analysis/compute_transition_kernel.py, and scripts/synthesis_transition_kernel.py.

2) `OnlineKernelCalibration` — Beta-Binomial running estimator queryable
   mid-loop. Thread-safe. Used by live agents that re-solve the Bayesian DP
   planner after each verify with an updated posterior. Small-N caveat: with
   fewer than ~10 transitions observed in a regime, the posterior is wide;
   the planner sees only the posterior mean. Inspect `.summary()` for the
   underlying counts when interpreting decisions.

3) `resolve_kernel(gen_dir, mode)` — three-way switch that callers use to pick
   between {measured-from-file, online-Beta-Binomial, hardcoded-default}.

`kernel_update(belief, kernel)` is the one-step belief propagation under the
kernel — same semantics whether the kernel is static or a snapshot from
`OnlineKernelCalibration.get()`.

Note: iter/swe_kernel.py keeps its own private compute_kernel because its
JSON output uses a different (flat, unsmoothed) schema consumed by the SWE
sections of analysis.ipynb. Migrating it would be a downstream-visible
change and is deliberately scoped out of this module.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


# Default values used by abbo + the synthesis-live agent for the initial
# uninformative kernel. Tunable per-pipeline by passing init_kernel= to
# OnlineKernelCalibration.
DEFAULT_KERNEL: dict = {"p_fix_broken": 0.50, "p_break_correct": 0.05}


def kernel_update(belief: float, kernel: Mapping[str, float]) -> float:
    """One-step belief propagation under a transition kernel.

    b' = b · (1 − p_break) + (1 − b) · p_fix

    Accepts either the lowercase schema {"p_fix_broken", "p_break_correct"}
    used by abbo / synthesis-live agents, or the uppercase schema
    {"P_fix_given_broken", "P_break_given_correct"} produced by
    `compute_transition_kernel_from_pairs` (typically under a `kernel_all`
    wrapper in JSON files).
    """
    p_fix = kernel.get("p_fix_broken", kernel.get("P_fix_given_broken"))
    p_break = kernel.get("p_break_correct", kernel.get("P_break_given_correct"))
    if p_fix is None or p_break is None:
        raise ValueError(
            "kernel missing p_fix_broken / p_break_correct "
            f"(or uppercase variants): {dict(kernel)!r}"
        )
    return belief * (1.0 - p_break) + (1.0 - belief) * p_fix


# ----------------------------------------------------------------------------
# Post-hoc kernel: Beta-smoothed point estimate from a fixed set of pairs
# ----------------------------------------------------------------------------

def _beta_smooth(success: int, total: int, alpha: float, beta: float) -> float:
    return (success + alpha) / (total + alpha + beta)


def compute_transition_kernel_from_pairs(
    pairs: Sequence[tuple[int, int]],
    *, alpha: float = 1.0, beta: float = 1.0,
) -> dict:
    """Beta(α, β)-smoothed transition kernel from (Y_t, Y_{t+1}) pairs.

    Y values must be 0 or 1; non-binary pairs are silently dropped to match
    legacy behavior. The Beta prior defaults to Laplace (alpha=beta=1).

    Schema (superset of the four variants this replaces):
        {
          "P_fix_given_broken":    p_fix,
          "P_stay_broken":         1 − p_fix     # smoothed independently
          "P_break_given_correct": p_break,
          "P_stay_correct":        1 − p_break   # smoothed independently
          "raw_counts": {"0->0": int, "0->1": int, "1->0": int, "1->1": int},
          "n_pairs": int,
          "n_broken_observed": int,    # pairs with Y_t == 0
          "n_correct_observed": int,   # pairs with Y_t == 1
          "smoothing": "Beta(alpha,beta)",
        }

    Note: when alpha == beta (e.g. Laplace), independent smoothing of stay-
    vs change-transitions exactly sums to 1 — both schemas are equivalent
    in that case. With asymmetric priors they can drift, which is by design
    (each Beta is its own posterior).
    """
    counts = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
    for y0, y1 in pairs:
        if y0 not in (0, 1) or y1 not in (0, 1):
            continue
        counts[f"{int(y0)}->{int(y1)}"] += 1
    n_broken = counts["0->0"] + counts["0->1"]
    n_correct = counts["1->0"] + counts["1->1"]
    p_fix = _beta_smooth(counts["0->1"], n_broken, alpha, beta)
    p_stay_broken = _beta_smooth(counts["0->0"], n_broken, alpha, beta)
    p_break = _beta_smooth(counts["1->0"], n_correct, alpha, beta)
    p_stay_correct = _beta_smooth(counts["1->1"], n_correct, alpha, beta)
    return {
        "P_fix_given_broken": p_fix,
        "P_stay_broken": p_stay_broken,
        "P_break_given_correct": p_break,
        "P_stay_correct": p_stay_correct,
        "raw_counts": counts,
        "n_pairs": n_broken + n_correct,
        "n_broken_observed": n_broken,
        "n_correct_observed": n_correct,
        "smoothing": f"Beta({alpha},{beta})",
    }


def pairs_from_trajectories(
    trajectories: Iterable[Iterable[Mapping]],
    *, y_key: str = "Y",
) -> list[tuple[int, int]]:
    """Build (Y_t, Y_{t+1}) pairs from per-instance step trajectories.

    Each trajectory is an iterable of step records, already sorted in step
    order. Pairs with missing or non-binary Y are dropped. This is the one
    helper the four post-hoc compute_kernel callers share — they each handle
    the upstream group-by-instance + sort themselves (the keys differ across
    scripts: "step", "patch_id", custom).
    """
    pairs: list[tuple[int, int]] = []
    for traj in trajectories:
        traj = list(traj)
        for i in range(len(traj) - 1):
            y0 = traj[i].get(y_key)
            y1 = traj[i + 1].get(y_key)
            if y0 is None or y1 is None:
                continue
            pairs.append((int(y0), int(y1)))
    return pairs


# ----------------------------------------------------------------------------
# Online kernel: Beta-Binomial running estimator
# ----------------------------------------------------------------------------

@dataclass
class _KernelCounts:
    n_broken: int = 0    # Y_before == 0
    k_fix: int = 0       # transitions 0 -> 1
    n_correct: int = 0   # Y_before == 1
    k_break: int = 0     # transitions 1 -> 0


class OnlineKernelCalibration:
    """Beta(α, β)-smoothed running estimator of P(fix|broken) + P(break|correct).

    Designed to be updated after each verify in a live evaluation loop.
    Falls back to `init_kernel` (or DEFAULT_KERNEL) when no observations
    have been recorded in a regime yet — i.e. the planner gets *some*
    kernel even on the very first instance.

    Thread-safe: .update() and .get() acquire an internal lock so multiple
    worker threads can share one estimator. The lock is uncontested in the
    typical sequential-loop case and adds negligible overhead.

    Usage:
        ok = OnlineKernelCalibration(init_kernel=measured_kernel)
        for step in trajectory:
            y_before = current_correctness_estimate
            run_action()
            y_after = verify()
            ok.update(y_before, y_after)
            current = ok.get()  # safe to feed back into planner re-solve
    """
    def __init__(
        self,
        init_kernel: Mapping[str, float] | None = None,
        *, alpha: float = 1.0, beta: float = 1.0,
        prior_counts: Mapping[str, int] | None = None,
    ) -> None:
        self.counts = _KernelCounts()
        self.alpha = float(alpha)
        self.beta = float(beta)
        # Train-fit raw counts treated as pseudo-observations. Used by sample()
        # so Thompson posterior is anchored to train evidence instead of a
        # flat Beta(1,1). Schema matches transition_kernel.json["raw_counts"]:
        #   {"0->0": int, "0->1": int, "1->0": int, "1->1": int}
        pc = dict(prior_counts) if prior_counts else {}
        self._prior_k_fix = int(pc.get("0->1", 0))
        self._prior_n_broken = int(pc.get("0->0", 0) + pc.get("0->1", 0))
        self._prior_k_break = int(pc.get("1->0", 0))
        self._prior_n_correct = int(pc.get("1->0", 0) + pc.get("1->1", 0))
        # Normalize init_kernel to lowercase-key schema so .get() can serve it
        # back uniformly to callers that expect kernel_update(belief, kernel).
        raw = dict(init_kernel) if init_kernel else dict(DEFAULT_KERNEL)
        if "P_fix_given_broken" in raw and "p_fix_broken" not in raw:
            raw["p_fix_broken"] = raw["P_fix_given_broken"]
        if "P_break_given_correct" in raw and "p_break_correct" not in raw:
            raw["p_break_correct"] = raw["P_break_given_correct"]
        # Validate at construction time so a malformed init_kernel raises
        # here, not later inside .get() on an unrelated call site (KeyError
        # in the middle of a hot loop is much harder to debug).
        missing = {"p_fix_broken", "p_break_correct"} - raw.keys()
        if missing:
            raise ValueError(
                "OnlineKernelCalibration init_kernel missing required keys "
                f"{sorted(missing)} (accepted: lowercase {{p_fix_broken, "
                f"p_break_correct}} or uppercase {{P_fix_given_broken, "
                f"P_break_given_correct}}); got: {dict(raw)!r}"
            )
        self._init = raw
        # RLock so summary() can call get() while holding the lock without
        # deadlocking. Re-entrant acquire is cheap in the uncontended case.
        self._lock = threading.RLock()

    def update(self, y_before: int, y_after: int) -> None:
        """Record one observed (Y_t, Y_{t+1}) transition.

        Raises ValueError if either value does not equal 0 or 1 — silently
        accepting non-binary input (e.g. None) was a bug surface because
        anything `!= 0` previously landed in the "correct" regime, so
        `update(None, 1)` would silently log a (correct → correct)
        transition. Callers must filter their own data; both production
        call sites already do.

        Note on accepted types: the check uses `==` equality, so bool
        (True/False) and float 0.0/1.0 are accepted in addition to int
        0/1 — they all round-trip to the same counts. Anything not equal
        to 0 or 1 (None, 2, strings, etc.) raises.
        """
        if y_before not in (0, 1) or y_after not in (0, 1):
            raise ValueError(
                f"OnlineKernelCalibration.update requires y_before and "
                f"y_after to equal 0 or 1; got y_before={y_before!r}, "
                f"y_after={y_after!r}"
            )
        with self._lock:
            if y_before == 0:
                self.counts.n_broken += 1
                self.counts.k_fix += int(y_after == 1)
            else:
                self.counts.n_correct += 1
                self.counts.k_break += int(y_after == 0)

    def get(self) -> dict:
        """Current posterior estimate (mean only).

        Shape matches kernel_update()'s expected lowercase-key schema:
          {"p_fix_broken": float, "p_break_correct": float}
        """
        with self._lock:
            c = self.counts
            if c.n_broken > 0:
                p_fix = (c.k_fix + self.alpha) / (c.n_broken + self.alpha + self.beta)
            else:
                p_fix = self._init["p_fix_broken"]
            if c.n_correct > 0:
                p_break = (c.k_break + self.alpha) / (c.n_correct + self.alpha + self.beta)
            else:
                p_break = self._init["p_break_correct"]
        return {"p_fix_broken": p_fix, "p_break_correct": p_break}

    def sample(self, rng) -> dict:
        """Thompson sample from Beta posterior.

        Returns a kernel dict drawn from
            Beta(alpha + train_k + live_k, beta + train_n + live_n - ...).
        If no prior_counts and no live observations exist, falls back to
        a point sample at the init_kernel mean (degenerate posterior).

        rng: numpy.random.Generator (rng.beta(a, b)) or random.Random
              (rng.betavariate(a, b)). Detected by attribute.
        """
        beta_fn = getattr(rng, "beta", None) or rng.betavariate
        with self._lock:
            c = self.counts
            # Combine prior (train raw_counts) + live observations
            k_fix_total = c.k_fix + self._prior_k_fix
            n_broken_total = c.n_broken + self._prior_n_broken
            k_break_total = c.k_break + self._prior_k_break
            n_correct_total = c.n_correct + self._prior_n_correct

            if n_broken_total > 0:
                a_fix = k_fix_total + self.alpha
                b_fix = (n_broken_total - k_fix_total) + self.beta
                p_fix = float(beta_fn(a_fix, b_fix))
            else:
                p_fix = float(self._init["p_fix_broken"])
            if n_correct_total > 0:
                a_break = k_break_total + self.alpha
                b_break = (n_correct_total - k_break_total) + self.beta
                p_break = float(beta_fn(a_break, b_break))
            else:
                p_break = float(self._init["p_break_correct"])
        return {"p_fix_broken": p_fix, "p_break_correct": p_break}

    def summary(self) -> dict:
        """Diagnostic snapshot of internal counts + current estimate.

        Useful for the small-N caveat: if n_broken or n_correct is below
        ~10, the posterior mean carries little signal. Persist this at
        end-of-run so analyses can decide whether the online posterior is
        worth trusting.
        """
        with self._lock:
            c = self.counts
            current = self.get()  # OK to call: get() re-acquires the same RLock
            return {
                "n_broken_observed": c.n_broken,
                "n_correct_observed": c.n_correct,
                "k_fix": c.k_fix,
                "k_break": c.k_break,
                "alpha": self.alpha,
                "beta": self.beta,
                "init_kernel": dict(self._init),
                "current_estimate": current,
            }


# ----------------------------------------------------------------------------
# File-level resolver used by live agents and the iter/refine.py CLI
# ----------------------------------------------------------------------------

def resolve_kernel(
    gen_dir: Path,
    mode: str = "measured",
    *,
    kernel_path: Path | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[dict, str, OnlineKernelCalibration | None]:
    """Return (kernel_dict, source_label, online_or_None) for the given mode.

    `mode` is one of:
      'measured'  — load <gen_dir>/transition_kernel.json (or `kernel_path`
                    if provided), fall back to DEFAULT_KERNEL if missing.
                    Third return value is None.
      'online'    — same starting point, plus an OnlineKernelCalibration
                    initialized from it; the caller should
                    .update(y_before, y_after) after each verify.
      'thompson'  — load measured kernel + raw_counts as Beta prior; returns
                    OnlineKernelCalibration whose .sample(rng) draws from the
                    posterior. Caller re-solves DP per instance with sample.
      'hardcoded' — always return DEFAULT_KERNEL, even if a measured file
                    exists. Third return value is None.

    Source label is one of {"measured", "default", "hardcoded"} for logging.
    """
    if mode not in {"measured", "online", "hardcoded", "thompson"}:
        raise ValueError(f"unknown kernel mode: {mode!r}")

    if mode == "hardcoded":
        return dict(DEFAULT_KERNEL), "hardcoded", None

    path = kernel_path or (gen_dir / "transition_kernel.json")
    raw_counts: dict = {}
    if path.exists():
        kj = json.loads(path.read_text())
        k_all = kj.get("kernel_all", kj)
        kernel = {
            "p_fix_broken": k_all["P_fix_given_broken"],
            "p_break_correct": k_all["P_break_given_correct"],
        }
        raw_counts = dict(k_all.get("raw_counts", {}))
        source = "measured"
    else:
        kernel = dict(DEFAULT_KERNEL)
        source = "default"

    if mode == "online":
        ok = OnlineKernelCalibration(init_kernel=kernel, alpha=alpha, beta=beta)
        return kernel, source, ok
    if mode == "thompson":
        ok = OnlineKernelCalibration(
            init_kernel=kernel, alpha=alpha, beta=beta,
            prior_counts=raw_counts,
        )
        return kernel, source, ok
    return kernel, source, None


# ----------------------------------------------------------------------------
# Conditional kernel: P(Y_{k+1} | Y_k, z_k) with Beta(1,1) smoothing
# ----------------------------------------------------------------------------
#
# Mathematical extension of the marginal kernel that conditions the regen
# transition on the *observed critic signature* z_k = (z^(1), ..., z^(K)).
#
# Standard kernel: P(Y_{k+1} | Y_k)            -> {p_01, p_10}
# Conditional:     P(Y_{k+1} | Y_k, z_k)        -> {p_01(z), p_10(z)}
#
# Why: critics observed before the generate call carry signal about the
# *kind* of bug (deceptive correctness vs. obvious test failure), which
# changes how likely a regenerate fixes it. The marginal kernel averages
# all signatures together; conditioning preserves this information.
#
# Data requirements: K binary critics -> 2^K cells, each conditioning a
# Beta(1,1) posterior. For K=3 (L0/L2/L3) -> 8 conditioning bins x 2
# regimes (Y_k in {0,1}) = 16 cells. Sparse cells fall back to the
# marginal kernel via the `fallback_to_marginal` flag.

CRITIC_FIELDS_DEFAULT = ("L0_syntax", "L2_public_tests", "L3_llm_review")


def _z_signature(rec: Mapping[str, object],
                 critic_fields: Sequence[str] = CRITIC_FIELDS_DEFAULT
                 ) -> tuple[int, ...] | None:
    """Pack a record's critic outcomes into a deterministic 0/1 tuple.

    Returns None if any critic field is missing or non-binary — caller
    should drop the transition rather than impute (imputation would bias
    the conditional posterior).
    """
    out: list[int] = []
    for c in critic_fields:
        v = rec.get(c)
        if v is None:
            return None
        if v in (True, 1, 1.0):
            out.append(1)
        elif v in (False, 0, 0.0):
            out.append(0)
        else:
            return None
    return tuple(out)


def conditional_pairs_from_trajectories(
    trajectories: Iterable[Iterable[Mapping]],
    *,
    y_key: str = "Y",
    critic_fields: Sequence[str] = CRITIC_FIELDS_DEFAULT,
) -> list[tuple[int, tuple[int, ...], int]]:
    """Build (Y_t, z_t, Y_{t+1}) triples from sorted per-instance trajectories.

    Pairs with missing Y on either side, or missing critic outcomes at
    step t, are dropped. Critics at t+1 are not needed (we observe the
    PRE-regen critic signature).
    """
    triples: list[tuple[int, tuple[int, ...], int]] = []
    for traj in trajectories:
        traj = list(traj)
        for i in range(len(traj) - 1):
            y0 = traj[i].get(y_key)
            y1 = traj[i + 1].get(y_key)
            if y0 not in (0, 1) or y1 not in (0, 1):
                continue
            z = _z_signature(traj[i], critic_fields)
            if z is None:
                continue
            triples.append((int(y0), z, int(y1)))
    return triples


class ConditionalKernel:
    """Tabular Beta(alpha,beta)-smoothed posterior over P(Y_{t+1} | Y_t, z_t).

    Builds a conditional kernel from observed (Y_t, z_t, Y_{t+1}) triples,
    where z_t is the pre-regen critic signature. Falls back to the
    marginal kernel for under-observed cells (when the (Y_t, z_t) bucket
    has fewer than `min_obs` transitions).

    Usage (post-hoc from cached trajectories):
        ck = ConditionalKernel.from_triples(triples, critic_fields)
        # During replay, after observing z_k pre-generate:
        p_fix   = ck.p_fix(z_k)
        p_break = ck.p_break(z_k)
        b_after = b_before * (1 - p_break) + (1 - b_before) * p_fix

    Online streaming usage:
        ck = ConditionalKernel(critic_fields=..., init_kernel=marginal_seed)
        for (y_t, z_t, y_t1) in stream:
            ck.update(y_t, z_t, y_t1)
        # Same getters as above.
    """

    def __init__(
        self,
        critic_fields: Sequence[str] = CRITIC_FIELDS_DEFAULT,
        *,
        init_kernel: Mapping[str, float] | None = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        min_obs: int = 3,
    ) -> None:
        self.critic_fields = tuple(critic_fields)
        self.K = len(self.critic_fields)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_obs = int(min_obs)
        # counts[(y_t, z_tuple)] = {"to_1": int, "to_0": int}
        self.counts: dict[tuple[int, tuple[int, ...]], dict[str, int]] = {}
        # Marginal fallback (used when conditioning bucket is sparse). If
        # init_kernel is None, defaults to the literature prior.
        raw = dict(init_kernel) if init_kernel else dict(DEFAULT_KERNEL)
        if "P_fix_given_broken" in raw and "p_fix_broken" not in raw:
            raw["p_fix_broken"] = raw["P_fix_given_broken"]
        if "P_break_given_correct" in raw and "p_break_correct" not in raw:
            raw["p_break_correct"] = raw["P_break_given_correct"]
        missing = {"p_fix_broken", "p_break_correct"} - raw.keys()
        if missing:
            raise ValueError(
                f"ConditionalKernel init_kernel missing keys {sorted(missing)}"
            )
        self._marginal_fallback = {
            "p_fix_broken": float(raw["p_fix_broken"]),
            "p_break_correct": float(raw["p_break_correct"]),
        }

    @classmethod
    def from_triples(
        cls,
        triples: Iterable[tuple[int, tuple[int, ...], int]],
        critic_fields: Sequence[str] = CRITIC_FIELDS_DEFAULT,
        *,
        init_kernel: Mapping[str, float] | None = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        min_obs: int = 3,
    ) -> "ConditionalKernel":
        """Build a ConditionalKernel by streaming pre-extracted triples."""
        ck = cls(critic_fields, init_kernel=init_kernel,
                  alpha=alpha, beta=beta, min_obs=min_obs)
        for y_t, z_t, y_t1 in triples:
            ck.update(y_t, z_t, y_t1)
        return ck

    def update(self, y_before: int, z_before: tuple[int, ...], y_after: int) -> None:
        if y_before not in (0, 1) or y_after not in (0, 1):
            raise ValueError(
                f"ConditionalKernel.update needs binary Y; got "
                f"y_before={y_before!r}, y_after={y_after!r}"
            )
        if len(z_before) != self.K:
            raise ValueError(
                f"z_before has length {len(z_before)}, expected K={self.K} "
                f"(critics={self.critic_fields})"
            )
        key = (int(y_before), tuple(int(z) for z in z_before))
        c = self.counts.setdefault(key, {"to_1": 0, "to_0": 0})
        if y_after == 1:
            c["to_1"] += 1
        else:
            c["to_0"] += 1

    def _bucket(self, y_before: int, z: tuple[int, ...]) -> dict[str, int]:
        return self.counts.get((int(y_before), tuple(int(zi) for zi in z)),
                                {"to_1": 0, "to_0": 0})

    def n_obs(self, y_before: int, z: tuple[int, ...]) -> int:
        c = self._bucket(y_before, z)
        return c["to_1"] + c["to_0"]

    def p_fix(self, z: tuple[int, ...]) -> float:
        """P(Y_{t+1}=1 | Y_t=0, z_t=z). Falls back to marginal if sparse."""
        c = self._bucket(0, z)
        n = c["to_1"] + c["to_0"]
        if n < self.min_obs:
            return self._marginal_fallback["p_fix_broken"]
        return (c["to_1"] + self.alpha) / (n + self.alpha + self.beta)

    def p_break(self, z: tuple[int, ...]) -> float:
        """P(Y_{t+1}=0 | Y_t=1, z_t=z). Falls back to marginal if sparse."""
        c = self._bucket(1, z)
        n = c["to_1"] + c["to_0"]
        if n < self.min_obs:
            return self._marginal_fallback["p_break_correct"]
        return (c["to_0"] + self.alpha) / (n + self.alpha + self.beta)

    def kernel_for(self, z: tuple[int, ...]) -> dict[str, float]:
        """Return {p_fix_broken, p_break_correct} for the given z_t.

        Lets callers slot a conditional kernel into code paths that expect
        the standard kernel dict (e.g. `kernel_update(belief, kernel)`).
        """
        return {
            "p_fix_broken": self.p_fix(z),
            "p_break_correct": self.p_break(z),
        }

    def sample(self, rng, z: tuple[int, ...]) -> dict[str, float]:
        """Thompson sample (p_fix, p_break) from Beta posterior of bucket z.

        For p_fix uses bucket (Y=0, z); for p_break uses bucket (Y=1, z).
        Falls back to marginal kernel mean when either bucket has < min_obs
        observations (degenerate posterior).

        rng: random.Random (uses betavariate) or numpy Generator (uses beta).
        """
        beta_fn = getattr(rng, "beta", None) or rng.betavariate
        # p_fix from bucket (Y_t=0, z)
        c0 = self._bucket(0, z); n0 = c0["to_1"] + c0["to_0"]
        if n0 >= self.min_obs:
            a = c0["to_1"] + self.alpha
            b = c0["to_0"] + self.beta
            p_fix = float(beta_fn(a, b))
        else:
            p_fix = float(self._marginal_fallback["p_fix_broken"])
        # p_break from bucket (Y_t=1, z)
        c1 = self._bucket(1, z); n1 = c1["to_1"] + c1["to_0"]
        if n1 >= self.min_obs:
            a = c1["to_0"] + self.alpha
            b = c1["to_1"] + self.beta
            p_break = float(beta_fn(a, b))
        else:
            p_break = float(self._marginal_fallback["p_break_correct"])
        return {"p_fix_broken": p_fix, "p_break_correct": p_break}

    def summary(self) -> dict:
        """Per-bucket counts + smoothed posteriors. For inspection / saving."""
        out: dict[str, object] = {
            "critic_fields": list(self.critic_fields),
            "alpha": self.alpha, "beta": self.beta,
            "min_obs": self.min_obs,
            "marginal_fallback": dict(self._marginal_fallback),
            "buckets": {},
        }
        all_keys = sorted(self.counts.keys())
        for (y, z) in all_keys:
            c = self.counts[(y, z)]
            n = c["to_1"] + c["to_0"]
            key = f"y{y}|z={'_'.join(map(str, z))}"
            posterior = (
                self.p_fix(z) if y == 0 else self.p_break(z)
            )
            sparse = n < self.min_obs
            out["buckets"][key] = {
                "y_before": y, "z": list(z),
                "to_1": c["to_1"], "to_0": c["to_0"], "n": n,
                "posterior": posterior,
                "fellback_to_marginal": sparse,
            }
        return out


__all__ = [
    "DEFAULT_KERNEL",
    "kernel_update",
    "compute_transition_kernel_from_pairs",
    "pairs_from_trajectories",
    "OnlineKernelCalibration",
    "resolve_kernel",
    # Conditional kernel extension:
    "CRITIC_FIELDS_DEFAULT",
    "ConditionalKernel",
    "conditional_pairs_from_trajectories",
]
