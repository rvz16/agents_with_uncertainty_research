"""UProp: uncertainty propagation by trajectory-dependent pointwise mutual information.

Reference: Jinhao Duan et al., "UProp: Investigating the Uncertainty Propagation
of LLMs in Multi-Step Agentic Decision-Making", arXiv:2506.17419 (2025). The
public repository holds no code; this follows the paper's equations.

The uncertainty of decision step t decomposes into an intrinsic part (the
entropy of the step's own decision distribution) and an extrinsic part (the
mutual information with every preceding decision). Both are estimated from N
samples drawn at every step of the *recorded* trajectory (one trajectory-
dependent decision process, Z = 1, whose realised decisions are the ones the
agent actually took):

* intrinsic  IU_t = mean_n [ -(1/L_n) log p(y_t^(n) | prefix) ]      (LN-PE, eq. 2.1)
* extrinsic  PMI_t = -log mean_n K_tau( d(y_t^(n), y_t^*) )           (eq. 8)
  where y_t^* is the realised decision, d a string distance and
  K_tau(x) = ((1/sqrt(2 pi)) exp(-x^2 / 2))^tau a Gaussian kernel;
* step t inherits the PMI of every earlier step: EU_t = sum_{i<t} PMI_i (eq. 5),
  H_t = IU_t + EU_t;
* step-length normalisation (Sec. 3.3): lambda = T + sum_t EU_t / IU_t and
  UProp = (1/lambda) sum_t H_t                                       (eq. 9).

The distance is a fuzzy string ratio (the paper uses thefuzz); here
``difflib.SequenceMatcher`` gives the same [0, 1] ratio without a dependency.
Steps whose samples are missing contribute nothing. Higher = more uncertain;
the report negates it into a confidence.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import math
from typing import Any

SQRT_2PI = math.sqrt(2.0 * math.pi)


def string_distance(a: str, b: str) -> float:
    """1 - fuzzy ratio, in [0, 1]; whitespace-normalised."""
    a, b = " ".join(a.split()), " ".join(b.split())
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def kernel(distance: float, tau: float = 1.0) -> float:
    return (math.exp(-0.5 * distance * distance) / SQRT_2PI) ** tau


def step_pmi(samples: list[str], realised: str, tau: float = 1.0) -> float | None:
    """-log of the kernel density of the realised decision among the step's samples."""
    if not samples:
        return None
    density = sum(kernel(string_distance(s, realised), tau) for s in samples) / len(samples)
    return -math.log(max(density, 1e-300))


def step_intrinsic(sample_nll: list[float]) -> float | None:
    """Length-normalised predictive entropy: mean per-token NLL over the samples."""
    ok = [v for v in sample_nll if v is not None and math.isfinite(v)]
    return sum(ok) / len(ok) if ok else None


def uprop(steps: list[dict[str, Any]], tau: float = 1.0, eps: float = 1e-6) -> dict[str, Any]:
    """Propagate over the recorded trajectory.

    ``steps[t]`` = {"realised": str, "samples": [str], "sample_nll": [float] | None,
    "iu": float | None}. ``iu`` overrides the sample-based intrinsic term (for a
    provider without logprobs, the recorded step's own entropy can stand in).
    Returns per-step IU / PMI / EU / H and the normalised total.
    """
    iu, pmi = [], []
    for s in steps:
        pmi_t = step_pmi(s.get("samples") or [], s.get("realised") or "", tau)
        iu_t = s.get("iu")
        if iu_t is None:
            iu_t = step_intrinsic(s.get("sample_nll") or [])
        if pmi_t is None or iu_t is None:
            continue
        iu.append(float(iu_t)); pmi.append(float(pmi_t))
    if not iu:
        return {"total": None, "iu": [], "pmi": [], "eu": [], "h": [], "lambda": None}
    eu, h = [], []
    running = 0.0
    for t in range(len(iu)):
        eu.append(running); h.append(iu[t] + running)
        running += pmi[t]
    lam = len(iu) + sum(e / max(i, eps) for e, i in zip(eu, iu))
    return {"total": sum(h) / lam, "iu": iu, "pmi": pmi, "eu": eu, "h": h, "lambda": lam,
            "iu_mean": sum(iu) / len(iu), "eu_mean": sum(eu) / len(eu), "pmi_mean": sum(pmi) / len(pmi)}
