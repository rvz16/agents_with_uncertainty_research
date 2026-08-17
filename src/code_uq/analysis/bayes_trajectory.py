"""Belief over episode success, updated along the trajectory.

The published scheme multiplies one likelihood ratio per critic verdict and then
pushes the belief through a transition kernel on every generation. Both hurt on
measured data. The kernel is the worse of the two: its constants are marked
"initial uninformative" in the sources and were never calibrated, and because it
is optimistic (fixed point around 0.91) every regeneration raises the belief --
while in practice many regenerations mean a hard task.

Here evidence is summed in log-odds instead:

    logit(belief) = logit(prior) + tau * sum_e  w_kind * gamma^i * logLR(e)

* `logLR` of a critic verdict comes from a 2x2 table with Beta(1,1) smoothing;
* `logLR` of a continuous per-generation feature is `logit P(Y=1|x) - logit(prior)`
  from a one-dimensional logistic fitted at the level of individual generations;
* `gamma^i` decays with generation index. Earlier generations turned out to be
  more informative than later ones, so the decay runs forward, not backward;
* `w_kind` is one weight per kind of evidence rather than per feature: on a
  hundred episodes, fitting a weight per feature does not pay for itself;
* `tau` is a temperature against overconfidence from correlated evidence, applied
  to the log ratio rather than to the probabilities.

There is no transition kernel. A regeneration on its own does not move the
belief; only observations do.

Calibrated per benchmark: `gamma`, `tau` and the weights are chosen by likelihood
on the training folds, as are the tables and the one-dimensional fits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9
CRITICS = ("critic_L0", "critic_L1", "critic_L2", "critic_L3")


@dataclass
class Calibration:
    prior: float = 0.5
    #: critic name -> (P(pass|Y=1), P(pass|Y=0))
    critic_theta: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: feature name -> (slope, intercept) of the one-dimensional logistic
    step_model: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: feature name -> (mean, sd) used for standardisation
    step_scale: dict[str, tuple[float, float]] = field(default_factory=dict)
    gamma: float = 1.0
    temperature: float = 1.0
    w_critic: float = 1.0
    w_step: float = 1.0


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def fit_critic_theta(rows: list[dict], labels: np.ndarray
                     ) -> dict[str, tuple[float, float]]:
    """P(pass | Y) with Beta(1,1) smoothing."""
    counts: dict[str, list[int]] = {}
    for row, label in zip(rows, labels):
        for name, value in row.items():
            if not name.startswith("critic_L"):
                continue
            slot = counts.setdefault(name, [0, 0, 0, 0])
            if label:
                slot[0] += int(value > 0.5)
                slot[1] += 1
            else:
                slot[2] += int(value > 0.5)
                slot[3] += 1
    return {name: ((p1 + 1) / (n1 + 2), (p0 + 1) / (n0 + 2))
            for name, (p1, n1, p0, n0) in counts.items() if n1 and n0}


def fit_step_model(values: list[float], labels: list[int]
                   ) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """One-dimensional logistic fitted at the level of individual generations.

    Trained on (generation score, outcome of its episode) pairs, so a feature is
    weighted independently of how many generations the episode had.
    """
    x = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=float)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if len(x) < 20 or len(set(y.tolist())) < 2:
        return None
    mu, sd = float(x.mean()), float(x.std())
    if sd < 1e-9:
        return None
    z = (x - mu) / sd
    # one-dimensional logistic by Newton steps, no dependencies
    w = b = 0.0
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(w * z + b)))
        gw = float(np.mean((p - y) * z))
        gb = float(np.mean(p - y))
        hw = float(np.mean(p * (1 - p) * z * z)) + 1e-4
        hb = float(np.mean(p * (1 - p))) + 1e-4
        w -= gw / hw
        b -= gb / hb
        if abs(gw) + abs(gb) < 1e-7:
            break
    return (w, b), (mu, sd)


def calibrate(rows: list[dict], labels: np.ndarray, step_series: list[dict],
              step_features: list[str], gammas=(1.0, 0.8, 0.6, 0.4),
              temps=(0.2, 0.4, 0.7, 1.0, 1.5)) -> Calibration:
    """Fit on the training folds: tables, one-dimensional models, gamma, tau, weights."""
    cal = Calibration(prior=float(np.clip(labels.mean(), 1e-6, 1 - 1e-6)))
    cal.critic_theta = fit_critic_theta(rows, labels)

    for feature in step_features:
        vals, labs = [], []
        for series, label in zip(step_series, labels):
            for value in series.get(feature, []):
                vals.append(value)
                labs.append(int(label))
        fitted = fit_step_model(vals, labs)
        if fitted:
            cal.step_model[feature] = fitted[0]
            cal.step_scale[feature] = fitted[1]

    # grid over gamma, tau and weights, scored by training likelihood
    best, best_ll = None, -np.inf
    for gamma in gammas:
        for temp in temps:
            for w_critic, w_step in ((1.0, 1.0), (1.0, 0.5), (0.5, 1.0),
                                     (1.0, 0.0), (0.0, 1.0)):
                cal.gamma, cal.temperature = gamma, temp
                cal.w_critic, cal.w_step = w_critic, w_step
                z = np.array([belief_logit(cal, r, s, step_features)
                              for r, s in zip(rows, step_series)])
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                ll = float(np.mean(labels * np.log(np.clip(p, EPS, 1))
                                   + (1 - labels) * np.log(np.clip(1 - p, EPS, 1))))
                if ll > best_ll:
                    best_ll = ll
                    best = (gamma, temp, w_critic, w_step)
    if best:
        cal.gamma, cal.temperature, cal.w_critic, cal.w_step = best
    return cal


def belief_logit(cal: Calibration, row: dict, series: dict,
                 step_features: list[str]) -> float:
    """Log-odds of the belief after walking the whole trajectory."""
    z = _logit(cal.prior)
    prior_logit = _logit(cal.prior)

    for name, (p1, p0) in cal.critic_theta.items():
        value = row.get(name)
        if value is None:
            continue
        a, b = (p1, p0) if value > 0.5 else (1.0 - p1, 1.0 - p0)
        z += cal.temperature * cal.w_critic * (
            math.log(max(a, EPS)) - math.log(max(b, EPS)))

    for feature in step_features:
        model = cal.step_model.get(feature)
        if model is None:
            continue
        w, b = model
        mu, sd = cal.step_scale[feature]
        for i, value in enumerate(series.get(feature, [])):
            if not math.isfinite(value):
                continue
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, w * (value - mu) / sd + b))))
            log_lr = _logit(p) - prior_logit
            z += cal.temperature * cal.w_step * (cal.gamma ** i) * log_lr
    return z


def belief(cal: Calibration, row: dict, series: dict,
           step_features: list[str]) -> float:
    z = belief_logit(cal, row, series, step_features)
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
