"""HTC: Holistic Trajectory Calibration (Zhang, Xiong, Wu, "Agentic Confidence Calibration", ICML 2026).

A trajectory is the sequence of its steps' token confidences; each step t has
n_t tokens with top-1 probabilities r_{t,i} and top-k mean probabilities. From
them 48 trajectory-level features (Appendix D of the paper) in four families
-- Dynamics (19), Position (14), Stability (10), Structure (5) -- feed a
logistic calibrator with L2 (HTC-Full) or L1 (HTC-Reduced) regularisation,
alpha chosen by inner cross-validation on a grid, liblinear, 5-fold CV outside.

The paper's supplementary code is not public; feature definitions follow
Appendix D.1 / Listing 1 exactly (names and order), undefined statistics
(S < 2, n_t < 2) are 0 as stated there. Within-step "attention" quantities
are the normalised token-confidence distribution pi_{t,i} = r_{t,i} / sum_j r_{t,j}.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

EPS = 1e-8
FEATURE_NAMES = (
    # Dynamics (19)
    "top1_gradient_mean", "top1_gradient_std", "top1_gradient_max", "top1_gradient_min", "top1_gradient_trend",
    "topk_gradient_mean", "topk_gradient_std", "topk_gradient_max", "topk_gradient_min", "topk_gradient_trend",
    "token_gradient_mean", "token_gradient_std", "token_gradient_max", "token_gradient_min",
    "step_progression_entropy", "step_progression_concentration", "step_progression_spread",
    "top1_confidence_change", "topk_confidence_change",
    # Position (14)
    "first_attention_entropy", "first_attention_concentration", "first_attention_spread", "first_confidence_volatility",
    "first_confidence_skewness", "first_top1_avg", "first_topk_avg",
    "last_attention_entropy", "last_attention_concentration", "last_attention_spread", "last_confidence_volatility",
    "last_confidence_skewness", "last_top1_avg", "last_topk_avg",
    # Stability (10)
    "attention_entropy_mean", "attention_entropy_std", "attention_concentration_mean", "attention_concentration_std",
    "attention_spread_mean", "attention_spread_std",
    "token_volatility_mean", "token_volatility_std", "token_skewness_mean", "token_skewness_std",
    # Structure (5)
    "normalized_step_count", "first_token_count", "last_token_count", "avg_tokens_per_step", "std_tokens_per_step",
)
FAMILIES = {"Dynamics": range(0, 19), "Position": range(19, 33), "Stability": range(33, 43), "Structure": range(43, 48)}
ALPHA_GRID = (0.001, 0.01, 0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0, 50.0)


def _step_stats(r: np.ndarray) -> tuple[float, float, float, float]:
    """(H_t, kappa_t, rho_t, skew_t) of one step's token confidences; zeros when n_t < 2."""
    if len(r) < 2:
        return 0.0, 0.0, 0.0, 0.0
    pi = r / (r.sum() + EPS)
    h = float(-(pi * np.log(pi + EPS)).sum())
    mu, sd = float(r.mean()), float(r.std())
    kappa = float(r.max() / (mu + EPS))
    rho = sd / (mu + EPS)
    skew = float(np.mean(((r - mu) / (sd + EPS)) ** 3))
    return h, kappa, rho, skew


def _stats(v: Sequence[float]) -> tuple[float, float, float, float]:
    a = np.asarray(v, dtype=float)
    if len(a) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(a.mean()), float(a.std()), float(a.max()), float(a.min())


def features(top1: Sequence[Sequence[float]], topk: Sequence[Sequence[float]] | None = None) -> np.ndarray:
    """The 48 HTC features of one trajectory.

    ``top1[t]`` = the top-1 token probabilities of step t; ``topk[t]`` = the
    per-token mean top-k probabilities (defaults to ``top1``).
    """
    steps = [np.asarray(s, dtype=float) for s in top1 if len(s) > 0]
    ksteps = [np.asarray(s, dtype=float) for s in (topk if topk is not None else top1) if len(s) > 0]
    if len(ksteps) != len(steps):
        ksteps = steps
    S = len(steps)
    if S == 0:
        return np.zeros(len(FEATURE_NAMES))
    x = np.array([s.mean() for s in steps]); y = np.array([s.mean() for s in ksteps])
    per = [_step_stats(s) for s in steps]
    H = np.array([p[0] for p in per]); K = np.array([p[1] for p in per]); R = np.array([p[2] for p in per]); SK = np.array([p[3] for p in per])
    n = np.array([len(s) for s in steps], dtype=float)
    f: list[float] = []
    # Dynamics
    dx = np.diff(x) if S >= 2 else np.array([]); dy = np.diff(y) if S >= 2 else np.array([])
    for d in (dx, dy):
        m, sd, mx, mn = _stats(d)
        f += [m, sd, mx, mn, float(d[-1] - d[0]) if len(d) >= 2 else 0.0]
    tok = np.concatenate([np.diff(s) for s in steps if len(s) >= 2]) if any(len(s) >= 2 for s in steps) else np.array([])
    f += list(_stats(tok))
    for v in (H, K, R):
        f.append(float(v.std() / (v.mean() + EPS)) if S >= 2 else 0.0)
    f += [float(x[-1] - x[0]) if S >= 2 else 0.0, float(y[-1] - y[0]) if S >= 2 else 0.0]
    # Position
    for idx in (0, S - 1):
        f += [float(H[idx]), float(K[idx]), float(R[idx]), float(R[idx]), float(SK[idx]), float(x[idx]), float(y[idx])]
    # Stability
    for v in (H, K, R):
        f += [float(v.mean()), float(v.std()) if S >= 2 else 0.0]
    f += [float(R.mean()), float(R.std()) if S >= 2 else 0.0, float(SK.mean()), float(SK.std()) if S >= 2 else 0.0]
    # Structure
    f += [S / 10.0, float(n[0]), float(n[-1]), float(n.mean()), float(n.std()) if S >= 2 else 0.0]
    out = np.asarray(f, dtype=float)
    assert out.shape == (len(FEATURE_NAMES),)
    out[~np.isfinite(out)] = 0.0
    return out


class HTCCalibrator:
    """Logistic calibrator on the 48 features; ``penalty`` 'l2' (HTC-Full) or 'l1' (HTC-Reduced).

    alpha is picked on the training set by inner stratified 3-fold CV over
    ALPHA_GRID, maximising AUROC and breaking ties by the Brier score, as the
    paper's combined criterion. Features are standardised with training
    statistics; columns of a family can be dropped with ``exclude``.
    """

    def __init__(self, penalty: str = "l2", exclude: Sequence[str] = (), seed: int = 42):
        self.penalty = penalty
        self.keep = np.array([i for i, name in enumerate(FEATURE_NAMES) if not any(name in FAMILIES_INV.get(fam, ()) for fam in exclude)])
        self.seed = seed
        self.mean = self.std = None
        self.model = None
        self.alpha = None

    def _prep(self, X: np.ndarray) -> np.ndarray:
        return (X[:, self.keep] - self.mean) / self.std

    def fit(self, X: np.ndarray, y: Sequence[int]) -> "HTCCalibrator":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, brier_score_loss
        from sklearn.model_selection import StratifiedKFold
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
        self.mean = X[:, self.keep].mean(axis=0); self.std = X[:, self.keep].std(axis=0) + 1e-9
        Z = self._prep(X)
        best = None
        n_inner = min(3, int(np.bincount(y).min()))
        for alpha in ALPHA_GRID:
            if n_inner >= 2:
                skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=self.seed)
                aucs, briers = [], []
                for tr, va in skf.split(Z, y):
                    if len(set(y[tr])) < 2 or len(set(y[va])) < 2:
                        continue
                    m = LogisticRegression(penalty=self.penalty, C=1.0 / alpha, solver="liblinear", max_iter=1000, random_state=self.seed).fit(Z[tr], y[tr])
                    p = m.predict_proba(Z[va])[:, 1]
                    aucs.append(roc_auc_score(y[va], p)); briers.append(brier_score_loss(y[va], p))
                score = (np.mean(aucs) if aucs else 0.0, -(np.mean(briers) if briers else 1.0))
            else:
                score = (0.0, 0.0)
            if best is None or score > best[0]:
                best = (score, alpha)
        self.alpha = best[1]
        self.model = LogisticRegression(penalty=self.penalty, C=1.0 / self.alpha, solver="liblinear", max_iter=1000, random_state=self.seed).fit(Z, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._prep(np.asarray(X, dtype=float)))[:, 1]

    @property
    def n_selected(self) -> int:
        return int((np.abs(self.model.coef_[0]) > 1e-9).sum())


FAMILIES_INV = {fam: tuple(FEATURE_NAMES[i] for i in idx) for fam, idx in FAMILIES.items()}
