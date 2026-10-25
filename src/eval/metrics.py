"""Scores every piece-3 model is judged with (D-022, D-024, D-025): the project's "answer key".

Built by Antigravity (Claude Opus 4.6 Thinking) from task card docs/handoff/WP-01-metrics.md,
then reviewed and hardened by Claude Code (review + eval-auditor findings, D-025).
Tests: tests/handoff/test_wp01_metrics.py.
Rules: pure functions only (no file reading, no printing, no global state); inputs may be lists or
numpy arrays; return plain Python floats (or None where a value is undefined), never numpy scalars.
y_true / y_val are 0/1 labels (1 = injury); score is any finite number where higher = more risky;
prob is a probability in [0, 1]. Invalid input raises ValueError instead of giving a number.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ArrayLike = Sequence[float] | np.ndarray
LOGIT_CLIP = 1e-6  # probabilities are clipped to [1e-6, 1 - 1e-6] before taking the logit


# ---------------------------------------------------------------- input checks


def _labels(y: ArrayLike) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("labels must be a non-empty 1-D sequence")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 (healthy) or 1 (injury)")
    return y


def _finite(x: ArrayLike, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return x


def _probabilities(p: ArrayLike) -> np.ndarray:
    p = _finite(p, "prob")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("prob must lie in [0, 1]: raw calibration is n/a for scores that are "
                         "not probabilities (D-025)")
    return p


def _pair(y: ArrayLike, x: np.ndarray) -> np.ndarray:
    y = _labels(y)
    if len(y) != len(x):
        raise ValueError(f"length mismatch: {len(y)} labels vs {len(x)} values")
    return y


def _both_classes(y: np.ndarray, what: str) -> None:
    if not (np.any(y == 1) and np.any(y == 0)):
        raise ValueError(f"{what} needs both classes (0 and 1)")


def logit(prob: ArrayLike) -> np.ndarray:
    """log(p / (1 - p)) with p clipped to [1e-6, 1 - 1e-6]. Platt is fitted on logit(p) for
    models whose output is a probability (D-025)."""
    p = np.clip(_probabilities(prob), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- ranking scores


def average_precision(y_true: ArrayLike, score: ArrayLike) -> float:
    """PR-AUC as average precision (step-wise, ties form one step; sklearn's formula).
    ValueError if y_true has no injury."""
    s = _finite(score, "score")
    y = _pair(y_true, s)
    if not np.any(y == 1):
        raise ValueError("y_true has no positive (injury) examples")
    return float(average_precision_score(y, s))


def roc_auc(y_true: ArrayLike, score: ArrayLike) -> float | None:
    """ROC-AUC (ties count half). None when y_true contains only one class."""
    s = _finite(score, "score")
    y = _pair(y_true, s)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


# ---------------------------------------------------------------- the alert level


def choose_alert_threshold(y_val: ArrayLike, score_val: ArrayLike) -> float:
    """The alert level picked on validation runners (D-022, D-024). Alert means score >= threshold.

    Candidates: every distinct score plus +infinity ("alert nobody"). Keep the smallest
    |sensitivity - specificity|; then the largest sensitivity + specificity; then the highest
    threshold. Floats are compared with a tolerance of 1e-12.
    """
    s = _finite(score_val, "score_val")
    y = _pair(y_val, s)
    _both_classes(y, "choose_alert_threshold")
    positives, negatives = np.sum(y == 1), np.sum(y == 0)

    results = []
    for t in list(np.unique(s)) + [math.inf]:
        alerted = s >= t
        sens = np.sum(alerted & (y == 1)) / positives
        spec = np.sum(~alerted & (y == 0)) / negatives
        results.append((abs(sens - spec), sens + spec, float(t)))

    min_gap = min(r[0] for r in results)
    kept = [r for r in results if r[0] <= min_gap + 1e-12]
    max_sum = max(r[1] for r in kept)
    kept = [r for r in kept if r[1] >= max_sum - 1e-12]
    return float(max(r[2] for r in kept))


def alert_numbers(y_true: ArrayLike, score: ArrayLike, threshold: float) -> dict:
    """Numbers at a fixed alert level (alert = score >= threshold): tp, fp, tn, fn, alerts,
    sensitivity, specificity, precision, false_alarms_per_caught_injury, alerts_per_100_events,
    mcc. Any value whose denominator is 0 is None (never 0)."""
    s = _finite(score, "score")
    y = _pair(y_true, s)
    if math.isnan(threshold):
        raise ValueError("threshold is NaN")
    alerted = s >= threshold
    tp = int(np.sum(alerted & (y == 1)))
    fp = int(np.sum(alerted & (y == 0)))
    tn = int(np.sum(~alerted & (y == 0)))
    fn = int(np.sum(~alerted & (y == 1)))

    def ratio(num: float, den: float) -> float | None:
        return float(num / den) if den > 0 else None

    mcc_den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)  # Python ints: no overflow
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "alerts": tp + fp,
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "precision": ratio(tp, tp + fp),
        "false_alarms_per_caught_injury": ratio(fp, tp),
        "alerts_per_100_events": float(100 * (tp + fp) / len(y)),
        "mcc": float((tp * tn - fp * fn) / math.sqrt(mcc_den)) if mcc_den > 0 else None,
    }


# ---------------------------------------------------------------- calibration


def fit_platt(score_val: ArrayLike, y_val: ArrayLike) -> tuple[float, float]:
    """Platt scaling fitted on validation runners: an UNPENALISED logistic regression
    P(injury) = 1 / (1 + exp(-(slope * score + intercept))). Returns (slope, intercept).

    Pass logit(p) as the score for models whose output is a probability (D-025).
    Constant scores give slope 0 and the logit of the injury rate. Warns when the validation
    data is perfectly separated (the slope is then not finite; it can be a sign of a leak).
    """
    s = _finite(score_val, "score_val")
    y = _pair(y_val, s)
    _both_classes(y, "fit_platt")
    rate = float(np.mean(y))
    mean, sd = float(np.mean(s)), float(np.std(s))
    if sd == 0:
        return 0.0, float(math.log(rate / (1 - rate)))
    if s[y == 1].min() > s[y == 0].max() or s[y == 1].max() < s[y == 0].min():
        warnings.warn("Platt: validation scores separate the classes perfectly; the fitted slope "
                      "is unreliable (check for leakage)", RuntimeWarning, stacklevel=2)
    z = (s - mean) / sd  # standardising keeps lbfgs accurate for any score scale
    # C=inf: no regularisation; tol=1e-10: reach the exact optimum (default stops early)
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000, tol=1e-10)
    lr.fit(z.reshape(-1, 1), y)
    b, c = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return b / sd, c - b * mean / sd


def apply_platt(score: ArrayLike, params: tuple[float, float]) -> np.ndarray:
    """Probabilities from fit_platt's (slope, intercept), as a float numpy array."""
    s = _finite(score, "score")
    slope, intercept = params
    return 1.0 / (1.0 + np.exp(-(slope * s + intercept)))


def platt_slope_flag(params: tuple[float, float]) -> bool:
    """True when the Platt slope is 0 or below (it would flatten or flip the model's ranking)."""
    return bool(params[0] <= 0)


def brier(y_true: ArrayLike, prob: ArrayLike) -> float:
    """Brier score: mean of (prob - y)^2. Lower is better."""
    p = _probabilities(prob)
    y = _pair(y_true, p)
    return float(np.mean((p - y) ** 2))


def _tie_aware_bins(p_sorted: np.ndarray, bins: int) -> list[np.ndarray]:
    """Equal-count groups of the sorted order, but tied probabilities always share a group:
    each numpy.array_split cut moves forward to the end of its tie block (D-025)."""
    n = len(p_sorted)
    cuts = np.cumsum([len(g) for g in np.array_split(np.arange(n), bins)])[:-1]
    edges = [0]
    for c in cuts:
        while 0 < c < n and p_sorted[c] == p_sorted[c - 1]:
            c += 1
        if edges[-1] < c < n:
            edges.append(int(c))
    edges.append(n)
    return [np.arange(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


def ece_equal_count(y_true: ArrayLike, prob: ArrayLike, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins in which tied probabilities share a bin
    (so the result never depends on row order): sum of (bin size / n) * |mean y - mean p|."""
    p = _probabilities(prob)
    y = _pair(y_true, p)
    order = np.argsort(p, kind="stable")
    y_sorted, p_sorted = y[order], p[order]
    ece = 0.0
    for g in _tie_aware_bins(p_sorted, bins):
        ece += len(g) / len(y) * abs(float(np.mean(y_sorted[g])) - float(np.mean(p_sorted[g])))
    return float(ece)


def calibration_slope_intercept(y_true: ArrayLike, prob: ArrayLike) -> tuple[float, float]:
    """(slope, intercept) of calibration (Van Calster et al. 2019), with L = logit(p).

    slope: coefficient b of an unpenalised logistic regression y ~ c + b * L (1 = perfect).
    intercept: the a solving mean(y) = mean(sigmoid(a + L)), L used as a fixed offset
    (0 = perfect; below 0 = risks too high on average).
    """
    lg = logit(prob)
    y = _pair(y_true, lg)
    _both_classes(y, "calibration_slope_intercept")
    slope = fit_platt(lg, y)[0]  # same unpenalised fit; 0.0 for a constant prediction
    mean_y = float(np.mean(y))

    def gap(a: float) -> float:  # rises with a, so a bracketing root-finder is exact
        return float(np.mean(1.0 / (1.0 + np.exp(-(a + lg))))) - mean_y

    return float(slope), float(brentq(gap, -50.0, 50.0, xtol=1e-12))


# ---------------------------------------------------------------- decision curve


def _thresholds(thresholds: ArrayLike) -> np.ndarray:
    ts = _finite(thresholds, "thresholds")
    if np.any((ts <= 0) | (ts >= 1)):
        raise ValueError("decision-curve thresholds must lie strictly between 0 and 1")
    return ts


def net_benefit(y_true: ArrayLike, prob: ArrayLike, thresholds: ArrayLike) -> np.ndarray:
    """Net benefit per threshold t (alert = prob >= t): TP/n - (FP/n) * t / (1 - t)."""
    p = _probabilities(prob)
    y = _pair(y_true, p)
    n = len(y)
    out = []
    for t in _thresholds(thresholds):
        alerted = p >= t
        tp, fp = np.sum(alerted & (y == 1)), np.sum(alerted & (y == 0))
        out.append(tp / n - (fp / n) * (t / (1 - t)))
    return np.asarray(out, dtype=float)


def net_benefit_treat_all(y_true: ArrayLike, thresholds: ArrayLike) -> np.ndarray:
    """Net benefit of alerting everyone: prevalence - (1 - prevalence) * t / (1 - t)."""
    prevalence = float(np.mean(_labels(y_true)))
    ts = _thresholds(thresholds)
    return prevalence - (1 - prevalence) * ts / (1 - ts)


# ---------------------------------------------------------------- activity (PAMAP2)


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Macro-F1 over the labels that appear in y_true only (labels that are only predicted are
    ignored); a label with no correct prediction scores 0."""
    y, yp = np.asarray(y_true), np.asarray(y_pred)
    if y.ndim != 1 or len(y) == 0 or y.shape != yp.shape:
        raise ValueError("y_true and y_pred must be non-empty 1-D sequences of equal length")
    return float(f1_score(y, yp, labels=np.unique(y), average="macro", zero_division=0))
