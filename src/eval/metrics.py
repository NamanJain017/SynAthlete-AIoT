"""Scores every piece-3 model is judged with (D-022, D-024): the project's "answer key" maths.

Task card: docs/handoff/WP-01-metrics.md. Tests: tests/handoff/test_wp01_metrics.py.
Rules: pure functions only (no file reading, no printing, no global state); inputs may be lists or
numpy arrays; return plain Python floats (or None where a value is undefined), never numpy scalars.
y_true / y_val are 0/1 labels (1 = injury); score is any number where higher = more risky;
prob is a probability in [0, 1].
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ArrayLike = Sequence[float] | np.ndarray

# Set to True only when every function below is written; the tests skip while it is False.
IMPLEMENTED = True


def _to_arrays(a: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to numpy arrays, raise ValueError if lengths differ."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Length mismatch: {a.shape[0]} vs {b.shape[0]}")
    return a, b


def average_precision(y_true: ArrayLike, score: ArrayLike) -> float:
    """PR-AUC as average precision: sum over thresholds of (recall step) x precision.

    Step-wise, with no straight-line interpolation between points (use
    sklearn.metrics.average_precision_score). Raise ValueError if y_true has no 1.
    """
    y, s = _to_arrays(y_true, score)
    if not np.any(y == 1):
        raise ValueError("y_true has no positive (injury) examples")
    return float(average_precision_score(y, s))


def roc_auc(y_true: ArrayLike, score: ArrayLike) -> float | None:
    """ROC-AUC (chance that a random injury scores higher than a random healthy event; ties
    count half). Return None when y_true contains only one class."""
    y, s = _to_arrays(y_true, score)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def choose_alert_threshold(y_val: ArrayLike, score_val: ArrayLike) -> float:
    """The alert level picked on validation runners (D-022, D-024). Alert means score >= threshold.

    1. Candidates: every distinct value in score_val, plus +infinity ("alert nobody").
    2. For each candidate: sensitivity = TP / positives, specificity = TN / negatives.
    3. Keep the candidates with the smallest |sensitivity - specificity|.
    4. If several remain: keep those with the largest sensitivity + specificity.
    5. If several still remain: return the highest threshold.
    Compare floats with a tolerance of 1e-12. Raise ValueError unless y_val has both classes.
    Returns a Python float (float("inf") is allowed).
    """
    y, s = _to_arrays(y_val, score_val)
    if len(np.unique(y)) < 2:
        raise ValueError("y_val must have both classes (0 and 1)")

    positives = np.sum(y == 1)
    negatives = np.sum(y == 0)

    # Candidates: distinct scores + infinity
    candidates = list(np.unique(s)) + [float("inf")]

    results = []
    for t in candidates:
        # alert = score >= threshold
        alerted = s >= t
        tp = float(np.sum(alerted & (y == 1)))
        tn = float(np.sum(~alerted & (y == 0)))
        sens = tp / positives
        spec = tn / negatives
        gap = abs(sens - spec)
        total = sens + spec
        results.append((gap, total, t))

    # Step 3: smallest gap (with tolerance)
    min_gap = min(r[0] for r in results)
    filtered = [r for r in results if r[0] <= min_gap + 1e-12]

    # Step 4: largest sum
    max_sum = max(r[1] for r in filtered)
    filtered = [r for r in filtered if r[1] >= max_sum - 1e-12]

    # Step 5: highest threshold
    best_t = max(r[2] for r in filtered)
    return float(best_t)


def alert_numbers(y_true: ArrayLike, score: ArrayLike, threshold: float) -> dict:
    """Numbers at a fixed alert level (alert = score >= threshold).

    Returns a dict with exactly these keys:
      tp, fp, tn, fn (int), alerts (int, = tp + fp),
      sensitivity = tp / (tp + fn), specificity = tn / (tn + fp), precision = tp / (tp + fp),
      false_alarms_per_caught_injury = fp / tp, alerts_per_100_events = 100 * alerts / n,
      mcc = (tp*tn - fp*fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn)).
    Any value whose denominator is 0 is None (never 0). Floats are Python floats.
    """
    y, s = _to_arrays(y_true, score)
    n = len(y)
    alerted = s >= threshold
    tp = int(np.sum(alerted & (y == 1)))
    fp = int(np.sum(alerted & (y == 0)))
    tn = int(np.sum(~alerted & (y == 0)))
    fn = int(np.sum(~alerted & (y == 1)))
    alerts = tp + fp

    # sensitivity = tp / (tp + fn)
    sens_denom = tp + fn
    sensitivity = float(tp / sens_denom) if sens_denom > 0 else None

    # specificity = tn / (tn + fp)
    spec_denom = tn + fp
    specificity = float(tn / spec_denom) if spec_denom > 0 else None

    # precision = tp / (tp + fp)
    prec_denom = tp + fp
    precision = float(tp / prec_denom) if prec_denom > 0 else None

    # false_alarms_per_caught_injury = fp / tp
    false_alarms = float(fp / tp) if tp > 0 else None

    # alerts_per_100_events = 100 * alerts / n
    alerts_per_100 = float(100 * alerts / n)

    # mcc = (tp*tn - fp*fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))
    mcc_denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if mcc_denom_sq == 0:
        mcc_val = None
    else:
        mcc_val = float((tp * tn - fp * fn) / math.sqrt(mcc_denom_sq))

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "alerts": alerts,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "false_alarms_per_caught_injury": false_alarms,
        "alerts_per_100_events": alerts_per_100,
        "mcc": mcc_val,
    }


def fit_platt(score_val: ArrayLike, y_val: ArrayLike) -> tuple[float, float]:
    """Platt scaling fitted on validation runners: an UNPENALISED logistic regression
    P(injury) = 1 / (1 + exp(-(slope * score + intercept))). Returns (slope, intercept)."""
    s = np.asarray(score_val, dtype=float).reshape(-1, 1)
    y = np.asarray(y_val, dtype=float)
    # Unpenalised: C=inf turns off regularisation (sklearn 1.9+)
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000)
    lr.fit(s, y)
    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    return (slope, intercept)


def apply_platt(score: ArrayLike, params: tuple[float, float]) -> np.ndarray:
    """Probabilities from fit_platt's (slope, intercept), as a float numpy array."""
    s = np.asarray(score, dtype=float)
    slope, intercept = params
    return 1.0 / (1.0 + np.exp(-(slope * s + intercept)))


def platt_slope_flag(params: tuple[float, float]) -> bool:
    """True when the Platt slope is 0 or below (it would flatten or flip the model's ranking)."""
    return params[0] <= 0


def brier(y_true: ArrayLike, prob: ArrayLike) -> float:
    """Brier score: mean of (prob - y)^2. Lower is better."""
    y, p = _to_arrays(y_true, prob)
    return float(np.mean((p - y) ** 2))


def ece_equal_count(y_true: ArrayLike, prob: ArrayLike, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins.

    Sort the events by prob (stable sort: numpy argsort kind="stable"), split the sorted order
    into `bins` nearly equal groups with numpy.array_split, then
    ECE = sum over groups of (group size / n) * |mean(y in group) - mean(prob in group)|.
    """
    y, p = _to_arrays(y_true, prob)
    n = len(y)
    order = np.argsort(p, kind="stable")
    y_sorted = y[order]
    p_sorted = p[order]
    groups = np.array_split(np.arange(n), bins)
    ece = 0.0
    for g in groups:
        if len(g) == 0:
            continue
        weight = len(g) / n
        ece += weight * abs(float(np.mean(y_sorted[g])) - float(np.mean(p_sorted[g])))
    return float(ece)


def calibration_slope_intercept(y_true: ArrayLike, prob: ArrayLike) -> tuple[float, float]:
    """(slope, intercept) of calibration (Van Calster et al. 2019).

    Let L = log(p / (1 - p)) with p clipped to [1e-6, 1 - 1e-6].
    slope: coefficient b of an unpenalised logistic regression y ~ c + b * L (1 = perfect).
    intercept: the a that solves mean(y) = mean(1 / (1 + exp(-(a + L)))), i.e. L used as a
    fixed offset (0 = perfect; below 0 = risks too high on average).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
    logit_p = np.log(p_clipped / (1 - p_clipped))

    # slope: unpenalised logistic regression y ~ c + b * L
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000)
    lr.fit(logit_p.reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0])

    # intercept: find a such that mean(y) = mean(1 / (1 + exp(-(a + L))))
    # This is a 1-D root-finding problem. Minimise |mean(y) - mean(sigmoid(a + L))|.
    mean_y = float(np.mean(y))

    def objective(a: float) -> float:
        pred = 1.0 / (1.0 + np.exp(-(a + logit_p)))
        return (float(np.mean(pred)) - mean_y) ** 2

    result = minimize_scalar(objective, bounds=(-10, 10), method="bounded")
    intercept = float(result.x)
    return (slope, intercept)


def net_benefit(y_true: ArrayLike, prob: ArrayLike, thresholds: ArrayLike) -> np.ndarray:
    """Decision-curve net benefit for each threshold t (alert = prob >= t):
    NB(t) = TP / n - (FP / n) * t / (1 - t). Returns a float numpy array, one value per t."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    ts = np.asarray(thresholds, dtype=float)
    n = len(y)
    result = np.empty(len(ts), dtype=float)
    for i, t in enumerate(ts):
        alerted = p >= t
        tp = float(np.sum(alerted & (y == 1)))
        fp = float(np.sum(alerted & (y == 0)))
        result[i] = tp / n - (fp / n) * (t / (1 - t))
    return result


def net_benefit_treat_all(y_true: ArrayLike, thresholds: ArrayLike) -> np.ndarray:
    """Net benefit of alerting everyone: prevalence - (1 - prevalence) * t / (1 - t)."""
    y = np.asarray(y_true, dtype=float)
    ts = np.asarray(thresholds, dtype=float)
    prevalence = float(np.mean(y))
    return prevalence - (1 - prevalence) * ts / (1 - ts)


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Macro-F1 averaged over the labels that appear in y_true only (labels that are only
    predicted are ignored); a label with no correct prediction scores 0 (zero_division=0)."""
    y = np.asarray(y_true)
    yp = np.asarray(y_pred)
    # Only labels present in y_true
    labels = np.unique(y)
    return float(f1_score(y, yp, labels=labels, average="macro", zero_division=0))
