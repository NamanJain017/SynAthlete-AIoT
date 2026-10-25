"""Scores every piece-3 model is judged with (D-022, D-024): the project's "answer key" maths.

Task card: docs/handoff/WP-01-metrics.md. Tests: tests/handoff/test_wp01_metrics.py.
Rules: pure functions only (no file reading, no printing, no global state); inputs may be lists or
numpy arrays; return plain Python floats (or None where a value is undefined), never numpy scalars.
y_true / y_val are 0/1 labels (1 = injury); score is any number where higher = more risky;
prob is a probability in [0, 1].
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray

# Set to True only when every function below is written; the tests skip while it is False.
IMPLEMENTED = False


def average_precision(y_true: ArrayLike, score: ArrayLike) -> float:
    """PR-AUC as average precision: sum over thresholds of (recall step) x precision.

    Step-wise, with no straight-line interpolation between points (use
    sklearn.metrics.average_precision_score). Raise ValueError if y_true has no 1.
    """
    raise NotImplementedError


def roc_auc(y_true: ArrayLike, score: ArrayLike) -> float | None:
    """ROC-AUC (chance that a random injury scores higher than a random healthy event; ties
    count half). Return None when y_true contains only one class."""
    raise NotImplementedError


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
    raise NotImplementedError


def alert_numbers(y_true: ArrayLike, score: ArrayLike, threshold: float) -> dict:
    """Numbers at a fixed alert level (alert = score >= threshold).

    Returns a dict with exactly these keys:
      tp, fp, tn, fn (int), alerts (int, = tp + fp),
      sensitivity = tp / (tp + fn), specificity = tn / (tn + fp), precision = tp / (tp + fp),
      false_alarms_per_caught_injury = fp / tp, alerts_per_100_events = 100 * alerts / n,
      mcc = (tp*tn - fp*fn) / sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn)).
    Any value whose denominator is 0 is None (never 0). Floats are Python floats.
    """
    raise NotImplementedError


def fit_platt(score_val: ArrayLike, y_val: ArrayLike) -> tuple[float, float]:
    """Platt scaling fitted on validation runners: an UNPENALISED logistic regression
    P(injury) = 1 / (1 + exp(-(slope * score + intercept))). Returns (slope, intercept)."""
    raise NotImplementedError


def apply_platt(score: ArrayLike, params: tuple[float, float]) -> np.ndarray:
    """Probabilities from fit_platt's (slope, intercept), as a float numpy array."""
    raise NotImplementedError


def platt_slope_flag(params: tuple[float, float]) -> bool:
    """True when the Platt slope is 0 or below (it would flatten or flip the model's ranking)."""
    raise NotImplementedError


def brier(y_true: ArrayLike, prob: ArrayLike) -> float:
    """Brier score: mean of (prob - y)^2. Lower is better."""
    raise NotImplementedError


def ece_equal_count(y_true: ArrayLike, prob: ArrayLike, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins.

    Sort the events by prob (stable sort: numpy argsort kind="stable"), split the sorted order
    into `bins` nearly equal groups with numpy.array_split, then
    ECE = sum over groups of (group size / n) * |mean(y in group) - mean(prob in group)|.
    """
    raise NotImplementedError


def calibration_slope_intercept(y_true: ArrayLike, prob: ArrayLike) -> tuple[float, float]:
    """(slope, intercept) of calibration (Van Calster et al. 2019).

    Let L = log(p / (1 - p)) with p clipped to [1e-6, 1 - 1e-6].
    slope: coefficient b of an unpenalised logistic regression y ~ c + b * L (1 = perfect).
    intercept: the a that solves mean(y) = mean(1 / (1 + exp(-(a + L)))), i.e. L used as a
    fixed offset (0 = perfect; below 0 = risks too high on average).
    """
    raise NotImplementedError


def net_benefit(y_true: ArrayLike, prob: ArrayLike, thresholds: ArrayLike) -> np.ndarray:
    """Decision-curve net benefit for each threshold t (alert = prob >= t):
    NB(t) = TP / n - (FP / n) * t / (1 - t). Returns a float numpy array, one value per t."""
    raise NotImplementedError


def net_benefit_treat_all(y_true: ArrayLike, thresholds: ArrayLike) -> np.ndarray:
    """Net benefit of alerting everyone: prevalence - (1 - prevalence) * t / (1 - t)."""
    raise NotImplementedError


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Macro-F1 averaged over the labels that appear in y_true only (labels that are only
    predicted are ignored); a label with no correct prediction scores 0 (zero_division=0)."""
    raise NotImplementedError
