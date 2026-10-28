"""Piece-8 explanation checks (EVAL_PROTOCOL "Explanation criteria", D-027(3)).

Works for any attribution: inputs X (n, ...) and attributions A of the same shape.
- deletion: set each row's top-10 attributed inputs to the baseline (0 = training mean) and
  compare the drop in predicted risk with setting 10 random inputs to 0 (pass: ratio >= 1.5);
- randomisation: rank correlation between attributions of the trained model and of a model
  that learned nothing (pass: below 0.5) (Adebayo et al. 2018 "sanity checks");
- stability: rank correlation of attributions for x and x + 1 % noise (pass: above 0.8).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import spearmanr

Predict = Callable[[np.ndarray], np.ndarray]


def _rowwise_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    vals = [spearmanr(x, y).statistic for x, y in zip(a, b, strict=True)]
    return float(np.nanmean(vals))


def deletion(predict: Predict, X: np.ndarray, A: np.ndarray, k: int = 10,
             seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    flat_x, flat_a = X.reshape(len(X), -1), A.reshape(len(A), -1)
    top, rand = flat_x.copy(), flat_x.copy()
    for i in range(len(X)):
        top[i, np.argsort(-flat_a[i])[:k]] = 0.0
        rand[i, rng.choice(flat_x.shape[1], k, replace=False)] = 0.0
    base = predict(X)
    drop_top = float(np.mean(base - predict(top.reshape(X.shape))))
    drop_rand = float(np.mean(base - predict(rand.reshape(X.shape))))
    ratio = drop_top / drop_rand if drop_rand > 0 else (np.inf if drop_top > 0 else 0.0)
    return {"drop_top": drop_top, "drop_random": drop_rand, "ratio": float(ratio),
            "passed": bool(ratio >= 1.5)}


def randomisation(A_trained: np.ndarray, A_random: np.ndarray) -> dict:
    rho = _rowwise_spearman(np.abs(A_trained), np.abs(A_random))
    return {"rank_correlation": rho, "passed": bool(rho < 0.5)}


def stability(attribute: Callable[[np.ndarray], np.ndarray], X: np.ndarray, A: np.ndarray,
              seed: int = 0) -> dict:
    """Inputs are scaled (SD 1), so 1 % noise = Gaussian noise with SD 0.01."""
    noisy = X + np.random.default_rng(seed).normal(0, 0.01, X.shape).astype(X.dtype)
    rho = _rowwise_spearman(A, attribute(noisy))
    return {"rank_correlation": rho, "passed": bool(rho > 0.8)}
