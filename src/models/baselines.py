"""Untuned and simple injury baselines (D-022, D-023, D-024) and the leak canary model.

- constant risk: the training injury rate for everyone (raises no alerts, D-024);
- ACWR-style rule: newest week's km / max(mean km of the two earlier weeks, 1 km), on RAW km;
- logistic regression: the tuned linear baseline (sklearn, deterministic).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.loaders import RunnerData


def constant_risk(y_train: np.ndarray, n: int) -> np.ndarray:
    return np.full(n, float(np.mean(y_train)))


def acwr_ratio(week: RunnerData, km_feature: str = "total kms",
               floor_km: float = 1.0) -> np.ndarray:
    """Uncoupled acute:chronic ratio per event, from the raw week file (oldest week first)."""
    km = week.X[:, :, week.feature_names.index(km_feature)].astype(np.float64)
    newest, earlier = km[:, 2], km[:, :2].mean(axis=1)
    return newest / np.maximum(earlier, floor_km)


def logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                        setting: dict) -> LogisticRegression:
    """Fit an L2 logistic regression with one setting (C, class_weight; fixed solver settings).

    The config says penalty l2: that is sklearn's default (its `penalty` argument is deprecated
    since 1.8), so it is checked rather than passed.
    """
    if setting.get("penalty", "l2") != "l2":
        raise ValueError("only the L2 penalty is declared (configs/baselines/lovdal.yaml)")
    params = {k: setting[k] for k in ("C", "class_weight", "solver", "max_iter")}
    return LogisticRegression(**params).fit(X_train, y_train)


def shuffled_labels(y_train: np.ndarray, seed: int) -> np.ndarray:
    """Leak canary: training labels shuffled among the training rows (Ojala & Garriga 2010)."""
    return np.random.default_rng(seed).permutation(y_train)
