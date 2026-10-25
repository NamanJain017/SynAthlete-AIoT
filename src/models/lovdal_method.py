"""The Lövdal et al. (2021) bagged XGBoost, from the authors' main.ipynb (cell index 1).

- athlete_balanced_bag: getBalancedSubset (lines 132-156): every training athlete with both
  classes gives floor(2048 / k) healthy and as many injury events, drawn with replacement.
- train_bags: 9 bags; each bag's depth and tree count drawn from {2, 3} and {256, 512}
  (lines 120-123, 360-368); learning rate 0.01.
- paper_scaling: getMeanStd + normalize2 (lines 45-57): each athlete z-scored with the mean and
  SD of THEIR OWN HEALTHY events (SD 0 -> 0.01). It reads the labels, and for test athletes the
  test labels (line 263): used only in the labelled reproduction check (D-021, D-023).
- reproduction: the exact copy on runners 0-53 -> 54-63, with per-bag Platt on a balanced
  sample of training events (lines 276-283, 310-316).
Deviations, documented: our random generator is seeded; Platt is our unpenalised fit_platt on the
bag probability (sklearn's CalibratedClassifierCV adds Platt's target smoothing).
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from src.eval import metrics


def athlete_balanced_bag(y: np.ndarray, athlete: np.ndarray, rng: np.random.Generator,
                         samples_per_class: int = 2048) -> np.ndarray:
    """Row indices of one balanced bag (with replacement), athlete by athlete."""
    both = [a for a in np.unique(athlete) if len(np.unique(y[athlete == a])) == 2]
    per_athlete = samples_per_class // len(both)
    rows = []
    for a in both:
        for label in (1, 0):
            pool = np.flatnonzero((athlete == a) & (y == label))
            rows.append(rng.choice(pool, size=per_athlete, replace=True))
    return np.concatenate(rows)


def train_bags(X: np.ndarray, y: np.ndarray, athlete: np.ndarray, seed: int, paper: dict,
               threads: int = -1) -> list[XGBClassifier]:
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(paper["bags"]):
        idx = athlete_balanced_bag(y, athlete, rng, paper["samples_per_class"])
        model = XGBClassifier(objective="binary:logistic", learning_rate=paper["learning_rate"],
                              max_depth=int(rng.choice(paper["max_depth_choices"])),
                              n_estimators=int(rng.choice(paper["n_estimators_choices"])),
                              random_state=int(rng.integers(2**31)), n_jobs=threads,
                              eval_metric="auc")
        models.append(model.fit(X[idx], y[idx]))
    return models


def bag_mean(models: list[XGBClassifier], X: np.ndarray) -> np.ndarray:
    """Label-free variant's score: the mean bag probability (Platt comes later, on validation)."""
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def paper_scaling(X: np.ndarray, y: np.ndarray, athlete: np.ndarray) -> np.ndarray:
    """Each athlete z-scored with their own healthy events' mean and SD (READS THE LABELS)."""
    out = np.empty(X.shape, dtype=np.float64)
    for a in np.unique(athlete):
        rows = athlete == a
        healthy = X[rows & (y == 0)].astype(np.float64)
        mean = healthy.mean(axis=0)
        sd = healthy.std(axis=0, ddof=1) if len(healthy) > 1 else np.full(X.shape[1], np.nan)
        out[rows] = (X[rows] - mean) / np.where(sd == 0, 0.01, sd)  # pandas std; 0 -> 0.01
    return out


def reproduction(X: np.ndarray, y: np.ndarray, athlete: np.ndarray, train_ids: range,
                 test_ids: range, seed: int, paper: dict) -> np.ndarray:
    """Exact-copy test probabilities for one seed (labelled reproduction check, D-021)."""
    train, test = np.isin(athlete, list(train_ids)), np.isin(athlete, list(test_ids))
    Xs = np.empty(X.shape, dtype=np.float64)
    Xs[train] = paper_scaling(X[train], y[train], athlete[train])
    Xs[test] = paper_scaling(X[test], y[test], athlete[test])  # the label-reading step
    models = train_bags(Xs[train], y[train], athlete[train], seed, paper)
    rng = np.random.default_rng(seed + 10_000)
    val = athlete_balanced_bag(y[train], athlete[train], rng, paper["samples_per_class"])
    X_val, y_val = Xs[train][val], y[train][val]
    probs = []
    for m in models:  # per-bag Platt on the balanced training sample, as in the notebook
        params = metrics.fit_platt(m.predict_proba(X_val)[:, 1], y_val)
        probs.append(metrics.apply_platt(m.predict_proba(Xs[test])[:, 1], params))
    return np.mean(probs, axis=0)
