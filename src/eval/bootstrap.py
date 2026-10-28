"""How sure are we? Runner-level bootstrap intervals for piece-3 scores (D-024).

Task card: docs/handoff/WP-02-bootstrap.md. Tests: tests/handoff/test_wp02_bootstrap.py.
A bootstrap re-draws the data many times and recomputes a score each time; the spread of those
scores shows how uncertain the score is. Here we re-draw whole RUNNERS (never single rows),
separately inside each test group, because one runner's days are related (cluster bootstrap,
Field & Welsh 2007). Pure functions only: no files, no printing, no global random state.

A "metric" is any function metric(y, score) -> float, that returns None or raises ValueError
when it cannot be computed (e.g. src.eval.metrics.roc_auc or average_precision).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray
Metric = Callable[[np.ndarray, np.ndarray], "float | None"]
Statistic = Callable[["np.ndarray | None"], "float | None"]

def resample_runners(athlete: ArrayLike, group: ArrayLike,
                     rng: np.random.Generator) -> np.ndarray:
    """Row indices of ONE bootstrap sample.

    For each test group (in sorted order of the group label): take that group's runner IDs
    (sorted), draw the same number of runners WITH replacement using `rng`, and add ALL rows of
    every drawn runner (a runner drawn twice adds its rows twice). Concatenate in drawing order.
    The draws must depend only on the runner lists, never on how many rows a runner has.
    ValueError if athlete and group have different lengths.
    """
    athlete = np.asarray(athlete)
    group = np.asarray(group)
    if len(athlete) != len(group):
        raise ValueError(f"Length mismatch: {len(athlete)} vs {len(group)}")

    all_indices = []
    # Process groups in sorted order so the rng draws are deterministic
    for g in np.sort(np.unique(group)):
        g_mask = group == g
        g_rows = np.flatnonzero(g_mask)
        # Sorted unique runners in this group
        runners = np.sort(np.unique(athlete[g_rows]))
        n_runners = len(runners)
        # Draw runners with replacement
        drawn = rng.choice(runners, size=n_runners, replace=True)
        # For each drawn runner, add all their rows (in original order)
        for r in drawn:
            row_indices = g_rows[athlete[g_rows] == r]
            all_indices.append(row_indices)

    return np.concatenate(all_indices)


def _safe_metric(metric: Metric, y: np.ndarray, score: np.ndarray) -> float | None:
    """Call metric, catching ValueError as None (same as a metric returning None)."""
    try:
        return metric(y, score)
    except ValueError:
        return None


def group_mean(metric: Metric, y: ArrayLike, score: ArrayLike, group: ArrayLike,
               idx: np.ndarray | None = None) -> float | None:
    """Mean over test groups of metric(y_g, score_g), using only the rows `idx` (all rows if None).

    Groups are those present among the rows used. If the metric returns None or raises
    ValueError for ANY group, return None. Returns a plain Python float.
    """
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    group = np.asarray(group)

    if idx is not None:
        y = y[idx]
        score = score[idx]
        group = group[idx]

    values = []
    for g in np.sort(np.unique(group)):
        mask = group == g
        val = _safe_metric(metric, y[mask], score[mask])
        if val is None:
            return None
        values.append(val)

    return float(np.mean(values))


def _bootstrap(stat: Statistic, athlete: np.ndarray, group: np.ndarray, resamples: int,
               seed: int, level: float) -> dict:
    """The loop shared by every public function below.

    stat(None) is the statistic on all rows; stat(idx) is the statistic on one resample's rows.
    A resample where stat returns None is discarded (counted).
    """
    estimate = stat(None)
    if estimate is None:
        raise ValueError("Metric is undefined on the observed data")

    rng = np.random.default_rng(seed)
    values = []
    discarded = 0
    for _ in range(resamples):
        val = stat(resample_runners(athlete, group, rng))
        if val is None:
            discarded += 1
        else:
            values.append(val)

    if discarded > resamples / 2:
        raise ValueError(f"More than half of resamples discarded: {discarded}/{resamples}")

    low, high = np.percentile(values, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {
        "estimate": float(estimate),
        "low": float(low),
        "high": float(high),
        "resamples_used": len(values),
        "resamples_discarded": discarded,
    }


def bootstrap_ci(metric: Metric, y: ArrayLike, score: ArrayLike, athlete: ArrayLike,
                 group: ArrayLike, resamples: int = 2000, seed: int = 0,
                 level: float = 0.95) -> dict:
    """Percentile interval of the group mean of `metric` over runner resamples.

    Use rng = numpy.random.default_rng(seed) and call resample_runners `resamples` times.
    A resample whose group_mean is None is discarded (counted). Raise ValueError if more than
    half of the resamples are discarded, or if the metric is undefined on the observed data.
    Returns {"estimate": group_mean on all rows, "low": percentile 100*(1-level)/2,
             "high": percentile 100*(1+level)/2 (numpy.percentile, default method),
             "resamples_used": int, "resamples_discarded": int}  (floats are Python floats)
    """
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    group = np.asarray(group)
    return _bootstrap(lambda idx: group_mean(metric, y, score, group, idx),
                      np.asarray(athlete), group, resamples, seed, level)


def paired_bootstrap_difference(metric: Metric, y: ArrayLike, score_a: ArrayLike,
                                score_b: ArrayLike, athlete: ArrayLike, group: ArrayLike,
                                resamples: int = 2000, seed: int = 0,
                                level: float = 0.95) -> dict:
    """Same as bootstrap_ci, but for the difference group_mean(a) - group_mean(b).

    PAIRED: both models are scored on the SAME resampled rows in every resample. A resample is
    discarded if either group mean is None. Same return keys and ValueError rules.
    """
    y = np.asarray(y, dtype=float)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    group = np.asarray(group)

    def difference(idx: np.ndarray | None) -> float | None:
        val_a = group_mean(metric, y, score_a, group, idx)
        val_b = group_mean(metric, y, score_b, group, idx)
        return None if val_a is None or val_b is None else val_a - val_b

    return _bootstrap(difference, np.asarray(athlete), group, resamples, seed, level)


def reproduction_check(y: ArrayLike, scores_by_seed: Sequence[ArrayLike], athlete: ArrayLike,
                       metric: Metric, resamples: int = 2000, seed: int = 0) -> dict:
    """D-024 pass rule for the Lövdal reproduction check (one group: the stand-in runners).

    Statistic = mean over seeds of metric(y, scores_of_that_seed) (each seed scored on the same
    resampled rows). Runner bootstrap as in bootstrap_ci with level 0.95.
    Returns {"estimate", "low", "high", "passed": low > 0.5 (a Python bool),
             "resamples_used", "resamples_discarded"}.
    """
    y = np.asarray(y, dtype=float)
    scores = [np.asarray(s, dtype=float) for s in scores_by_seed]
    group = np.zeros(len(y), dtype=int)  # one group: all rows

    def mean_over_seeds(idx: np.ndarray | None) -> float | None:
        vals = [group_mean(metric, y, s, group, idx) for s in scores]
        return None if any(v is None for v in vals) else float(np.mean(vals))

    result = _bootstrap(mean_over_seeds, np.asarray(athlete), group, resamples, seed, 0.95)
    return {**result, "passed": bool(result["low"] > 0.5)}


def paired_seed_mean_difference(metric: Metric, y: ArrayLike,
                                scores_a_by_seed: Sequence[ArrayLike],
                                scores_b_by_seed: Sequence[ArrayLike], athlete: ArrayLike,
                                group: ArrayLike, resamples: int = 2000, seed: int = 0,
                                level: float = 0.95) -> dict:
    """Paired difference of seed-averaged group means: mean over A's seeds of
    group_mean(metric) minus the same for B, both on the SAME resampled rows (D-023(10)).
    With one seed each this equals paired_bootstrap_difference. Same keys and ValueError rules."""
    y = np.asarray(y, dtype=float)
    group = np.asarray(group)
    a = [np.asarray(s, dtype=float) for s in scores_a_by_seed]
    b = [np.asarray(s, dtype=float) for s in scores_b_by_seed]

    def mean_over(scores, idx):
        vals = [group_mean(metric, y, s, group, idx) for s in scores]
        return None if any(v is None for v in vals) else float(np.mean(vals))

    def difference(idx: np.ndarray | None) -> float | None:
        va, vb = mean_over(a, idx), mean_over(b, idx)
        return None if va is None or vb is None else va - vb

    return _bootstrap(difference, np.asarray(athlete), group, resamples, seed, level)
