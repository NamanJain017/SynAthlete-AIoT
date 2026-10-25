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

# Set to True only when every function below is written; the tests skip while it is False.
IMPLEMENTED = True


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
    athlete = np.asarray(athlete)
    group = np.asarray(group)

    estimate = group_mean(metric, y, score, group)
    if estimate is None:
        raise ValueError("Metric is undefined on the observed data")

    rng = np.random.default_rng(seed)
    boot_values = []
    discarded = 0

    for _ in range(resamples):
        idx = resample_runners(athlete, group, rng)
        val = group_mean(metric, y, score, group, idx)
        if val is None:
            discarded += 1
        else:
            boot_values.append(val)

    if discarded > resamples / 2:
        raise ValueError(f"More than half of resamples discarded: {discarded}/{resamples}")

    boot_arr = np.array(boot_values)
    alpha = 100 * (1 - level) / 2
    low = float(np.percentile(boot_arr, alpha))
    high = float(np.percentile(boot_arr, 100 - alpha))

    return {
        "estimate": float(estimate),
        "low": low,
        "high": high,
        "resamples_used": len(boot_values),
        "resamples_discarded": discarded,
    }


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
    athlete = np.asarray(athlete)
    group = np.asarray(group)

    est_a = group_mean(metric, y, score_a, group)
    est_b = group_mean(metric, y, score_b, group)
    if est_a is None or est_b is None:
        raise ValueError("Metric is undefined on the observed data for one or both models")
    estimate = est_a - est_b

    rng = np.random.default_rng(seed)
    diffs = []
    discarded = 0

    for _ in range(resamples):
        idx = resample_runners(athlete, group, rng)
        val_a = group_mean(metric, y, score_a, group, idx)
        val_b = group_mean(metric, y, score_b, group, idx)
        if val_a is None or val_b is None:
            discarded += 1
        else:
            diffs.append(val_a - val_b)

    if discarded > resamples / 2:
        raise ValueError(f"More than half of resamples discarded: {discarded}/{resamples}")

    diff_arr = np.array(diffs)
    alpha = 100 * (1 - level) / 2
    low = float(np.percentile(diff_arr, alpha))
    high = float(np.percentile(diff_arr, 100 - alpha))

    return {
        "estimate": float(estimate),
        "low": low,
        "high": high,
        "resamples_used": len(diffs),
        "resamples_discarded": discarded,
    }


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
    athlete = np.asarray(athlete)
    # One group: all rows belong to the same group
    group = np.zeros(len(y), dtype=int)

    def mean_over_seeds(y_sub: np.ndarray, _score_unused: np.ndarray) -> float | None:
        """Metric that averages the real metric over all seeds for the given row subset."""
        # We need to figure out which rows were selected. Since group_mean passes y[mask]
        # and score[mask], but we need the row indices to index into each seed's scores,
        # we use a different approach: compute via bootstrap_ci directly.
        # Actually, this helper won't work in group_mean because group_mean slices by index.
        # We need a different approach.
        raise NotImplementedError  # placeholder, see below

    # Direct implementation: replicate the bootstrap loop with the mean-over-seeds statistic
    def _stat_on_rows(idx: np.ndarray | None) -> float | None:
        """Mean over seeds of metric(y[idx], scores_seed[idx])."""
        if idx is None:
            rows_y = y
            rows_scores = scores
        else:
            rows_y = y[idx]
            rows_scores = [s[idx] for s in scores]
        vals = []
        for s in rows_scores:
            v = _safe_metric(metric, rows_y, s)
            if v is None:
                return None
            vals.append(v)
        return float(np.mean(vals))

    estimate = _stat_on_rows(None)
    if estimate is None:
        raise ValueError("Metric is undefined on the observed data")

    rng = np.random.default_rng(seed)
    boot_values = []
    discarded = 0

    for _ in range(resamples):
        idx = resample_runners(athlete, group, rng)
        val = _stat_on_rows(idx)
        if val is None:
            discarded += 1
        else:
            boot_values.append(val)

    if discarded > resamples / 2:
        raise ValueError(f"More than half of resamples discarded: {discarded}/{resamples}")

    boot_arr = np.array(boot_values)
    low = float(np.percentile(boot_arr, 2.5))
    high = float(np.percentile(boot_arr, 97.5))

    return {
        "estimate": float(estimate),
        "low": low,
        "high": high,
        "passed": bool(low > 0.5),
        "resamples_used": len(boot_values),
        "resamples_discarded": discarded,
    }
