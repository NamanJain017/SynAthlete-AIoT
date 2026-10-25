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
IMPLEMENTED = False


def resample_runners(athlete: ArrayLike, group: ArrayLike,
                     rng: np.random.Generator) -> np.ndarray:
    """Row indices of ONE bootstrap sample.

    For each test group (in sorted order of the group label): take that group's runner IDs
    (sorted), draw the same number of runners WITH replacement using `rng`, and add ALL rows of
    every drawn runner (a runner drawn twice adds its rows twice). Concatenate in drawing order.
    The draws must depend only on the runner lists, never on how many rows a runner has.
    ValueError if athlete and group have different lengths.
    """
    raise NotImplementedError


def group_mean(metric: Metric, y: ArrayLike, score: ArrayLike, group: ArrayLike,
               idx: np.ndarray | None = None) -> float | None:
    """Mean over test groups of metric(y_g, score_g), using only the rows `idx` (all rows if None).

    Groups are those present among the rows used. If the metric returns None or raises
    ValueError for ANY group, return None. Returns a plain Python float.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def paired_bootstrap_difference(metric: Metric, y: ArrayLike, score_a: ArrayLike,
                                score_b: ArrayLike, athlete: ArrayLike, group: ArrayLike,
                                resamples: int = 2000, seed: int = 0,
                                level: float = 0.95) -> dict:
    """Same as bootstrap_ci, but for the difference group_mean(a) - group_mean(b).

    PAIRED: both models are scored on the SAME resampled rows in every resample. A resample is
    discarded if either group mean is None. Same return keys and ValueError rules.
    """
    raise NotImplementedError


def reproduction_check(y: ArrayLike, scores_by_seed: Sequence[ArrayLike], athlete: ArrayLike,
                       metric: Metric, resamples: int = 2000, seed: int = 0) -> dict:
    """D-024 pass rule for the Lövdal reproduction check (one group: the stand-in runners).

    Statistic = mean over seeds of metric(y, scores_of_that_seed) (each seed scored on the same
    resampled rows). Runner bootstrap as in bootstrap_ci with level 0.95.
    Returns {"estimate", "low", "high", "passed": low > 0.5 (a Python bool),
             "resamples_used", "resamples_discarded"}.
    """
    raise NotImplementedError
