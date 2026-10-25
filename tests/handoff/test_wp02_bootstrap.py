"""Answer checker for task card WP-02 (docs/handoff/WP-02-bootstrap.md): src/eval/bootstrap.py.

Written by Claude Code before the code exists; builders must not edit this file.
The bootstrap resamples whole runners inside each test group (cluster bootstrap, D-024).
Tests check properties that any correct implementation has, plus small hand-worked cases.
Skipped until bootstrap.IMPLEMENTED is True.
"""

import numpy as np
import pytest

from src.eval import bootstrap
from src.eval.metrics import average_precision, roc_auc

pytestmark = pytest.mark.skipif(not bootstrap.IMPLEMENTED, reason="WP-02 not built yet")

KEYS = {"estimate", "low", "high", "resamples_used", "resamples_discarded"}


def _toy(n_runners: int = 12, rows: int = 5, groups: int = 3, seed: int = 0):
    """Made-up data: runner IDs, test groups, labels and noisy scores; both classes per group."""
    rng = np.random.default_rng(seed)
    athlete = np.repeat(np.arange(n_runners), rows)
    group = athlete % groups
    y = (rng.random(len(athlete)) < 0.3).astype(int)
    for g in range(groups):
        rows_g = np.flatnonzero(group == g)
        y[rows_g[0]], y[rows_g[1]] = 1, 0
    score = y + rng.normal(size=len(y))
    return y, score, athlete, group


# ---------------------------------------------------------------- one resample


def test_resample_is_deterministic_and_keeps_each_groups_runner_count():
    y, s, athlete, group = _toy()
    idx = bootstrap.resample_runners(athlete, group, np.random.default_rng(5))
    assert np.array_equal(idx, bootstrap.resample_runners(athlete, group, np.random.default_rng(5)))
    assert not np.array_equal(idx, bootstrap.resample_runners(athlete, group,
                                                              np.random.default_rng(6)))
    for g in np.unique(group):
        drawn_rows = idx[group[idx] == g]
        per_runner = np.bincount(athlete[drawn_rows], minlength=athlete.max() + 1)
        assert np.all(per_runner % 5 == 0)              # whole runners only (5 rows each)
        assert per_runner.sum() // 5 == len(np.unique(athlete[group == g]))  # same runner count


def test_resample_takes_every_row_of_a_drawn_runner():
    athlete = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3])  # runners with 3, 1, 2 and 4 rows
    group = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    sizes = np.bincount(athlete)
    for seed in range(20):
        idx = bootstrap.resample_runners(athlete, group, np.random.default_rng(seed))
        counts = np.bincount(athlete[idx], minlength=len(sizes))
        assert np.all(counts % sizes == 0)
        assert counts[[0, 1]].sum() >= 1 and set(group[idx]) == {0, 1}


# ---------------------------------------------------------------- the per-group mean


def test_group_mean_by_hand():
    # group 0: y [0, 1], s [0.2, 0.9] -> AUC 1.0
    # group 1: y [0, 1, 1], s [0.5, 0.4, 0.6] -> pairs (0.4 vs 0.5) wrong, (0.6 vs 0.5) right -> 0.5
    y = np.array([0, 1, 0, 1, 1])
    s = np.array([0.2, 0.9, 0.5, 0.4, 0.6])
    group = np.array([0, 0, 1, 1, 1])
    assert bootstrap.group_mean(roc_auc, y, s, group) == pytest.approx(0.75)
    assert bootstrap.group_mean(roc_auc, y, s, group, np.array([0, 1])) == pytest.approx(1.0)


def test_group_mean_is_none_when_any_group_is_undefined():
    y = np.array([0, 1, 0, 0])
    s = np.array([0.1, 0.9, 0.3, 0.4])
    group = np.array([0, 0, 1, 1])                       # group 1 has no injury
    assert bootstrap.group_mean(roc_auc, y, s, group) is None            # metric returns None
    assert bootstrap.group_mean(average_precision, y, s, group) is None  # metric raises


# ---------------------------------------------------------------- intervals


def test_bootstrap_ci_shape_and_determinism():
    y, s, athlete, group = _toy()
    r = bootstrap.bootstrap_ci(roc_auc, y, s, athlete, group, resamples=300, seed=1)
    assert set(r) == KEYS
    assert r["estimate"] == pytest.approx(bootstrap.group_mean(roc_auc, y, s, group))
    assert r["low"] <= r["estimate"] <= r["high"]
    assert r["resamples_used"] + r["resamples_discarded"] == 300
    assert r == bootstrap.bootstrap_ci(roc_auc, y, s, athlete, group, resamples=300, seed=1)
    narrow = bootstrap.bootstrap_ci(roc_auc, y, s, athlete, group, resamples=300, seed=1,
                                    level=0.5)
    assert r["low"] <= narrow["low"] <= narrow["high"] <= r["high"]


def test_paired_difference_of_identical_models_is_exactly_zero():
    y, s, athlete, group = _toy()
    r = bootstrap.paired_bootstrap_difference(roc_auc, y, s, s.copy(), athlete, group,
                                              resamples=200, seed=2)
    assert set(r) == KEYS
    assert (r["estimate"], r["low"], r["high"]) == (0.0, 0.0, 0.0)


def test_paired_difference_detects_a_clearly_better_model():
    y, _, athlete, group = _toy(n_runners=30, rows=6, seed=3)
    rng = np.random.default_rng(4)
    good = y + 0.3 * rng.normal(size=len(y))
    noise = rng.normal(size=len(y))
    r = bootstrap.paired_bootstrap_difference(roc_auc, y, good, noise, athlete, group,
                                              resamples=300, seed=5)
    assert r["estimate"] > 0 and r["low"] > 0


def test_runner_bootstrap_is_unchanged_by_duplicating_rows():
    """Copies of a runner's rows move together, so they must not shrink the interval
    (a row-level bootstrap would wrongly look more certain)."""
    y, s, athlete, group = _toy(n_runners=20, rows=1, groups=2, seed=6)
    k = 30
    once = bootstrap.bootstrap_ci(roc_auc, y, s, athlete, group, resamples=300, seed=7)
    many = bootstrap.bootstrap_ci(roc_auc, np.repeat(y, k), np.repeat(s, k),
                                  np.repeat(athlete, k), np.repeat(group, k),
                                  resamples=300, seed=7)
    for key in ("estimate", "low", "high"):
        assert many[key] == pytest.approx(once[key])


def test_undefined_resamples_are_counted_and_too_many_raise():
    # one group, 10 runners with 3 rows each, only runner 0 ever injured:
    # a resample misses runner 0 about 35 % of the time -> those resamples are discarded
    athlete = np.repeat(np.arange(10), 3)
    group = np.zeros(30, dtype=int)
    y = (athlete == 0).astype(int)
    y[1] = 0                                             # runner 0 has both classes
    s = np.random.default_rng(8).normal(size=30)
    r = bootstrap.bootstrap_ci(roc_auc, y, s, athlete, group, resamples=400, seed=9)
    assert r["resamples_discarded"] > 0
    assert r["resamples_used"] + r["resamples_discarded"] == 400
    # three such groups: a resample needs runner "0" of every group -> most are undefined
    athlete3 = np.concatenate([athlete, athlete + 10, athlete + 20])
    group3 = np.repeat([0, 1, 2], 30)
    with pytest.raises(ValueError):
        bootstrap.bootstrap_ci(roc_auc, np.tile(y, 3), np.tile(s, 3), athlete3, group3,
                               resamples=400, seed=9)


# ---------------------------------------------------------------- the reproduction pass rule


def test_reproduction_check_pass_rule():
    athlete = np.repeat(np.arange(10), 8)
    y = np.tile([0, 0, 0, 1, 0, 0, 0, 0], 10)
    perfect = [y + 0.0, y * 2.0, y * 3.0]                # 3 seeds, all rank perfectly
    r = bootstrap.reproduction_check(y, perfect, athlete, roc_auc, resamples=300, seed=0)
    assert r["estimate"] == pytest.approx(1.0) and r["passed"] is True
    constant = [np.zeros(80), np.zeros(80), np.zeros(80)]  # no skill: AUC 0.5 everywhere
    r = bootstrap.reproduction_check(y, constant, athlete, roc_auc, resamples=300, seed=0)
    assert r["estimate"] == pytest.approx(0.5) and r["low"] == pytest.approx(0.5)
    assert r["passed"] is False                          # pass needs low ABOVE 0.5


def test_module_has_no_file_or_data_access():
    from pathlib import Path

    text = Path(bootstrap.__file__).read_text(encoding="utf-8")
    for forbidden in ("open(", "read_csv", "np.load", "load_dev", "data/", "Path("):
        assert forbidden not in text, f"bootstrap.py must not use {forbidden!r}"
