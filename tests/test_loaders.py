"""Tests for the dataset loaders (piece 2): tiny fake data, plus order checks on the real
Lövdal files that skip only when the raw data is not on this machine."""

import numpy as np
import pandas as pd
import pytest

from src.data import loaders, readers


def _runner_frame(perceived: bool = False) -> pd.DataFrame:
    """2 features x 3 steps; cell = 10 * suffix + feature number (+ 100 in the first row), so
    both the step order and the row each value belongs to are visible."""
    names = ["perceived exertion", "perceived trainingSuccess", "perceived recovery"] if perceived \
        else ["a", "b"]
    data = {}
    for s in range(3):
        for j, n in enumerate(names):
            data[n if s == 0 else f"{n}.{s}"] = [100.0 + 10.0 * s + j, 10.0 * s + j]
    data.update({"Athlete ID": [1, 0], "injury": [1, 0], "Date": [5, 3]})
    return pd.DataFrame(data)


def test_day_file_keeps_suffix_order_oldest_to_newest():
    # README v2: day suffix 0 = 7 days before (oldest), highest suffix = the day before (newest).
    d = loaders.lovdal_arrays(_runner_frame(), "day")
    assert d.X.shape == (2, 3, 2)
    assert d.X[0, 0, 0] == 0.0 and d.X[0, -1, 0] == 20.0  # step 0 = suffix 0, last = suffix 2
    assert d.X[1, 0, 0] == 100.0 and d.X[1, -1, 1] == 121.0  # values moved with their row
    assert d.feature_names == ["a", "b"]


def test_week_file_is_reversed_to_oldest_to_newest():
    # README v2: week suffix 0 = the week before (newest), suffix 2 = three weeks before (oldest).
    frame = _runner_frame()
    frame["rel total kms week 0_1"] = [1.5, 0.5]
    w = loaders.lovdal_arrays(frame, "week")
    assert w.X[0, 0, 0] == 20.0 and w.X[0, -1, 0] == 0.0  # first = suffix 2, last = suffix 0
    assert w.X[1, 0, 1] == 121.0 and w.extra[1, 0] == 1.5  # athlete 1 keeps its own values
    assert w.extra_names == ("rel total kms week 0_1",)
    assert w.extra.shape == (2, 1)


def test_rows_are_sorted_by_athlete_then_date():
    d = loaders.lovdal_arrays(_runner_frame(), "day")
    assert d.athlete_id.tolist() == [0, 1] and d.date.tolist() == [3, 5] and d.y.tolist() == [0, 1]


def test_rest_day_mask_marks_days_with_all_perceived_values_at_minus_0_01():
    frame = _runner_frame(perceived=True)
    for n in ("perceived exertion.1", "perceived trainingSuccess.1", "perceived recovery.1"):
        frame[n] = -0.01
    d = loaders.lovdal_arrays(frame, "day")
    assert d.rest_day.tolist() == [[False, True, False], [False, True, False]]


def test_steps_with_different_features_are_rejected():
    frame = _runner_frame().rename(columns={"b.2": "c.2"})
    with pytest.raises(ValueError, match="different features"):
        loaders.lovdal_arrays(frame, "day")


def test_shared_days_align_rows_and_check_labels():
    day = loaders.lovdal_arrays(_runner_frame(), "day")
    week = loaders.lovdal_arrays(_runner_frame().iloc[::-1].reset_index(drop=True), "week")
    i_day, i_week = loaders.shared_days(day, week)
    assert np.array_equal(day.athlete_id[i_day], week.athlete_id[i_week])
    week.y = 1 - week.y
    with pytest.raises(ValueError, match="labels differ"):
        loaders.shared_days(day, week)


# ---------------------------------------------------------------- real Lövdal files (if present)


@pytest.fixture(scope="module")
def real():
    files = readers.find_runner_files(loaders.RAW / "lovdal")
    if not {"day", "week"} <= set(files):
        pytest.skip("raw Lövdal files are not on this machine")
    return loaders.load_lovdal("day"), loaders.load_lovdal("week")


def test_real_day_windows_slide_one_day_oldest_to_newest(real):
    """Day d+1's window must equal day d's window moved one step: proves the step order."""
    day, _ = real
    nxt = (day.athlete_id[1:] == day.athlete_id[:-1]) & (day.date[1:] == day.date[:-1] + 1)
    earlier, later = day.X[:-1][nxt], day.X[1:][nxt]
    assert nxt.sum() > 40_000
    assert np.array_equal(earlier[:, 1:], later[:, :-1])


def test_real_week_change_uses_the_two_newest_weeks(real):
    """'rel total kms week 0_1' = newest week / the week before: the last two steps."""
    _, week = real
    km = week.feature_names.index("total kms")
    rel = week.extra[:, week.extra_names.index("rel total kms week 0_1")]
    ok = week.X[:, -2, km] > 0
    ratio = week.X[ok, -1, km] / week.X[ok, -2, km]
    assert np.median(np.abs(rel[ok] - ratio)) < 1e-3


def test_real_day_and_week_files_hold_the_same_runners(real):
    day, week = real
    assert set(np.unique(day.athlete_id)) == set(np.unique(week.athlete_id))


def test_own_history_counts_earlier_events_within_the_window():
    athlete = np.array([1, 1, 1, 1, 2, 2], dtype=np.int16)
    date = np.array([0, 1, 2, 30, 5, 6], dtype=np.int32)
    count, gap, first = loaders.own_history(athlete, date, days=21)
    assert count.tolist() == [0, 1, 2, 0, 0, 1]  # day 30: days 0-2 are more than 21 days back
    assert np.isnan(gap[0]) and gap[1:4].tolist() == [1, 1, 28] and gap[5] == 1
    assert first.tolist() == [True, False, False, False, True, False]
