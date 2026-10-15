"""Tests for the dataset loaders (piece 2), on tiny fake data."""

import numpy as np
import pandas as pd
import pytest

from src.data import loaders


def _runner_frame(perceived: bool = False) -> pd.DataFrame:
    """2 features x 3 steps; each cell = 10 * suffix + feature number, so the order is visible."""
    names = ["perceived exertion", "perceived trainingSuccess", "perceived recovery"] if perceived \
        else ["a", "b"]
    data = {}
    for s in range(3):
        for j, n in enumerate(names):
            data[n if s == 0 else f"{n}.{s}"] = [10.0 * s + j, 10.0 * s + j]
    data.update({"Athlete ID": [1, 0], "injury": [1, 0], "Date": [5, 3]})
    return pd.DataFrame(data)


def test_day_file_keeps_suffix_order_oldest_to_newest():
    # README v2: day suffix 0 = 7 days before (oldest), highest suffix = the day before (newest).
    d = loaders.lovdal_arrays(_runner_frame(), "day")
    assert d.X.shape == (2, 3, 2)
    assert d.X[0, 0, 0] == 0.0 and d.X[0, -1, 0] == 20.0  # step 0 = suffix 0, last = suffix 2
    assert d.feature_names == ["a", "b"]


def test_week_file_is_reversed_to_oldest_to_newest():
    # README v2: week suffix 0 = the week before (newest), suffix 2 = three weeks before (oldest).
    frame = _runner_frame()
    frame["rel total kms week 0_1"] = [1.5, 0.5]
    w = loaders.lovdal_arrays(frame, "week")
    assert w.X[0, 0, 0] == 20.0 and w.X[0, -1, 0] == 0.0  # first = suffix 2, last = suffix 0
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
