"""Leakage tests for the person-level splits (piece 2, D-017).

Unit tests use fake data. The checks on data/splits/*.json run once those files exist
(they are created once by `python tasks.py splits` and committed).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.data import loaders, splits

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits"


def _load(name: str) -> dict:
    path = SPLITS / name
    if not path.exists():
        pytest.skip(f"{name} not created yet")
    return json.loads(path.read_text(encoding="utf-8"))


def _roles_disjoint(fold: dict) -> bool:
    roles = [set(fold.get(k, [])) for k in ("train_ids", "val_ids", "test_ids")]
    return all(not (roles[i] & roles[j]) for i in range(3) for j in range(i + 1, 3))


def _fake_runners(n_athletes: int = 30, per_athlete: int = 20) -> loaders.RunnerData:
    athlete = np.repeat(np.arange(n_athletes), per_athlete).astype(np.int16)
    y = np.zeros(len(athlete), dtype=np.int8)
    y[::10] = 1  # 2 injuries per athlete: balanced by construction
    date = np.tile(np.arange(per_athlete), n_athletes).astype(np.int32)
    X = np.zeros((len(y), 7, 2), dtype=np.float32)
    return loaders.RunnerData("day", X, y, athlete, date, ["a", "b"])


# ---------------------------------------------------------------- unit tests (fake data)


def test_write_once_never_overwrites(tmp_path):
    path = tmp_path / "x_folds.json"
    assert splits.write_once(path, {"v": 1}) is True
    assert splits.write_once(path, {"v": 2}) is False
    assert json.loads(path.read_text(encoding="utf-8")) == {"v": 1}


def test_balance_rule_stops_on_unbalanced_groups():
    splits.check_balance([100, 105, 95, 110, 90], "ok")
    with pytest.raises(splits.BalanceError):
        splits.check_balance([100, 100, 100, 100, 40], "bad")


def test_lovdal_split_structure_on_fake_runners():
    day = _fake_runners()
    week = loaders.RunnerData("week", day.X[:, :3], day.y.copy(), day.athlete_id.copy(),
                              day.date.copy(), ["a", "b"])
    s = splits.lovdal_split(day, week)
    assert s["lockbox_ids"] == list(range(20, 30))
    dev = set(range(20))
    tested = [a for f in s["folds"].values() for a in f["test_ids"]]
    assert sorted(tested) == sorted(dev)  # every development athlete tested exactly once
    for f in s["folds"].values():
        assert _roles_disjoint(f)
        assert set(f["train_ids"]) | set(f["val_ids"]) | set(f["test_ids"]) == dev
    assert set(s["final"]["train_ids"]) | set(s["final"]["val_ids"]) == dev
    assert s["final"]["test_ids"] == s["lockbox_ids"]


def test_leave_one_out_tests_each_person_once():
    folds = splits.leave_one_out(["a", "b", "c"])
    assert sorted(f["test_ids"][0] for f in folds.values()) == ["a", "b", "c"]
    assert all(f["test_ids"][0] not in f["train_ids"] for f in folds.values())


# ---------------------------------------------------------------- checks on the saved split files


def test_lovdal_saved_split_is_leak_free():
    s = _load("lovdal_folds.json")
    assert s["lockbox_ids"] == list(range(64, 74))
    dev = set(range(64))
    tested = [a for f in s["folds"].values() for a in f["test_ids"]]
    assert sorted(tested) == sorted(dev) and len(tested) == len(set(tested))
    for f in s["folds"].values():
        assert _roles_disjoint(f)
        used = set(f["train_ids"]) | set(f["val_ids"]) | set(f["test_ids"])
        assert not (set(s["lockbox_ids"]) & used)
    assert set(s["final"]["train_ids"]) | set(s["final"]["val_ids"]) == dev
    assert s["final"]["test_ids"] == s["lockbox_ids"]
    for view in s["balance"].values():
        counts = view["test_group_injuries"]
        mean = np.mean(counts)
        assert all(abs(c - mean) <= splits.BALANCE_TOLERANCE * mean for c in counts)
    assert s["meta"]["seed"] == 0


def test_lovdal_lockbox_is_the_papers_test_set():
    s = _load("lovdal_folds.json")
    assert s["balance"]["day"]["lockbox_injuries"] == 50
    assert s["balance"]["day"]["lockbox_events"] == 3044


@pytest.mark.parametrize("name", ["pamap2_folds.json", "wesad_folds.json"])
def test_sensor_splits_test_each_subject_once(name):
    s = _load(name)
    tested = [x for f in s["folds"].values() for x in f["test_ids"]]
    assert len(tested) == len(set(tested))
    for f in s["folds"].values():
        assert _roles_disjoint(f)
    if name == "pamap2_folds.json":
        assert "109" not in tested
        for f in s["folds"].values():
            assert "109" not in f["val_ids"] and "109" in f["train_ids"]


def test_sportdb_groups_cover_each_recording_once_and_keep_unnamed_together():
    s = _load("sportdb2_folds.json")
    recs = [r for f in s["folds"].values() for r in f["test_recordings"]]
    assert len(recs) == len(set(recs))
    unnamed = [k for f in s["folds"].values() for k in f["test_ids"] if "<no subject folder>" in k]
    assert len(unnamed) == len(set(unnamed))  # each unnamed group appears in exactly one fold


@pytest.mark.parametrize("name", ["soccermon_lopo.json", "pmdata_lopo.json"])
def test_leave_one_out_files(name):
    s = _load(name)
    tested = [f["test_ids"][0] for f in s["folds"].values()]
    assert len(tested) == len(set(tested))
    assert all(f["test_ids"][0] not in f["train_ids"] for f in s["folds"].values())
    if name == "pmdata_lopo.json":
        assert "p08" not in tested


def test_sirp_demo_split_rows_are_disjoint():
    s = _load("sirp600_demo_split.json")
    assert not (set(s["train_rows"]) & set(s["test_rows"]))
    assert len(s["train_rows"]) + len(s["test_rows"]) == 600
