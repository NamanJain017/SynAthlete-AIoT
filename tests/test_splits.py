"""Leakage tests for the person-level splits (piece 2, D-017, D-020).

Unit tests use fake data. The checks on data/splits/ FAIL (never skip) when a file is missing:
the split files are committed, so every clone must have them. Completeness is checked against
data/MANIFEST.csv, which is committed too, so these tests need no raw data.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.data import audit, loaders, splits

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
MANIFEST = ROOT / "data" / "MANIFEST.csv"
# sha256 of data/splits/SHA256SUMS as sealed on 2026-10-15 (commit b34af23): changing a split
# file AND its pinned hash still fails here.
SEAL_DIGEST = "97a603e4eda39f5f9e8aac81a98c43f7c478a8d4942b5e884aa6b3c396fc71f4"


def _load(name: str) -> dict:
    path = SPLITS / name
    assert path.exists(), f"{name} is missing: split files are committed, never regenerated"
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(dataset: str) -> dict[str, str]:
    """file path inside the dataset folder -> sha256, from the committed MANIFEST."""
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return {r["file"].split("/", 1)[1]: r["sha256"]
                for r in csv.DictReader(f) if r["dataset"] == dataset}


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


def test_write_all_writes_everything_or_nothing(tmp_path):
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    with pytest.raises(splits.SplitError):
        splits.write_all({"a.json": {"v": 1}, "b.json": {"v": 2}}, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["b.json"]  # a.json not written
    splits.write_all({"c.json": {"v": 3}, "d.json": {"v": 4}}, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["b.json", "c.json", "d.json"]


def test_balance_rule_stops_on_unbalanced_groups():
    splits.check_balance([100, 105, 95, 110, 90], "ok")
    with pytest.raises(splits.BalanceError):
        splits.check_balance([100, 100, 100, 100, 40], "bad")


def test_rotating_folds_never_reuse_a_validation_person():
    people = [f"S{i}" for i in range(15)]
    folds = splits.rotating_folds(people, 5, 2)
    vals = [p for f in folds.values() for p in f["val_ids"]]
    assert len(vals) == len(set(vals)) == 10
    tests = [p for f in folds.values() for p in f["test_ids"]]
    assert sorted(tests) == sorted(people)
    for f in folds.values():
        assert _roles_disjoint(f)
        assert set(f["train_ids"]) | set(f["val_ids"]) | set(f["test_ids"]) == set(people)


def test_remake_refuses_when_sealed_unlogged_or_not_allowed(tmp_path, monkeypatch):
    decisions = tmp_path / "DECISIONS.md"
    decisions.write_text("- D-020 | 2026-10-15 | ok\n", encoding="utf-8")
    monkeypatch.setattr(splits, "DECISIONS", decisions)
    monkeypatch.setattr(splits, "SEAL", tmp_path / "missing")
    with pytest.raises(splits.SplitError, match="not logged"):
        splits.remake(["wesad_folds.json"], "D-999")
    with pytest.raises(splits.SplitError, match="may not be re-made"):
        splits.remake(["lovdal_folds.json"], "D-020")
    seal = tmp_path / "SHA256SUMS"
    seal.write_text("", encoding="utf-8")
    monkeypatch.setattr(splits, "SEAL", seal)
    with pytest.raises(splits.SplitError, match="sealed"):
        splits.remake(["wesad_folds.json"], "D-020")


def _remake_sandbox(tmp_path, monkeypatch, old_folds: dict) -> None:
    """A temporary data/splits with one old wesad file and a logged decision."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "wesad_folds.json").write_text(
        json.dumps({"meta": {"created_utc": "t0", "git_commit": "c0"}, "folds": old_folds}),
        encoding="utf-8")
    decisions = tmp_path / "DECISIONS.md"
    decisions.write_text("- D-020 | 2026-10-15 | ok\n", encoding="utf-8")
    for name, value in {"SPLITS": splits_dir, "SUPERSEDED": splits_dir / "superseded",
                        "SEAL": splits_dir / "SHA256SUMS", "DECISIONS": decisions,
                        "ROOT": tmp_path}.items():
        monkeypatch.setattr(splits, name, value)
    new = {"0": {"train_ids": ["c"], "val_ids": ["b"], "test_ids": ["a"]}}
    monkeypatch.setattr(splits, "wesad_split", lambda: {"meta": {}, "folds": new})


def test_remake_refuses_changed_test_groups(tmp_path, monkeypatch):
    _remake_sandbox(tmp_path, monkeypatch,
                    {"0": {"train_ids": ["a"], "val_ids": ["b"], "test_ids": ["c"]}})
    with pytest.raises(splits.SplitError, match="test groups would change"):
        splits.remake(["wesad_folds.json"], "D-020")
    assert not (tmp_path / "splits" / "superseded").exists()  # nothing moved


def test_remake_keeps_the_old_file_and_runs_only_once(tmp_path, monkeypatch):
    _remake_sandbox(tmp_path, monkeypatch,
                    {"0": {"train_ids": ["b"], "val_ids": ["c"], "test_ids": ["a"]}})
    splits.remake(["wesad_folds.json", "wesad_folds.json"], "D-020")  # repeat is collapsed
    kept = tmp_path / "splits" / "superseded" / "wesad_folds.v1.json"
    assert json.loads(kept.read_text(encoding="utf-8"))["meta"]["git_commit"] == "c0"
    new = json.loads((tmp_path / "splits" / "wesad_folds.json").read_text(encoding="utf-8"))
    assert new["meta"]["decision"] == "D-017, D-020"
    assert new["meta"]["supersedes"]["file"] == "splits/superseded/wesad_folds.v1.json"
    with pytest.raises(splits.SplitError, match="already re-made"):
        splits.remake(["wesad_folds.json"], "D-020")


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


# ---------------------------------------------------------------- the saved split files


def test_split_files_match_their_pinned_hashes():
    seal = SPLITS / "SHA256SUMS"
    assert seal.exists(), "data/splits is not sealed (python tasks.py splits --seal)"
    pinned = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ", 1)
        pinned[name] = sha
    on_disk = {p.relative_to(SPLITS).as_posix() for p in SPLITS.rglob("*.json")}
    assert on_disk == set(pinned), "a split file is not pinned, or a pinned file is missing"
    assert set(splits.FILES) <= set(pinned)
    for name, sha in pinned.items():
        digest = hashlib.sha256((SPLITS / name).read_bytes()).hexdigest()
        assert digest == sha, f"{name} changed after sealing"
    assert hashlib.sha256(seal.read_bytes()).hexdigest() == SEAL_DIGEST


@pytest.mark.parametrize("name", splits.FILES)
def test_every_split_file_records_clean_code_and_seed(name):
    m = _load(name)["meta"]
    assert m["seed"] == splits.SEED
    assert len(m["git_commit"]) == 40 and m["git_dirty"] is False
    assert m["data_fingerprint"]


@pytest.mark.parametrize("name,dataset", [
    ("lovdal_folds.json", "lovdal"), ("pamap2_folds.json", "pamap2"),
    ("wesad_folds.json", "wesad"), ("sportdb2_folds.json", "sportdb2"),
    ("sirp600_demo_split.json", "sirp600"),
])
def test_fingerprint_matches_the_audited_data(name, dataset):
    assert _load(name)["meta"]["data_fingerprint"] == audit.fingerprint(_manifest(dataset))


def test_exploratory_fingerprints_match_the_audited_files():
    soccer = _manifest("soccermon")
    picked = ("subjective/training-load/daily_load.csv", "subjective/injury/injury.csv")
    assert all(f in soccer for f in picked), "SoccerMon files missing from MANIFEST"
    assert _load("soccermon_lopo.json")["meta"]["data_fingerprint"] == \
        audit.fingerprint({f: soccer[f] for f in picked})
    pm = _manifest("pmdata")
    s = _load("pmdata_lopo.json")
    injury = {f"{p}/pmsys/injury.csv" for f in s["folds"].values() for p in f["test_ids"]}
    assert injury <= set(pm), "PMData injury files missing from MANIFEST"
    assert s["meta"]["data_fingerprint"] == audit.fingerprint({f: pm[f] for f in injury})


def test_lovdal_saved_split_is_leak_free():
    s = _load("lovdal_folds.json")
    lockbox = set(s["lockbox_ids"])
    assert s["lockbox_ids"] == list(range(64, 74))
    dev = set(range(64))
    tested = [a for f in s["folds"].values() for a in f["test_ids"]]
    assert sorted(tested) == sorted(dev) and len(tested) == len(set(tested))
    for f in s["folds"].values():
        assert _roles_disjoint(f) and f["val_ids"] and f["train_ids"]
        assert set(f["train_ids"]) | set(f["val_ids"]) | set(f["test_ids"]) == dev
    final = s["final"]
    assert _roles_disjoint(final)
    assert set(final["train_ids"]) | set(final["val_ids"]) == dev
    assert final["val_ids"] == s["folds"]["0"]["test_ids"]
    assert set(final["test_ids"]) == lockbox
    paper = s["paper_split"]
    assert set(paper["train_ids"]) == dev and set(paper["test_ids"]) == lockbox
    for view in s["balance"].values():
        counts = view["test_group_injuries"]
        mean = np.mean(counts)
        assert all(abs(c - mean) <= splits.BALANCE_TOLERANCE * mean for c in counts)


def test_lovdal_validation_overlap_is_as_disclosed():
    """Lövdal validation groups are drawn per fold, so some runners validate in several folds
    (disclosed in docs/VIVA_NOTES.md): 46 slots, 30 runners, 20/21/33/37 three times each."""
    vals = [a for f in _load("lovdal_folds.json")["folds"].values() for a in f["val_ids"]]
    counts = {a: vals.count(a) for a in set(vals)}
    assert len(vals) == 46 and len(counts) == 30
    assert sorted(a for a, n in counts.items() if n == 3) == [20, 21, 33, 37]
    assert max(counts.values()) == 3


def test_lovdal_lockbox_is_the_papers_test_set():
    s = _load("lovdal_folds.json")
    assert s["balance"]["day"]["lockbox_injuries"] == 50
    assert s["balance"]["day"]["lockbox_events"] == 3044


@pytest.mark.parametrize("name,dataset", [("pamap2_folds.json", "pamap2"),
                                          ("wesad_folds.json", "wesad")])
def test_sensor_splits_are_complete_and_rotate_validation(name, dataset):
    s = _load(name)
    everyone = set(s["meta"]["subjects"])
    audited = {Path(f).stem.removeprefix("subject") for f in _manifest(dataset)
               if f.endswith((".dat", ".pkl"))}
    assert everyone == audited, "subject list differs from the audited files"
    tested = [x for f in s["folds"].values() for x in f["test_ids"]]
    vals = [x for f in s["folds"].values() for x in f["val_ids"]]
    assert len(tested) == len(set(tested)) and len(vals) == len(set(vals))  # D-020
    for f in s["folds"].values():
        assert _roles_disjoint(f) and f["val_ids"]
        assert set(f["train_ids"]) | set(f["val_ids"]) | set(f["test_ids"]) == everyone
    if dataset == "pamap2":
        assert set(tested) == everyone - {"109"}
        assert all("109" in f["train_ids"] for f in s["folds"].values())
    else:
        assert set(tested) == everyone


def test_sportdb_groups_cover_every_audited_recording_once():
    s = _load("sportdb2_folds.json")
    recs = [r for f in s["folds"].values() for r in f["test_recordings"]]
    audited = {f.rsplit("/", 1)[0] for f in _manifest("sportdb2") if f.endswith("Data.mat")}
    assert len(recs) == len(set(recs)) == len(audited)
    keys = [k for f in s["folds"].values() for k in f["test_ids"]]
    assert len(keys) == len(set(keys))  # a subject (or the unnamed group) is in one fold only


@pytest.mark.parametrize("name,dataset", [("soccermon_lopo.json", "soccermon"),
                                          ("pmdata_lopo.json", "pmdata")])
def test_leave_one_out_files_are_complete(name, dataset):
    s = _load(name)
    tested = [f["test_ids"][0] for f in s["folds"].values()]
    assert len(tested) == len(set(tested))
    everyone = set(tested)
    for f in s["folds"].values():
        assert f["test_ids"][0] not in f["train_ids"]
        assert set(f["train_ids"]) | set(f["test_ids"]) == everyone
    if dataset == "soccermon":
        assert len(tested) == s["meta"]["players"]
    else:
        people = {f.split("/", 1)[0] for f in _manifest("pmdata")}
        assert everyone | set(s["meta"]["excluded_from_injury_tasks"]) == people
        assert "p08" not in everyone


def test_sirp_demo_split_rows_are_disjoint():
    s = _load("sirp600_demo_split.json")
    assert not (set(s["train_rows"]) & set(s["test_rows"]))
    assert len(s["train_rows"]) + len(s["test_rows"]) == 600
