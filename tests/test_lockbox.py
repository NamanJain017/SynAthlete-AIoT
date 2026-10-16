"""The lockbox guard (D-017): locked runners stay apart and open only at the evaluation freeze."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.data import loaders, lockbox

HEAD = "a" * 40
TAG_OBJECT = "t" * 40
SRC = Path(__file__).resolve().parents[1] / "src"


def _runners(ids: list[int]) -> loaders.RunnerData:
    athlete = np.repeat(np.array(ids, dtype=np.int16), 3)
    n = len(athlete)
    return loaders.RunnerData("day", np.zeros((n, 7, 2), np.float32), np.zeros(n, np.int8),
                              athlete, np.tile(np.arange(3, dtype=np.int32), len(ids)),
                              ["a", "b"], rest_day=np.zeros((n, 7), bool))


def _state(monkeypatch, **changes) -> None:
    state = {"tag_type": "tag", "tag_object": TAG_OBJECT, "tag_commit": HEAD,
             "remote_tag_object": TAG_OBJECT, "git_commit": HEAD, "git_dirty": False, **changes}
    monkeypatch.setattr(lockbox, "freeze_state", lambda: state)


@pytest.fixture
def locked(monkeypatch, tmp_path):
    """Fake locked arrays (runners 64, 65) of the split's data version, and a temporary log."""
    loaders.save_runner_data(_runners([64, 65]), tmp_path / "locked", source={"fingerprint": "v1"})
    monkeypatch.setattr(loaders, "LOCKBOX_DIR", tmp_path / "locked")
    monkeypatch.setattr(loaders, "lovdal_split", lambda: {
        "lockbox_ids": list(range(64, 74)), "meta": {"data_fingerprint": "v1"}})
    monkeypatch.setattr(lockbox, "LOG", tmp_path / "access.log")
    return tmp_path


@pytest.mark.parametrize("changes,message", [
    ({"tag_commit": "unavailable"}, "evaluation freeze"),
    ({"tag_type": "commit"}, "annotated"),
    ({"remote_tag_object": "unavailable"}, "push the tag"),
    ({"git_commit": "b" * 40}, "check out"),
    ({"git_dirty": True}, "check out"),
])
def test_lockbox_refuses_outside_a_proper_freeze(monkeypatch, locked, changes, message):
    _state(monkeypatch, **changes)
    with pytest.raises(lockbox.LockboxSealed, match=message):
        lockbox.open_lockbox("day", "final evaluation")
    assert not (locked / "access.log").exists()


def test_lockbox_opens_at_the_freeze_and_logs_it(monkeypatch, locked):
    _state(monkeypatch)
    data = lockbox.open_lockbox("day", "final evaluation")
    assert sorted(np.unique(data.athlete_id)) == [64, 65]
    entry = json.loads((locked / "access.log").read_text(encoding="utf-8"))
    assert entry["commit"] == HEAD and entry["reason"] == "final evaluation"
    lockbox.open_lockbox("day", "same frozen commit again")  # e.g. the week arm: allowed
    assert len((locked / "access.log").read_text(encoding="utf-8").splitlines()) == 2


def test_lockbox_opens_at_one_frozen_commit_only(monkeypatch, locked):
    _state(monkeypatch)
    lockbox.open_lockbox("day", "final evaluation")
    _state(monkeypatch, git_commit="c" * 40, tag_commit="c" * 40)  # a second, later freeze
    with pytest.raises(lockbox.LockboxSealed, match="already opened"):
        lockbox.open_lockbox("day", "try again")


def test_lockbox_refuses_another_data_version(monkeypatch, locked):
    monkeypatch.setattr(loaders, "lovdal_split", lambda: {
        "lockbox_ids": list(range(64, 74)), "meta": {"data_fingerprint": "v2"}})
    _state(monkeypatch)
    with pytest.raises(lockbox.LockboxSealed, match="data version"):
        lockbox.open_lockbox("day", "final evaluation")


@pytest.mark.parametrize("porcelain,dirty", [
    ("", False),
    ("?? experiments/baselines/run1/", False),  # new outputs are fine
    ("?? src/models/extra.py", True),           # untracked code is not frozen
    ("?? configs/new.yaml", True),
    ("?? numpy.py", True),                      # could shadow an import
    ("M src/data/loaders.py", True),            # any tracked change
    ("unavailable", True),                      # git failed: fail closed
])
def test_dirty_rule(porcelain, dirty):
    assert lockbox.is_dirty(porcelain) is dirty


def test_real_repository_lockbox_is_still_sealed():
    """No eval-freeze tag and no access log exist before the evaluation freeze (D-017)."""
    if lockbox.audit._git("rev-parse", "--git-dir") == "unavailable":
        pytest.skip("not a git checkout")
    assert lockbox.audit._git("tag", "-l", lockbox.FREEZE_TAG) == ""
    assert not lockbox.LOG.exists()


def test_no_module_reads_raw_or_locked_lovdal_data_around_the_guard():
    """Only the data modules may touch the whole raw Lövdal files or the locked arrays."""
    allowed = {"loaders.py", "splits.py", "lockbox.py", "audit.py", "readers.py"}
    pattern = re.compile(r"load_lovdal\b|LOCKBOX_DIR|lovdal_lockbox|find_runner_files|"
                         r"read_runner_csv|raw[\"'/\\ ,)]+[\"']?lovdal", re.IGNORECASE)
    offenders = [p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py")
                 if p.name not in allowed and pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == [], f"use loaders.load_dev() instead: {offenders}"


def _fake_split(monkeypatch, tmp_path, ids: list[int], source_fp: str) -> None:
    loaders.save_runner_data(_runners(ids), tmp_path / "lovdal", source={"fingerprint": source_fp})
    monkeypatch.setattr(loaders, "PROCESSED", tmp_path)
    monkeypatch.setattr(loaders, "lovdal_split", lambda: {
        "lockbox_ids": list(range(64, 74)), "meta": {"data_fingerprint": "split-version"}})


def test_load_dev_refuses_a_file_with_locked_runners(monkeypatch, tmp_path):
    _fake_split(monkeypatch, tmp_path, [1, 70], "split-version")
    with pytest.raises(RuntimeError, match="locked runners"):
        loaders.load_dev("day")


def test_load_dev_refuses_another_data_version(monkeypatch, tmp_path):
    _fake_split(monkeypatch, tmp_path, [1, 2], "other-version")
    with pytest.raises(RuntimeError, match="data version"):
        loaders.load_dev("day")


def test_load_dev_returns_matching_development_data(monkeypatch, tmp_path):
    _fake_split(monkeypatch, tmp_path, [1, 2], "split-version")
    assert sorted(np.unique(loaders.load_dev("day").athlete_id)) == [1, 2]


def test_saved_development_files_hold_no_locked_runner():
    folder = loaders.PROCESSED / "lovdal"
    if not (folder / "day_meta.json").exists():
        pytest.skip("processed Lövdal files are not on this machine (python tasks.py data)")
    locked = set(loaders.lovdal_lockbox_ids())
    for kind in ("day", "week"):
        meta = json.loads((folder / f"{kind}_meta.json").read_text(encoding="utf-8"))
        assert not locked & set(meta["athlete_ids"])
        assert not np.isin(loaders.load_dev(kind).athlete_id, list(locked)).any()
