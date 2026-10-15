"""The lockbox guard (D-017): locked runners stay apart and open only at the evaluation freeze."""

import json

import numpy as np
import pytest

from src.data import loaders, lockbox

HEAD = "a" * 40


def _runners(ids: list[int]) -> loaders.RunnerData:
    athlete = np.repeat(np.array(ids, dtype=np.int16), 3)
    n = len(athlete)
    return loaders.RunnerData("day", np.zeros((n, 7, 2), np.float32), np.zeros(n, np.int8),
                              athlete, np.tile(np.arange(3, dtype=np.int32), len(ids)),
                              ["a", "b"], rest_day=np.zeros((n, 7), bool))


def _state(monkeypatch, tag: str, head: str = HEAD, dirty: bool = False) -> None:
    monkeypatch.setattr(lockbox, "freeze_state",
                        lambda: {"tag_commit": tag, "git_commit": head, "git_dirty": dirty})


def test_lockbox_refuses_before_the_freeze_tag(monkeypatch):
    _state(monkeypatch, tag="unavailable")
    with pytest.raises(lockbox.LockboxSealed, match="eval-freeze"):
        lockbox.open_lockbox("day", "final evaluation")


@pytest.mark.parametrize("head,dirty", [("b" * 40, False), (HEAD, True)])
def test_lockbox_refuses_away_from_the_frozen_code(monkeypatch, head, dirty):
    _state(monkeypatch, tag=HEAD, head=head, dirty=dirty)
    with pytest.raises(lockbox.LockboxSealed):
        lockbox.open_lockbox("day", "final evaluation")


def test_lockbox_opens_at_the_freeze_and_logs_it(monkeypatch, tmp_path):
    loaders.save_runner_data(_runners([64, 65]), tmp_path / "locked")
    monkeypatch.setattr(loaders, "LOCKBOX_DIR", tmp_path / "locked")
    monkeypatch.setattr(lockbox, "LOG", tmp_path / "access.log")
    _state(monkeypatch, tag=HEAD)
    data = lockbox.open_lockbox("day", "final evaluation")
    assert sorted(np.unique(data.athlete_id)) == [64, 65]
    entry = json.loads((tmp_path / "access.log").read_text(encoding="utf-8"))
    assert entry["commit"] == HEAD and entry["reason"] == "final evaluation"


def test_load_dev_refuses_a_file_with_locked_runners(monkeypatch, tmp_path):
    loaders.save_runner_data(_runners([1, 70]), tmp_path / "lovdal")
    monkeypatch.setattr(loaders, "PROCESSED", tmp_path)
    monkeypatch.setattr(loaders, "lovdal_lockbox_ids", lambda: list(range(64, 74)))
    with pytest.raises(RuntimeError, match="locked runners"):
        loaders.load_dev("day")


def test_saved_development_files_hold_no_locked_runner():
    folder = loaders.PROCESSED / "lovdal"
    if not (folder / "day_meta.json").exists():
        pytest.skip("processed Lövdal files are not on this machine (python tasks.py data)")
    locked = set(loaders.lovdal_lockbox_ids())
    for kind in ("day", "week"):
        meta = json.loads((folder / f"{kind}_meta.json").read_text(encoding="utf-8"))
        assert not locked & set(meta["athlete_ids"])
        assert not np.isin(loaders.load_dev(kind).athlete_id, list(locked)).any()
