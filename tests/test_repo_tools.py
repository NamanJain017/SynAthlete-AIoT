"""Tests for the repo tooling: the path guard hook and the snapshot filter."""

import importlib.util
from pathlib import Path

import tasks

ROOT = Path(__file__).resolve().parents[1]


def _load_guard():
    spec = importlib.util.spec_from_file_location(
        "guard_paths", ROOT / ".claude" / "hooks" / "guard_paths.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_blocks_protected_paths():
    guard = _load_guard()
    root = str(ROOT)
    for rel in [
        "data/raw/pamap2/subject101.dat",
        "data/splits/lovdal_folds.json",
        "experiments/baselines/run1/metrics.json",
        ".env",
    ]:
        assert guard.is_protected(str(ROOT / rel), root), rel


def test_guard_allows_normal_paths():
    guard = _load_guard()
    root = str(ROOT)
    for rel in ["src/data/audit.py", "docs/STATE.md", "experiments/baselines/run1/config.yaml"]:
        assert not guard.is_protected(str(ROOT / rel), root), rel


def test_snapshot_skips_raw_data_and_secrets():
    assert not tasks.snapshot_includes(Path("data/raw/pamap2/subject101.dat"))
    assert not tasks.snapshot_includes(Path(".env"))
    assert not tasks.snapshot_includes(Path("backups/snapshot.zip"))
    assert tasks.snapshot_includes(Path("src/data/audit.py"))
