"""The raw-file check every builder runs before reading data (D-019, AGENTS.md rule 11)."""

import csv

import pytest

from src.data import hashing, manifest


def _setup(tmp_path, content: bytes = b"a,b\n1,2\n"):
    raw = tmp_path / "raw"
    (raw / "ds").mkdir(parents=True)
    f = raw / "ds" / "x.csv"
    f.write_bytes(content)
    man = tmp_path / "MANIFEST.csv"
    with open(man, "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=["dataset", "file", "sha256"])
        w.writeheader()
        w.writerow({"dataset": "ds", "file": "ds/x.csv", "sha256": hashing.sha256_file(f)})
    return raw, f, man


def test_matching_files_give_the_audit_style_fingerprint(tmp_path):
    raw, f, man = _setup(tmp_path)
    rec = manifest.verify([f], raw / "ds", raw=raw, manifest=man)
    assert rec == {"files_read": 1, "not_in_manifest": [],
                   "fingerprint": hashing.fingerprint({"x.csv": hashing.sha256_file(f)})}


def test_a_changed_file_stops_the_run(tmp_path):
    raw, f, man = _setup(tmp_path)
    f.write_bytes(b"a,b\n1,3\n")
    with pytest.raises(manifest.ManifestMismatch, match="differ"):
        manifest.verify([f], raw / "ds", raw=raw, manifest=man)


def test_a_never_audited_file_stops_the_run(tmp_path):
    raw, f, man = _setup(tmp_path)
    extra = raw / "ds" / "new.csv"
    extra.write_bytes(b"z\n")
    with pytest.raises(manifest.ManifestMismatch, match="never audited"):
        manifest.verify([f, extra], raw / "ds", raw=raw, manifest=man)


def test_a_missing_manifest_stops_the_run(tmp_path):
    raw, f, _ = _setup(tmp_path)
    with pytest.raises(manifest.ManifestMismatch, match="missing"):
        manifest.verify([f], raw / "ds", raw=raw, manifest=tmp_path / "none.csv")
