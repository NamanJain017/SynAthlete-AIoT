"""Check that the raw files a builder reads are the files the audit hashed (data/MANIFEST.csv).

Every builder passes the exact paths it reads to verify(). A file whose sha256 differs from the
MANIFEST stops the run; the returned record (files read + fingerprint) is saved next to the
outputs, so each processed file names its data version (AGENTS.md rule 11).
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.data import hashing

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "MANIFEST.csv"


class ManifestMismatch(RuntimeError):
    """A raw file changed after the audit: re-run the audit and ask before using it."""


def recorded(path: Path = MANIFEST) -> dict[str, str]:
    """MANIFEST as {path under data/raw: sha256}; empty if there is no MANIFEST yet."""
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["file"]: row["sha256"] for row in csv.DictReader(f)}


def verify(paths: list[Path], dataset_folder: Path, raw: Path = RAW,
           manifest: Path = MANIFEST) -> dict:
    """Hash the files a builder read and compare them with the MANIFEST.

    Files the audit never saw are listed (not an error: re-run the audit to add them).
    The fingerprint uses paths inside the dataset folder, like the audit's fingerprint.
    """
    known = recorded(manifest)
    hashes = hashing.sha256_files(sorted(set(paths)))
    changed, unlisted = [], []
    for path, sha in hashes.items():
        rel = path.relative_to(raw).as_posix()
        if rel not in known:
            unlisted.append(rel)
        elif known[rel] != sha:
            changed.append(rel)
    if changed:
        raise ManifestMismatch(f"{len(changed)} raw file(s) differ from data/MANIFEST.csv, "
                               f"e.g. {changed[0]}: re-run the audit and ask before using them")
    if unlisted:
        print(f"    note: {len(unlisted)} file(s) not in data/MANIFEST.csv yet (run the audit)")
    inside = {p.relative_to(dataset_folder).as_posix(): sha for p, sha in hashes.items()}
    return {"files_read": len(hashes), "fingerprint": hashing.fingerprint(inside),
            "not_in_manifest": sorted(unlisted)}
