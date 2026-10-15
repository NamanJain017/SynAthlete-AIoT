"""File hashes and dataset fingerprints, shared by the audit, the loaders and the split builder."""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_files(paths: list[Path]) -> dict[Path, str]:
    """Hash many files at once; threads suffice because hashlib and file reads release the GIL."""
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        return dict(zip(paths, pool.map(sha256_file, paths), strict=True))


def fingerprint(hashes: dict[str, str]) -> str:
    """One hash for a whole dataset version: the hash of its sorted (file, sha256) list."""
    lines = "\n".join(f"{name}:{sha}" for name, sha in sorted(hashes.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()
