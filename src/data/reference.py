"""Reference and demo datasets (piece 2): Sport DB 2.0 heart rate, SIRP-600 and ziya07 tables.

Sport DB 2.0 is only a realism reference for synthetic heart rate (D-012); SIRP-600 and ziya07 are
demo data and never evidence (D-012, D-013). Raw values only.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import manifest, readers
from src.data.parallel import parallel_map, pick_workers

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
# Plausible heart rate for the realism check (D-019). 0 bpm is the device's "no signal" value.
HR_VALID_BPM = (30.0, 220.0)


# ---------------------------------------------------------------- Sport DB 2.0


def sportdb_hr_rows(crd: Path, root: Path) -> pd.DataFrame:
    """Continuous heart-rate (and RR) series of one recording, long format; empty if it has none."""
    parts = crd.relative_to(root).parts
    has_subject = len(parts) >= 3
    rec = readers.read_sportdb_record(crd)
    sig = rec["signals"]
    hr_keys = [k for k in sig if re.search(r"(^|\.)HR(_|$)", k)]
    span = (max(e for _, _, e in rec["phases"]) - min(s for _, s, _ in rec["phases"])
            if rec["phases"] else None)
    frames = []
    for k in hr_keys:
        hr = np.asarray(sig[k], dtype=np.float64).ravel()
        rr_key = "RR" + k[2:]
        rr = np.asarray(sig[rr_key], dtype=np.float64).ravel() if rr_key in sig else None
        frames.append(pd.DataFrame({
            "recording": "/".join(parts),
            "sport": parts[0],
            "group_key": "/".join(parts[:-1]) if has_subject else f"{parts[0]}/<no subject folder>",
            "segment": k,
            "sample": np.arange(len(hr)),
            "hr_bpm": hr,
            "rr_ms": rr[: len(hr)] if rr is not None and len(rr) >= len(hr) else np.nan,
            # samples per second from the timed notes; only meaningful for one-segment recordings
            "implied_hz": len(hr) / span if span and len(hr_keys) == 1 else np.nan,
        }))
    return pd.concat(frames) if frames else pd.DataFrame()


def build_sportdb(workers: int | None = None) -> dict:
    mats = sorted((RAW / "sportdb2").rglob("Data.mat"))
    read = [m.parent / f for m in mats for f in ("Data.mat", "Dem.txt", "TrNote.txt")
            if (m.parent / f).exists()]  # the files read_sportdb_record opens
    source = manifest.verify(read, RAW / "sportdb2")
    root = Path(os.path.commonpath([m.parent for m in mats]))
    biggest = max(m.stat().st_size for m in mats)
    n = pick_workers(len(mats), 6 * biggest, workers)
    parts = parallel_map(partial(sportdb_hr_rows, root=root), [m.parent for m in mats], n,
                         label="Sport DB recordings")
    table = pd.concat([p for p in parts if len(p)], ignore_index=True)
    lo, hi = HR_VALID_BPM
    table["hr_valid"] = table["hr_bpm"].between(lo, hi)  # raw values kept; NaN -> False
    out = PROCESSED / "sportdb2"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "hr.parquet")
    per_rec = table.groupby("recording")
    hz = table.groupby("recording")["implied_hz"].first().dropna()
    meta = {
        "recordings_with_hr": int(per_rec.ngroups),
        "hr_samples": int(len(table)),
        "implied_hz_median": round(float(hz.median()), 3) if len(hz) else None,
        "implied_hz_note": "rates are not stored in the files: inferred from timed notes [VERIFY]",
        "hr_bpm_min_max": [float(table["hr_bpm"].min()), float(table["hr_bpm"].max())],
        "hr_nan": int(table["hr_bpm"].isna().sum()),
        "hr_zero_no_signal": int((table["hr_bpm"] == 0).sum()),
        "hr_valid_rule_bpm": list(HR_VALID_BPM),
        "hr_valid_percent": round(float(table["hr_valid"].mean() * 100), 2),
        "recordings_per_sport": {k: int(v) for k, v in
                                 table.groupby("sport")["recording"].nunique().items()},
        "source": source,
    }
    (out / "hr_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------- demo tables


def only_file(folder: Path, pattern: str) -> Path:
    """The dataset's single table file; more than one would make the choice ambiguous."""
    found = sorted(folder.rglob(pattern))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one {pattern} in {folder}, found {len(found)}")
    return found[0]


def build_demo(name: str) -> dict:
    folder = RAW / name
    path = only_file(folder, "*.xlsx" if name == "sirp600" else "*.csv")
    source = manifest.verify([path], folder)
    if name == "sirp600":
        df = readers.read_sirp_xlsx(path)
        label = "Injury_Risk"
    else:
        df = readers.read_ziya_csv(path)
        label = "Activity_Status"
    out = PROCESSED / name
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "table.parquet")
    meta = {"rows": int(len(df)), "columns": list(df.columns), "label": label,
            "label_counts": {str(k): int(v)
                             for k, v in df[label].value_counts().sort_index().items()},
            "status": "DEMO ONLY: unknown origin, never evidence (D-012)", "source": source}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for name in argv or ["sportdb2", "sirp600", "ziya07_wearable"]:
        t0 = time.time()
        meta = build_sportdb() if name == "sportdb2" else build_demo(name)
        shown = {k: v for k, v in meta.items() if k != "columns"}
        print(f"[{name}] {shown} ({time.time() - t0:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
