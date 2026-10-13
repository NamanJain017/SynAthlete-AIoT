"""Dataset audit (piece 1): facts about every raw dataset, computed from the files themselves.

    python -m src.data.audit [--only NAME]        (same as: python tasks.py audit [--only NAME])

Reads data/raw/ (never writes there) and configs/datasets.yaml. Writes:
- experiments/audit/<run_id>/: audit.json, audit_table.md, config.yaml, run_info.json
- data/MANIFEST.csv: sha256 of every raw file the audit read
- docs/DATASETS.md: the audit table between the AUDIT markers (full runs only)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import date, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.data import readers

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CONFIG = ROOT / "configs" / "datasets.yaml"
MANIFEST = ROOT / "data" / "MANIFEST.csv"
DATASETS_MD = ROOT / "docs" / "DATASETS.md"
RUNS = ROOT / "experiments" / "audit"
SEED = 0
MANIFEST_FIELDS = ["dataset", "file", "source_url", "licence", "access_date", "sha256"]
TABLE_START, TABLE_END = "<!-- AUDIT:START -->", "<!-- AUDIT:END -->"
TABLE_HEADER = (
    "| # | Real / simulated / unknown | Subjects | Rows or hours | Channels / features | Labels "
    "| Injury labels? | Time series? | Licence | Role decided |\n"
    "|---|---|---|---|---|---|---|---|---|---|"
)
TABLE_KEYS = ("subjects", "size", "channels", "labels", "injury", "time_series")


# ---------------------------------------------------------------- small helpers


def new_result() -> dict:
    """Same keys for every dataset, so the outputs can be compared side by side."""
    return {
        "status": "ok", "subjects": {}, "rows": None, "hours": None, "channels": {},
        "labels": {}, "injury": {"has_labels": False}, "time_series": {}, "missingness": {},
        "duplicates": None, "plausibility": {}, "details": {}, "notes": [], "table": {},
        "_files": [],
    }


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fingerprint(hashes: dict[str, str]) -> str:
    """One hash for a whole dataset version: the hash of its sorted (file, sha256) list."""
    lines = "\n".join(f"{name}:{sha}" for name, sha in sorted(hashes.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def r(x: float | None, digits: int = 3) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), digits)


def value_counts(s: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


def nan_percent(df: pd.DataFrame) -> dict[str, float]:
    pct = df.isna().mean() * 100
    return {c: r(v, 2) for c, v in pct.items() if v > 0}


def uniformity(values) -> dict:
    """KS test against an even spread over the observed range. High p = looks uniform."""
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(x) < 3 or x.max() == x.min():
        return {"n": int(len(x)), "note": "too few values or constant"}
    res = stats.kstest((x - x.min()) / (x.max() - x.min()), "uniform")
    return {"n": int(len(x)), "ks_D": r(res.statistic), "p": float(f"{res.pvalue:.3g}")}


def correlation(a, b) -> dict:
    df = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"),
                       "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(df) < 3:
        return {"n": int(len(df)), "note": "too few pairs"}
    res = stats.pearsonr(df["a"], df["b"])
    return {"n": int(len(df)), "r": r(res.statistic), "p": float(f"{res.pvalue:.3g}")}


def mean_contrast(high: np.ndarray, low: np.ndarray) -> dict:
    """Mean of two groups, their difference and a Welch t-test p-value."""
    high, low = np.asarray(high, float), np.asarray(low, float)
    if len(high) < 2 or len(low) < 2:
        return {"n_high": int(len(high)), "n_low": int(len(low)), "note": "too few values"}
    p = stats.ttest_ind(high, low, equal_var=False).pvalue
    return {
        "n_high": int(len(high)), "n_low": int(len(low)), "mean_high": r(high.mean(), 1),
        "mean_low": r(low.mean(), 1), "difference": r(high.mean() - low.mean(), 1),
        "p": float(f"{p:.3g}"),
    }


def label_rule_probe(X: pd.DataFrame, y: pd.Series) -> dict:
    """Origin probe, never a reported result: can a tree with at most 3 questions predict y?

    Real injury outcomes are noisy, so a tiny tree should not get close to 100 %. A label made
    from a formula of the features often can. Row-level CV, because the file has no athlete IDs.
    """
    tree = DecisionTreeClassifier(max_depth=2, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc = cross_val_score(tree, X, y, cv=cv, scoring="accuracy")
    tree.fit(X, y)
    used = [X.columns[i] for i in tree.tree_.feature if i >= 0]
    return {
        "model": "decision tree, max_depth=2, 5-fold stratified CV, seed 0",
        "cv_accuracy_mean": r(acc.mean()),
        "cv_accuracy_folds": [r(a) for a in acc],
        "majority_class_rate": r(y.value_counts(normalize=True).max()),
        "features_used": list(dict.fromkeys(used)),
    }


# ---------------------------------------------------------------- runner logs (#1, #2)


def runner_facts(df: pd.DataFrame) -> dict:
    features = [c for c in df.columns if c not in readers.RUNNER_ID_COLS]
    relative = [c for c in features if c.startswith("rel ")]
    stepped = [c for c in features if c not in relative]
    # pandas names repeated headers "x", "x.1", "x.2": the suffix is the day (or week) index.
    steps = Counter(int(m.group(1)) if (m := re.search(r"\.(\d+)$", c)) else 0 for c in stepped)
    negatives: Counter = Counter()
    for c in features:
        n = int((df[c] < 0).sum())
        if n:
            negatives[re.sub(r"\.\d+$", "", c)] += n
    values = df[features].to_numpy(float).ravel()
    neg_values = Counter(np.round(values[values < 0], 4).tolist()).most_common(5)
    per_athlete = df.groupby("Athlete ID").size()
    positive = df["injury"] == 1
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "athletes": int(df["Athlete ID"].nunique()),
        "day_index_range": [int(df["Date"].min()), int(df["Date"].max())],
        "injury_counts": value_counts(df["injury"]),
        "injury_rate": r(positive.mean(), 4),
        "injured_athletes": int(df.loc[positive, "Athlete ID"].nunique()),
        "rows_per_athlete": {
            "min": int(per_athlete.min()), "median": r(per_athlete.median(), 1),
            "max": int(per_athlete.max()),
        },
        "steps": len(steps),
        "features_per_step": sorted(set(steps.values())),
        "relative_features": len(relative),
        "negative_values_by_feature": dict(negatives),
        "most_common_negative_values": [[v, n] for v, n in neg_values],
        "nan_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_athlete_day": int(df.duplicated(["Athlete ID", "Date"]).sum()),
    }


def audit_runners(folder: Path) -> dict:
    files = readers.find_runner_files(folder)
    if not files:
        found = [p.name for p in folder.rglob("*") if p.is_file()]
        return {"status": f"no day/week approach files found (files: {found})", "table": {}}
    res = new_result()
    res["_files"] = list(files.values())
    facts = {kind: runner_facts(readers.read_runner_csv(p)) for kind, p in files.items()}
    res["details"] = facts
    first = next(iter(facts.values()))
    res["subjects"] = {"count": first["athletes"], "id": "Athlete ID (masked)"}
    res["rows"] = {kind: f["rows"] for kind, f in facts.items()}
    res["labels"] = {kind: f["injury_counts"] for kind, f in facts.items()}
    res["injury"] = {
        "has_labels": True,
        **{kind: {"positive_rows": f["injury_counts"].get("1", 0), "rate": f["injury_rate"],
                  "injured_athletes": f["injured_athletes"]} for kind, f in facts.items()},
    }
    res["time_series"] = {"value": True,
                          "detail": "one row per athlete-day, day index (masked dates)"}
    res["missingness"] = {kind: {"nan_cells": f["nan_cells"],
                                 "negative_values_by_feature": f["negative_values_by_feature"]}
                          for kind, f in facts.items()}
    res["duplicates"] = {kind: f["duplicate_rows"] for kind, f in facts.items()}
    athletes = sorted({f["athletes"] for f in facts.values()})
    lo = min(f["day_index_range"][0] for f in facts.values())
    hi = max(f["day_index_range"][1] for f in facts.values())

    def shape(f: dict) -> str:
        per_step = "/".join(str(n) for n in f["features_per_step"])
        rel = f" + {f['relative_features']} relative" if f["relative_features"] else ""
        return f"{f['steps']} steps x {per_step} features{rel}"

    res["table"] = {
        "subjects": " / ".join(str(a) for a in athletes) + " athletes",
        "size": "; ".join(f"{k}: {f['rows']:,} windows" for k, f in facts.items()),
        "channels": "; ".join(f"{k}: {shape(f)}" for k, f in facts.items()),
        "labels": "injury 0/1",
        "injury": "yes; " + "; ".join(
            f"{k}: {f['injury_counts'].get('1', 0):,} positive ({f['injury_rate']:.2%}), "
            f"{f['injured_athletes']} athletes injured" for k, f in facts.items()),
        "time_series": f"daily training logs, day index {lo}-{hi} (dates masked)",
    }
    return res


def compare_tables(
    a: pd.DataFrame, b: pd.DataFrame, keys: tuple[str, ...] = (), tol: float = 1e-9
) -> dict:
    """Cell-by-cell comparison after putting both tables in the same row order.

    Rows are ordered by the key columns first, so one changed value counts as one differing
    cell instead of shifting the row order of everything after it.
    """
    out: dict = {"shape_a": list(a.shape), "shape_b": list(b.shape),
                 "same_columns": list(a.columns) == list(b.columns)}
    if a.shape != b.shape or not out["same_columns"]:
        out.update(
            identical=False, differing_cells=None,
            columns_only_in_a=[c for c in a.columns if c not in b.columns],
            columns_only_in_b=[c for c in b.columns if c not in a.columns],
        )
        return out
    cols = list(a.columns)
    order = [k for k in keys if k in cols] + [c for c in cols if c not in keys]
    a2 = a.sort_values(order, kind="mergesort").reset_index(drop=True)
    b2 = b.sort_values(order, kind="mergesort").reset_index(drop=True)
    differing = 0
    for c in cols:
        x, y = a2[c], b2[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            same = np.isclose(x.to_numpy(float), y.to_numpy(float), rtol=0, atol=tol)
            same |= x.isna().to_numpy() & y.isna().to_numpy()
        else:
            same = (x.astype(str) == y.astype(str)).to_numpy()
        differing += int((~same).sum())
    out.update(identical=differing == 0, differing_cells=differing)
    return out


def mirror_check(original: Path, mirror: Path) -> dict:
    """Is the mirror (e.g. Kaggle #2) the same data as the original (Lövdal #1)?"""
    a_files, b_files = readers.find_runner_files(original), readers.find_runner_files(mirror)
    out = {}
    for kind in readers.RUNNER_KINDS:
        if kind not in a_files or kind not in b_files:
            out[kind] = {"identical": None, "note": "file missing on one side"}
            continue
        if sha256_file(a_files[kind]) == sha256_file(b_files[kind]):
            out[kind] = {"identical": True, "same_sha256": True}
        else:
            a = readers.read_runner_csv(a_files[kind])
            b = readers.read_runner_csv(b_files[kind])
            out[kind] = {"same_sha256": False,
                         **compare_tables(a, b, keys=("Athlete ID", "Date"))}
    return out


# ---------------------------------------------------------------- PAMAP2 (#3)


def audit_pamap2(folder: Path) -> dict:
    files = sorted(folder.rglob("subject*.dat"))
    if not files:
        return {"status": "no subject*.dat files found", "table": {}}
    res = new_result()
    res["_files"] = files
    sensor_cols = [c for c in readers.PAMAP2_COLUMNS
                   if c not in ("timestamp", "activity_id") and "orient" not in c]
    per_file, act_seconds = [], Counter()
    hr_by_act: dict[int, list[np.ndarray]] = {1: [], 5: []}  # lying, running (readme II.2)
    nan_totals, total_rows = pd.Series(0.0, index=readers.PAMAP2_COLUMNS), 0
    for path in files:
        df = readers.read_pamap2_dat(path)
        step = float(np.median(np.diff(df["timestamp"].to_numpy())))
        covered_s = len(df) * step
        hr_valid = int(df["heart_rate"].notna().sum())
        for act, n in df["activity_id"].value_counts().items():
            act_seconds[int(act)] += n * step
        for act, chunks in hr_by_act.items():
            chunks.append(df.loc[df["activity_id"] == act, "heart_rate"].dropna().to_numpy())
        nan_totals += df.isna().sum()
        total_rows += len(df)
        per_file.append({
            "file": f"{path.parent.name}/{path.name}",
            "subject": path.stem.removeprefix("subject"),
            "session": path.parent.name,
            "rows": int(len(df)),
            "median_step_s": r(step, 4),
            "hours": r(covered_s / 3600),
            "timestamp_span_hours": r((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 3600),
            "labelled_hours": r((df["activity_id"] != 0).sum() * step / 3600),
            "hr_samples_per_s": r(hr_valid / covered_s, 2),
            "imu_missing_pct": {p: r(df[f"{p}_acc16_x"].isna().mean() * 100, 2)
                                for p in readers.PAMAP2_IMU_POSITIONS},
            "activities": sorted(int(a) for a in df["activity_id"].unique() if a != 0),
            "duplicate_rows": int(df.duplicated().sum()),
        })
        del df
    sessions: dict[str, list[dict]] = {}
    for f in per_file:
        sessions.setdefault(f["session"], []).append(f)
    subjects = {s: sorted(f["subject"] for f in fs) for s, fs in sessions.items()}
    all_subjects = sorted({f["subject"] for f in per_file})
    hours = {s: r(sum(f["hours"] for f in fs), 2) for s, fs in sessions.items()}
    labelled = r(sum(f["labelled_hours"] for f in per_file), 2)
    hr_rates = [f["hr_samples_per_s"] for f in per_file]
    steps = sorted({f["median_step_s"] for f in per_file})
    activities = {readers.PAMAP2_ACTIVITIES.get(a, str(a)): r(s / 60, 1)
                  for a, s in sorted(act_seconds.items())}
    n_acts = sum(1 for a in act_seconds if a != 0)

    res["subjects"] = {"count": len(all_subjects), "ids": all_subjects, "by_session": subjects}
    res["rows"] = int(total_rows)
    res["hours"] = {"by_session": hours, "labelled_total": labelled}
    res["channels"] = {"count": len(sensor_cols), "names": sensor_cols,
                       "note": "orientation columns excluded (readme: invalid)"}
    res["labels"] = {"activity_minutes": activities, "n_activities": n_acts}
    res["time_series"] = {"value": True, "median_step_s": steps,
                          "hr_samples_per_s": {"min": min(hr_rates), "max": max(hr_rates),
                                               "median": r(float(np.median(hr_rates)), 2)}}
    res["missingness"] = {c: r(v / total_rows * 100, 2) for c, v in nan_totals.items() if v > 0}
    res["duplicates"] = sum(f["duplicate_rows"] for f in per_file)
    res["details"] = {"files": per_file}
    res["plausibility"] = {
        "hr_running_vs_lying": mean_contrast(np.concatenate(hr_by_act[5]),
                                             np.concatenate(hr_by_act[1])),
    }
    opt = subjects.get("Optional", [])
    res["table"] = {
        "subjects": f"{len(all_subjects)} ({len(opt)} also in Optional)",
        "size": "; ".join(f"{s}: {h} h" for s, h in hours.items()) + f" ({labelled} h labelled)",
        "channels": f"HR + 3 IMUs: {len(sensor_cols)} sensor channels",
        "labels": f"{n_acts} activities (+ id 0 transient)",
        "injury": "no",
        "time_series": f"yes: one row per {steps[0]} s; HR median "
                       f"{res['time_series']['hr_samples_per_s']['median']} samples/s",
    }
    return res


# ---------------------------------------------------------------- WESAD (#4)


def _width(a: np.ndarray) -> int:
    return int(a.shape[1]) if a.ndim == 2 else 1


def audit_wesad(folder: Path) -> dict:
    pkls = sorted(folder.rglob("S*.pkl"), key=lambda p: int(p.stem[1:]))
    if not pkls:
        return {"status": "no S*.pkl files found", "table": {}}
    res = new_result()
    per_subject, label_seconds, demographics = [], Counter(), []
    channels: dict[str, int] = {}
    for pkl in pkls:
        print(f"    {pkl.stem} ...", flush=True)
        readme = pkl.parent / f"{pkl.stem}_readme.txt"
        res["_files"] += [pkl] + ([readme] if readme.exists() else [])
        d = readers.read_wesad_pickle(pkl)
        label = np.asarray(d["label"])
        chest, wrist = d["signal"]["chest"], d["signal"]["wrist"]
        chest_s = len(label) / readers.WESAD_CHEST_HZ
        codes, counts = np.unique(label, return_counts=True)
        for code, n in zip(codes.tolist(), counts.tolist(), strict=True):
            label_seconds[int(code)] += n / readers.WESAD_CHEST_HZ
        wrist_s = {k: v.shape[0] / readers.WESAD_WRIST_HZ[k]
                   for k, v in wrist.items() if k in readers.WESAD_WRIST_HZ}
        channels = {"chest": sum(_width(v) for v in chest.values()),
                    "wrist": sum(_width(v) for v in wrist.values())}
        signals = list(chest.values()) + list(wrist.values())
        info = readers.read_wesad_readme(readme) if readme.exists() else {}
        demographics.append(info)
        per_subject.append({
            "subject": str(d.get("subject", pkl.stem)),
            "hours": r(chest_s / 3600),
            "chest_lengths_match_label": all(v.shape[0] == len(label) for v in chest.values()),
            "wrist_minus_chest_s": {k: r(s - chest_s, 1) for k, s in wrist_s.items()},
            "label_minutes": {str(c): r(n / readers.WESAD_CHEST_HZ / 60, 1)
                              for c, n in zip(codes.tolist(), counts.tolist(), strict=True)},
            "nan_values": int(sum(np.isnan(v).sum() for v in signals if v.dtype.kind == "f")),
            "chest_signals": {k: list(v.shape) for k, v in chest.items()},
            "wrist_signals": {k: list(v.shape) for k, v in wrist.items()},
            "readme": info,
        })
        del d, chest, wrist, signals, label
    ids = [int(s["subject"].lstrip("S")) for s in per_subject if s["subject"].lstrip("S").isdigit()]
    absent = sorted(set(range(1, max(ids) + 1)) - set(ids)) if ids else []
    hours = r(sum(s["hours"] for s in per_subject), 2)
    gaps = [abs(v) for s in per_subject for v in s["wrist_minus_chest_s"].values()]
    ages = pd.to_numeric(pd.Series([i.get("Age") for i in demographics]), errors="coerce")
    minutes = {readers.WESAD_LABELS.get(c, str(c)) + f" ({c})": r(s / 60, 1)
               for c, s in sorted(label_seconds.items())}

    res["subjects"] = {"count": len(per_subject), "ids": [s["subject"] for s in per_subject],
                       "absent_ids_up_to_max": [f"S{i}" for i in absent]}
    res["hours"] = hours
    res["channels"] = {"count": channels, "note": "chest RespiBAN + wrist Empatica E4"}
    res["labels"] = {"minutes_by_code": minutes}
    res["time_series"] = {"value": True, "chest_hz_documented": readers.WESAD_CHEST_HZ,
                          "wrist_hz_documented": readers.WESAD_WRIST_HZ,
                          "max_abs_wrist_minus_chest_s": r(max(gaps), 1) if gaps else None}
    res["missingness"] = {"nan_values": sum(s["nan_values"] for s in per_subject)}
    res["details"] = {
        "subjects": per_subject,
        "demographics": {
            "age_mean": r(ages.mean(), 1), "age_min": r(ages.min(), 0), "age_max": r(ages.max(), 0),
            "gender": dict(Counter(i.get("Gender", "?") for i in demographics)),
        },
    }
    if absent:
        res["notes"].append(f"absent subject folders: {', '.join(f'S{i}' for i in absent)} "
                            "(wesad_readme.pdf I.2: discarded after sensor malfunction)")
    main = [readers.WESAD_LABELS[c] + f" {label_seconds[c] / 60:.0f} min" for c in (1, 2, 3, 4)
            if c in label_seconds]
    hz = readers.WESAD_WRIST_HZ.values()
    res["table"] = {
        "subjects": str(len(per_subject)),
        "size": f"{hours} h",
        "channels": f"chest {channels['chest']} + wrist {channels['wrist']}",
        "labels": ", ".join(main),
        "injury": "no",
        "time_series": f"yes: chest {readers.WESAD_CHEST_HZ} Hz, wrist {min(hz)}-{max(hz)} Hz "
                       "(documented)",
    }
    return res


# ---------------------------------------------------------------- Sport DB 2.0 (#5)


def _is_hr(key: str) -> bool:
    return re.search(r"(^|\.)HR(_|$)", key) is not None


def audit_sportdb(folder: Path) -> dict:
    mats = sorted(folder.rglob("Data.mat"))
    if not mats:
        return {"status": "no Data.mat files found", "table": {}}
    res = new_result()
    root = Path(os.path.commonpath([m.parent for m in mats]))
    if len(mats) == 1:
        root = root.parent
    records, schemas, phase_names = [], Counter(), Counter()
    for mat in mats:
        crd = mat.parent
        parts = crd.relative_to(root).parts  # e.g. AMF/S1/CRD1, BAS/BAS1/S1/CRD1, SKA/CRD1
        res["_files"] += [p for p in (mat, crd / "Dem.txt", crd / "TrNote.txt") if p.exists()]
        rec = readers.read_sportdb_record(crd)
        sig = rec["signals"]
        lengths = {k: int(max(v.shape)) if v.ndim else 1 for k, v in sig.items()}
        continuous = any(_is_hr(k) for k in sig)
        span = (max(e for _, _, e in rec["phases"]) - min(s for _, s, _ in rec["phases"])
                if rec["phases"] else None)
        ecg_per_hr = [lengths["ECG" + k[2:]] / lengths[k] for k in sig
                      if k.startswith("HR") and "ECG" + k[2:] in sig]
        hr_keys = [k for k in sig if _is_hr(k)]
        schemas[tuple(sorted(sig))] += 1
        phase_names.update(name for name, _, _ in rec["phases"])
        records.append({
            "recording": "/".join(parts),
            "sport": parts[0],
            "subject": "/".join(parts[:-1]),
            "has_subject_folder": len(parts) >= 3,
            "continuous": continuous,
            "signals": {k: list(v.shape) for k, v in sig.items()},
            "timed_span_s": r(span, 1),
            "implied_hr_hz": r(lengths[hr_keys[0]] / span) if len(hr_keys) == 1 and span else None,
            "ecg_samples_per_hr_sample": [r(x, 2) for x in ecg_per_hr],
            "nan_values": int(sum(np.isnan(v).sum() for v in sig.values() if v.dtype.kind == "f")),
            "dem": rec["dem"],
        })
    cont = [x for x in records if x["continuous"]]
    clips = [x for x in records if not x["continuous"]]
    timed = [x for x in records if x["timed_span_s"]]
    hours = r(sum(x["timed_span_s"] for x in timed) / 3600, 2)
    subject_folders = sorted({x["subject"] for x in records if x["has_subject_folder"]})
    no_folder = [x["recording"] for x in records if not x["has_subject_folder"]]
    # One demographics row per person: per subject folder, or per recording when there is none.
    people: dict[str, dict] = {}
    inconsistent = set()
    for x in records:
        key = x["subject"] if x["has_subject_folder"] else x["recording"]
        if key in people and people[key] != x["dem"]:
            inconsistent.add(key)
        people.setdefault(key, x["dem"])
    dem = pd.DataFrame(list(people.values()))
    for col in ("Sex", "Age", "Height", "Weight"):
        if col in dem:
            dem[col] = pd.to_numeric(dem[col], errors="coerce")
    hr_hz = [x["implied_hr_hz"] for x in records if x["implied_hr_hz"]]
    ratios = [v for x in records for v in x["ecg_samples_per_hr_sample"]]
    cont_names = sorted({re.sub(r"(_.*|\..*)$", "", k) for x in cont for k in x["signals"]})

    res["subjects"] = {"subject_folders": len(subject_folders),
                       "recordings_without_subject_folder": no_folder,
                       "people_upper_bound": len(people),
                       "subjects_with_differing_dem_across_recordings": sorted(inconsistent)}
    res["rows"] = {"recordings": len(records), "continuous": len(cont), "short_clips": len(clips)}
    res["hours"] = {"with_timed_notes": hours, "recordings_with_timed_notes": len(timed)}
    res["channels"] = {"schemas": {" ".join(k): n for k, n in schemas.most_common()}}
    res["labels"] = {"type": "free-text phase notes, no class labels",
                     "top_phase_names": dict(phase_names.most_common(15))}
    res["time_series"] = {
        "value": True, "continuous_recordings": len(cont),
        "implied_hr_hz_median": r(float(np.median(hr_hz))) if hr_hz else None,
        "implied_hr_hz_range": [r(min(hr_hz)), r(max(hr_hz))] if hr_hz else None,
        "ecg_samples_per_hr_sample": sorted(set(ratios)),
        "note": "sampling rates are not stored in the files; implied rates [VERIFY] against the "
                "dataset description",
    }
    res["missingness"] = {"nan_values": sum(x["nan_values"] for x in records),
                          "dem_na_percent": nan_percent(dem)}
    res["details"] = {
        "recordings_per_sport": dict(Counter(x["sport"] for x in records)),
        "demographics": {
            "sex_counts": value_counts(dem["Sex"]) if "Sex" in dem else None,
            "age": ({"mean": r(dem["Age"].mean(), 1), "min": r(dem["Age"].min(), 0),
                     "max": r(dem["Age"].max(), 0)} if "Age" in dem else None),
        },
        "recordings": records,
    }
    if "Height" in dem and "Weight" in dem:
        res["plausibility"] = {
            "height_weight_correlation": correlation(dem["Height"], dem["Weight"]),
            "uniformity": {c: uniformity(dem[c]) for c in ("Age", "Height", "Weight") if c in dem},
        }
    if no_folder:
        res["notes"].append(f"{len(no_folder)} recordings have no subject folder, so the person "
                            f"behind them is unknown: {', '.join(no_folder)}")
    res["notes"].append("subject folders restart per sport/team: the same person in two sports "
                        "cannot be detected")
    hz = res["time_series"]["implied_hr_hz_median"]
    res["table"] = {
        "subjects": f"{len(subject_folders)} subject folders + {len(no_folder)} recordings "
                    "without one",
        "size": f"{len(records)} recordings; {hours} h in {len(timed)} with timed notes",
        "channels": f"continuous {'/'.join(cont_names)} in {len(cont)}; short ECG clips only "
                    f"in {len(clips)}",
        "labels": "none (free-text phase notes)",
        "injury": "no",
        "time_series": f"yes in {len(cont)} recordings (implied HR ~{hz} Hz [VERIFY])",
    }
    return res


# ---------------------------------------------------------------- Kaggle tabular (#6, #7)


def audit_sirp(folder: Path) -> dict:
    files = sorted(folder.rglob("*.xlsx"))
    if not files:
        return {"status": "no .xlsx file found", "table": {}}
    path = files[0]
    df = readers.read_sirp_xlsx(path)
    label = "Injury_Risk"
    features = [c for c in df.columns if c != label]
    res = new_result()
    res["_files"] = [path]
    meta = readers.xlsx_metadata(path)
    if "created" in meta and "modified" in meta:
        created = datetime.fromisoformat(meta["created"].replace("Z", "+00:00"))
        modified = datetime.fromisoformat(meta["modified"].replace("Z", "+00:00"))
        meta["seconds_created_to_modified"] = (modified - created).total_seconds()
    counts = value_counts(df[label])
    res["subjects"] = {"count": None, "note": "no athlete ID column"}
    res["rows"] = int(len(df))
    res["channels"] = {"count": len(features), "names": features}
    res["labels"] = {label: counts}
    res["injury"] = {"has_labels": True, "label": label, "counts": counts,
                     "note": "risk label of unknown origin, not observed injuries"}
    res["time_series"] = {"value": False, "detail": "one row per record, no time column"}
    res["missingness"] = nan_percent(df)
    res["duplicates"] = int(df.duplicated().sum())
    numeric = df[features].select_dtypes("number")
    plaus: dict = {
        "uniformity": {c: uniformity(numeric[c]) for c in numeric if numeric[c].nunique() > 10},
        "label_rule_probe": label_rule_probe(numeric, df[label]),
    }
    if {"Height_cm", "Weight_kg"} <= set(df.columns):
        plaus["height_weight_correlation"] = correlation(df["Height_cm"], df["Weight_kg"])
    if {"BMI", "Height_cm", "Weight_kg"} <= set(df.columns):
        bmi = df["Weight_kg"] / (df["Height_cm"] / 100) ** 2
        plaus["bmi_equals_weight_over_height2_share"] = r(((df["BMI"] - bmi).abs() <= 0.011).mean())
    res["plausibility"] = plaus
    res["details"] = {"file": path.name, "file_metadata": meta,
                      "describe": numeric.describe().round(2).to_dict()}
    res["table"] = {
        "subjects": "no IDs",
        "size": f"{len(df):,} rows",
        "channels": f"{len(features)} features",
        "labels": f"{label} " + ", ".join(f"{k}: {v}" for k, v in counts.items()),
        "injury": "risk label only (origin unknown)",
        "time_series": "no",
    }
    return res


def audit_ziya(folder: Path) -> dict:
    files = sorted(folder.rglob("*.csv"))
    if not files:
        return {"status": "no .csv file found", "table": {}}
    path = files[0]
    df = readers.read_ziya_csv(path)
    res = new_result()
    res["_files"] = [path]
    ts = pd.to_datetime(df["Timestamp"])
    step_min = (ts.diff().dropna().dt.total_seconds() / 60).value_counts()
    by_athlete = df.assign(_ts=ts).sort_values("_ts").groupby("Athlete_ID")["_ts"]
    gaps = by_athlete.diff().dropna().dt.total_seconds() / 60
    acts = df["Activity_Status"].value_counts()
    bp = df["Blood_Pressure"].str.split("/", expand=True).apply(pd.to_numeric, errors="coerce")
    hr = df.groupby("Activity_Status")["Heart_Rate"]
    features = [c for c in df.columns if c not in ("Record_ID", "Athlete_ID", "Timestamp")]
    res["subjects"] = {"count": int(df["Athlete_ID"].nunique()),
                       "ids": sorted(df["Athlete_ID"].unique().tolist()),
                       "records_per_athlete": value_counts(df["Athlete_ID"])}
    res["rows"] = int(len(df))
    res["channels"] = {"count": len(features), "names": features}
    res["labels"] = {"Activity_Status": {k: int(v) for k, v in acts.items()}}
    res["time_series"] = {
        "value": False,
        "global_step_minutes": {str(r(k, 1)): int(v) for k, v in step_min.items()},
        "per_athlete_gap_minutes": {"median": r(gaps.median(), 1), "min": r(gaps.min(), 1),
                                    "max": r(gaps.max(), 1)},
        "range": [str(ts.min()), str(ts.max())],
        "detail": "one global clock; consecutive rows usually belong to different athletes",
    }
    res["missingness"] = nan_percent(df)
    res["duplicates"] = int(df.drop(columns=["Record_ID"]).duplicated().sum())
    res["plausibility"] = {
        "mean_hr_by_activity": {k: r(v, 1) for k, v in hr.mean().items()},
        "hr_running_vs_resting": mean_contrast(
            df.loc[df["Activity_Status"] == "Running", "Heart_Rate"],
            df.loc[df["Activity_Status"] == "Resting", "Heart_Rate"]),
        "systolic_above_diastolic_share": r((bp[0] > bp[1]).mean()),
        "uniformity": {c: uniformity(v) for c, v in {
            "Heart_Rate": df["Heart_Rate"], "Body_Temperature": df["Body_Temperature"],
            "Blood_Oxygen": df["Blood_Oxygen"], "Step_Count": df["Step_Count"],
            "systolic": bp[0], "diastolic": bp[1]}.items()},
    }
    res["details"] = {
        "file": path.name,
        "secure_transmission_status": value_counts(df["Secure_Transmission_Status"]),
        "describe": df.select_dtypes("number").describe().round(2).to_dict(),
    }
    res["table"] = {
        "subjects": f"{res['subjects']['count']} athlete IDs",
        "size": f"{len(df):,} rows",
        "channels": f"{len(features)} fields",
        "labels": "Activity_Status " + ", ".join(f"{k}: {v}" for k, v in acts.items()),
        "injury": "no",
        "time_series": "no real per-athlete series (one 5-min clock, athletes interleaved)"
        if not step_min.empty and step_min.index[0] == 5 else "no",
    }
    return res


AUDITORS = {
    "runners": audit_runners,
    "pamap2": audit_pamap2,
    "wesad": audit_wesad,
    "sportdb": audit_sportdb,
    "sirp": audit_sirp,
    "ziya": audit_ziya,
}


def has_files(folder: Path) -> bool:
    return folder.is_dir() and any(p.is_file() for p in folder.rglob("*"))


def audit_dataset(folder: Path, reader: str) -> dict:
    if not has_files(folder):
        return {"status": "not downloaded", "table": {}}
    if reader not in AUDITORS:
        return {"status": f"no auditor for reader '{reader}'", "table": {}}
    return AUDITORS[reader](folder)


# ---------------------------------------------------------------- outputs


def table_row(cfg: dict, res: dict, mirror: dict | None = None) -> str:
    status = res.get("status", "not audited")
    cells = res.get("table", {})
    computed = [cells.get(k) or (status if status != "ok" else "") for k in TABLE_KEYS]
    origin = cfg.get("origin", "")
    if cfg.get("mirror_of"):
        if not mirror:
            origin += f"; mirror check pending ({cfg['mirror_of']} not available)"
        else:
            verdicts = {kind: m.get("identical") for kind, m in mirror.items()}
            origin += "; mirror check: " + ", ".join(
                f"{k} {'identical' if v else 'differs' if v is False else 'n/a'}"
                for k, v in verdicts.items())
    row = [str(cfg.get("number", "")), origin, *computed, str(cfg.get("licence", "")),
           str(cfg.get("role", "pending"))]
    return "| " + " | ".join(c.replace("|", "/") for c in row) + " |"


def render_table(cfgs: dict, results: dict, mirrors: dict, run_id: str) -> str:
    rows = [table_row(cfgs[n], results[n], mirrors.get(n))
            for n in sorted(results, key=lambda n: cfgs[n]["number"])]
    return (f"Generated by `python tasks.py audit` (run `{run_id}`, see "
            f"experiments/audit/{run_id}/). Origin, licence and role come from "
            "configs/datasets.yaml; all other cells from audit.json.\n\n"
            + TABLE_HEADER + "\n" + "\n".join(rows))


def replace_table_block(text: str, table_md: str) -> str:
    start, end = text.find(TABLE_START), text.find(TABLE_END)
    if start < 0 or end < start:
        raise ValueError(f"{TABLE_START} / {TABLE_END} markers missing")
    return text[: start + len(TABLE_START)] + "\n" + table_md + "\n" + text[end:]


def update_manifest(rows_by_dataset: dict[str, list[dict]], path: Path = MANIFEST) -> None:
    """Replace the rows of the audited datasets, keep the others."""
    kept = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            kept = [row for row in csv.DictReader(f) if row["dataset"] not in rows_by_dataset]
    rows = kept + [row for rows in rows_by_dataset.values() for row in rows]
    rows.sort(key=lambda row: (row["dataset"], row["file"]))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def clean(obj):
    """Make results JSON-safe: numpy types to Python, NaN to None, paths and dates to text."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        return clean(obj.item())
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return obj


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run_info(run_id: str, only: str | None, fingerprints: dict[str, str | None]) -> dict:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "PyYAML")
    return {
        "run_id": run_id,
        "command": " ".join(sys.argv),
        "only": only,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _git("status", "--porcelain") not in ("", "unavailable"),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in packages},
        "seed": SEED,
        "data_fingerprints": fingerprints,
    }


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def run_audit(only: str | None = None) -> Path:
    cfgs = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["datasets"]
    if only and only not in cfgs:
        raise SystemExit(f"unknown dataset {only!r}; choose from: {', '.join(cfgs)}")
    names = [only] if only else sorted(cfgs, key=lambda n: cfgs[n]["number"])
    results, manifest_rows, fingerprints = {}, {}, {}
    for name in names:
        cfg = cfgs[name]
        print(f"[{cfg['number']}] {name}", flush=True)
        t0 = time.time()
        res = audit_dataset(RAW / cfg["folder"], cfg["reader"])
        files = res.pop("_files", [])
        hashes = {p.relative_to(RAW).as_posix(): sha256_file(p) for p in files}
        res["files"] = sorted(hashes)
        res["size_mb"] = r(sum(p.stat().st_size for p in files) / 1e6, 1)
        res["fingerprint"] = fingerprints[name] = fingerprint(hashes) if hashes else None
        manifest_rows[name] = [
            {"dataset": name, "file": f, "source_url": cfg.get("source_url", ""),
             "licence": cfg.get("licence", ""), "access_date": str(cfg.get("access_date") or ""),
             "sha256": sha} for f, sha in sorted(hashes.items())]
        results[name] = res
        cells = res.get("table", {})
        print(f"    {res['status']} | {len(files)} files | subjects: {cells.get('subjects', '-')} "
              f"| {cells.get('size', '-')} | {time.time() - t0:.0f} s", flush=True)

    mirrors = {}
    for name in names:
        source = cfgs[name].get("mirror_of")
        if source and results[name]["status"] == "ok":
            src_folder = RAW / cfgs[source]["folder"]
            src_res = results.get(source)  # None on an --only run of the mirror
            ready = has_files(src_folder) if src_res is None else src_res["status"] == "ok"
            if ready:
                mirrors[name] = mirror_check(src_folder, RAW / cfgs[name]["folder"])
                results[name]["mirror"] = {"of": source, **mirrors[name]}

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    table = render_table(cfgs, results, mirrors, run_id)
    write_json(run_dir / "audit.json", {"run_id": run_id, "datasets": results})
    (run_dir / "audit_table.md").write_text(table + "\n", encoding="utf-8")
    shutil.copyfile(CONFIG, run_dir / "config.yaml")
    write_json(run_dir / "run_info.json", run_info(run_id, only, fingerprints))
    update_manifest(manifest_rows)
    if not only:
        text = DATASETS_MD.read_text(encoding="utf-8")
        DATASETS_MD.write_text(replace_table_block(text, table), encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252
    parser = argparse.ArgumentParser(description="Audit the raw datasets in data/raw/.")
    parser.add_argument("--only", help="audit one dataset (name from configs/datasets.yaml)")
    args = parser.parse_args(argv)
    run_dir = run_audit(args.only)
    print("wrote", run_dir.relative_to(ROOT).as_posix(), "and data/MANIFEST.csv"
          + ("" if args.only else " and the table in docs/DATASETS.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
