"""Turn raw datasets into model-ready arrays (piece 2).

Loaders keep raw values: no scaling, no imputation. Scalers and imputers are fitted on training
folds only, later (AGENTS.md rule 3). File parsing lives in readers.py; this module only reshapes.
The Lövdal locked final-test runners are saved apart from the development runners (D-017):
use load_dev() for all development work; only src/data/lockbox.py reads the locked ones.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import manifest, readers

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
LOCKBOX_DIR = PROCESSED / "lovdal_lockbox"  # read only through src/data/lockbox.py

LOVDAL_PERCEIVED = ("perceived exertion", "perceived trainingSuccess", "perceived recovery")
REST_DAY_CODE = -0.01  # authors' README: perceived values on a rest day


# ---------------------------------------------------------------- Lövdal runner logs (Track A)


@dataclass
class RunnerData:
    """One Lövdal file as arrays. Axis 1 of X runs oldest -> newest (last = nearest the event)."""

    kind: str  # "day" or "week"
    X: np.ndarray  # (N, steps, features) float32
    y: np.ndarray  # (N,) int8, 1 = injury on the event day
    athlete_id: np.ndarray  # (N,) int16
    date: np.ndarray  # (N,) int32, event day index counted from the first record
    feature_names: list[str]
    extra: np.ndarray | None = None  # week file: (N, 3) relative mileage changes
    extra_names: tuple[str, ...] = ()
    rest_day: np.ndarray | None = None  # day file: (N, 7) True where all perceived values are -0.01


def _steps_by_suffix(columns: list[str]) -> dict[int, list[str]]:
    """pandas names repeated headers 'x', 'x.1', 'x.2': group the columns by that suffix."""
    steps: dict[int, list[str]] = {}
    for c in columns:
        m = re.search(r"\.(\d+)$", c)
        steps.setdefault(int(m.group(1)) if m else 0, []).append(c)
    return steps


def lovdal_arrays(df: pd.DataFrame, kind: str) -> RunnerData:
    """Reshape one runner file to (N, steps, features), oldest step first.

    Column order from the authors' README v2 (2024):
    - day file: suffix 0 (no suffix) = 7 days before the event, suffix 6 = the day before,
      so increasing suffix is already oldest -> newest;
    - week file: suffix 0 = the week before the event, suffix 2 = three weeks before,
      so the order must be reversed.
    """
    if kind not in ("day", "week"):
        raise ValueError(f"kind must be 'day' or 'week', got {kind!r}")
    df = df.sort_values(["Athlete ID", "Date"], kind="mergesort").reset_index(drop=True)
    features = [c for c in df.columns if c not in readers.RUNNER_ID_COLS]
    rel = [c for c in features if c.startswith("rel ")]
    steps = _steps_by_suffix([c for c in features if c not in rel])
    names = [re.sub(r"\.\d+$", "", c) for c in steps[0]]
    for s, cols in steps.items():
        if [re.sub(r"\.\d+$", "", c) for c in cols] != names:
            raise ValueError(f"{kind} file: step {s} has different features than step 0")
    order = sorted(steps) if kind == "day" else sorted(steps, reverse=True)
    X = np.stack([df[steps[s]].to_numpy(np.float32) for s in order], axis=1)
    data = RunnerData(
        kind=kind,
        X=X,
        y=df["injury"].to_numpy(np.int8),
        athlete_id=df["Athlete ID"].to_numpy(np.int16),
        date=df["Date"].to_numpy(np.int32),
        feature_names=names,
    )
    if rel:
        data.extra = df[rel].to_numpy(np.float32)
        data.extra_names = tuple(rel)
    perceived = [names.index(p) for p in LOVDAL_PERCEIVED if p in names]
    if kind == "day" and len(perceived) == len(LOVDAL_PERCEIVED):
        data.rest_day = np.isclose(X[:, :, perceived], REST_DAY_CODE).all(axis=-1)
    return data


def load_lovdal(kind: str, folder: Path = RAW / "lovdal") -> RunnerData:
    files = readers.find_runner_files(folder)
    if kind not in files:
        raise FileNotFoundError(f"no {kind}-approach file in {folder}")
    return lovdal_arrays(readers.read_runner_csv(files[kind]), kind)


def shared_days(day: RunnerData, week: RunnerData) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of the athlete-days present in both files (for fusion, D-014).

    Raises if a shared athlete-day has different injury labels in the two files.
    """
    a = pd.DataFrame({"athlete": day.athlete_id, "date": day.date, "i_day": range(len(day.y))})
    b = pd.DataFrame({"athlete": week.athlete_id, "date": week.date, "i_week": range(len(week.y))})
    m = a.merge(b, on=["athlete", "date"], how="inner", validate="one_to_one")
    i_day, i_week = m["i_day"].to_numpy(), m["i_week"].to_numpy()
    if not np.array_equal(day.y[i_day], week.y[i_week]):
        raise ValueError("injury labels differ between the day and week file on shared days")
    return i_day, i_week


def lovdal_split() -> dict:
    """The committed Lövdal split file (D-017)."""
    return json.loads((SPLITS / "lovdal_folds.json").read_text(encoding="utf-8"))


def lovdal_lockbox_ids() -> list[int]:
    """The locked final-test runners, from the committed split file (D-017)."""
    return lovdal_split()["lockbox_ids"]


def select(data: RunnerData, keep: np.ndarray) -> RunnerData:
    """The same file restricted to the rows where keep is True."""
    return dataclasses.replace(
        data, X=data.X[keep], y=data.y[keep], athlete_id=data.athlete_id[keep],
        date=data.date[keep],
        extra=None if data.extra is None else data.extra[keep],
        rest_day=None if data.rest_day is None else data.rest_day[keep],
    )


def save_runner_data(data: RunnerData, out_dir: Path, contents: str = "",
                     source: dict | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"X": data.X, "y": data.y, "athlete_id": data.athlete_id, "date": data.date}
    if data.extra is not None:
        arrays["extra"] = data.extra
    if data.rest_day is not None:
        arrays["rest_day"] = data.rest_day
    path = out_dir / f"{data.kind}.npz"
    np.savez_compressed(path, **arrays)
    meta = {
        "kind": data.kind,
        "contents": contents,
        "shape": list(data.X.shape),
        "axis1": "oldest -> newest step (last = just before the event day)",
        "feature_names": data.feature_names,
        "extra_names": list(data.extra_names),
        "events": int(len(data.y)),
        "injuries": int(data.y.sum()),
        "athletes": int(len(np.unique(data.athlete_id))),
        "athlete_ids": [int(a) for a in np.unique(data.athlete_id)],
        "rest_day_cells": int(data.rest_day.sum()) if data.rest_day is not None else None,
        "source": source,
    }
    (out_dir / f"{data.kind}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_runner_data(folder: Path, kind: str) -> RunnerData:
    """Read back what save_runner_data wrote."""
    meta = json.loads((folder / f"{kind}_meta.json").read_text(encoding="utf-8"))
    with np.load(folder / f"{kind}.npz") as z:
        return RunnerData(
            kind=kind, X=z["X"], y=z["y"], athlete_id=z["athlete_id"], date=z["date"],
            feature_names=meta["feature_names"],
            extra=z["extra"] if "extra" in z else None,
            extra_names=tuple(meta["extra_names"]),
            rest_day=z["rest_day"] if "rest_day" in z else None,
        )


def load_dev(kind: str) -> RunnerData:
    """Development runners only (IDs 0-63): the data for all work before the evaluation freeze.

    Refuses files built from another data version than the splits, or holding a locked runner.
    """
    folder, split = PROCESSED / "lovdal", lovdal_split()
    meta = json.loads((folder / f"{kind}_meta.json").read_text(encoding="utf-8"))
    if (meta.get("source") or {}).get("fingerprint") != split["meta"]["data_fingerprint"]:
        raise RuntimeError("processed Lövdal data is not the data version the splits were made "
                           "from: re-run python tasks.py data lovdal")
    data = load_runner_data(folder, kind)
    if np.isin(data.athlete_id, split["lockbox_ids"]).any():
        raise RuntimeError("development file contains locked runners: re-run python tasks.py data")
    return data


# ---------------------------------------------------------------- command line


def lovdal_facts(day: RunnerData, week: RunnerData, i_day: np.ndarray) -> dict:
    """Whole-dataset counts quoted in the docs (dataset-level, like the audit; D-017)."""
    per = pd.DataFrame({"athlete": day.athlete_id, "date": day.date, "y": day.y})
    injured = per.groupby("athlete")["y"].sum()
    inj = per[per["y"] == 1].sort_values(["athlete", "date"])
    gaps = inj.groupby("athlete")["date"].diff().dropna()
    return {
        "athletes": int(len(injured)),
        "day_events": int(len(day.y)), "day_injuries": int(day.y.sum()),
        "week_events": int(len(week.y)), "week_injuries": int(week.y.sum()),
        "athletes_injured_day_file": int((injured > 0).sum()),
        "athletes_never_injured_day_file": int((injured == 0).sum()),
        "shared_athlete_days": int(len(i_day)), "shared_injuries": int(day.y[i_day].sum()),
        "min_days_between_injuries_same_runner": int(gaps.min()) if len(gaps) else None,
    }


def prepare_lovdal() -> dict:
    """Whole files are checked (shapes, shared days), then saved as development and locked parts."""
    files = readers.find_runner_files(RAW / "lovdal")
    source = manifest.verify([files["day"], files["week"]], RAW / "lovdal")
    day, week = load_lovdal("day"), load_lovdal("week")
    i_day, _ = shared_days(day, week)
    locked_ids = lovdal_lockbox_ids()
    out = {"source_fingerprint": source["fingerprint"], "whole_day": list(day.X.shape),
           "whole_week": list(week.X.shape), "whole_shared_days": int(len(i_day))}
    facts = lovdal_facts(day, week, i_day)
    (PROCESSED / "lovdal").mkdir(parents=True, exist_ok=True)
    (PROCESSED / "lovdal" / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    for data in (day, week):
        locked = np.isin(data.athlete_id, locked_ids)
        dev = select(data, ~locked)
        save_runner_data(dev, PROCESSED / "lovdal", "development runners 0-63 (D-017)", source)
        save_runner_data(select(data, locked), LOCKBOX_DIR,
                         "LOCKED final-test runners 64-73: open only via lockbox.py", source)
        out[f"dev_{data.kind}"] = list(dev.X.shape)
        out[f"dev_{data.kind}_injuries"] = int(dev.y.sum())
    return out


def _short(meta: dict) -> dict:
    """The headline numbers of a builder's metadata, for the console."""
    return {k: v for k, v in meta.items()
            if not isinstance(v, (list, dict)) or k in ("shape", "windows_per_label")
            or (isinstance(v, list) and len(v) <= 3 and k.startswith(("dev_", "whole_")))}


def main(argv: list[str] | None = None) -> int:
    """python tasks.py data [NAME ...]: every builder (or the named ones), then the data cards."""
    import time

    from src.data import daily, datacards, reference, windows

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    jobs = {
        "lovdal": prepare_lovdal,
        "pamap2": lambda: windows.build("pamap2"),
        "wesad": lambda: windows.build("wesad"),
        "soccermon": lambda: daily.build("soccermon"),
        "pmdata": lambda: daily.build("pmdata"),
        "sportdb2": reference.build_sportdb,
        "sirp600": lambda: reference.build_demo("sirp600"),
        "ziya07_wearable": lambda: reference.build_demo("ziya07_wearable"),
    }
    unknown = [n for n in argv or [] if n not in jobs]
    if unknown:
        print(f"unknown dataset(s) {unknown}; choose from: {', '.join(jobs)}")
        return 2
    for name in argv or list(jobs):
        t0 = time.time()
        print(f"[{name}] {_short(jobs[name]())} ({time.time() - t0:.1f} s)", flush=True)
    print("wrote data/processed/ (git-ignored)")
    return datacards.main()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
