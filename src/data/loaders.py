"""Turn raw datasets into model-ready arrays (piece 2).

Loaders keep raw values: no scaling, no imputation. Scalers and imputers are fitted on training
folds only, later (AGENTS.md rule 3). File parsing lives in readers.py; this module only reshapes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import readers

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

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


def save_runner_data(data: RunnerData, out_dir: Path) -> Path:
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
        "shape": list(data.X.shape),
        "axis1": "oldest -> newest step (last = just before the event day)",
        "feature_names": data.feature_names,
        "extra_names": list(data.extra_names),
        "events": int(len(data.y)),
        "injuries": int(data.y.sum()),
        "athletes": int(len(np.unique(data.athlete_id))),
        "rest_day_cells": int(data.rest_day.sum()) if data.rest_day is not None else None,
    }
    (out_dir / f"{data.kind}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------- command line


def prepare_lovdal() -> dict:
    day, week = load_lovdal("day"), load_lovdal("week")
    i_day, _ = shared_days(day, week)
    for data in (day, week):
        save_runner_data(data, PROCESSED / "lovdal")
    return {"day": list(day.X.shape), "day_injuries": int(day.y.sum()),
            "week": list(week.X.shape), "week_injuries": int(week.y.sum()),
            "week_extra": list(week.extra.shape), "shared_days": int(len(i_day)),
            "shared_injuries": int(day.y[i_day].sum())}


def main() -> int:
    import sys
    import time

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for name, job in {"lovdal": prepare_lovdal}.items():
        t0 = time.time()
        print(f"[{name}] {job()} ({time.time() - t0:.1f} s)", flush=True)
    print("wrote data/processed/ (git-ignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
