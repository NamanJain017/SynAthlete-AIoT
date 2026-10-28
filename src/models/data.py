"""Round builder for piece 3: one round's data, scaled on that round's training people only.

Leak-sensitive, so written by Claude Code (not a task card). Rules it enforces:
- people, never rows, decide train / validation / test (D-017, D-020);
- one z-score scaler per round, fitted on training rows only (D-023, AGENTS.md rule 3);
- model inputs are exactly the declared 70 (day) / 69 (week) features: never an ID, date, event
  count, gap or history length (D-023, leak canary part a, D-024);
- the locked Lövdal runners are never touched (load_dev refuses them; checked again here).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from src.data import loaders

DECLARED_INPUTS = {"day": 70, "week": 69}
# Words that would mean an identity, a time stamp or the runner's own event history.
FORBIDDEN_INPUT = re.compile(r"\b(athlete|id|date|index|gap|history|events?)\b", re.IGNORECASE)


@dataclass
class Part:
    """Rows of one role (train, val or test) in a round."""

    X: np.ndarray
    y: np.ndarray
    person: np.ndarray  # runner or subject ID per row (for grouping and the bootstrap)


@dataclass
class Round:
    k: int
    kind: str  # "day", "week" or "pamap2"
    train: Part
    val: Part
    test: Part
    input_names: list[str]
    scaler_mean: np.ndarray
    scaler_std: np.ndarray


# ---------------------------------------------------------------- inputs


def input_names(data: loaders.RunnerData) -> list[str]:
    """Names of the flattened inputs, step-major (oldest step first), matching flatten()."""
    steps = data.X.shape[1]
    unit = "d" if data.kind == "day" else "w"
    names = [f"{f} [{unit}-{steps - t}]" for t in range(steps) for f in data.feature_names]
    return names + list(data.extra_names)


def flatten(data: loaders.RunnerData) -> np.ndarray:
    """(N, steps, features) -> (N, steps * features), plus the week file's 3 relative changes."""
    flat = data.X.reshape(len(data.X), -1)
    return flat if data.extra is None else np.hstack([flat, data.extra])


def check_declared_inputs(names: list[str], kind: str) -> None:
    """Leak canary part (a): exactly the declared number of inputs and no forbidden one."""
    if len(names) != DECLARED_INPUTS[kind]:
        raise ValueError(f"{kind}: {len(names)} inputs, declared {DECLARED_INPUTS[kind]}")
    bad = [n for n in names if FORBIDDEN_INPUT.search(n)]
    if bad:
        raise ValueError(f"forbidden model inputs: {bad}")


# ---------------------------------------------------------------- scaling


def fit_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and SD over training rows (and over steps for sequences).

    A feature that is constant in training gets SD 1, so it stays constant instead of dividing
    by zero.
    """
    axes = tuple(range(X_train.ndim - 1))
    mean = X_train.mean(axis=axes, dtype=np.float64)
    std = X_train.std(axis=axes, dtype=np.float64)
    return mean, np.where(std > 0, std, 1.0)


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


# ---------------------------------------------------------------- rounds


def split_rows(person: np.ndarray, fold: dict, locked: list | None = None) -> dict[str, np.ndarray]:
    """Boolean row masks for train / val / test from a fold's person lists.

    Refuses overlapping person lists, locked people, and rows that belong to no role.
    """
    ids = {role: {str(p) for p in fold[f"{role}_ids"]} for role in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if ids[a] & ids[b]:
            raise ValueError(f"people in both {a} and {b}: {sorted(ids[a] & ids[b])}")
    person_str = person.astype(str)
    if locked is not None and np.isin(person_str, [str(p) for p in locked]).any():
        raise ValueError("locked people in the development data")
    masks = {role: np.isin(person_str, sorted(ids[role])) for role in ids}
    if not (masks["train"] | masks["val"] | masks["test"]).all():
        raise ValueError("rows whose person is in no role of this fold")
    return masks


def make_round(k: int, kind: str, X: np.ndarray, y: np.ndarray, person: np.ndarray,
               fold: dict, names: list[str], locked: list | None = None) -> Round:
    """Split by person, fit the scaler on training rows only, scale all three parts."""
    masks = split_rows(person, fold, locked)
    mean, std = fit_scaler(X[masks["train"]])
    parts = {role: Part(apply_scaler(X[m], mean, std), y[m], person[m])
             for role, m in masks.items()}
    return Round(k, kind, parts["train"], parts["val"], parts["test"], names, mean, std)


def lovdal_rounds(kind: str, layout: str = "flat") -> Iterator[Round]:
    """The 5 development rounds of one Lövdal file.

    layout "flat": (N, 70 or 69) for table models; "sequence": (N, steps, features) for networks
    (day file only; the week file's 3 relative changes have no time step).
    """
    if layout not in ("flat", "sequence") or (layout == "sequence" and kind != "day"):
        raise ValueError(f"layout {layout!r} not available for the {kind} file")
    data = loaders.load_dev(kind)
    names = input_names(data)
    check_declared_inputs(names, kind)
    X = flatten(data) if layout == "flat" else data.X
    split = loaders.lovdal_split()
    for k in sorted(split["folds"], key=int):
        yield make_round(int(k), kind, X, data.y.astype(np.int64), data.athlete_id,
                         split["folds"][k], names, split["lockbox_ids"])


def lovdal_shared_rounds() -> Iterator[tuple[Round, Round]]:
    """Per round: (day sequences, week features) on the SAME athlete-days, same row order
    (the shared days, D-014), each with its own training-rows scaler."""
    day, week = loaders.load_dev("day"), loaders.load_dev("week")
    check_declared_inputs(input_names(day), "day")
    check_declared_inputs(input_names(week), "week")
    i_day, i_week = loaders.shared_days(day, week)
    y, person = day.y[i_day].astype(np.int64), day.athlete_id[i_day]
    X_seq, X_tab = day.X[i_day], flatten(week)[i_week]
    split = loaders.lovdal_split()
    for k in sorted(split["folds"], key=int):
        fold = split["folds"][k]
        yield (make_round(int(k), "day", X_seq, y, person, fold, input_names(day),
                          split["lockbox_ids"]),
               make_round(int(k), "week", X_tab, y, person, fold, input_names(week),
                          split["lockbox_ids"]))


def pamap2_rounds() -> Iterator[Round]:
    """The 4 PAMAP2 rounds: 128 x 10 clips, scaled per channel on training subjects (D-020)."""
    with np.load(loaders.PROCESSED / "pamap2" / "windows.npz") as z:
        X, y, subject = z["X"], z["y"].astype(np.int64), z["subject"]
    meta = json.loads((loaders.PROCESSED / "pamap2" / "windows_meta.json").read_text("utf-8"))
    split = json.loads((loaders.SPLITS / "pamap2_folds.json").read_text("utf-8"))
    if meta["source"]["fingerprint"] != split["meta"]["data_fingerprint"]:
        raise RuntimeError("PAMAP2 windows are not the data version the splits were made from")
    for k in sorted(split["folds"], key=int):
        yield make_round(int(k), "pamap2", X, y, subject, split["folds"][k], meta["channels"])
