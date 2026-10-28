"""Data for the coach dashboard (piece 9). Reads saved run outputs; never the locked runners.

The replayed athletes are development runners of one round's TEST group (never 64-73 before the
evaluation freeze; PLAN piece 9). Risk 0-100 = percentile of the calibrated fused risk among
that round's validation events (D-027(4)); an alert = raw fused score at or above the round's
validation-chosen alert level (D-022).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import loaders
from src.models import data

ROOT = loaders.ROOT
EXP = ROOT / "experiments"


def latest_run(phase: str) -> Path | None:
    """Newest experiments/<phase>/<run_id>/ that has a metrics.json."""
    folder = EXP / phase
    runs = sorted(p for p in folder.glob("20*") if (p / "metrics.json").exists()) \
        if folder.exists() else []
    return runs[-1] if runs else None


def test_dates(k: int) -> np.ndarray:
    """Event day index of the round's test rows on the shared days, in the round's row order."""
    day, week = loaders.load_dev("day"), loaders.load_dev("week")
    i_day, _ = loaders.shared_days(day, week)
    split = loaders.lovdal_split()
    masks = data.split_rows(day.athlete_id[i_day], split["folds"][str(k)], split["lockbox_ids"])
    return day.date[i_day][masks["test"]]


def runner_events(fused_run: Path, k: int = 0, seed: int = 0) -> pd.DataFrame:
    """One row per scored test event of round k: runner, day, risk 0-100, alert, injury."""
    z = np.load(fused_run / "predictions" / f"fused_s{seed}_r{k}.npz")
    rnd = json.loads((fused_run / "metrics.json").read_text("utf-8"))["fused"]["rounds"][k]
    threshold = rnd["seeds"][seed]["alert_threshold"]
    df = pd.DataFrame({"runner": z["person_test"].astype(int), "day": test_dates(k),
                       "risk_0_100": np.round(z["risk_test"], 1), "score": z["s_test"],
                       "injury": z["y_test"].astype(int)})
    df["alert"] = (df["score"] >= threshold) if threshold is not None else False
    df["row"] = np.arange(len(df))  # position in the round's test rows (for explanations)
    return df.sort_values(["runner", "day"]).reset_index(drop=True)


class Explainer:
    """The round's chosen tabular arm, refit with seed 0 on that round's training runners;
    TreeSHAP reasons for any test event (week inputs)."""

    def __init__(self, fused_run: Path, k: int = 0) -> None:
        from src.models import fitters
        rnd = json.loads((fused_run / "metrics.json").read_text("utf-8"))["fused"]["rounds"][k]
        tab = rnd["tabular_arm"]
        _, self.r = list(data.lovdal_shared_rounds())[k]
        build = (fitters.random_forest_model if tab["model"] == "random_forest"
                 else fitters.catboost_model)
        self.model, self.arm = build(self.r, tab["setting"], 0), tab["model"]

    def reasons(self, row: int, top: int = 3) -> tuple[str, pd.DataFrame]:
        from src.xai import explain
        x = self.r.test.X[row:row + 1]
        sv = explain.tree_shap(self.model, x)[0]
        table = pd.DataFrame({"input": self.r.input_names, "value_vs_usual":
                              np.where(x[0] > 0, "higher", "lower"), "contribution": sv})
        table = table.reindex(table["contribution"].abs().sort_values(ascending=False).index)
        return explain.reason_sentence(self.r.input_names, x[0], sv, top), table.head(8)


def ziya07_stream() -> pd.DataFrame:
    """Demo-only wearable records (unknown origin, D-012): never evidence."""
    return pd.read_parquet(loaders.PROCESSED / "ziya07_wearable" / "table.parquet")
