"""Daily tables for the exploratory datasets SoccerMon and PMData (piece 2, Track C, D-018).

One row per person per calendar day. Raw values: days without a report stay NaN (no imputation).
Pain reports are grouped into episodes with the Lövdal-style rule:
- a report at most 21 days after the same person's previous report continues the episode;
- a gap of 22 days or more starts a new episode (Lövdal's injuries are all >= 22 days apart).
An episode's first day is a positive event; a day with no pain report within 21 days before and
after is a healthy event. Days within 21 days of the start or end of a person's data cannot be
judged and are neither. No report on a day is read as "no pain" (assumption, D-018).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import manifest, readers

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
EPISODE_GAP_DAYS = 21
SOCCERMON_LOAD = ("daily_load", "weekly_load", "atl", "ctl28", "ctl42", "acwr", "monotony",
                  "strain")
SOCCERMON_WELLNESS = ("fatigue", "mood", "readiness", "sleep_duration", "sleep_quality",
                      "soreness", "stress")
PMDATA_USED = ("fitbit/resting_heart_rate.json", "fitbit/steps.json",
               "fitbit/very_active_minutes.json", "fitbit/moderately_active_minutes.json",
               "fitbit/lightly_active_minutes.json", "fitbit/sedentary_minutes.json",
               "fitbit/sleep_score.csv", "pmsys/wellness.csv", "pmsys/srpe.csv",
               "pmsys/injury.csv")


# ---------------------------------------------------------------- the episode rule (pure)


def label_days(days: pd.DatetimeIndex, report_days: set[pd.Timestamp],
               gap: int = EPISODE_GAP_DAYS) -> pd.DataFrame:
    """Per calendar day: pain_report, episode_start, healthy_event (see module docstring)."""
    days = pd.DatetimeIndex(sorted(days))
    start, end = days.min(), days.max()
    reports = sorted(d for d in report_days if start <= d <= end)
    starts, prev = set(), None
    for d in reports:
        new = prev is None or (d - prev).days > gap
        # A first report too close to the data start may continue an unseen earlier episode.
        if new and (d - start).days > gap:
            starts.add(d)
        prev = d
    rep = np.array([d in report_days for d in days])
    near = np.zeros(len(days), dtype=bool)
    for d in reports:
        near |= np.abs((days - d).days) <= gap
    judgeable = ((days - start).days > gap) & ((end - days).days > gap)
    return pd.DataFrame({
        "pain_report": rep,
        "episode_start": [d in starts for d in days],
        "healthy_event": judgeable & ~near,
    }, index=days)


def _long(wide: pd.DataFrame, name: str) -> pd.DataFrame:
    out = wide.reset_index().melt(id_vars="date", var_name="person", value_name=name)
    return out.set_index(["person", "date"])


# ---------------------------------------------------------------- SoccerMon


def soccermon_files(folder: Path) -> list[Path]:
    """Every file soccermon_daily reads."""
    return ([folder / "training-load" / f"{m}.csv" for m in SOCCERMON_LOAD]
            + [folder / "wellness" / f"{m}.csv" for m in SOCCERMON_WELLNESS]
            + [folder / "training-load" / "session.json", folder / "illness" / "illness.csv",
               folder / "injury" / "injury.csv"])


def soccermon_daily(folder: Path = RAW / "soccermon" / "subjective") -> pd.DataFrame:
    parts = [_long(readers.read_soccermon_wide(folder / "training-load" / f"{m}.csv"), m)
             for m in SOCCERMON_LOAD]
    parts += [_long(readers.read_soccermon_wide(folder / "wellness" / f"{m}.csv"), m)
              for m in SOCCERMON_WELLNESS]
    table = pd.concat(parts, axis=1).sort_index()
    sessions = readers.read_soccermon_sessions(folder / "training-load" / "session.json")
    n_sessions = sessions.groupby(["player", "date"]).size().rename("n_sessions")
    n_sessions.index = n_sessions.index.set_names(["person", "date"])
    table = table.join(n_sessions)
    table["n_sessions"] = table["n_sessions"].fillna(0).astype(int)
    illness = readers.read_soccermon_reports(folder / "illness" / "illness.csv")
    ill = set(zip(illness["player_name"], illness["date"], strict=True))
    table["illness_report"] = [k in ill for k in table.index]
    injury = readers.read_soccermon_reports(folder / "injury" / "injury.csv")
    injury = injury[injury["type"].map(lambda s: len(json.loads(s)) > 0)]
    return _add_labels(table, injury.groupby("player_name")["date"].apply(set).to_dict(),
                       observed_cols=SOCCERMON_WELLNESS + ("n_sessions",))


# ---------------------------------------------------------------- PMData


PMDATA_MISSING: dict[str, list[str]] = {}  # person -> files the dataset itself does not contain


def _fitbit_daily(folder: Path) -> pd.DataFrame:
    """Daily Fitbit values; a file missing from the dataset gives an all-NaN column (recorded)."""
    fb = folder / "fitbit"
    cols: dict[str, pd.Series] = {}

    def daily(name: str, how: str) -> pd.Series:
        path = fb / f"{name}.json"
        if not path.exists():
            PMDATA_MISSING.setdefault(folder.name, []).append(f"fitbit/{name}.json")
            return pd.Series(dtype=float)
        m = readers.read_fitbit_json(path)
        grouped = m.groupby(m["date_time"].dt.normalize())["value"]
        return grouped.sum(min_count=1) if how == "sum" else grouped.first()

    cols["resting_hr"] = daily("resting_heart_rate", "first")
    cols["steps"] = daily("steps", "sum")
    for name in ("very_active_minutes", "moderately_active_minutes", "lightly_active_minutes",
                 "sedentary_minutes"):
        cols[name] = daily(name, "first")
    sleep = pd.read_csv(fb / "sleep_score.csv")
    sleep["date"] = readers.utc_date(sleep["timestamp"])
    first = sleep.sort_values("timestamp").groupby("date").first()
    for c in ("overall_score", "deep_sleep_in_minutes", "restlessness"):
        cols[f"sleep_{c}"] = first[c]
    cols["sleep_resting_hr"] = first["resting_heart_rate"]
    out = pd.DataFrame(cols)
    out.index.name = "date"
    return out


def _pmsys_daily(folder: Path) -> pd.DataFrame:
    pm = folder / "pmsys"
    well = readers.read_pmdata_pmsys(pm / "wellness.csv", "effective_time_frame")
    keep = ["fatigue", "mood", "readiness", "sleep_duration_h", "sleep_quality", "soreness",
            "stress"]
    out = well.sort_values("effective_time_frame").groupby("date")[keep].first()
    srpe = readers.read_pmdata_pmsys(pm / "srpe.csv", "end_date_time")
    srpe["load"] = srpe["perceived_exertion"] * srpe["duration_min"]
    out = out.join(srpe.groupby("date")["load"].sum().rename("srpe_load"), how="outer")
    out = out.join(srpe.groupby("date").size().rename("n_sessions"), how="outer")
    out.index.name = "date"
    return out


def pmdata_person(folder: Path) -> tuple[pd.DataFrame, set | None]:
    table = _fitbit_daily(folder).join(_pmsys_daily(folder), how="outer")
    injury_file = folder / "pmsys" / "injury.csv"
    if not injury_file.exists():
        PMDATA_MISSING.setdefault(folder.name, []).append("pmsys/injury.csv")
        return table, None  # p08: missing file is NOT "no injuries"
    inj = readers.read_pmdata_injuries(injury_file)
    return table, set(inj.loc[inj["has_injury"], "date"])


def pmdata_files(folder: Path) -> list[Path]:
    """Every file pmdata_daily reads (files missing from the dataset itself are skipped)."""
    people = sorted(p for p in folder.iterdir() if p.is_dir() and p.name.startswith("p"))
    return [p / u for p in people for u in PMDATA_USED if (p / u).exists()]


def pmsys_late_rows(folder: Path) -> dict:
    """PMSys rows stamped 22:00-24:00 UTC: their UTC date may differ from the local date.

    PMSys times are UTC ('Z') while Fitbit days are the device's local days, so these rows
    could sit one day off from the Fitbit values of the same local day.
    """
    total = late = 0
    for person in sorted(p for p in folder.iterdir() if p.is_dir() and p.name.startswith("p")):
        for name, col in (("wellness.csv", "effective_time_frame"),
                          ("injury.csv", "effective_time_frame"), ("srpe.csv", "end_date_time")):
            path = person / "pmsys" / name
            if path.exists():
                t = pd.to_datetime(pd.read_csv(path)[col], utc=True, format="ISO8601")
                total, late = total + len(t), late + int((t.dt.hour >= 22).sum())
    return {"pmsys_rows": total, "stamped_22_to_24_utc": late}


def pmdata_daily(folder: Path = RAW / "pmdata") -> pd.DataFrame:
    tables, reports = [], {}
    for person in sorted(p.name for p in folder.iterdir() if p.is_dir() and p.name.startswith("p")):
        table, rep = pmdata_person(folder / person)
        table = table.assign(person=person).reset_index().set_index(["person", "date"])
        tables.append(table)
        if rep is not None:
            reports[person] = rep
    table = pd.concat(tables).sort_index()
    observed = ("fatigue", "resting_hr", "steps")
    return _add_labels(table, reports, observed_cols=observed, unlabelled_ok=True)


# ---------------------------------------------------------------- shared


def _add_labels(table: pd.DataFrame, reports: dict[str, set], observed_cols: tuple[str, ...],
                unlabelled_ok: bool = False) -> pd.DataFrame:
    """Restrict each person to their observed period and add the episode labels."""
    pieces = []
    for person, rows in table.groupby(level="person"):
        rows = rows.droplevel("person")
        seen = rows[list(observed_cols)].notna().any(axis=1)
        if "n_sessions" in rows and "n_sessions" in observed_cols:
            seen |= rows["n_sessions"] > 0
        if not seen.any():
            continue
        first, last = seen[seen].index.min(), seen[seen].index.max()
        days = pd.date_range(first, last, freq="D")
        rows = rows.reindex(days)
        rows.index.name = "date"
        if person in reports:
            rows = rows.join(label_days(days, reports[person]))
        elif unlabelled_ok:
            rows = rows.assign(pain_report=np.nan, episode_start=np.nan, healthy_event=np.nan)
        else:
            rows = rows.join(label_days(days, set()))
        pieces.append(rows.assign(person=person).reset_index().set_index(["person", "date"]))
    return pd.concat(pieces).sort_index()


def summarise(table: pd.DataFrame) -> dict:
    labelled = table["pain_report"].notna()
    t = table[labelled]
    per = t.groupby(level="person")[["pain_report", "episode_start"]].sum()
    return {
        "people": int(table.index.get_level_values("person").nunique()),
        "people_with_labels": int(t.index.get_level_values("person").nunique()),
        "person_days": int(len(table)),
        "pain_report_days": int(t["pain_report"].sum()),
        "episodes": int(t["episode_start"].sum()),
        "healthy_event_days": int(t["healthy_event"].sum()),
        "people_with_an_episode": int((per["episode_start"] > 0).sum()),
        "episodes_per_person": {k: int(v) for k, v in per["episode_start"].items() if v > 0},
        "missing_percent": {c: round(float(table[c].isna().mean() * 100), 1)
                            for c in table.columns if table[c].isna().any()},
    }


def build(name: str) -> dict:
    folder = RAW / name
    files = soccermon_files(folder / "subjective") if name == "soccermon" else pmdata_files(folder)
    source = manifest.verify(files, folder)
    table = soccermon_daily() if name == "soccermon" else pmdata_daily()
    out = PROCESSED / name
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "daily.parquet")
    meta = {"columns": list(table.columns), "episode_gap_days": EPISODE_GAP_DAYS,
            **summarise(table)}
    meta["source"] = source
    if name == "pmdata":
        meta["utc_date_check"] = pmsys_late_rows(folder)
        meta["files_missing_in_dataset"] = {k: sorted(v) for k, v in sorted(PMDATA_MISSING.items())}
    (out / "daily_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for name in argv or ["soccermon", "pmdata"]:
        t0 = time.time()
        m = build(name)
        print(f"[{name}] people {m['people']} ({m['people_with_labels']} labelled), person-days "
              f"{m['person_days']}, pain-report days {m['pain_report_days']}, episodes "
              f"{m['episodes']} in {m['people_with_an_episode']} people, healthy days "
              f"{m['healthy_event_days']} ({time.time() - t0:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
