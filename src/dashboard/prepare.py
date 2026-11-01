"""Prepare everything the dashboard shows: python tasks.py demo --prepare.

The dashboard itself never trains (except the Training lab, D-016), so the live demo is instant
and cannot fail halfway. Each part is built from saved run outputs; a part whose run does not
exist yet is skipped and the page says so. Output: experiments/dashboard/bundle/ (git-ignored:
it holds per-runner predictions derived from the private data).

Data roles (honesty rule, D-001/D-012): coach pages use REAL Lövdal injuries (development
runners of round 0's test group, never the locked 64-73); the sensor and synthetic pages use the
wearable datasets (no injury labels); SIRP-600 and ziya07 appear only as labelled demos.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.dashboard import data as dd
from src.data import loaders

BUNDLE = loaders.ROOT / "experiments" / "dashboard" / "bundle"
ROUND = 0
DATASETS = [  # D-012 / D-013 / D-018 roles, as recorded in docs/DECISIONS.md
    ("Lövdal runners (74 runners, 7 years of daily logs)", "real", "the only injury evidence: "
     "baselines, combined model, explanations, coach view"),
    ("Kaggle 'Injury Prediction for Competitive Runners'", "real (copy)", "byte-identical copy of "
     "Lövdal: no separate role"),
    ("PAMAP2 (9 people, wearable sensors)", "real", "edge sensor stream, activity recognition, "
     "generator training and fidelity"),
    ("WESAD (15 people, wrist sensors)", "real", "second dataset for generator generalisation"),
    ("Sport DB 2.0 (55 heart-rate recordings)", "real", "heart-rate realism check and second edge "
     "stream"),
    ("SoccerMon, PMData", "real", "exploratory only (Track C), never main evidence"),
    ("SIRP-600", "unknown origin", "demo only: shows the SHAP code runs"),
    ("ziya07 wearable", "unknown origin", "demo only: dashboard demo stream"),
]


def _save(name: str, obj) -> None:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(BUNDLE / f"{name}.parquet")
    else:
        (BUNDLE / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float), "utf-8")


def coach(fused: Path) -> None:
    """Risk per event + TreeSHAP reasons for every event of round 0's test runners."""
    from src.xai import explain
    ev = dd.runner_events(fused, ROUND)
    ex = dd.Explainer(fused, ROUND)
    sv = explain.tree_shap(ex.model, ex.r.test.X)  # all test rows, round order
    names = ex.r.input_names
    ev["reason"] = [explain.reason_sentence(names, ex.r.test.X[i], sv[i]) for i in ev["row"]]
    top = []
    for i in ev["row"]:
        order = np.argsort(-np.abs(sv[i]))[:6]
        top.append(json.dumps([[names[j], float(sv[i, j]), "higher" if ex.r.test.X[i, j] > 0
                                else "lower"] for j in order]))
    ev["top_inputs"] = top
    _save("events", ev.drop(columns=["score"]))
    _save("coach_meta", {"fused_run": fused.name, "round": ROUND, "tabular_arm": ex.arm})


def results() -> None:
    """Headline tables, copied from the metrics files written by code."""
    out = {}
    for phase in ("baselines", "fused", "generators", "tiers", "track_c", "explain", "lockbox",
                  "comparisons"):
        runs = sorted((loaders.ROOT / "experiments" / phase).glob("20*"))
        out[phase] = []
        for r in runs:
            files = {p.name: json.loads(p.read_text("utf-8")) for p in r.glob("metrics*.json")}
            if files:
                out[phase].append({"run": r.name, "files": files})
    _save("results", out)


def sensor_stream() -> None:
    """A PAMAP2 test person of round 0, replayed window by window (no overlap)."""
    windows = loaders.PROCESSED / "pamap2" / "windows.npz"
    window_meta = loaders.PROCESSED / "pamap2" / "windows_meta.json"
    if not windows.is_file() or not window_meta.is_file():
        print("sensor stream: skipped (processed PAMAP2 windows not found)", flush=True)
        return
    from src.pipeline import bench
    raw, clf, person = bench.pamap2_stream()
    acts = clf(raw)
    meta = json.loads(window_meta.read_text("utf-8"))
    hr = raw[:, :, meta["channels"].index("heart_rate")]
    acc = np.linalg.norm(raw[:, :, 7:10], axis=2)  # ankle accelerometer magnitude
    df = pd.DataFrame({"window": np.arange(len(raw)), "seconds": np.arange(len(raw)) * 5.12,
                       "activity_predicted": acts, "hr_mean": hr.mean(1), "hr_max": hr.max(1),
                       "ankle_acc_mean": acc.mean(1), "reflex_alert": hr.max(1) > 180})
    _save("sensor_stream", df)
    _save("sensor_meta", {"pamap2_person": person, "window_seconds": 5.12,
                          "activity_names": ACTIVITIES, "data_role": "real wearable data, "
                          "different people from the runners (systems demo)"})


ACTIVITIES = {1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running", 6: "cycling",
              7: "Nordic walking", 9: "watching TV", 10: "computer work", 11: "car driving",
              12: "ascending stairs", 13: "descending stairs", 16: "vacuum cleaning",
              17: "ironing", 18: "folding laundry", 19: "house cleaning", 20: "playing soccer",
              24: "rope jumping"}


def synthetic_samples() -> None:
    """Real vs synthetic heart-rate clips (bpm) of the newest generator run, round 0."""
    runs = sorted((loaders.ROOT / "experiments" / "generators").glob("20*/samples"))
    if not runs:
        return
    if not (loaders.PROCESSED / "pamap2" / "windows.npz").is_file():
        print("synthetic samples: skipped (processed PAMAP2 windows not found)", flush=True)
        return
    rows = []
    for f in sorted(runs[-1].glob("pamap2_*_r0.npz")):
        with np.load(f) as z:
            bpm = z["X"] * z["hr_sd"] + z["hr_mean"]
            y = z["y"]
        gen = f.stem.replace("pamap2_", "").replace("_r0", "")
        for act in (1, 4, 5, 6):  # lying, walking, running, cycling
            idx = np.flatnonzero(y == act)[:3]
            for j, i in enumerate(idx):
                rows += [{"generator": gen, "activity": ACTIVITIES[act], "clip": j, "t": t,
                          "bpm": float(v)} for t, v in enumerate(bpm[i])]
    from src.models import data
    r = next(data.pamap2_rounds())
    for act in (1, 4, 5, 6):
        idx = np.flatnonzero(r.train.y == act)[:3]
        for j, i in enumerate(idx):
            bpm = r.train.X[i, :, 0] * r.scaler_std[0] + r.scaler_mean[0]
            rows += [{"generator": "real", "activity": ACTIVITIES[act], "clip": j, "t": t,
                      "bpm": float(v)} for t, v in enumerate(bpm)]
    _save("synthetic_samples", pd.DataFrame(rows))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    _save("datasets", [{"dataset": d, "origin": o, "role": r} for d, o, r in DATASETS])
    ziya07_path = loaders.PROCESSED / "ziya07_wearable" / "table.parquet"
    if ziya07_path.is_file():
        _save("ziya07_demo", dd.ziya07_stream().head(60).astype(str))
        print("ziya07 demo: done (unknown-origin demo data)", flush=True)
    else:
        print("ziya07 demo: skipped (optional processed data not found)", flush=True)
    results()
    print("results: done", flush=True)
    sensor_stream()
    print("sensor stream: done", flush=True)
    synthetic_samples()
    print("synthetic samples: done (if a generator run exists)", flush=True)
    fused = dd.latest_run("fused")
    lovdal_day = loaders.PROCESSED / "lovdal" / "day.npz"
    if fused is not None and lovdal_day.is_file():
        coach(fused)
        print(f"coach view: done (fused run {fused.name})", flush=True)
    else:
        reason = "no fused run yet" if fused is None else "processed Lövdal development data not found"
        print(f"coach view: skipped ({reason})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
