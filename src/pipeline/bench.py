"""Piece-6 benchmark: python -m src.pipeline.bench (tasks.py pipeline).

Streams: (1) a PAMAP2 test person of round 0 (heart rate + 3 accelerometers, 25 Hz, 5-s
windows, no overlap); (2) a Sport DB 2.0 heart-rate recording (about 1 Hz, 128-sample windows).
Each stream runs tiered (edge -> fog -> cloud) and cloud-only, 3 repetitions each, with every
process set up before the first window and one window sent every tiers.INTERVAL_S (D-034).
Edge uses the piece-3 activity CNN (round 0's chosen setting, trained on round 0's training
people) on the CPU; cloud scores a round-0 test runner's days with piece-3 logistic regression.
Writes experiments/tiers/<run_id>/{run_info.json, metrics.json, summary.md}.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from src.data import loaders
from src.models import baselines, data, nets
from src.models import run as base_run
from src.pipeline import tiers

OUT = loaders.ROOT / "experiments" / "tiers"
P3_GPU = loaders.ROOT / "experiments" / "baselines" / "2026-10-26_022902" / "metrics.json"
P3_CPU = loaders.ROOT / "experiments" / "baselines" / "2026-10-26_020012" / "metrics.json"
REPEATS = 3


class ActivityClassifier:
    """Picklable: rebuilt from its weights inside the edge process; runs on the CPU."""

    def __init__(self, state: dict, setting: dict, n_channels: int, classes: np.ndarray,
                 mean: np.ndarray, std: np.ndarray) -> None:
        self.state, self.setting, self.n_channels = state, setting, n_channels
        self.classes, self.mean, self.std, self._model = classes, mean, std, None

    def __call__(self, w: np.ndarray) -> np.ndarray:
        if self._model is None:
            torch.set_num_threads(1)
            self._model = nets.build_model("cnn1d", self.n_channels, len(self.classes),
                                           self.setting)
            self._model.load_state_dict(self.state)
        x = ((w - self.mean) / self.std).astype(np.float32)
        return nets.predict_labels(self._model, x, self.classes)


class DailyModel:
    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(rows)[:, 1]


def _chosen(path, name, k=0, kind=None):
    res = json.loads(path.read_text("utf-8"))[name]
    rnd = (res[kind] if kind else res)["rounds"][k]
    return rnd["trials"][rnd["chosen"]]["setting"]


def pamap2_stream() -> tuple[np.ndarray, ActivityClassifier, str]:
    r = next(data.pamap2_rounds())
    setting = _chosen(P3_GPU, "pamap2_cnn1d")
    nets.seed_everything(0)
    classes = np.unique(r.train.y)
    model = nets.build_model("cnn1d", 10, len(classes), setting)
    res = nets.train_model(model, "multiclass", r.train.X, r.train.y, r.val.X, r.val.y,
                           lr=setting["lr"], batch_size=setting["batch_size"],
                           max_epochs=setting["max_epochs"], patience=setting["patience"],
                           seed=0, device=nets.pick_device())
    state = {k: v.cpu() for k, v in res["model"].state_dict().items()}
    clf = ActivityClassifier(state, setting, 10, classes, r.scaler_mean, r.scaler_std)
    with np.load(loaders.PROCESSED / "pamap2" / "windows.npz") as z:
        person = sorted(set(r.test.person.tolist()))[0]
        keep = z["subject"] == person
        order = np.lexsort((z["start_s"][keep], z["session"][keep]))
        raw = z["X"][keep][order][::2]  # stride 64 of 128: every second window = no overlap
    return raw.astype(np.float32), clf, str(person)


def sportdb_stream() -> tuple[np.ndarray, str]:
    hr = pd.read_parquet(loaders.PROCESSED / "sportdb2" / "hr.parquet")
    rec = hr.loc[hr["hr_valid"]].groupby("recording").size().idxmax()  # longest valid recording
    x = hr.loc[(hr["recording"] == rec) & hr["hr_valid"], "hr_bpm"].to_numpy(np.float32)
    n = len(x) // 128
    return x[: n * 128].reshape(n, 128, 1), str(rec)


def daily() -> tuple[np.ndarray, DailyModel, int]:
    r = next(data.lovdal_rounds("day"))
    model = baselines.logistic_regression(r.train.X, r.train.y,
                                          _chosen(P3_CPU, "logistic_regression", kind="day"))
    runner = int(sorted(set(r.test.person.tolist()))[0])
    return r.test.X[r.test.person == runner], DailyModel(model), runner


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    code = base_run._code_state()
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = OUT / run_id
    folder.mkdir(parents=True)
    t_start = time.perf_counter()
    pam, clf, person = pamap2_stream()
    sdb, recording = sportdb_stream()
    rows, dmodel, runner = daily()
    streams = {
        "pamap2": (pam, tiers.ReflexRule(hr_channel=0, acc_channels=(4, 5, 6)), clf),
        "sportdb2_hr": (sdb, tiers.ReflexRule(hr_channel=0), None),
    }
    results = {"note": "simulated on one laptop; wearable person and runner are different "
                       "people (systems demo, not injury evidence)",
               "pamap2_person": person, "sportdb2_recording": recording,
               "lovdal_runner_round0_test": runner, "repeats": REPEATS,
               "send_interval_s": tiers.INTERVAL_S, "streams": {}}
    for name, (windows, rule, classify) in streams.items():
        out = {}
        for mode in ("tiered", "cloud_only"):
            reps = [tiers.run(windows, rule, classify, rows, dmodel, mode)
                    for _ in range(REPEATS)]
            out[mode] = {k: (float(np.median([r[k] for r in reps]))
                             if isinstance(reps[0][k], float) else reps[0][k]) for k in reps[0]}
            print(f"{name} {mode}: into cloud {out[mode]['bytes_into_cloud']:,} bytes, "
                  f"median decision {out[mode]['decision_latency_ms_median']:.2f} ms", flush=True)
        out["bytes_reduction_factor"] = (out["cloud_only"]["bytes_into_cloud"]
                                         / max(out["tiered"]["bytes_into_cloud"], 1))
        results["streams"][name] = out
    info = {**code, "runtime_s": round(time.perf_counter() - t_start, 1),
            "data_fingerprints": base_run._fingerprints(), "torch": torch.__version__}
    (folder / "run_info.json").write_text(json.dumps(info, indent=2), "utf-8")
    (folder / "metrics.json").write_text(json.dumps(results, indent=2), "utf-8")
    lines = [f"# Piece-6 tiers, run {run_id}", "", results["note"], "",
             "| Stream | Mode | Windows | Bytes into cloud | Median / p95 decision latency (ms) |",
             "|---|---|---|---|---|"]
    for name, out in results["streams"].items():
        for mode in ("tiered", "cloud_only"):
            m = out[mode]
            lines.append(f"| {name} | {mode} | {m['windows']} | {m['bytes_into_cloud']:,} | "
                         f"{m['decision_latency_ms_median']:.2f} / "
                         f"{m['decision_latency_ms_p95']:.2f} |")
        lines.append(f"| {name} | reduction | | {out['bytes_reduction_factor']:.0f}x fewer | |")
    (folder / "summary.md").write_text("\n".join(lines) + "\n", "utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
