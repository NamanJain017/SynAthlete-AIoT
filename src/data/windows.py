"""Sensor windows for PAMAP2 and WESAD (piece 2, DATA_CONTRACT Track B).

Each recording is resampled to 25 Hz and cut into 128-step windows (stride 64). A window is kept
only if all its samples are valid and it (plus a small margin) carries one allowed label.
Raw sensor units are kept: scaling is fitted on training subjects only, later (rule 3).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import decimate, resample_poly

from src.data import manifest, readers
from src.data.parallel import parallel_map, pick_workers

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
# scipy's FIR decimate uses 2 * 10 * q taps, so it smears 10 * q source samples to each side.
FIR_HALF_LEN_PER_Q = 10


def load_channels(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"channels_{name}.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- small pure helpers


def window_starts(n: int, window: int, stride: int) -> np.ndarray:
    """Start indices of all full windows: floor((n - window) / stride) + 1 of them."""
    if n < window:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, n - window + 1, stride)


def nan_runs(x: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of each run of NaNs; end is exclusive."""
    isnan = np.isnan(x).astype(np.int8)
    edges = np.diff(np.concatenate(([0], isnan, [0])))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True))


def fill_short_gaps(x: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill NaN runs of at most max_gap samples. Returns (filled copy, valid mask).

    Longer runs and runs touching either end stay invalid. They are filled too, only so that
    filters can run over them; the mask makes sure their windows are dropped.
    """
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    if valid.all():
        return x.copy(), valid
    if not valid.any():
        return np.zeros_like(x), valid
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[valid], x[valid])
    ok = valid.copy()
    for start, end in nan_runs(x):
        if end - start <= max_gap and start > 0 and end < len(x):
            ok[start:end] = True
    return filled, ok


def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    """Also mark k samples on each side of every True sample (filter edge effects)."""
    if k <= 0 or not mask.any():
        return mask
    return np.convolve(mask.astype(np.float64), np.ones(2 * k + 1), mode="same") > 0.5


def pure_window_starts(labels: np.ndarray, valid: np.ndarray, window: int, stride: int,
                       margin: int, keep: set[int]) -> np.ndarray:
    """Windows whose samples are all valid and whose label (also `margin` steps around) is one
    single value from `keep`."""
    out = []
    for s in window_starts(len(labels), window, stride):
        lo, hi = max(0, s - margin), min(len(labels), s + window + margin)
        seg = labels[lo:hi]
        if int(seg[0]) in keep and (seg == seg[0]).all() and valid[s:s + window].all():
            out.append(s)
    return np.asarray(out, dtype=np.int64)


def nearest_index(t: np.ndarray, hz: float, n: int) -> np.ndarray:
    """Index of the sample at (or nearest to) each time t in a series sampled at hz.

    Rounding, not truncation: k / 25 * 700 can come out as 27.999..., which truncation
    would turn into the sample before the intended one.
    """
    return np.minimum(np.rint(np.asarray(t) * hz).astype(np.int64), n - 1)


def cut(signal: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    if len(starts) == 0:
        return np.empty((0, window, signal.shape[1]), dtype=np.float32)
    return np.stack([signal[s:s + window] for s in starts]).astype(np.float32)


# ---------------------------------------------------------------- PAMAP2


def pamap2_file_windows(path: Path) -> dict:
    """One subject file -> 25 Hz windows. Module-level so worker processes can run it."""
    cfg = load_channels("pamap2")
    src, dst = cfg["source_hz"], cfg["sampling_hz"]
    q = src // dst
    df = readers.read_pamap2_dat(path.read_bytes(), name=path.name).sort_values("timestamp")
    ts = df["timestamp"].to_numpy()
    breaks = np.where(np.diff(ts) > 1.5 / src)[0] + 1  # none in the current files: a guard
    keep = set(np.unique(df["activity_id"]).astype(int)) - set(cfg["drop_labels"])
    xs, ys, starts_s = [], [], []
    for seg in np.split(np.arange(len(df)), breaks):
        part = df.iloc[seg]
        if len(part) < q * cfg["window"]:
            continue
        cols, valid = [], np.ones(len(part), dtype=bool)
        for ch in cfg["channels"]:
            gap_s = cfg["max_gap_s"]["heart_rate" if ch == "heart_rate" else "imu"]
            filled, ok = fill_short_gaps(part[ch].to_numpy(), int(gap_s * src))
            cols.append(filled)
            valid &= ok
        sig = np.stack([decimate(c, q, ftype="fir", zero_phase=True) for c in cols], axis=1)
        bad = dilate(~valid, FIR_HALF_LEN_PER_Q * q)[::q][: len(sig)]
        labels = part["activity_id"].to_numpy()[::q][: len(sig)]
        starts = pure_window_starts(labels, ~bad, cfg["window"], cfg["stride"],
                                    cfg["label_margin"], keep)
        xs.append(cut(sig, starts, cfg["window"]))
        ys.append(labels[starts].astype(np.int16))
        starts_s.append(part["timestamp"].to_numpy()[0] + starts / dst)
    X = np.concatenate(xs) if xs else np.empty((0, cfg["window"], len(cfg["channels"])), np.float32)
    n = len(X)
    return {"subject": path.stem.removeprefix("subject"), "session": path.parent.name, "X": X,
            "y": np.concatenate(ys) if ys else np.empty(0, np.int16),
            "start_s": np.concatenate(starts_s) if starts_s else np.empty(0),
            "subject_col": np.full(n, path.stem.removeprefix("subject")),
            "session_col": np.full(n, path.parent.name)}


# ---------------------------------------------------------------- WESAD


def wesad_subject_windows(pkl: Path) -> dict:
    """One subject pickle (read with the numpy-only unpickler) -> 25 Hz wrist windows."""
    cfg = load_channels("wesad")
    dst = cfg["sampling_hz"]
    d = readers.read_wesad_pickle(pkl)
    label = np.asarray(d["label"]).ravel()
    wrist = {k: np.asarray(v, dtype=np.float64) for k, v in d["signal"]["wrist"].items()}
    del d
    hz = readers.WESAD_WRIST_HZ
    n = int(len(label) / cfg["label_hz"] * dst)
    t = np.arange(n) / dst

    def poly(x: np.ndarray, src: int) -> np.ndarray:
        g = np.gcd(dst, src)
        return resample_poly(x, dst // g, src // g, axis=0)

    def interp(x: np.ndarray, src: int) -> np.ndarray:
        return np.interp(t, np.arange(len(x)) / src, x.ravel())

    acc = poly(wrist["ACC"], hz["ACC"])
    parts = {"BVP": poly(wrist["BVP"].ravel(), hz["BVP"]), "EDA": interp(wrist["EDA"], hz["EDA"]),
             "TEMP": interp(wrist["TEMP"], hz["TEMP"]),
             "ACC_x": acc[:, 0], "ACC_y": acc[:, 1], "ACC_z": acc[:, 2]}
    n = min(n, *(len(v) for v in parts.values()))
    sig = np.column_stack([parts[ch][:n] for ch in cfg["channels"]])
    lab = label[nearest_index(t[:n], cfg["label_hz"], len(label))]
    keep = {int(k) for k in cfg["keep_labels"]}
    starts = pure_window_starts(lab, np.isfinite(sig).all(axis=1), cfg["window"], cfg["stride"],
                                cfg.get("label_margin", 10), keep)
    X = cut(sig, starts, cfg["window"])
    return {"subject": pkl.stem, "session": "wrist", "X": X, "y": lab[starts].astype(np.int16),
            "start_s": starts / dst, "subject_col": np.full(len(X), pkl.stem),
            "session_col": np.full(len(X), "wrist")}


# ---------------------------------------------------------------- build + save

WESAD_BYTES_PER_FILE_BYTE = 1.6  # measured unpickle peak 1.1x file size + resampled float64 copies
PAMAP2_BYTES_PER_FILE_BYTE = 8  # text -> float64 table + filled copies (estimate)


def build(name: str, workers: int | None = None) -> dict:
    cfg = load_channels(name)
    if name == "pamap2":
        files = sorted((RAW / "pamap2").rglob("subject*.dat"))
        func, factor = pamap2_file_windows, PAMAP2_BYTES_PER_FILE_BYTE
    else:
        files = sorted((RAW / "wesad").rglob("S*.pkl"), key=lambda p: int(p.stem[1:]))
        func, factor = wesad_subject_windows, WESAD_BYTES_PER_FILE_BYTE
    source = manifest.verify(files, RAW / name)
    biggest = max(p.stat().st_size for p in files)
    n_workers = pick_workers(len(files), int(factor * biggest), workers)
    outs = parallel_map(func, files, n_workers, label=f"{name} files")
    X = np.concatenate([o["X"] for o in outs])
    y = np.concatenate([o["y"] for o in outs])
    subject = np.concatenate([o["subject_col"] for o in outs]).astype(str)
    session = np.concatenate([o["session_col"] for o in outs]).astype(str)
    start_s = np.concatenate([o["start_s"] for o in outs])
    out_dir = PROCESSED / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "windows.npz", X=X, y=y, subject=subject, session=session,
                        start_s=start_s)
    labels, counts = np.unique(y, return_counts=True)
    subs, scounts = np.unique(subject, return_counts=True)
    meta = {
        "shape": list(X.shape), "channels": cfg["channels"], "sampling_hz": cfg["sampling_hz"],
        "window": cfg["window"], "stride": cfg["stride"], "workers": n_workers,
        "windows_per_label": {str(k): int(v) for k, v in zip(labels, counts, strict=True)},
        "windows_per_subject": {str(k): int(v) for k, v in zip(subs, scounts, strict=True)},
        "nan_values": int(np.isnan(X).sum()),
        "source": source,
    }
    (out_dir / "windows_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    names = argv or ["pamap2", "wesad"]
    for name in names:
        t0 = time.time()
        meta = build(name)
        print(f"[{name}] windows {meta['shape']}, per label {meta['windows_per_label']}, "
              f"NaN {meta['nan_values']}, {meta['workers']} workers, {time.time() - t0:.0f} s",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
