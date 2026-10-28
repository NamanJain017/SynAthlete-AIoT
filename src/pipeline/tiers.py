"""Edge-fog-cloud tiers as three plain processes joined by queues (piece 6, D-008).

- edge "reflex": gets a wearable stream one 5-s window (128 samples x channels at 25 Hz) at a
  time; a quick check (heart-rate and motion rule, plus the activity CNN) and a ~100-byte
  summary instead of the raw window;
- fog "supervisor": groups window summaries into sessions and sends a session summary;
- cloud "deep": runs the daily injury model on a development runner's days and logs sessions.
Mode "cloud_only" sends every raw window straight to the cloud, which does all the work.
Messages are pickled; bytes = length of the pickled message. Times: time.perf_counter_ns.
Latency = time from a window being made to its decision (fog in tiered mode, cloud in
cloud-only mode). So that it measures processing and transfer, not start-up backlog (D-034):
every process first finishes its own set-up (imports, model warm-up, the daily model) and waits
at a shared barrier; then windows are sent at a steady pace (one every `interval_s`), not all
at once.
A systems demo on one laptop: the wearable person and the runner are different people, so
nothing here is evidence about injuries (D-001).
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import time
from dataclasses import dataclass

import numpy as np

STOP = "STOP"
INTERVAL_S = 0.1  # one window every 0.1 s: a steady load, faster than real time (5-s windows)


@dataclass
class ReflexRule:
    hr_channel: int | None      # index of heart rate in the window, or None
    hr_high_bpm: float = 180.0  # heart rate above this -> immediate alert
    acc_channels: tuple = ()    # accelerometer channels (m/s^2)
    acc_spike: float = 60.0     # |a| above this (about 6 g) -> immediate alert

    def check(self, w: np.ndarray) -> dict:
        out = {"alert": False}
        if self.hr_channel is not None:
            hr = w[:, self.hr_channel]
            out["hr_mean"] = float(hr.mean())
            out["alert"] |= bool(hr.max() > self.hr_high_bpm)
        if self.acc_channels:
            mag = np.linalg.norm(w[:, list(self.acc_channels)], axis=1)
            out["acc_mean"] = float(mag.mean())
            out["alert"] |= bool(mag.max() > self.acc_spike)
        return out


def _send(q, msg, counter) -> None:
    counter.value += len(pickle.dumps(msg))
    q.put(msg)


def _warm_up(classify, stream) -> None:
    """Build the activity model before the clock starts (its first call loads the weights)."""
    if classify is not None and len(stream.windows):
        classify(stream.windows[:1])


def edge(stream, rule: ReflexRule, classify, out_q, sent_bytes, ready) -> None:
    """Reflex tier: one summary message per window (or the raw window in cloud-only mode)."""
    _warm_up(classify, stream)
    ready.wait()
    for i, (t_made, w) in enumerate(stream):
        t0 = time.perf_counter_ns()
        summary = {"i": i, "t_made": t_made, **rule.check(w),
                   "activity": None if classify is None else int(classify(w[None])[0])}
        summary["edge_ns"] = time.perf_counter_ns() - t0
        _send(out_q, summary, sent_bytes)
    _send(out_q, STOP, sent_bytes)


def fog(in_q, out_q, sent_bytes, session_windows: int, results, ready) -> None:
    """Supervisor tier: session summaries every `session_windows` windows."""
    buf, latencies = [], []
    ready.wait()
    while (msg := in_q.get()) != STOP:
        t_now = time.perf_counter_ns()
        latencies.append(t_now - msg["t_made"])  # sensor -> fog decision (alerts are seen here)
        buf.append(msg)
        if len(buf) == session_windows:
            _send(out_q, _session(buf), sent_bytes)
            buf = []
    if buf:
        _send(out_q, _session(buf), sent_bytes)
    _send(out_q, STOP, sent_bytes)
    results["fog_latency_ns"] = latencies


def _session(buf: list[dict]) -> dict:
    acts = [m["activity"] for m in buf if m["activity"] is not None]
    return {"windows": len(buf), "alerts": int(sum(m["alert"] for m in buf)),
            "hr_mean": float(np.mean([m.get("hr_mean", np.nan) for m in buf])),
            "acc_mean": float(np.mean([m.get("acc_mean", np.nan) for m in buf])),
            "activities": {int(a): acts.count(a) for a in set(acts)}}


def cloud(in_q, daily_rows, daily_model, results, raw_mode, rule, classify,
          session_windows, ready, warm_window) -> None:
    """Deep tier: daily injury risk for the runner's days, plus whatever arrives from below.
    In cloud-only mode it also does the reflex check and the session summaries itself."""
    t0 = time.perf_counter_ns()
    results["daily_risk"] = [float(v) for v in daily_model(daily_rows)]
    results["daily_model_ns"] = time.perf_counter_ns() - t0
    if raw_mode and classify is not None and warm_window is not None:
        classify(warm_window)
    sessions, latencies, buf = [], [], []
    ready.wait()
    while (msg := in_q.get()) != STOP:
        if raw_mode:  # msg = (index, t_made, raw window)
            i, t_made, w = msg
            summary = {"i": i, "t_made": t_made, **rule.check(w),
                       "activity": None if classify is None else int(classify(w[None])[0])}
            latencies.append(time.perf_counter_ns() - t_made)
            buf.append(summary)
            if len(buf) == session_windows:
                sessions.append(_session(buf))
                buf = []
        else:
            sessions.append(msg)
    if raw_mode and buf:
        sessions.append(_session(buf))
    results["sessions"] = len(sessions)
    results["alerts"] = int(sum(s["alerts"] for s in sessions))
    if raw_mode:
        results["cloud_latency_ns"] = latencies


def _raw_edge(stream, out_q, sent_bytes, ready) -> None:
    ready.wait()
    for i, (t_made, w) in enumerate(stream):
        _send(out_q, (i, t_made, w), sent_bytes)
    _send(out_q, STOP, sent_bytes)


class _Stream:
    """Replays windows at a steady pace (one every `interval_s`, from the first one); each
    window gets its 'made' time when it is released."""

    def __init__(self, windows: np.ndarray, interval_s: float = INTERVAL_S) -> None:
        self.windows, self.interval_ns = windows, int(interval_s * 1e9)

    def __iter__(self):
        start = time.perf_counter_ns()
        for i, w in enumerate(self.windows):
            wait = start + i * self.interval_ns - time.perf_counter_ns()
            if wait > 0:
                time.sleep(wait / 1e9)
            yield time.perf_counter_ns(), w


def run(windows: np.ndarray, rule: ReflexRule, classify, daily_rows, daily_model,
        mode: str, session_windows: int = 12, interval_s: float = INTERVAL_S) -> dict:
    """Run one mode end to end in separate processes and return its measurements."""
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager()
    results = mgr.dict()
    b_edge, b_fog = ctx.Value("q", 0), ctx.Value("q", 0)
    q1, q2 = ctx.Queue(), ctx.Queue()
    stream = _Stream(windows, interval_s)
    warm = windows[:1] if len(windows) else None
    t0 = time.perf_counter()
    if mode == "tiered":
        ready = ctx.Barrier(3)
        procs = [ctx.Process(target=edge, args=(stream, rule, classify, q1, b_edge, ready)),
                 ctx.Process(target=fog, args=(q1, q2, b_fog, session_windows, results, ready)),
                 ctx.Process(target=cloud, args=(q2, daily_rows, daily_model, results, False,
                                                 rule, classify, session_windows, ready, warm))]
    elif mode == "cloud_only":
        ready = ctx.Barrier(2)
        procs = [ctx.Process(target=_raw_edge, args=(stream, q2, b_edge, ready)),
                 ctx.Process(target=cloud, args=(q2, daily_rows, daily_model, results, True,
                                                 rule, classify, session_windows, ready, warm))]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0
    res = dict(results)
    lat = np.array(res.get("fog_latency_ns" if mode == "tiered" else "cloud_latency_ns", [0]))
    mgr.shutdown()
    return {"mode": mode, "windows": int(len(windows)), "send_interval_s": interval_s,
            "wall_s": wall,
            "bytes_from_device": int(b_edge.value), "bytes_fog_to_cloud": int(b_fog.value),
            "bytes_into_cloud": int(b_fog.value if mode == "tiered" else b_edge.value),
            "decision_latency_ms_median": float(np.median(lat) / 1e6),
            "decision_latency_ms_p95": float(np.percentile(lat, 95) / 1e6),
            "sessions": res.get("sessions"), "alerts": res.get("alerts"),
            "daily_days_scored": len(res.get("daily_risk", []))}
