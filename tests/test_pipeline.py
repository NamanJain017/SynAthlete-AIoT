"""Piece-6 tiers on a small made-up stream: both modes process every window, tiered sends
far fewer bytes into the cloud, and alerts fire on a heart-rate spike."""

import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.pipeline import bench, tiers


def test_tiered_and_cloud_only_process_every_window():
    rng = np.random.default_rng(0)
    windows = rng.normal(100, 5, (24, 128, 4)).astype(np.float32)
    windows[5, 10, 0] = 200.0  # one heart-rate spike above 180 bpm
    rows = rng.normal(size=(6, 3))
    model = bench.DailyModel(LogisticRegression().fit(rows, [0, 1, 0, 1, 0, 1]))
    rule = tiers.ReflexRule(hr_channel=0, acc_channels=(1, 2, 3), acc_spike=1e9)
    t = tiers.run(windows, rule, None, rows, model, "tiered", session_windows=6)
    c = tiers.run(windows, rule, None, rows, model, "cloud_only", session_windows=6)
    for m in (t, c):
        assert m["windows"] == 24 and m["sessions"] == 4 and m["alerts"] == 1
        assert m["daily_days_scored"] == 6
    assert c["bytes_into_cloud"] > 20 * t["bytes_into_cloud"]


class SlowDailyModel:
    """A cloud model that needs a second to load, like a real one."""

    def __call__(self, rows):
        time.sleep(1.0)
        return np.zeros(len(rows))


def test_latency_excludes_start_up_and_follows_the_pace():
    """Windows wait until every tier is ready (D-034): a slow cloud start-up must not count as
    decision time (it was about 1,000 ms here before the fix), and sending keeps its pace."""
    windows = np.random.default_rng(1).normal(100, 5, (10, 128, 4)).astype(np.float32)
    m = tiers.run(windows, tiers.ReflexRule(hr_channel=0), None, np.zeros((3, 2)),
                  SlowDailyModel(), "cloud_only", session_windows=5, interval_s=0.05)
    assert m["decision_latency_ms_median"] < 200
    assert m["wall_s"] >= 9 * 0.05 and m["send_interval_s"] == 0.05
