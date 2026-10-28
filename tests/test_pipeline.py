"""Piece-6 tiers on a small made-up stream: both modes process every window, tiered sends
far fewer bytes into the cloud, and alerts fire on a heart-rate spike."""

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
