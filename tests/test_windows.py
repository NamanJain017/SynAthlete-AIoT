"""Tests for the sensor window helpers (piece 2, Track B), on synthetic signals."""

import numpy as np
from scipy.signal import decimate, resample_poly

from src.data import windows


def test_window_count_formula():
    for n in (127, 128, 191, 192, 300, 1000):
        expected = 0 if n < 128 else (n - 128) // 64 + 1
        assert len(windows.window_starts(n, 128, 64)) == expected


def test_short_gaps_filled_long_and_edge_gaps_invalid():
    x = np.array([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan, 10.0, np.nan])
    filled, ok = windows.fill_short_gaps(x, max_gap=3)
    assert np.allclose(filled[:4], [1, 2, 3, 4])  # 2-sample gap interpolated
    assert ok[:4].all()
    assert not ok[4:9].any()  # 5-sample gap: too long
    assert not ok[10]  # gap touching the end


def test_dilate_marks_neighbours():
    m = np.zeros(10, dtype=bool)
    m[5] = True
    assert windows.dilate(m, 2).tolist() == [False] * 3 + [True] * 5 + [False] * 2


def test_pure_windows_drop_mixed_labels_invalid_samples_and_dropped_labels():
    labels = np.array([1] * 200 + [2] * 200 + [0] * 200)
    valid = np.ones(600, dtype=bool)
    valid[450] = False
    starts = windows.pure_window_starts(labels, valid, window=128, stride=64, margin=0,
                                        keep={1, 2})
    # [0,128) and [64,192) are all 1; [128,256) and [192,320) mix 1/2; [256,384) is all 2;
    # later windows mix 2/0, contain label 0 or the invalid sample 450.
    assert starts.tolist() == [0, 64, 256]
    with_margin = windows.pure_window_starts(labels, valid, 128, 64, margin=10, keep={1, 2})
    # [64,192) plus 10 steps reaches index 201, past the 1 -> 2 change at 200: dropped.
    assert with_margin.tolist() == [0, 256]


def test_resampled_lengths():
    x = np.sin(np.linspace(0, 20, 4000))
    assert len(decimate(x, 4, ftype="fir", zero_phase=True)) == 1000  # 100 Hz -> 25 Hz
    assert len(resample_poly(np.ones(640), 25, 64)) == 250  # 64 Hz -> 25 Hz


def test_cut_shapes():
    sig = np.arange(1000 * 3, dtype=float).reshape(1000, 3)
    X = windows.cut(sig, np.array([0, 64]), 128)
    assert X.shape == (2, 128, 3) and X.dtype == np.float32
    assert X[1, 0, 0] == sig[64, 0]


def test_nearest_index_hits_the_same_instant_despite_float_error():
    k = np.arange(200_000)
    idx = windows.nearest_index(k / 25, 700, 10**9)
    assert np.array_equal(idx, 28 * k)  # 700 / 25 = 28 label samples per 25 Hz step
    assert windows.nearest_index(np.array([10.0]), 700, 50)[0] == 49  # clipped to the end
