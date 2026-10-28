"""Piece-3 round builder, tuning, baselines and scoring (src/models, written by Claude Code).

The leak checks live here: scaler sees training rows only, people never cross roles, declared
inputs only, locked runners refused. Real-data checks skip when the private data is absent (CI).
"""

import math

import numpy as np
import pytest

from src.data import loaders
from src.models import baselines, data, evaluate, tuning

HAS_LOVDAL = (loaders.PROCESSED / "lovdal" / "day.npz").exists()
HAS_PAMAP2 = (loaders.PROCESSED / "pamap2" / "windows.npz").exists()


def runner_data(kind="week", n=6, steps=3, features=("total kms", "nr. sessions")):
    X = np.arange(n * steps * len(features), dtype=np.float32).reshape(n, steps, len(features))
    return loaders.RunnerData(kind=kind, X=X, y=np.zeros(n, dtype=np.int8),
                              athlete_id=np.arange(n, dtype=np.int16),
                              date=np.zeros(n, dtype=np.int32), feature_names=list(features))


FOLD = {"train_ids": [0, 1, 2], "val_ids": [3], "test_ids": [4, 5]}


# ---------------------------------------------------------------- inputs

def test_flatten_matches_input_names():
    d = runner_data(kind="day", steps=3)
    d.X[:] = 0
    for t in range(3):
        for f in range(2):
            d.X[:, t, f] = 10 * t + f
    names, flat = data.input_names(d), data.flatten(d)
    assert names == ["total kms [d-3]", "nr. sessions [d-3]", "total kms [d-2]",
                     "nr. sessions [d-2]", "total kms [d-1]", "nr. sessions [d-1]"]
    assert flat[0].tolist() == [0, 1, 10, 11, 20, 21]


def test_week_flatten_appends_relative_changes():
    d = runner_data()
    d.extra = np.ones((6, 3), dtype=np.float32)
    d.extra_names = ("rel a", "rel b", "rel c")
    assert data.flatten(d).shape == (6, 9)
    assert data.input_names(d)[-3:] == ["rel a", "rel b", "rel c"]


def test_declared_inputs_guard():
    data.check_declared_inputs([f"x{i}" for i in range(70)], "day")
    with pytest.raises(ValueError):
        data.check_declared_inputs([f"x{i}" for i in range(71)], "day")
    for bad in ("Athlete ID", "Date", "own events 21 days", "gap to previous", "history length"):
        with pytest.raises(ValueError):
            data.check_declared_inputs([f"x{i}" for i in range(68)] + [bad], "week")


# ---------------------------------------------------------------- people and scaling

def test_split_rows_refuses_mixing_and_locked_people():
    person = np.arange(6)
    masks = data.split_rows(person, FOLD)
    assert [int(m.sum()) for m in masks.values()] == [3, 1, 2]
    with pytest.raises(ValueError):
        data.split_rows(person, {**FOLD, "val_ids": [3, 4]})      # 4 also in test
    with pytest.raises(ValueError):
        data.split_rows(person, FOLD, locked=[5])                  # locked runner present
    with pytest.raises(ValueError):
        data.split_rows(np.arange(7), FOLD)                        # person 6 has no role


def test_scaler_sees_training_rows_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 4)).astype(np.float32)
    y, person = np.array([0, 1, 0, 1, 0, 1]), np.arange(6)
    a = data.make_round(0, "week", X, y, person, FOLD, ["a", "b", "c", "d"])
    X2 = X.copy()
    X2[3:] += 1000.0  # change validation and test people only
    b = data.make_round(0, "week", X2, y, person, FOLD, ["a", "b", "c", "d"])
    assert np.array_equal(a.scaler_mean, b.scaler_mean) and np.array_equal(a.train.X, b.train.X)
    assert np.allclose(a.train.X.mean(axis=0), 0, atol=1e-6)
    assert set(a.train.person) == {0, 1, 2} and set(a.test.person) == {4, 5}


def test_sequence_scaler_is_per_channel_and_constant_channels_survive():
    X = np.zeros((6, 5, 2), dtype=np.float32)
    X[:, :, 0] = np.arange(30).reshape(6, 5)
    mean, std = data.fit_scaler(X[:3])
    assert mean.shape == (2,) and std[1] == 1.0  # constant channel: SD 1, no division by zero
    assert np.isclose(mean[0], np.arange(15).mean())


# ---------------------------------------------------------------- tuning

GRID = {"C": {"loguniform": [1e-4, 1e2]}, "class_weight": [None, "balanced"]}


def test_draw_settings_is_seeded_and_in_range():
    a = tuning.draw_settings(GRID, 8, seed=1000, fixed={"max_iter": 2000})
    assert a == tuning.draw_settings(GRID, 8, seed=1000, fixed={"max_iter": 2000})
    assert a != tuning.draw_settings(GRID, 8, seed=1001, fixed={"max_iter": 2000})
    assert all(1e-4 <= s["C"] <= 1e2 and s["class_weight"] in (None, "balanced")
               and s["max_iter"] == 2000 for s in a)
    reordered = {"class_weight": GRID["class_weight"], "C": GRID["C"]}
    assert tuning.draw_settings(reordered, 8, seed=1000) == tuning.draw_settings(GRID, 8, seed=1000)
    with pytest.raises(ValueError):
        tuning.draw_settings({"C": []}, 1, seed=0)


def test_choose_uses_ap_then_auc_then_order():
    trials = [{"val_ap": 0.2, "val_auc": 0.6}, {"val_ap": 0.3, "val_auc": 0.5},
              {"val_ap": 0.3, "val_auc": 0.7}, {"val_ap": 0.3, "val_auc": 0.7}]
    assert tuning.choose(trials) == 2
    assert tuning.choose([{"val_ap": 0.1, "val_auc": None}, {"val_ap": 0.1, "val_auc": 0.5}]) == 1


# ---------------------------------------------------------------- baselines and scoring

def test_acwr_ratio_with_the_1_km_floor():
    d = runner_data(n=3)
    d.X[:, :, 0] = [[10, 30, 40], [0, 0, 5], [0, 1, 3]]  # oldest -> newest week km
    assert baselines.acwr_ratio(d).tolist() == [2.0, 5.0, 3.0]


def test_shuffled_labels_keep_the_counts():
    y = np.array([0] * 90 + [1] * 10)
    s = baselines.shuffled_labels(y, 2000)
    assert s.sum() == 10 and np.array_equal(s, baselines.shuffled_labels(y, 2000))
    assert not np.array_equal(s, y)


def test_constant_risk_raises_no_alerts():
    rng = np.random.default_rng(0)
    y_val, y_test = (rng.random(200) < 0.1).astype(int), (rng.random(300) < 0.1).astype(int)
    y_val[:2], y_test[:2] = [0, 1], [0, 1]
    s_val, s_test = baselines.constant_risk(y_val, 200), baselines.constant_risk(y_val, 300)
    out = evaluate.injury_scores(y_val, s_val, y_test, s_test, probability=True)
    assert out["alert_nobody"] is True and out["alert"] is None and out["alert_threshold"] is None
    assert out["roc_auc"] == 0.5 and out["pr_auc"] == pytest.approx(y_test.mean())
    assert out["platt"]["slope_flag"] is True


def test_injury_scores_for_a_ranking_score():
    rng = np.random.default_rng(1)
    y_val, y_test = (rng.random(300) < 0.2).astype(int), (rng.random(300) < 0.2).astype(int)
    s_val, s_test = y_val + rng.normal(size=300) * 2, y_test + rng.normal(size=300) * 2
    out = evaluate.injury_scores(y_val, s_val, y_test, s_test, probability=False)
    assert out["calibration_raw"] is None and out["alert"]["alerts"] > 0
    assert 0 < out["calibration_platt"]["brier"] < 1 and len(out["net_benefit"]["model"]) == 20
    assert out["net_benefit"]["risks"][0] == 0.005 and out["net_benefit"]["risks"][-1] == 0.1


def test_summarise():
    s = evaluate.summarise([{"pr_auc": 0.2, "roc_auc": None}, {"pr_auc": 0.4, "roc_auc": 0.6}])
    assert s["pr_auc"]["mean"] == pytest.approx(0.3) and s["roc_auc"] is None
    assert s["pr_auc"]["sd"] == pytest.approx(math.sqrt(0.02))


# ---------------------------------------------------------------- real data (skipped in CI)

@pytest.mark.skipif(not HAS_LOVDAL, reason="private Lövdal data not present")
@pytest.mark.parametrize("kind,layout,width", [("day", "flat", 70), ("week", "flat", 69),
                                               ("day", "sequence", 10)])
def test_real_lovdal_rounds(kind, layout, width):
    split = loaders.lovdal_split()
    rounds = list(data.lovdal_rounds(kind, layout))
    assert len(rounds) == 5
    for r in rounds:
        fold = split["folds"][str(r.k)]
        assert r.train.X.shape[-1] == width and r.train.X.dtype == np.float32
        for part, role in ((r.train, "train"), (r.val, "val"), (r.test, "test")):
            assert set(part.person.tolist()) == set(fold[f"{role}_ids"])
            assert not np.isin(part.person, split["lockbox_ids"]).any()
        assert np.allclose(r.train.X.reshape(-1, width).mean(axis=0), 0, atol=1e-3)


@pytest.mark.skipif(not HAS_LOVDAL, reason="private Lövdal data not present")
def test_week_file_has_no_sequence_layout():
    with pytest.raises(ValueError):
        next(data.lovdal_rounds("week", "sequence"))


@pytest.mark.skipif(not HAS_PAMAP2, reason="private PAMAP2 data not present")
def test_real_pamap2_rounds():
    rounds = list(data.pamap2_rounds())
    assert len(rounds) == 4
    for r in rounds:
        assert r.train.X.shape[1:] == (128, 10)
        assert "109" in set(r.train.person.tolist())  # always in training (D-020)
        assert not set(r.train.person) & (set(r.val.person) | set(r.test.person))


# ---------------------------------------------------------------- Lövdal method and seeds

def test_athlete_balanced_bag_counts():
    from src.models import lovdal_method
    y = np.array([0, 0, 0, 1] * 3 + [0, 0, 0, 0])          # athletes 0-2 have both classes
    athlete = np.repeat([0, 1, 2, 3], 4)
    idx = lovdal_method.athlete_balanced_bag(y, athlete, np.random.default_rng(0), 30)
    assert len(idx) == 3 * 2 * 10                          # floor(30 / 3) per class per athlete
    assert y[idx].sum() == 30 and 3 not in set(athlete[idx])


def test_paper_scaling_uses_each_athletes_healthy_rows():
    from src.models import lovdal_method
    X = np.array([[1.0], [3.0], [100.0], [10.0], [14.0], [12.0]])
    y = np.array([0, 0, 1, 0, 0, 1])
    athlete = np.array([0, 0, 0, 1, 1, 1])
    z = lovdal_method.paper_scaling(X, y, athlete)
    sd0, sd1 = np.std([1, 3], ddof=1), np.std([10, 14], ddof=1)
    assert np.allclose(z[:, 0], [(1 - 2) / sd0, (3 - 2) / sd0, (100 - 2) / sd0,
                                 (10 - 12) / sd1, (14 - 12) / sd1, 0.0])


def test_mean_over_seeds():
    base = {"test_events": 10, "test_injuries": 2, "test_injury_rate": 0.2, "val_pr_auc": 0.3,
            "val_roc_auc": 0.6, "alert": None,
            "calibration_platt": {"brier": 0.1, "ece": 0.0, "slope": 1.0, "intercept": 0.0}}
    out = evaluate.mean_over_seeds([{**base, "pr_auc": 0.2, "roc_auc": 0.6},
                                    {**base, "pr_auc": 0.4, "roc_auc": None}])
    assert out["pr_auc"] == pytest.approx(0.3) and out["roc_auc"] is None and out["alert"] is None


# ---------------------------------------------------------------- fused model (piece 7)

def test_oof_predictions_never_score_a_row_with_a_model_that_trained_on_it():
    from src.models import fused
    person = np.repeat(np.arange(10), 4)
    X = np.arange(40, dtype=np.float32).reshape(40, 1)
    r = data.Round(0, "week", data.Part(X, np.tile([0, 1], 20), person),
                   data.Part(X[:2], np.array([0, 1]), np.array([99, 99])),
                   data.Part(X[:2], np.array([0, 1]), np.array([98, 98])), ["x"],
                   np.zeros(1), np.ones(1))
    seen = []

    def fit(sub, setting, seed):
        trained = set(sub.train.person.tolist())
        seen.append(trained)
        # the "prediction" says whether the scored row's runner was in this model's training
        return lambda Xq: np.isin(person[Xq[:, 0].astype(int)], list(trained)).astype(float)

    oof = fused.oof_predictions(fit, r, {}, 0)
    assert len(seen) == 5 and not np.isnan(oof).any()
    assert oof.sum() == 0  # no training row was scored by a model that saw its runner


def test_risk_score_is_a_validation_percentile():
    from src.models import fused
    p_val = np.array([0.01, 0.02, 0.03, 0.04])
    assert fused.risk_score(p_val, np.array([0.005, 0.025, 0.05])).tolist() == [0.0, 50.0, 100.0]


def test_paired_seed_mean_difference():
    from src.eval import bootstrap
    from src.eval.metrics import roc_auc
    rng = np.random.default_rng(0)
    person = np.repeat(np.arange(12), 10)
    group = person % 3
    y = (rng.random(120) < 0.3).astype(int)
    a, b = y + rng.normal(size=120), rng.normal(size=120)
    one = bootstrap.paired_seed_mean_difference(roc_auc, y, [a], [b], person, group, 200, 0)
    ref = bootstrap.paired_bootstrap_difference(roc_auc, y, a, b, person, group, 200, 0)
    assert one == pytest.approx(ref)
    same = bootstrap.paired_seed_mean_difference(roc_auc, y, [a, 2 * a], [a], person, group, 50)
    assert same["estimate"] == pytest.approx(0.0) and same["low"] == pytest.approx(0.0)
