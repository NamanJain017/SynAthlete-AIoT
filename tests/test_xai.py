"""Piece-8 explanation code on small made-up data (CPU, seconds)."""

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from src.models import nets
from src.xai import checks, explain

CPU = torch.device("cpu")


def toy(n=200, f=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float32)
    y = (X[:, 0] + 0.1 * rng.normal(size=n) > 0.8).astype(int)  # only feature 0 matters
    return X, y


def test_tree_shap_finds_the_feature_that_matters_and_deletion_passes():
    X, y = toy()
    rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    sv = explain.tree_shap(rf, X)
    assert sv.shape == X.shape and np.abs(sv).mean(0).argmax() == 0
    d = checks.deletion(lambda Z: rf.predict_proba(Z)[:, 1], X[y == 1], sv[y == 1], k=1)
    assert d["passed"] and d["drop_top"] > d["drop_random"]


def test_randomisation_and_stability():
    X, y = toy()
    rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    sv = explain.tree_shap(rf, X)
    shuffled = RandomForestClassifier(n_estimators=50, random_state=0).fit(
        X, np.random.default_rng(1).permutation(y))
    assert checks.randomisation(sv, explain.tree_shap(shuffled, X))["rank_correlation"] < 0.9
    st = checks.stability(lambda Z: explain.tree_shap(rf, Z), X[:50], sv[:50])
    assert st["rank_correlation"] > 0.5


def test_integrated_gradients_and_attention_shapes():
    nets.seed_everything(0)
    X = np.random.default_rng(0).normal(size=(8, 7, 3)).astype(np.float32)
    gru = nets.GRUNet(3, 1, hidden=4, layers=1, dropout=0.0)
    assert explain.integrated_gradients(gru, X, CPU, steps=4).shape == X.shape
    attn = explain.AttentionGRU(3, 1, hidden=4, layers=1, dropout=0.0)
    w = explain.attention_weights(attn, X, CPU)
    assert w.shape == (8, 7) and np.allclose(w.sum(1), 1.0, atol=1e-5)


def test_grad_cam_shape_and_range():
    nets.seed_everything(0)
    cnn = nets.CNN1D(3, 4, channels=8, kernel=3, dropout=0.0)
    cam = explain.grad_cam_1d(cnn, np.random.default_rng(0).normal(size=(5, 16, 3)), CPU)
    assert cam.shape == (5, 16) and cam.min() >= 0 and cam.max() <= 1 + 1e-6


def test_reason_sentence():
    s = explain.reason_sentence(["total kms [w-1]", "avg recovery [w-1]"],
                                np.array([1.2, -0.8]), np.array([0.3, 0.1]))
    assert s == ("Risk is higher mainly because total kms [w-1] was higher than usual; "
                 "avg recovery [w-1] was lower than usual.")


def test_explain_round_resumes_from_its_checkpoint(tmp_path):
    import json

    from src.xai import run
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    (ck / "explain_r3.json").write_text(json.dumps({"round": 3, "saved": True}))
    # no data or model is touched: the finished round is read back as it was saved
    assert run.explain_round(3, "no-such-fused-run", str(tmp_path)) == {"round": 3, "saved": True}
