"""Piece-9 dashboard logic (the Streamlit pages themselves are checked by running them)."""

import pytest

from src.dashboard import lab
from src.data import loaders

HAS_LOVDAL = (loaders.PROCESSED / "lovdal" / "day.npz").exists()


def test_training_lab_refuses_settings_outside_safe_ranges():
    lab.check("logistic_regression", {"C": 1.0, "class_weight": None})
    lab.check("gru", {"hidden": 32, "layers": 2, "dropout": 0.2, "lr": 1e-3})
    for model, s in [("logistic_regression", {"C": 1e6, "class_weight": None}),
                     ("random_forest", {"n_estimators": 5000, "max_depth": 4,
                                        "min_samples_leaf": 1, "max_features": "sqrt",
                                        "class_weight": None}),
                     ("gru", {"hidden": 32, "layers": 3, "dropout": 0.2, "lr": 1e-3}),
                     ("transformer", {})]:
        with pytest.raises((ValueError, KeyError)):
            lab.check(model, s)


@pytest.mark.skipif(not HAS_LOVDAL, reason="private Lövdal data not present")
def test_lab_scores_on_validation_only_and_saves_a_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "OUT", tmp_path)
    res = lab.train("logistic_regression", {"C": 1.0, "class_weight": None})
    assert res["scored_on"].startswith("validation") and 0 < res["val_roc_auc"] < 1
    assert list(tmp_path.glob("*/demo_result.json"))
