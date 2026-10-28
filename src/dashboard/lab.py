"""Training lab (D-016): retrain a fast model with changed settings inside safe ranges, on ONE
development group (round 0), scored on that group's VALIDATION runners only.

Runs are saved to experiments/training_lab/<timestamp>/ labelled "demo": they never overwrite
the official models or results, never use the locked runners, and never look at test runners.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np

from src.eval import metrics
from src.models import baselines, data, fitters

OUT = data.loaders.ROOT / "experiments" / "training_lab"

# Safe ranges = the pre-declared piece-3 grids (configs/baselines/lovdal.yaml).
SAFE = {
    "logistic_regression": {"C": (1e-4, 1e2), "class_weight": (None, "balanced")},
    "random_forest": {"n_estimators": (100, 500), "max_depth": (4, 6, 8, 12, None),
                      "min_samples_leaf": (1, 50), "max_features": ("sqrt", 0.3, 0.5),
                      "class_weight": (None, "balanced_subsample")},
    "catboost": {"learning_rate": (0.01, 0.1), "depth": (4, 6, 8), "l2_leaf_reg": (1.0, 10.0),
                 "auto_class_weights": (None, "Balanced")},
    "gru": {"hidden": (16, 32, 64), "layers": (1, 2), "dropout": (0.0, 0.2, 0.4),
            "lr": (3e-4, 3e-3)},
}


def check(model: str, setting: dict) -> None:
    """ValueError for an unknown model or any value outside its safe range."""
    if model not in SAFE:
        raise ValueError(f"unknown model {model!r}")
    for key, allowed in SAFE[model].items():
        v = setting[key]
        numeric_range = (len(allowed) == 2 and all(isinstance(a, (int, float))
                                                   and not isinstance(a, bool) for a in allowed)
                         and key not in ("layers",))
        ok = (allowed[0] <= v <= allowed[1]) if numeric_range and v is not None else v in allowed
        if not ok:
            raise ValueError(f"{key}={v!r} is outside the safe range {allowed}")


def train(model: str, setting: dict) -> dict:
    check(model, setting)
    if model == "gru":
        r = next(data.lovdal_rounds("day", "sequence"))
        s = {**setting, "batch_size": 128, "max_epochs": 30, "patience": 5,
             "weight_decay": 0.0, "pos_weight": "balanced"}
        p_val = fitters.gru(r, s, 0)(r.val.X)
    else:
        r = next(data.lovdal_rounds("week"))
        if model == "logistic_regression":
            s = {**setting, "solver": "lbfgs", "max_iter": 2000}
            m = baselines.logistic_regression(r.train.X, r.train.y, s)
            p_val = m.predict_proba(r.val.X)[:, 1]
        elif model == "random_forest":
            p_val = fitters.random_forest(r, setting, 0)(r.val.X)
        else:
            s = {**setting, "iterations": 2000, "early_stopping_rounds": 100,
                 "eval_metric": "PRAUC"}
            p_val = fitters.catboost(r, s, 0)(r.val.X)
    result = {"label": "TRAINING LAB DEMO: not an official result (D-016)", "model": model,
              "setting": setting, "round": 0, "scored_on": "validation runners of round 0",
              "val_pr_auc": metrics.average_precision(r.val.y, p_val),
              "val_roc_auc": metrics.roc_auc(r.val.y, p_val),
              "val_injury_rate": float(np.mean(r.val.y))}
    folder = OUT / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "demo_result.json").write_text(json.dumps(result, indent=2, default=str), "utf-8")
    return result
