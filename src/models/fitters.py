"""One fit function per tuned model: fit(round, setting, seed) -> predict(X) -> scores.

Each fit sees ONE round's training rows (and its validation rows only where D-021 allows:
early stopping). The runner chooses settings on validation and scores test once.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from src.models import lovdal_method, nets
from src.models.data import Round

Predict = Callable[[np.ndarray], np.ndarray]


def random_forest(r: Round, s: dict, seed: int) -> Predict:
    keys = ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "class_weight")
    model = RandomForestClassifier(**{k: s[k] for k in keys}, n_jobs=-1, random_state=seed)
    model.fit(r.train.X, r.train.y)
    return lambda X: model.predict_proba(X)[:, 1]


def catboost(r: Round, s: dict, seed: int) -> Predict:
    model = CatBoostClassifier(iterations=s["iterations"], learning_rate=s["learning_rate"],
                               depth=s["depth"], l2_leaf_reg=s["l2_leaf_reg"],
                               auto_class_weights=s["auto_class_weights"],
                               eval_metric=s["eval_metric"], random_seed=seed, thread_count=-1,
                               verbose=0, allow_writing_files=False)
    # early stopping on this round's validation runners (D-021 allows it)
    model.fit(r.train.X, r.train.y, eval_set=(r.val.X, r.val.y),
              early_stopping_rounds=s["early_stopping_rounds"], use_best_model=True)
    return lambda X: model.predict_proba(X)[:, 1]


def sequence(name: str) -> Callable[[Round, dict, int], Predict]:
    """Injury sequence network (gru, lstm or tcn) on day sequences (7 days x 10 features)."""
    return lambda r, s, seed: _sequence(name, r, s, seed)


def gru(r: Round, s: dict, seed: int) -> Predict:
    """Injury GRU on the day file; early stopping on validation PR-AUC."""
    return _sequence("gru", r, s, seed)


def _sequence(name: str, r: Round, s: dict, seed: int) -> Predict:
    device = nets.pick_device()
    nets.seed_everything(seed)
    model = nets.build_model(name, r.train.X.shape[2], 1, s)
    res = nets.train_model(model, "binary", r.train.X, r.train.y, r.val.X, r.val.y, lr=s["lr"],
                           batch_size=s["batch_size"], max_epochs=s["max_epochs"],
                           patience=s["patience"], seed=seed, weight_decay=s["weight_decay"],
                           pos_weight=nets.positive_weight(r.train.y, s["pos_weight"]),
                           device=device)
    return lambda X: nets.predict_scores(res["model"], X, device=device)


def lovdal_label_free(r: Round, s: dict, seed: int) -> Predict:
    """Paper method without its peek: our training-rows scaling (round builder) instead of the
    label-reading per-athlete scaling; Platt and alert level on validation like every model."""
    models = lovdal_method.train_bags(r.train.X, r.train.y, r.train.person, seed, s)
    return lambda X: lovdal_method.bag_mean(models, X)


def pamap2_net(name: str) -> Callable[[Round, dict, int], tuple[Callable, float]]:
    """PAMAP2 activity network; returns (predict_labels, validation macro-F1)."""
    def fit(r: Round, s: dict, seed: int) -> tuple[Callable, float]:
        device = nets.pick_device()
        nets.seed_everything(seed)
        classes = np.unique(r.train.y)
        model = nets.build_model(name, r.train.X.shape[2], len(classes), s)
        res = nets.train_model(model, "multiclass", r.train.X, r.train.y, r.val.X, r.val.y,
                               lr=s["lr"], batch_size=s["batch_size"],
                               max_epochs=s["max_epochs"], patience=s["patience"], seed=seed,
                               device=device)
        return (lambda X: nets.predict_labels(res["model"], X, res["classes"], device=device),
                res["best_score"])
    return fit
