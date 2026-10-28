"""Track C (exploratory, D-018, D-027(6)): python -m src.models.track_c.

SoccerMon and PMData: leave-one-person-out, predictions pooled, fixed default settings (no
tuning: there is no validation person), threshold-free metrics only (EVAL_PROTOCOL Track C).
- Event rows: episode starts (positive) and healthy events (negative), labelled in piece 2;
  inputs are that day's columns of the full daily table (features were built from all days,
  never from the selected rows); no report, illness or label column is an input.
- Imputation (median) and scaling (z-score) are fitted on the training people of each fold.
- SoccerMon is also scored within its main team (TeamA), with a team-only baseline (each
  team's training injury rate as the score); PMData with and without p12.
Never main injury evidence (D-018). Writes experiments/track_c/<run_id>/.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data import loaders
from src.eval import metrics
from src.models import run as base_run

OUT = loaders.ROOT / "experiments" / "track_c"
NOT_INPUTS = {"pain_report", "illness_report", "episode_start", "healthy_event"}
MODELS = {
    "logistic_regression": lambda seed: LogisticRegression(max_iter=2000),
    "random_forest": lambda seed: RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                                         random_state=seed),
}


def events(name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    t = pd.read_parquet(loaders.PROCESSED / name / "daily.parquet")
    e = t[t["episode_start"] | t["healthy_event"]]
    X = e[[c for c in t.columns if c not in NOT_INPUTS]].astype(float)
    return X, e["episode_start"].to_numpy(int), e.index.get_level_values("person").to_numpy()


def lopo(X: pd.DataFrame, y: np.ndarray, person: np.ndarray, model, seed: int = 0) -> np.ndarray:
    """Pooled leave-one-person-out scores; median imputation + z-score from training people."""
    scores = np.full(len(y), np.nan)
    for p in np.unique(person):
        test, train = person == p, person != p
        if y[train].sum() == 0:
            continue
        med = X[train].median()
        Xtr, Xte = X[train].fillna(med).fillna(0.0), X[test].fillna(med).fillna(0.0)
        mu, sd = Xtr.mean(), Xtr.std().replace(0, 1.0).fillna(1.0)
        m = model(seed).fit((Xtr - mu) / sd, y[train])
        scores[test] = m.predict_proba((Xte - mu) / sd)[:, 1]
    return scores


def score(y, s) -> dict:
    keep = ~np.isnan(s)
    return {"events": int(keep.sum()), "positives": int(y[keep].sum()),
            "chance_pr_auc": float(y[keep].mean()),
            "pr_auc": metrics.average_precision(y[keep], s[keep]),
            "roc_auc": metrics.roc_auc(y[keep], s[keep])}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    code = base_run._code_state()
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = OUT / run_id
    folder.mkdir(parents=True)
    res = {"label": "exploratory (D-018): never main injury evidence", "soccermon": {},
           "pmdata": {}}
    X, y, person = events("soccermon")
    team = np.array([p.split("-")[0] for p in person])
    for name, model in MODELS.items():
        s = lopo(X, y, person, model)
        res["soccermon"][name] = {"all": score(y, s),
                                  "TeamA_only": score(y[team == "TeamA"], s[team == "TeamA"])}
    # team-only baseline: the person's team injury rate among the OTHER (training) people
    base = np.empty(len(y))
    for p in np.unique(person):
        rows = person == p
        base[rows] = y[(team == team[rows][0]) & ~rows].mean()
    res["soccermon"]["team_only_baseline"] = score(y, base)
    X, y, person = events("pmdata")
    for name, model in MODELS.items():
        no12 = person != "p12"
        res["pmdata"][name] = {"with_p12": score(y, lopo(X, y, person, model)),
                               "without_p12": score(y[no12], lopo(X[no12], y[no12],
                                                                  person[no12], model))}
    info = {**code, "data_fingerprints": base_run._fingerprints()}
    (folder / "run_info.json").write_text(json.dumps(info, indent=2), "utf-8")
    (folder / "metrics.json").write_text(json.dumps(res, indent=2), "utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
