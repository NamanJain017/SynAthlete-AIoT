"""All piece-3 numbers for one model in one round (D-022, D-024, D-025).

Validation rows choose the alert level and fit Platt; test rows are only scored, once.
"""

from __future__ import annotations

import math

import numpy as np

from src.eval import metrics

# Decision-curve risks 0.5 % to 10 % in steps of 0.5 % (exploratory, EVAL_PROTOCOL).
RISKS = [round(0.005 * i, 3) for i in range(1, 21)]


def _calibration(y: np.ndarray, prob: np.ndarray) -> dict:
    slope, intercept = metrics.calibration_slope_intercept(y, prob)
    return {"brier": metrics.brier(y, prob), "ece": metrics.ece_equal_count(y, prob),
            "slope": slope, "intercept": intercept}


def injury_scores(y_val: np.ndarray, s_val: np.ndarray, y_test: np.ndarray, s_test: np.ndarray,
                  probability: bool) -> dict:
    """Every reported number for one model in one round.

    probability: True if the scores are probabilities (raw calibration is then reported, and
    Platt is fitted on logit(p)); False for ranking scores such as the ACWR ratio.
    """
    threshold = metrics.choose_alert_threshold(y_val, s_val)
    nobody = math.isinf(threshold)
    platt_in = metrics.logit if probability else np.asarray
    params = metrics.fit_platt(platt_in(s_val), y_val)
    p_test = metrics.apply_platt(platt_in(s_test), params)
    return {
        "test_events": int(len(y_test)),
        "test_injuries": int(np.sum(y_test)),
        "test_injury_rate": float(np.mean(y_test)),  # the chance level of PR-AUC
        "pr_auc": metrics.average_precision(y_test, s_test),
        "roc_auc": metrics.roc_auc(y_test, s_test),
        "val_pr_auc": metrics.average_precision(y_val, s_val),
        "val_roc_auc": metrics.roc_auc(y_val, s_val),
        "alert_nobody": nobody,
        "alert_threshold": None if nobody else threshold,
        # "alert nobody" (e.g. the constant-risk model): no alerts, so alert numbers are n/a (D-024)
        "alert": None if nobody else metrics.alert_numbers(y_test, s_test, threshold),
        "platt": {"slope": params[0], "intercept": params[1],
                  "slope_flag": metrics.platt_slope_flag(params)},
        "calibration_raw": _calibration(y_test, s_test) if probability else None,
        "calibration_platt": _calibration(y_test, p_test),
        "net_benefit": {
            "risks": RISKS,
            "model": [float(v) for v in metrics.net_benefit(y_test, p_test, RISKS)],
            "treat_all": [float(v) for v in metrics.net_benefit_treat_all(y_test, RISKS)],
        },
    }


def mean_over_seeds(per_seed: list[dict]) -> dict:
    """One round's numbers averaged over seeds (D-024); a value is None if any seed lacks it."""
    def mean(values):
        return None if any(v is None for v in values) else float(np.mean(values))

    out = {k: per_seed[0][k] for k in ("test_events", "test_injuries", "test_injury_rate")}
    for k in ("pr_auc", "roc_auc", "val_pr_auc", "val_roc_auc"):
        out[k] = mean([s[k] for s in per_seed])
    out["calibration_platt"] = {k: mean([s["calibration_platt"][k] for s in per_seed])
                                for k in ("brier", "ece", "slope", "intercept")}
    alerts = [s["alert"] for s in per_seed]
    out["alert"] = None if any(a is None for a in alerts) else {
        k: mean([a[k] for a in alerts]) for k in ("sensitivity", "specificity", "precision",
                                                  "false_alarms_per_caught_injury",
                                                  "alerts_per_100_events", "mcc")}
    return out


def summarise(rounds: list[dict], keys: tuple[str, ...] = ("pr_auc", "roc_auc")) -> dict:
    """Mean and SD (ddof 1) over rounds for the headline numbers; None if any round lacks one."""
    out = {}
    for key in keys:
        values = [r[key] for r in rounds]
        if any(v is None for v in values):
            out[key] = None
            continue
        out[key] = {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))
                    if len(values) > 1 else None, "per_round": [float(v) for v in values]}
    return out
