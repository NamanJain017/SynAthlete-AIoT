"""Answer checker for task card WP-01 (docs/handoff/WP-01-metrics.md): src/eval/metrics.py.

Written by Claude Code before the code existed; builders must not edit this file.
Expected values are worked out by hand on tiny examples (shown in the comments) or come from
an independent reference (scikit-learn). Strengthened after the eval-auditor review of WP-01:
property tests that only an exact, unpenalised fit can pass, tie cases, invalid inputs (D-025).
The tests always run (the original "skip until built" switch was removed after the merge).
"""

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.eval import metrics

# Example A: 2 healthy (0) and 2 injury (1) events.
Y_A = [0, 0, 1, 1]
S_A = [0.1, 0.4, 0.35, 0.8]


def _is_plain(x) -> bool:
    return isinstance(x, (float, int)) and not isinstance(x, (np.floating, np.integer))


# ---------------------------------------------------------------- ranking scores


def test_average_precision_by_hand_and_against_sklearn():
    # Sorted by score: 0.8 (injury) -> precision 1 at recall 0.5; 0.4 (healthy); 0.35 (injury)
    # -> precision 2/3 at recall 1.  AP = 0.5 * 1 + 0.5 * 2/3 = 0.8333...
    ap = metrics.average_precision(Y_A, S_A)
    assert ap == pytest.approx(5 / 6)
    assert _is_plain(ap)
    rng = np.random.default_rng(0)
    y, s = rng.integers(0, 2, 500), rng.normal(size=500)
    assert metrics.average_precision(y, s) == pytest.approx(average_precision_score(y, s))


def test_average_precision_needs_an_injury():
    with pytest.raises(ValueError):
        metrics.average_precision([0, 0, 0], [0.1, 0.2, 0.3])


def test_roc_auc_by_hand_and_one_class():
    # Injury scores {0.35, 0.8} vs healthy {0.1, 0.4}: 3 of 4 pairs ranked right -> 0.75.
    assert metrics.roc_auc(Y_A, S_A) == pytest.approx(0.75)
    assert metrics.roc_auc([1, 0, 1], [0.5, 0.5, 0.5]) == pytest.approx(0.5)  # ties count half
    assert metrics.roc_auc([0, 0, 0], [0.1, 0.2, 0.3]) is None
    rng = np.random.default_rng(1)
    y, s = rng.integers(0, 2, 500), rng.normal(size=500)
    assert metrics.roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))


# ---------------------------------------------------------------- the alert level


def test_alert_threshold_smallest_gap():
    # t=0.1 (1, 0); t=0.35 (1, .5); t=0.4 (.5, .5) gap 0; t=0.8 (.5, 1); inf (0, 1) -> 0.4
    t = metrics.choose_alert_threshold(Y_A, S_A)
    assert t == pytest.approx(0.4)
    assert _is_plain(t)


def test_alert_threshold_tie_goes_to_larger_sum_even_if_lower():
    # (sensitivity, specificity): t=0.8 -> (.5, .75) gap .25 sum 1.25;
    #                             t=0.7 -> (1, .75)  gap .25 sum 1.75  -> pick 0.7 (lower!)
    y = [1, 0, 1, 0, 0, 0]
    s = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2]
    assert metrics.choose_alert_threshold(y, s) == pytest.approx(0.7)


def test_constant_score_alerts_nobody():
    # Candidates 0.5 -> (1, 0) and inf -> (0, 1): same gap, same sum -> the higher one: inf.
    assert metrics.choose_alert_threshold([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == math.inf


def test_alert_threshold_needs_both_classes():
    with pytest.raises(ValueError):
        metrics.choose_alert_threshold([0, 0, 0], [0.1, 0.2, 0.3])


def test_alert_numbers_by_hand():
    # threshold 0.4: alerts are 0.4 (healthy) and 0.8 (injury) -> tp 1, fp 1, tn 1, fn 1
    r = metrics.alert_numbers(Y_A, S_A, 0.4)
    assert set(r) == {"tp", "fp", "tn", "fn", "alerts", "sensitivity", "specificity",
                      "precision", "false_alarms_per_caught_injury",
                      "alerts_per_100_events", "mcc"}
    assert (r["tp"], r["fp"], r["tn"], r["fn"], r["alerts"]) == (1, 1, 1, 1, 2)
    assert r["sensitivity"] == pytest.approx(0.5)
    assert r["specificity"] == pytest.approx(0.5)
    assert r["precision"] == pytest.approx(0.5)
    assert r["false_alarms_per_caught_injury"] == pytest.approx(1.0)
    assert r["alerts_per_100_events"] == pytest.approx(50.0)
    assert r["mcc"] == pytest.approx(0.0)
    assert all(_is_plain(v) for v in r.values())


def test_alert_numbers_undefined_values_are_none_not_zero():
    # threshold inf: no alerts -> tp 0, fp 0, tn 2, fn 2
    r = metrics.alert_numbers(Y_A, S_A, math.inf)
    assert (r["tp"], r["fp"], r["tn"], r["fn"], r["alerts"]) == (0, 0, 2, 2, 0)
    assert r["sensitivity"] == pytest.approx(0.0)
    assert r["specificity"] == pytest.approx(1.0)
    assert r["precision"] is None                       # 0 / 0
    assert r["false_alarms_per_caught_injury"] is None  # no caught injury
    assert r["alerts_per_100_events"] == pytest.approx(0.0)
    assert r["mcc"] is None                             # a zero factor under the root


def test_alert_numbers_mcc_against_formula():
    # threshold 0.35: alerts 0.4, 0.35, 0.8 -> tp 2, fp 1, tn 1, fn 0
    r = metrics.alert_numbers(Y_A, S_A, 0.35)
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (2, 1, 1, 0)
    assert r["mcc"] == pytest.approx((2 * 1 - 1 * 0) / math.sqrt(3 * 2 * 2 * 1))
    assert r["false_alarms_per_caught_injury"] == pytest.approx(0.5)


# ---------------------------------------------------------------- calibration


def _logistic_data(n: int, slope: float, intercept: float, seed: int):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(slope * s + intercept)))
    return s, (rng.random(n) < p).astype(int)


def test_platt_recovers_known_parameters():
    s, y = _logistic_data(20000, slope=2.0, intercept=-1.0, seed=0)
    slope, intercept = metrics.fit_platt(s, y)
    assert slope == pytest.approx(2.0, abs=0.1)
    assert intercept == pytest.approx(-1.0, abs=0.1)
    assert _is_plain(slope) and _is_plain(intercept)
    assert metrics.platt_slope_flag((slope, intercept)) is False


def test_platt_keeps_the_ranking_and_flags_a_flipped_slope():
    p = metrics.apply_platt([0.1, 0.5, 2.0], (2.0, -1.0))
    assert isinstance(p, np.ndarray)
    assert np.all(np.diff(p) > 0)
    assert p[1] == pytest.approx(1 / (1 + math.exp(-(2.0 * 0.5 - 1.0))))
    s, y = _logistic_data(5000, slope=-1.5, intercept=0.0, seed=1)
    assert metrics.platt_slope_flag(metrics.fit_platt(s, y)) is True
    assert metrics.platt_slope_flag((0.0, 1.0)) is True


def test_brier_by_hand():
    # ((0.2 - 0)^2 + (0.6 - 1)^2) / 2 = (0.04 + 0.16) / 2 = 0.10
    assert metrics.brier([0, 1], [0.2, 0.6]) == pytest.approx(0.10)


def test_ece_equal_count_by_hand():
    # 2 bins of 2: (.1,.2) vs y (0,0): |0 - .15| = .15 ; (.3,.4) vs y (1,1): |1 - .35| = .65
    assert metrics.ece_equal_count([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], bins=2) == \
        pytest.approx(0.5 * 0.15 + 0.5 * 0.65)
    # 5 events, 2 bins -> sizes 3 and 2 (numpy.array_split); input order shuffled on purpose.
    # (.1,.2,.3) vs y (0,0,1): |1/3 - .2| ; (.4,.5) vs y (1,1): |1 - .45| = .55
    y = [1, 0, 1, 0, 1]
    p = [0.5, 0.2, 0.3, 0.1, 0.4]
    assert metrics.ece_equal_count(y, p, bins=2) == \
        pytest.approx(0.6 * abs(1 / 3 - 0.2) + 0.4 * 0.55)


def test_calibration_slope_and_intercept():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.02, 0.98, 50000)
    y = (rng.random(50000) < p).astype(int)
    logit = np.log(p / (1 - p))
    slope, intercept = metrics.calibration_slope_intercept(y, p)          # well calibrated
    assert slope == pytest.approx(1.0, abs=0.05)
    assert intercept == pytest.approx(0.0, abs=0.05)
    too_extreme = 1 / (1 + np.exp(-2 * logit))
    assert metrics.calibration_slope_intercept(y, too_extreme)[0] == pytest.approx(0.5, abs=0.05)
    too_high = 1 / (1 + np.exp(-(logit + 1.0)))
    assert metrics.calibration_slope_intercept(y, too_high)[1] == pytest.approx(-1.0, abs=0.05)


# ---------------------------------------------------------------- decision curve


def test_net_benefit_by_hand():
    y = [1, 0, 0, 1]
    p = [0.8, 0.6, 0.1, 0.3]
    # t=0.5: alerts .8, .6 -> tp 1, fp 1 -> 1/4 - 1/4 * (0.5/0.5) = 0
    # t=0.2: alerts .8, .6, .3 -> tp 2, fp 1 -> 2/4 - 1/4 * (0.2/0.8) = 0.4375
    nb = metrics.net_benefit(y, p, [0.5, 0.2])
    assert isinstance(nb, np.ndarray)
    assert nb == pytest.approx([0.0, 0.4375])
    # alert everyone: prevalence .5 -> .5 - .5 * t/(1-t): t=.5 -> 0, t=.2 -> .375
    assert metrics.net_benefit_treat_all(y, [0.5, 0.2]) == pytest.approx([0.0, 0.375])


# ---------------------------------------------------------------- activity (PAMAP2)


def test_macro_f1_uses_only_true_labels():
    # label 1: precision 1, recall .5 -> F1 2/3 ; label 2: precision .5, recall .5 -> .5
    # label 3 is only predicted -> ignored.  Macro = (2/3 + 1/2) / 2
    y_true, y_pred = [1, 1, 2, 2], [1, 2, 2, 3]
    assert metrics.macro_f1(y_true, y_pred) == pytest.approx((2 / 3 + 0.5) / 2)
    assert metrics.macro_f1(y_true, y_pred) == pytest.approx(
        f1_score(y_true, y_pred, labels=[1, 2], average="macro", zero_division=0))


# ---------------------------------------------------------------- input checks


@pytest.mark.parametrize("func", [metrics.average_precision, metrics.roc_auc, metrics.brier,
                                  metrics.ece_equal_count])
def test_mismatched_lengths_are_rejected(func):
    with pytest.raises(ValueError):
        func([0, 1, 1], [0.1, 0.2])


def test_module_has_no_file_or_data_access():
    """Pure maths only: the score functions must not read files or project data."""
    from pathlib import Path

    text = Path(metrics.__file__).read_text(encoding="utf-8")
    for forbidden in ("open(", "read_csv", "np.load", "load_dev", "data/", "Path("):
        assert forbidden not in text, f"metrics.py must not use {forbidden!r}"


# ---------------------------------------------------------------- added after the WP-01 review


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def _small_overlapping_set(n: int = 40, seed: int = 11):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    y = (rng.random(n) < _sigmoid(1.2 * s - 0.5)).astype(int)
    y[0], y[1] = 1, 0
    return s, y


def test_platt_solves_its_own_equations():
    # The exact unpenalised fit makes both score equations zero; a penalised, smoothed or
    # early-stopped fit cannot (catches C=1, Platt target smoothing, default tol).
    s, y = _small_overlapping_set()
    p_hat = metrics.apply_platt(s, metrics.fit_platt(s, y))
    assert abs(np.sum(y - p_hat)) < 1e-6
    assert abs(np.sum(s * (y - p_hat))) < 1e-6


def test_platt_is_accurate_far_from_zero_and_for_constant_scores():
    s, y = _small_overlapping_set(n=400, seed=12)
    far = s + 5000.0                                   # same data, shifted far from zero
    slope, intercept = metrics.fit_platt(far, y)
    p_hat = metrics.apply_platt(far, (slope, intercept))
    assert abs(np.sum(y - p_hat)) < 1e-5 and abs(np.sum(s * (y - p_hat))) < 1e-5
    assert slope == pytest.approx(metrics.fit_platt(s, y)[0], rel=1e-6)
    assert metrics.fit_platt([1.0, 1.0, 1.0, 1.0], [0, 1, 0, 0]) == \
        pytest.approx((0.0, math.log(0.25 / 0.75)))


def test_platt_warns_on_perfectly_separated_validation_data():
    with pytest.warns(RuntimeWarning):
        metrics.fit_platt([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])


def test_platt_slope_flag_returns_a_plain_bool():
    assert metrics.platt_slope_flag((np.float64(-1.0), 0.0)) is True
    assert metrics.platt_slope_flag((np.float64(2.0), 0.0)) is False


def test_calibration_intercept_uses_logit_as_an_offset():
    # Predictions with a slope far from 1: the offset intercept a must make the mean predicted
    # risk equal the injury rate exactly; the intercept of the joint fit does not.
    rng = np.random.default_rng(13)
    p_true = rng.uniform(0.05, 0.6, 3000)
    y = (rng.random(3000) < p_true).astype(int)
    pred = _sigmoid(2 * np.log(p_true / (1 - p_true)) + 1)
    slope, intercept = metrics.calibration_slope_intercept(y, pred)
    lg = metrics.logit(pred)
    assert np.mean(_sigmoid(intercept + lg)) == pytest.approx(np.mean(y), abs=1e-10)
    assert slope == pytest.approx(0.5, abs=0.1)


def test_calibration_slope_solves_its_own_equations():
    from scipy.optimize import brentq

    s, y = _small_overlapping_set(n=60, seed=14)
    prob = _sigmoid(0.7 * s)
    b = metrics.calibration_slope_intercept(y, prob)[0]
    lg = metrics.logit(prob)
    c = brentq(lambda c_: np.sum(y - _sigmoid(c_ + b * lg)), -50, 50, xtol=1e-14)
    assert abs(np.sum(lg * (y - _sigmoid(c + b * lg)))) < 1e-6   # catches a penalised slope


def test_threshold_rule_uses_the_float_tolerance():
    # Gaps for 0.5 and 0.9 are both 2/3 but differ in the last float digit; with the 1e-12
    # tolerance the tie goes to the larger sensitivity + specificity -> 0.5 (exact compare: 0.9)
    assert metrics.choose_alert_threshold([0, 1, 1, 0, 0], [0.1, 0.5, 0.5, 0.5, 0.9]) == \
        pytest.approx(0.5)


def test_average_precision_treats_tied_scores_as_one_step():
    # 0.9 (injury): precision 1 at recall .5; the three tied 0.5s together: precision .5 at
    # recall 1 -> AP = .5 * 1 + .5 * .5 = 0.75 (breaking the tie by input order would give 1.0)
    assert metrics.average_precision([1, 0, 0, 1], [0.5, 0.5, 0.5, 0.9]) == pytest.approx(0.75)


def test_ece_keeps_tied_probabilities_in_one_bin_so_row_order_never_matters():
    # D-025: constant risk 0.02 with injury rate 0.02 is perfectly calibrated -> ECE 0 always
    y = np.zeros(100, dtype=int)
    y[:2] = 1
    p = np.full(100, 0.02)
    for order in (np.arange(100), np.arange(100)[::-1], np.random.default_rng(15).permutation(100)):
        assert metrics.ece_equal_count(y[order], p[order]) == pytest.approx(0.0)
    rng = np.random.default_rng(16)
    p5 = rng.choice([0.01, 0.05, 0.1, 0.2, 0.4], size=300)
    y5 = (rng.random(300) < p5).astype(int)
    values = {round(metrics.ece_equal_count(y5[o], p5[o]), 12)
              for o in (np.arange(300), rng.permutation(300), rng.permutation(300))}
    assert len(values) == 1


def test_no_injury_gives_none_not_zero():
    assert metrics.alert_numbers([0, 0, 0], [0.1, 0.2, 0.3], 0.2)["sensitivity"] is None


def test_net_benefit_alerts_when_risk_equals_the_threshold():
    # both events have risk exactly 0.2 -> both alerted: 1/2 - 1/2 * 0.25 = 0.375
    assert metrics.net_benefit([1, 0], [0.2, 0.2], [0.2]) == pytest.approx([0.375])


def test_logit_clips_like_the_specification():
    assert metrics.logit([0.5])[0] == pytest.approx(0.0)
    assert metrics.logit([0.0])[0] == pytest.approx(math.log(1e-6 / (1 - 1e-6)))


@pytest.mark.parametrize("call", [
    lambda: metrics.roc_auc([0, 2, 1], [0.1, 0.2, 0.3]),                 # label not 0/1
    lambda: metrics.choose_alert_threshold([0, 0.5, 1], [0.1, 0.2, 0.3]),
    lambda: metrics.choose_alert_threshold([0, 1, 1], [0.1, float("nan"), 0.3]),
    lambda: metrics.alert_numbers([0, 1, 1], [0.1, float("inf"), 0.3], 0.2),
    lambda: metrics.average_precision([0, 1], [float("nan"), 0.3]),
    lambda: metrics.brier([0, 1], [0.2, 1.5]),                           # not a probability
    lambda: metrics.ece_equal_count([0, 1], [-0.1, 0.5]),
    lambda: metrics.calibration_slope_intercept([0, 1, 1], [0.2, 3.0, 0.4]),
    lambda: metrics.net_benefit([0, 1], [0.2, 1.2], [0.1]),
    lambda: metrics.net_benefit([0, 1], [0.2, 0.4], [1.0]),              # threshold not in (0, 1)
    lambda: metrics.net_benefit([0, 1, 1], [0.2, 0.4], [0.1]),           # lengths
    lambda: metrics.alert_numbers([0, 1, 1], [0.2, 0.4], 0.3),
    lambda: metrics.choose_alert_threshold([0, 1, 1], [0.2, 0.4]),
    lambda: metrics.calibration_slope_intercept([0, 1, 1], [0.2, 0.4]),
    lambda: metrics.fit_platt([0.2, 0.4], [0, 1, 1]),
    lambda: metrics.macro_f1([1, 2, 2], [1, 2]),
    lambda: metrics.brier([], []),                                       # empty
])
def test_invalid_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()
