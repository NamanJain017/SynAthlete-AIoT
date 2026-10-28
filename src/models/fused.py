"""Piece-7 fused model (D-023(2), D-024(3), D-027(4)): python -m src.models.fused (tasks.py train).

Per development round, on the shared days (D-014), using that round's runners only:
1. Arms: sequence {GRU, LSTM, TCN} on day sequences, tabular {random forest, CatBoost} on week
   features; 8 settings each (seed 1000 + k, the piece-3 grids unchanged); the best sequence
   candidate and the best tabular candidate are chosen on validation AP (then ROC-AUC).
2. Combiner: logistic regression on the logits of the two arms' scores, trained on INNER
   5-fold out-of-fold predictions grouped by runner over the training runners only (a row's
   arm scores always come from arms that never trained on that row); C from {0.01, 0.1, 1, 10}
   chosen on validation AP. 44 tries per round in total.
3. The chosen pipeline is refit with seeds 0-2; validation fits Platt and the alert level;
   test is scored once (src.models.evaluate). Risk score 0-100 = percentile of the calibrated
   risk among the round's validation events (D-027(4)).
Rounds run in parallel processes. Writes experiments/fused/<run_id>/.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from src.data import loaders
from src.eval import metrics
from src.models import data, evaluate, fitters, tuning
from src.models import run as base_run

ROOT = loaders.ROOT
FUSED_CFG = ROOT / "configs" / "fused_candidates.yaml"
BASE_CFG = ROOT / "configs" / "baselines" / "lovdal.yaml"
OUT = ROOT / "experiments" / "fused"
SEQ, TAB = ("gru", "lstm", "tcn"), ("random_forest", "catboost")


def candidates(fcfg: dict, bcfg: dict) -> dict[str, dict]:
    """Grid and fixed values of every arm candidate, from the pre-declared configs."""
    gru = bcfg["models"]["gru"]
    tcn = fcfg["sequence_arm"]["tcn"]
    out = {"gru": gru, "lstm": gru, "tcn": {"fixed": {**tcn["fixed"], "pos_weight": 1,
                                                       "weight_decay": 0.0}, "grid": tcn["grid"]}}
    for name in TAB:
        out[name] = bcfg["models"][name]
    return out


def fit_fn(name: str):
    return fitters.sequence(name) if name in SEQ else getattr(fitters, name)


def oof_predictions(fit, r: data.Round, setting: dict, seed: int, folds: int = 5) -> np.ndarray:
    """Out-of-fold arm scores for every training row: each row is scored by a model fitted on
    the OTHER training runners only (GroupKFold by runner). Early stopping inside uses the
    round's validation runners, as for every arm fit."""
    out = np.full(len(r.train.y), np.nan)
    for inner_train, held in GroupKFold(n_splits=folds).split(r.train.X, r.train.y,
                                                              r.train.person):
        sub = dataclasses.replace(r, train=data.Part(r.train.X[inner_train],
                                                     r.train.y[inner_train],
                                                     r.train.person[inner_train]))
        out[held] = fit(sub, setting, seed)(r.train.X[held])
    return out


def _features(seq: np.ndarray, tab: np.ndarray) -> np.ndarray:
    return np.column_stack([metrics.logit(seq), metrics.logit(tab)])


def risk_score(p_val: np.ndarray, p: np.ndarray) -> np.ndarray:
    """0-100: share of the round's validation events with a lower calibrated risk."""
    ref = np.sort(p_val)
    return 100.0 * np.searchsorted(ref, p, side="right") / len(ref)


def _tune_arm(names, cands, rs: data.Round, proc: dict) -> dict:
    """Try 8 settings of every candidate in `names`; keep the best on validation."""
    trials, best = [], None
    for name in names:
        c = cands[name]
        for s in tuning.draw_settings(c["grid"], proc["settings_per_round"],
                                      proc["search_seed_base"] + rs.k, c.get("fixed")):
            p_val = fit_fn(name)(rs, s, proc["tune_seed"])(rs.val.X)
            trials.append({"model": name, "setting": s,
                           "val_ap": metrics.average_precision(rs.val.y, p_val),
                           "val_auc": metrics.roc_auc(rs.val.y, p_val)})
            if tuning.choose(trials) == len(trials) - 1:
                best = (name, s)
    return {"model": best[0], "setting": best[1], "trials": trials}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_pipeline(r_seq: data.Round, r_tab: data.Round, proc: dict, cands: dict,
                 fcfg: dict) -> dict:
    """The whole fused procedure on one split (used by every development round AND the final
    model, so both follow exactly the same steps): tune both arms on validation, then for each
    refit seed the out-of-fold combiner (C chosen on validation) and the scores of the
    validation and test rows. Test rows are scored once and never used for a choice."""
    assert np.array_equal(r_seq.train.person, r_tab.train.person)  # same rows, same order
    seq = _tune_arm(SEQ, cands, r_seq, proc)
    tab = _tune_arm(TAB, cands, r_tab, proc)
    seeds = []
    for seed in proc["refit_seeds"]:
        f_seq, f_tab = fit_fn(seq["model"]), fit_fn(tab["model"])
        oof = _features(oof_predictions(f_seq, r_seq, seq["setting"], seed),
                        oof_predictions(f_tab, r_tab, tab["setting"], seed))
        p_seq, p_tab = f_seq(r_seq, seq["setting"], seed), f_tab(r_tab, tab["setting"], seed)
        val_f = _features(p_seq(r_seq.val.X), p_tab(r_tab.val.X))
        test_f = _features(p_seq(r_seq.test.X), p_tab(r_tab.test.X))
        tries = []
        for C in fcfg["combiner"]["grid"]["C"]:  # chosen on validation AP only
            comb = LogisticRegression(C=C, max_iter=2000).fit(oof, r_seq.train.y)
            pv = comb.predict_proba(val_f)[:, 1]
            tries.append({"C": C, "val_ap": metrics.average_precision(r_seq.val.y, pv),
                          "val_auc": metrics.roc_auc(r_seq.val.y, pv), "model": comb})
        best = tuning.choose(tries)
        comb = tries[best]["model"]
        seeds.append({"seed": seed, "s_val": comb.predict_proba(val_f)[:, 1],
                      "s_test": comb.predict_proba(test_f)[:, 1],
                      "seq_val": _sigmoid(val_f[:, 0]), "seq_test": _sigmoid(test_f[:, 0]),
                      "tab_val": _sigmoid(val_f[:, 1]), "tab_test": _sigmoid(test_f[:, 1]),
                      "predict_seq": p_seq, "predict_tab": p_tab,
                      "combiner_C": tries[best]["C"],
                      "combiner_coef": [float(v) for v in comb.coef_[0]],
                      "combiner_trials": [{k2: v for k2, v in t.items() if k2 != "model"}
                                          for t in tries]})
    return {"sequence_arm": seq, "tabular_arm": tab, "seeds": seeds,
            "tries": len(seq["trials"]) + len(tab["trials"]) + len(fcfg["combiner"]["grid"]["C"])}


def score_seed(r: data.Round, sd: dict) -> dict:
    """All reported numbers of one refit seed (validation fits Platt and the alert level)."""
    scores = evaluate.injury_scores(r.val.y, sd["s_val"], r.test.y, sd["s_test"], True)
    return {**scores, "combiner_C": sd["combiner_C"], "combiner_coef": sd["combiner_coef"],
            "arms_test": {a: {"pr_auc": metrics.average_precision(r.test.y, sd[f"{a}_test"]),
                              "roc_auc": metrics.roc_auc(r.test.y, sd[f"{a}_test"])}
                          for a in ("seq", "tab")}}


def configs() -> tuple[dict, dict, dict, dict]:
    fcfg, bcfg = yaml.safe_load(FUSED_CFG.read_text("utf-8")), yaml.safe_load(
        BASE_CFG.read_text("utf-8"))
    return fcfg, bcfg, bcfg["procedure"], candidates(fcfg, bcfg)


def fused_round(k: int, folder: str) -> dict:
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 5) // 5))
    fcfg, _, proc, cands = configs()
    r_seq, r_tab = list(data.lovdal_shared_rounds())[k]
    fit = fit_pipeline(r_seq, r_tab, proc, cands, fcfg)
    pred = os.path.join(folder, "predictions")
    os.makedirs(pred, exist_ok=True)
    per_seed = []
    for sd in fit["seeds"]:
        params = metrics.fit_platt(metrics.logit(sd["s_val"]), r_seq.val.y)
        p_cal_val = metrics.apply_platt(metrics.logit(sd["s_val"]), params)
        p_cal_test = metrics.apply_platt(metrics.logit(sd["s_test"]), params)
        np.savez_compressed(os.path.join(pred, f"fused_s{sd['seed']}_r{k}.npz"),
                            y_val=r_seq.val.y, y_test=r_seq.test.y, person_val=r_seq.val.person,
                            person_test=r_seq.test.person, s_val=sd["s_val"],
                            s_test=sd["s_test"], seq_val=sd["seq_val"], seq_test=sd["seq_test"],
                            tab_val=sd["tab_val"], tab_test=sd["tab_test"],
                            risk_test=risk_score(p_cal_val, p_cal_test))
        per_seed.append(score_seed(r_seq, sd))
    out = {**evaluate.mean_over_seeds(per_seed), "seeds": per_seed, "round": k,
           "sequence_arm": fit["sequence_arm"], "tabular_arm": fit["tabular_arm"],
           "combiner_trials": [sd["combiner_trials"] for sd in fit["seeds"]],
           "tries": fit["tries"]}
    print(f"  fused round {k}: test PR-AUC {out['pr_auc']:.4f} ROC-AUC {out['roc_auc']:.3f} "
          f"(arms {fit['sequence_arm']['model']} + {fit['tabular_arm']['model']})", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    code = base_run._code_state()
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = OUT / run_id
    folder.mkdir(parents=True)
    shutil.copyfile(FUSED_CFG, folder / "config_fused.yaml")
    shutil.copyfile(BASE_CFG, folder / "config.yaml")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=5) as ex:
        rounds = list(ex.map(fused_round, range(5), [str(folder)] * 5))
    results = {"fused": {"summary": evaluate.summarise(rounds), "rounds": rounds},
               "arms_summary": {a: evaluate.summarise(
                   [{"pr_auc": np.mean([s["arms_test"][a]["pr_auc"] for s in rd["seeds"]]),
                     "roc_auc": np.mean([s["arms_test"][a]["roc_auc"] for s in rd["seeds"]])}
                    for rd in rounds]) for a in ("seq", "tab")}}
    info = {**code, "runtime_s": round(time.perf_counter() - t0, 1),
            "data_fingerprints": base_run._fingerprints()}
    (folder / "run_info.json").write_text(json.dumps(info, indent=2), "utf-8")
    (folder / "metrics.json").write_text(json.dumps(results, indent=2, default=float), "utf-8")
    s = results["fused"]["summary"]
    lines = [f"# Piece-7 fused model, run {run_id}", "",
             "Shared days; mean ± SD over the 5 development rounds; seeds 0-2 averaged.", "",
             "| Model | PR-AUC | ROC-AUC |", "|---|---|---|",
             f"| fused | {s['pr_auc']['mean']:.3f} ± {s['pr_auc']['sd']:.3f} | "
             f"{s['roc_auc']['mean']:.3f} ± {s['roc_auc']['sd']:.3f} |"]
    for a, label in (("seq", "sequence arm (chosen)"), ("tab", "tabular arm (chosen)")):
        v = results["arms_summary"][a]
        lines.append(f"| {label} | {v['pr_auc']['mean']:.3f} ± {v['pr_auc']['sd']:.3f} | "
                     f"{v['roc_auc']['mean']:.3f} ± {v['roc_auc']['sd']:.3f} |")
    lines += ["", "Chosen arms per round: " + ", ".join(
        f"r{rd['round']} {rd['sequence_arm']['model']}+{rd['tabular_arm']['model']}"
        for rd in rounds)]
    (folder / "summary.md").write_text("\n".join(lines) + "\n", "utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
