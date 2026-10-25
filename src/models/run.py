"""Piece-3 runner: python -m src.models.run [--only NAME,NAME] (or python tasks.py baselines).

Writes experiments/baselines/<run_id>/: config.yaml, run_info.json, metrics.json, summary.md,
and git-ignored predictions/ (scores per row, for the bootstrap). Every number comes from here.
Names: see ORDER. One family per run keeps each run short (AGENTS.md: under about 2 hours).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from src.data import audit, loaders
from src.eval import bootstrap, metrics
from src.models import baselines, data, evaluate, fitters, lovdal_method, tuning

ROOT = loaders.ROOT
CONFIG = ROOT / "configs" / "baselines" / "lovdal.yaml"
PAMAP2_CONFIG = ROOT / "configs" / "baselines" / "pamap2.yaml"
OUT = ROOT / "experiments" / "baselines"
INJURY = ("constant_risk", "acwr_style", "logistic_regression", "random_forest", "catboost", "gru",
          "lovdal_label_free")
ORDER = ("canary", *INJURY, "lovdal_reproduction", "pamap2_cnn1d", "pamap2_gru")


def _save_predictions(folder: Path, name: str, r: data.Round, s_val, s_test) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{name}_r{r.k}.npz", y_val=r.val.y, s_val=s_val,
                        person_val=r.val.person, y_test=r.test.y, s_test=s_test,
                        person_test=r.test.person)


def run_canary(cfg: dict) -> dict:
    """Leak canary part (b): logistic regression on shuffled training labels must score at chance
    on test (D-024). Part (a), the declared-input check, runs inside data.lovdal_rounds."""
    c = cfg["canary"]
    limit = c["fail_if_abs_mean_test_roc_auc_minus_0.5_above"]
    out = {}
    for kind in c["files"]:
        aucs = []
        for r in data.lovdal_rounds(kind):
            y_shuffled = baselines.shuffled_labels(r.train.y, c["seed_base"] + r.k)
            model = baselines.logistic_regression(r.train.X, y_shuffled, c["fixed"])
            aucs.append(metrics.roc_auc(r.test.y, model.predict_proba(r.test.X)[:, 1]))
        mean = float(np.mean(aucs))
        out[kind] = {"test_roc_auc_per_round": aucs, "mean": mean,
                     "passed": bool(abs(mean - 0.5) <= limit)}
    out["passed"] = all(out[k]["passed"] for k in c["files"])
    out["rule"] = f"|mean test ROC-AUC - 0.5| <= {limit} for every file"
    return out


def run_constant_risk(cfg: dict, pred: Path) -> dict:
    out = {}
    for kind in cfg["models"]["constant_risk"]["files"]:
        rounds = []
        for r in data.lovdal_rounds(kind):
            s_val = baselines.constant_risk(r.train.y, len(r.val.y))
            s_test = baselines.constant_risk(r.train.y, len(r.test.y))
            _save_predictions(pred, f"constant_risk_{kind}", r, s_val, s_test)
            rounds.append(evaluate.injury_scores(r.val.y, s_val, r.test.y, s_test, True))
        out[kind] = {"summary": evaluate.summarise(rounds), "rounds": rounds}
    return out


def run_acwr(cfg: dict, pred: Path) -> dict:
    m = cfg["models"]["acwr_style"]
    week = loaders.load_dev("week")
    ratio = baselines.acwr_ratio(week, m["km_feature"], m["floor_km"])
    split = loaders.lovdal_split()
    rounds = []
    for r in data.lovdal_rounds("week"):  # same rows and order as the masks below
        masks = data.split_rows(week.athlete_id, split["folds"][str(r.k)], split["lockbox_ids"])
        if not (np.array_equal(week.y[masks["val"]], r.val.y)
                and np.array_equal(week.y[masks["test"]], r.test.y)):
            raise RuntimeError("ACWR rows do not line up with the round's rows")
        s_val, s_test = ratio[masks["val"]], ratio[masks["test"]]
        _save_predictions(pred, "acwr_style_week", r, s_val, s_test)
        rounds.append(evaluate.injury_scores(r.val.y, s_val, r.test.y, s_test, False))
    return {"week": {"summary": evaluate.summarise(rounds), "rounds": rounds}}


def run_logistic(cfg: dict, pred: Path) -> dict:
    m, proc = cfg["models"]["logistic_regression"], cfg["procedure"]
    out = {}
    for kind in m["files"]:
        rounds = []
        for r in data.lovdal_rounds(kind):
            settings = tuning.draw_settings(m["grid"], proc["settings_per_round"],
                                            proc["search_seed_base"] + r.k, m["fixed"])
            trials, models = [], []
            for s in settings:  # validation of THIS round only (D-021)
                model = baselines.logistic_regression(r.train.X, r.train.y, s)
                p_val = model.predict_proba(r.val.X)[:, 1]
                trials.append({"setting": s, "val_ap": metrics.average_precision(r.val.y, p_val),
                               "val_auc": metrics.roc_auc(r.val.y, p_val)})
                models.append(model)
            best = tuning.choose(trials)
            s_val = models[best].predict_proba(r.val.X)[:, 1]
            s_test = models[best].predict_proba(r.test.X)[:, 1]  # the one look at this test group
            _save_predictions(pred, f"logistic_regression_{kind}", r, s_val, s_test)
            scores = evaluate.injury_scores(r.val.y, s_val, r.test.y, s_test, True)
            rounds.append({**scores, "chosen": best, "trials": trials})
        out[kind] = {"summary": evaluate.summarise(rounds), "rounds": rounds}
    return out


def _seeded(label: str, r: data.Round, pred: Path, setting: dict, seeds: list[int], fit,
            first=None, tune_seed: int = 0) -> dict:
    """Refit the chosen setting with every seed on the training runners (D-024) and average the
    round's numbers over seeds. `first` reuses the tuning fit for the tuning seed."""
    per_seed = []
    for seed in seeds:
        predict = first if (first is not None and seed == tune_seed) else fit(r, setting, seed)
        s_val, s_test = predict(r.val.X), predict(r.test.X)  # the one look at this test group
        _save_predictions(pred, f"{label}_s{seed}", r, s_val, s_test)
        per_seed.append(evaluate.injury_scores(r.val.y, s_val, r.test.y, s_test, True))
    return {**evaluate.mean_over_seeds(per_seed), "seeds": per_seed}


def run_tuned(name: str, cfg: dict, pred: Path, fit, layout: str = "flat") -> dict:
    """8 settings tried with seed 0, chosen on this round's validation runners (val AP, then
    val ROC-AUC); the chosen one refit with seeds 0-2 (D-021, D-024)."""
    m, proc = cfg["models"][name], cfg["procedure"]
    out = {}
    for kind in m["files"]:
        rounds = []
        for r in data.lovdal_rounds(kind, layout):
            settings = tuning.draw_settings(m["grid"], proc["settings_per_round"],
                                            proc["search_seed_base"] + r.k, m.get("fixed"))
            trials, best_predict = [], None
            for s in settings:
                predict = fit(r, s, proc["tune_seed"])
                p_val = predict(r.val.X)
                trials.append({"setting": s, "val_ap": metrics.average_precision(r.val.y, p_val),
                               "val_auc": metrics.roc_auc(r.val.y, p_val)})
                if tuning.choose(trials) == len(trials) - 1:  # keep only the best model so far
                    best_predict = predict
            best = tuning.choose(trials)
            seeded = _seeded(f"{name}_{kind}", r, pred, settings[best], proc["refit_seeds"], fit,
                             best_predict, proc["tune_seed"])
            rounds.append({**seeded, "chosen": best, "trials": trials})
            print(f"  {name} {kind} round {r.k}: test PR-AUC {seeded['pr_auc']:.4f}", flush=True)
        out[kind] = {"summary": evaluate.summarise(rounds), "rounds": rounds}
    return out


def run_lovdal_label_free(cfg: dict, pred: Path) -> dict:
    """The paper's bagged XGBoost (untuned) with our training-rows scaling; seeds 0-2."""
    m, proc = cfg["models"]["lovdal_xgboost"], cfg["procedure"]
    out = {}
    for kind in m["files"]:
        rounds = [_seeded(f"lovdal_label_free_{kind}", r, pred, m["paper"], proc["refit_seeds"],
                          fitters.lovdal_label_free) for r in data.lovdal_rounds(kind)]
        out[kind] = {"summary": evaluate.summarise(rounds), "rounds": rounds}
    return out


def run_reproduction(cfg: dict) -> dict:
    """Labelled reproduction check (D-021, D-024): the exact copy, label-reading scaling included,
    trained on runners 0-53 and tested on 54-63, day file as in the paper. Never used to choose."""
    m = cfg["models"]["lovdal_xgboost"]
    rc = m["reproduction_check"]
    train = range(rc["train_runners"][0], rc["train_runners"][1] + 1)
    test = range(rc["test_runners"][0], rc["test_runners"][1] + 1)
    out = {"label": "exact copy incl. the paper's label-reading scaling; never used to choose",
           "train_runners": rc["train_runners"], "test_runners": rc["test_runners"]}
    for kind in m["files"]:  # day and week (D-023(8)); the notebook ran the day file
        d = loaders.load_dev(kind)
        X, y, athlete = data.flatten(d), d.y.astype(np.int64), d.athlete_id
        scores = [lovdal_method.reproduction(X, y, athlete, train, test, s, m["paper"])
                  for s in rc["seeds"]]
        in_test = np.isin(athlete, list(test))
        check = bootstrap.reproduction_check(y[in_test], scores, athlete[in_test],
                                             metrics.roc_auc,
                                             resamples=cfg["procedure"]["bootstrap"]["resamples"])
        out[kind] = {"test_events": int(in_test.sum()), "test_injuries": int(y[in_test].sum()),
                     "test_roc_auc_per_seed": [metrics.roc_auc(y[in_test], s) for s in scores],
                     **check}
    return out


def run_pamap2(name: str, pcfg: dict) -> dict:
    """PAMAP2 activity network: 8 settings on validation macro-F1, refit seeds 0-2, test macro-F1
    over the activities present in the test people (D-024)."""
    m, proc = pcfg["models"][name], pcfg["procedure"]
    fit = fitters.pamap2_net(name)
    rounds = []
    for r in data.pamap2_rounds():
        settings = tuning.draw_settings(m["grid"], proc["settings_per_round"],
                                        proc["search_seed_base"] + r.k, m["fixed"])
        trials = []
        for s in settings:
            _, val_f1 = fit(r, s, proc["tune_seed"])
            trials.append({"setting": s, "val_macro_f1": float(val_f1)})
        scores = [t["val_macro_f1"] for t in trials]
        best = scores.index(max(scores))  # ties -> the earlier setting
        test_f1 = []
        for seed in proc["refit_seeds"]:
            predict, _ = fit(r, settings[best], seed)
            test_f1.append(metrics.macro_f1(r.test.y, predict(r.test.X)))
        rounds.append({"macro_f1": float(np.mean(test_f1)), "macro_f1_per_seed": test_f1,
                       "chosen": best, "trials": trials, "test_clips": int(len(r.test.y))})
        print(f"  pamap2 {name} round {r.k}: test macro-F1 {rounds[-1]['macro_f1']:.3f}",
              flush=True)
    return {"summary": evaluate.summarise(rounds, ("macro_f1",)), "rounds": rounds}


def _code_state() -> dict:
    """Commit and dirty flag; earlier run outputs (untracked files under experiments/) are
    results, not code, so they do not make the code 'dirty'."""
    state = audit.code_state()
    porcelain = audit._git("status", "--porcelain")
    if porcelain == "unavailable":
        return {**state, "git_dirty": True}
    lines = [ln for ln in porcelain.splitlines() if ln.strip()]
    state["git_dirty"] = any(not (ln.startswith("?? ") and ln[3:].startswith("experiments/"))
                             for ln in lines)
    return state


def _fingerprints() -> dict:
    meta = json.loads((loaders.PROCESSED / "lovdal" / "day_meta.json").read_text("utf-8"))
    return {"lovdal_source": (meta.get("source") or {}).get("fingerprint"),
            "lovdal_split": loaders.lovdal_split()["meta"]["data_fingerprint"],
            "pamap2_source": json.loads((loaders.PROCESSED / "pamap2" / "windows_meta.json")
                                        .read_text("utf-8"))["source"]["fingerprint"],
            "pamap2_split": json.loads((loaders.SPLITS / "pamap2_folds.json")
                                       .read_text("utf-8"))["meta"]["data_fingerprint"]}


def _summary_md(run_id: str, results: dict) -> str:
    lines = [f"# Piece-3 baselines, run {run_id}", "",
             "Mean ± SD over the 5 development rounds (test groups); chance level of PR-AUC = "
             "the test injury rate. Written by src/models/run.py.", ""]
    if "canary" in results:
        c = results["canary"]
        lines += [f"Leak canary: {'PASSED' if c['passed'] else 'FAILED'} ({c['rule']}); "
                  + ", ".join(f"{k} mean test ROC-AUC {c[k]['mean']:.3f}"
                              for k in c if isinstance(c[k], dict)), ""]
    lines += ["| Model | File | PR-AUC | ROC-AUC | Test injury rate |", "|---|---|---|---|---|"]
    for name in INJURY:
        for kind, res in results.get(name, {}).items():
            s = res["summary"]
            rate = np.mean([r["test_injury_rate"] for r in res["rounds"]])
            cell = lambda v: "n/a" if v is None else f"{v['mean']:.3f} ± {v['sd']:.3f}"  # noqa: E731
            lines.append(f"| {name} | {kind} | {cell(s['pr_auc'])} | {cell(s['roc_auc'])} "
                         f"| {rate:.4f} |")
    if "lovdal_reproduction" in results:
        rc = results["lovdal_reproduction"]
        lines += ["", f"Reproduction check (exact copy, runners {rc['train_runners']} -> "
                  f"{rc['test_runners']}, label-reading scaling, never used to choose; "
                  "pass = 95% runner-bootstrap lower bound above 0.5):"]
        for kind in ("day", "week"):
            if kind in rc:
                c = rc[kind]
                lines.append(f"- {kind}: test ROC-AUC {c['estimate']:.3f} ({c['low']:.3f} to "
                             f"{c['high']:.3f}), {'PASSED' if c['passed'] else 'FAILED'}")
    pam = [n for n in ("pamap2_cnn1d", "pamap2_gru") if n in results]
    if pam:
        lines += ["", "| PAMAP2 model | Test macro-F1 (4 rounds) |", "|---|---|"]
        for n in pam:
            v = results[n]["summary"]["macro_f1"]
            lines.append(f"| {n} | {v['mean']:.3f} ± {v['sd']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated names: " + ", ".join(ORDER))
    args = parser.parse_args(argv)
    names = ORDER if not args.only else tuple(n.strip() for n in args.only.split(","))
    unknown = [n for n in names if n not in ORDER]
    if unknown:
        print(f"unknown name(s) {unknown}; choose from {', '.join(ORDER)}")
        return 2
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    code = _code_state()  # before anything is written
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = OUT / run_id
    pred = folder / "predictions"
    folder.mkdir(parents=True)
    shutil.copyfile(CONFIG, folder / "config.yaml")
    cfg = yaml.safe_load(CONFIG.read_text("utf-8"))
    pcfg = yaml.safe_load(PAMAP2_CONFIG.read_text("utf-8"))
    if any(n.startswith("pamap2") for n in names):
        shutil.copyfile(PAMAP2_CONFIG, folder / "config_pamap2.yaml")

    jobs = {"canary": lambda: run_canary(cfg),
            "constant_risk": lambda: run_constant_risk(cfg, pred),
            "acwr_style": lambda: run_acwr(cfg, pred),
            "logistic_regression": lambda: run_logistic(cfg, pred),
            "random_forest": lambda: run_tuned("random_forest", cfg, pred, fitters.random_forest),
            "catboost": lambda: run_tuned("catboost", cfg, pred, fitters.catboost),
            "gru": lambda: run_tuned("gru", cfg, pred, fitters.gru, layout="sequence"),
            "lovdal_label_free": lambda: run_lovdal_label_free(cfg, pred),
            "lovdal_reproduction": lambda: run_reproduction(cfg),
            "pamap2_cnn1d": lambda: run_pamap2("cnn1d", pcfg),
            "pamap2_gru": lambda: run_pamap2("gru", pcfg)}
    results, runtime = {}, {}
    for name in [n for n in ORDER if n in names]:
        t0 = time.perf_counter()
        results[name] = jobs[name]()
        runtime[name] = round(time.perf_counter() - t0, 1)
        print(f"{name}: done in {runtime[name]} s", flush=True)
        if name == "canary" and not results[name]["passed"]:
            print("LEAK CANARY FAILED: stopping before any real model (D-024)")
            break

    command = "python -m src.models.run " + " ".join(argv if argv is not None else sys.argv[1:])
    info = {"run_id": run_id, "command": command,
            "only": args.only, **code, "python": platform.python_version(),
            "packages": {p: version(p) for p in ("numpy", "scikit-learn", "scipy", "PyYAML",
                                                 "xgboost", "catboost", "torch")},
            "seeds": {"search_seed_base": cfg["procedure"]["search_seed_base"],
                      "canary_seed_base": cfg["canary"]["seed_base"],
                      "refit_seeds": cfg["procedure"]["refit_seeds"],
                      "pamap2_search_seed_base": pcfg["procedure"]["search_seed_base"]},
            "cpu_count": os.cpu_count(), "runtime_s": runtime, "data_fingerprints": _fingerprints()}
    (folder / "run_info.json").write_text(json.dumps(info, indent=2), "utf-8")
    (folder / "metrics.json").write_text(json.dumps(results, indent=2), "utf-8")
    (folder / "summary.md").write_text(_summary_md(run_id, results), "utf-8")
    print((folder / "summary.md").read_text("utf-8"))
    return 0 if results.get("canary", {"passed": True})["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
