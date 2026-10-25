"""Piece-3 runner: python -m src.models.run [--only NAME,NAME] (or python tasks.py baselines).

Writes experiments/baselines/<run_id>/: config.yaml, run_info.json, metrics.json, summary.md,
and git-ignored predictions/ (scores per row, for the bootstrap). Every number comes from here.
Names: canary, constant_risk, acwr_style, logistic_regression (more models are added later).
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
from src.eval import metrics
from src.models import baselines, data, evaluate, tuning

ROOT = loaders.ROOT
CONFIG = ROOT / "configs" / "baselines" / "lovdal.yaml"
OUT = ROOT / "experiments" / "baselines"
ORDER = ("canary", "constant_risk", "acwr_style", "logistic_regression")


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


def _fingerprints() -> dict:
    meta = json.loads((loaders.PROCESSED / "lovdal" / "day_meta.json").read_text("utf-8"))
    return {"lovdal_source": (meta.get("source") or {}).get("fingerprint"),
            "lovdal_split": loaders.lovdal_split()["meta"]["data_fingerprint"]}


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
    for name in ORDER[1:]:
        for kind, res in results.get(name, {}).items():
            s = res["summary"]
            rate = np.mean([r["test_injury_rate"] for r in res["rounds"]])
            cell = lambda v: "n/a" if v is None else f"{v['mean']:.3f} ± {v['sd']:.3f}"  # noqa: E731
            lines.append(f"| {name} | {kind} | {cell(s['pr_auc'])} | {cell(s['roc_auc'])} "
                         f"| {rate:.4f} |")
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

    code = audit.code_state()  # before anything is written
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = OUT / run_id
    pred = folder / "predictions"
    folder.mkdir(parents=True)
    shutil.copyfile(CONFIG, folder / "config.yaml")
    cfg = yaml.safe_load(CONFIG.read_text("utf-8"))

    jobs = {"canary": lambda: run_canary(cfg),
            "constant_risk": lambda: run_constant_risk(cfg, pred),
            "acwr_style": lambda: run_acwr(cfg, pred),
            "logistic_regression": lambda: run_logistic(cfg, pred)}
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
            "packages": {p: version(p) for p in ("numpy", "scikit-learn", "scipy", "PyYAML")},
            "seeds": {"search_seed_base": cfg["procedure"]["search_seed_base"],
                      "canary_seed_base": cfg["canary"]["seed_base"]},
            "cpu_count": os.cpu_count(), "runtime_s": runtime, "data_fingerprints": _fingerprints()}
    (folder / "run_info.json").write_text(json.dumps(info, indent=2), "utf-8")
    (folder / "metrics.json").write_text(json.dumps(results, indent=2), "utf-8")
    (folder / "summary.md").write_text(_summary_md(run_id, results), "utf-8")
    print((folder / "summary.md").read_text("utf-8"))
    return 0 if results.get("canary", {"passed": True})["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
