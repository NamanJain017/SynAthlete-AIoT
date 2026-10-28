"""Final model and the ONE locked test (D-017, D-021, D-023(11)): python -m src.models.final.

    python -m src.models.final --dry-run   # whole path on a development round; no lockbox
    python -m src.models.final             # only at the eval-freeze commit (lockbox.py checks)

Final model: the fused procedure (src.models.fused.fit_pipeline, identical to the development
rounds) on the final split of the development runners (train 51 / validation 13; D-017),
search seed 1000 + 5. It is scored once on the shared days of the locked runners 64-73
(primary), next to the 5-round headline, never instead of it. Secondary, scored once and never
used to choose anything: the chosen arms on their own files, and the Lövdal label-free variant
trained on runners 0-63 (paper_split), threshold-free only (D-030). Uncertainty: runner
bootstrap over the 10 locked runners (2,000 resamples), which understates uncertainty with so few
runners (disclosed, EVAL_PROTOCOL).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np

from src.data import loaders, lockbox
from src.eval import bootstrap, metrics
from src.models import data, evaluate, fitters, fused
from src.models import run as base_run

OUT = loaders.ROOT / "experiments" / "lockbox"
FINAL_K = 5  # search seed 1000 + 5: the final model's own seeded random search


def _shared(day: loaders.RunnerData, week: loaders.RunnerData):
    i_day, i_week = loaders.shared_days(day, week)
    return (day.X[i_day], data.flatten(week)[i_week], day.y[i_day].astype(np.int64),
            day.athlete_id[i_day])


def _round(kind, X_dev, y_dev, p_dev, X_test, y_test, p_test, fold, names) -> data.Round:
    """Train/validation from the development runners (one scaler on training rows only);
    test = rows scaled with that same scaler."""
    masks = data.split_rows(p_dev, {**fold, "test_ids": []})
    mean, std = data.fit_scaler(X_dev[masks["train"]])
    part = lambda m: data.Part(data.apply_scaler(X_dev[m], mean, std), y_dev[m], p_dev[m])  # noqa: E731
    return data.Round(FINAL_K, kind, part(masks["train"]), part(masks["val"]),
                      data.Part(data.apply_scaler(X_test, mean, std), y_test, p_test),
                      names, mean, std)


def _ci(y, scores_by_seed, person, metric) -> dict:
    """Seed-averaged metric with a runner bootstrap (2,000 resamples) over the test runners."""
    r = bootstrap.reproduction_check(y, scores_by_seed, person, metric, resamples=2000)
    return {k: r[k] for k in ("estimate", "low", "high", "resamples_used")}


def run(dry_run: bool) -> dict:
    split = loaders.lovdal_split()
    dev_day, dev_week = loaders.load_dev("day"), loaders.load_dev("week")
    names_seq, names_tab = data.input_names(dev_day), data.input_names(dev_week)
    data.check_declared_inputs(names_seq, "day")
    data.check_declared_inputs(names_tab, "week")
    if dry_run:  # development round 1 stands in: its test group plays the locked runners
        fold = split["folds"]["1"]
        test_ids = fold["test_ids"]
        keep = lambda d: loaders.select(d, ~np.isin(d.athlete_id, test_ids))  # noqa: E731
        lock = lambda d: loaders.select(d, np.isin(d.athlete_id, test_ids))  # noqa: E731
        lock_day, lock_week = lock(dev_day), lock(dev_week)
        dev_day, dev_week = keep(dev_day), keep(dev_week)
        fold = {"train_ids": fold["train_ids"], "val_ids": fold["val_ids"]}
    else:
        fold = {"train_ids": split["final"]["train_ids"], "val_ids": split["final"]["val_ids"]}
        reason = "final evaluation of the declared fused model (D-021, D-023(11))"
        lock_day = lockbox.open_lockbox("day", reason)
        lock_week = lockbox.open_lockbox("week", reason)
    Xs, Xt, y, p = _shared(dev_day, dev_week)
    Ls, Lt, ly, lp = _shared(lock_day, lock_week)
    r_seq = _round("day", Xs, y, p, Ls, ly, lp, fold, names_seq)
    r_tab = _round("week", Xt, y, p, Lt, ly, lp, fold, names_tab)

    fcfg, bcfg, proc, cands = fused.configs()
    fit = fused.fit_pipeline(r_seq, r_tab, proc, cands, fcfg)
    per_seed = [fused.score_seed(r_seq, sd) for sd in fit["seeds"]]
    s_test = [sd["s_test"] for sd in fit["seeds"]]
    out = {"label": "DRY RUN on development round 1 (not the lockbox)" if dry_run else
           "LOCKED FINAL TEST: runners 64-73, opened once",
           "test_runners": sorted(int(v) for v in np.unique(lp)),
           "test_events": int(len(ly)), "test_injuries": int(ly.sum()),
           "fused": {**evaluate.mean_over_seeds(per_seed), "seeds": per_seed,
                     "pr_auc_ci": _ci(ly, s_test, lp, metrics.average_precision),
                     "roc_auc_ci": _ci(ly, s_test, lp, metrics.roc_auc)},
           "sequence_arm": fit["sequence_arm"], "tabular_arm": fit["tabular_arm"],
           "tries": fit["tries"]}

    # secondary 1: the chosen arms on their own files (all locked rows of that file)
    own = {}
    for arm, rr, d, key in (("sequence", r_seq, lock_day, "predict_seq"),
                            ("tabular", r_tab, lock_week, "predict_tab")):
        X = d.X if arm == "sequence" else data.flatten(d)
        Xn = data.apply_scaler(X, rr.scaler_mean, rr.scaler_std)
        scores = [sd[key](Xn) for sd in fit["seeds"]]
        yy = d.y.astype(int)
        own[arm] = {"file": "day" if arm == "sequence" else "week", "events": int(len(d.y)),
                    "injuries": int(d.y.sum()),
                    "pr_auc_ci": _ci(yy, scores, d.athlete_id, metrics.average_precision),
                    "roc_auc_ci": _ci(yy, scores, d.athlete_id, metrics.roc_auc)}
    out["arms_on_own_files"] = own

    # secondary 2: Lövdal label-free variant trained on runners 0-63 (paper_split), threshold-free
    paper = bcfg["models"]["lovdal_xgboost"]["paper"]
    lf = {}
    for kind, dev, lk in (("day", dev_day, lock_day), ("week", dev_week, lock_week)):
        X = data.flatten(dev)
        mean, std = data.fit_scaler(X)
        train = data.Part(data.apply_scaler(X, mean, std), dev.y.astype(np.int64), dev.athlete_id)
        rr = data.Round(FINAL_K, kind, train, train, train, data.input_names(dev), mean, std)
        Xl = data.apply_scaler(data.flatten(lk), mean, std)
        scores = [fitters.lovdal_label_free(rr, paper, seed)(Xl) for seed in proc["refit_seeds"]]
        lf[kind] = {"pr_auc_ci": _ci(lk.y.astype(int), scores, lk.athlete_id,
                                     metrics.average_precision),
                    "roc_auc_ci": _ci(lk.y.astype(int), scores, lk.athlete_id, metrics.roc_auc)}
    out["lovdal_label_free_paper_split"] = lf
    f = out["fused"]
    out["summary_sentence"] = (
        f"On the {len(out['test_runners'])} locked runners ({out['test_events']} shared days, "
        f"{out['test_injuries']} injuries) the declared fused model reached ROC-AUC "
        f"{f['roc_auc_ci']['estimate']:.3f} (95\\% runner bootstrap {f['roc_auc_ci']['low']:.3f}"
        f"--{f['roc_auc_ci']['high']:.3f}) and PR-AUC {f['pr_auc_ci']['estimate']:.3f}; with "
        "only 10 runners this interval understates the uncertainty.")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    code = base_run._code_state()
    t0 = time.perf_counter()
    out = run(args.dry_run)
    phase = "lockbox_dry_run" if args.dry_run else "lockbox"
    folder = loaders.ROOT / "experiments" / phase / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder.mkdir(parents=True)
    info = {**code, "runtime_s": round(time.perf_counter() - t0, 1),
            "data_fingerprints": base_run._fingerprints(), "dry_run": args.dry_run}
    (folder / "run_info.json").write_text(json.dumps(info, indent=2), "utf-8")
    (folder / "metrics.json").write_text(json.dumps(out, indent=2, default=float), "utf-8")
    print(out["label"])
    print(out["summary_sentence"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
