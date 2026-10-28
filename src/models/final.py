"""Final model and the ONE locked test (D-017, D-021, D-023(11), D-030): python -m src.models.final.

    python -m src.models.final --dry-run [--quick]  # whole path on a development round; no lockbox
    python -m src.models.final                      # only at the eval-freeze commit (lockbox.py)
    python -m src.models.final --summarise RUN      # recompute metrics.json from the saved scores

Final model: the fused procedure (src.models.fused.fit_pipeline, identical to the development
rounds) on the final split of the development runners (train 51 / validation 13; D-017), search
seed 1000 + 5. It is scored once on the shared days of the locked runners 64-73 (primary), next to
the 5-round headline, never instead of it. Secondary, scored once and never used to choose
anything: the chosen arms on their own files, and the Lövdal label-free variant trained on runners
0-63 (paper_split), threshold-free only (D-030). Uncertainty: runner bootstrap over the 10 locked
runners (2,000 resamples), which understates uncertainty with so few runners (disclosed).

Safety: run_info and the configs are written before anything is trained; the locked rows are
loaded only after every model is trained; every raw score (validation and locked rows, with the
row indices) is saved before any number is computed, first for the primary model and again after
the secondaries. All reported numbers come from those files (--summarise), so a failure after the
lockbox opened never loses the scoring and never needs a second opening.
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
from types import SimpleNamespace

import numpy as np

from src.data import loaders, lockbox
from src.eval import bootstrap, metrics
from src.models import data, evaluate, fitters, fused
from src.models import run as base_run

OUT = loaders.ROOT / "experiments"
FINAL_K = 5  # search seed 1000 + 5: the final model's own seeded random search (D-030)
BOOT_SEED = 0
REASON = "final evaluation of the declared fused model (D-021, D-023(11), D-030)"
SECONDARY = "secondary: scored once, never used to choose anything (D-030)"


def _shared(day: loaders.RunnerData, week: loaders.RunnerData):
    i_day, i_week = loaders.shared_days(day, week)
    return (day.X[i_day], data.flatten(week)[i_week], day.y[i_day].astype(np.int64),
            day.athlete_id[i_day], i_day, i_week)


def _fit_round(kind, X_dev, y_dev, p_dev, fold, names) -> data.Round:
    """Train/validation from the development runners (one scaler on training rows only). The
    test slot holds the validation rows while the models train (fit_pipeline needs one and never
    uses it for a choice): the locked rows are loaded only after training."""
    masks = data.split_rows(p_dev, {**fold, "test_ids": []})
    mean, std = data.fit_scaler(X_dev[masks["train"]])

    def part(m):
        return data.Part(data.apply_scaler(X_dev[m], mean, std), y_dev[m], p_dev[m])

    val = part(masks["val"])
    return data.Round(FINAL_K, kind, part(masks["train"]), val, val, names, mean, std)


def _write(folder: Path, facts: dict, raw: dict) -> None:
    (folder / "predictions").mkdir(exist_ok=True)  # git-ignored, like every run's predictions
    np.savez_compressed(folder / "predictions" / "predictions.npz", **raw)
    (folder / "scoring.json").write_text(json.dumps(facts, indent=1, default=float), "utf-8")


def score_once(folder: Path, dry_run: bool, quick: bool) -> tuple[dict, dict]:
    """Train the final model, then load the locked rows and score them ONCE.
    Returns (scoring facts, raw scores); both are saved in `folder` as soon as they exist."""
    split = loaders.lovdal_split()
    dev_day, dev_week = loaders.load_dev("day"), loaders.load_dev("week")
    names_seq, names_tab = data.input_names(dev_day), data.input_names(dev_week)
    data.check_declared_inputs(names_seq, "day")
    data.check_declared_inputs(names_tab, "week")
    if dry_run:  # development round 1 stands in: its test group plays the locked runners
        stand_in = split["folds"]["1"]["test_ids"]
        lock_day = loaders.select(dev_day, np.isin(dev_day.athlete_id, stand_in))
        lock_week = loaders.select(dev_week, np.isin(dev_week.athlete_id, stand_in))
        dev_day = loaders.select(dev_day, ~np.isin(dev_day.athlete_id, stand_in))
        dev_week = loaders.select(dev_week, ~np.isin(dev_week.athlete_id, stand_in))
        f1 = split["folds"]["1"]
        fold = {"train_ids": f1["train_ids"], "val_ids": f1["val_ids"]}
    else:
        fold = {"train_ids": split["final"]["train_ids"], "val_ids": split["final"]["val_ids"]}
    Xs, Xt, y, p, _, _ = _shared(dev_day, dev_week)
    r_seq = _fit_round("day", Xs, y, p, fold, names_seq)
    r_tab = _fit_round("week", Xt, y, p, fold, names_tab)
    fcfg, bcfg, proc, cands = fused.configs()
    if quick:  # dry runs only: the whole path with the smallest budget
        proc = {**proc, "settings_per_round": 1, "refit_seeds": [0]}
    fit = fused.fit_pipeline(r_seq, r_tab, proc, cands, fcfg)
    print("final model trained; scoring the locked rows", flush=True)

    if not dry_run:  # every model is trained: only now are the locked rows loaded
        lock_day = lockbox.open_lockbox("day", REASON)
        lock_week = lockbox.open_lockbox("week", REASON)
    if data.input_names(lock_day) != names_seq or data.input_names(lock_week) != names_tab:
        raise ValueError("locked files do not have the development files' inputs")
    Ls, Lt, ly, lp, i_day, i_week = _shared(lock_day, lock_week)
    Ls = data.apply_scaler(Ls, r_seq.scaler_mean, r_seq.scaler_std)
    Lt = data.apply_scaler(Lt, r_tab.scaler_mean, r_tab.scaler_std)
    raw = {"val_y": r_seq.val.y, "val_person": r_seq.val.person,
           "shared_y": ly, "shared_person": lp, "shared_i_day": i_day, "shared_i_week": i_week,
           "day_y": lock_day.y.astype(int), "day_person": lock_day.athlete_id,
           "week_y": lock_week.y.astype(int), "week_person": lock_week.athlete_id}
    for sd in fit["seeds"]:  # the same steps as fit_pipeline's test rows
        f = fused._features(sd["predict_seq"](Ls), sd["predict_tab"](Lt))
        s = sd["seed"]
        raw[f"fused_val_s{s}"] = sd["s_val"]
        raw[f"fused_s{s}"] = sd["combiner"].predict_proba(f)[:, 1]
        raw[f"seq_val_s{s}"], raw[f"tab_val_s{s}"] = sd["seq_val"], sd["tab_val"]
        raw[f"seq_shared_s{s}"], raw[f"tab_shared_s{s}"] = fused._sigmoid(f[:, 0]), \
            fused._sigmoid(f[:, 1])
    facts = {"label": "DRY RUN on development round 1 (not the lockbox)" if dry_run else
             "LOCKED FINAL TEST: runners 64-73, opened once",
             "quick_budget": quick, "seeds": [sd["seed"] for sd in fit["seeds"]],
             "combiner": [{"seed": sd["seed"], "C": sd["combiner_C"], "coef": sd["combiner_coef"],
                           "trials": sd["combiner_trials"]} for sd in fit["seeds"]],
             "sequence_arm": fit["sequence_arm"], "tabular_arm": fit["tabular_arm"],
             "tries": fit["tries"]}
    _write(folder, facts, raw)  # primary scores safe before any secondary runs
    print("primary scores saved", flush=True)

    # secondary 1: the chosen arms on their own files (all locked rows of that file)
    for arm, rr, d, key, file in (("sequence", r_seq, lock_day, "predict_seq", "day"),
                                  ("tabular", r_tab, lock_week, "predict_tab", "week")):
        X = d.X if arm == "sequence" else data.flatten(d)
        Xn = data.apply_scaler(X, rr.scaler_mean, rr.scaler_std)
        for sd in fit["seeds"]:
            raw[f"own_{arm}_{file}_s{sd['seed']}"] = sd[key](Xn)
    _write(folder, facts, raw)
    # secondary 2: Lövdal label-free variant trained on runners 0-63 (paper_split), threshold-free
    paper = bcfg["models"]["lovdal_xgboost"]["paper"]
    for kind, dev, lk in (("day", dev_day, lock_day), ("week", dev_week, lock_week)):
        X = data.flatten(dev)
        mean, std = data.fit_scaler(X)
        train = data.Part(data.apply_scaler(X, mean, std), dev.y.astype(np.int64), dev.athlete_id)
        rr = data.Round(FINAL_K, kind, train, train, train, data.input_names(dev), mean, std)
        Xl = data.apply_scaler(data.flatten(lk), mean, std)
        for seed in proc["refit_seeds"]:
            raw[f"lovdal_{kind}_s{seed}"] = fitters.lovdal_label_free(rr, paper, seed)(Xl)
    _write(folder, facts, raw)
    return facts, raw


def _ci(y, scores_by_seed, person, metric) -> dict:
    """Mean over seeds of the metric, runner bootstrap (2,000 resamples) over the test runners."""
    r = bootstrap.reproduction_check(y, scores_by_seed, person, metric, resamples=2000,
                                     seed=BOOT_SEED)
    return {**{k: r[k] for k in ("estimate", "low", "high", "resamples_used",
                                 "resamples_discarded")}, "bootstrap_seed": BOOT_SEED}


def _both(y, scores, person) -> dict:
    out = {"events": int(len(y)), "injuries": int(np.sum(y)),
           "runners_with_injury": int(len(np.unique(np.asarray(person)[np.asarray(y) == 1])))}
    for name, metric in (("pr_auc", metrics.average_precision), ("roc_auc", metrics.roc_auc)):
        try:
            out[f"{name}_ci"] = _ci(y, scores, person, metric)
        except ValueError as e:  # e.g. too few injured runners for a bootstrap: say so
            out[f"{name}_ci"] = {"error": str(e)}
    return out


def summarise(facts: dict, raw: dict) -> dict:
    """All reported numbers, from the saved raw scores (rerun without the lockbox)."""
    seeds = facts["seeds"]
    rows = SimpleNamespace(val=SimpleNamespace(y=raw["val_y"]),
                           test=SimpleNamespace(y=raw["shared_y"]))
    per_seed = [fused.score_seed(rows, {"s_val": raw[f"fused_val_s{s}"],
                                        "s_test": raw[f"fused_s{s}"],
                                        "seq_test": raw[f"seq_shared_s{s}"],
                                        "tab_test": raw[f"tab_shared_s{s}"],
                                        "combiner_C": c["C"], "combiner_coef": c["coef"]})
                for s, c in zip(seeds, facts["combiner"], strict=True)]
    out = {k: facts[k] for k in ("label", "quick_budget", "sequence_arm", "tabular_arm", "tries")}
    out["test_runners"] = sorted(int(v) for v in np.unique(raw["shared_person"]))
    out["fused"] = {"role": "primary: the model declared for the locked test (D-021, D-023)",
                    **evaluate.mean_over_seeds(per_seed), "seeds": per_seed,
                    **_both(raw["shared_y"], [raw[f"fused_s{s}"] for s in seeds],
                            raw["shared_person"])}
    out["arms_on_own_files"] = {
        arm: {"role": SECONDARY, "file": file,
              **_both(raw[f"{file}_y"], [raw[f"own_{arm}_{file}_s{s}"] for s in seeds],
                      raw[f"{file}_person"])}
        for arm, file in (("sequence", "day"), ("tabular", "week"))}
    out["lovdal_label_free_paper_split"] = {
        kind: {"role": SECONDARY + "; threshold-free only; trained on runners 0-63 with "
                                   "training-rows scaling, so not the paper's own number",
               **_both(raw[f"{kind}_y"], [raw[f"lovdal_{kind}_s{s}"] for s in seeds],
                       raw[f"{kind}_person"])} for kind in ("day", "week")}
    out["not_a_comparison"] = ("fused (shared days) and the secondaries (own files) are scored on "
                               "different rows; no paired comparison is made here")
    f = out["fused"]
    roc = f.get("roc_auc_ci", {})
    if "estimate" in roc:
        n = len(out["test_runners"])
        out["summary_sentence"] = (
            f"On the {n} locked runners ({f['events']} shared days, {f['injuries']} injuries) "
            f"the declared fused model reached ROC-AUC {roc['estimate']:.3f} (95% runner "
            f"bootstrap {roc['low']:.3f}-{roc['high']:.3f}) and PR-AUC "
            f"{f['pr_auc_ci'].get('estimate', float('nan')):.3f}; with only {n} runners this "
            "interval understates the uncertainty.")
    return out


def _run_info(argv: list[str], dry_run: bool, quick: bool) -> dict:
    proc = fused.configs()[2]
    return {**base_run._code_state(),
            "command": "python -m src.models.final " + " ".join(argv),
            "dry_run": dry_run, "quick_budget": quick, "python": platform.python_version(),
            "packages": {p: version(p) for p in ("numpy", "scikit-learn", "scipy", "PyYAML",
                                                 "xgboost", "catboost", "torch")},
            "seeds": {"search_seed": proc["search_seed_base"] + FINAL_K,
                      "tune_seed": proc["tune_seed"],
                      "refit_seeds": [0] if quick else proc["refit_seeds"],
                      "bootstrap_seed": BOOT_SEED},
            "cpu_count": os.cpu_count(), "data_fingerprints": base_run._fingerprints()}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quick", action="store_true", help="dry runs only: 1 setting, 1 seed")
    p.add_argument("--summarise", help="run id: recompute metrics.json from its saved scores")
    args = p.parse_args(argv)
    if args.quick and not args.dry_run:
        p.error("--quick is only for --dry-run")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.summarise:
        folder = Path(args.summarise)
        folder = folder if folder.is_absolute() else OUT / "lockbox" / args.summarise
        facts = json.loads((folder / "scoring.json").read_text("utf-8"))
        with np.load(folder / "predictions" / "predictions.npz") as z:
            raw = {k: z[k] for k in z.files}
    else:
        info = _run_info(argv, args.dry_run, args.quick)  # before anything is trained or opened
        phase = "lockbox_dry_run" if args.dry_run else "lockbox"
        folder = OUT / phase / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        folder.mkdir(parents=True)
        shutil.copyfile(fused.FUSED_CFG, folder / "config_fused.yaml")
        shutil.copyfile(fused.BASE_CFG, folder / "config.yaml")
        (folder / "run_info.json").write_text(json.dumps(info, indent=2), "utf-8")
        t0 = time.perf_counter()
        facts, raw = score_once(folder, args.dry_run, args.quick)
        info["runtime_s"] = round(time.perf_counter() - t0, 1)
        (folder / "run_info.json").write_text(json.dumps(info, indent=2), "utf-8")
        print(f"scores saved in {folder}", flush=True)
    out = summarise(facts, raw)
    (folder / "metrics.json").write_text(json.dumps(out, indent=2, default=float), "utf-8")
    summ = {**base_run._code_state(), "time": datetime.now().astimezone().isoformat(
        timespec="seconds")}
    (folder / "summarise_info.json").write_text(json.dumps(summ, indent=2), "utf-8")
    print(out["label"])
    print(out.get("summary_sentence", "summary: see metrics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
