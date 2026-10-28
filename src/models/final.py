"""Final model and the ONE locked test (D-017, D-021, D-023(11), D-030): python -m src.models.final.

    python -m src.models.final --dry-run        # whole path on a development round; no lockbox
    python -m src.models.final                  # only at the eval-freeze commit (lockbox.py checks)
    python -m src.models.final --summarise RUN  # recompute the numbers from saved predictions

Final model: the fused procedure (src.models.fused.fit_pipeline, identical to the development
rounds) on the final split of the development runners (train 51 / validation 13; D-017), search
seed 1000 + 5. It is scored once on the shared days of the locked runners 64-73 (primary), next to
the 5-round headline, never instead of it. Secondary, scored once and never used to choose
anything: the chosen arms on their own files, and the Lövdal label-free variant trained on runners
0-63 (paper_split), threshold-free only (D-030). Uncertainty: runner bootstrap over the 10 locked
runners (2,000 resamples), which understates uncertainty with so few runners (disclosed).

Safety: the raw test scores are saved the moment they exist (predictions/ + scoring.json), and
the summary numbers are computed from those files, so a failure after the lockbox opened never
loses the scoring and never needs a second opening.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data import loaders, lockbox
from src.eval import bootstrap, metrics
from src.models import data, evaluate, fitters, fused
from src.models import run as base_run

OUT = loaders.ROOT / "experiments"
FINAL_K = 5  # search seed 1000 + 5: the final model's own seeded random search (D-030)


def _shared(day: loaders.RunnerData, week: loaders.RunnerData):
    i_day, i_week = loaders.shared_days(day, week)
    return (day.X[i_day], data.flatten(week)[i_week], day.y[i_day].astype(np.int64),
            day.athlete_id[i_day])


def _round(kind, X_dev, y_dev, p_dev, X_test, y_test, p_test, fold, names) -> data.Round:
    """Train/validation from the development runners (one scaler on training rows only);
    test = rows scaled with that same scaler."""
    masks = data.split_rows(p_dev, {**fold, "test_ids": []})
    mean, std = data.fit_scaler(X_dev[masks["train"]])

    def part(m):
        return data.Part(data.apply_scaler(X_dev[m], mean, std), y_dev[m], p_dev[m])

    return data.Round(FINAL_K, kind, part(masks["train"]), part(masks["val"]),
                      data.Part(data.apply_scaler(X_test, mean, std), y_test, p_test),
                      names, mean, std)


def score_once(dry_run: bool) -> tuple[dict, dict]:
    """Train the final model and score the test rows ONCE. Returns (scoring facts, raw scores)."""
    split = loaders.lovdal_split()
    dev_day, dev_week = loaders.load_dev("day"), loaders.load_dev("week")
    names_seq, names_tab = data.input_names(dev_day), data.input_names(dev_week)
    data.check_declared_inputs(names_seq, "day")
    data.check_declared_inputs(names_tab, "week")
    if dry_run:  # development round 1 stands in: its test group plays the locked runners
        fold = split["folds"]["1"]
        test_ids = fold["test_ids"]
        lock_day = loaders.select(dev_day, np.isin(dev_day.athlete_id, test_ids))
        lock_week = loaders.select(dev_week, np.isin(dev_week.athlete_id, test_ids))
        dev_day = loaders.select(dev_day, ~np.isin(dev_day.athlete_id, test_ids))
        dev_week = loaders.select(dev_week, ~np.isin(dev_week.athlete_id, test_ids))
        fold = {"train_ids": fold["train_ids"], "val_ids": fold["val_ids"]}
    else:
        fold = {"train_ids": split["final"]["train_ids"], "val_ids": split["final"]["val_ids"]}
        reason = "final evaluation of the declared fused model (D-021, D-023(11), D-030)"
        lock_day = lockbox.open_lockbox("day", reason)
        lock_week = lockbox.open_lockbox("week", reason)
    Xs, Xt, y, p = _shared(dev_day, dev_week)
    Ls, Lt, ly, lp = _shared(lock_day, lock_week)
    r_seq = _round("day", Xs, y, p, Ls, ly, lp, fold, names_seq)
    r_tab = _round("week", Xt, y, p, Lt, ly, lp, fold, names_tab)

    fcfg, bcfg, proc, cands = fused.configs()
    fit = fused.fit_pipeline(r_seq, r_tab, proc, cands, fcfg)
    raw = {"shared_y": ly, "shared_person": lp,
           "day_y": lock_day.y.astype(int), "day_person": lock_day.athlete_id,
           "week_y": lock_week.y.astype(int), "week_person": lock_week.athlete_id}
    for sd in fit["seeds"]:
        raw[f"fused_s{sd['seed']}"] = sd["s_test"]
    # secondary 1: the chosen arms on their own files (all locked rows of that file)
    for arm, rr, d, key, file in (("sequence", r_seq, lock_day, "predict_seq", "day"),
                                  ("tabular", r_tab, lock_week, "predict_tab", "week")):
        X = d.X if arm == "sequence" else data.flatten(d)
        Xn = data.apply_scaler(X, rr.scaler_mean, rr.scaler_std)
        for sd in fit["seeds"]:
            raw[f"own_{arm}_{file}_s{sd['seed']}"] = sd[key](Xn)
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
    facts = {"label": "DRY RUN on development round 1 (not the lockbox)" if dry_run else
             "LOCKED FINAL TEST: runners 64-73, opened once",
             "seeds": [sd["seed"] for sd in fit["seeds"]],
             "fused_per_seed": [fused.score_seed(r_seq, sd) for sd in fit["seeds"]],
             "sequence_arm": fit["sequence_arm"], "tabular_arm": fit["tabular_arm"],
             "tries": fit["tries"]}
    return facts, raw


def _ci(y, scores_by_seed, person, metric) -> dict:
    """Seed-averaged metric with a runner bootstrap (2,000 resamples) over the test runners."""
    r = bootstrap.reproduction_check(y, scores_by_seed, person, metric, resamples=2000)
    return {k: r[k] for k in ("estimate", "low", "high", "resamples_used")}


def _both(y, scores, person) -> dict:
    out = {"events": int(len(y)), "injuries": int(np.sum(y))}
    for name, metric in (("pr_auc", metrics.average_precision), ("roc_auc", metrics.roc_auc)):
        try:
            out[f"{name}_ci"] = _ci(y, scores, person, metric)
        except ValueError as e:  # e.g. too few injured runners for a bootstrap: say so
            out[f"{name}_ci"] = {"error": str(e)}
    return out


def summarise(facts: dict, raw: dict) -> dict:
    """All reported numbers, from the saved raw scores (can be rerun without the lockbox)."""
    seeds = facts["seeds"]
    out = {k: facts[k] for k in ("label", "sequence_arm", "tabular_arm", "tries")}
    out["test_runners"] = sorted(int(v) for v in np.unique(raw["shared_person"]))
    out["fused"] = {**evaluate.mean_over_seeds(facts["fused_per_seed"]),
                    "seeds": facts["fused_per_seed"],
                    **_both(raw["shared_y"], [raw[f"fused_s{s}"] for s in seeds],
                            raw["shared_person"])}
    out["arms_on_own_files"] = {
        arm: {"file": file, **_both(raw[f"{file}_y"],
                                    [raw[f"own_{arm}_{file}_s{s}"] for s in seeds],
                                    raw[f"{file}_person"])}
        for arm, file in (("sequence", "day"), ("tabular", "week"))}
    out["lovdal_label_free_paper_split"] = {
        kind: _both(raw[f"{kind}_y"], [raw[f"lovdal_{kind}_s{s}"] for s in seeds],
                    raw[f"{kind}_person"]) for kind in ("day", "week")}
    f = out["fused"]
    roc = f.get("roc_auc_ci", {})
    if "estimate" in roc:
        out["summary_sentence"] = (
            f"On the {len(out['test_runners'])} locked runners ({f['events']} shared days, "
            f"{f['injuries']} injuries) the declared fused model reached ROC-AUC "
            f"{roc['estimate']:.3f} (95\\% runner bootstrap {roc['low']:.3f}--{roc['high']:.3f})"
            f" and PR-AUC {f['pr_auc_ci'].get('estimate', float('nan')):.3f}; with only "
            f"{len(out['test_runners'])} runners this interval understates the uncertainty.")
    return out


def _write(folder: Path, facts: dict, raw: dict) -> None:
    (folder / "predictions").mkdir(exist_ok=True)  # git-ignored, like every run's predictions
    np.savez_compressed(folder / "predictions" / "predictions.npz", **raw)
    (folder / "scoring.json").write_text(json.dumps(facts, indent=1, default=float), "utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summarise", help="run id: recompute metrics.json from its saved scores")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.summarise:
        folder = Path(args.summarise)
        folder = folder if folder.is_absolute() else OUT / "lockbox" / args.summarise
        facts = json.loads((folder / "scoring.json").read_text("utf-8"))
        with np.load(folder / "predictions" / "predictions.npz") as z:
            raw = {k: z[k] for k in z.files}
    else:
        code = base_run._code_state()  # before anything is written
        phase = "lockbox_dry_run" if args.dry_run else "lockbox"
        folder = OUT / phase / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        folder.mkdir(parents=True)
        t0 = time.perf_counter()
        facts, raw = score_once(args.dry_run)
        _write(folder, facts, raw)  # saved at once: nothing is lost if a later step fails
        info = {**code, "runtime_s": round(time.perf_counter() - t0, 1),
                "data_fingerprints": base_run._fingerprints(), "dry_run": args.dry_run}
        (folder / "run_info.json").write_text(json.dumps(info, indent=2), "utf-8")
        print(f"scores saved in {folder}", flush=True)
    out = summarise(facts, raw)
    (folder / "metrics.json").write_text(json.dumps(out, indent=2, default=float), "utf-8")
    print(out["label"])
    print(out.get("summary_sentence", "summary: see metrics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
