"""Paired model comparisons (D-023(10)): python -m src.paper.compare.

Primary comparisons fixed in advance: each model vs the Lövdal label-free variant on the same
file; the fused model vs each of its chosen arms (shared days). Statistic: difference of the
5-round mean metric, seeds averaged within each model, paired on the same rows; runner bootstrap
within test groups (2,000 resamples, percentile 95 % interval; Field & Welsh 2007).
Inputs: the local per-row prediction files of the runs (git-ignored). A model whose files are
not complete for all 5 rounds is skipped and listed. Writes experiments/comparisons/<run_id>/.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from src.data import loaders
from src.eval import bootstrap, metrics
from src.models import run as base_run

EXP = loaders.ROOT / "experiments"
ROUNDS = range(5)
MODELS = {"day": ("constant_risk", "logistic_regression", "random_forest", "catboost", "gru"),
          "week": ("constant_risk", "acwr_style", "logistic_regression", "random_forest",
                   "catboost")}
REFERENCE = "lovdal_label_free"


def _load(files_by_round: dict[int, list[Path]], key: str = "s_test") -> dict:
    """Concatenate the 5 rounds' test rows; one score array per seed (column `key`)."""
    ys, ps, gs, seeds = [], [], [], None
    for k in ROUNDS:
        zs = [np.load(f) for f in files_by_round[k]]
        ys.append(zs[0]["y_test"])
        ps.append(zs[0]["person_test"])
        gs.append(np.full(len(zs[0]["y_test"]), k))
        cols = [z[key] for z in zs]
        if seeds is None:
            seeds = [[c] for c in cols]
        else:
            seeds = [s + [c] for s, c in zip(seeds, cols, strict=True)]
    return {"y": np.concatenate(ys).astype(int), "person": np.concatenate(ps),
            "group": np.concatenate(gs), "scores": [np.concatenate(s) for s in seeds]}


def find(model: str, kind: str) -> dict | None:
    """Newest baseline run with a complete set (all 5 rounds, same seeds each round)."""
    for run in sorted((EXP / "baselines").glob("20*"), reverse=True):
        pred = run / "predictions"
        by_round = {}
        for k in ROUNDS:
            fs = sorted(pred.glob(f"{model}_{kind}_s*_r{k}.npz")) or \
                sorted(pred.glob(f"{model}_{kind}_r{k}.npz"))
            if fs:
                by_round[k] = fs
        if len(by_round) == 5 and len({len(v) for v in by_round.values()}) == 1:
            return {**_load(by_round), "run": run.name}
    return None


def fused_parts() -> dict | None:
    """Fused model and its two chosen arms, on the same shared-day test rows."""
    for run in sorted((EXP / "fused").glob("20*"), reverse=True):
        files = {k: sorted((run / "predictions").glob(f"fused_s*_r{k}.npz")) for k in ROUNDS}
        if all(files.values()):
            return {part: {**_load(files, key), "run": run.name}
                    for part, key in (("fused", "s_test"), ("sequence arm", "seq_test"),
                                      ("tabular arm", "tab_test"))}
    return None


def compare(job: tuple[str, dict, dict]) -> dict:
    label, a, b = job
    if not (np.array_equal(a["y"], b["y"]) and np.array_equal(a["person"], b["person"])):
        raise ValueError(f"{label}: the two models were not scored on the same rows")
    out = {"comparison": label, "runs": [a["run"], b["run"]]}
    for name, metric in (("pr_auc", metrics.average_precision), ("roc_auc", metrics.roc_auc)):
        r = bootstrap.paired_seed_mean_difference(metric, a["y"], a["scores"], b["scores"],
                                                  a["person"], a["group"], 2000, 0)
        out[name] = {**r, "interval_excludes_0": bool(r["low"] > 0 or r["high"] < 0)}
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    code = base_run._code_state()
    jobs, skipped = [], []
    for kind, models in MODELS.items():
        ref = find(REFERENCE, kind)
        for m in models:
            got = find(m, kind)
            if got is None or ref is None:
                skipped.append(f"{m} ({kind})")
                continue
            jobs.append((f"{m} - {REFERENCE} ({kind})", got, ref))
    fp = fused_parts()
    if fp:
        for arm in ("sequence arm", "tabular arm"):
            jobs.append((f"fused - {arm} (shared days)", fp["fused"], fp[arm]))
    else:
        skipped.append("fused vs arms")
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(compare, jobs))
    folder = EXP / "comparisons" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder.mkdir(parents=True)
    (folder / "run_info.json").write_text(json.dumps({**code, "resamples": 2000}, indent=2),
                                          "utf-8")
    (folder / "metrics.json").write_text(json.dumps(
        {"comparisons": results, "skipped_incomplete": skipped}, indent=2, default=float),
        "utf-8")
    for r in results:
        p = r["pr_auc"]
        print(f"{r['comparison']}: PR-AUC diff {p['estimate']:+.4f} "
              f"[{p['low']:+.4f}, {p['high']:+.4f}]")
    print("skipped (incomplete prediction files):", skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

