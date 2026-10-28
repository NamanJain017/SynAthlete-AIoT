"""Piece-5 runner: python -m src.gen.run pamap2 | followup --from RUN_ID  (tasks.py gen).

pamap2:   4 PAMAP2 rounds in parallel processes. Per round and generator: 8 settings chosen by
          validation TSTR macro-F1 (test people not looked at), then the chosen setting with
          seeds 0-2 scored on the round's real test people: discriminative accuracy,
          autocorrelation / correlation error, privacy, TSTR and real+synthetic macro-F1.
followup: WESAD generalisation (fidelity only, 5 folds in parallel), Sport DB 2.0 heart-rate
          realism (descriptive), Lövdal injury augmentation (5 rounds, training rows only).
Writes experiments/generators/<run_id>/: config.yaml, run_info.json, metrics.json, summary.md;
synthetic samples and per-round files stay local (git-ignored samples/).
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import itertools
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.data import loaders
from src.models import data, fitters, nets, tuning
from src.models import evaluate as inj_eval
from src.models import run as base_run

ROOT = loaders.ROOT
CONFIG = ROOT / "configs" / "gen" / "generators.yaml"
OUT = ROOT / "experiments" / "generators"
PIECE3_GPU_RUN = ROOT / "experiments" / "baselines" / "2026-10-26_022902" / "metrics.json"
GENERATORS = ("timegan", "wgan_gp", "timevae")


def _cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text("utf-8"))


def _piece3_choice(model: str, k: int) -> tuple[dict, list[float] | None]:
    """The round's piece-3 chosen setting (and its test scores), so nothing is re-tuned."""
    res = json.loads(PIECE3_GPU_RUN.read_text("utf-8"))[model]
    rnd = (res["rounds"] if "rounds" in res else res["day"]["rounds"])[k]
    return rnd["trials"][rnd["chosen"]]["setting"], rnd.get("macro_f1_per_seed")


def _device():
    return nets.pick_device()


def _rounds(name: str):
    """PAMAP2 rounds from the round builder; WESAD rounds built the same way (D-020 splits)."""
    if name == "pamap2":
        return data.pamap2_rounds()
    with np.load(loaders.PROCESSED / "wesad" / "windows.npz") as z:
        X, y, subject = z["X"], z["y"].astype(np.int64), z["subject"]
    meta = json.loads((loaders.PROCESSED / "wesad" / "windows_meta.json").read_text("utf-8"))
    split = json.loads((loaders.SPLITS / "wesad_folds.json").read_text("utf-8"))
    return (data.make_round(int(k), "wesad", X, y, subject, split["folds"][k], meta["channels"])
            for k in sorted(split["folds"], key=int))  # one fold at a time


def _free() -> None:
    """Release finished models now: without this, old generators piled up in GPU memory, the
    GPU spilled into RAM and Windows swapped to disk (each setting got slower; D-029)."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


GPU_SHARE = 0.30  # per worker: the largest TimeGAN needs ~1.36 GB; 0.23 (1.38 GB) was too tight
RETRIES = 4       # a brief out-of-memory moment waits and retries instead of crashing the run


def _retry(fn, *args, **kwargs):
    """Run fn; on a CUDA out-of-memory error free memory, wait and try again (same inputs and
    seed, so the result is the same computation). The last attempt's error is raised."""
    import torch
    for attempt in range(1, RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except torch.OutOfMemoryError:
            if attempt == RETRIES:
                raise
            _free()
            print(f"  [{datetime.now():%H:%M}] out of GPU memory: retry {attempt} in 60 s",
                  flush=True)
            time.sleep(60)


def _save_json(path: Path, obj) -> None:
    """Checkpoint written atomically, so a crash never leaves half a file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), "utf-8")
    tmp.replace(path)


def _limit_gpu_memory() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_SHARE)


def _round_k(rounds, k: int) -> data.Round:
    """Only round k (the rounds are built one at a time; earlier ones are dropped)."""
    return next(itertools.islice(rounds, k, None))


def _progress(what: str, t0: float) -> None:
    """One line per finished step, so the real speed is visible within minutes."""
    print(f"  [{datetime.now():%H:%M}] {what}: {(time.perf_counter() - t0) / 60:.1f} min",
          flush=True)


def _fidelity(r: data.Round, syn: np.ndarray, seed: int, device) -> dict:
    from src.gen import evaluate as ge
    return {"discriminative_accuracy": ge.discriminative_accuracy(r.test.X, syn, seed, device),
            "acf_error": ge.acf_error(r.test.X, syn), "correlation_error":
            ge.correlation_error(r.test.X, syn), **ge.privacy(r.train.X, r.test.X, syn, device)}


def pamap2_round(k: int, folder: str) -> dict:
    """One PAMAP2 round, run in its own process."""
    import torch

    from src.gen import evaluate as ge
    from src.gen import models

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 4))
    cfg, device = _cfg(), _device()
    _limit_gpu_memory()
    proc = cfg["procedure"]
    r = _round_k(data.pamap2_rounds(), k)
    classes = np.unique(r.train.y)
    y_idx = np.searchsorted(classes, r.train.y)
    cnn, real_f1 = _piece3_choice("pamap2_cnn1d", k)
    out = {"round": k, "real_only_test_macro_f1": float(np.mean(real_f1)), "generators": {}}
    samples = Path(folder) / "samples"
    samples.mkdir(parents=True, exist_ok=True)

    def score(name, sampler_for_seed):
        per_seed = []
        for seed in proc["seeds"]:
            t0 = time.perf_counter()
            sampler = _retry(sampler_for_seed, seed)
            syn = _retry(sampler, y_idx)
            if seed == proc["seeds"][0]:
                np.savez_compressed(samples / f"pamap2_{name}_r{k}.npz", X=syn[:, :, 0],
                                    y=classes[y_idx], hr_mean=r.scaler_mean[0],
                                    hr_sd=r.scaler_std[0])
            _, tstr = _retry(ge.cnn_macro_f1, syn, classes[y_idx], r, cnn, seed, device)
            _, aug = _retry(ge.cnn_macro_f1, np.concatenate([r.train.X, syn]),
                            np.concatenate([r.train.y, classes[y_idx]]), r, cnn, seed, device)
            per_seed.append({**_retry(_fidelity, r, syn, seed, device),
                             "tstr_test_macro_f1": tstr,
                             "real_plus_synthetic_test_macro_f1": aug})
            _progress(f"round {k} {name} seed {seed} scored", t0)
            del sampler, syn
            _free()
        return {key: float(np.mean([s[key] for s in per_seed])) for key in per_seed[0]} | {
            "seeds": per_seed}

    ckpt = Path(folder) / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    for name in GENERATORS:
        done = ckpt / f"pamap2_r{k}_{name}.json"
        if done.exists():  # finished before a crash: resume instead of redoing it
            out["generators"][name] = json.loads(done.read_text("utf-8"))
            print(f"  [{datetime.now():%H:%M}] round {k} {name}: resumed from checkpoint",
                  flush=True)
            continue
        g = cfg["generators"][name]
        if g.get("tuned", True):
            settings = tuning.draw_settings(g["grid"], proc["settings_per_generator_per_round"],
                                            proc["search_seed_base"] + k, g["fixed"])
        else:  # untuned (D-029): one setting declared in the config before the run
            settings = [{**g["fixed"], **g["declared_setting"]}]
        trials, first = [], None
        for i, s in enumerate(settings, 1):  # choose on the REAL validation person only
            t0 = time.perf_counter()
            sampler = _retry(models.TRAINERS[name], r.train.X, y_idx, len(classes), s, 0, device)
            syn = _retry(sampler, y_idx)
            diverged = not np.isfinite(syn).all()  # blown-up setting: worst score, never chosen
            val = float("-inf") if diverged else _retry(
                ge.cnn_macro_f1, syn, classes[y_idx], r, cnn, 0, device, score_test=False)[0]
            del syn
            trials.append({"setting": s, "val_tstr_macro_f1": val, "diverged": diverged})
            _progress(f"round {k} {name} setting {i}/{len(settings)}", t0)
            scores = [t["val_tstr_macro_f1"] for t in trials]
            if scores.index(max(scores)) == len(trials) - 1:
                first = sampler
            del sampler
            _free()
        best = [t["val_tstr_macro_f1"] for t in trials]
        best = best.index(max(best))
        chosen = settings[best]
        res = score(name, lambda seed, c=chosen, f=first, n=name: f if seed == proc["seeds"][0]
                    else models.TRAINERS[n](r.train.X, y_idx, len(classes), c, seed, device))
        out["generators"][name] = {**res, "chosen": best, "trials": trials}
        _save_json(done, out["generators"][name])
        print(f"  pamap2 round {k} {name}: TSTR {res['tstr_test_macro_f1']:.3f}", flush=True)
        del first
        _free()
    done = ckpt / f"pamap2_r{k}_classic.json"
    if done.exists():
        out["generators"]["classic"] = json.loads(done.read_text("utf-8"))
    else:
        c = cfg["generators"]["classic"]["fixed"]
        out["generators"]["classic"] = score(
            "classic", lambda seed: models.classic_sampler(r.train.X, y_idx, c, seed))
        _save_json(done, out["generators"]["classic"])
    return out


def _consensus(rounds: list[dict], name: str) -> dict:
    """D-027 'most often chosen PAMAP2 setting': per key the most common value (median for
    continuous ones) of the 4 rounds' chosen settings."""
    chosen = [rd["generators"][name]["trials"][rd["generators"][name]["chosen"]]["setting"]
              for rd in rounds]
    out = {}
    for key in chosen[0]:
        vals = [c[key] for c in chosen]
        if all(isinstance(v, float) for v in vals) and len(set(vals)) == len(vals):
            out[key] = float(np.median(vals))
        else:
            out[key] = Counter(map(json.dumps, vals)).most_common(1)[0][0]
            out[key] = json.loads(out[key])
    return out


def wesad_fold(k: int, settings: dict, folder: str) -> dict:
    import torch

    from src.gen import models
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 4))
    device = _device()
    _limit_gpu_memory()
    r = _round_k(_rounds("wesad"), k)
    classes = np.unique(r.train.y)
    y_idx = np.searchsorted(classes, r.train.y)
    out = {"fold": k, "generators": {}}
    ckpt = Path(folder) / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    for name in GENERATORS:
        done = ckpt / f"wesad_f{k}_{name}.json"
        if done.exists():
            out["generators"][name] = json.loads(done.read_text("utf-8"))
            continue
        per_seed = []
        for seed in _cfg()["procedure"]["seeds"]:
            t0 = time.perf_counter()
            sampler = _retry(models.TRAINERS[name], r.train.X, y_idx, len(classes),
                             settings[name], seed, device)
            syn = _retry(sampler, y_idx)
            per_seed.append(_retry(_fidelity, r, syn, seed, device))
            _progress(f"wesad fold {k} {name} seed {seed}", t0)
            del sampler, syn
            _free()
        out["generators"][name] = {key: float(np.mean([s[key] for s in per_seed]))
                                   for key in per_seed[0]}
        _save_json(done, out["generators"][name])
    return out


def injury_round(k: int, setting: dict, folder: str) -> dict:
    """Lövdal round k: GRU (piece-3 setting) on real training rows plus synthetic injury
    examples (TimeGAN or classic), training rows only; validation/test stay real."""
    import torch

    from src.gen import models
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 4))
    cfg, device = _cfg(), _device()
    _limit_gpu_memory()
    r = _round_k(data.lovdal_rounds("day", "sequence"), k)
    gru_setting, _ = _piece3_choice("gru", k)
    pos = r.train.y == 1
    n_pos = int(pos.sum())
    out = {"round": k, "synthetic_injuries_added": n_pos}
    ckpt = Path(folder) / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    for name in ("timegan", "classic"):
        done = ckpt / f"injury_r{k}_{name}.json"
        if done.exists():
            out[name] = json.loads(done.read_text("utf-8"))
            continue
        per_seed = []
        for seed in cfg["procedure"]["seeds"]:
            t0 = time.perf_counter()
            if name == "timegan":
                sampler = _retry(models.TRAINERS["timegan"], r.train.X, r.train.y, 2, setting,
                                 seed, device)
            else:
                sampler = models.classic_sampler(r.train.X, r.train.y,
                                                 cfg["generators"]["classic"]["fixed"], seed)
            syn = _retry(sampler, np.ones(n_pos, dtype=np.int64))
            train = data.Part(np.concatenate([r.train.X, syn]),
                              np.concatenate([r.train.y, np.ones(n_pos, dtype=r.train.y.dtype)]),
                              np.concatenate([r.train.person, np.full(n_pos, -1)]))
            predict = _retry(fitters.gru, dataclasses.replace(r, train=train), gru_setting, seed)
            per_seed.append(inj_eval.injury_scores(r.val.y, predict(r.val.X), r.test.y,
                                                   predict(r.test.X), True))
            _progress(f"injury round {k} {name} seed {seed}", t0)
            del sampler, predict
            _free()
        out[name] = {**inj_eval.mean_over_seeds(per_seed), "seeds": per_seed}
        _save_json(done, out[name])
    return out


def sportdb_realism(folder: Path) -> dict:
    """Descriptive: synthetic heart rate (bpm, PAMAP2 generators) vs Sport DB 2.0 real HR."""
    import pandas as pd
    from scipy.stats import ks_2samp
    hr = pd.read_parquet(loaders.PROCESSED / "sportdb2" / "hr.parquet")
    real = hr.loc[hr["hr_valid"], "hr_bpm"].to_numpy(float)
    out = {"sportdb2_hr": {"n": int(len(real)), "mean": float(real.mean()),
                           "sd": float(real.std())}}
    for f in sorted((folder / "samples").glob("pamap2_*_r*.npz")):
        with np.load(f) as z:
            bpm = (z["X"] * z["hr_sd"] + z["hr_mean"]).ravel()
        name = f.stem.split("_r")[0].replace("pamap2_", "")
        out.setdefault(name, []).append({"mean": float(bpm.mean()), "sd": float(bpm.std()),
                                         "ks_vs_sportdb2": float(ks_2samp(bpm, real).statistic)})
    return out


def _summarise(rounds: list[dict], key: str, names) -> dict:
    out = {}
    for name in names:
        rows = [rd[key][name] for rd in rounds]
        out[name] = {m: {"mean": float(np.mean([x[m] for x in rows])),
                         "sd": float(np.std([x[m] for x in rows], ddof=1))}
                     for m in rows[0] if isinstance(rows[0][m], float)}
    return out


def criteria_check(summary: dict, real_f1: float, crit: dict, tstr: bool = True) -> dict:
    """Pass or fail per generator against the pre-registered criteria (D-027), on round means."""
    out = {}
    for name, m in summary.items():
        checks = {
            "discriminative": abs(m["discriminative_accuracy"]["mean"] - 0.5)
            <= crit["discriminative_accuracy_within_of_0.5"],
            "acf": m["acf_error"]["mean"] < crit["acf_mean_abs_diff_below"],
            "correlation": m["correlation_error"]["mean"] < crit["correlation_mean_abs_diff_below"],
            "privacy": m["share_synthetic_closer"]["mean"]
            <= crit["privacy_max_share_closer_than_real_p5"]}
        if tstr and "tstr_test_macro_f1" in m:
            checks["tstr"] = (real_f1 - m["tstr_test_macro_f1"]["mean"]
                              <= crit["tstr_macro_f1_within_of_real"])
        out[name] = {k: bool(v) for k, v in checks.items()}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["pamap2", "followup"])
    p.add_argument("--from", dest="src", help="run id of the pamap2 stage (for followup)")
    p.add_argument("--resume", help="run id to continue after a crash (finished parts are kept)")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    code = base_run._code_state()
    cfg = _cfg()
    t0 = time.perf_counter()
    if args.stage == "pamap2":
        if args.resume:
            folder = OUT / args.resume
        else:
            folder = OUT / datetime.now().strftime("%Y-%m-%d_%H%M%S")
            folder.mkdir(parents=True)
            shutil.copyfile(CONFIG, folder / "config.yaml")
        # 4 rounds at once: measured faster than 1-3 at a time once strict mode is off (D-029)
        with ProcessPoolExecutor(max_workers=4) as ex:
            rounds = list(ex.map(pamap2_round, range(4), [str(folder)] * 4))
        summary = _summarise(rounds, "generators", (*GENERATORS, "classic"))
        real = float(np.mean([rd["real_only_test_macro_f1"] for rd in rounds]))
        results = {"pamap2_rounds": rounds, "pamap2_summary": summary,
                   "real_only_test_macro_f1": [rd["real_only_test_macro_f1"] for rd in rounds],
                   "criteria": criteria_check(summary, real, cfg["criteria"])}
        name = "metrics_pamap2.json"
    else:
        folder = OUT / args.src
        prev = json.loads((folder / "metrics_pamap2.json").read_text("utf-8"))["pamap2_rounds"]
        settings = {g: _consensus(prev, g) for g in GENERATORS}
        with ProcessPoolExecutor(max_workers=4) as ex:
            wesad = list(ex.map(wesad_fold, range(5), [settings] * 5, [str(folder)] * 5))
            injury = list(ex.map(injury_round, range(5), [settings["timegan"]] * 5,
                                 [str(folder)] * 5))
        results = {"consensus_settings": settings, "wesad_folds": wesad,
                   "wesad_summary": _summarise(wesad, "generators", GENERATORS),
                   "wesad_criteria": criteria_check(_summarise(wesad, "generators", GENERATORS),
                                                    0.0, cfg["criteria"], tstr=False),
                   "sportdb2_hr_realism": sportdb_realism(folder),
                   "injury_rounds": injury,
                   "injury_summary": {n: inj_eval.summarise([rd[n] for rd in injury])
                                      for n in ("timegan", "classic")}}
        name = "metrics_followup.json"
    info = {"stage": args.stage, **code, "runtime_s": round(time.perf_counter() - t0, 1),
            "seeds": cfg["procedure"]["seeds"], "data_fingerprints": base_run._fingerprints()}
    (folder / f"run_info_{args.stage}.json").write_text(json.dumps(info, indent=2), "utf-8")
    (folder / name).write_text(json.dumps(results, indent=2, default=float), "utf-8")
    print(f"wrote {folder / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
