"""Piece-8 runner: python -m src.xai.run --fused RUN_ID  (tasks.py explain).

Per development round (in parallel processes), the fused run's chosen arms are refit with
seed 0 on that round's training runners and explained on its test events (all injuries plus
500 random healthy events): TreeSHAP (tabular arm), integrated gradients (sequence arm), the
attention view (explanation-only GRU), the three checks (deletion, randomisation, stability),
agreement with the pre-written prior list, and plain-language reasons for the riskiest events.
Also: 1D Grad-CAM on the PAMAP2 activity CNN (round 0), and SIRP-600 TreeSHAP as a code demo
(unknown origin, never evidence; D-013). Writes experiments/explain/<run_id>/.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from src.data import loaders
from src.models import data, fitters, nets
from src.models import run as base_run
from src.xai import checks, explain

ROOT = loaders.ROOT
OUT = ROOT / "experiments" / "explain"
P3_GPU = ROOT / "experiments" / "baselines" / "2026-10-26_022902" / "metrics.json"
HEALTHY_SAMPLE = 500

# Pre-written prior list (EVAL_PROTOCOL): high recent load -> higher risk; low recovery -> higher
# risk; monotony/strain have no input of their own in these files (not testable here).
PRIOR = {"recent_load_raises_risk": (("km", "sessions", "tough", "interval"), +1),
         "low_recovery_raises_risk": (("recovery",), -1)}


def _explain_rows(r: data.Round, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(r.test.y == 1), np.flatnonzero(r.test.y == 0)
    return np.sort(np.r_[pos, rng.choice(neg, min(HEALTHY_SAMPLE, len(neg)), replace=False)])


def prior_agreement(names: list[str], X: np.ndarray, A: np.ndarray) -> dict:
    """Per prior: share of matching features whose attribution moves with the feature value in
    the expected direction (Spearman sign), only for the most recent step ([d-1] / [w-1])."""
    out = {}
    for prior, (words, sign) in PRIOR.items():
        idx = [i for i, n in enumerate(names) if any(w in n.lower() for w in words)
               and ("[d-1]" in n or "[w-1]" in n or n.startswith("rel "))]
        rhos = [spearmanr(X[:, i], A[:, i]).statistic for i in idx]
        rhos = [r for r in rhos if np.isfinite(r)]
        out[prior] = {"features": [names[i] for i in idx], "rank_correlations": rhos,
                      "share_in_expected_direction": float(np.mean([np.sign(r) == sign
                                                                    for r in rhos]))
                      if rhos else None}
    out["monotony_strain"] = "not testable: no monotony or strain input in the Lövdal files"
    return out


def explain_round(k: int, fused_run: str) -> dict:
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 5) // 5))
    fused = json.loads((ROOT / "experiments" / "fused" / fused_run / "metrics.json")
                       .read_text("utf-8"))["fused"]["rounds"][k]
    seq, tab = fused["sequence_arm"], fused["tabular_arm"]
    r_seq, r_tab = list(data.lovdal_shared_rounds())[k]
    rows = _explain_rows(r_seq)
    out = {"round": k, "sequence_arm": seq["model"], "tabular_arm": tab["model"],
           "explained_events": int(len(rows)), "explained_injuries": int(r_seq.test.y[rows].sum())}

    # tabular arm: TreeSHAP + checks
    build = (fitters.random_forest_model if tab["model"] == "random_forest"
             else fitters.catboost_model)
    model = build(r_tab, tab["setting"], 0)
    Xt = r_tab.test.X[rows]
    shap_v = explain.tree_shap(model, Xt)
    predict_t = lambda X: model.predict_proba(X)[:, 1]  # noqa: E731
    shuffled = dataclasses.replace(r_tab, train=data.Part(
        r_tab.train.X, np.random.default_rng(1).permutation(r_tab.train.y), r_tab.train.person))
    shap_rand = explain.tree_shap(build(shuffled, tab["setting"], 0), Xt)
    out["tabular"] = {
        "global_importance": dict(sorted(zip(r_tab.input_names, np.abs(shap_v).mean(0).tolist(),
                                             strict=True), key=lambda kv: -kv[1])[:15]),
        "deletion": checks.deletion(predict_t, Xt, shap_v),
        "randomisation": checks.randomisation(shap_v, shap_rand),
        "stability": checks.stability(lambda X: explain.tree_shap(model, X), Xt, shap_v),
        "prior_agreement": prior_agreement(r_tab.input_names, Xt, shap_v)}

    # sequence arm: integrated gradients + checks
    net, device = fitters.sequence_model(seq["model"], r_seq, seq["setting"], 0)
    Xs = r_seq.test.X[rows]
    ig = explain.integrated_gradients(net, Xs, device)
    predict_s = lambda X: nets.predict_scores(net, X, device=device)  # noqa: E731
    nets.seed_everything(99)
    untrained = nets.build_model(seq["model"], Xs.shape[2], 1, seq["setting"])
    ig_rand = explain.integrated_gradients(untrained, Xs, device)
    day_names = [f"{f} [d-{7 - t}]" for t in range(7) for f in
                 loaders.load_runner_data(loaders.PROCESSED / "lovdal", "day").feature_names]
    flat_ig = ig.reshape(len(ig), -1)
    out["sequence"] = {
        "global_importance": dict(sorted(zip(day_names, np.abs(flat_ig).mean(0).tolist(),
                                             strict=True), key=lambda kv: -kv[1])[:15]),
        "per_day_importance": np.abs(ig).sum(axis=2).mean(0).tolist(),
        "deletion": checks.deletion(predict_s, Xs, ig),
        "randomisation": checks.randomisation(ig, ig_rand),
        "stability": checks.stability(lambda X: explain.integrated_gradients(net, X, device),
                                      Xs, ig),
        "prior_agreement": prior_agreement(day_names, Xs.reshape(len(Xs), -1), flat_ig)}

    # attention view (explanation-only model), always next to IG per day (invariant 12)
    gs, _ = base_run_choice("gru", k)
    nets.seed_everything(0)
    attn = explain.AttentionGRU(Xs.shape[2], 1, gs["hidden"], gs["layers"], gs["dropout"])
    res = nets.train_model(attn, "binary", r_seq.train.X, r_seq.train.y, r_seq.val.X, r_seq.val.y,
                           lr=gs["lr"], batch_size=gs["batch_size"], max_epochs=gs["max_epochs"],
                           patience=gs["patience"], seed=0, weight_decay=gs["weight_decay"],
                           pos_weight=nets.positive_weight(r_seq.train.y, gs["pos_weight"]),
                           device=device)
    w = explain.attention_weights(res["model"], Xs, device)
    ig_day = np.abs(ig).sum(axis=2)
    out["attention"] = {"mean_weight_per_day": w.mean(0).tolist(),
                        "rank_correlation_with_ig_per_day": float(np.nanmean(
                            [spearmanr(a, b).statistic for a, b in zip(w, ig_day, strict=True)])),
                        "note": "explanation-only model, never part of the risk score"}

    # plain-language reasons for the 5 riskiest explained events (tabular SHAP, week inputs)
    risk = predict_t(Xt)
    top = np.argsort(-risk)[:5]
    person, label = r_tab.test.person[rows], r_tab.test.y[rows]
    out["reasons"] = [{"runner": int(person[i]), "injury": int(label[i]),
                       "sentence": explain.reason_sentence(r_tab.input_names, Xt[i], shap_v[i])}
                      for i in top]
    print(f"  explain round {k}: done", flush=True)
    return out


def base_run_choice(model: str, k: int):
    res = json.loads(P3_GPU.read_text("utf-8"))[model]
    rnd = (res["rounds"] if "rounds" in res else res["day"]["rounds"])[k]
    return rnd["trials"][rnd["chosen"]]["setting"], rnd


def grad_cam_pamap2() -> dict:
    r = next(data.pamap2_rounds())
    s, _ = base_run_choice("pamap2_cnn1d", 0)
    device = nets.pick_device()
    nets.seed_everything(0)
    classes = np.unique(r.train.y)
    res = nets.train_model(nets.build_model("cnn1d", 10, len(classes), s), "multiclass",
                           r.train.X, r.train.y, r.val.X, r.val.y, lr=s["lr"],
                           batch_size=s["batch_size"], max_epochs=s["max_epochs"],
                           patience=s["patience"], seed=0, device=device)
    rows = np.random.default_rng(0).choice(len(r.test.y), min(500, len(r.test.y)), replace=False)
    X = r.test.X[rows]
    cam = explain.grad_cam_1d(res["model"], X, device)
    nets.seed_everything(99)
    cam_rand = explain.grad_cam_1d(nets.build_model("cnn1d", 10, len(classes), s), X, device)
    probs = nets.predict_scores(res["model"], X, device=device)
    cls = probs.argmax(1)

    def predict(Xq):  # probability of the originally predicted class
        return nets.predict_scores(res["model"], Xq, device=device)[np.arange(len(Xq)), cls]

    A = np.repeat(cam[:, :, None], X.shape[2], axis=2)  # time importance for every channel
    return {"windows": int(len(rows)), "randomisation": checks.randomisation(cam, cam_rand),
            "deletion_13_steps_x_all_channels": checks.deletion(predict, X, A, k=13 * 10),
            "mean_cam_over_time": cam.mean(0)[::8].tolist()}


def sirp600_demo() -> dict:
    """D-013: shows the RF + TreeSHAP code runs on a snapshot table; unknown origin, no
    person IDs, never evidence."""
    from sklearn.ensemble import RandomForestClassifier
    t = pd.read_parquet(loaders.PROCESSED / "sirp600" / "table.parquet")
    meta = json.loads((loaders.PROCESSED / "sirp600" / "meta.json").read_text("utf-8"))
    label = meta["label"]
    X = t.drop(columns=[label]).select_dtypes("number").fillna(0)
    y = t[label]
    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(X, y)
    sv = explain.tree_shap(rf, X.to_numpy()[:200])
    imp = dict(sorted(zip(X.columns, np.abs(sv).mean(0).tolist(), strict=True),
                      key=lambda kv: -kv[1])[:8])
    return {"label": "demo only: unknown-origin data (D-013), never evidence",
            "target": str(label), "top_features_mean_abs_shap": imp}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fused", required=True, help="run id under experiments/fused/")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    code = base_run._code_state()
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder = OUT / run_id
    folder.mkdir(parents=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=5) as ex:
        rounds = list(ex.map(explain_round, range(5), [args.fused] * 5))
    crit = yaml.safe_load((ROOT / "configs" / "gen" / "generators.yaml").read_text("utf-8"))
    results = {"fused_run": args.fused, "rounds": rounds, "grad_cam_pamap2": grad_cam_pamap2(),
               "sirp600_demo": sirp600_demo(),
               "pass_counts": {arm: {c: int(sum(rd[arm][c]["passed"] for rd in rounds))
                                     for c in ("deletion", "randomisation", "stability")}
                               for arm in ("tabular", "sequence")},
               "criteria_source": "EVAL_PROTOCOL Explanation criteria (D-027)",
               "generator_config_checked": bool(crit)}
    info = {**code, "runtime_s": round(time.perf_counter() - t0, 1),
            "data_fingerprints": base_run._fingerprints()}
    (folder / "run_info.json").write_text(json.dumps(info, indent=2), "utf-8")
    (folder / "metrics.json").write_text(json.dumps(results, indent=2, default=float), "utf-8")
    print(json.dumps(results["pass_counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
