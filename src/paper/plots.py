"""python -m src.paper.plots: result graphs as PNG files in figures/, drawn only from the saved
result files (experiments/**/metrics*.json). No training, no hand-typed numbers."""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data import loaders  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.paper import figures  # noqa: E402

OUT = loaders.ROOT / "figures"
EXP_DIR = loaders.ROOT / "experiments"
NAVY, RED, GREY, GREEN = "#1f3a5f", "#c0392b", "#888888", "#2e7d32"
INJURY = [("constant_risk", "Constant risk"), ("acwr_style", "ACWR rule"),
          ("logistic_regression", "Logistic reg."), ("random_forest", "Random forest"),
          ("catboost", "CatBoost"), ("gru", "GRU"), ("lovdal_label_free", "Lövdal (label-free)")]


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"figures/{name}")


def injury_models() -> None:
    b = figures.baselines()
    rows = [(f"{lab} ({k})", b[(m, k)]) for m, lab in INJURY for k in ("day", "week")
            if (m, k) in b]
    fused = figures._runs("fused")
    if fused:
        rows.append(("FUSED (declared in advance)", fused[-1][1]["fused"]["summary"]))
    names = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, metric, chance, title in ((axes[0], "roc_auc", 0.5, "ROC-AUC"),
                                      (axes[1], "pr_auc", None, "PR-AUC")):
        mean = [r[1][metric]["mean"] for r in rows]
        sd = [r[1][metric]["sd"] for r in rows]
        colors = [RED if n.startswith("FUSED") else NAVY for n in names]
        ax.barh(names, mean, xerr=sd, color=colors, capsize=3)
        if chance:
            ax.axvline(chance, color=GREY, ls="--", lw=1, label="chance")
            ax.set_xlim(0.45, max(mean) + 0.05)
        ax.set_title(f"{title} (mean ± SD, 5 development rounds)")
        ax.invert_yaxis()
    axes[0].legend(loc="lower right")
    fig.suptitle("Injury prediction on runners never seen in training")
    _save(fig, "1_injury_models.png")


def locked_test() -> None:
    lock, fused = figures._runs("lockbox"), figures._runs("fused")
    if not lock or not fused:
        return
    m, s = lock[-1][1], fused[-1][1]["fused"]["summary"]
    f = m["fused"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = ["Fused: 5 development rounds", "Fused: sealed final test (primary)",
              "GRU arm, day file (secondary)", "CatBoost arm, week file (secondary)",
              "Lövdal label-free, day (secondary)", "Lövdal label-free, week (secondary)"]
    cis = [None, f["roc_auc_ci"], m["arms_on_own_files"]["sequence"]["roc_auc_ci"],
           m["arms_on_own_files"]["tabular"]["roc_auc_ci"],
           m["lovdal_label_free_paper_split"]["day"]["roc_auc_ci"],
           m["lovdal_label_free_paper_split"]["week"]["roc_auc_ci"]]
    for i, ci in enumerate(cis):
        if ci is None:
            est, lo, hi = s["roc_auc"]["mean"], s["roc_auc"]["mean"] - s["roc_auc"]["sd"], \
                s["roc_auc"]["mean"] + s["roc_auc"]["sd"]
        else:
            est, lo, hi = ci["estimate"], ci["low"], ci["high"]
        color = RED if i == 1 else (NAVY if i == 0 else GREY)
        ax.errorbar(est, i, xerr=[[est - lo], [hi - est]], fmt="o", color=color, capsize=4)
        ax.text(hi + 0.005, i, f"{est:.3f}", va="center", fontsize=9)
    ax.axvline(0.5, color=GREY, ls="--", lw=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC (development: mean ± SD; sealed test: 95% runner bootstrap)")
    ax.set_title(f"Sealed final test: {len(m['test_runners'])} runners, {f['events']:,} days, "
                 f"{f['injuries']} injuries (secondaries are not a comparison)", fontsize=10)
    _save(fig, "2_locked_final_test.png")


def comparisons() -> None:
    runs = figures._runs("comparisons")
    if not runs:
        return
    comps = runs[-1][1]["comparisons"]
    names = [c["comparison"].replace("_", " ").replace(" - lovdal label free", " vs label-free")
             .replace(" - ", " vs ") for c in comps]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, metric, title in ((axes[0], "pr_auc", "Δ PR-AUC"), (axes[1], "roc_auc", "Δ ROC-AUC")):
        for i, c in enumerate(comps):
            r = c[metric]
            clear = r["low"] > 0 or r["high"] < 0
            ax.errorbar(r["estimate"], i, xerr=[[r["estimate"] - r["low"]],
                                                [r["high"] - r["estimate"]]],
                        fmt="o", capsize=3, color=GREEN if r["low"] > 0 else
                        (RED if clear else GREY))
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"{title} [95% interval]")
    axes[0].set_yticks(range(len(names)), names)
    axes[0].invert_yaxis()
    fig.suptitle("Paired comparisons on the same rows (green = clearly better, red = clearly "
                 f"worse, grey = not clear; {2 * len(comps)} intervals, not adjusted)")
    _save(fig, "3_paired_comparisons.png")


def generators() -> None:
    runs = figures._runs("generators", "metrics_pamap2.json")
    if not runs:
        return
    g = runs[-1][1]
    names = list(g["pamap2_summary"])
    real = g.get("real_only_test_macro_f1")
    real = (real.get("mean") if isinstance(real, dict)
            else float(np.mean(real)) if isinstance(real, list) else real)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    disc = [g["pamap2_summary"][n]["discriminative_accuracy"]["mean"] for n in names]
    axes[0].bar(x, disc, color=NAVY)
    axes[0].axhspan(0.35, 0.65, color=GREEN, alpha=0.15, label="pass zone (0.5 ± 0.15)")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Real-vs-synthetic classifier accuracy (0.5 = fooled)")
    axes[0].legend()
    w = 0.38
    for off, key, lab, col in ((-w / 2, "tstr_test_macro_f1", "trained on synthetic only", GREY),
                               (w / 2, "real_plus_synthetic_test_macro_f1", "real + synthetic",
                                NAVY)):
        axes[1].bar(x + off, [g["pamap2_summary"][n][key]["mean"] for n in names], w,
                    label=lab, color=col)
    if real:
        axes[1].axhline(real, color=RED, ls="--", label=f"real only ({real:.3f})")
    axes[1].set_title("Activity recognition macro-F1 on real test people")
    axes[1].legend(fontsize=8)
    labels = [("wgan_gp\n(untuned)" if n == "wgan_gp" else n) for n in names]
    for ax in axes:
        ax.set_xticks(x, labels)
    fig.suptitle("Synthetic wearable data (PAMAP2, 4 rounds)")
    _save(fig, "4_generators.png")


def tiers() -> None:
    runs = figures._runs("tiers")
    if not runs:
        return
    st = runs[-1][1]["streams"]
    names = {"pamap2": "PAMAP2 wearable", "sportdb2_hr": "Sport DB heart rate"}
    x = np.arange(len(st))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for off, mode, col in ((-w / 2, "tiered", GREEN), (w / 2, "cloud_only", NAVY)):
        axes[0].bar(x + off, [o[mode]["bytes_into_cloud"] for o in st.values()], w,
                    label=mode.replace("_", "-"), color=col)
        axes[1].bar(x + off, [o[mode]["decision_latency_ms_median"] for o in st.values()], w,
                    label=mode.replace("_", "-"), color=col)
    for i, o in enumerate(st.values()):
        axes[0].text(i, o["cloud_only"]["bytes_into_cloud"] * 1.3,
                     f"{o['bytes_reduction_factor']:.0f}x less", ha="center")
    axes[0].set_yscale("log")
    axes[0].set_ylim(top=max(o["cloud_only"]["bytes_into_cloud"] for o in st.values()) * 6)
    axes[0].set_title("Bytes sent into the cloud (log scale)")
    axes[1].set_title("Median decision time (ms; one laptop, no network delay)")
    for ax in axes:
        ax.set_xticks(x, [names.get(k, k) for k in st])
        ax.legend()
    fig.suptitle("Edge-fog-cloud vs cloud-only (simulated on one laptop)")
    _save(fig, "5_edge_fog_cloud.png")


def explanations() -> None:
    runs = figures._runs("explain")
    if not runs:
        return
    m = runs[-1][1]
    pc = m["pass_counts"]
    tests = ["deletion", "randomisation", "stability"]
    x = np.arange(len(tests))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for off, arm, col in ((-0.2, "tabular", NAVY), (0.2, "sequence", GREEN)):
        axes[0].bar(x + off, [pc[arm][t] for t in tests], 0.4,
                    label=f"{arm} ({'SHAP' if arm == 'tabular' else 'integrated gradients'})",
                    color=col)
    axes[0].set_xticks(x, tests)
    axes[0].set_ylim(0, 5.5)
    axes[0].set_ylabel("rounds passed (of 5)")
    axes[0].set_title("Faithfulness tests passed")
    axes[0].legend(fontsize=8)
    shares: dict[str, list[float]] = {}
    for rd in m["rounds"]:
        for arm in ("tabular", "sequence"):
            for prior, v in rd[arm].get("prior_agreement", {}).items():
                if isinstance(v, dict) and "share_in_expected_direction" in v:
                    shares.setdefault(prior.replace("_", " "), []).append(
                        v["share_in_expected_direction"])
    axes[1].boxplot(list(shares.values()))
    axes[1].set_xticks(range(1, len(shares) + 1), list(shares))
    axes[1].axhline(0.5, color=GREY, ls="--", lw=1)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Expectations written before running:\nshare of inputs in the "
                      "expected direction (10 arm-rounds)")
    fig.suptitle("Explanations: tested, not assumed")
    _save(fig, "6_explanations.png")


def augmentation() -> None:
    runs, b = figures._runs("generators", "metrics_followup.json"), figures.baselines()
    gru = b.get(("gru", "day"))
    if not runs or not gru:
        return
    inj = runs[-1][1]["injury_summary"]
    labels = ["real only", "+ TimeGAN examples", "+ classic augmentation"]
    vals = [gru, inj["timegan"], inj["classic"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in ((axes[0], "pr_auc", "PR-AUC"), (axes[1], "roc_auc", "ROC-AUC")):
        ax.bar(labels, [v[metric]["mean"] for v in vals], yerr=[v[metric]["sd"] for v in vals],
               color=[NAVY, GREY, GREY], capsize=4)
        ax.set_title(f"{title} (mean ± SD, 5 rounds)")
        ax.tick_params(axis="x", labelsize=8)
    axes[1].set_ylim(0.5, 0.7)
    fig.suptitle("Synthetic injury examples (training rows only): no gain claimed")
    _save(fig, "7_injury_augmentation.png")


# ---------------------------------------------------------------- from saved per-row predictions
# These read the git-ignored prediction files (experiments/**/predictions/), so they can only be
# redrawn on the laptop that ran the experiments. Curves pool the test rows of the 5 rounds
# (seed 0); the reported AUCs are per-round means (graph 1).
DAY = [("logistic_regression", "Logistic reg."), ("random_forest", "Random forest"),
       ("catboost", "CatBoost"), ("gru", "GRU"), ("lovdal_label_free", "Lövdal (label-free)")]
WEEK = [("acwr_style", "ACWR rule"), ("logistic_regression", "Logistic reg."),
        ("random_forest", "Random forest"), ("catboost", "CatBoost"),
        ("lovdal_label_free", "Lövdal (label-free)")]


def _fused_round_files(seed: int = 0) -> list:
    fused = figures._runs("fused")
    if not fused:
        return []
    pred = EXP_DIR / "fused" / fused[-1][0] / "predictions"
    files = [pred / f"fused_s{seed}_r{k}.npz" for k in range(5)]
    return [np.load(f) for f in files] if all(f.exists() for f in files) else []


def roc_pr_curves() -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    from src.paper import compare
    fz = _fused_round_files()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row, (kind, models) in enumerate((("day", DAY), ("week", WEEK))):
        curves = [(lab, compare.find(m, kind)) for m, lab in models]
        if kind == "day" and fz:
            curves.append(("FUSED (shared days)", {
                "y": np.concatenate([z["y_test"] for z in fz]),
                "scores": [np.concatenate([z["s_test"] for z in fz])]}))
        rate = None
        for lab, c in curves:
            if not c:
                continue
            y, s = np.asarray(c["y"]).astype(int), np.asarray(c["scores"][0])
            fpr, tpr, _ = roc_curve(y, s)
            prec, rec, _ = precision_recall_curve(y, s)
            style = dict(color="black", lw=2.4) if lab.startswith("FUSED") else dict(lw=1.3)
            axes[row, 0].plot(fpr, tpr, label=lab, **style)
            axes[row, 1].plot(rec, prec, label=lab, **style)
            rate = y.mean()
        axes[row, 0].plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="chance")
        if rate:
            axes[row, 1].axhline(rate, color=GREY, ls="--", lw=1,
                                 label=f"chance (injury rate {rate:.3f})")
        axes[row, 0].set(xlabel="False-alarm rate (1 - specificity)",
                         ylabel="Injuries caught (sensitivity)",
                         title=f"ROC curves, {kind} file")
        axes[row, 1].set(xlabel="Injuries caught (recall)", ylabel="Precision",
                         title=f"Precision-recall curves, {kind} file", ylim=(0, 0.12))
        for ax in axes[row]:
            ax.legend(fontsize=8)
    fig.suptitle("Injury models on unseen runners (test rows of the 5 rounds pooled, seed 0)")
    _save(fig, "8_roc_pr_curves.png")


def _confusion(y, s, thr) -> np.ndarray:
    alert = s >= thr  # alert if score >= the level chosen on validation (EVAL_PROTOCOL)
    return np.array([[np.sum(~alert & (y == 0)), np.sum(alert & (y == 0))],
                     [np.sum(~alert & (y == 1)), np.sum(alert & (y == 1))]])


def _draw_confusion(ax, cm, title) -> None:
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    names = [["healthy day, no alert\n(correct)", "healthy day, ALERT\n(false alarm)"],
             ["injury, no alert\n(missed)", "injury, ALERT\n(caught)"]]
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]:,}\n{share:.0%} of row\n{names[i][j]}", ha="center",
                    va="center", fontsize=9, color="white" if share > 0.5 else "black")
    ax.set_xticks([0, 1], ["no alert", "alert"])
    ax.set_yticks([0, 1], ["healthy", "injury"])
    ax.set_xlabel("Model decision")
    ax.set_ylabel("What happened")
    ax.set_title(title, fontsize=10)


def confusion_injury() -> None:
    fz = _fused_round_files()
    lock = figures._runs("lockbox")
    panels = []
    if fz:
        cm = sum(_confusion(z["y_test"], z["s_test"],
                            metrics.choose_alert_threshold(z["y_val"], z["s_val"])) for z in fz)
        panels.append((cm, "Fused model, 5 development rounds (seed 0)"))
    if lock:
        p = EXP_DIR / "lockbox" / lock[-1][0] / "predictions" / "predictions.npz"
        if p.exists():
            with np.load(p) as z:
                thr = metrics.choose_alert_threshold(z["val_y"], z["fused_val_s0"])
                panels.append((_confusion(z["shared_y"], z["fused_s0"], thr),
                               "Fused model, sealed final test (seed 0)"))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.5))
    for ax, (cm, title) in zip(np.atleast_1d(axes), panels, strict=True):
        _draw_confusion(ax, cm, title)
    fig.suptitle("Confusion matrices at the alert level chosen on validation runners "
                 "(never on test)")
    _save(fig, "9_confusion_injury.png")


def calibration_and_risk() -> None:
    fz = _fused_round_files()
    if not fz:
        return
    ys, ps, risk = [], [], []
    for z in fz:  # Platt fitted on each round's validation runners, applied to its test runners
        params = metrics.fit_platt(metrics.logit(z["s_val"]), z["y_val"])
        ps.append(metrics.apply_platt(metrics.logit(z["s_test"]), params))
        ys.append(z["y_test"])
        risk.append(z["risk_test"])
    y, p, risk = np.concatenate(ys).astype(int), np.concatenate(ps), np.concatenate(risk)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    edges = np.quantile(p, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 9)
    mean_p = [p[idx == b].mean() for b in range(10)]
    obs = [y[idx == b].mean() for b in range(10)]
    top = max(max(mean_p), max(obs)) * 1.1
    axes[0].plot([0, top], [0, top], color=GREY, ls="--", lw=1, label="perfect calibration")
    axes[0].plot(mean_p, obs, "o-", color=NAVY, label="fused model (10 equal-size groups)")
    axes[0].set(xlabel="Predicted injury probability (after Platt)",
                ylabel="Observed injury rate", title="Calibration: does 2% risk mean 2%?")
    axes[0].legend(fontsize=8)
    bins = np.linspace(0, 100, 21)
    axes[1].hist(risk[y == 0], bins=bins, density=True, alpha=0.6, color=NAVY,
                 label=f"healthy days (median {np.median(risk[y == 0]):.0f})")
    axes[1].hist(risk[y == 1], bins=bins, density=True, alpha=0.6, color=RED,
                 label=f"injury days (median {np.median(risk[y == 1]):.0f})")
    axes[1].set(xlabel="Risk score 0-100 shown to the coach", ylabel="share of days (density)",
                title="Risk score: injured vs healthy days overlap a lot")
    axes[1].legend(fontsize=8)
    fig.suptitle("Fused model, test runners of the 5 development rounds (seed 0)")
    _save(fig, "10_calibration_and_risk_score.png")


def per_round() -> None:
    b = figures.baselines()
    fused = figures._runs("fused")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, kind in zip(axes, ("day", "week"), strict=True):
        for m, lab in INJURY:
            v = b.get((m, kind))
            if m == "constant_risk" or not v or not v["roc_auc"].get("per_round"):
                continue
            ax.plot(range(5), v["roc_auc"]["per_round"], "o-", lw=1.4, label=lab)
        if kind == "day" and fused:
            pr = fused[-1][1]["fused"]["summary"]["roc_auc"].get("per_round")
            if pr:
                ax.plot(range(5), pr, "o-", color="black", lw=2.5, label="FUSED (shared days)")
        ax.axhline(0.5, color=GREY, ls="--", lw=1, label="chance / constant risk")
        ax.set_xticks(range(5), [f"round {k}" for k in range(5)])
        ax.set_title(f"{kind} file")
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Test ROC-AUC")
    fig.suptitle("Every model in every development round (the spread behind the ± numbers)")
    _save(fig, "11_per_round_scores.png")


def fused_vs_arms() -> None:
    fused = figures._runs("fused")
    if not fused:
        return
    m = fused[-1][1]
    parts = [("Fused\n(both arms)", m["fused"]["summary"]),
             ("Sequence arm\n(day by day)", m["arms_summary"]["seq"]),
             ("Tabular arm\n(weekly totals)", m["arms_summary"]["tab"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in ((axes[0], "roc_auc", "ROC-AUC"), (axes[1], "pr_auc", "PR-AUC")):
        ax.bar([p[0] for p in parts], [p[1][metric]["mean"] for p in parts],
               yerr=[p[1][metric]["sd"] for p in parts], color=[RED, NAVY, NAVY], capsize=4)
        ax.set_title(f"{title} (mean ± SD, 5 rounds, shared days)")
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylim(0.5, 0.7)
    fig.suptitle("Does combining the two arms help? Only a borderline gain (see graph 3)")
    _save(fig, "12_fused_vs_arms.png")


PAMAP2_NAMES = {1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
                6: "cycling", 7: "Nordic walking", 9: "watching TV", 10: "computer work",
                11: "car driving", 12: "stairs up", 13: "stairs down", 16: "vacuuming",
                17: "ironing", 18: "folding laundry", 19: "house cleaning",
                20: "playing soccer", 24: "rope jumping"}  # PAMAP2 readme activity IDs


def confusion_activity() -> None:
    runs = figures._runs("pamap2_confusion")
    if not runs:
        return
    m = runs[-1][1]
    cm = np.array(m["confusion"], dtype=float)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    names = [PAMAP2_NAMES.get(v, str(v)) for v in m["labels"]]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, f"{share[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if share[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    rows = [n if cm[i].sum() else f"{n} (no test clips)" for i, n in enumerate(names)]
    ax.set_yticks(range(len(names)), rows, fontsize=8)
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    note = "reproduces the recorded run" if m["reproduces_record"] else "RE-RUN, differs"
    ax.set_title(f"Activity CNN, PAMAP2 round 0 test people {m['test_people']} "
                 f"({m['test_clips']} clips)\nmacro-F1 {m['macro_f1']:.3f} ({note}); "
                 "rows = share of each true activity", fontsize=10)
    _save(fig, "13_confusion_activity.png")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    for f in (injury_models, locked_test, comparisons, generators, tiers, explanations,
              augmentation, roc_pr_curves, confusion_injury, calibration_and_risk, per_round,
              fused_vs_arms, confusion_activity):
        f()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
