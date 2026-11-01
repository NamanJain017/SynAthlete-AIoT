"""python tasks.py paper-figures: every number and table of the paper, from the metrics files.

Writes paper/generated/{numbers.tex, table_datasets.tex, table_injury.tex,
table_generators.tex, table_tiers.tex}. A result that does not exist yet is written as "n/a",
so the paper always compiles and never shows a hand-typed number (AGENTS.md invariant 1).
"""

from __future__ import annotations

import json
import sys

from src.dashboard.prepare import DATASETS
from src.data import loaders

EXP = loaders.ROOT / "experiments"
GEN = loaders.ROOT / "paper" / "generated"
NA = "n/a"


def _runs(phase: str, file: str = "metrics.json") -> list[tuple[str, dict]]:
    out = []
    for r in sorted((EXP / phase).glob("20*")):
        if (r / file).exists():
            out.append((r.name, json.loads((r / file).read_text("utf-8"))))
    return out


def _tex(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def baselines() -> dict:
    """Newest result per (model, file) over all baseline runs."""
    best = {}
    for run, m in _runs("baselines"):
        for model, per in m.items():
            if not isinstance(per, dict):
                continue
            for kind, res in per.items():
                if isinstance(res, dict) and res.get("summary", {}).get("pr_auc"):
                    best[(model, kind)] = {**res["summary"], "run": run}
            if model == "lovdal_reproduction":
                best[("lovdal_reproduction", "-")] = per
            if model == "canary":
                best[("canary", "-")] = per
            if model.startswith("pamap2") and "summary" in per:
                best[(model, "pamap2")] = per["summary"]
    return best


def fmt(v: dict | None, key: str = "roc_auc") -> str:
    if not v or not v.get(key):
        return NA
    return f"{v[key]['mean']:.3f} $\\pm$ {v[key]['sd']:.3f}"


def numbers(b: dict) -> dict[str, str]:
    learned = [v["roc_auc"]["mean"] for (m, k), v in b.items()
               if m in ("logistic_regression", "random_forest", "catboost", "gru",
                        "lovdal_label_free")]
    n = {"injuryAucRange": f"{min(learned):.2f}--{max(learned):.2f}" if learned else NA}
    c = b.get(("canary", "-"))
    n["canaryAuc"] = (f"{c['day']['mean']:.3f} (day) and {c['week']['mean']:.3f} (week)"
                      if c else NA)
    rc = b.get(("lovdal_reproduction", "-"))
    n["reproDayAuc"] = (f"{rc['day']['estimate']:.3f} (95\\% CI {rc['day']['low']:.3f}--"
                        f"{rc['day']['high']:.3f})" if rc and "day" in rc else NA)
    lf = [b.get(("lovdal_label_free", k)) for k in ("day", "week")]
    n["labelFreeAuc"] = (f"{lf[0]['roc_auc']['mean']:.3f} (day) and "
                         f"{lf[1]['roc_auc']['mean']:.3f} (week)" if all(lf) else NA)
    p = b.get(("pamap2_cnn1d", "pamap2"))
    n["pamapRealFone"] = f"{p['macro_f1']['mean']:.3f}" if p else NA
    facts = loaders.PROCESSED / "lovdal" / "facts.json"
    n["lovdalRunners"] = (str(json.loads(facts.read_text("utf-8"))["athletes"])
                          if facts.exists() else NA)
    fused = _runs("fused")
    if fused:
        s = fused[-1][1]["fused"]["summary"]
        n["fusedAuc"], n["fusedPrAuc"] = fmt(s, "roc_auc"), fmt(s, "pr_auc")
    else:
        n["fusedAuc"] = n["fusedPrAuc"] = NA
    tiers = _runs("tiers")
    factor = tiers[-1][1]["streams"]["pamap2"]["bytes_reduction_factor"] if tiers else None
    n["tiersReduction"] = f"{factor:.0f}$\\times$" if factor else NA
    ex = _runs("explain")
    if ex:
        pc = ex[-1][1]["pass_counts"]
        n["explainSummary"] = ("Out of five rounds, the tabular-arm SHAP explanations passed the "
                               f"deletion test in {pc['tabular']['deletion']}, randomisation in "
                               f"{pc['tabular']['randomisation']} and stability in "
                               f"{pc['tabular']['stability']}; the sequence-arm integrated "
                               f"gradients passed {pc['sequence']['deletion']}, "
                               f"{pc['sequence']['randomisation']} and "
                               f"{pc['sequence']['stability']}.")
    else:
        n["explainSummary"] = "Explanation checks: n/a."
    fu = _runs("generators", "metrics_followup.json")
    if fu:
        f = fu[-1][1]
        label = {"timegan": "TimeGAN", "wgan_gp": "WGAN-GP", "timevae": "TimeVAE"}
        parts = []
        for g, c in f["wesad_criteria"].items():
            p = [k for k, ok in c.items() if ok]
            parts.append(f"{label.get(g, _tex(g))} met "
                         + (f"only the {' and '.join(p)} criterion" if p else "none")
                         + f" of its {len(c)} fidelity and privacy criteria")
        n["wesadSummary"] = ("On WESAD (five folds, settings carried over from PAMAP2 without "
                             "tuning), " + "; ".join(parts) + ".")
        inj, gru = f["injury_summary"], b.get(("gru", "day"))
        n["augmentSummary"] = (
            "Adding synthetic injury examples to the training rows only gave the GRU PR-AUC "
            f"{fmt(inj['timegan'], 'pr_auc')} with TimeGAN and {fmt(inj['classic'], 'pr_auc')}"
            f" with classic augmentation, against {fmt(gru, 'pr_auc')} real-only (ROC-AUC "
            f"{fmt(inj['timegan'])}, {fmt(inj['classic'])} and {fmt(gru)}); there is no paired "
            "interval, so no improvement is claimed.")
    else:
        n["wesadSummary"] = n["augmentSummary"] = "n/a."
    lock = _runs("lockbox")
    if lock:
        m = lock[-1][1]
        f = m["fused"]
        r, p, a = f["roc_auc_ci"], f["pr_auc_ci"], f["alert"]
        lov = m["lovdal_label_free_paper_split"]
        n["lockboxSummary"] = _tex(
            f"On the {len(m['test_runners'])} sealed runners ({f['events']:,} shared days, "
            f"{f['injuries']} injuries in {f['runners_with_injury']} runners), opened once after "
            f"the evaluation freeze, the declared fused model reached ROC-AUC "
            f"{r['estimate']:.3f} ({r['low']:.3f}-{r['high']:.3f}) and PR-AUC {p['estimate']:.3f}"
            f" ({p['low']:.3f}-{p['high']:.3f}; chance {f['test_injury_rate']:.3f}); its alert "
            f"level caught {a['sensitivity']:.0%} of injuries with "
            f"{a['false_alarms_per_caught_injury']:.0f} false alarms per caught injury. "
            "Secondary results, scored once and not a comparison (other rows and more training "
            f"runners): the label-free variant reached ROC-AUC "
            f"{lov['day']['roc_auc_ci']['estimate']:.3f} (day) and "
            f"{lov['week']['roc_auc_ci']['estimate']:.3f} (week); no model was switched. With so "
            "few runners the intervals are wide and understate the uncertainty.")
    else:
        n["lockboxSummary"] = "Pending the evaluation freeze."
    return n


def write_numbers(n: dict[str, str]) -> None:
    lines = ["% generated by src/paper/figures.py: do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}\\xspace}}" for k, v in sorted(n.items())]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n", "utf-8")


def table_datasets() -> None:
    rows = "\n".join(f"{_tex(d)} & {_tex(o)} & {_tex(r)} \\\\" for d, o, r in DATASETS)
    (GEN / "table_datasets.tex").write_text(
        "\\begin{table}[t]\\centering\\caption{Datasets and their single role}\\label{tab:data}\n"
        "\\footnotesize\\begin{tabular}{p{2.6cm}p{1.2cm}p{3.6cm}}\\toprule\n"
        "Dataset & Origin & Role \\\\\\midrule\n" + rows +
        "\n\\bottomrule\\end{tabular}\\end{table}\n", "utf-8")


def table_injury(b: dict) -> None:
    names = {"constant_risk": "Constant risk", "acwr_style": "ACWR-style rule",
             "logistic_regression": "Logistic regression", "random_forest": "Random forest",
             "catboost": "CatBoost", "gru": "GRU", "lovdal_label_free": "L\\\"ovdal, label-free"}
    rows = []
    for m, label in names.items():
        for k in ("day", "week"):
            if (m, k) in b:
                rows.append(f"{label} & {k} & {fmt(b[(m, k)], 'pr_auc')} & "
                            f"{fmt(b[(m, k)], 'roc_auc')} \\\\")
    fused = _runs("fused")
    if fused:
        s = fused[-1][1]["fused"]["summary"]
        rows.append(f"\\textbf{{Fused}} & shared & {fmt(s, 'pr_auc')} & {fmt(s, 'roc_auc')} \\\\")
    (GEN / "table_injury.tex").write_text(
        "\\begin{table}[t]\\centering\\caption{Injury prediction, mean $\\pm$ SD over five "
        "development rounds (chance PR-AUC $\\approx$ injury rate)}\\label{tab:injury}\n"
        "\\footnotesize\\begin{tabular}{llcc}\\toprule\nModel & File & PR-AUC & ROC-AUC "
        "\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n",
        "utf-8")


def table_generators() -> None:
    runs = _runs("generators", "metrics_pamap2.json")
    body = "\\multicolumn{5}{c}{n/a} \\\\"
    if runs:
        g = runs[-1][1]
        rows = []
        for name, m in g["pamap2_summary"].items():
            ok = sum(g["criteria"][name].values())
            label = "wgan_gp (untuned)" if name == "wgan_gp" else name
            rows.append(f"{_tex(label)} & {m['discriminative_accuracy']['mean']:.3f} & "
                        f"{m['tstr_test_macro_f1']['mean']:.3f} & "
                        f"{m['real_plus_synthetic_test_macro_f1']['mean']:.3f} & "
                        f"{ok}/{len(g['criteria'][name])} \\\\")
        body = "\n".join(rows)
    (GEN / "table_generators.tex").write_text(
        "\\begin{table}[t]\\centering\\caption{Generators on PAMAP2 (4 rounds): discriminative "
        "accuracy (0.5 = indistinguishable), TSTR and real+synthetic macro-F1, pre-registered "
        "criteria passed}\\label{tab:gen}\n\\footnotesize\\begin{tabular}{lcccc}\\toprule\n"
        "Generator & Disc.\\ acc. & TSTR & Real+syn. & Passed \\\\\\midrule\n" + body +
        "\n\\bottomrule\\end{tabular}\\end{table}\n", "utf-8")


def table_tiers() -> None:
    runs = _runs("tiers")
    body = "\\multicolumn{4}{c}{n/a} \\\\"
    if runs:
        rows = []
        for name, o in runs[-1][1]["streams"].items():
            for mode in ("tiered", "cloud_only"):
                rows.append(f"{_tex(name)} & {_tex(mode)} & {o[mode]['bytes_into_cloud']:,} & "
                            f"{o[mode]['decision_latency_ms_median']:.2f} \\\\")
        body = "\n".join(rows)
    (GEN / "table_tiers.tex").write_text(
        "\\begin{table}[t]\\centering\\caption{Edge--fog--cloud vs cloud-only (simulated on one "
        "computer)}\\label{tab:tiers}\n\\footnotesize\\begin{tabular}{llrr}\\toprule\n"
        "Stream & Mode & Bytes into cloud & Median decision (ms) \\\\\\midrule\n" + body +
        "\n\\bottomrule\\end{tabular}\\end{table}\n", "utf-8")


def table_comparisons() -> None:
    runs = _runs("comparisons")
    body = r"\multicolumn{3}{c}{n/a} \\"
    if runs:
        rows = []
        for c in runs[-1][1]["comparisons"]:
            p, r = c["pr_auc"], c["roc_auc"]
            name = c["comparison"].replace(" - lovdal_label_free", " vs label-free")
            name = name.replace("_", " ").replace(" - ", " vs ")
            rows.append(f"{_tex(name)} & {p['estimate']:+.4f} [{p['low']:+.4f}, "
                        f"{p['high']:+.4f}] & {r['estimate']:+.3f} [{r['low']:+.3f}, "
                        f"{r['high']:+.3f}] " + r"\\")
        body = "\n".join(rows)
    (GEN / "table_comparisons.tex").write_text(
        r"\begin{table}[t]\centering\caption{Paired differences (5-round mean, runner "
        r"bootstrap 95\% interval, 2,000 resamples)}\label{tab:compare}" + "\n"
        r"\footnotesize\begin{tabular}{lcc}\toprule" + "\n"
        r"Comparison & $\Delta$ PR-AUC & $\Delta$ ROC-AUC \\\midrule" + "\n"
        + body + "\n" + r"\bottomrule\end{tabular}\end{table}" + "\n", "utf-8")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    GEN.mkdir(parents=True, exist_ok=True)
    b = baselines()
    n = numbers(b)
    write_numbers(n)
    table_datasets()
    table_injury(b)
    table_generators()
    table_tiers()
    table_comparisons()
    for k, v in sorted(n.items()):
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
