"""SynAthlete-AIoT dashboard (piece 9): python tasks.py demo   (prepare first: --prepare).

Eight pages in pipeline order, so each page is one step of the project:
Home -> Athletes -> Athlete detail -> Live sensor -> Synthetic data -> Results -> Training lab ->
Data & honesty. Reads only the prepared bundle (experiments/dashboard/bundle/), so the demo is
instant; only the Training lab trains, inside safe ranges, on validation only (D-016).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # streamlit runs this file directly
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.dashboard import lab  # noqa: E402
from src.dashboard.prepare import ACTIVITIES, BUNDLE  # noqa: E402

st.set_page_config(page_title="SynAthlete-AIoT", layout="wide")


@st.cache_data
def load(name: str):
    p_parquet, p_json = BUNDLE / f"{name}.parquet", BUNDLE / f"{name}.json"
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    if p_json.exists():
        return json.loads(p_json.read_text("utf-8"))
    return None


def missing(what: str, how: str) -> None:
    st.info(f"{what} is not ready yet. {how}")


def latest(results: dict, phase: str, file: str = "metrics.json") -> dict | None:
    for run in reversed(results.get(phase, [])):
        if file in run["files"]:
            return {"run": run["run"], **run["files"][file]}
    return None


def baseline_table(results: dict) -> pd.DataFrame:
    """Every injury model's 5-round mean, from all baseline runs (newest wins per model)."""
    rows = {}
    for run in results.get("baselines", []):
        m = run["files"].get("metrics.json", {})
        for model, per_file in m.items():
            if not isinstance(per_file, dict):
                continue
            for kind, res in per_file.items():
                if isinstance(res, dict) and "summary" in res and res["summary"].get("pr_auc"):
                    rows[(model, kind)] = {"model": model, "file": kind,
                                           "PR-AUC": res["summary"]["pr_auc"]["mean"],
                                           "ROC-AUC": res["summary"]["roc_auc"]["mean"],
                                           "run": run["run"]}
    return pd.DataFrame(rows.values())


PIPELINE = "\n".join([  # "edge" is a reserved word in DOT, so the edge tier is node e1
    "digraph { rankdir=LR; node [shape=box, style=rounded];",
    'data [label="Data\\nLövdal injuries + wearables"];',
    'gen [label="Synthetic data\\nTimeGAN, WGAN-GP, TimeVAE"];',
    'e1 [label="Edge (reflex)"]; fog [label="Fog (supervisor)"]; cloud [label="Cloud (deep)"];',
    'model [label="Combined model\\nsequence + tabular"];',
    'xai [label="Explanations\\nSHAP, IG, attention"];',
    'coach [label="Coach view\\nrisk 0-100 + reasons"];',
    "data -> gen; data -> e1 -> fog -> cloud -> model -> xai -> coach;",
    "gen -> model [style=dashed]; }"])

results = load("results") or {}
pages = ["Home", "Athletes", "Athlete detail", "Live sensor", "Synthetic data", "Results",
         "Training lab", "Data & honesty"]
st.sidebar.title("SynAthlete-AIoT")
st.sidebar.caption("Explainable AIoT injury-risk system")
page = st.sidebar.radio("Page", pages)
st.sidebar.markdown("---")
st.sidebar.caption("Injury numbers come only from real injury labels. Synthetic data is never "
                   "evidence. Every number comes from a file written by code.")

# ------------------------------------------------------------------ 1 Home
if page == "Home":
    st.title("SynAthlete-AIoT")
    st.write("From a runner's training log and a wearable stream to an explained risk score "
             "for the coach.")
    st.graphviz_chart(PIPELINE)
    bt = baseline_table(results)
    fused = latest(results, "fused")
    tiers = latest(results, "tiers")
    c = st.columns(4)
    if not bt.empty:
        learned = bt[~bt["model"].isin(["constant_risk", "acwr_style"])]
        c[0].metric("Injury models, ROC-AUC (5 rounds)",
                    f"{learned['ROC-AUC'].min():.2f}-{learned['ROC-AUC'].max():.2f}", "chance 0.50")
    if fused:
        c[1].metric("Combined model ROC-AUC", f"{fused['fused']['summary']['roc_auc']['mean']:.3f}")
    if tiers:
        f = tiers["streams"]["pamap2"]["bytes_reduction_factor"]
        c[2].metric("Edge tier: data sent to cloud", f"{f:.0f}x less")
    pam = bt[bt["model"].str.startswith("pamap2")] if not bt.empty else bt
    for run in results.get("baselines", []):
        m = run["files"].get("metrics.json", {})
        if "pamap2_cnn1d" in m:
            c[3].metric("Activity recognition (macro-F1)",
                        f"{m['pamap2_cnn1d']['summary']['macro_f1']['mean']:.2f}", "chance ~0.06")
    st.caption("Demo order: Athletes -> Athlete detail -> Live sensor -> Synthetic data -> "
               "Results -> Data & honesty.")

# ------------------------------------------------------------------ 2 Athletes
elif page == "Athletes":
    ev = load("events")
    if ev is None:
        missing("The coach view", "It appears after the combined model run and "
                "`python tasks.py demo --prepare`.")
    else:
        meta = load("coach_meta")
        st.header("Athletes")
        st.caption(f"Real runners from development round {meta['round']}'s test group: the model "
                   "never trained on them. Not the 10 locked runners.")
        last = ev.sort_values("day").groupby("runner").tail(1).set_index("runner")
        table = pd.DataFrame({"latest risk (0-100)": last["risk_0_100"],
                              "alerts": ev.groupby("runner")["alert"].sum(),
                              "scored days": ev.groupby("runner").size(),
                              "injuries in data": ev.groupby("runner")["injury"].sum()})
        st.dataframe(table.sort_values("latest risk (0-100)", ascending=False),
                     use_container_width=True)
        st.caption("Risk 0-100 = riskier than that share of validation events (a ranking, not a "
                   "probability of injury).")

# ------------------------------------------------------------------ 3 Athlete detail
elif page == "Athlete detail":
    ev = load("events")
    if ev is None:
        missing("The athlete detail", "It appears after the combined model run.")
    else:
        runner = st.selectbox("Runner", sorted(ev["runner"].unique()))
        sub = ev[ev["runner"] == runner].sort_values("day")
        st.header(f"Runner {runner}")
        base = alt.Chart(sub).encode(x=alt.X("day:Q", title="day"))
        line = base.mark_line().encode(y=alt.Y("risk_0_100:Q", title="risk 0-100"))
        inj = base.transform_filter("datum.injury == 1").mark_point(
            color="red", size=90, filled=True).encode(y="risk_0_100:Q")
        st.altair_chart(line + inj, use_container_width=True)
        st.caption("Red dots = injuries recorded in the data.")
        day = st.select_slider("Replay day", options=sub["day"].tolist(), value=sub["day"].iloc[-1])
        row = sub[sub["day"] == day].iloc[0]
        c = st.columns(3)
        c[0].metric("Risk 0-100", f"{row['risk_0_100']:.0f}")
        c[1].metric("Alert", "YES" if row["alert"] else "no")
        c[2].metric("Injury recorded", "yes" if row["injury"] else "no")
        st.subheader("Why?")
        st.success(row["reason"])
        top = pd.DataFrame(json.loads(row["top_inputs"]),
                           columns=["input", "contribution", "value vs usual"])
        st.altair_chart(alt.Chart(top).mark_bar().encode(
            x=alt.X("contribution:Q", title="push on risk (TreeSHAP)"),
            y=alt.Y("input:N", sort="-x"),
            color=alt.condition("datum.contribution > 0", alt.value("#c0392b"),
                                alt.value("#2e86c1"))), use_container_width=True)
        st.caption("Red raises risk, blue lowers it. This explains the model, it does not prove "
                   "a cause.")

# ------------------------------------------------------------------ 4 Live sensor
elif page == "Live sensor":
    s, m = load("sensor_stream"), load("sensor_meta")
    st.header("Live sensor: edge - fog - cloud")
    if s is None:
        missing("The sensor stream", "Run `python tasks.py demo --prepare`.")
    else:
        st.caption(f"PAMAP2 person {m['pamap2_person']} replayed in 5-second windows. "
                   f"{m['data_role']}.")
        upto = st.slider("Replay up to window", 1, len(s), min(120, len(s)))
        part = s.iloc[:upto].copy()
        part["activity"] = part["activity_predicted"].map(lambda a: ACTIVITIES.get(int(a), str(a)))
        c = st.columns(3)
        c[0].metric("Activity now (edge CNN)", part["activity"].iloc[-1])
        c[1].metric("Heart rate now", f"{part['hr_mean'].iloc[-1]:.0f} bpm")
        c[2].metric("Reflex alerts so far", int(part["reflex_alert"].sum()))
        st.line_chart(part.set_index("seconds")[["hr_mean"]], height=200)
        st.altair_chart(alt.Chart(part).mark_tick(thickness=4).encode(
            x=alt.X("seconds:Q"), y=alt.Y("activity:N", title="")), use_container_width=True)
    tiers = latest(results, "tiers")
    st.subheader("Why tiers? Data sent and decision speed")
    if tiers is None:
        missing("The tiers benchmark", "Run `python tasks.py pipeline`.")
    else:
        rows = [{"stream": n, "mode": mode, "bytes into cloud": o[mode]["bytes_into_cloud"],
                 "median decision (ms)": round(o[mode]["decision_latency_ms_median"], 2)}
                for n, o in tiers["streams"].items() for mode in ("tiered", "cloud_only")]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.caption(tiers["note"])

# ------------------------------------------------------------------ 5 Synthetic data
elif page == "Synthetic data":
    st.header("Synthetic data")
    st.caption("Generators learn from training people only; synthetic clips are used only for "
               "training, never for testing, and never as evidence about injuries.")
    smp = load("synthetic_samples")
    if smp is None:
        missing("Synthetic samples", "They appear after the generator run.")
    else:
        act = st.selectbox("Activity", sorted(smp["activity"].unique()))
        sub = smp[(smp["activity"] == act) & (smp["clip"] == 0)]
        st.altair_chart(alt.Chart(sub).mark_line().encode(
            x=alt.X("t:Q", title="sample (25 per second)"), y=alt.Y("bpm:Q", title="heart rate"),
            color="generator:N").properties(height=280), use_container_width=True)
    gen = latest(results, "generators", "metrics_pamap2.json")
    if gen:
        rows = []
        for name, m in gen["pamap2_summary"].items():
            mean = {k: v["mean"] for k, v in m.items()}
            rows.append({"generator": name,
                         "real-vs-fake accuracy (0.5 = fooled)": mean["discriminative_accuracy"],
                         "TSTR macro-F1": mean["tstr_test_macro_f1"],
                         "real + synthetic macro-F1": mean["real_plus_synthetic_test_macro_f1"],
                         "too-close share (privacy)": mean["share_synthetic_closer"],
                         "passes": ", ".join(k for k, v in gen["criteria"][name].items() if v)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        real = sum(gen["real_only_test_macro_f1"]) / len(gen["real_only_test_macro_f1"])
        st.caption(f"Real-data-only macro-F1 for comparison: {real:.3f}. Pass marks fixed in "
                   "advance (D-027).")

# ------------------------------------------------------------------ 6 Results
elif page == "Results":
    st.header("Results")
    bt = baseline_table(results)
    if not bt.empty:
        inj = bt[~bt["model"].str.startswith("pamap2")]
        st.subheader("Injury models (mean over 5 development rounds)")
        st.dataframe(inj.drop(columns=["run"]).round(3), use_container_width=True)
        st.caption("Chance: PR-AUC about 0.013 (the injury rate), ROC-AUC 0.50.")
    fused = latest(results, "fused")
    if fused:
        s = fused["fused"]["summary"]
        st.metric("Combined model (shared days)",
                  f"ROC-AUC {s['roc_auc']['mean']:.3f} ± {s['roc_auc']['sd']:.3f}",
                  f"PR-AUC {s['pr_auc']['mean']:.3f}")
    for run in results.get("baselines", []):
        rc = run["files"].get("metrics.json", {}).get("lovdal_reproduction")
        if rc and "day" in rc:
            st.subheader("Paper copy check (exact method on stand-in runners 54-63)")
            st.write({k: f"ROC-AUC {rc[k]['estimate']:.3f} ({rc[k]['low']:.3f} to "
                         f"{rc[k]['high']:.3f}) {'PASSED' if rc[k]['passed'] else 'FAILED'}"
                      for k in ("day", "week") if k in rc})
    st.subheader("Locked final test (10 newest runners)")
    st.write("Opened once, after the evaluation freeze. Shown here when available.")

# ------------------------------------------------------------------ 7 Training lab
elif page == "Training lab":
    st.header("Training lab (demo)")
    st.caption("Retrain a fast model with other settings inside safe ranges, on development round "
               "0, scored on its validation runners only. Saved as a demo; never official.")
    model = st.selectbox("Model", list(lab.SAFE))
    s: dict = {}
    if model == "logistic_regression":
        s["C"] = st.select_slider("C", [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], value=1.0)
        s["class_weight"] = st.selectbox("class_weight", [None, "balanced"])
    elif model == "random_forest":
        s["n_estimators"] = st.slider("trees", 100, 500, 300, 50)
        s["max_depth"] = st.selectbox("max_depth", [4, 6, 8, 12, None])
        s["min_samples_leaf"] = st.slider("min_samples_leaf", 1, 50, 5)
        s["max_features"] = st.selectbox("max_features", ["sqrt", 0.3, 0.5])
        s["class_weight"] = st.selectbox("class_weight", [None, "balanced_subsample"])
    elif model == "catboost":
        s["learning_rate"] = st.slider("learning rate", 0.01, 0.1, 0.03)
        s["depth"] = st.selectbox("depth", [4, 6, 8])
        s["l2_leaf_reg"] = st.slider("l2_leaf_reg", 1.0, 10.0, 3.0)
        s["auto_class_weights"] = st.selectbox("class weights", [None, "Balanced"])
    else:
        s["hidden"] = st.selectbox("hidden", [16, 32, 64])
        s["layers"] = st.selectbox("layers", [1, 2])
        s["dropout"] = st.selectbox("dropout", [0.0, 0.2, 0.4])
        s["lr"] = st.select_slider("learning rate", [3e-4, 1e-3, 3e-3], value=1e-3)
    if st.button("Train"):
        with st.spinner("Training..."):
            res = lab.train(model, s)
        st.success(res["label"])
        st.json(res)

# ------------------------------------------------------------------ 8 Data & honesty
else:
    st.header("Data & honesty")
    ds = load("datasets")
    if ds:
        st.dataframe(pd.DataFrame(ds), use_container_width=True)
    st.markdown("""
- **Tested on new people only:** every score comes from runners or people the model never
  trained on; settings were chosen on separate validation people.
- **Locked final test:** the 10 newest runners stay sealed until the evaluation freeze.
- **Leak checks:** a shuffled-label alarm, an input guard (no IDs, dates or history), training-only
  scaling, an independent code review.
- **Synthetic data** is labelled synthetic everywhere and only ever used for training.
- **The system view** uses wearable data from different people than the runners: a demo of the
  IoT design, not injury evidence.
""")
    z = load("ziya07_demo")
    if z is not None:
        with st.expander("ziya07 demo stream (unknown origin: demo only, never evidence)"):
            st.dataframe(z, use_container_width=True)
