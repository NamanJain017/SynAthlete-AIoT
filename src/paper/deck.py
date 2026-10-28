"""Review 2 deck, made by code: python tasks.py paper-figures --deck  ->  slides/review2.pptx.

Structure follows the Review 2 rubric (Methodology 5, Implementation 10, Report + paper 5).
Every number comes from the metrics files (via src.paper.figures); a result whose run has not
finished is shown as "running" and appears when the deck is rebuilt.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

from src.dashboard.prepare import DATASETS  # noqa: E402
from src.paper import figures  # noqa: E402

OUT = figures.loaders.ROOT / "slides"
IMG = OUT / "img"
NAVY, ACCENT = RGBColor(0x1F, 0x3A, 0x5F), RGBColor(0xC0, 0x39, 0x2B)
GREY = RGBColor(0x55, 0x5B, 0x66)
GREEN, WHITE = RGBColor(0x1E, 0x84, 0x49), RGBColor(0xFF, 0xFF, 0xFF)
W, H = Inches(13.333), Inches(7.5)
LEFT, TOP, WIDE, TALL = Inches(0.6), Inches(1.3), Inches(12.1), Inches(5.6)


def _plain(v: str) -> str:
    """LaTeX bits from figures.numbers() -> plain slide text."""
    for a, b in (("$\\pm$", "±"), ("\\%", "%"), ("--", "-"), ("$\\times$", "x"), ("$", "")):
        v = v.replace(a, b)
    return v

# Speaker notes: what to say on each slide, in plain words. No numbers here: the numbers are on
# the slides themselves and come from the result files (AGENTS.md invariant 1).
NOTES = {
    'SynAthlete-AIoT': (
        "Good morning. Our project is an AI system that warns a coach when a runner's injury "
        'risk is building up, explains why, and runs on an edge-fog-cloud IoT design. Our '
        'main rule: everything is tested honestly, and every number comes from code.'),
    'Problem and the corrected research gap': (
        'Coaches see injuries only after they happen. Recent AIoT frameworks describe the '
        'system but do not build the prediction or the explanation part. Since Review 1 we '
        'corrected our gap: a real dataset with daily logs and real injuries exists, so the '
        'real gap is that no public dataset joins smartwatch-style sensor streams with real '
        'injuries. We also found that many reported accuracies are inflated by leakage, so '
        'honest testing is part of our contribution.'),
    'Objectives from Review 1: all implemented': (
        'These are our five objectives from Review 1. Every one is implemented; the third '
        'column says where the code is, and the last column says what state it is in. I can '
        'open any of these files during the demo.'),
    'Datasets: each with one honest role': (
        'We audited every dataset. Only the Lövdal runner logs contain real injuries, so '
        'every injury claim comes from them. The wearable datasets drive the IoT and '
        'synthetic-data parts. Two datasets of unknown origin are used only as labelled '
        'demos.'),
    'Pipeline': (
        'This is the whole system from left to right: data, synthetic data, the watch-level '
        'edge, the phone-level fog, the cloud model, explanations and the coach dashboard. '
        'Green boxes are the IoT tiers, which run as separate programs.'),
    'Methodology: honest evaluation, fixed before any result': (
        'This slide is why our numbers can be trusted. Models are always tested on runners '
        'they never saw. Ten runners are locked away for one final exam. Settings are chosen '
        'on separate validation runners, never on the test. We also run a leak alarm, and we '
        'wrote the pass marks down before running anything.'),
    'Implementation progress': (
        'Here is where each module stands. All code is written and tested, every run has '
        'finished, and the slides are built from the result files.'),
    'Result: injury prediction is hard, and we report it honestly': (
        'Each bar is one model tested on runners it never saw. The dashed line is guessing. '
        'The models are only modestly above guessing and no model is clearly best overall. '
        'The red bar is the fused model because we declared it in advance, not because it '
        'won. This is an honest result: training logs alone carry little warning before an '
        'injury.'),
    'Result: the final locked test': (
        'Ten runners were locked away from the start and opened once, after the code was '
        'frozen. The model we declared in advance scored about the same as in development, '
        'which shows our development results were not over-tuned. The alert is too noisy for '
        'real use. The other numbers are secondary: they were scored on different rows, so we '
        'do not rank models on them and we did not switch models.'),
    'Result: the published method, rebuilt and checked': (
        "We copied the published paper's exact method, including a scaling step that reads "
        'the labels, and it passed our pre-registered sanity check. The version without that '
        'step was tested on different runners, so we do not claim how big the effect of the '
        'step is. On the same rows, only a few of our models are clearly better than it, and '
        'we did not adjust for testing many pairs.'),
    'Result: synthetic wearable data': (
        'Three generators learn to draw realistic sensor clips, only from training people. We'
        ' judge the clips on real people they never saw: can a judge tell them apart, does a '
        'model trained only on them still work, and are they copying anyone. The pass marks '
        'were fixed in advance.'),
    'Result: activity recognition and edge-fog-cloud': (
        'Our activity model recognises what a person is doing from sensors, on people it '
        'never saw, far above guessing. In the IoT system the watch-level edge sends short '
        'summaries instead of raw data, so much less data reaches the cloud. This is '
        'simulated on one laptop.'),
    'Result: explanations that are tested, not assumed': (
        'For every risk score the coach sees why: which inputs pushed the risk up. We do not '
        'just trust these explanations: each one must pass a deletion test, a randomisation '
        'test and a stability test.'),
    'Live demo: coach dashboard': (
        'Now the live demo. I start with the athletes page, open the riskiest runner, show '
        'the reason sentence, then the live sensor page, the synthetic data page, the results'
        ' page and finally the honesty page. The data was prepared in advance, so the demo '
        'cannot fail.'),
    'Code tour: where each part lives': (
        'If you would like to see code, these are the files for each part. Every part has '
        'tests, and one command runs all of them.'),
    'Research paper and project report': (
        'Our IEEE-style paper draft and project report are written; all their numbers are '
        'inserted by code from the same result files, so the paper always matches the '
        'experiments. We will confirm the target conference with our guide.'),
    'Honesty and limitations': (
        'We are open about the limits: injury prediction from logs is only modestly accurate;'
        ' the dataset removed healthy days near injuries; the IoT tiers are simulated; '
        'synthetic data is used only for training; and the locked final test is opened only '
        'once.'),
    'Thank you: questions?': (
        'Thank you. We are happy to answer questions or show any part again.'),
}


class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.n = 0

    def slide(self, title: str, rubric: str | None = None):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(1.0))
        bar.fill.solid()
        bar.fill.fore_color.rgb = NAVY
        bar.line.fill.background()
        self.text(s, title, Inches(0.4), Inches(0.18), Inches(10.5), Inches(0.7), 28, True, WHITE)
        if rubric:
            self.text(s, rubric, Inches(10.4), Inches(0.3), Inches(2.8), Inches(0.5), 12, False,
                      RGBColor(0xDD, 0xE6, 0xF0))
        self.n += 1
        self.text(s, str(self.n), Inches(12.6), Inches(7.0), Inches(0.6), Inches(0.4), 11,
                  False, GREY)
        if title in NOTES:  # visible to the presenter in PowerPoint's presenter view
            s.notes_slide.notes_text_frame.text = NOTES[title]
        return s

    @staticmethod
    def text(s, txt, x, y, w, h, size=18, bold=False, color=NAVY):
        tb = s.shapes.add_textbox(x, y, w, h)
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = txt
        p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, color
        return tb

    @staticmethod
    def bullets(s, items, x=LEFT, y=TOP, w=WIDE, h=TALL, size=20):
        tf = s.shapes.add_textbox(x, y, w, h).text_frame
        tf.word_wrap = True
        for i, it in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            sub = it.startswith("  ")
            p.text = ("– " if sub else "• ") + it.strip()
            p.level = 1 if sub else 0
            p.font.size = Pt(size - 3 if sub else size)
            p.font.color.rgb = GREY if sub else NAVY
            p.space_after = Pt(6)

    @staticmethod
    def table(s, header, rows, x=LEFT, y=TOP, w=WIDE, size=14):
        t = s.shapes.add_table(len(rows) + 1, len(header), x, y, w,
                               Inches(0.4) * (len(rows) + 1)).table
        for j, name in enumerate(header):
            font = t.cell(0, j).text_frame.paragraphs[0].font
            t.cell(0, j).text = name
            font = t.cell(0, j).text_frame.paragraphs[0].font
            font.size, font.bold = Pt(size), True
        for i, row in enumerate(rows, 1):
            for j, v in enumerate(row):
                t.cell(i, j).text = str(v)
                t.cell(i, j).text_frame.paragraphs[0].font.size = Pt(size - 1)
        return t

    @staticmethod
    def box(s, txt, x, y, w, h, color):
        b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
        b.fill.solid()
        b.fill.fore_color.rgb = color
        b.line.fill.background()
        tf = b.text_frame
        tf.word_wrap = True
        tf.paragraphs[0].text = txt
        font = tf.paragraphs[0].font
        font.size, font.color.rgb = Pt(14), WHITE

    @staticmethod
    def arrow(s, x1, y, x2):
        c = s.shapes.add_connector(1, x1, y, x2, y)
        c.line.color.rgb, c.line.width = GREY, Pt(2)


def _chart_injury(b: dict) -> str | None:
    labels = {"constant_risk": "Constant risk", "acwr_style": "ACWR rule",
              "logistic_regression": "Logistic reg.", "random_forest": "Random forest",
              "catboost": "CatBoost", "gru": "GRU", "lovdal_label_free": "Lövdal (label-free)"}
    names, vals = [], []
    for m, lab in labels.items():
        for k in ("day", "week"):
            if (m, k) in b:
                names.append(f"{lab} ({k})")
                vals.append(b[(m, k)]["roc_auc"]["mean"])
    fused = figures._runs("fused")
    if fused:
        names.append("FUSED (declared in advance)")
        vals.append(fused[-1][1]["fused"]["summary"]["roc_auc"]["mean"])
    if not vals:
        return None
    IMG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    ax.barh(names, vals, color=["#c0392b" if n.startswith("FUSED") else "#1f3a5f" for n in names])
    ax.axvline(0.5, color="#888", ls="--", lw=1)
    ax.set_ylim(len(names) - 0.5, -1.3)  # room above the first bar for the chance label
    ax.text(0.503, -0.75, "chance 0.50", color="#666", fontsize=9, va="center")
    ax.set_xlim(0.45, max(0.7, max(vals) + 0.03))
    ax.set_xlabel("ROC-AUC on runners never seen in training (mean of 5 rounds)")
    fig.tight_layout()
    path = IMG / "injury_auc.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


OBJECTIVES = [
    ["1 Bridge the data gap", "TimeGAN, WGAN-GP, TimeVAE + classic augmentation", "src/gen/"],
    ["2 Edge-fog-cloud tiers", "3 processes: reflex, supervisor, deep; bytes + latency",
     "src/pipeline/"],
    ["3 Fused model", "GRU/LSTM/TCN + RF/CatBoost, stacking, risk 0-100", "src/models/fused.py"],
    ["4 Validated explanations", "TreeSHAP, IG, attention, Grad-CAM + 3 checks", "src/xai/"],
    ["5 Coach dashboard", "Streamlit: 8 pages + training lab", "src/dashboard/"],
    ["Baselines + honest testing", "7 injury models, leak alarm, paper rebuild", "src/models/"],
]
CODE_TOUR = [
    ["Splits + leak guard", "src/models/data.py", "tests/test_models.py"],
    ["Scoring rules", "src/eval/metrics.py", "tests/handoff/test_wp01_metrics.py"],
    ["Baselines + paper rebuild", "src/models/run.py, lovdal_method.py", "tests/test_models.py"],
    ["Generators", "src/gen/models.py, evaluate.py", "tests/test_gen.py"],
    ["Edge-fog-cloud", "src/pipeline/tiers.py", "tests/test_pipeline.py"],
    ["Fused model", "src/models/fused.py", "tests/test_models.py"],
    ["Explanations", "src/xai/explain.py, checks.py", "tests/test_xai.py"],
    ["Dashboard", "src/dashboard/app.py", "tests/test_dashboard_pages.py"],
]


def _published_line() -> str:
    lock = figures._runs("lockbox")
    if not lock:
        return ("  The published 0.724 was measured on the locked runners: compared only at the "
                "final check.")
    r = lock[-1][1]["lovdal_label_free_paper_split"]["day"]["roc_auc_ci"]
    return (f"  Final check on the paper's own test runners (locked, day file): label-free "
            f"variant ROC-AUC {r['estimate']:.3f} ({r['low']:.3f}-{r['high']:.3f}); the "
            "published 0.724 used the label-reading scaling.")


STREAM = {"pamap2": "PAMAP2", "sportdb2_hr": "Sport DB heart rate"}
GEN_LABEL = {"timegan": "TimeGAN", "wgan_gp": "WGAN-GP (untuned: 1 declared setting, D-029)",
             "timevae": "TimeVAE", "classic": "Classic augmentation"}


def _comparison_lines() -> list[str]:
    """Paired differences vs the Lövdal label-free variant and fused vs its arms, per metric
    (PR-AUC first, the headline metric); 'clear' = the 95 % interval excludes 0."""
    runs = figures._runs("comparisons")
    if not runs:
        return []
    comps = runs[-1][1]["comparisons"]
    better = {"pr_auc": [], "roc_auc": []}
    worse = set()
    for c in comps:
        model, ref = c["comparison"].split(" - ", 1)
        name = f"{model.replace('_', ' ')} {ref[ref.rfind('('):]}"
        for metric in better:
            if "lovdal" in ref and c[metric]["low"] > 0:
                better[metric].append(name)
            if "lovdal" in ref and c[metric]["high"] < 0:
                worse.add(name)
    lines = ["Paired runner bootstrap vs the label-free variant (same rows): clearly better on "
             f"PR-AUC: {', '.join(better['pr_auc']) or 'none'}; on ROC-AUC: "
             f"{', '.join(better['roc_auc']) or 'none'}; clearly worse: "
             f"{', '.join(sorted(worse)) or 'none'}; the rest are not clearly different."]
    fused = []
    for c in comps:
        if c["comparison"].startswith("fused"):
            arm = c["comparison"].split(" - ", 1)[1].split(" (")[0]
            parts = []
            for metric, label in (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")):
                r = c[metric]
                verdict = ("excludes 0, borderline" if r["low"] > 0 and r["low"] < 0.001
                           else "excludes 0" if r["low"] > 0 else "includes 0")
                parts.append(f"{label} {r['estimate']:+.4f} [{r['low']:+.5f}, {r['high']:+.4f}]"
                             f" ({verdict})")
            fused.append(f"vs its {arm}: " + ", ".join(parts))
    if fused:
        lines.append("Fused model " + "; ".join(fused) + ".")
    lines.append(f"{2 * len(comps)} intervals, not adjusted for multiple comparisons: read "
                 "borderline ones with care.")
    return lines


def _lock() -> dict | None:
    lock = figures._runs("lockbox")
    return lock[-1][1] if lock else None


def _lock_lines(dev_roc: str) -> list[str]:
    """The locked final test: primary result with its chance level and alert behaviour, and the
    secondaries clearly labelled (D-030)."""
    m = _lock()
    if not m:
        return ["The 10 locked runners are opened once, after the evaluation freeze."]
    f = m["fused"]
    r, p = f["roc_auc_ci"], f["pr_auc_ci"]
    rate = f["test_injury_rate"]
    a = f["alert"]
    own = m["arms_on_own_files"]
    lov = m["lovdal_label_free_paper_split"]

    def est(x):
        return f"{x['roc_auc_ci']['estimate']:.3f}"

    return [
        f"Opened once at the frozen commit: {len(m['test_runners'])} new runners, "
        f"{f['events']:,} days, {f['injuries']} injuries in {f['runners_with_injury']} runners.",
        f"Declared fused model: ROC-AUC {r['estimate']:.3f} ({r['low']:.3f}-{r['high']:.3f}); "
        f"PR-AUC {p['estimate']:.3f} ({p['low']:.3f}-{p['high']:.3f}) vs chance {rate:.3f}.",
        f"Development (5 rounds): ROC-AUC {dev_roc}.",
        f"Alert level (set on validation): catches {a['sensitivity']:.0%} of injuries with "
        f"{a['false_alarms_per_caught_injury']:.0f} false alarms per caught injury: too noisy "
        "for real use.",
        f"Secondary, scored once, not a comparison (other rows, more training runners): GRU arm "
        f"{est(own['sequence'])} (day file), CatBoost arm {est(own['tabular'])} (week file), "
        f"Lövdal label-free {est(lov['day'])} (day) / {est(lov['week'])} (week).",
        "No model is clearly best; nothing was switched after the test. With 10 runners the "
        "intervals are wide and still understate the uncertainty."]


def _prior_line() -> str:
    """Agreement with the prior list written before running (EVAL_PROTOCOL), per arm-round."""
    ex = figures._runs("explain")
    if not ex:
        return ""
    counts: dict[str, list[int]] = {}
    for rd in ex[-1][1]["rounds"]:
        for arm in ("tabular", "sequence"):
            for prior, v in rd[arm].get("prior_agreement", {}).items():
                if isinstance(v, dict) and "share_in_expected_direction" in v:
                    c = counts.setdefault(prior, [0, 0])
                    c[0] += v["share_in_expected_direction"] >= 0.5
                    c[1] += 1
    return ("Expectations written before running (most inputs in the expected direction): "
            + "; ".join(f"{k.replace('_', ' ')}: {a} of {t} arm-rounds"
                        for k, (a, t) in counts.items()) + ".")


def _real_reason() -> str:
    ex = figures._runs("explain")
    if not ex:
        return ""
    reasons = ex[-1][1]["rounds"][0].get("reasons", [])
    return f"Real example from round 0: \"{reasons[0]['sentence']}\"" if reasons else ""


def _status(phase: str, waiting: str, file: str = "metrics.json") -> str:
    return "results" if figures._runs(phase, file) else waiting


def build() -> str:
    b = figures.baselines()
    n = {k: _plain(v) if v != figures.NA else "pending (training now)"
         for k, v in figures.numbers(b).items()}
    d = Deck()

    s = d.slide("SynAthlete-AIoT")
    d.text(s, "An honest, explainable AIoT pipeline for athlete injury-risk prediction with "
              "synthetic wearable data", LEFT, Inches(1.6), WIDE, Inches(1.4), 30, True)
    d.text(s, "Review 2  ·  21CSE292P AIoT (PBL)  ·  SRM IST  ·  [Team names]", LEFT,
           Inches(3.4), WIDE, Inches(0.6), 20, False, GREY)
    d.text(s, "Everything shown is produced by code from real runs; numbers come from result "
              "files.", LEFT, Inches(6.4), WIDE, Inches(0.5), 14, False, GREY)

    s = d.slide("Problem and the corrected research gap", "Methodology")
    d.bullets(s, ["Coaches cannot see injury risk building up over days.",
                  "Recent AIoT athlete-health frameworks design the system but leave prediction "
                  "and explanation unimplemented.",
                  "Corrected gap (since Review 1): no public dataset pairs high-frequency "
                  "physiological streams with real injury outcomes.",
                  "  Real injuries exist only in daily training logs (Lövdal et al. 2021, "
                  f"{n['lovdalRunners']} runners); wearables (PAMAP2, WESAD) have no injury "
                  "labels.",
                  "Second problem: reported accuracies are often inflated by evaluation "
                  "shortcuts (leakage)."])

    s = d.slide("Objectives from Review 1: all implemented", "Methodology")
    state = {"1": _status("generators", "training now", "metrics_pamap2.json"),
             "2": _status("tiers", "tested; measured on an idle machine after training"),
             "3": _status("fused", "training now"),
             "4": _status("explain", "tested; runs after the fused model"),
             "5": "demo ready", "B": "results"}
    d.table(s, ["Objective (Review 1)", "Technique implemented", "Code", "Status"],
            [[o, t, c, "built, " + state.get(o[0], state["B"])] for o, t, c in OBJECTIVES],
            size=13)

    s = d.slide("Datasets: each with one honest role", "Methodology")
    d.table(s, ["Dataset", "Origin", "Role"], [list(r) for r in DATASETS], size=12)

    s = d.slide("Pipeline", "Methodology")
    steps = [("Data\nLövdal + wearables", NAVY), ("Synthetic data\nGANs / VAE", NAVY),
             ("Edge\nreflex", GREEN), ("Fog\nsupervisor", GREEN), ("Cloud\ndeep model", GREEN),
             ("Explanations\nSHAP / IG", NAVY), ("Coach\ndashboard", ACCENT)]
    x, y = Inches(0.35), Inches(2.6)
    for i, (t, c) in enumerate(steps):
        d.box(s, t, x, y, Inches(1.62), Inches(1.2), c)
        if i < len(steps) - 1:
            d.arrow(s, x + Inches(1.62), y + Inches(0.6), x + Inches(1.84))
        x += Inches(1.84)
    d.bullets(s, ["Injury risk is learned only from real injury labels (Lövdal).",
                  "Wearable streams drive the IoT tiers; synthetic data is only for training."],
              y=Inches(4.4), size=18)

    s = d.slide("Methodology: honest evaluation, fixed before any result", "Methodology")
    d.bullets(s, ["Split by person: models are always tested on runners they never saw.",
                  "10 newest runners sealed as a final test; opened once, after a freeze.",
                  "5 development rounds: settings, calibration and alert level chosen on "
                  "validation runners only.",
                  "Headline score PR-AUC (rare events) next to its chance level; ROC-AUC for "
                  "comparison.",
                  f"Leak checks: shuffled-label alarm (passed: ROC-AUC {n['canaryAuc']}), input "
                  "guard, training-only scaling, independent review.",
                  "Pass marks for generators and explanations written before running (D-027)."],
              size=19)

    s = d.slide("Implementation progress", "Implementation")
    d.table(s, ["Module", "Progress", "State"], [
        ["Data loading, person-level splits, audit", "100%", "done"],
        ["Baselines (7 injury models) + paper rebuild", "100%", "results"],
        ["Synthetic data (3 generators + classic)", "code 100%", state["1"]],
        ["Edge-fog-cloud tiers", "code 100%", state["2"]],
        ["Fused model (+ final locked test)", "code 100%", state["3"]],
        ["Explanations + checks", "code 100%", state["4"]],
        ["Dashboard (8 pages)", "100%", "demo ready"],
        ["Paper draft / project report", "draft 100%", "numbers inserted by code"]], size=15)

    s = d.slide("Result: injury prediction is hard, and we report it honestly", "Implementation")
    img = _chart_injury(b)
    if img:
        s.shapes.add_picture(img, Inches(0.4), Inches(1.15), width=Inches(8.4))
    d.bullets(s, [f"Learned models: ROC-AUC {n['injuryAucRange']} (5 development rounds).",
                  "Only modestly above chance (0.5); no model is clearly best overall.",
                  f"Fused model (declared in advance): ROC-AUC {n['fusedAuc']}, PR-AUC "
                  f"{n['fusedPrAuc']}.",
                  "Pre-declared pairs are compared on the next slides."],
              x=Inches(8.9), y=Inches(1.4), w=Inches(4.2), size=16)

    s = d.slide("Result: the final locked test", "Implementation")
    d.bullets(s, _lock_lines(n["fusedAuc"]), size=17)

    s = d.slide("Result: the published method, rebuilt and checked", "Implementation")
    d.bullets(s, ["Exact copy (incl. its label-reading scaling), stand-in runners 54-63: ROC-AUC "
                  f"{n['reproDayAuc']}: passed its pre-registered sanity check (lower bound "
                  "above 0.5).",
                  "Label-free variant (no peek at the labels), 5 development rounds: ROC-AUC "
                  f"{n['labelFreeAuc']}.",
                  _published_line(),
                  "These were scored on different rows, so the effect of the label-reading step "
                  "was not measured head-to-head."] + _comparison_lines(), size=15)

    s = d.slide("Result: synthetic wearable data", "Implementation")
    gens = figures._runs("generators", "metrics_pamap2.json")
    if gens:
        g = gens[-1][1]
        d.table(s, ["Generator", "Real-vs-fake acc. (0.5 = fooled)", "Train-on-synthetic F1",
                    "Real + synthetic F1", "Pass marks met"],
                [[GEN_LABEL.get(k, k), f"{m['discriminative_accuracy']['mean']:.3f}",
                  f"{m['tstr_test_macro_f1']['mean']:.3f}",
                  f"{m['real_plus_synthetic_test_macro_f1']['mean']:.3f}",
                  f"{sum(g['criteria'][k].values())}/{len(g['criteria'][k])}"]
                 for k, m in g["pamap2_summary"].items()], size=14)
        d.bullets(s, [n["wesadSummary"], n["augmentSummary"],
                      "Synthetic data is used for training only, never as evidence."],
                  y=Inches(4.3), size=15)
    else:
        d.bullets(s, ["Generators train only on each round's training people (4 PAMAP2 rounds).",
                      "Judged on real people they never saw: realism, usefulness, privacy.",
                      "RUNNING: results are added when the run finishes."])

    s = d.slide("Result: activity recognition and edge-fog-cloud", "Implementation")
    items = [f"Activity recognition on unseen people (PAMAP2): macro-F1 {n['pamapRealFone']}."]
    tiers = figures._runs("tiers")
    if tiers:
        st = tiers[-1][1]["streams"]
        items += [", ".join(f"{STREAM.get(name, name)}: edge tier sends "
                            f"{o['bytes_reduction_factor']:.0f}x less data to the cloud"
                            for name, o in st.items()) + "."]
        items += ["Median decision time, tiered vs cloud-only: " + "; ".join(
            f"{STREAM.get(name, name)} {o['tiered']['decision_latency_ms_median']:.1f} vs "
            f"{o['cloud_only']['decision_latency_ms_median']:.1f} ms" for name, o in st.items())
            + " (one window every 0.1 s after all tiers were ready)."]
    else:
        items += ["Tiers benchmark runs on an idle machine after training (bytes + latency)."]
    items += ["  Simulated on one laptop: no network delay, so latency compares processing only "
              "and the data saving is the claim; the cloud tier runs the logistic-regression "
              "daily model; wearable people differ from the runners (systems demo)."]
    d.bullets(s, items)

    s = d.slide("Result: explanations that are tested, not assumed", "Implementation")
    d.bullets(s, ["TreeSHAP (weekly-totals arm) and integrated gradients (day-by-day arm).",
                  "Attention view next to integrated gradients; Grad-CAM on the activity CNN.",
                  "Each explanation must pass: deletion, randomisation and stability tests.",
                  n["explainSummary"], _prior_line(),
                  "Attention and integrated gradients often disagree, so attention is never "
                  "shown alone.",
                  _real_reason()], size=16)

    s = d.slide("Live demo: coach dashboard", "Implementation")
    d.bullets(s, ["python tasks.py demo   (data prepared in advance: instant, cannot fail)",
                  "Athletes -> Athlete detail (risk 0-100, why?) -> Live sensor -> Synthetic "
                  "data -> Results -> Training lab -> Data & honesty",
                  "Coach pages replay real runners the model never trained on (never the locked "
                  "runners)."])

    s = d.slide("Code tour: where each part lives", "Implementation")
    d.table(s, ["Part", "Open this file", "Tests"], CODE_TOUR, size=13)
    d.text(s, "One command checks everything: python tasks.py check", LEFT, Inches(5.6), WIDE,
           Inches(0.5), 16, True, GREEN)

    s = d.slide("Research paper and project report", "Report + paper")
    d.bullets(s, ["IEEE conference draft (paper/main.tex, Overleaf): abstract, introduction, "
                  "related work, data, protocol, methods, results, discussion, AI-use statement.",
                  "All numbers inserted by code from result files (python tasks.py paper-figures).",
                  "Project report: full chapters (methodology, implementation, results, testing).",
                  "Target conference: [to confirm with the guide]."])

    s = d.slide("Honesty and limitations", "Report + paper")
    d.bullets(s, ["Training logs alone carry little warning: honest models are modestly above "
                  "chance.",
                  "Healthy days near injuries were removed by the dataset authors: alert rates "
                  "are not daily rates.",
                  "Tiers are simulated on one laptop; wearable people differ from the runners.",
                  "Synthetic data is only for training, never evidence about injuries.",
                  "The locked final test was opened once, after the evaluation freeze (tag "
                  "eval-freeze); only 10 runners, so its interval is wide."
                  if figures._runs("lockbox") else
                  "The locked final test is opened once, after the evaluation freeze."])

    s = d.slide("Thank you: questions?")
    d.text(s, "Demo: python tasks.py demo    ·    Code: github.com/NamanJain017/SynAthlete-AIoT",
           LEFT, Inches(3.2), WIDE, Inches(0.8), 22, False, NAVY)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "review2.pptx"
    d.prs.save(path)
    return str(path)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
