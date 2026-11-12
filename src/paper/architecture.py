"""python -m src.paper.architecture: the system architecture diagram (figures/0_architecture.png).

Drawn with boxes and arrows from what the code does (src/pipeline, src/models, src/gen,
src/xai, src/dashboard); it contains no result numbers.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.data import loaders  # noqa: E402

OUT = loaders.ROOT / "figures" / "0_architecture.png"
COL = {"data": "#dbe7f3", "iot": "#d8efd9", "model": "#f6e2c8", "xai": "#eadcf0",
       "app": "#f8d7d4", "guard": "#ffffff"}


def box(ax, x, y, w, h, lines, kind, bold=False, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=COL[kind], ec="#333333", lw=1.2,
                                ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center", fontsize=8.6,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, label=None, dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 color="#333333", lw=1.2, ls="--" if dashed else "-"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.14, label, ha="center", fontsize=7.2,
                style="italic", color="#444444")


def track_b(ax) -> None:
    """Wearable stream through the edge, fog and cloud tiers."""
    ax.text(0.2, 7.85, "Track B: wearable stream and IoT tiers (systems evidence)",
            fontsize=10, fontweight="bold", color="#2e7d32")
    box(ax, 0.2, 6.4, 2.2, 1.1, ["Wearable sensors", "PAMAP2 (HR + 3 IMUs)",
                                 "Sport DB 2.0 (HR)"], "data")
    box(ax, 3.0, 6.4, 2.4, 1.1, ["EDGE tier (reflex)", "5-s window check:",
                                 "HR / motion rule +", "activity CNN"], "iot", True)
    box(ax, 6.0, 6.4, 2.2, 1.1, ["FOG tier (supervisor)", "session summaries",
                                 "(12 windows)"], "iot", True)
    box(ax, 8.8, 6.4, 2.2, 1.1, ["CLOUD tier", "daily injury model,", "session log"],
        "iot", True)
    box(ax, 11.4, 6.4, 1.4, 1.1, ["Cloud-only", "baseline:", "raw to cloud"], "guard",
        dashed=True)
    arrow(ax, 2.4, 6.95, 3.0, 6.95, "raw")
    arrow(ax, 5.4, 6.95, 6.0, 6.95, "summary")
    arrow(ax, 8.2, 6.95, 8.8, 6.95, "session")
    arrow(ax, 11.4, 6.95, 11.0, 6.95, dashed=True)


def track_a(ax) -> None:
    """Injury evidence: runner logs to the 0-100 risk score."""
    ax.text(0.2, 5.75, "Track A: injury-risk model (the only injury evidence)",
            fontsize=10, fontweight="bold", color="#b26a00")
    box(ax, 0.2, 4.1, 2.2, 1.35, ["Lövdal runner logs", "74 runners, real injuries",
                                  "day file (7 x 10)", "week file (3 x 22 + 3)"], "data")
    box(ax, 3.0, 4.1, 2.4, 1.35, ["Person-level splits", "5 development rounds",
                                  "(train / validation / test)",
                                  "+ SEALED test: 10 runners"], "guard", True)
    box(ax, 6.0, 4.75, 2.2, 0.7, ["Sequence arm", "GRU / LSTM / TCN"], "model")
    box(ax, 6.0, 3.95, 2.2, 0.7, ["Tabular arm", "Random forest / CatBoost"], "model")
    box(ax, 8.8, 4.1, 2.2, 1.35, ["Stacking combiner", "(out-of-fold, by runner)",
                                  "+ Platt calibration", "+ alert level (validation)"],
        "model", True)
    box(ax, 11.4, 4.1, 1.4, 1.35, ["Risk score", "0-100", "+ alert"], "app", True)
    arrow(ax, 2.4, 4.78, 3.0, 4.78)
    arrow(ax, 5.4, 5.05, 6.0, 5.05)
    arrow(ax, 5.4, 4.3, 6.0, 4.3)
    arrow(ax, 8.2, 5.05, 8.8, 4.9)
    arrow(ax, 8.2, 4.3, 8.8, 4.45)
    arrow(ax, 11.0, 4.78, 11.4, 4.78)
    arrow(ax, 9.9, 6.4, 9.9, 5.45)
    ax.text(10.0, 5.95, "runs the daily model", fontsize=7.2, style="italic",
            color="#444444")


def support(ax) -> None:
    """Generators, explanations, dashboard and the protocol strip."""
    box(ax, 0.2, 1.75, 2.6, 1.5, ["Synthetic data", "TimeGAN, WGAN-GP,",
                                  "TimeVAE, classic aug.", "(training people only)"], "data")
    box(ax, 3.3, 1.75, 2.6, 1.5, ["Generator checks", "real-vs-fake, TSTR,",
                                  "ACF / correlation,", "privacy distance"], "guard")
    box(ax, 6.4, 1.75, 3.0, 1.5, ["Explanations", "TreeSHAP (tabular), integrated",
                                  "gradients (sequence), attention,", "Grad-CAM; deletion /",
                                  "randomisation / stability tests"], "xai", True)
    box(ax, 9.9, 1.75, 2.9, 1.5, ["Coach dashboard", "(8 Streamlit pages)",
                                  "risk, reasons, live", "sensor, results, lab"], "app", True)
    arrow(ax, 2.8, 2.5, 3.3, 2.5)
    arrow(ax, 1.5, 3.25, 1.5, 4.1, dashed=True)
    ax.text(1.6, 3.62, "training rows only", fontsize=7.2, style="italic", color="#444444")
    arrow(ax, 9.9, 4.1, 8.0, 3.25)
    ax.text(9.2, 3.75, "explain", fontsize=7.2, style="italic", color="#444444")
    arrow(ax, 12.1, 4.1, 11.6, 3.25)
    arrow(ax, 9.4, 2.5, 9.9, 2.5)
    box(ax, 0.2, 0.25, 12.6, 0.95, [
        "Evaluation protocol (fixed before any run): person-level splits; scalers, generators "
        "and thresholds fitted on training / validation runners only;",
        "leak alarm; sealed test opened once after an annotated eval-freeze tag; runner "
        "bootstrap; every number written by code"], "guard")


def main() -> int:
    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    track_b(ax)
    track_a(ax)
    support(ax)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
