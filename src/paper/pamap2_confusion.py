"""python -m src.paper.pamap2_confusion: activity confusion matrix for the reported PAMAP2 CNN.

The piece-3 run (experiments/baselines/2026-10-26_022902) saved scores, not per-clip predictions.
This re-trains exactly its recorded model: round 0, seed 0, the setting chosen on validation
(no new tuning), on round 0's training people, and predicts round 0's test people. It records
whether the re-trained model reproduces the recorded test macro-F1; the confusion matrix is only
described as the reported model's if it does. Writes experiments/pamap2_confusion/<run_id>/.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import confusion_matrix

from src.data import loaders
from src.eval import metrics
from src.models import data, fitters
from src.models import run as base_run

SOURCE = "2026-10-26_022902"
OUT = loaders.ROOT / "experiments" / "pamap2_confusion"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    code = base_run._code_state()
    src = json.loads((loaders.ROOT / "experiments" / "baselines" / SOURCE / "metrics.json")
                     .read_text("utf-8"))["pamap2_cnn1d"]["rounds"][0]
    setting = src["trials"][src["chosen"]]["setting"]
    r = next(data.pamap2_rounds())
    t0 = time.perf_counter()
    predict, _ = fitters.pamap2_net("cnn1d")(r, setting, 0)
    pred = np.asarray(predict(r.test.X))
    f1 = metrics.macro_f1(r.test.y, pred)
    recorded = src["macro_f1_per_seed"][0]
    labels = sorted(set(r.test.y.tolist()) | set(pred.tolist()))
    out = {"source_run": SOURCE, "round": 0, "seed": 0, "setting": setting,
           "test_people": sorted(int(p) for p in np.unique(r.test.person)),
           "test_clips": int(len(r.test.y)), "macro_f1": float(f1),
           "recorded_macro_f1": float(recorded),
           "reproduces_record": bool(abs(f1 - recorded) < 1e-6),
           "labels": [int(v) for v in labels],
           "confusion": confusion_matrix(r.test.y, pred, labels=labels).tolist()}
    folder = OUT / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder.mkdir(parents=True)
    info = {**code, "runtime_s": round(time.perf_counter() - t0, 1),
            "data_fingerprints": base_run._fingerprints()}
    (folder / "run_info.json").write_text(json.dumps(info, indent=2), "utf-8")
    (folder / "metrics.json").write_text(json.dumps(out, indent=2), "utf-8")
    print(f"macro-F1 {f1:.4f} (recorded {recorded:.4f}); reproduces: {out['reproduces_record']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
