"""Seeded random search and the D-024 choice rule (written by Claude Code: leak-sensitive).

Every function here sees ONE round's validation scores only; nothing averages across rounds
(D-021: one round's validation runners are another round's test runners).
"""

from __future__ import annotations

import math

import numpy as np


def draw_settings(grid: dict, n: int, seed: int, fixed: dict | None = None) -> list[dict]:
    """n settings by seeded random search (Bergstra & Bengio 2012).

    A list in the grid is drawn uniformly; {"loguniform": [low, high]} is drawn log-uniformly.
    Keys are visited in sorted order, so the draws do not depend on the YAML key order.
    `fixed` values are copied into every setting.
    """
    rng = np.random.default_rng(seed)
    settings = []
    for _ in range(n):
        s = dict(fixed or {})
        for key in sorted(grid):
            spec = grid[key]
            if isinstance(spec, dict) and set(spec) == {"loguniform"}:
                low, high = spec["loguniform"]
                s[key] = float(math.exp(rng.uniform(math.log(low), math.log(high))))
            elif isinstance(spec, list) and spec:
                s[key] = spec[int(rng.integers(len(spec)))]
            else:
                raise ValueError(f"grid entry {key!r} must be a non-empty list or loguniform")
        settings.append(s)
    return settings


def choose(trials: list[dict]) -> int:
    """Index of the chosen trial: highest validation average precision, ties -> higher
    validation ROC-AUC, then the earlier trial (D-024). Trials need "val_ap" and "val_auc"."""
    if not trials:
        raise ValueError("no trials")
    key = [(t["val_ap"], -math.inf if t["val_auc"] is None else t["val_auc"]) for t in trials]
    return max(range(len(trials)), key=lambda i: (key[i], -i))
