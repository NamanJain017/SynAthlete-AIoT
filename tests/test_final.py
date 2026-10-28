"""Final-test program (D-030, D-031) on made-up scores: no data, no lockbox, CPU, seconds."""

import numpy as np
import pytest

from src.data import lockbox
from src.models import final


def _fake(seeds=(0, 1), runners=6, rows=40, seed=0):
    rng = np.random.default_rng(seed)

    def labels():
        y = (rng.random(runners * rows) < 0.1).astype(int)
        y[::rows] = 1  # every runner has an injury, so every bootstrap resample is usable
        return y

    person = np.repeat(np.arange(runners), rows)
    raw = {"val_y": labels(), "val_person": person, "shared_y": labels(),
           "shared_person": person + 64, "day_y": labels(), "day_person": person + 64,
           "week_y": labels(), "week_person": person + 64}
    for s in seeds:
        for key in ("fused_val", "fused", "seq_shared", "tab_shared", "own_sequence_day",
                    "own_tabular_week", "lovdal_day", "lovdal_week"):
            raw[f"{key}_s{s}"] = rng.random(runners * rows)
    facts = {"label": "test", "quick_budget": False, "seeds": list(seeds),
             "combiner": [{"seed": s, "C": 1.0, "coef": [0.5, 0.5], "trials": []} for s in seeds],
             "sequence_arm": {"model": "gru"}, "tabular_arm": {"model": "catboost"}, "tries": 44}
    return facts, raw


@pytest.fixture(autouse=True)
def few_resamples(monkeypatch):
    """The bootstrap is tested in handoff/test_wp02_bootstrap.py; here only the summary around
    it, so 50 resamples keep these tests fast."""
    real = final.bootstrap.reproduction_check
    monkeypatch.setattr(final.bootstrap, "reproduction_check",
                        lambda *a, **k: real(*a, **{**k, "resamples": 50}))


def test_summarise_reports_every_part_from_the_saved_scores():
    out = final.summarise(*_fake())
    assert out["test_runners"] == list(range(64, 70))
    assert len(out["fused"]["seeds"]) == 2 and "estimate" in out["fused"]["roc_auc_ci"]
    assert out["fused"]["roc_auc_ci"]["resamples_used"] == 50
    assert out["fused"]["roc_auc_ci"]["bootstrap_seed"] == 0
    assert "estimate" in out["lovdal_label_free_paper_split"]["week"]["pr_auc_ci"]
    assert out["arms_on_own_files"]["tabular"]["role"].startswith("secondary")
    assert "6 runners" in out["summary_sentence"]


def test_summarise_keeps_the_primary_when_a_secondary_never_ran():
    facts, raw = _fake()
    for k in [k for k in raw if k.startswith(("lovdal_day", "lovdal_week"))]:
        del raw[k]
    out = final.summarise(facts, raw)
    assert out["lovdal_label_free_paper_split"]["day"]["not_available"]
    assert "estimate" in out["fused"]["roc_auc_ci"]
    assert "estimate" in out["arms_on_own_files"]["sequence"]["roc_auc_ci"]


def test_waits_for_a_short_network_drop_then_gives_up(monkeypatch):
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise lockbox.LockboxSealed("push the tag to GitHub first")

    monkeypatch.setattr(final, "_check_openable", flaky)
    monkeypatch.setattr(final.time, "sleep", lambda s: None)
    final._wait_until_openable(tries=5, pause_s=0)
    assert len(calls) == 3

    def never():
        raise lockbox.LockboxSealed("push the tag to GitHub first")

    monkeypatch.setattr(final, "_check_openable", never)
    with pytest.raises(lockbox.LockboxSealed):
        final._wait_until_openable(tries=2, pause_s=0)


def test_quick_budget_only_for_dry_runs():
    with pytest.raises(SystemExit):
        final.main(["--quick"])
