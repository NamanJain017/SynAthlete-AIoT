# TODO: add the final results to the dashboard (task #23)

## Why
The dashboard's **Results** page shows baselines, the fused model, generators, tiers, Track C and explanations. Three results from 28–29 Oct are missing:
1. the **final locked test**;
2. the **paired comparisons**;
3. the **generator follow-up**: WESAD and the synthetic injury examples.

## Where things are
- `src/dashboard/prepare.py` builds `experiments/dashboard/bundle/results.json`. Today it has the keys `baselines`, `fused`, `generators`, `tiers`, `track_c` and `explain`.
- `src/dashboard/app.py` holds the 8 Streamlit pages. The Results page reads `results.json`.
- `src/dashboard/data.py` holds the loading helpers.
- Tests: `tests/test_dashboard.py` and `tests/test_dashboard_pages.py`. Every page must render with no errors.

## Steps
1. **In `prepare.py`**, where `results.json` is written, add three keys. Read each from the newest run folder, the same way the existing keys do:
   - `lockbox`: from `experiments/lockbox/2026-10-29_010127/metrics.json`. Take `fused.roc_auc_ci`, `fused.pr_auc_ci`, `fused.test_injury_rate`, `fused.events`, `fused.injuries`, `fused.runners_with_injury`, `fused.alert` (sensitivity, false_alarms_per_caught_injury), `arms_on_own_files`, `lovdal_label_free_paper_split` and `not_a_comparison`.
   - `comparisons`: from `experiments/comparisons/2026-10-29_020232/metrics.json`, the `comparisons` list. Each entry has `comparison`, and `pr_auc` and `roc_auc` with `estimate`, `low`, `high` and `interval_excludes_0`.
   - `followup`: from `experiments/generators/2026-10-28_172535/metrics_followup.json`. Take `wesad_summary`, `wesad_criteria` and `injury_summary`.

   If a file is missing, write `None` so the page shows "not available". Never type numbers by hand.
2. **In `app.py`**, Results page, add three sections:
   - **"Final locked test (opened once)":**
     - ROC-AUC and PR-AUC with their intervals, and chance (the injury rate);
     - "7 of 10 runners injured";
     - the alert line ("catches X % of injuries, Y false alarms per caught injury");
     - the secondaries, under the caption *"secondary, scored once, not a comparison; no model was switched"*.
   - **"Paired comparisons":** a table with `comparison`, ΔPR-AUC [low, high] and ΔROC-AUC [low, high], plus a ✔ when `interval_excludes_0`. Caption: *"runner bootstrap, 2,000 resamples; 24 intervals, not adjusted for multiple comparisons"*.
   - **"Synthetic data beyond PAMAP2":** the WESAD criteria per generator, and the injury augmentation PR-AUC and ROC-AUC against real-only GRU (0.023 PR-AUC, from the baselines). Caption: *"no improvement claimed"*.
   - The slide code `src/paper/deck.py` (`_lock_lines`, `_comparison_lines`) already builds this wording from the same files. Copy that logic.
3. **Run:**
   - `python tasks.py demo --prepare` (about 1 minute);
   - `python tasks.py check` (all tests must pass);
   - `python tasks.py demo`, then open the Results page and check it.
4. **Commit:** "Dashboard: final locked test, comparisons, generator follow-up".

## Rules to keep
- **Coach pages:** they must never show the locked runners (64–73), only development runners. Only the Results page shows the locked-test summary numbers.
- **Numbers:** they come only from the result files.
- **Attention:** it is never shown without integrated gradients next to it.
