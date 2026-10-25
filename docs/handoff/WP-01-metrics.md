# WP-01 — The score calculator (`src/eval/metrics.py`)

| | |
|---|---|
| Status | **Ready for a builder** |
| Piece | 3 (first real models) |
| Written by | Claude Code, 2026-10-25 |
| Best builder | Antigravity with **Claude Opus 4.6 (Thinking)** (until 2 Nov 2026), else Gemini 3.8 Flash (High); backup: ChatGPT Go Codex |
| Branch | `wp-01-metrics` |

## 1. Goal (one sentence)
Write the maths that turns a model's risk numbers into the project's scores, so every model in piece 3 is marked with exactly the same answer key (decisions D-022 and D-024).

## 2. Files
- **Change:** `src/eval/metrics.py` only. **First** set `IMPLEMENTED = True` at the top (the 22 tests are skipped until you do), then fill in every function body (they currently `raise NotImplementedError`) until all tests pass.
- **May add:** one line at the end of `docs/AI_LOG.md` (format below).
- **Never change:** anything else. Especially not `tests/`, `docs/handoff/`, `configs/`, `data/`, `AGENTS.md`, `CLAUDE.md`, `docs/DECISIONS.md`. A git hook blocks commits to these.
- **Never open:** `data/raw`, `data/processed`, `data/splits`. This task needs no data: the tests use tiny made-up examples.

## 3. What to write
The exact rules are in each function's docstring in `src/eval/metrics.py` (that is the specification). Summary:

| Function | Returns |
|---|---|
| `average_precision(y_true, score)` | PR-AUC as average precision (step-wise, no straight-line interpolation: use `sklearn.metrics.average_precision_score`). ValueError if there is no injury |
| `roc_auc(y_true, score)` | ROC-AUC; `None` if only one class |
| `choose_alert_threshold(y_val, score_val)` | the alert level: candidates = distinct scores + infinity; smallest \|sensitivity − specificity\|; ties → larger sensitivity + specificity; then the higher threshold |
| `alert_numbers(y_true, score, threshold)` | dict with tp, fp, tn, fn, alerts, sensitivity, specificity, precision, false_alarms_per_caught_injury, alerts_per_100_events, mcc; **None** whenever a denominator is 0 |
| `fit_platt`, `apply_platt`, `platt_slope_flag` | Platt scaling = unpenalised logistic regression on the raw score; flag a slope ≤ 0 |
| `brier(y_true, prob)` | mean squared error of probabilities |
| `ece_equal_count(y_true, prob, bins=10)` | calibration error with equal-count bins (stable sort + `numpy.array_split`) |
| `calibration_slope_intercept(y_true, prob)` | slope from `y ~ c + b·logit(p)`; intercept from `logit(p)` used as an offset |
| `net_benefit`, `net_benefit_treat_all` | decision-curve net benefit per threshold |
| `macro_f1(y_true, y_pred)` | macro-F1 over labels present in `y_true` only |

## 4. Rules
- Pure functions: no reading or writing files, no printing, no global state, no randomness.
- Accept lists or numpy arrays. Return plain Python `float`/`int` (not numpy scalars), `None` where undefined, numpy arrays only where the docstring says so.
- Raise `ValueError` when inputs have different lengths, and in the cases the docstrings name.
- Use only numpy, scipy and scikit-learn (already installed). Do not add packages.
- Do not rename functions or change their arguments. Do not add new files.
- Keep the code short and readable, with a few comments that explain *why*. The project's owner must be able to explain it in a viva.
- **"Unpenalised" logistic regression:** turn off regularisation. In scikit-learn 1.9 check the current way (e.g. `C=numpy.inf` or `penalty=None`), or minimise the log-loss with `scipy.optimize`.

## 5. Worked examples (all are in the tests)
- `y = [0, 0, 1, 1]`, `score = [0.1, 0.4, 0.35, 0.8]`:
  - average precision **0.8333**;
  - ROC-AUC **0.75**;
  - alert threshold **0.4**;
  - at 0.4: tp 1, fp 1, tn 1, fn 1, alerts per 100 events 50, MCC 0.
- At threshold `inf` (nobody alerted): precision, false alarms per caught injury and MCC are **None**.
- `y = [1, 0, 1, 0, 0, 0]`, `score = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2]` → threshold **0.7**. The tie with 0.8 is broken by the larger sensitivity + specificity.
- A constant score → threshold **inf** ("alert nobody").

## 6. Tricky cases
- Ties in scores: "distinct values" means `numpy.unique`.
- Floating-point comparisons in the threshold rule: use a tolerance of `1e-12`.
- `roc_auc` with one class returns `None`, but `average_precision` without injuries raises `ValueError`.
- ECE: groups can have unequal sizes (`array_split` handles it).

## 7. How to check your work
In PowerShell, from the project folder:
```
.venv\Scripts\python.exe tasks.py check
```
Done when ruff says `All checks passed!` and pytest shows **no failures**. `tests/handoff/test_wp01_metrics.py` has 22 tests; they run only while `IMPLEMENTED = True`, so never set it back to False.

## 8. Definition of done
1. All 22 WP-01 tests pass, and every other test still passes.
2. Only `src/eval/metrics.py` (and optionally `docs/AI_LOG.md`) changed: check with `git status`.
3. Committed on branch `wp-01-metrics` with the message `WP-01: score calculator (src/eval/metrics.py)`, and pushed (`git push -u origin wp-01-metrics`).
4. One line added to `docs/AI_LOG.md`:
   `2026-MM-DD | Naman | <tool> (<model>) | WP-01 score calculator | Implemented src/eval/metrics.py from the task card | 22 WP-01 tests pass; python tasks.py check green`

## 9. If something is unclear
Stop and write the question for Naman. Do not guess, and do not change a test to make it pass.
If a test looks wrong, say which one and why. Claude Code will check it.

## 10. Prompt to paste into the builder
> You are working in the SynAthlete-AIoT repository. First read `AGENTS.md` (especially the section "Builder tools and task cards") and then `docs/handoff/WP-01-metrics.md`. Do exactly what the task card says and nothing more. Create and switch to a git branch named `wp-01-metrics`. In `src/eval/metrics.py`, first set `IMPLEMENTED = True` (this switches the 22 tests on), then fill in every function. Run `.venv\Scripts\python.exe tasks.py check` and fix your code until everything passes. Then add the AI_LOG line, commit with the message in the card and push the branch. Do not edit any test, task card, config or data file, and do not open the data folders. If anything is unclear, stop and ask me.
