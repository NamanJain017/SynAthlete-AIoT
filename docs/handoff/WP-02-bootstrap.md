# WP-02 — How sure are we? The runner bootstrap (`src/eval/bootstrap.py`)

| | |
|---|---|
| Status | **Merged** (2026-10-26; review: correct, cleaned up a leftover unfinished helper and a loop copied 3 times) |
| Piece | 3 (first real models) |
| Written by | Claude Code, 2026-10-25 |
| Best builder | Antigravity with **Claude Opus 4.6 (Thinking)** (until 2 Nov 2026), else Gemini 3.8 Flash (High); backup: ChatGPT Go Codex |
| Branch | `wp-02-bootstrap` (start it from the latest `main`) |

## 1. Goal (one sentence)
Measure how uncertain each score is by re-drawing whole runners many times (a "cluster bootstrap"). Every result can then be reported as "0.62 (0.55 to 0.69)" instead of a single number. The decision behind this is D-024.

## 2. Files
- **Change:** `src/eval/bootstrap.py` only. **First** set `IMPLEMENTED = True` at the top (the 11 tests are skipped until you do), then fill in every function until all tests pass.
- **May add:** one line at the end of `docs/AI_LOG.md` (format in section 8).
- **Never change:** anything else, especially `tests/`, `docs/handoff/`, `configs/`, `data/`, `src/eval/metrics.py`, `AGENTS.md`, `CLAUDE.md`, `docs/DECISIONS.md`. A git hook blocks commits to these.
- **Never open:** `data/raw`, `data/processed`, `data/splits`. The tests use small made-up data.

## 3. What to write
The docstrings in `src/eval/bootstrap.py` are the exact specification.

| Function | What it does |
|---|---|
| `resample_runners(athlete, group, rng)` | row indices of one bootstrap sample: inside each test group, draw that group's runners with replacement (same number of runners) and take ALL rows of each drawn runner |
| `group_mean(metric, y, score, group, idx=None)` | mean over test groups of `metric` on the chosen rows; `None` if the metric is undefined for any group |
| `bootstrap_ci(metric, y, score, athlete, group, resamples, seed, level)` | the score on all rows + its percentile interval; undefined resamples are discarded and counted |
| `paired_bootstrap_difference(metric, y, score_a, score_b, athlete, group, ...)` | interval of the difference between two models, both scored on the SAME resamples |
| `reproduction_check(y, scores_by_seed, athlete, metric, ...)` | the D-024 pass rule: mean ROC-AUC over seeds; passes if the interval's lower end is above 0.5 |

## 4. Rules
- Re-draw RUNNERS, never single rows. The draws must depend only on the runner lists (a runner with 50 rows has the same chance as one with 5).
- Paired means paired: both models use the same resampled rows in each resample.
- A resample where the metric cannot be computed is skipped and counted, never treated as 0. If more than half are skipped, raise `ValueError`.
- Use only `numpy.random.default_rng(seed)` for randomness, never the global random state. The same seed must give the same answer.
- Pure functions: no files, no printing. Return plain Python floats, ints and bools. Use only numpy. Do not rename functions or change their arguments, and do not add new files.
- Keep the code short and readable, with a few comments that explain *why*.

## 5. Worked examples (all are in the tests)
- Group 0: `y=[0,1]`, `score=[0.2,0.9]` gives AUC 1.0. Group 1: `y=[0,1,1]`, `score=[0.5,0.4,0.6]` gives AUC 0.5. So `group_mean` = **0.75**.
- Two identical models: the difference is exactly **0** (estimate, low and high all 0.0).
- Copying every row 30 times must not change the interval. Copies move together with their runner.
- Constant scores: AUC 0.5 in every resample, so low = 0.5 and `passed` is **False** (the rule needs *above* 0.5).

## 6. Tricky cases
- A runner drawn twice adds its rows twice: that is how a bootstrap works.
- Groups present: use only the groups among the rows you were given.
- `numpy.percentile` with its default method gives the interval ends.

## 7. How to check your work
In PowerShell, from the project folder:
```
.venv\Scripts\python.exe tasks.py check
```
Done when ruff says `All checks passed!` and pytest shows no failures. `tests/handoff/test_wp02_bootstrap.py` has 11 tests.

## 8. Definition of done
1. All 11 WP-02 tests pass, and every other test still passes.
2. Only `src/eval/bootstrap.py` (and optionally `docs/AI_LOG.md`) changed: check with `git status`.
3. Committed on branch `wp-02-bootstrap` with the message `WP-02: runner bootstrap (src/eval/bootstrap.py)`, and pushed (`git push -u origin wp-02-bootstrap`).
4. One line added to `docs/AI_LOG.md`:
   `2026-MM-DD | Naman | <tool> (<model>) | WP-02 runner bootstrap | Implemented src/eval/bootstrap.py from the task card | 11 WP-02 tests pass; python tasks.py check green`

## 9. If something is unclear
Stop and write the question for Naman. Do not guess, and do not change a test to make it pass.

## 10. Prompt to paste into the builder
> You are working in the SynAthlete-AIoT repository. First run `git checkout main` and `git pull`. Then read `AGENTS.md` (especially the section "Builder tools and task cards") and `docs/handoff/WP-02-bootstrap.md`. Do exactly what the task card says and nothing more. Create and switch to a git branch named `wp-02-bootstrap`. In `src/eval/bootstrap.py`, first set `IMPLEMENTED = True` (this switches the 11 tests on), then fill in every function. Run `.venv\Scripts\python.exe tasks.py check` and fix your code until everything passes. Then add the AI_LOG line, commit with the message in the card and push the branch. In PowerShell, separate commands with `;` (not `&&`). Do not edit any test, task card, config or data file, and do not open the data folders. If anything is unclear, stop and ask me.
