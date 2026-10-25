# Handoff: building parts of the project with other AI tools

This sheet is the practical "how to" for giving coding jobs to other tools (Antigravity, ChatGPT Go / Codex) while keeping the project honest. The simple explanation of *why* we do this is in `docs/STUDY_GUIDE.md` (section "How the code is built") and `docs/VIVA_NOTES.md`.

## The idea in one picture
- **Claude Code = the teacher.** It writes each job as a **task card**, plus an **answer checker** (tests) and an empty code file with the function names.
- **Antigravity / Codex = the student.** It writes the code.
- **The answer checker** says pass or fail, and automatic guards stop rule-breaking.
- **Claude Code then marks the final copy:** it reviews the code, runs it on the real data, and merges it.

Hard, leak-sensitive parts (how rounds are built, the tuning loop, the paper-method copy, the leak alarm, the combined model) stay with Claude Code.

## One-time setup (already done on Naman's laptop)
```
.venv\Scripts\python.exe tasks.py hooks
```
This turns on the git guard (`.githooks/`). It refuses any commit that touches protected files.

## The loop for each task card
1. **Pick a card** from the table below with status *Ready for a builder* (e.g. `docs/handoff/WP-01-metrics.md`).
2. **Open the project folder in Antigravity** and choose the model:
   - **Claude Opus 4.6 (Thinking)** while it is in the list (Google removes it on 2 Nov 2026);
   - after that, **Gemini 3.8 Flash (High)**;
   - if Antigravity's limit runs out, use **ChatGPT Go → Codex** with the same card.
3. **Paste the prompt from section 10 of the card.** In short: *"Read AGENTS.md and docs/handoff/WP-01-metrics.md and do exactly what it says."*
4. **Let it work.** It creates the branch (e.g. `wp-01-metrics`), switches `IMPLEMENTED = True`, writes the code, and runs `.venv\Scripts\python.exe tasks.py check` until everything passes. Then it commits and pushes the branch.
5. **Optional second opinion.** In Codex: *"Check branch wp-01-metrics against docs/handoff/WP-01-metrics.md and list any mistakes. Do not change files."*
6. **Come back to Claude Code** and say **"review WP-01"**. Claude reviews the change, runs it on the real data where that matters, fixes small things, merges it into `main`, and updates the explanation docs.

## What the guards do
| Guard | What it stops |
|---|---|
| **Answer checker** (`tests/handoff/test_wpNN_*.py`) | Wrong maths, wrong edge cases, missing "n/a" values |
| **Git hook** (`.githooks/`) | Commits that touch: tests, task cards, configs, data, results files, rulebooks, DECISIONS, the guards themselves. Only the owner can override it, with `ALLOW_PROTECTED=1`, and the change then shows up in review |
| **GitHub check** (`.github/workflows/check.yml`) | Any push or pull request where ruff or pytest fails. Shown as a red ✗ on GitHub |
| **Claude Code review** | Leaks, rule breaks, unclear code, anything the tests did not cover |
| **Rules in `AGENTS.md`** | Builders read them: stay inside the card's files, never open the data folders, never write numbers into documents, ask when unsure |

## Task cards
| Card | File | Status |
|---|---|---|
| WP-01 | `src/eval/metrics.py` (score calculator) | **Merged** (review found 2 problems + weak tests; fixed, D-025) |
| WP-02 | `src/eval/bootstrap.py` (uncertainty by resampling runners) | **Merged** (correct; review removed dead code and a loop copied 3 times) |
| WP-03 | `src/models/har.py` (PAMAP2 activity models) | Planned |

## If something goes wrong
- **Tests keep failing:** let the builder keep fixing its code. If it claims a test is wrong, bring that to Claude Code. Never edit the test.
- **The hook blocks a commit:** the builder touched a protected file. Undo that change (`git restore --staged <file>` and `git restore <file>`), then commit only the card's files.
- **The tool's limit ran out:** continue in the other tool on the same branch.
- **You are unsure what changed:** run `git status` and `git diff`, or ask Claude Code.

## Small glossary
- **Branch:** a separate copy of the code where a job is done without touching the main version.
- **Commit:** a saved snapshot of changes, with a message.
- **Push:** sending commits to GitHub.
- **Merge:** joining a finished branch into `main`.
- **Pull request:** a request on GitHub to merge a branch, where the changes can be reviewed.
- **Test:** a small program that checks the code gives the right answers.
- **Hook:** a script git runs automatically, here before every commit.
- **CI (continuous integration):** automatic checks GitHub runs on every push.
