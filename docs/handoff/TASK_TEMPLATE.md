# WP-NN — <short name> (`<file path>`)

| | |
|---|---|
| Status | Draft / Ready for a builder / Built (awaiting review) / Merged |
| Piece | <n> |
| Written by | Claude Code, <date> |
| Best builder | <tool + model> |
| Branch | `wp-NN-<name>` |

## 1. Goal (one sentence)
What this file does, and why the project needs it.

## 2. Files
- **Change:** the file(s) to fill in (stubs already exist, with `IMPLEMENTED = False`).
- **May add:** one line in `docs/AI_LOG.md`.
- **Never change:** everything else (tests, task cards, configs, data, rulebooks, DECISIONS).
- **Never open:** `data/raw`, `data/processed`, `data/splits`.

## 3. What to write
A table: function → what it returns. The docstrings in the stub file are the exact specification.

## 4. Rules
Project rules that apply here (leakage, purity, return types, allowed libraries, no new files).

## 5. Worked examples
Small inputs with their exact expected outputs (all of them are in the tests).

## 6. Tricky cases
Ties, empty inputs, undefined values, floating-point tolerance, ...

## 7. How to check your work
`.venv\Scripts\python.exe tasks.py check`, with the number of tests this card adds.

## 8. Definition of done
Tests green; only the allowed files changed; committed on the branch with the given message and pushed; AI_LOG line added.

## 9. If something is unclear
Stop and ask. Never change a test to make it pass.

## 10. Prompt to paste into the builder
A ready-made message telling the builder to read `AGENTS.md` and this card, use the branch, switch `IMPLEMENTED` on, make the tests pass, and stay inside the allowed files.
