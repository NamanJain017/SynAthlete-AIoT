# WP-03 — Neural networks and their training loop (`src/models/nets.py`)

| | |
|---|---|
| Status | **Ready for a builder** |
| Piece | 3 (first real models), also used in piece 7 |
| Written by | Claude Code, 2026-10-26 |
| Best builder | Antigravity with **Claude Opus 4.6 (Thinking)** (until 2 Nov 2026), else Gemini 3.8 Flash (High); backup: ChatGPT Go Codex |
| Folder and branch | the second folder `C:\Users\jainn\Documents\SynAthlete-wp03`, already on branch `wp-03-nets` |

## 1. Goal (one sentence)
Write the project's neural networks and ONE training loop that all of them share. That covers the activity models for PAMAP2 (piece 3), the injury-risk GRU (piece 3) and the GRU, LSTM and TCN of the combined model (piece 7). The loop trains with early stopping on the validation score and keeps the best epoch's weights.

## 2. Files
- **Change:** `src/models/nets.py` only. **First** set `IMPLEMENTED = True` at the top (the 43 tests are skipped until you do), then fill in every class and function until all tests pass.
- **May add:** one line at the end of `docs/AI_LOG.md` (format in section 8).
- **Never change:** anything else, especially `tests/`, `docs/handoff/`, `configs/`, `src/eval/`, `AGENTS.md`, `CLAUDE.md` and `docs/DECISIONS.md`. A git hook blocks commits to these.
- **Never open or load data:** this folder has no data on purpose. The tests use small made-up arrays.

## 3. What to write
The docstrings in `src/models/nets.py` are the exact specification: layer order, shapes, return keys and errors.

| Name | What it does |
|---|---|
| `seed_everything(seed)` | seeds Python, NumPy and torch; turns on PyTorch's deterministic settings |
| `pick_device(prefer_gpu)` | GPU if available, else CPU (teammates have no GPU) |
| `CNN1D` | 1D-CNN: `conv_blocks` × [Conv1d "same" → BatchNorm → ReLU → MaxPool(2)] → mean over time → Dropout → Linear |
| `GRUNet`, `LSTMNet` | GRU / LSTM → output at the last time step → Dropout → Linear |
| `TCN` | `levels` residual blocks, dilation 2^i, two causal convolutions each (zeros added on the LEFT only), 1×1 shortcut when widths differ → last step → Linear |
| `build_model(name, n_channels, n_outputs, params)` | builds one network from a settings dict; ignores training keys; refuses unknown or missing keys |
| `EarlyStopping(patience)` | remembers the best score; only a STRICTLY higher score counts; `should_stop` after `patience` epochs without one |
| `positive_weight(y, setting)` | `"balanced"` → number of 0s / number of 1s; a number → that number |
| `make_loss(task, pos_weight)` | BCE-with-logits (binary, with `pos_weight`) or cross-entropy (multiclass) |
| `make_optimizer(model, lr, weight_decay)` | Adam |
| `predict_scores(model, X, ...)` | probabilities in row order (sigmoid for 1 output, softmax for more); eval mode, no gradients, in batches |
| `predict_labels(model, X, classes, ...)` | `classes[argmax]`: turns output positions back into the real labels (e.g. activity 24) |
| `train_model(model, task, X_train, y_train, X_val, y_val, ...)` | the shared training loop with early stopping; returns the model with its best weights plus a history |

## 4. Rules
- **No data access.** The function receives arrays that were split and scaled elsewhere. Never read files, never import from `src.data`, and never use validation rows for anything except the validation score.
- **Repeatable.** Use the one `torch.Generator(seed)` for shuffling, as the docstring says. Never use NumPy's global random functions inside training. On the CPU, the same seed must give exactly the same history twice.
- **Count the model's outputs from its output shape** (run it on a few rows). Never read layer attributes, because the tests also use small stand-in models.
- **Labels:** multiclass training uses each label's position in `classes = numpy.unique(y_train)`. Validation may contain labels that are missing from training.
- Use only `numpy`, `torch` and `src.eval.metrics` (`macro_f1` and `average_precision`). No new libraries and no new files. Keep the stub's names, arguments and docstrings.
- Return plain Python `int`/`float` in the result and history, not NumPy or torch numbers.
- **Tidy code (every card):**
  - no copied blocks: if the same few lines would appear twice, write one small helper;
  - no leftovers: no unfinished functions, placeholders, `NotImplementedError`, commented-out code or `print`;
  - use the card's formulas exactly as written: never hand-type a constant that a formula defines;
  - short comments that explain *why*, not *what*.

## 5. Worked examples (all are in the tests)
- **Sizes** are checked by counting parameters (the learnable numbers) against formulas taken from the docstrings' layer lists. For example, `CNN1D(10, 18, channels=32, kernel=5, dropout=0.2)` has (10·32·5 + 32 + 2·32) + 2·(32·32·5 + 32 + 2·32) + (32·18 + 18) parameters. BatchNorm adds 2 per channel; its running averages are not parameters.
- **TCN reach:** with kernel 2 and 2 levels, the last output depends on exactly the last 1 + 2·1·3 = **7** steps. Changing the 7th-last step changes the output; changing anything older does not.
- **EarlyStopping(2)** with scores 0.1, 0.3, 0.3, 0.2 → new best at epochs 0 and 1 only (the tie is not a gain); best epoch 1; stops after the 4th score.
- **positive_weight([0, 0, 0, 1], "balanced") = 3.0.**
- **make_loss("binary", 3.0)** on logit 0 with targets 1 and 0 = (3·ln 2 + ln 2) / 2.
- **predict_labels** with `classes = [3, 7, 24]`: logits [0, 5, 1] → 7; logits [2, 0, 2] → a tie → lowest position → 3.

## 6. Tricky cases
- `padding="same"` for CNN1D. TCN causal padding: add `(kernel − 1) · dilation` zeros on the left only, so the length stays the same and nothing from the future leaks in.
- GRU/LSTM with 1 layer: pass `dropout=0.0` to the recurrent layer (PyTorch warns otherwise); the Dropout before the Linear head still uses `dropout`.
- Binary targets must have the same shape as the model output `(batch, 1)`, as float.
- Keep a **copy** of the best weights (e.g. `copy.deepcopy(model.state_dict())`), not a reference that later epochs overwrite.
- `train_loss` = mean loss per training row over the epoch (the last batch may be smaller, so weight each batch by its size).
- The GPU test runs only on a machine with a GPU. Your code must also work with `device="cpu"`.

## 7. How to check your work
In PowerShell, from `C:\Users\jainn\Documents\SynAthlete-wp03`:
```
C:\Users\jainn\Documents\SynAthlete-AIoT\.venv\Scripts\python.exe tasks.py check
```
(This folder shares the main folder's Python environment.) You're done when ruff says `All checks passed!` and pytest shows no failures. `tests/handoff/test_wp03_nets.py` has 43 tests, and the training tests take a few seconds on a CPU. Tests that need the private data are skipped in this folder; that is expected.

## 8. Definition of done
1. All 43 WP-03 tests pass, and every other test still passes.
2. Only `src/models/nets.py` (and optionally `docs/AI_LOG.md`) changed: check with `git status`.
3. Committed on branch `wp-03-nets` with the message `WP-03: neural networks and training loop (src/models/nets.py)`, and pushed (`git push -u origin wp-03-nets`).
4. One line added to `docs/AI_LOG.md`:
   `2026-MM-DD | Naman | <tool> (<model>) | WP-03 networks + training loop | Implemented src/models/nets.py from the task card | 43 WP-03 tests pass; python tasks.py check green`

## 9. If something is unclear
Stop and write the question for Naman. If a test still fails after 2–3 honest attempts and you believe the test is wrong, stop and explain why. Do not guess, and never change a test to make it pass.

## 10. Prompt to paste into the builder
Open the folder `C:\Users\jainn\Documents\SynAthlete-wp03` in Antigravity (not the main folder), then paste:
> You are working in a second copy of the SynAthlete-AIoT repository, already on branch `wp-03-nets`. Do not switch branches. Read `AGENTS.md` (especially "Builder tools and task cards") and `docs/handoff/WP-03-nets.md`, and do exactly what the task card says and nothing more. In `src/models/nets.py`, first set `IMPLEMENTED = True` (this switches the 43 tests on), then fill in every class and function. Check with `C:\Users\jainn\Documents\SynAthlete-AIoT\.venv\Scripts\python.exe tasks.py check` and fix your code until everything passes. Then add the AI_LOG line, commit with the message in the card and push the branch. In PowerShell, separate commands with `;` (not `&&`). Do not edit any test, task card, config or data file, and do not look for data. If anything is unclear, or a test still fails after 2–3 honest attempts, stop and ask me.
