# SETUP on Windows (PowerShell)

Run these in PowerShell inside your empty project folder.

## 1. Get the code
```powershell
git clone https://github.com/NamanJain017/SynAthlete-AIoT.git .
```
(The trailing dot clones into the current, empty folder.)

## 2. Python environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

## 3. PyTorch with CUDA (GPU)
Plain `pip install torch` on Windows gives the CPU-only build. Pick the command for your CUDA version from the selector at https://pytorch.org/get-started/locally/ (Windows, Pip, Python, your CUDA). Check your CUDA version with `nvidia-smi`. Then confirm:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
It must print `True` and your RTX 4050.

## 4. Everything else
```powershell
pip install -r requirements.txt
pip freeze > requirements.lock.txt
python tasks.py check
```
`check` runs ruff (lint) and pytest. It must end without errors.

## 5. Datasets
Follow docs/DATASETS.md. Put each dataset in its own folder under `data/raw/`. These folders are git-ignored; never commit them.

## 6. Start Claude Code
```powershell
claude
```
First message: "Read AGENTS.md and docs/PROJECT_BRIEF.md, docs/PLAN.md, docs/STATE.md. We are on piece 0/1. Check the setup, then help me with the dataset audit."
Run `/hooks` once and confirm the guard hook is listed.
