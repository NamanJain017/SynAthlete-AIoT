#!/usr/bin/env bash
# Fresh-clone reproducibility check: clone, copy audited raw files, audit, build, test, compare.
set -euo pipefail
SP="/c/Users/jainn/AppData/Local/Temp/claude/C--Users-jainn-Documents-SynAthlete-AIoT/f050f711-67a9-4004-8f6f-a5c9af6c1e4d/scratchpad"
MAIN="/c/Users/jainn/Documents/SynAthlete-AIoT"
CLONE="$SP/repro_clone"
PY="$MAIN/.venv/Scripts/python.exe"
[ -e "$CLONE" ] && { echo "clone folder already exists: stop"; exit 2; }
t0=$(date +%s)
echo "== clone"; git clone -q "$MAIN" "$CLONE"; git -C "$CLONE" log --oneline -1
echo "== copy audited raw files"; "$PY" -I "$SP/copy_raw.py" "$MAIN" "$CLONE"
cd "$CLONE"
echo "== audit (fresh clone)"; "$PY" tasks.py audit 2>&1 | grep -v "files: [0-9]*/\|subjects: [0-9]*/\|recordings: [0-9]*/"
echo "== data builders (fresh clone)"; "$PY" tasks.py data 2>&1 | grep -v "files: [0-9]*/\|subjects: [0-9]*/\|recordings: [0-9]*/\|^wrote docs"
echo "== tests (fresh clone)"; "$PY" -m ruff check --no-cache . && "$PY" -m pytest -q -p no:cacheprovider 2>&1 | tail -2
echo "== compare with the main working copy"; "$PY" "$SP/compare_repro.py" "$MAIN" "$CLONE"
echo "== total $(( $(date +%s) - t0 )) s"
