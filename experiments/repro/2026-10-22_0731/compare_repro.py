"""Compare a fresh-clone rebuild with the main working copy (read-only on both sides).

Run with the clone as the current directory so `src` is the clone's code:
    python compare_repro.py <main repo> <clone>
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

main, clone = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(clone))
problems: list[str] = []
checked: dict[str, int] = {}


def note(kind: str) -> None:
    checked[kind] = checked.get(kind, 0) + 1


# 1. MANIFEST rows
def rows(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if rows(main / "data/MANIFEST.csv") != rows(clone / "data/MANIFEST.csv"):
    problems.append("MANIFEST rows differ")
note("MANIFEST")

# 2. audit facts (newest clone run vs the committed main run)
main_run = main / "experiments/audit/2026-10-15_0819"
clone_runs = sorted(p for p in (clone / "experiments/audit").iterdir()
                    if (p / "audit.json").exists() and p.name != main_run.name)
a = json.loads((main_run / "audit.json").read_text(encoding="utf-8"))
b = json.loads((clone_runs[-1] / "audit.json").read_text(encoding="utf-8"))
a.pop("run_id"), b.pop("run_id")
if a != b:
    diff = [k for k in a["datasets"] if a["datasets"][k] != b["datasets"].get(k)]
    problems.append(f"audit facts differ for {diff}")
note("audit.json")
fa = json.loads((main_run / "run_info.json").read_text(encoding="utf-8"))["data_fingerprints"]
fb = json.loads((clone_runs[-1] / "run_info.json").read_text(encoding="utf-8"))["data_fingerprints"]
if fa != fb:
    problems.append("data fingerprints differ")
note("fingerprints")

# 3. every processed file
RUNTIME = {"workers"}
for p in sorted((main / "data/processed").rglob("*")):
    if not p.is_file() or p.name == ".gitkeep":
        continue
    q = clone / "data/processed" / p.relative_to(main / "data/processed")
    rel = p.relative_to(main).as_posix()
    if not q.exists():
        problems.append(f"missing in clone: {rel}")
        continue
    if p.suffix == ".npz":
        with np.load(p) as x, np.load(q) as y:
            if sorted(x.files) != sorted(y.files):
                problems.append(f"{rel}: different arrays")
            for k in x.files:
                u, v = x[k], y[k]
                same = (np.array_equal(u, v, equal_nan=True) if u.dtype.kind == "f"
                        else np.array_equal(u, v))
                if not same or u.dtype != v.dtype:
                    problems.append(f"{rel}[{k}] differs")
        note("npz")
    elif p.suffix == ".json":
        x = {k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items() if k not in RUNTIME}
        y = {k: v for k, v in json.loads(q.read_text(encoding="utf-8")).items() if k not in RUNTIME}
        if x != y:
            problems.append(f"{rel}: keys differ {[k for k in x if x[k] != y.get(k)]}")
        note("json")
    elif p.suffix == ".parquet":
        if not pd.read_parquet(p).equals(pd.read_parquet(q)):
            problems.append(f"{rel} differs")
        note("parquet")
extra = {q.relative_to(clone / "data/processed").as_posix()
         for q in (clone / "data/processed").rglob("*") if q.is_file()} - \
        {p.relative_to(main / "data/processed").as_posix()
         for p in (main / "data/processed").rglob("*") if p.is_file()}
if extra:
    problems.append(f"only in clone: {sorted(extra)}")

# 4. data cards (the clone's cards name its own audit run)
for p in sorted((main / "docs/data_cards").glob("*.md")):
    x = p.read_text(encoding="utf-8")
    y = (clone / "docs/data_cards" / p.name).read_text(encoding="utf-8")
    if x != y.replace(clone_runs[-1].name, main_run.name):
        problems.append(f"data card differs: {p.name}")
    note("data card")

# 5. split files rebuilt in memory from the clone's raw copy (never written)
from src.data import splits  # noqa: E402  (the clone's code)

DROP = {"created_utc", "git_commit", "git_dirty", "decision", "supersedes"}
built = splits.build_all()
for name, obj in built.items():
    saved = json.loads((clone / "data/splits" / name).read_text(encoding="utf-8"))
    for d in (obj, saved):
        d["meta"] = {k: v for k, v in d["meta"].items() if k not in DROP}
    if json.loads(json.dumps(obj)) != saved:
        problems.append(f"split rebuilt in memory differs: {name}")
    note("split file")

print("checked:", checked)
print("PROBLEMS:" if problems else "ALL IDENTICAL", *problems, sep="\n  ")
sys.exit(1 if problems else 0)
