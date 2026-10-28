"""Windows-friendly task runner (replaces make). Usage: python tasks.py <task>"""

import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent

# Never put these into a snapshot zip: raw data, caches, old snapshots, secrets.
SNAPSHOT_SKIP_DIRS = {
    "backups", "raw", ".git", "__pycache__", ".venv", ".ruff_cache", ".pytest_cache",
}
SNAPSHOT_MAX_BYTES = 50_000_000


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def check() -> int:
    lint = run([sys.executable, "-m", "ruff", "check", "."])
    return lint or run([sys.executable, "-m", "pytest", "-q"])


def snapshot_includes(rel: Path) -> bool:
    if set(rel.parts) & SNAPSHOT_SKIP_DIRS:
        return False
    return not rel.name.startswith(".env")


def snapshot() -> int:
    """Zip the project (no raw data, backups, caches or secrets) into backups/."""
    out_dir = ROOT / "backups"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"snapshot_{datetime.now():%Y-%m-%d_%H%M}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob("*"):
            rel = p.relative_to(ROOT)
            if p.is_file() and snapshot_includes(rel) and p.stat().st_size < SNAPSHOT_MAX_BYTES:
                z.write(p, rel)
    print("wrote", out)
    return 0


def audit() -> int:
    """Dataset audit (piece 1); extra arguments pass through, e.g. --only lovdal."""
    return run([sys.executable, "-m", "src.data.audit", *sys.argv[2:]])


def data() -> int:
    """Loaders (piece 2): raw datasets -> data/processed/ (git-ignored), then the data cards.
    Optional dataset names limit the run, e.g. python tasks.py data lovdal."""
    return run([sys.executable, "-m", "src.data.loaders", *sys.argv[2:]])


def baselines() -> int:
    """First real models (piece 3): writes experiments/baselines/<run_id>/.
    Optional --only NAME,NAME, e.g. python tasks.py baselines --only canary."""
    return run([sys.executable, "-m", "src.models.run", *sys.argv[2:]])


def gen() -> int:
    """Generators (piece 5): python tasks.py gen pamap2 | followup --from RUN_ID."""
    return run([sys.executable, "-m", "src.gen.run", *sys.argv[2:]])


def pipeline() -> int:
    """Edge-fog-cloud tiers benchmark (piece 6)."""
    return run([sys.executable, "-m", "src.pipeline.bench", *sys.argv[2:]])


def train() -> int:
    """Fused model on the 5 development rounds (piece 7)."""
    return run([sys.executable, "-m", "src.models.fused", *sys.argv[2:]])


def splits() -> int:
    """Person-level splits (piece 2): written once to data/splits/, never overwritten.
    --seal pins their hashes; --remake exists only for the logged one-time exception (D-020)."""
    return run([sys.executable, "-m", "src.data.splits", *sys.argv[2:]])


def hooks() -> int:
    """Turn on the git guard that blocks commits to protected files (once per clone)."""
    code = run(["git", "config", "core.hooksPath", ".githooks"])
    if code == 0:
        print("git hooks enabled: commits touching protected files are now blocked")
    return code


def not_implemented(piece: str):
    def _f() -> int:
        print(f"not implemented yet ({piece}, see docs/PLAN.md)")
        return 1

    return _f


TASKS = {
    "check": check,
    "snapshot": snapshot,
    "audit": audit,
    "data": data,
    "splits": splits,
    "hooks": hooks,
    "baselines": baselines,
    "gen": gen,
    "pipeline": pipeline,
    "train": train,
    "explain": not_implemented("piece 8"),
    "demo": not_implemented("piece 9"),
    "paper-figures": not_implemented("piece 10"),
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in TASKS:
        print("tasks:", ", ".join(TASKS))
        sys.exit(2)
    sys.exit(TASKS[name]())
