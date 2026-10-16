"""Create the person-level splits ONCE (piece 2, AGENTS.md rule 2, decisions D-017, D-020).

    python -m src.data.splits             create all files (same as: python tasks.py splits)
    python -m src.data.splits --remake FILE... --decision D-xxx   one-time re-make, logged first
    python -m src.data.splits --seal      pin every file's sha256 in data/splits/SHA256SUMS

All files are written together or not at all, and never overwritten. Each file records seed,
method, library version, git commit and the data fingerprint. A Lövdal test group whose injury
count is over 20 % from the mean stops the run (no re-seeding). Once SHA256SUMS exists the
folder is sealed: nothing can be re-made, and the tests compare every file with its pinned hash.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedShuffleSplit

from src.data import audit, loaders, readers
from src.data.reference import only_file

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"
MANIFEST = ROOT / "data" / "MANIFEST.csv"
DECISIONS = ROOT / "docs" / "DECISIONS.md"
SEAL = SPLITS / "SHA256SUMS"
SUPERSEDED = SPLITS / "superseded"  # earlier versions are moved here, never deleted
SEED = 0
BALANCE_TOLERANCE = 0.20
N_LOCKBOX = 10
FILES = (
    "lovdal_folds.json", "pamap2_folds.json", "wesad_folds.json", "sportdb2_folds.json",
    "soccermon_lopo.json", "pmdata_lopo.json", "sirp600_demo_split.json",
)
METHODS = {
    "lovdal": "lockbox = the 10 highest athlete IDs (= most recently joined = the paper's "
              "test set); dev test groups: StratifiedGroupKFold(5, shuffle, seed 0) on the "
              "day file; val: first group of the same procedure on the remaining dev athletes",
    "pamap2": "seed-0 permutation of subjects 101-108 into 4 test groups of 2; "
              "val of fold k = 1 subject of test group k+1 (rotating, D-020); "
              "109 always in training",
    "wesad": "seed-0 permutation into 5 test groups of 3; "
             "val of fold k = 2 subjects of test group k+1 (rotating, D-020)",
    "sportdb2": "GroupKFold(5) on subject folders; recordings without a subject folder form "
                "one group; test groups only (realism-check reference)",
    "soccermon": "leave-one-player-out; predictions pooled into one score; "
                 "fixed default hyperparameters (no validation)",
    "pmdata": "leave-one-person-out over people with an injury file; pooled; "
              "results reported with and without p12",
    "sirp600": "StratifiedShuffleSplit(test_size=0.2, seed 0) by rows: DEMO ONLY, "
               "no athlete IDs exist, never evidence",
}


class BalanceError(RuntimeError):
    """A test group is outside the balance rule: stop and ask (D-017)."""


class SplitError(RuntimeError):
    """A request that would mix, overwrite or re-make split files outside the rules."""


# ---------------------------------------------------------------- helpers


def ints(values) -> list:
    """JSON-friendly, sorted IDs (numpy scalars become Python values)."""
    return sorted(v.item() if isinstance(v, np.generic) else v for v in values)


def stratified_groups(labels: np.ndarray, groups: np.ndarray, k: int) -> list[list]:
    """k test groups of whole people, keeping the injury rate similar in each group."""
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    dummy = np.zeros(len(labels))
    return [ints(np.unique(groups[te])) for _, te in sgkf.split(dummy, labels, groups)]


def check_balance(counts: list[int], label: str) -> None:
    mean = float(np.mean(counts))
    if any(abs(c - mean) > BALANCE_TOLERANCE * mean for c in counts):
        raise BalanceError(
            f"{label}: injuries per test group {counts} are not within "
            f"±{BALANCE_TOLERANCE:.0%} of the mean {mean:.1f}. Stop and ask (D-017)."
        )


def manifest_fingerprint(dataset: str) -> str | None:
    """Fingerprint of a dataset's files as recorded by the audit (paths relative to its folder)."""
    if not MANIFEST.exists():
        return None
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] == dataset]
    if not rows:
        return None
    return audit.fingerprint({r["file"].split("/", 1)[1]: r["sha256"] for r in rows})


def files_fingerprint(paths: list[Path], folder: Path) -> str:
    hashes = {p.relative_to(folder).as_posix(): audit.sha256_file(p) for p in paths}
    return audit.fingerprint(hashes)


def meta(dataset: str, fingerprint: str | None, **extra) -> dict:
    return {
        "dataset": dataset,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **audit.code_state(),
        "seed": SEED,
        "method": METHODS[dataset],
        "scikit_learn": version("scikit-learn"),
        "data_fingerprint": fingerprint,
        "decision": "D-017",
        **extra,
    }


def write_once(path: Path, obj: dict) -> bool:
    """Write a split file only if it does not exist yet. Returns True if written."""
    if path.exists():
        print(f"    {path.name} already exists: left untouched (splits are never regenerated)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    print(f"    wrote {path.name}")
    return True


def write_all(built: dict[str, dict], folder: Path = SPLITS) -> None:
    """Write every file or none: all are written to temporary files first, then renamed."""
    clash = [name for name in built if (folder / name).exists()]
    if clash:
        raise SplitError(f"{clash} already exist: nothing written (splits are never regenerated)")
    folder.mkdir(parents=True, exist_ok=True)
    tmps: dict[str, Path] = {}
    try:
        for name, obj in built.items():
            tmps[name] = folder / f"{name}.tmp"
            tmps[name].write_text(json.dumps(obj, indent=1), encoding="utf-8")
    except OSError:
        for tmp in tmps.values():  # only our own temporary files
            tmp.unlink(missing_ok=True)
        raise
    for name, tmp in tmps.items():
        os.replace(tmp, folder / name)
        print(f"    wrote {name}")


def injuries_per(ids_list: list[list], athlete: np.ndarray, y: np.ndarray) -> list[int]:
    return [int(y[np.isin(athlete, ids)].sum()) for ids in ids_list]


# ---------------------------------------------------------------- Lövdal


def lovdal_split(day: loaders.RunnerData, week: loaders.RunnerData) -> dict:
    """Lockbox = the 10 newest runners (the paper's test set); 5 balanced development groups.

    The groups are built ONCE from the day file and used unchanged for the week file and fusion.
    """
    ids = ints(np.unique(day.athlete_id))
    lockbox, dev = ids[-N_LOCKBOX:], ids[:-N_LOCKBOX]
    in_dev = np.isin(day.athlete_id, dev)
    tests = stratified_groups(day.y[in_dev], day.athlete_id[in_dev], 5)
    folds = {}
    for k, test in enumerate(tests):
        rest = in_dev & ~np.isin(day.athlete_id, test)
        val = stratified_groups(day.y[rest], day.athlete_id[rest], 5)[0]
        train = ints(set(dev) - set(test) - set(val))
        folds[str(k)] = {"train_ids": train, "val_ids": val, "test_ids": test}
    i_day, _ = loaders.shared_days(day, week)
    views = {
        "day": (day.athlete_id, day.y),
        "week": (week.athlete_id, week.y),
        "shared_days": (day.athlete_id[i_day], day.y[i_day]),
    }
    vals = [f["val_ids"] for f in folds.values()]
    balance = {}
    for name, (athlete, y) in views.items():
        test_inj = injuries_per(tests, athlete, y)
        check_balance(test_inj, f"Lövdal {name}")
        in_lock = np.isin(athlete, lockbox)
        balance[name] = {
            "test_group_injuries": test_inj,
            "val_group_injuries": injuries_per(vals, athlete, y),
            "lockbox_events": int(in_lock.sum()),
            "lockbox_injuries": int(y[in_lock].sum()),
        }
    return {
        "meta": meta(
            "lovdal",
            manifest_fingerprint("lovdal"),
            applies_to=["day file", "week file", "fusion (shared days)"],
            balance_rule=f"test-group injuries within ±{BALANCE_TOLERANCE:.0%} of the mean",
        ),
        "lockbox_ids": lockbox,
        "folds": folds,
        "final": {
            "train_ids": ints(set(dev) - set(tests[0])),
            "val_ids": tests[0],
            "test_ids": lockbox,
        },
        "paper_split": {
            "train_ids": dev,
            "test_ids": lockbox,
            "note": "only for rebuilding the Lövdal et al. 2021 method",
        },
        "balance": balance,
    }


# ---------------------------------------------------------------- sensor datasets


def rotating_folds(subjects: list[str], n_groups: int, n_val: int,
                   always_train: tuple[str, ...] = ()) -> dict:
    """Seed-0 permutation into test groups; fold k validates on n_val people of test group k+1.

    Taking the validation people from the next fold's test group gives every fold different
    validation people (D-020). The test groups are the same as in the first version.
    """
    perm = [str(s) for s in np.random.default_rng(SEED).permutation(subjects)]
    tests = [sorted(perm[i::n_groups]) for i in range(n_groups)]
    folds = {}
    for k, test in enumerate(tests):
        nxt = tests[(k + 1) % n_groups]
        val = sorted([s for s in perm if s in nxt][:n_val])  # permutation order = seeded pick
        remaining = [s for s in subjects if s not in test]
        train = sorted(set(remaining) - set(val)) + sorted(always_train)
        folds[str(k)] = {"train_ids": sorted(train), "val_ids": val, "test_ids": test}
    return folds


def pamap2_split(folder: Path = RAW / "pamap2") -> dict:
    def ids(pattern: str) -> list[str]:
        return sorted({p.stem.removeprefix("subject") for p in folder.rglob(pattern)})

    subjects, protocol = ids("subject*.dat"), ids("Protocol/subject*.dat")
    keep = ("109",)  # ~85 s of protocol data, 1 activity: always training (D-017)
    folds = rotating_folds([s for s in protocol if s not in keep], 4, 1, keep)
    return {"meta": meta("pamap2", manifest_fingerprint("pamap2"), subjects=subjects),
            "folds": folds}


def wesad_split(folder: Path = RAW / "wesad") -> dict:
    subjects = sorted({p.stem for p in folder.rglob("S*.pkl")}, key=lambda s: int(s[1:]))
    return {"meta": meta("wesad", manifest_fingerprint("wesad"), subjects=subjects),
            "folds": rotating_folds(subjects, 5, 2)}


def sportdb_split(folder: Path = RAW / "sportdb2") -> dict:
    mats = sorted(folder.rglob("Data.mat"))
    root = Path(os.path.commonpath([m.parent for m in mats]))
    recordings, keys = [], []
    for m in mats:
        parts = m.parent.relative_to(root).parts
        recordings.append("/".join(parts))
        # Recordings without a subject folder (skating) may be one person: keep them together.
        has_subject = len(parts) >= 3
        keys.append("/".join(parts[:-1]) if has_subject else f"{parts[0]}/<no subject folder>")
    dummy = np.zeros(len(recordings))
    folds = {}
    for k, (_, te) in enumerate(GroupKFold(n_splits=5).split(dummy, groups=keys)):
        folds[str(k)] = {"test_ids": ints({keys[i] for i in te}),
                         "test_recordings": ints(recordings[i] for i in te)}
    return {"meta": meta("sportdb2", manifest_fingerprint("sportdb2")), "folds": folds}


# ---------------------------------------------------------------- exploratory datasets


def leave_one_out(people: list[str]) -> dict:
    return {str(i): {"train_ids": [q for q in people if q != p], "test_ids": [p]}
            for i, p in enumerate(people)}


def soccermon_split(folder: Path = RAW / "soccermon") -> dict:
    load = folder / "subjective" / "training-load" / "daily_load.csv"
    injury = folder / "subjective" / "injury" / "injury.csv"
    players = sorted(c for c in pd.read_csv(load, nrows=0).columns if c != "Date")
    fp = files_fingerprint([load, injury], folder)
    return {"meta": meta("soccermon", fp, players=len(players)), "folds": leave_one_out(players)}


def pmdata_split(folder: Path = RAW / "pmdata") -> dict:
    people = sorted(p.name for p in folder.iterdir() if p.is_dir() and p.name.startswith("p"))
    injury_files = {p: folder / p / "pmsys" / "injury.csv" for p in people}
    usable = [p for p in people if injury_files[p].exists()]
    excluded = {p: "injury file missing" for p in people if p not in usable}
    fp = files_fingerprint([injury_files[p] for p in usable], folder)
    return {"meta": meta("pmdata", fp, excluded_from_injury_tasks=excluded,
                         sensitivity_check="report every result with and without p12"),
            "folds": leave_one_out(usable)}


def sirp_split(folder: Path = RAW / "sirp600") -> dict:
    path = only_file(folder, "*.xlsx")
    y = readers.read_sirp_xlsx(path)["Injury_Risk"].to_numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train, test = next(sss.split(np.zeros(len(y)), y))
    return {"meta": meta("sirp600", manifest_fingerprint("sirp600")),
            "train_rows": ints(train), "test_rows": ints(test)}


# ---------------------------------------------------------------- main


def build_all() -> dict[str, dict]:
    """Build every split in memory (running the balance checks) before anything is written."""
    day, week = loaders.load_lovdal("day"), loaders.load_lovdal("week")
    built = [lovdal_split(day, week), pamap2_split(), wesad_split(), sportdb_split(),
             soccermon_split(), pmdata_split(), sirp_split()]
    return dict(zip(FILES, built, strict=True))


def create() -> None:
    existing = [name for name in FILES if (SPLITS / name).exists()]
    if len(existing) == len(FILES):
        print("all split files already exist: nothing to do (splits are never regenerated)")
        return
    if existing:
        raise SplitError(f"only some split files exist {existing}: refusing to mix files "
                         "from different runs. Ask before changing anything.")
    write_all(build_all())


def remake(names: list[str], decision: str | None) -> None:
    """One-time re-make of a split file, allowed only before sealing and only for D-020.

    The decision must already be logged in docs/DECISIONS.md. Test groups must come out
    identical; the old file is moved to data/splits/superseded/ (kept in git, never deleted).
    """
    builders = {"pamap2_folds.json": pamap2_split, "wesad_folds.json": wesad_split}
    if SEAL.exists():
        raise SplitError("data/splits is sealed (SHA256SUMS exists): nothing can be re-made")
    if not decision or f"- {decision} |" not in DECISIONS.read_text(encoding="utf-8"):
        raise SplitError(f"decision {decision!r} is not logged in docs/DECISIONS.md")
    names = list(dict.fromkeys(names))  # a repeated name would overwrite the kept v1 file
    bad = [name for name in names if name not in builders]
    if bad:
        raise SplitError(f"{bad} may not be re-made (only {sorted(builders)}, D-020)")
    built = {}
    for name in names:
        path, kept = SPLITS / name, SUPERSEDED / f"{Path(name).stem}.v1.json"
        if kept.exists():
            raise SplitError(f"{name} was already re-made once ({kept.name} exists)")
        old = json.loads(path.read_text(encoding="utf-8"))
        new = builders[name]()
        old_tests = [f["test_ids"] for f in old["folds"].values()]
        if old_tests != [f["test_ids"] for f in new["folds"].values()]:
            raise SplitError(f"{name}: the test groups would change; refused (D-020)")
        new["meta"]["decision"] = f"D-017, {decision}"
        new["meta"]["supersedes"] = {
            "file": kept.relative_to(ROOT).as_posix(),
            "sha256": audit.sha256_file(path),
            "created_utc": old["meta"]["created_utc"],
            "git_commit": old["meta"]["git_commit"],
            "reason": "validation people repeated across folds; test groups unchanged",
        }
        built[name] = new
    SUPERSEDED.mkdir(parents=True, exist_ok=True)
    for name, new in built.items():
        os.replace(SPLITS / name, SUPERSEDED / f"{Path(name).stem}.v1.json")  # moved, kept
        write_once(SPLITS / name, new)


def seal() -> None:
    """Pin the sha256 of every split file in data/splits/SHA256SUMS (written once)."""
    if SEAL.exists():
        raise SplitError("already sealed: SHA256SUMS is never rewritten")
    missing = [name for name in FILES if not (SPLITS / name).exists()]
    if missing:
        raise SplitError(f"cannot seal: {missing} missing")
    files = sorted(SPLITS.rglob("*.json"))
    lines = [f"{audit.sha256_file(p)}  {p.relative_to(SPLITS).as_posix()}" for p in files]
    SEAL.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    sealed {len(files)} files in {SEAL.name}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="person-level splits (written once)")
    parser.add_argument("--remake", nargs="+", metavar="FILE",
                        help="one-time re-make before sealing (D-020 files only)")
    parser.add_argument("--decision", help="DECISIONS.md entry that approved the re-make")
    parser.add_argument("--seal", action="store_true", help="write SHA256SUMS")
    args = parser.parse_args(argv)
    try:
        if args.remake:
            remake(args.remake, args.decision)
        elif args.seal:
            seal()
        else:
            create()
    except (BalanceError, SplitError) as e:
        print(f"STOP: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
