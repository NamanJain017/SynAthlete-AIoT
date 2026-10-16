"""The locked final-test runners (Lövdal IDs 64-73): opened once, at the evaluation freeze (D-017).

Their arrays are saved apart from the development data (data/processed/lovdal_lockbox/), and
this module is the only reader. It refuses unless:
- `eval-freeze` is an annotated tag (it records who made it and when) that is pushed to GitHub;
- the repository is at that commit, with no changes to tracked files and no untracked files in
  src/ or configs/ (code that is not part of the frozen commit);
- the lockbox was never opened at another commit (a second freeze would let results steer it);
- the locked arrays come from the same raw-data version as the split file.
Every opening is appended to experiments/lockbox_access.log; commit that log after the final run.
The raw CSV files still contain these runners, so this is a guard rail plus an audit trail,
not encryption.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone

from src.data import audit, loaders

FREEZE_TAG = "eval-freeze"
LOG = loaders.ROOT / "experiments" / "lockbox_access.log"
GUARDED_UNTRACKED = ("src/", "configs/")  # untracked files here could change the frozen run


class LockboxSealed(RuntimeError):
    """The lockbox may not be opened (yet)."""


def is_dirty(porcelain: str) -> bool:
    """Tracked changes anywhere, or untracked files under src/ or configs/, count as changes."""
    if porcelain == "unavailable":
        return True  # cannot tell: fail closed
    for line in porcelain.splitlines():
        if not line.startswith("??"):
            return True
        path = line[3:].strip('"')
        if path.startswith(GUARDED_UNTRACKED) or path.endswith(".py"):  # .py: could be imported
            return True
    return False


def remote_tag_object() -> str:
    """The tag object hash on GitHub ('unavailable' if it cannot be read)."""
    try:
        out = subprocess.run(
            ["git", "ls-remote", "--tags", "origin", f"refs/tags/{FREEZE_TAG}"],
            cwd=loaders.ROOT, capture_output=True, text=True, check=True, timeout=60,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return out[0] if out else "unavailable"


def freeze_state() -> dict:
    status = audit._git("status", "--porcelain", "--", ".",
                        ":(exclude)experiments/lockbox_access.log")
    return {
        "tag_type": audit._git("cat-file", "-t", FREEZE_TAG),
        "tag_object": audit._git("rev-parse", FREEZE_TAG),
        "tag_commit": audit._git("rev-list", "-n", "1", FREEZE_TAG),
        "remote_tag_object": remote_tag_object(),
        "git_commit": audit._git("rev-parse", "HEAD"),
        "git_dirty": is_dirty(status),
    }


def earlier_commits() -> set[str]:
    if not LOG.exists():
        return set()
    lines = LOG.read_text(encoding="utf-8").splitlines()
    return {json.loads(line)["commit"] for line in lines if line.strip()}


def open_lockbox(kind: str, reason: str) -> loaders.RunnerData:
    """Locked runners of the day or week file, for the one final evaluation."""
    if not reason.strip():
        raise ValueError("give a reason: it is written to the access log")
    state = freeze_state()
    if state["tag_commit"] in ("", "unavailable"):
        raise LockboxSealed(f"the lockbox opens only after the evaluation freeze: "
                            f"git tag -a {FREEZE_TAG} -m '...' (D-017)")
    if state["tag_type"] != "tag":
        raise LockboxSealed(f"{FREEZE_TAG} must be an annotated tag (git tag -a), "
                            "so it records who made it and when")
    if state["remote_tag_object"] != state["tag_object"]:
        raise LockboxSealed(f"push the tag to GitHub first: git push origin {FREEZE_TAG}")
    if state["git_commit"] != state["tag_commit"] or state["git_dirty"]:
        raise LockboxSealed(f"check out the commit tagged {FREEZE_TAG}, with no changes to "
                            "tracked files and no untracked files in src/ or configs/")
    other = earlier_commits() - {state["git_commit"]}
    if other:
        raise LockboxSealed(f"the lockbox was already opened at commit {sorted(other)[0][:7]}: "
                            "it is opened at one frozen commit only (D-017)")
    meta = json.loads((loaders.LOCKBOX_DIR / f"{kind}_meta.json").read_text(encoding="utf-8"))
    if (meta.get("source") or {}).get("fingerprint") != \
            loaders.lovdal_split()["meta"]["data_fingerprint"]:
        raise LockboxSealed("locked arrays are not the data version the splits were made from")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind,
             "commit": state["git_commit"], "tag_object": state["tag_object"], "reason": reason}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return loaders.load_runner_data(loaders.LOCKBOX_DIR, kind)
