"""The locked final-test runners (Lövdal IDs 64-73): opened once, at the evaluation freeze (D-017).

Their arrays are saved apart from the development data (data/processed/lovdal_lockbox/), and
this module is the only reader. It refuses unless the repository is at the commit tagged
`eval-freeze` with no uncommitted changes to tracked files, and it appends every opening to
experiments/lockbox_access.log. The raw CSV files still contain these runners, so this is a
guard rail plus an audit trail, not encryption.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.data import audit, loaders

FREEZE_TAG = "eval-freeze"
LOG = loaders.ROOT / "experiments" / "lockbox_access.log"


class LockboxSealed(RuntimeError):
    """The lockbox may not be opened yet."""


def freeze_state() -> dict:
    # Untracked outputs and the access log itself do not count as changes to the frozen code.
    status = audit._git("status", "--porcelain", "--untracked-files=no", "--", ".",
                        ":(exclude)experiments/lockbox_access.log")
    return {
        "tag_commit": audit._git("rev-list", "-n", "1", FREEZE_TAG),
        "git_commit": audit._git("rev-parse", "HEAD"),
        "git_dirty": status != "",
    }


def open_lockbox(kind: str, reason: str) -> loaders.RunnerData:
    """Locked runners of the day or week file, for the one final evaluation."""
    if not reason.strip():
        raise ValueError("give a reason: it is written to the access log")
    state = freeze_state()
    if state["tag_commit"] in ("", "unavailable"):
        raise LockboxSealed(f"the lockbox opens only after `git tag {FREEZE_TAG}` "
                            "(evaluation freeze, D-017)")
    if state["git_commit"] != state["tag_commit"] or state["git_dirty"]:
        raise LockboxSealed(f"check out the commit tagged {FREEZE_TAG}, "
                            "with no uncommitted changes, before opening the lockbox")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind,
             "commit": state["git_commit"], "reason": reason}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return loaders.load_runner_data(loaders.LOCKBOX_DIR, kind)
