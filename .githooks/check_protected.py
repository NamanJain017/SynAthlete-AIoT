"""Git pre-commit guard: refuse commits that touch protected files, whichever tool makes them.

Claude Code has its own guard (.claude/hooks/guard_paths.py); this one also covers builder tools
(Antigravity, Codex, Copilot) and people. Owner override: set ALLOW_PROTECTED=1 for one commit
(Claude Code / Naman only; the change is still visible in the review).
Standard library only, so any Python 3 can run it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

PROTECTED = [
    (r"^data/(raw|splits|processed)/", "private data and the sealed splits"),
    (r"(^|/)metrics\.json$", "results are written only by the experiment code"),
    (r"(^|/)\.env(\..*)?$", "secrets"),
    (r"^tests/", "tests are the answer checker: builders never edit them"),
    (r"^docs/handoff/", "task cards are written by Claude Code"),
    (r"^docs/DECISIONS\.md$", "decisions are logged only with Naman's approval"),
    (r"^configs/", "settings are fixed before results exist"),
    (r"^(\.githooks|\.github|\.claude)/", "the guards themselves"),
    (r"^(AGENTS|CLAUDE)\.md$", "the rulebook"),
]


def staged_files() -> list[str]:
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRD"],
                         capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def blocked(files: list[str]) -> list[tuple[str, str]]:
    return [(f, why) for f in files for pattern, why in PROTECTED if re.search(pattern, f)]


def main() -> int:
    hits = blocked(staged_files())
    if not hits:
        return 0
    if os.environ.get("ALLOW_PROTECTED") == "1":
        print(f"pre-commit: ALLOW_PROTECTED=1, committing {len(hits)} protected file(s).")
        return 0
    print("pre-commit: BLOCKED. This commit touches protected files:")
    for f, why in hits:
        print(f"  {f}  ({why})")
    print("Builder tools: undo these changes (git restore --staged <file>) "
          "and follow the task card.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
