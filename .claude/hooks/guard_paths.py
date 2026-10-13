"""Claude Code PreToolUse hook: block edits to raw data, splits, metrics files and secrets."""

import json
import os
import re
import sys

PROTECTED = [
    r"(^|/)data/raw/",
    r"(^|/)data/splits/",
    r"(^|/)experiments/.*metrics\.json$",
    r"(^|/)\.env(\..*)?$",
    r"(^|/)\.git/",
]


def is_protected(path: str, root: str) -> bool:
    abs_path = os.path.abspath(path)
    try:
        rel = os.path.relpath(abs_path, root)
    except ValueError:
        # Different drive on Windows: outside the project, so not ours to guard.
        return False
    rel = rel.replace(os.sep, "/")
    return any(re.search(pat, rel) for pat in PROTECTED)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0
    tool_input = payload.get("tool_input") or {}
    path = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not path:
        return 0
    root = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
    if is_protected(path, root):
        sys.stderr.write(
            f"BLOCKED: {path} is protected (raw data, splits, metrics or secrets). "
            "Generate it with code, or ask the human to change it.\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
