@AGENTS.md

## Claude Code specifics
- Use plan mode for any task touching more than two files or any split, metric or training code.
- Run the eval-auditor subagent (.claude/agents) after changes to splits, metrics or training, and before results go into slides or the paper.
- A hook in .claude/settings.json blocks edits to data/raw, data/splits, metrics.json and .env. If a write is blocked, do not work around it; ask Naman.
- Shell is PowerShell on Windows: use `python`, backslash-safe paths via pathlib, no `make`.
- When compacting, keep: current piece, open decisions, files changed, failing checks.
