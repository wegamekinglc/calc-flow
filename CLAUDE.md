# CLAUDE.md

Follow [AGENTS.md](AGENTS.md) for repository commands, architecture, coding
style, verification, release rules, and git conventions. It is the maintained
source of truth for agents working on Calc Flow.

Published documentation lives under [docs/](docs/), starting with
[the introduction](docs/introduction.md). Keep current behavior in those docs
and historical changes in [CHANGELOG.md](CHANGELOG.md).

The specialist team is defined in [`.codex/agents/`](.codex/agents/README.md).
Use its workflow for implementation and independent review. Route changes to
user-visible behavior or APIs through `cf-doc-writer` to reconcile published
docs. `.claude/agents/` is a compatibility mirror; synchronize team changes
from `.codex/agents/` to `.claude/agents/`.
