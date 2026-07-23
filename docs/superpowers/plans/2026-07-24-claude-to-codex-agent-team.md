# Claude-to-Codex Agent Team Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a complete Codex-native mirror of the preserved Claude guidance,
specialist team, and active artifacts, publish it in a pull request, remediate
CI and review findings, and merge the verified final head.

**Architecture:** Keep `AGENTS.md` as the repository-wide source of truth,
extend `.codex/config.toml` with additive multi-agent configuration, define
each specialist as a standalone `.codex/agents/*.toml` configuration layer,
and keep future team artifacts under `.codex/artifacts/`. Convert the existing
agent bodies faithfully with a small, explicit terminology/path mapping while
leaving `.claude/**` and `CLAUDE.md` byte-for-byte unchanged.

**Tech Stack:** Codex project configuration (TOML), Markdown, Python 3.13
`tomllib`/`unittest`, Git, GitHub CLI, GitHub Actions.

## Global Constraints

- Preserve every tracked file under `.claude/` and preserve `CLAUDE.md`;
  neither may be edited, renamed, or deleted.
- Use the approved design at
  `docs/superpowers/specs/2026-07-24-claude-to-codex-agent-team-design.md`.
- Follow `AGENTS.md`: branch from `main` as
  `feature/codex-agent-team-migration`, use imperative commit summaries under
  72 characters, and use a category-prefixed PR title under 70 characters.
- Keep prose guidance out of `.codex/rules/`; that directory remains reserved
  for command approval rules.
- Do not pin a model, reasoning effort, concurrency limit, credential, MCP
  endpoint, or user-specific absolute path in custom-agent files.
- Do not mutate remote GitHub state until local scope and verification are
  confirmed.
- The active workspace mounts `.codex` and `.git` read-only. Execute the plan
  in the existing writable staging clone, or recreate one under `/tmp`, and
  report the distinction accurately.
- The user explicitly authorized pushing, opening the PR, fixing CI and review
  issues, and merging after the exact final head is green.

## File Map

**Create:**

- `.codex/agents/README.md` — Codex-native roster, workflow, invocation, and
  hand-off guide.
- `.codex/agents/cf-api-designer.toml` — developer-facing API specialist.
- `.codex/agents/cf-critic.toml` — pre-implementation adversarial reviewer.
- `.codex/agents/cf-doc-writer.toml` — normative documentation specialist.
- `.codex/agents/cf-implementer.toml` — TDD implementation specialist.
- `.codex/agents/cf-orchestrator.toml` — dispatch-only coordinator.
- `.codex/agents/cf-performancer.toml` — noise-aware performance specialist.
- `.codex/agents/cf-reviewer.toml` — code and PR review specialist.
- `.codex/agents/cf-simplifier.toml` — behavior-preserving simplification
  specialist.
- `.codex/agents/cf-spec-writer.toml` — requirements specialist.
- `.codex/agents/cf-tester.toml` — cross-surface testing specialist.
- `.codex/artifacts/api-notes/docs-examples.md` — exact mirror of the current
  API note.
- `.codex/artifacts/specs/head-operator.md` — exact mirror of the current
  specification.
- `.codex/guidance/code-style.md` — exact mirror of the detailed style guide.
- `scripts/test_codex_agents.py` — structural and preservation-oriented
  regression tests.
- `docs/superpowers/specs/2026-07-24-claude-to-codex-agent-team-design.md` —
  approved design.
- `docs/superpowers/plans/2026-07-24-claude-to-codex-agent-team.md` — this
  implementation plan.

**Modify:**

- `.codex/config.toml` — add `[agents] enabled = true`.
- `AGENTS.md` — document the Codex specialist team and artifact namespace.

**Must remain unchanged:**

- `CLAUDE.md`
- `.claude/**`
- `.codex/rules/default.rules`
- engine, Python package, Studio, schemas, OpenAPI, and generated TypeScript
  sources.

---

### Task 0: Prepare the Writable Feature Branch

**Files:**

- Commit:
  `docs/superpowers/specs/2026-07-24-claude-to-codex-agent-team-design.md`
- Commit:
  `docs/superpowers/plans/2026-07-24-claude-to-codex-agent-team.md`

**Interfaces:**

- Consumes: current `origin/main`, the approved design, and this plan.
- Produces: writable branch `feature/codex-agent-team-migration` with design
  and plan commits before implementation begins.

- [ ] **Step 1: Confirm or recreate the writable staging clone**

Use the existing clone recorded during design approval. If it no longer
exists, create a new clone:

```bash
migration_stage=$(mktemp -d /tmp/calc-flow-codex-agent-team.XXXXXX)
git clone --no-hardlinks \
  /home/wegamekinglc/dev/github/my-claude/workspace/calc-flow \
  "$migration_stage"
```

Expected: the staging clone has a writable `.git` directory and writable
`.codex` directory.

- [ ] **Step 2: Create the feature branch**

Run in the staging clone:

```bash
git switch -c feature/codex-agent-team-migration
```

Expected: the branch starts from current `main` plus the approved design
commit when the existing staging clone is reused.

- [ ] **Step 3: Copy and commit this implementation plan**

Copy the byte-identical workspace plan into
`docs/superpowers/plans/2026-07-24-claude-to-codex-agent-team.md`, then run:

```bash
git add \
  docs/superpowers/specs/2026-07-24-claude-to-codex-agent-team-design.md \
  docs/superpowers/plans/2026-07-24-claude-to-codex-agent-team.md
git diff --cached --check
git commit -m "docs: plan Codex agent-team migration"
```

Expected: the staging branch contains the approved design and plan; if the
design was already committed, the new commit contains only the plan.

---

### Task 1: Add the Failing Codex-Agent Structure Test

**Files:**

- Create: `scripts/test_codex_agents.py`
- Read: `.claude/agents/*.md`
- Read: `.codex/config.toml`
- Read: `AGENTS.md`

**Interfaces:**

- Consumes: the approved ten-agent roster and source/destination mapping.
- Produces: `CodexAgentConfigTests`, executable with
  `python -m unittest scripts.test_codex_agents`.

- [ ] **Step 1: Record the preserved-source baseline**

Run:

```bash
git ls-files -z .claude CLAUDE.md \
  | xargs -0 sha256sum \
  > /tmp/calc-flow-preserved-claude.sha256
```

Expected: the file contains one hash for `CLAUDE.md` and every tracked
`.claude` file.

- [ ] **Step 2: Create the structural test**

Create `scripts/test_codex_agents.py` with:

```python
from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_AGENTS = {
    "cf-api-designer",
    "cf-critic",
    "cf-doc-writer",
    "cf-implementer",
    "cf-orchestrator",
    "cf-performancer",
    "cf-reviewer",
    "cf-simplifier",
    "cf-spec-writer",
    "cf-tester",
}

MIRRORS = {
    ".claude/api-notes/docs-examples.md":
        ".codex/artifacts/api-notes/docs-examples.md",
    ".claude/specs/head-operator.md":
        ".codex/artifacts/specs/head-operator.md",
    ".claude/rules/code-style.md":
        ".codex/guidance/code-style.md",
}

LEGACY_AGENT_MARKERS = (
    ".claude/",
    "EnterWorktree",
    "`Agent`",
    "`SendMessage`",
    "`TaskCreate`",
    "`TaskUpdate`",
    "`TaskList`",
    "`TaskGet`",
    "`Bash`",
    "`Read`",
    "`Write`",
    "`Edit`",
    "`NotebookEdit`",
    "`CronCreate`",
    "`ScheduleWakeup`",
)


class CodexAgentConfigTests(unittest.TestCase):
    def _agent_configs(self) -> dict[str, dict[str, object]]:
        agent_dir = ROOT / ".codex/agents"
        return {
            path.stem: tomllib.loads(path.read_text())
            for path in sorted(agent_dir.glob("*.toml"))
        }

    def test_project_config_enables_agents_without_changing_permissions(
        self,
    ) -> None:
        config = tomllib.loads((ROOT / ".codex/config.toml").read_text())

        self.assertEqual(config["approval_policy"], "on-request")
        self.assertEqual(config["sandbox_mode"], "workspace-write")
        self.assertIs(config["agents"]["enabled"], True)
        self.assertEqual(set(config["agents"]), {"enabled"})

    def test_expected_custom_agent_roster_is_complete(self) -> None:
        configs = self._agent_configs()

        self.assertEqual(set(configs), EXPECTED_AGENTS)
        declared_names = {config["name"] for config in configs.values()}
        self.assertEqual(declared_names, EXPECTED_AGENTS)
        for filename, config in configs.items():
            with self.subTest(agent=filename):
                self.assertEqual(config["name"], filename)
                self.assertIsInstance(config["description"], str)
                self.assertTrue(config["description"].strip())
                self.assertIsInstance(config["developer_instructions"], str)
                self.assertTrue(config["developer_instructions"].strip())
                self.assertNotIn("model", config)
                self.assertNotIn("model_reasoning_effort", config)

    def test_custom_agents_use_codex_paths_and_terminology(self) -> None:
        for filename, config in self._agent_configs().items():
            text = "\n".join(
                (
                    str(config["description"]),
                    str(config["developer_instructions"]),
                )
            )
            with self.subTest(agent=filename):
                for marker in LEGACY_AGENT_MARKERS:
                    self.assertNotIn(marker, text)

    def test_team_readme_lists_every_agent_and_codex_artifact_path(self) -> None:
        readme = (ROOT / ".codex/agents/README.md").read_text()

        for agent in EXPECTED_AGENTS:
            self.assertIn(f"`{agent}`", readme)
        for path in (
            ".codex/artifacts/specs/",
            ".codex/artifacts/api-notes/",
            ".codex/artifacts/critiques/",
            ".codex/guidance/code-style.md",
        ):
            self.assertIn(path, readme)
        self.assertNotIn("EnterWorktree", readme)
        self.assertNotIn(".claude/", readme)

    def test_generic_guidance_and_artifacts_are_exact_mirrors(self) -> None:
        for source, destination in MIRRORS.items():
            with self.subTest(destination=destination):
                self.assertEqual(
                    (ROOT / destination).read_bytes(),
                    (ROOT / source).read_bytes(),
                )

    def test_agents_guide_points_to_codex_team(self) -> None:
        guide = (ROOT / "AGENTS.md").read_text()

        self.assertIn(".codex/agents/README.md", guide)
        self.assertIn(".codex/artifacts/", guide)
        self.assertIn("preserved compatibility", guide)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the focused test and confirm the expected failure**

Run:

```bash
python -m unittest \
  scripts.test_codex_agents.CodexAgentConfigTests.test_expected_custom_agent_roster_is_complete
```

Expected: `FAIL`; the observed set is empty because `.codex/agents/*.toml`
does not exist yet.

---

### Task 2: Add Codex Configuration, Guidance, and Artifact Mirrors

**Files:**

- Modify: `.codex/config.toml`
- Modify: `AGENTS.md`
- Create: `.codex/artifacts/api-notes/docs-examples.md`
- Create: `.codex/artifacts/specs/head-operator.md`
- Create: `.codex/guidance/code-style.md`
- Test: `scripts/test_codex_agents.py`

**Interfaces:**

- Consumes: the current Codex permission values and the three generic Claude
  Markdown sources.
- Produces: enabled Codex subagents, byte-identical Markdown mirrors, and the
  root discovery pointer used by future sessions.

- [ ] **Step 1: Add the project agent switch**

Append this exact section to `.codex/config.toml`:

```toml

[agents]
enabled = true
```

Do not change `approval_policy`, `sandbox_mode`, or
`.codex/rules/default.rules`.

- [ ] **Step 2: Copy the generic Markdown artifacts exactly**

Run:

```bash
mkdir -p \
  .codex/artifacts/api-notes \
  .codex/artifacts/specs \
  .codex/guidance
cp .claude/api-notes/docs-examples.md \
  .codex/artifacts/api-notes/docs-examples.md
cp .claude/specs/head-operator.md \
  .codex/artifacts/specs/head-operator.md
cp .claude/rules/code-style.md \
  .codex/guidance/code-style.md
```

Expected: each destination has the same SHA-256 hash as its source.

- [ ] **Step 3: Add the specialist-team discovery section**

Insert this section in `AGENTS.md` after `## Git conventions` and before
`## Coding style`:

```markdown
## Specialist agents

The Codex-native calc-flow specialist team is defined in
`.codex/agents/`; its roster, workflow, and invocation examples live in
`.codex/agents/README.md`. Team-produced specifications, API notes, and
critiques belong under `.codex/artifacts/` with one shared kebab-case slug per
work item.

The `.claude/` tree is preserved compatibility guidance for Claude users.
Codex agents read and write the Codex paths above and must not rewrite the
preserved compatibility files during ordinary work.
```

- [ ] **Step 4: Run the foundational tests**

Run:

```bash
python -m unittest \
  scripts.test_codex_agents.CodexAgentConfigTests.test_project_config_enables_agents_without_changing_permissions \
  scripts.test_codex_agents.CodexAgentConfigTests.test_generic_guidance_and_artifacts_are_exact_mirrors \
  scripts.test_codex_agents.CodexAgentConfigTests.test_agents_guide_points_to_codex_team
```

Expected: `Ran 3 tests ... OK`.

---

### Task 3: Convert the Ten Specialist Definitions

**Files:**

- Create: `.codex/agents/cf-api-designer.toml`
- Create: `.codex/agents/cf-critic.toml`
- Create: `.codex/agents/cf-doc-writer.toml`
- Create: `.codex/agents/cf-implementer.toml`
- Create: `.codex/agents/cf-orchestrator.toml`
- Create: `.codex/agents/cf-performancer.toml`
- Create: `.codex/agents/cf-reviewer.toml`
- Create: `.codex/agents/cf-simplifier.toml`
- Create: `.codex/agents/cf-spec-writer.toml`
- Create: `.codex/agents/cf-tester.toml`
- Test: `scripts/test_codex_agents.py`

**Interfaces:**

- Consumes: each matching `.claude/agents/cf-*.md` body after its second
  `---` delimiter.
- Produces: one independently loadable Codex configuration layer per role,
  using `name`, `description`, and `developer_instructions`.

- [ ] **Step 1: Use the exact agent descriptions**

Use these `description` values:

| Agent             | Description                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------- |
| `cf-api-designer` | Design and critique calc-flow's public Rust, Python, REST, error, and example-code surfaces.  |
| `cf-critic`       | Adversarially review calc-flow specifications and API proposals before implementation.        |
| `cf-doc-writer`   | Reconcile calc-flow's normative documentation and changelog with current source behavior.     |
| `cf-implementer`  | Implement approved calc-flow changes test-first across Rust, Python, and Studio surfaces.     |
| `cf-orchestrator` | Route end-to-end work through the calc-flow specialist team without implementing it directly. |
| `cf-performancer` | Measure calc-flow changes with paired, noise-aware benchmarks and coverage analysis.          |
| `cf-reviewer`     | Review calc-flow diffs and pull requests for correctness, safety, tests, and documentation.   |
| `cf-simplifier`   | Find behavior-preserving simplifications in completed calc-flow implementations.              |
| `cf-spec-writer`  | Turn calc-flow requests and issues into explicit, testable requirement specifications.        |
| `cf-tester`       | Add and repair focused tests across calc-flow's Rust, Python, backend, and frontend surfaces. |

For each file, set `name` to the exact agent string in the first column and
`description` to the complete sentence in the second column. Follow those two
fields with `developer_instructions = """`, the converted body from Steps 2
and 3, and the closing `"""`. Do not add any other top-level key.

- [ ] **Step 2: Apply the common body conversion exactly**

For every role except `cf-orchestrator`, copy the complete Markdown body after
the source file's second `---` delimiter into `developer_instructions`, then
make these exact semantic replacements:

| Claude text/path                         | Codex text/path                                      |
| ---------------------------------------- | ---------------------------------------------------- |
| `.claude/rules/code-style.md`            | `.codex/guidance/code-style.md`                      |
| `.claude/specs/`                         | `.codex/artifacts/specs/`                            |
| `.claude/api-notes/`                     | `.codex/artifacts/api-notes/`                        |
| `.claude/critiques/`                     | `.codex/artifacts/critiques/`                        |
| `.claude/worktrees/`                     | `.worktrees/`                                        |
| `EnterWorktree`                          | `superpowers:using-git-worktrees`                    |
| “the `EnterWorktree` tool”               | “the `superpowers:using-git-worktrees` skill”        |
| “via `EnterWorktree`”                    | “via `superpowers:using-git-worktrees`”              |

Review the resulting prose for grammar without shortening the operational
checklists, verification commands, domain constraints, or role boundaries.
Remove YAML-only `model` and `color` metadata by not copying the frontmatter.

- [ ] **Step 3: Convert the orchestrator's tool contract**

Copy the complete post-frontmatter body of
`.claude/agents/cf-orchestrator.md`, apply the common path mapping, and replace
the `HARD RULES — Tool Restrictions` section with:

```markdown
## HARD RULES — Tool Restrictions

Use only Codex subagent coordination and task-tracking capabilities:
`spawn_agent`, `send_message`, `followup_task`, `wait_agent`, `list_agents`,
`interrupt_agent`, and `update_plan`, when those capabilities are available in
the current client.

Do not run shell commands, read or edit repository files, browse the web,
call GitHub, run tests, or create implementation artifacts. If the current
client does not expose subagent coordination, report that limitation to the
parent instead of implementing the work yourself.

Before every action, ask: “Am I delegating or tracking work, or am I doing a
specialist's work?” Stop if the action belongs to a specialist.
```

Also make these exact terminology changes throughout the remaining body:

- `Agent spawned` becomes `Subagent spawned`.
- “agent tool” becomes “subagent coordination capability”.
- `SendMessage` becomes `send_message` or `followup_task`, matching whether
  the target agent is already running or idle.
- Claude task-tool references become `update_plan`.
- Any remaining backticked Claude tool name is replaced by plain capability
  language.

- [ ] **Step 4: Validate the TOML files**

Run:

```bash
python -c 'import pathlib, tomllib; [
    tomllib.loads(path.read_text())
    for path in pathlib.Path(".codex/agents").glob("*.toml")
]'
```

Expected: exit code `0` with no output.

- [ ] **Step 5: Run the roster and terminology tests**

Run:

```bash
python -m unittest \
  scripts.test_codex_agents.CodexAgentConfigTests.test_expected_custom_agent_roster_is_complete \
  scripts.test_codex_agents.CodexAgentConfigTests.test_custom_agents_use_codex_paths_and_terminology
```

Expected: `Ran 2 tests ... OK`.

---

### Task 4: Add the Codex Team Guide

**Files:**

- Create: `.codex/agents/README.md`
- Reference: `.claude/agents/README.md`
- Test: `scripts/test_codex_agents.py`

**Interfaces:**

- Consumes: the existing roster, route, working agreements, and the Codex
  agent/artifact paths introduced in Tasks 2 and 3.
- Produces: the human- and agent-readable entry point linked from `AGENTS.md`.

- [ ] **Step 1: Write the team guide**

Adapt the complete structure of `.claude/agents/README.md` with these required
sections and content:

````markdown
# Calc Flow Codex Agent Team

The Codex-native calc-flow team coordinates specialized work across the
Rust-native engine, PyO3/Python adapters, FastAPI backend, and React Studio.
The orchestrator routes work through the specialists; each specialist keeps a
bounded responsibility and returns a concise, verified hand-off.

## Team Roster

| Role         | Agent             | Reads                                    | Writes                                        |
| ------------ | ----------------- | ---------------------------------------- | --------------------------------------------- |
| Orchestrator | `cf-orchestrator` | request and specialist reports           | task plan and consolidated hand-offs          |
| Spec writer  | `cf-spec-writer`  | request, issues, architecture, guidance  | `.codex/artifacts/specs/<slug>.md`            |
| API designer | `cf-api-designer` | spec, exports, stubs, OpenAPI, examples  | `.codex/artifacts/api-notes/<slug>.md`        |
| Critic       | `cf-critic`       | spec and API note                        | `.codex/artifacts/critiques/<slug>.md`        |
| Implementer  | `cf-implementer`  | approved artifacts and relevant source   | source and focused tests in a worktree        |
| Tester       | `cf-tester`       | behavior, source, and existing tests     | tests for the affected surfaces               |
| Reviewer     | `cf-reviewer`     | diff, requirements, tests, documentation | review report and requested remediations      |
| Performancer | `cf-performancer` | finished change, benchmarks, baselines   | performance and coverage advisory             |
| Simplifier   | `cf-simplifier`   | finished change and adjacent modules     | simplification report or approved edits       |
| Doc writer   | `cf-doc-writer`   | current source, guidance, normative docs | `docs/` and qualifying `CHANGELOG.md` entries |

## Workflow

```text
request -> spec writer -> API designer when public -> critic
        -> implementer and tester -> reviewer -> doc writer

out of band and on demand: performancer, simplifier
```

The orchestrator may shorten the route for narrow work, but it does not invent
missing requirements or bypass a requested review. A `Block` verdict returns
to the upstream author. Reviewer findings return to the implementer and
tester, followed by another review of the updated head.

## Artifact Layout

| Path                            | Owner             | Purpose                                        |
| ------------------------------- | ----------------- | ---------------------------------------------- |
| `.codex/artifacts/specs/`       | `cf-spec-writer`  | explicit, testable requirements                |
| `.codex/artifacts/api-notes/`   | `cf-api-designer` | developer-facing API decisions                 |
| `.codex/artifacts/critiques/`   | `cf-critic`       | adversarial pre-implementation review          |
| `.codex/guidance/code-style.md` | repository        | detailed Python, web, test, and Markdown rules |

One kebab-case slug follows a work item across its spec, API note, and
critique.

## How to Invoke the Team

- End to end: “Use `cf-orchestrator` to handle issue #12 through review.”
- Direct specialist: “Use `cf-spec-writer` to specify the requested head
  operator.”
- Adversarial review: “Use `cf-critic` on
  `.codex/artifacts/specs/head-operator.md`.”
- Out-of-band sweep: “Use `cf-performancer` to check this branch for
  regressions” or “Use `cf-simplifier` to review the finished diff.”

## Conventions Each Agent Honors

- `AGENTS.md` is authoritative for commands, architecture, tests, releases,
  and Git conventions.
- `.codex/guidance/code-style.md` supplies detailed functional,
  input-immutability, testing, and Markdown guidance.
- `docs/introduction.md` supplies the domain vocabulary and execution model.
- Agents preserve caller-owned inputs and use the exact verification relevant
  to the changed surfaces.

## Team Working Agreements

- File-changing specialists use `superpowers:using-git-worktrees` when
  available and otherwise follow the repository's safe isolated-worktree
  practice.
- Behavior changes proceed red, green, refactor: observe the focused test fail
  for the expected reason before implementation.
- Planning agents write only their assigned artifact. One agent owns an
  artifact at a time.
- No agent pushes, edits a PR, resolves review threads, merges, or performs
  another remote mutation without explicit user authority.

## Hand-off Etiquette

- Give every specialist a self-contained prompt with paths, decisions,
  acceptance criteria, and expected output.
- Verify an artifact or change before advancing to the next role.
- Return concise evidence: changed paths, commands run, observed results,
  unresolved questions, and the recommended next role.
- Do not fan out multiple writers onto the same file or artifact.
````

Replace every legacy artifact path and Claude tool reference. Do not include a
color column because Codex custom-agent TOML has no color field. Keep every
Markdown table aligned according to the repository rule.

- [ ] **Step 2: Run the team-guide test**

Run:

```bash
python -m unittest \
  scripts.test_codex_agents.CodexAgentConfigTests.test_team_readme_lists_every_agent_and_codex_artifact_path
```

Expected: `Ran 1 test ... OK`.

- [ ] **Step 3: Run the complete focused suite**

Run:

```bash
python -m unittest scripts.test_codex_agents
```

Expected: all six tests pass.

---

### Task 5: Verify Preservation, Formatting, and the Full Repository Matrix

**Files:**

- Verify: all intended files in the File Map.
- Verify unchanged: `.claude/**`, `CLAUDE.md`,
  `.codex/rules/default.rules`, generated contracts.

**Interfaces:**

- Consumes: the complete migration from Tasks 1–4.
- Produces: local evidence strong enough to publish and diagnose CI failures.

- [ ] **Step 1: Prove preserved files are byte-identical**

Run:

```bash
sha256sum --check /tmp/calc-flow-preserved-claude.sha256
git diff --exit-code -- .claude CLAUDE.md
```

Expected: every checksum reports `OK`; `git diff` exits `0`.

- [ ] **Step 2: Check the intended diff and generated contracts**

Run:

```bash
git status --short
git diff --check
git diff --exit-code -- \
  schemas/project-v2.schema.json \
  web-ui/openapi.json \
  web-ui/src/api/schema.d.ts
```

Expected: only files in the File Map are changed; no whitespace errors; no
generated-contract diff.

- [ ] **Step 3: Run focused Python quality checks**

Run:

```bash
UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv run ruff check scripts/test_codex_agents.py
UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv run ruff format --check scripts/test_codex_agents.py
python -m unittest scripts.test_codex_agents
python -m unittest scripts.test_inspect_wheel scripts.test_release_config
```

Expected: all commands pass.

- [ ] **Step 4: Run the Rust command group**

Run with repository-local build outputs:

```bash
export CARGO_TARGET_DIR="$PWD/target/codex-agent-team"
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo llvm-cov --workspace --all-features --fail-under-lines 90
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177
cargo deny --locked check
```

Expected: every command exits `0`; coverage is at least 90%.

- [ ] **Step 5: Run the Python binding and adapter command group**

Run:

```bash
export UV_CACHE_DIR="$PWD/target/uv-cache"
export CARGO_TARGET_DIR="$PWD/target/codex-agent-team"
uv sync --extra dev
uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests -q
uv run ruff check .
uv run ruff format --check .
find python/calc_flow -maxdepth 1 -type f -name '_native*.so' -print -delete
test -z "$(find python/calc_flow -maxdepth 1 -type f -name '_native*.so' -print -quit)"
```

Expected: all commands pass and no generated
`python/calc_flow/_native*.so` remains.

- [ ] **Step 6: Run the Studio command groups**

Run:

```bash
cd web-ui/backend
UV_CACHE_DIR="$OLDPWD/target/uv-cache" \
  uv run --project . --extra dev pytest --cov=calc_flow_studio
cd ..
npm ci
npm run sync:api
npm run build
npm test
npm run test:e2e
npm audit --omit=dev
cd ..
```

Expected: backend coverage meets its 85% floor; build, unit tests, browser
tests, audit, and API sync pass; generated contracts remain unchanged.

- [ ] **Step 7: Re-run final diff guards**

Run:

```bash
git diff --exit-code -- \
  schemas/project-v2.schema.json \
  web-ui/openapi.json \
  web-ui/src/api/schema.d.ts
git diff --check
git status --short
```

Expected: only the intended migration files remain changed.

---

### Task 6: Commit, Push, and Open the Pull Request

**Files:**

- Stage only the files in the File Map.
- Do not stage `target/`, `.venv/`, `node_modules/`, generated native modules,
  caches, logs, or runtime state.

**Interfaces:**

- Consumes: a clean, verified intended diff.
- Produces: remote branch `feature/codex-agent-team-migration` and a PR
  targeting `main`.

- [ ] **Step 1: Stage the migration scope explicitly**

Run:

```bash
git add \
  AGENTS.md \
  .codex/config.toml \
  .codex/agents \
  .codex/artifacts \
  .codex/guidance \
  scripts/test_codex_agents.py
git diff --cached --name-status
git diff --cached --check
```

Expected: the staged list contains only the migration files; it contains no
`.claude` or `CLAUDE.md` entry.

- [ ] **Step 2: Commit the migration**

Run:

```bash
git commit -m "chore: add Codex-native agent team"
```

Expected: the commit succeeds with the verified migration scope.

- [ ] **Step 3: Confirm GitHub prerequisites and push**

Run:

```bash
gh --version
gh auth status
GIT_SSH_COMMAND='ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=accept-new' \
  git push -u origin feature/codex-agent-team-migration
```

Expected: `gh` is authenticated and the remote branch points to the local
head.

- [ ] **Step 4: Open the draft PR**

Use the connected GitHub app when available, falling back to `gh pr create`.
Use:

- Base: `main`
- Head: `feature/codex-agent-team-migration`
- Title: `chore: migrate Claude agent team to Codex`
- Body:

```markdown
## Summary

- add Codex-native TOML definitions for the full calc-flow specialist team
- mirror active Claude guidance and artifacts while preserving the originals
- enable project subagents and document the Codex artifact workflow
- add structural tests for roster, paths, configuration, and preservation

## Test plan

- `python -m unittest scripts.test_codex_agents`
- `python -m unittest scripts.test_inspect_wheel scripts.test_release_config`
- full Rust, Python, Studio backend, and Studio frontend command groups from
  `AGENTS.md`
- `git diff --exit-code -- schemas/project-v2.schema.json web-ui/openapi.json web-ui/src/api/schema.d.ts`
- `git diff --check`
```

Expected: a draft PR URL targeting `main`.

- [ ] **Step 5: Mark the PR ready after local checks are recorded**

Run:

```bash
gh pr ready
```

Expected: the PR is ready for review before merge evaluation.

---

### Task 7: Remediate CI and Review Issues to an Exact Green Head

**Files:**

- Modify only files required by verified CI failures or actionable review
  findings.
- Preserve the scope and invariants from the approved design.

**Interfaces:**

- Consumes: PR check runs, logs, review threads, and the exact PR head SHA.
- Produces: a final pushed head with all required checks green and no
  actionable unresolved review finding.

- [ ] **Step 1: Inspect the PR head, checks, and mergeability**

Run:

```bash
gh pr view --json \
  number,url,state,isDraft,mergeable,headRefName,headRefOid,statusCheckRollup
```

Expected: the head branch and SHA match the pushed local branch.

- [ ] **Step 2: Diagnose and fix any failing GitHub Actions check**

If any required check fails, invoke `github:gh-fix-ci`, inspect the failing
Actions logs, reproduce the failure locally, add or refine a focused test when
behavior changes, apply the narrowest technically correct fix, rerun relevant
local and full-scope checks, commit, and push.

Expected: each fix is tied to concrete log evidence; tests are not weakened.

- [ ] **Step 3: Inspect and address actionable review threads**

If review comments or requested changes exist, invoke
`github:gh-address-comments`, fetch thread-level state, verify each suggestion
against source and tests, apply only technically correct changes, rerun
verification, commit, push, and resolve or reply to the addressed threads.

Expected: no actionable unresolved review thread remains on the final head.

- [ ] **Step 4: Repeat until the exact current head is green**

After every push, capture the new local and remote head SHA and rerun:

```bash
git rev-parse HEAD
gh pr view --json headRefOid,mergeable,statusCheckRollup
```

Expected: local `HEAD` equals `headRefOid`; every required current-head check
is successful; `mergeable` is not `CONFLICTING`.

---

### Task 8: Merge and Verify the Final State

**Files:**

- No additional file changes unless a final required check exposes a defect.

**Interfaces:**

- Consumes: the exact green PR head and explicit user authorization to merge.
- Produces: a merged PR and `origin/main` containing the final migration.

- [ ] **Step 1: Perform the pre-merge completion audit**

Verify each approved acceptance criterion against current files, focused test
output, preservation hashes, generated-contract diffs, PR reviews, and the
current-head check rollup. Do not use stale checks from an earlier SHA.

Expected: every criterion has direct evidence and no required work remains.

- [ ] **Step 2: Merge using the repository's established merge-commit style**

Run:

```bash
gh pr merge --merge --delete-branch
```

If branch protection queues or delays the merge, monitor until GitHub reports
the terminal merged state.

- [ ] **Step 3: Verify the merged PR and main**

Run:

```bash
gh pr view --json number,url,state,mergedAt,mergeCommit,headRefOid
GIT_SSH_COMMAND='ssh -F /dev/null -o BatchMode=yes' \
  git fetch origin main
git merge-base --is-ancestor HEAD origin/main
git rev-parse origin/main
```

Expected: PR state is `MERGED`, `mergedAt` and `mergeCommit` are present, and
the final feature head is an ancestor of `origin/main`.

- [ ] **Step 4: Report the verified hand-off**

Report the PR URL, final feature head, merge commit, merged time, local
verification, final CI count/state, review-thread state, preservation proof,
and any intentionally retained local staging path.
