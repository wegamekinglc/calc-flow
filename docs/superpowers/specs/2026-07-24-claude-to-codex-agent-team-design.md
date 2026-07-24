# Claude-to-Codex Agent Team Migration Design

**Status:** Implemented and merged in PR #25 (historical design)

## Goal

Create a complete Codex-native mirror of the repository's Claude guidance,
specialist team, and active planning artifacts without deleting or modifying
the existing `CLAUDE.md` or `.claude/` tree.

The result must be directly usable by current Codex clients:

- durable repository guidance remains in `AGENTS.md`;
- project settings remain in `.codex/config.toml`;
- command approval rules remain in `.codex/rules/*.rules`;
- specialist subagents live in `.codex/agents/*.toml`; and
- team-produced planning artifacts live under `.codex/artifacts/`.

## Current State

The repository already has the following Codex surfaces:

- `AGENTS.md`, which is the authoritative repository guide;
- `.codex/config.toml`, which selects project sandbox and approval defaults;
  and
- `.codex/rules/default.rules`, which controls approval for selected commands.

The Claude tree contains:

- ten role definitions and a team guide under `.claude/agents/`;
- detailed style guidance at `.claude/rules/code-style.md`;
- one specification at `.claude/specs/head-operator.md`;
- one API note at `.claude/api-notes/docs-examples.md`; and
- an empty, untracked `.claude/worktrees/` runtime directory.

Current Codex supports project-scoped custom agents as standalone TOML files
under `.codex/agents/`. Each file requires `name`, `description`, and
`developer_instructions`. Model and reasoning settings may be omitted so the
agent inherits the parent session's current configuration.

## Selected Approach

Use an additive, faithful native mirror.

The Claude files remain available to Claude users and as migration evidence.
Codex receives independent TOML agent definitions whose instructions are
adapted to Codex concepts and paths. Generic Markdown artifacts are copied
without changing their content. New Codex agents write only to Codex paths.

This is preferred over thin wrappers that tell Codex to read
`.claude/agents/*.md`: wrappers would retain Claude-specific tools, make the
Codex team dependent on legacy files, and leave future artifacts under the
wrong namespace.

This is also preferred over converting every role into a skill. Skills encode
reusable workflows, while this request requires named specialists that can be
selected for subagent work. Existing workflow skills may still be invoked by
the migrated agents where applicable.

## File Layout

The migration adds the following files:

```text
.codex/
├── agents/
│   ├── README.md
│   ├── cf-api-designer.toml
│   ├── cf-critic.toml
│   ├── cf-doc-writer.toml
│   ├── cf-implementer.toml
│   ├── cf-orchestrator.toml
│   ├── cf-performancer.toml
│   ├── cf-reviewer.toml
│   ├── cf-simplifier.toml
│   ├── cf-spec-writer.toml
│   └── cf-tester.toml
├── artifacts/
│   ├── api-notes/
│   │   └── docs-examples.md
│   └── specs/
│       └── head-operator.md
├── guidance/
│   └── code-style.md
├── rules/
│   └── default.rules
└── config.toml
```

`.codex/artifacts/critiques/` is created on demand by `cf-critic`; Git does
not track an empty directory. The empty `.claude/worktrees/` runtime directory
is not copied. File-changing Codex agents use the repository's worktree
workflow rather than storing worktrees inside the configuration directory.

The root `AGENTS.md` gains a short specialist-team section that points Codex
to `.codex/agents/README.md`, describes the artifact namespace, and states
that `.claude/` is a preserved compatibility tree rather than the write
target for Codex work.

## Custom Agent Conversion

Each `.claude/agents/cf-*.md` file maps one-to-one to
`.codex/agents/cf-*.toml`.

The TOML files use this shape:

```toml
name = "cf-role-name"
description = """
Concise guidance describing when to use this specialist.
"""
developer_instructions = """
The migrated and Codex-adapted role instructions.
"""
```

The conversion follows these rules:

1. Preserve the existing `cf-` agent names and role boundaries.
2. Convert the YAML `description` into a concise TOML description.
3. Put the complete operational role guidance in
   `developer_instructions`.
4. Omit `model: inherit`; omission is Codex's native inheritance behavior.
5. Omit Claude's color metadata because Codex custom-agent TOML has no
   equivalent field.
6. Replace `.claude/specs/`, `.claude/api-notes/`, and
   `.claude/critiques/` write targets with the corresponding directories
   under `.codex/artifacts/`.
7. Replace `.claude/rules/code-style.md` references with
   `.codex/guidance/code-style.md`.
8. Keep `AGENTS.md` as the authoritative source for commands, architecture,
   test layout, release invariants, and Git conventions.
9. Replace Claude-specific `Agent`, `SendMessage`, and `EnterWorktree`
   language with Codex subagent coordination and the available worktree
   workflow.
10. Do not embed fixed model names, reasoning levels, credentials, MCP
    endpoints, or user-specific absolute paths.

## Team Semantics

The migrated roster and phase ownership remain unchanged:

| Role         | Agent             | Primary responsibility                                |
| ------------ | ----------------- | ----------------------------------------------------- |
| Orchestrator | `cf-orchestrator` | route work and coordinate specialist hand-offs        |
| Spec writer  | `cf-spec-writer`  | write testable requirements                           |
| API designer | `cf-api-designer` | design Rust, Python, and Studio developer-facing APIs |
| Critic       | `cf-critic`       | adversarially review specs and API notes              |
| Implementer  | `cf-implementer`  | implement through red, green, and refactor            |
| Tester       | `cf-tester`       | add and repair tests across the affected surfaces     |
| Reviewer     | `cf-reviewer`     | review correctness and run the relevant verification  |
| Performancer | `cf-performancer` | perform noise-aware benchmark analysis                |
| Simplifier   | `cf-simplifier`   | identify or apply behavior-preserving simplifications |
| Doc writer   | `cf-doc-writer`   | reconcile normative docs and `CHANGELOG.md`           |

The default route remains:

```text
request -> spec writer -> API designer when public -> critic
        -> implementer and tester -> reviewer -> doc writer
```

`cf-performancer` and `cf-simplifier` remain optional, out-of-band advisory
roles. The reviewer remains an in-band gate. The orchestrator may shorten the
route for narrow work, but it must not invent missing requirements or bypass a
requested review.

The team guide includes invocation examples for direct specialists and
orchestrated work. It also tells Codex to delegate only when the user asks for
subagents or applicable repository or skill guidance requests delegation.

## Permissions and Worktrees

The project config explicitly enables subagent operation:

```toml
[agents]
enabled = true
```

It does not pin a concurrency limit, model, or reasoning effort. Those values
remain session- and user-controlled.

Subagents inherit the parent session's sandbox and approval choices unless a
custom agent overrides them. The migration does not weaken the existing
workspace-write sandbox or command approval rules.

Planning agents may write their assigned Markdown artifacts directly in the
active worktree. Agents that change source, tests, runtime documentation, or
benchmarks must use the available isolated-worktree workflow before editing.
The reviewer may use a read-only or detached review worktree as appropriate.

No migrated agent may push, create or edit a pull request, merge, resolve
review threads, or otherwise mutate remote state unless the user explicitly
requests that action.

## Guidance and Artifact Migration

The detailed style guide is copied from
`.claude/rules/code-style.md` to `.codex/guidance/code-style.md` without
changing its behavioral requirements. The Codex copy is a prose guidance
document, not a `.rules` file: `.codex/rules/*.rules` remains reserved for
command approval policy.

The existing specification and API note are copied byte-for-byte:

| Source                                   | Codex destination                                  |
| ---------------------------------------- | -------------------------------------------------- |
| `.claude/specs/head-operator.md`         | `.codex/artifacts/specs/head-operator.md`          |
| `.claude/api-notes/docs-examples.md`     | `.codex/artifacts/api-notes/docs-examples.md`      |
| `.claude/rules/code-style.md`            | `.codex/guidance/code-style.md`                    |

Future Codex-created specs, API notes, and critiques use one shared kebab-case
slug under `.codex/artifacts/`, preserving the current traceability model.

## Preservation Contract

Preservation means:

- no tracked file under `.claude/` is deleted, renamed, or edited;
- `CLAUDE.md` is not edited;
- existing `.codex/config.toml` and `.codex/rules/default.rules` behavior is
  retained except for the additive `[agents]` configuration;
- existing specs and notes are copied rather than moved; and
- the migration does not rewrite historical `docs/superpowers/` artifacts.

Before implementation, hashes of all tracked `.claude/` files and
`CLAUDE.md` are recorded. After implementation, the same hashes must match,
and `git diff --exit-code -- .claude CLAUDE.md` must report no changes.

## Validation

The change starts with a focused structural test in
`scripts/test_codex_agents.py`. It initially fails because the Codex team does
not yet exist. The test then verifies:

1. `.codex/config.toml` and every `.codex/agents/*.toml` file parse with
   Python's `tomllib`.
2. Exactly the ten expected agent names exist and each name is unique.
3. Every agent defines non-empty `name`, `description`, and
   `developer_instructions`.
4. Each filename corresponds to its declared agent name.
5. New agent instructions do not use `.claude/` artifact paths,
   `EnterWorktree`, or `SendMessage`; the orchestrator uses current Codex
   subagent coordination terminology.
6. The team README roster matches the TOML files on disk.
7. The three generic Markdown mirrors are byte-for-byte equal to their Claude
   sources.
8. `AGENTS.md` points to the Codex team and artifact namespace.
9. `.codex/config.toml` enables agents without changing the existing approval
   or sandbox values.

Verification commands are:

```bash
python -m unittest scripts.test_codex_agents
python -c 'import pathlib, tomllib; [
    tomllib.loads(path.read_text())
    for path in pathlib.Path(".codex/agents").glob("*.toml")
]'
git diff --exit-code -- .claude CLAUDE.md
git diff --check
```

Because this migration changes guidance and configuration rather than engine,
Python package, or Studio behavior, the full Rust, Python, and web test matrix
is not required. The focused structural test and diff checks are the
proportionate verification.

## Delivery Constraint

In the current managed checkout, both `.codex` and `.git` are mounted
read-only. The approved contents can therefore be authored and verified in a
writable temporary clone, then exported as an exact patch or published only
after separate user authorization.

The migration must not pretend that a staged patch has already changed the
read-only checkout. Final hand-off must distinguish:

- files actually visible in the current workspace;
- files committed only in the writable staging clone; and
- any patch or archive the user must apply after the mount restriction is
  removed.

No remote branch or pull request is created without an explicit request.

## Acceptance Criteria

- All ten Claude roles have Codex-native custom-agent TOML equivalents.
- The Codex team preserves the current role boundaries and orchestration
  workflow while using Codex terminology, tools, and artifact paths.
- `AGENTS.md` documents how to find and invoke the Codex team.
- Existing style guidance, specification, and API note have Codex-side
  mirrors.
- `.codex/config.toml` explicitly enables agents without pinning models or
  weakening permissions.
- Structural validation passes.
- `.claude/**` and `CLAUDE.md` remain unchanged.
- No remote state changes occur without explicit authorization.
