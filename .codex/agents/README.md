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
        -> implementer -> tester -> reviewer -> doc writer

out of band and on demand: performancer, simplifier
```

The orchestrator may shorten the route for narrow work, but it never skips the
reviewer. It does not invent missing requirements. A `Block` verdict returns to
the upstream author. Reviewer findings return to the implementer and tester,
followed by another review of the updated head.

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
  practice. The skill is an optional external installation; this repository
  does not vendor its `SKILL.md`.
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
