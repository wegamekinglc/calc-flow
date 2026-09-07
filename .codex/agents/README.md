# Calc Flow Codex Agent Team

The Codex-native calc-flow team coordinates specialized work across the
Rust-native engine, PyO3/Python adapters, FastAPI backend, and React Studio.
The orchestrator routes work through the specialists; each specialist keeps a
bounded responsibility and returns a concise, verified hand-off.

The TOML definitions and descriptions in this directory are canonical.
Downstream compatibility manifests mirror these definitions; synchronize team
changes from this directory to those mirrors, never in the reverse direction.

## Team Roster

| Role         | Agent             | Reads                                    | Writes                                        |
|--------------|-------------------|------------------------------------------|-----------------------------------------------|
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
clear request -> implementer -> reviewer -> doc writer when needed
test coverage -> tester -> reviewer
documentation -> doc writer -> reviewer
```

Add spec/API/critic only when ambiguity, public-contract changes, or high-risk semantics
need them. Add tester for focused independent coverage or failure diagnosis. Performance
and simplification specialists run on demand; an observed required gate failure remains
blocking. The orchestrator uses the specialist's own completion evidence to advance;
it does not inspect source or specialist artifacts. Reviewer blockers return to the
appropriate author, followed by focused review of the updated head.

## Shared delivery policy

Clear work proceeds to implementation with concise decisions and testable acceptance
criteria. Spec/API/critic stages are conditional on ambiguity, public-contract changes,
or high-risk semantics. Trim templates to the task; use one main artifact plus at most
one blocking correction. Critic blocks only wrong implementation, compatibility breaks,
or acceptance failures; other findings are caveats. Routine documentation alignment
does not require a new concept critique. Final specialist review remains required.

Follow [AGENTS.md Verification](../../AGENTS.md#verification): smallest local tests and
necessary module checks; full regression and routine performance gates in CI; local
full runs only by explicit request, CI failure diagnosis, or clear high risk without
CI coverage. State the reason and limited scope. Keep the Rust 90% line and Studio
backend 85% coverage floors. Do not repeat unchanged passing checks. Take at most one
non-blocking CI snapshot after a commit/push or at review handoff; do not wait or poll
unless final results were explicitly requested. Pending permits handoff, not merge;
required failures and unresolved checks still block merge.

Client adaptations may change tool names, worktree/style entry points, model
inheritance in compatibility manifests, or use persistent issue delegation and
platform lifecycle rules. Preserve authorized compact-delivery overlays that agree
with this policy. Synchronization means canonical semantics with those adaptations
listed, not identical instruction bytes. Fix genuine conflicts in canonical first,
then synchronize downstream text; do not alter non-text agent configuration.
Orchestrator tools remain limited to the authorized context reads, dispatch, and
handoff exception in its definition, including only required comment body files.
This grants no implementation, testing, arbitrary repository editing, PR, or merge
authority. Respect client lifecycle limits; do not abandon run-owned background work.

## Shared Repository Surfaces

| Path                                 | Owner             | Purpose                                      |
|--------------------------------------|-------------------|----------------------------------------------|
| `.codex/artifacts/specs/`            | `cf-spec-writer`  | explicit, testable requirements              |
| `.codex/artifacts/api-notes/`        | `cf-api-designer` | developer-facing API decisions               |
| `.codex/artifacts/critiques/`        | `cf-critic`       | adversarial pre-implementation review        |
| `.agents/skills/code-style/SKILL.md` | repository        | shared Python, web, test, and Markdown rules |

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
- `$code-style` supplies detailed functional, input-immutability, testing, and
  Markdown guidance. Every custom agent enables it through `skills.config`.
- `docs/introduction.md` supplies the domain vocabulary and execution model.
- Agents preserve caller-owned inputs and use the exact verification relevant
  to the changed surfaces.

## Team Working Agreements

- File-changing specialists use `superpowers:using-git-worktrees` when
  available and otherwise follow the repository's safe isolated-worktree
  practice. That worktree skill is an optional external installation; this
  repository vendors only its own project-specific skills.
- Behavior changes proceed red, green, refactor: observe the focused test fail
  for the expected reason before implementation.
- Planning agents write only their assigned artifact. One agent owns an
  artifact at a time.
- No agent pushes, edits a PR, resolves review threads, merges, or performs
  another remote mutation without explicit user authority.

## Hand-off Etiquette

- Give every specialist a self-contained prompt with paths, decisions,
  acceptance criteria, and expected output.
- Specialists verify their own artifacts; the orchestrator advances from their reports.
- Return concise evidence: changed paths, commands run, observed results,
  unresolved questions, and the recommended next role.
- Do not fan out multiple writers onto the same file or artifact.
