# Calc Flow Agent Team

A coordinated team of specialist agents for calc-flow, the Rust-native micro-batch /
streaming calculation engine with Python adapters and a local web studio. Each agent owns
one phase of the spec → design → critique → implement → review → document pipeline. The
orchestrator routes work between them.

This directory is a Claude compatibility mirror. The canonical agent definitions and
descriptions live in `.codex/agents/`; synchronize team changes from `.codex/agents/` to
this directory, never in the reverse direction.

## Team Roster

| Role         | Agent             | Color  | Reads                                      | Writes                                      |
|--------------|-------------------|--------|--------------------------------------------|---------------------------------------------|
| Orchestrator | `cf-orchestrator` | purple | authorized context and completion reports  | task list                                   |
| Spec writer  | `cf-spec-writer`  | orange | issues, introduction.md, AGENTS.md, rules  | `.codex/artifacts/specs/<slug>.md`          |
| API designer | `cf-api-designer` | pink   | spec, crate exports, stubs, openapi.json   | `.codex/artifacts/api-notes/<slug>.md`      |
| Critic       | `cf-critic`       | red    | spec, api-note                             | `.codex/artifacts/critiques/<slug>.md`      |
| Implementer  | `cf-implementer`  | green  | spec, api-note, critique                   | source code, tests, TDD in worktree         |
| Tester       | `cf-tester`       | cyan   | source under-test, conventions             | tests for the touched surfaces, in worktree |
| Reviewer     | `cf-reviewer`     | amber  | PR diff, all upstream artifacts            | review report; merge on explicit request    |
| Performancer | `cf-performancer` | yellow | finished impl, benchmark suites, baselines | perf-regression report, coverage advisory   |
| Simplifier   | `cf-simplifier`   | blue   | finished impl, existing modules            | simplification report; optional apply edits |
| Doc writer   | `cf-doc-writer`   | teal   | current source, AGENTS.md, docs            | `docs/` and `CHANGELOG.md`                  |

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

## Artifact Layout

| Path                          | Owner           | Purpose                                                        |
|-------------------------------|-----------------|----------------------------------------------------------------|
| `.codex/artifacts/specs/`     | cf-spec-writer  | testable requirement specifications (created on demand)        |
| `.codex/artifacts/api-notes/` | cf-api-designer | public-API surface notes (created on demand)                   |
| `.codex/artifacts/critiques/` | cf-critic       | adversarial reviews of specs and api-notes (created on demand) |
| `docs/`                       | cf-doc-writer   | normative engine and usage docs (referenced by all agents)     |
| `CHANGELOG.md`                | cf-doc-writer   | existing dated log of fundamental changes                      |
| `.claude/rules/`              | (existing)      | normative coding/test conventions                              |

Filenames share a single kebab-case slug derived from the request, so work traces through
`.codex/artifacts/specs/tumbling-window.md` →
`.codex/artifacts/api-notes/tumbling-window.md` →
`.codex/artifacts/critiques/tumbling-window.md` end-to-end.

## How to Invoke the Team

- **End-to-end on a GitHub issue.** "Use `cf-orchestrator` to handle issue #12." The
  orchestrator analyzes the issue, plans the route, and delegates to teammates.
- **A single specialist.** Address the role directly: "Use `cf-spec-writer` to spec the
  tumbling-window operator described in issue #12."
- **Adversarial review of an existing plan.** "Use `cf-critic` on the spec at
  `.codex/artifacts/specs/tumbling-window.md`."
- **Out-of-band sweep.** "Use `cf-performancer` to check the branch for benchmark
  regressions" or "Use `cf-simplifier` on the diff before I merge."

## Conventions Each Agent Honors

- `.claude/rules/code-style.md` — functional-first Python/TypeScript, immutability, no
  caller-owned mutation, Arrow/Array-API backing rules, aligned markdown tables
- `AGENTS.md` — authoritative build/test/verify commands per surface; `CLAUDE.md`
  mirrors the maintained operational guidance for Claude users
- `docs/introduction.md` — domain vocabulary (Batch, Port, Operator, Pipeline, Checkpoint,
  engines, runners); behavioral claims must match these docs
- Repo git conventions (`AGENTS.md`) — `feature/`·`fix/` branches, imperative commits
  under 72 chars with a why-body, no tool-attribution trailer unless requested,
  category-prefixed PR titles, `## Summary` / `## Test plan` PR bodies (embedded in
  `cf-implementer` and `cf-reviewer`)

## Team Working Agreements

Two practices are mandatory for every agent that changes files in the repository
(`cf-implementer`, `cf-tester`, `cf-doc-writer`, `cf-performancer` when it adds a benchmark,
and `cf-simplifier` when the user has opted into apply mode; `cf-reviewer` also reviews
inside a worktree):

- **Worktree isolation.** Enter an isolated git worktree (`EnterWorktree`) before creating
  or editing any file. All edits, builds, iteration, and any explicitly requested
  commit/PR happen inside it, keeping the main working tree clean. The planning agents
  (spec writer, API designer, critic) write only into the shared `.codex/artifacts/`
  directories (created on demand) and do not need a worktree.
- **Test-driven development (TDD).** The implementer works strictly red → green → refactor:
  write a failing test for the next behavior, confirm it fails for the right reason, write
  the minimum code to pass, then refactor while green. Production code is never written
  ahead of a test that demands it. The doc writer is exempt from TDD (there is no code to
  test), but still works in a worktree.
- **Remote authority.** No agent pushes, creates or edits a PR, resolves review threads,
  merges, or performs another remote mutation without explicit user authority.

## Hand-off Etiquette

- One agent at a time per artifact. Don't fan out the same artifact to two agents in
  parallel.
- Self-contained prompts. The teammate agent doesn't see the parent conversation, so the
  invocation must include all paths, decisions, and acceptance criteria it needs.
- Verify before advancing. The orchestrator confirms each step's completion from the
  specialist's own report — specialists verify their own artifacts — before
  dispatching the next step.
- A `Block` verdict from the critic routes work back to the upstream author, not forward
  to the implementer.
