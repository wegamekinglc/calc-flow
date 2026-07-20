# Calc Flow Agent Team

A coordinated team of specialist agents for calc-flow, the Rust-native micro-batch /
streaming calculation engine with Python adapters and a local web studio. Each agent owns
one phase of the spec → design → critique → implement → review → document pipeline. The
orchestrator routes work between them.

## Team Roster

| Role         | Agent             | Color  | Reads                                      | Writes                                      |
|--------------|-------------------|--------|--------------------------------------------|---------------------------------------------|
| Orchestrator | `cf-orchestrator` | purple | GitHub issues, all artifacts               | task list                                   |
| Spec writer  | `cf-spec-writer`  | orange | issues, introduction.md, AGENTS.md, rules  | `.claude/specs/<slug>.md`                   |
| API designer | `cf-api-designer` | pink   | spec, crate exports, stubs, openapi.json   | `.claude/api-notes/<slug>.md`               |
| Critic       | `cf-critic`       | red    | spec, api-note                             | `.claude/critiques/<slug>.md`               |
| Implementer  | `cf-implementer`  | green  | spec, api-note, critique                   | source code, tests, TDD in worktree         |
| Tester       | `cf-tester`       | cyan   | source under-test, conventions             | tests for the touched surfaces, in worktree |
| Reviewer     | `cf-reviewer`     | amber  | PR diff, all upstream artifacts            | review report; merge on explicit request    |
| Performancer | `cf-performancer` | yellow | finished impl, benchmark suites, baselines | perf-regression report, coverage advisory   |
| Simplifier   | `cf-simplifier`   | blue   | finished impl, existing modules            | simplification report; optional apply edits |
| Doc writer   | `cf-doc-writer`   | teal   | current source, AGENTS.md, docs            | `docs/` and `CHANGELOG.md`                  |

## Workflow

```
request ──► spec-writer ──► api-designer ──► critic
                             (if public)       │
                                               ▼
                            implementer (+ tester) ◄──┘
                                   │
                                   ▼
                               reviewer        ◄── in-band gate (blocking, every iteration)
                                   │
                                   ▼
                            doc-writer (reconcile docs/ + CHANGELOG.md)
                                   │
                                   ▼
                              merged PR

  out-of-band quality sweeps (separate context, often background, on demand):
    ┌────────────────┐          ┌──────────────┐
    │ cf-performancer│          │ cf-simplifier│
    └───────┬────────┘          └──────┬───────┘
            │ consume finished implementation │
            └────────────► (advisory; do not block merge, do not gate doc-writer)
```

The orchestrator is the only agent that decides which steps to skip. Most requests take a
subset of the pipeline (see `cf-orchestrator.md` for the routing table). The main loop ends
at `cf-reviewer` → `cf-doc-writer`; `cf-performancer` and `cf-simplifier` are out-of-band
sweeps invoked on demand in a separate context (often background), not prerequisites to
merge.

## Artifact Layout

| Path                 | Owner           | Purpose                                                               |
|----------------------|-----------------|-----------------------------------------------------------------------|
| `.claude/specs/`     | cf-spec-writer  | testable requirement specifications (created on demand)               |
| `.claude/api-notes/` | cf-api-designer | public-API surface notes (created on demand)                          |
| `.claude/critiques/` | cf-critic       | adversarial reviews of specs and api-notes (created on demand)        |
| `docs/`              | cf-doc-writer   | normative engine and usage docs (referenced by all agents)            |
| `CHANGELOG.md`       | cf-doc-writer   | dated log of fundamental changes (created on first qualifying change) |
| `.claude/rules/`     | (existing)      | normative coding/test conventions                                     |

Filenames share a single kebab-case slug derived from the request, so work traces through
`specs/tumbling-window.md → api-notes/tumbling-window.md → critiques/tumbling-window.md`
end-to-end.

## How to Invoke the Team

- **End-to-end on a GitHub issue.** "Use `cf-orchestrator` to handle issue #12." The
  orchestrator analyzes the issue, plans the route, and delegates to teammates.
- **A single specialist.** Address the role directly: "Use `cf-spec-writer` to spec the
  tumbling-window operator described in issue #12."
- **Adversarial review of an existing plan.** "Use `cf-critic` on the spec at
  `.claude/specs/tumbling-window.md`."
- **Out-of-band sweep.** "Use `cf-performancer` to check the branch for benchmark
  regressions" or "Use `cf-simplifier` on the diff before I merge."

## Conventions Each Agent Honors

- `.claude/rules/code-style.md` — functional-first Python/TypeScript, immutability, no
  caller-owned mutation, Arrow/Array-API backing rules, aligned markdown tables
- `AGENTS.md` — build/test/verify commands per surface; the source of truth where the
  stale `CLAUDE.md` (retired `src/calc_flow/` layout) disagrees
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
  or editing any file. All edits, builds, iteration, and the commit/PR happen inside it,
  keeping the main working tree clean. The planning agents (spec writer, API designer,
  critic) write only into the shared `.claude/` artifact directories (created on demand)
  and do not need a worktree.
- **Test-driven development (TDD).** The implementer works strictly red → green → refactor:
  write a failing test for the next behavior, confirm it fails for the right reason, write
  the minimum code to pass, then refactor while green. Production code is never written
  ahead of a test that demands it. The doc writer is exempt from TDD (there is no code to
  test), but still works in a worktree.

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
