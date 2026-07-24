# Calc Flow Agent Team Design

**Status:** Implemented and merged in PR #19; later mirrored for Codex in PR #25

## Context

The sibling project `workspace/Derivatives-Algorithms-Lib` (DAL) operates a
coordinated team of ten specialist agents under `.claude/agents/`: an
orchestrator dispatches work through a spec → design → critique → implement →
review → document pipeline, with performancer and simplifier as out-of-band
advisory sweeps. That team is proven in daily use. This design ports it to
calc-flow.

Calc Flow differs from DAL in stack, not in kind. DAL is a single-language C++
library; Calc Flow 2.0 is a Rust-native engine (`crates/calc-flow`,
`crates/calc-flow-python` PyO3 bindings) with Python adapters
(`python/calc_flow/`), a FastAPI studio backend (`web-ui/backend/`), and a
React/TypeScript studio frontend (`web-ui/`). The pipeline mechanics of the
DAL team are project-agnostic; the project-context, conventions, and
verification content of each agent must be rewritten for this stack.

## Decisions

Settled with the user before this design:

1. **Full port, generalists.** All ten DAL roles are ported. Each agent is a
   generalist across the whole stack — the implementer implements in Rust,
   Python, or TypeScript depending on the task, mirroring how DAL's single
   implementer covers all of C++. No per-surface specialist split.
2. **Artifact layout.** The team adopts DAL's `.claude/` artifact directories:
   `.claude/specs/`, `.claude/api-notes/`, `.claude/critiques/`, with a shared
   kebab-case slug per work item. The existing `docs/superpowers/specs|plans`
   directories remain as historical record; the team's pipeline is the
   workflow going forward.
3. **Naming.** Agents use the `cf-` prefix, mirroring how `dal-` abbreviates
   Derivatives-Algorithms-Lib: `cf-orchestrator`, `cf-implementer`, etc.
4. **Approach: faithful port, content swapped.** Roster, pipeline, and README
   structure are copied exactly. Each agent body stays self-contained (full
   process text inline, as in DAL) rather than referencing superpowers plugin
   skills — no plugin dependency, no subagent skill-invocation failure mode.

## File and artifact layout

Eleven new files under `.claude/agents/`; the agent-definition layout is additive, so no existing agent file is modified. (The PR that lands the team additionally carries this design + plan under `docs/superpowers/` and, as a smoke test, adds a `.claude/specs/head-operator.md` artifact and a small snippet to the existing `docs/getting-started.md`; those are side artifacts of the rollout, not part of the agent file layout below.)

```
.claude/agents/
├── README.md               # roster, workflow, conventions, agreements
├── cf-orchestrator.md      # purple
├── cf-spec-writer.md       # orange
├── cf-api-designer.md      # pink
├── cf-critic.md            # red
├── cf-implementer.md       # green
├── cf-tester.md            # cyan
├── cf-reviewer.md          # amber
├── cf-performancer.md      # yellow
├── cf-simplifier.md        # blue
└── cf-doc-writer.md        # teal
```

Artifact directories are created on demand by the agents themselves:

| Path                   | Owner             | Purpose                                      |
| ---------------------- | ----------------- | -------------------------------------------- |
| `.claude/specs/`       | cf-spec-writer    | testable requirement specifications          |
| `.claude/api-notes/`   | cf-api-designer   | public-API surface notes                     |
| `.claude/critiques/`   | cf-critic         | adversarial reviews of specs and api-notes   |

Filenames share one kebab-case slug derived from the work item, so work traces
end-to-end: `specs/lazy-source.md → api-notes/lazy-source.md →
critiques/lazy-source.md`.

Every agent file has YAML frontmatter with `name`, a `description` including
trigger examples, `model: inherit`, and its color. No `tools:` frontmatter —
tool restrictions live in the agent body where they are explained, matching
DAL parity.

## Pipeline

Identical to DAL's:

```
request ──► spec-writer ──► api-designer ──► critic
                             (if public)       │
                                               ▼
                            implementer (+ tester) ◄──┘
                                   │
                                   ▼
                               reviewer        ◄── blocking gate, every iteration
                                   │
                                   ▼
                            doc-writer (reconcile docs/)
                                   │
                                   ▼
                              merged PR

out-of-band, on demand, advisory:  performancer, simplifier
```

The orchestrator is a pure dispatcher: its body restricts it to `Agent`,
`SendMessage`, and the task-tracking tools, and forbids `Bash`, `Read`,
`Write`, `Edit`, and web tools. It is the only agent that decides which steps
to skip, and it never skips `cf-reviewer`. Its routing table covers calc-flow
work types:

- Engine change (Rust) or Python binding change: spec-writer → (api-designer
  if public) → critic → implementer → tester → reviewer → doc-writer
- Studio backend/frontend: spec-writer → (api-designer if REST/OpenAPI) →
  critic → implementer → tester → reviewer → doc-writer
- Bug fix with clear scope: implementer → tester → reviewer → doc-writer
- Benchmark regression: performancer (advisory, out-of-band)
- Docs-only change: doc-writer → reviewer
- Test coverage gap: tester → reviewer

## Per-agent adaptations

Process text ports from DAL unchanged; project-context, conventions, and
verification sections are rewritten:

- **cf-spec-writer** — domain vocabulary: Batch envelopes,
  ports/operators/pipelines, checkpoints, DataFusion engine, runners, studio.
  Reads `docs/introduction.md` and `AGENTS.md`.
- **cf-api-designer** — public surface = Rust crate API (`crates/calc-flow`
  exports), Python API (`python/calc_flow/` + `_native.pyi`), studio REST API
  (`web-ui/openapi.json`).
- **cf-critic** — domain checklist: Batch immutability, Arrow/Array-API
  ownership, DataFusion-only table math, no Python `eval` in the array providers,
  UDF registry versioning.
- **cf-implementer** — TDD + worktree mandatory (unchanged). Per-surface
  verification matrix below. Style from `.claude/rules/code-style.md`
  (functional-first, no caller mutation).
- **cf-tester** — test layout: Rust unit tests in-crate,
  `python/tests/test_*.py`, `web-ui/backend` pytest, `web-ui`
  Vitest/Playwright. Independent of the implementer.
- **cf-reviewer** — verifies against all upstream artifacts, runs the full
  matrix, applies parent-repo git conventions (see below). Merges only when
  the user explicitly asks.
- **cf-performancer** — runs the benchmark command below, compares against
  baseline, writes an advisory regression report; aware of the all-scales
  benchmark CI.
- **cf-simplifier** — same report shape as DAL (duplication, dead code,
  verbose constructs); heuristics span Rust/Python/TypeScript; apply mode
  only on user opt-in.
- **cf-doc-writer** — owns the `docs/` tree (introduction, python-api,
  rust-api, api-reference, getting-started) and `CHANGELOG.md` (created on
  first use — the repo has none today). Worktree, TDD-exempt.

### Verification matrix

Embedded in `cf-implementer`, `cf-tester`, and `cf-reviewer` (the benchmark
command below is the `cf-performancer`'s, and stays advisory — it never gates):

- **Rust:** `cargo fmt --all --check` ·
  `cargo clippy --workspace --all-targets --all-features -- -D warnings` ·
  `cargo test --workspace --all-targets --all-features` ·
  `cargo llvm-cov --workspace --all-features --fail-under-lines 90`
- **Python:** `uv sync --extra dev` · `uv run maturin develop` ·
  `JAX_PLATFORMS=cpu uv run pytest python/tests -q` ·
  `uv run ruff check .` · `uv run ruff format --check .`
- **Studio backend:** `cd web-ui/backend &&
  uv run --project . --extra dev pytest --cov=calc_flow_studio`
- **Studio frontend:** `cd web-ui && npm ci && npm run sync:api &&
  npm run build && npm test` (plus `npm run test:e2e` when UI behavior
  changes)
- **Benchmarks:** `CALC_FLOW_BENCHMARK_SCALE=overhead JAX_PLATFORMS=cpu
  uv run --extra benchmark pytest benchmarks --benchmark-only`

### Git conventions

Embedded in `cf-implementer` and `cf-reviewer`, inherited from the parent
repo and qualified by this repo's `AGENTS.md`: branches `feature/<description>`
or `fix/<description>` off `main`; imperative commit summaries under 72 chars
with a why-explaining body and no tool-attribution trailer unless requested
(`AGENTS.md` overrides the parent repo here); PR titles with a category prefix
(`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `style:`, `perf:`,
`ci:`) under 70 chars; PR bodies with `## Summary` and `## Test plan`.

### Source of truth caveat

`AGENTS.md` is the current repository guidance (Rust-native 2.0). The
checked-in `CLAUDE.md` describes the retired pure-Python `src/calc_flow/`
layout and is stale. Every agent's project-context section names `AGENTS.md`
as the source of truth where the two disagree. Refreshing `CLAUDE.md` is out
of scope for this design.

## Working agreements and hand-off etiquette

Ported from DAL in substance:

- **Worktree isolation.** Every file-changing agent (implementer, tester,
  doc-writer, performancer when adding benchmarks, simplifier in apply mode)
  enters an isolated git worktree via `EnterWorktree` before the first edit.
  The reviewer reviews inside a worktree. Planning agents (spec-writer,
  api-designer, critic) write only into `.claude/` artifact directories and
  need no worktree.
- **Test-driven development.** The implementer works strictly red → green →
  refactor: a failing test witnessed failing for the right reason, then the
  minimum code to pass, then refactor while green. Production code is never
  written ahead of a test that demands it. The doc writer is TDD-exempt.
- **Hand-offs.** One agent at a time per artifact. Prompts are self-contained
  (teammates cannot see the parent conversation): they carry issue/request
  context, upstream artifact paths, the acceptance criteria for that step, and
  prior decisions. The orchestrator confirms each step's completion from the
  specialist's own report — specialists verify their own artifacts — before
  advancing.

## Failure handling

- A `Block` verdict from the critic routes work back to the upstream author,
  never forward to the implementer.
- A verification failure in the implementer or tester is fixed in the
  implementation — tests are never weakened to pass.
- Reviewer findings route back to the implementer; the reviewer re-runs after
  fixes. The reviewer is a blocking gate on every iteration.
- A missing upstream artifact routes to the missing step first; agents never
  improvise around a missing spec.
- An agent blocked by ambiguity reports back to the orchestrator or user
  rather than guessing.

## Validation

No automated test harness exists for agent files (DAL has none either); the
team is validated structurally at implementation time and by use afterward:

1. **Structural check** — every `.claude/agents/*.md` parses as YAML
   frontmatter; each `name` matches its filename; the README roster matches
   the files on disk.
2. **Smoke test** — one direct specialist invocation (`cf-spec-writer` on a
   small real feature) and one orchestrated docs-only run end-to-end,
   confirming prompts are self-contained and the orchestrator's tool
   restrictions hold.

## Out of scope

- Refreshing the stale `CLAUDE.md` (flagged; can be an early task for
  `cf-doc-writer` once the team exists).
- Migrating or reformatting the historical `docs/superpowers/` specs and
  plans.
- Any automated CI validation of agent files beyond the structural check
  above.
