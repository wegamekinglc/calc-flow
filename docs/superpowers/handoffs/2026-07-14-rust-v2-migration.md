# Rust V2 Migration Handoff

## Current State

- Branch: `feature/rust-v2-migration`
- Handoff base: `a43d69c`
- Migration implementation tasks completed: none
- Approved implementation tasks remaining: 25 of 25

The repository contains the approved architecture and the complete implementation
plan. No Rust workspace, Rust source, PyO3 binding, FastAPI v2 route, or frontend
v2 change has been implemented yet.

## Committed Work

| Commit | Subject | Contents |
| ------ | ------- | -------- |
| `f9f781b` | `docs: design Rust v2 migration` | Approved architecture, API boundaries, compatibility decisions, rollout, and risk controls. |
| `f6b004a` | `docs: plan Rust v2 migration` | Twenty-five test-driven implementation tasks with exact paths, commands, interfaces, and release gates. |
| `a43d69c` | `chore: ignore local worktrees` | Ignores `.worktrees/` so an isolated checkout can be created safely. |

Primary artifacts:

- `docs/superpowers/specs/2026-07-13-rust-v2-migration-design.md`
- `docs/superpowers/plans/2026-07-13-rust-v2-migration.md`

## Decisions Already Approved

- V2 is a breaking release; there is no v1 project or checkpoint converter.
- The complete calculation core, graph compiler, runtime, state, stores, and
  runners move to Rust.
- Public APIs are provided as a native Rust crate and a Rust-backed Python
  package.
- NumPy and JAX remain Python-hosted external providers. Rust-only hosts reject
  graphs that require them.
- The Studio backend remains Python and FastAPI, with spawned bounded workers.
- The migration uses a Rust-core-first rewrite and freezes Python v1 until the
  final removal task.
- The supported wheel matrix is Linux x86_64/aarch64, macOS x86_64/arm64, and
  Windows x86_64 for Python 3.13 or newer.

## Environment Findings

The original machine could not create a linked worktree because the managed
Windows sandbox rejected `git worktree add`. The clean checkout was therefore
switched directly to `feature/rust-v2-migration`; no feature edits were made on
`main` after that switch.

The sandbox also rejected `uv sync --extra dev`, so a trustworthy locked baseline
could not be established. A diagnostic run using the globally installed Python
environment collected 229 tests: 201 passed, 7 failed, and 21 errored. Those
results are not a repository baseline because the environment had:

- no installed editable `calc-flow` package or Hypothesis dependency;
- a DataFusion Python API version incompatible with the lockfile;
- an unwritable default pytest temporary directory.

Do not treat those failures as migration regressions. Re-run the locked baseline
on the new machine before Task 1.

## Resume Procedure

1. Make the commits available on the new machine, then check out
   `feature/rust-v2-migration` at or after this handoff commit.
2. Prefer an isolated worktree if the new environment supports it:

   ```bash
   git worktree add .worktrees/rust-v2-migration feature/rust-v2-migration
   cd .worktrees/rust-v2-migration
   ```

3. Install the locked development environment and establish a clean baseline:

   ```bash
   uv sync --extra dev
   uv run pytest
   uv run ruff check .
   uv run ruff format --check .
   cd web-ui/backend && uv run --project . --extra dev pytest --cov=calc_flow_studio
   cd ../.. && cd web-ui && npm ci && npm run build && npm test && npm run test:e2e
   ```

4. Start with Task 1 in
   `docs/superpowers/plans/2026-07-13-rust-v2-migration.md`. No task is recorded
   as complete, so do not skip the v1 semantic fixture freeze.
5. Follow each task's red-green-refactor sequence, focused verification, review
   checkpoint, and commit step. Do not start Python binding work until the Task 14
   Rust parity gate passes.

## Completion Gate

Before releasing v2, run the complete verification block in Task 25 and the
commands in `AGENTS.md`. In particular, require at least 90% Rust core line
coverage and 85% Studio backend coverage, regenerate the Rust project schema and
OpenAPI client types, and confirm the generated files have no diff.
