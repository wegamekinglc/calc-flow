# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Calc Flow is a micro-batch / streaming stateful calculation engine. Data flows
through compiled graphs in immutable `Batch` envelopes. Apache DataFusion is
the only table query/calculation engine; NumPy and JAX are optional array
engines. See `docs/introduction.md` for requirements and data flow. This project
lives under `workspace/calc-flow/` in the parent repo — git conventions (branch
naming, commit messages, PR titles) are inherited from the parent repo's
`CLAUDE.md`.

## Commands

```bash
uv sync                                  # install dependencies
uv run pytest                            # run all tests
uv run pytest -n auto                    # run tests across available CPU workers
uv run pytest --cov=calc_flow            # enforce the 90% core coverage floor
uv run pytest -k checkpoint
uv run ruff check .
uv run ruff format --check .
uv run ruff format .                     # apply formatting
CALC_FLOW_BENCHMARK_SCALE=overhead uv run --extra benchmark pytest benchmarks --benchmark-only
cd web-ui && npm ci
cd web-ui && npm run build
cd web-ui && npm test
cd web-ui && npm run test:e2e
./scripts/start_web_ui.sh                 # start the local API and Vite studio
./scripts/stop_web_ui.sh                  # stop both managed process groups
```

## Coding style

Target Python 3.13 or newer. Use four-space indentation, double quotes, and
modern type syntax (`list[str]`, `dict[str, Any]`, `A | B`). Keep `from
__future__ import annotations` in Python modules unless the project removes it
consistently. Prefer small, explicit modules over placeholder abstractions.
Keep table behavior Arrow-backed and array behavior Array API-backed. Do not
add incomplete stubs, unused fixtures, unused CLIs, or placeholder modules just
to reserve future structure. See `.claude/rules/code-style.md`.

## Architecture

Data flows through the system in immutable `Batch` envelopes. Table batches
contain `pa.Table` payloads and are calculated exclusively with DataFusion.
Array batches contain Python Array API objects owned by NumPy or JAX. Never
pass raw tables or arrays to pipelines, operators, runners, or engines.

### Operator, Pipeline, Checkpoint cycle

- **`Port`** — named operator boundary with a `BatchKind`, required flag, and
  optional exact Arrow schema.
- **`Operator`** (ABC) — declares ports and implements
  `process(inputs, context) -> Mapping[str, Batch]`. `snapshot`, `restore`, and
  `reset` form the checkpoint lifecycle.
- **`StatelessOperator`** — pure mapping transform. Construct with a callable,
  or subclass and override `process`.
- **`StatefulOperator`** — maintains `self._state: dict` across items.
  Subclasses implement `process`.
- **`Pipeline`** — DAG builder using `add_node`/`connect`; `then`/`add` are
  single-port linear sugar. `compile()` validates topology and port contracts.
- **`ExecutionPlan`** — immutable topology whose `execute(inputs)` returns named
  outputs, per-node timings, DataFusion metrics, and run metadata.
- **`Checkpoint`** — versioned state with pipeline fingerprint, source cursor
  and sequence, node-keyed JSON state, and creation time.
- **`CheckpointStore` / `FileCheckpointStore`** — persistence protocol and
  atomic local JSON implementation.

### Engines (`engine/`)

- **`Engine`** (ABC) — `evaluate(expression, data: Batch) -> Batch`.
- **`DataFusionEngine`** is the sole table engine. `evaluate()` handles
  expressions and assignments; `sql()` executes a single `SELECT` or CTE over
  named table batches.
- **`DataFusionRuntime`** owns a `SessionContext` per run, rejects mutating SQL,
  cleans up input aliases, and records query plans and timings.
- **`ExpressionOperator`** performs table calculations/projections/filters;
  **`SqlOperator`** performs multi-input DataFusion SQL.
- **`NumpyEngine`** and **`JaxEngine`** are optional array engines. Their
  expression evaluator interprets an allowlisted AST and must not use Python
  `eval`.
- **`UdfRegistry`** owns trusted vectorized implementations. Operators carry
  `UdfReference(name, version)` values; configurations and catalogs never carry
  source, callable objects, or import paths.
- **`UdfRegistrySnapshot`** is captured during compilation. Unknown versions and
  conflicting DataFusion versions fail compilation. Runs install only selected
  scalar UDFs and array expressions allow only selected array UDF calls.
- **`ArrayExpressionOperator`** executes restricted NumPy/JAX expressions inside
  the DAG while enforcing array-backend ownership.

Expression handling is centralized in `expression.py`:

- `split_assignment("c = a + b")` → `("c", "a + b")`. Returns `None` for
  non-assignment expressions. The regex guards against comparison operators
  (`==`, `!=`, `<=`, `>=`).
- `sql_projection("c = a + b", "input")` →
  `"SELECT *, (a + b) AS c FROM input"`. For non-assignment:
  `"SELECT (expr) AS result FROM input"`.

Array engines also expose a set of programmatic operation methods: `add`,
`subtract`, `multiply`, `divide`, `matmul`, `sum`, `mean`, `max`, `min`,
`transpose`, `reshape`. They accept an array `Batch` as the primary operand and
return an array `Batch` while preserving metadata.

### Runtime modes (`runtime/`)

- **`Source`** — replays batches from a JSON cursor; **`BatchingSource`** groups
  records by row/byte limits; **`Sink`** writes one output batch.
- **`MicroBatchRunner`** — yields `RunResult` objects, delivers sinks, and
  checkpoints source cursor/state after sink success.
- **`StreamingRunner`** — processes one batch into a `RunResult` and checkpoints
  after successful sinks. Delivery is at-least-once.

### Configuration and local studio

- **`ProjectConfig`** and related Pydantic models are the strict data-only graph
  format. `compile_project()` maps them to runtime operators and ports.
- **`FileProjectStore`** atomically stores canonical JSON under hashed IDs; YAML
  uses safe import/export only.
- **`calc_flow.web`** provides the optional loopback-only FastAPI `/api/v1`
  service and bounded spawned preview workers.
- **`web-ui/`** contains the React/TypeScript/Vite/React Flow studio. Its API
  types are generated from the checked-in OpenAPI document. Run its build and
  Vitest suite for UI changes.

### Test layout

Tests mirror the source tree under `tests/calc_flow/`. Every non-`__init__.py`
source module in `src/calc_flow/` must have a corresponding `test_<module>.py`
file. Name test functions `test_<behavior>()`. Add focused tests for public
behavior, regressions, and state recovery paths. Avoid tests that only preserve
unused scaffolding. Fixtures are defined locally in test files (no shared
`conftest.py`).
