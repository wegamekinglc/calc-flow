# Repository Guidance

Calc Flow is a micro-batch and streaming stateful calculation engine. Data
flows through compiled graphs in immutable `Batch` envelopes. Apache DataFusion
is the only table query/calculation engine; NumPy and JAX are optional array
engines. See `docs/introduction.md` for requirements and data flow.

## Commands

```bash
uv sync --extra dev                      # install runtime and development dependencies
uv run pytest                            # run all tests
uv run pytest -n auto                    # run tests across available CPU workers
uv run pytest --cov=calc_flow            # enforce the 90% core coverage floor
uv run pytest -k checkpoint              # run checkpoint-focused tests
uv run ruff check .                      # lint
uv run ruff format --check .             # check formatting
uv run ruff format .                     # apply formatting
CALC_FLOW_BENCHMARK_SCALE=overhead uv run --extra benchmark pytest benchmarks --benchmark-only
cd web-ui && npm ci                      # install locked frontend dependencies
cd web-ui && npm run build               # type-check and build the studio
cd web-ui && npm test                    # run Vitest
cd web-ui && npm run test:e2e            # run the Playwright browser workflow
./scripts/start_web_ui.sh                 # start the local API and Vite studio
./scripts/stop_web_ui.sh                  # stop both managed process groups
```

## Git conventions

- Name feature branches `feature/<description>` and bug-fix branches
  `fix/<description>`. Use `main` as the base branch.
- Write commit messages with an imperative summary under 72 characters, then a
  blank line and a body explaining why when a body is useful. Do not add
  tool-specific attribution unless the user requests it.
- Prefix PR titles with one of `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`,
  `test:`, `style:`, `perf:`, or `ci:`. Keep them under 70 characters and use
  imperative mood.
- Use `gh pr create` or `gh pr edit`. PR bodies contain `## Summary` and
  `## Test plan` sections.

## Coding style

- Target Python 3.13 or newer.
- Use four-space indentation and double quotes.
- Keep `from __future__ import annotations` in Python modules unless the project
  removes it consistently.
- Use modern built-in type syntax such as `list[str]`, `dict[str, Any]`, and
  `A | B`.
- Prefer small, explicit modules over compatibility shims, duplicate abstraction
  layers, or placeholder abstractions.
- Keep table behavior Arrow-backed and array behavior Array API-backed,
  matching `docs/introduction.md`.
- Do not add incomplete stubs, unused fixtures, unused CLIs, or placeholder
  modules merely to reserve future structure.

For Markdown tables, align columns with pipes and pad each separator row so its
dashes span the full column width, including the spaces around cell content.

## Architecture

Data flows through the system in immutable `Batch` envelopes. Table batches
contain `pa.Table` payloads and are calculated exclusively with DataFusion.
Array batches contain Python Array API objects owned by NumPy or JAX. Never
pass raw tables or arrays to pipelines, operators, runners, or engines.

### Operator, Pipeline, Checkpoint cycle

- **`Port`** — named operator boundary with a `BatchKind`, required flag, and
  optional exact Arrow schema.
- **`Operator`** (ABC) — declares input/output ports and implements
  `process(inputs, context) -> Mapping[str, Batch]`. `snapshot()`, `restore()`,
  and `reset()` form the checkpoint lifecycle.
- **`StatelessOperator`** — pure mapping transform. Construct with a callable or
  subclass and override `process`.
- **`StatefulOperator`** — maintains `self._state: dict` across items.
  `snapshot`, `restore`, and `reset` deep-copy this dict. Subclasses implement
  `process`.
- **`Pipeline`** — mutable graph builder. `add_node()` and `connect()` construct
  a DAG; `then()` and `add()` are single-port linear sugar. `compile()` checks
  endpoints, kinds, schemas, single-writer inputs, and cycles.
- **`ExecutionPlan`** — immutable compiled topology. `execute(inputs)` creates a
  `RunContext`, restores state on failure, and returns a `RunResult` with named
  outputs, node timings, DataFusion metrics, and run metadata.
- **`Checkpoint`** — versioned value containing pipeline name/fingerprint,
  source cursor/sequence, node-keyed JSON state, and creation time.
- **`CheckpointStore` / `FileCheckpointStore`** — protocol plus atomic local
  JSON implementation. Pipeline names are hashed into safe filenames.

### Engines (`src/calc_flow/engine/`)

- **`Engine`** (ABC) — `evaluate(expression, data: Batch) -> Batch`.
- **`DataFusionEngine`** is the sole table engine. `evaluate()` handles
  expressions and assignments; `sql()` executes a single `SELECT` or CTE over
  named table batches.
- **`DataFusionRuntime`** owns one `SessionContext` per graph run, cleans up
  temporary table aliases, rejects DDL/DML/utility SQL, and collects query plans
  and timings.
- **`ExpressionOperator`** performs DataFusion calculations, projections, and
  filtering. **`SqlOperator`** performs multi-input table queries.
- **`NumpyEngine`** and **`JaxEngine`** are optional array engines. Their
  expression evaluator interprets an allowlisted AST and must not use Python
  `eval`.
- **`UdfRegistry`** owns trusted implementations. DataFusion scalar UDFs declare
  Arrow fields and volatility; array UDFs declare argument count and must retain
  the active backend. Operators use serializable `UdfReference(name, version)`
  values. Never put callable objects, source, or import paths in configuration.
- **`UdfRegistrySnapshot`** is captured at compile time. Compilation rejects
  unknown versions and conflicting DataFusion versions; each run registers only
  selected scalar UDFs. Catalog output must remain JSON-compatible metadata.
- **`ArrayExpressionOperator`** is the DAG operator for restricted NumPy/JAX
  expressions and explicitly referenced array UDFs.

Expression handling is centralized in `expression.py`:

- `split_assignment("c = a + b")` returns `("c", "a + b")`. It returns `None`
  for non-assignment expressions, and its regex guards against comparison
  operators (`==`, `!=`, `<=`, `>=`).
- `sql_projection("c = a + b", "input")` returns
  `"SELECT *, (a + b) AS c FROM input"`. For a non-assignment expression it
  returns `"SELECT (expr) AS result FROM input"`.

Array engines also expose programmatic operations: `add`, `subtract`,
`multiply`, `divide`, `matmul`, `sum`, `mean`, `max`, `min`, `transpose`,
`reshape`. They accept an array `Batch` as the primary operand and return an
array `Batch` while preserving metadata.

### Runtime modes (`src/calc_flow/runtime/`)

- **`Source`** — implements `read(cursor=None) -> Iterator[Batch]`.
  **`BatchingSource`** groups records by row/byte limits. **`Sink`** implements
  `write(batch)`.
- **`MicroBatchRunner`** — recovers a source cursor, yields every `RunResult`,
  writes configured sinks, and periodically checkpoints only after sink success.
- **`StreamingRunner`** — processes one formed batch into a `RunResult` and
  checkpoints after successful sinks. Both runners provide at-least-once
  delivery and roll back in-memory state when delivery fails.

### Configuration and local web service

- **`ProjectConfig`** and related strict Pydantic models are the canonical
  data-only graph format. `compile_project()` must produce the same plan as the
  Python builder. Table nodes never expose a backend selector.
- **`FileProjectStore`** stores sorted formatted JSON atomically under hashed
  IDs. YAML is safe import/export only. Never deserialize executable objects.
- **`calc_flow.web`** is optional behind the `web` extra. Its FastAPI routes live
  under `/api/v1`; `serve()` must reject non-loopback hosts.
- **`RunManager`** decodes bounded Arrow inputs in the parent, then uses spawned
  worker processes with timeout, CPU, resident-memory, output, cancellation,
  and lifecycle controls.
- **`web-ui/`** is React, TypeScript, Vite, and React Flow. API types are
  generated from `web-ui/openapi.json`; regenerate them after route/model
  changes. Keep React Flow node type maps outside render functions.

## Tests

- Mirror source files under `tests/calc_flow/`.
- Every source file in `src/calc_flow/`, except `__init__.py` and `__main__.py`,
  should have a corresponding `test_<module>.py` in the mirrored test path.
- Name test functions `test_<behavior>()`.
- Add focused tests for public behavior, regressions, and state-recovery paths.
  Avoid tests that only preserve unused scaffolding.
- Define fixtures locally in test files; do not add a shared `conftest.py`.

## Verification

Before considering a change complete, run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
cd web-ui && npm run build && npm test && npm run test:e2e
cd web-ui && npm audit --omit=dev
```
