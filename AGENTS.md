# Repository Guidance

Calc Flow 2.0 is a Rust-native micro-batch and streaming calculation engine.
The `calc-flow` crate owns immutable `Batch` values, graph compilation,
DataFusion execution, projects, stores, checkpoints, and runners. The Python
package under `python/calc_flow/` is a PyO3 binding plus functional adapters;
it is not a second engine. `calc-flow-studio` is a separate local FastAPI and
React application. See `docs/introduction.md`.

## Commands

```bash
# Rust core
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
cargo llvm-cov --workspace --all-features --fail-under-lines 90
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps

# PyO3 package and Python adapters
uv sync --extra dev
uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests -q
uv run ruff check .
uv run ruff format --check .
uv run ruff format .

# Studio backend
cd web-ui/backend
uv run --project . --extra dev pytest --cov=calc_flow_studio

# Studio frontend and generated API
cd web-ui
npm ci
npm run sync:api
npm run build
npm test
npm run test:e2e
npm audit --omit=dev

# Supply chain and release helpers
cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177
cargo deny --locked check
python -m unittest scripts.test_inspect_wheel scripts.test_release_config
```

Keep Cargo, Maturin, uv, coverage, and release outputs under the repository
`target/` tree when working from a constrained mirror. Never leave a generated
`python/calc_flow/_native*.so` in source.

Run informational benchmarks with:

```bash
CALC_FLOW_BENCHMARK_SCALE=overhead \
  JAX_PLATFORMS=cpu \
  uv run --extra benchmark pytest benchmarks --benchmark-only
```

Use `./web-ui/scripts/start_web_ui.sh` and
`./web-ui/scripts/stop_web_ui.sh` for the managed local Studio.

## Git conventions

- Name feature branches `feature/<description>` and fixes
  `fix/<description>`; use `main` as the base.
- Write imperative commit summaries under 72 characters. Add a body explaining
  why when useful. Do not add tool attribution unless requested.
- Prefix PR titles with `feat:`, `fix:`, `docs:`, `chore:`,
  `refactor:`, `test:`, `style:`, `perf:`, or `ci:`; keep them under
  70 characters.
- PR bodies contain `## Summary` and `## Test plan`.
- Preserve unrelated user changes in dirty worktrees and stage only the
  intended scope.

## Coding style

### Rust

- Use Rust 2024 and the workspace MSRV in `Cargo.toml`.
- Keep `unsafe_code = "forbid"`; do not weaken workspace lints.
- Prefer immutable values, pure transformations, explicit error variants, and
  deterministic `BTreeMap` ordering where output is serialized or observed.
- Return `calc_flow::Result<T>` from fallible public core operations. Preserve
  the failing field/path and source error where applicable.
- Async traits and functions own I/O, cancellation, stores, sources, sinks, and
  runner lifecycles. Do not block Tokio executor threads.
- Add public rustdoc and examples for new public APIs. Keep
  `RUSTDOCFLAGS="-D warnings"` green.

### Python

- Target Python 3.13 or newer, use four spaces and double quotes, and retain
  `from __future__ import annotations`.
- Use built-in type syntax such as `list[str]`, `dict[str, object]`, and
  `A | B`.
- Prefer functions and frozen/slot data containers. Add a class only for
  identity, protocol, lifecycle, resource ownership, or explicitly stateful
  behavior.
- Never mutate caller-owned mappings, sequences, Arrow tables, arrays, or JSON
  values. Defensively copy before passing data to libraries that mutate.
- Keep pure CPU-local transforms synchronous. Await file, stream, process, and
  network I/O; blocking convenience methods must reject a running event loop.
- Do not add compatibility shims for the removed Python v1 implementation.

### Web

- Apply the same functional-first and input-immutability rules to FastAPI and
  TypeScript.
- Use asynchronous FastAPI I/O and thread/process boundaries for blocking or
  CPU work.
- Update React state immutably; use functional updates when new state depends
  on previous state. Clean up streams, timers, listeners, requests, and child
  processes.
- Keep React Flow node-type maps outside render functions.

For Markdown tables, align columns with pipes and pad separator rows so their
dashes span the full column width, including cell spaces.

## Architecture

### Rust core

- `Batch` is the only graph/runner data envelope. Table batches contain Arrow
  record batches; external batches contain an explicitly registered provider
  payload.
- `Port` declares name, `BatchKind`, required flag, and optional exact Arrow
  schema.
- `Operator` owns async processing and checkpoint lifecycle. Built-ins are
  `ExpressionOperator` and `SqlOperator`; external operators resolve through
  `ProviderRegistry`.
- `PipelineBuilder` consumes immutable graph-building steps.
  `compile()` validates endpoints, kinds, schemas, one-writer inputs, UDFs,
  cycles, deterministic topology, inputs/outputs, and fingerprint.
- `ExecutionPlan::execute` creates one run-scoped DataFusion session and
  returns `RunResult` with named outputs, node timings, DataFusion metrics, and
  metadata.
- `UdfRegistry` owns trusted native implementations.
  `UdfRegistrySnapshot` is captured at compile time; configurations contain
  only `UdfReference` values.
- `Source`, `Sink`, `MicroBatchRunner`, and `StreamingRunner` provide
  at-least-once delivery. Checkpoints commit only after all sinks succeed.
- `FileProjectStore` and `FileCheckpointStore` write bounded documents
  atomically under hashed names.

Apache DataFusion 54 is the sole table engine. Table operations accept one
expression/projection/filter node or one read-only `SELECT`/CTE SQL node.
Reject DDL, DML, utility statements, multiple statements, and table backend
selectors.

### Python binding

- Public Python source lives under `python/calc_flow/`; native bindings live in
  `crates/calc-flow-python/`.
- `Batch.from_pyarrow` and `Batch.from_array` construct Python-facing native
  envelopes.
- Functional `PipelineBuilder` emits strict project format v2 and compiles
  through Rust `Runtime`.
- Python scalar UDFs are trusted vectorized callbacks registered with exact
  Arrow types/provider/name/version/volatility and selected explicitly by
  nodes.
- NumPy/JAX are optional explicitly registered external providers. Their
  bounded AST evaluator must never use Python `eval`.
- `ProjectDocument` and the generated JSON Schema are the canonical data-only
  project contract. Never deserialize executable objects.
- Python runner/store adapters expose async methods and explicit blocking
  convenience methods.

### Studio

- `web-ui/backend/` is the independent `calc-flow-studio` package. Its
  FastAPI routes live under `/api/v2`; `serve()` must reject non-loopback
  hosts.
- `RunManager` decodes bounded Arrow inputs in the parent, then uses spawned
  workers with timeout, CPU, resident-memory, output, cancellation, and
  lifecycle controls.
- `web-ui/` is React, TypeScript, Vite, and React Flow. API types are generated
  from `web-ui/openapi.json`; regenerate both after route/model/version
  changes.

## Tests

- Rust unit tests live beside source; integration tests live under
  `crates/calc-flow/tests/` and `crates/calc-flow-python/tests/`.
- Python binding/adapter tests live under `python/tests/`. Define fixtures in
  focused test modules; do not create a shared `conftest.py`.
- Preserve `tests/fixtures/v1/` unchanged as historical parity evidence. It is
  not a v2 runtime or package path.
- Studio backend tests live under `web-ui/backend/tests/` and enforce the
  independent 85% coverage floor.
- Frontend unit tests use Vitest; browser workflow tests use Playwright.
- Every behavior change starts with a focused failing test and records the
  expected failure before implementation.

## Release invariants

- Workspace crate, Python core, Studio, and frontend versions move together.
  The binding's Rust dependency is exact; Studio accepts the current v2 major.
- Core wheels contain only `calc_flow`, native module, metadata/SBOM, and
  license. Studio wheels contain only the Studio package/static assets,
  metadata, and license.
- Source distributions contain essential Rust/Python build sources but never
  frozen `src/calc_flow`, Studio, fixtures, or guidance directories.
- The published crate includes license/source/examples/benches and excludes
  repository-only integration tests.
- Preserve pinned workflow actions, Maturin/Rust versions, audit policies, and
  artifact inspectors.

## Verification

Before considering a change complete, run the full command groups in
`Commands`, then:

```bash
git diff --exit-code -- \
  schemas/project-v2.schema.json \
  web-ui/openapi.json \
  web-ui/src/api/schema.d.ts
git diff --check
```

For a release change, additionally build the core wheel, source distribution,
crate, and Studio wheel; run `scripts/inspect_wheel.py` for each; install the
wheels in clean environments; and run the core/Studio smoke checks.
