# Repository Guidance

Calc Flow 3.0 is a Rust-native micro-batch and streaming calculation engine.
The `calc-flow` crate owns immutable `Batch` values, graph compilation,
DataFusion execution, projects, stores, checkpoints, and runners. The Python
package under `python/calc_flow/` is a PyO3 binding plus functional adapters;
it is not a second engine. `calc-flow-connectors` delivers connector implementations behind
feature gates. `calc-flow-studio` is a separate local FastAPI and
React application serving the `/api/v3` continuous job API. See `docs/introduction.md`.

## Commands

```bash
# Rust core and PyO3 Rust unit tests
uv sync --extra dev
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
uv run python scripts/run_rust_tests.py
CALC_FLOW_CONNECTOR_CONTAINERS=1 \
  CALC_FLOW_KAFKA_BOOTSTRAP=localhost:9092 \
  CALC_FLOW_PG_TEST_URL=postgresql://postgres:postgres@localhost:5432/postgres \
  CH_TEST_URL=http://localhost:8123 \
  uv run python scripts/run_rust_coverage.py
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps

# PyO3 package and Python adapters
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
cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177 --ignore RUSTSEC-2026-0235
cargo deny --locked check
python -m unittest scripts.test_run_rust_tests scripts.test_run_rust_coverage scripts.test_inspect_wheel scripts.test_release_config
```

Keep Cargo, Maturin, uv, coverage, and release outputs under the repository
`target/` tree when working from a constrained mirror. Never leave a generated
`python/calc_flow/_native*.so` in source.

The Rust coverage gate includes the opt-in Kafka, PostgreSQL change data
capture (CDC), and ClickHouse container tests in one llvm-cov profile set.
Start those three services before running `scripts/run_rust_coverage.py`; the
runner fails before compilation when any service environment variable is
missing and preserves the 90% workspace line floor without excluding connector
source.

`uv sync --extra dev` installs the NumPy and PyArrow dependencies imported by
the PyO3 Rust tests. Run the harness through `uv run python` so Cargo and the
test binary inherit that managed Python environment. The harness runs the core
targets normally, then compiles the `calc_flow_python` lib test and discovers
its executable before starting the runtime-only timeout. The compiled PyO3
test runs serially with a five-minute limit; the harness adds the managed
interpreter's library directory to the test process's loader path. Pass
`--python-stress-runs N` to repeat that isolated PyO3 test process. If the
PyO3 build is configured with `PYO3_PYTHON`, invoke the harness through that
same interpreter so its NumPy, PyArrow, and shared-library paths stay aligned.

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

## Specialist agents

The Codex-native calc-flow specialist team is defined in
`.codex/agents/`; its roster, workflow, and invocation examples live in
`.codex/agents/README.md`. Team-produced specifications, API notes, and
critiques belong under `.codex/artifacts/` with one shared kebab-case slug per
work item.

The `.codex/agents/` definitions and descriptions are canonical. The
`.claude/agents/` tree is a preserved compatibility mirror for Claude users;
when a team definition changes, synchronize its semantics from
`.codex/agents/` to `.claude/agents/`, never in the reverse direction.
Ordinary feature work must not rewrite either team definition.

## Coding style

### Rust

- Use Rust 2024 and the workspace MSRV in `Cargo.toml`.
- Keep `unsafe_code = "forbid"`; do not weaken workspace lints.
- Prefer a functional style: immutable values, pure transformations, explicit
  error variants, and deterministic `BTreeMap` ordering where output is
  serialized or observed. Confine necessary mutation to owned stateful
  boundaries. Treat caller-owned inputs as read-only; defensively copy when an
  underlying library requires mutation.
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

- `Batch` is the public graph/runner data envelope. Table batches contain
  Arrow record batches; external batches contain an explicitly registered
  provider payload.
- The streaming surface moves stream traffic on one typed `StreamMessage` per
  edge under `crates/calc-flow/src/runtime/streaming/`, with typed `EventTime`
  and `Epoch` values under `crates/calc-flow/src/time/`. Only the data variant
  wrapping an immutable `Batch` is publicly constructable; control messages are
  created through crate-private constructors, so there is no public runner
  control-injection API. The crate-root public `StreamingRunner` consumes a
  `StreamExecutionPlan` and returns an owning `StreamingJob` with whole-job
  preflight, bounded source/operator/sink tasks, job-scoped event-time progress,
  aligned epoch checkpoints, managed v3 manifest recovery for durable restart,
  per-output delivery proof, and deterministic metrics. Control construction,
  task/coordinator/supervisor internals, and reaper ownership remain
  crate-private. See `docs/runtime-envelope.md`.
- `Port` declares name, `BatchKind`, required flag, and optional exact Arrow
  schema.
- `OperatorMetadata` owns shared graph metadata. `BatchOperator` and
  `StreamOperator` split finite processing from continuous processing and
  checkpoint lifecycle. `ExpressionOperator` and `SqlOperator` implement both
  lifecycles; `UnionOperator` and `WindowAggregateOperator` are stream-only.
  External operators resolve through lifecycle-specific factories in
  `ProviderRegistry`.
- `PipelineBuilder` consumes immutable graph-building steps.
  `compile_batch()` and `compile_stream()` validate endpoints, kinds, schemas,
  one-writer inputs, UDFs, cycles, deterministic topology, inputs/outputs, and
  fingerprint.
- `BatchExecutionPlan::execute` creates one run-scoped DataFusion session and
  returns `RunResult` with named outputs, node timings, DataFusion metrics, and
  metadata.
- `UdfRegistry` owns trusted native implementations.
  `UdfRegistrySnapshot` is captured at compile time; configurations contain
  only `UdfReference` values.
- `StateBackend` opens lineage-exclusive state sessions.
  `LocalStateBackend` publishes immutable, checksum-verified segments, while
  `CheckpointManifest` is the strict v3 state-manifest contract and the managed
  continuous runtime's durable recovery truth.
- The old v2 `Source`/`Sink`, `MicroBatchRunner`, formed-batch push runner, and
  public `FileCheckpointStore` are removed. `FileProjectStore` remains the
  bounded atomic project-document store.

Apache DataFusion 54 is the sole table engine. Table operations accept one
expression/projection/filter node or one read-only `SELECT`/CTE SQL node.
Reject DDL, DML, utility statements, multiple statements, and table backend
selectors.

### Python binding

- Public Python source lives under `python/calc_flow/`; native bindings live in
  `crates/calc-flow-python/`.
- `Batch.from_pyarrow` and `Batch.from_array` construct Python-facing native
  envelopes.
- Functional `PipelineBuilder` emits strict project format v3 and compiles
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
  FastAPI routes live under `/api/v3`; by default `serve()` rejects non-loopback
  hosts, and changing that security boundary requires a separately reviewed
  exception.
- `RunManager` owns spawned continuous-job workers with concurrency,
  resident-memory, checkpoint-disk, cancellation, shutdown, and lifecycle
  controls.
- `web-ui/` is React, TypeScript, Vite, and React Flow. API types are generated
  from `web-ui/openapi.json`; regenerate both after route/model/version
  changes.

## Tests

- Rust unit tests live beside source; integration tests live under
  `crates/calc-flow/tests/`. The `calc-flow-python` PyO3 binding has no
  separate Rust integration-test directory — its behavior is covered by inline
  unit tests and the Python suite under `python/tests/`.
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
  schemas/project-v3.schema.json \
  web-ui/openapi.json \
  web-ui/src/api/schema.d.ts
git diff --check
```

For a release change, additionally build the core wheel, source distribution,
crate, and Studio wheel; run `scripts/inspect_wheel.py` for each; install the
wheels in clean environments; and run the core/Studio smoke checks.
