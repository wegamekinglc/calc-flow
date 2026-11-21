# Repository Guidance

Calc Flow is a Python calculation library for Arrow tables and stateful streams.
The application API under `python/calc_flow/` exposes immutable expressions,
`compute`/`compute_async`, SQL and `pipe` composition, reusable `Program`
collection, owned async stream results, synchronous stream sessions, and
integrations. The
`calc-flow` crate is the internal runtime and owns immutable `Batch` values,
graph compilation, DataFusion execution, projects, stores, checkpoints, and
runners. PyO3 and Python lowering connect the API to that single table runtime. `calc-flow-connectors` delivers connector implementations behind
feature gates. The Studio application lives in the separate
[calc-flow-studio](https://github.com/wegamekinglc/calc-flow-studio)
repository and depends on `calc-flow-python` as a published package. See
`docs/introduction.md`.

## Commands

These are complete command references for CI and explicitly scoped full
verification. Select local checks under [Verification](#verification).
Start each command group at the repository root.

```bash
# Rust core and PyO3 Rust unit tests
uv sync --extra dev
cargo fmt --all --check
cargo clippy --workspace --lib --bins --tests --examples --all-features -- -D warnings
uv run python scripts/run_rust_tests.py
CALC_FLOW_CONNECTOR_CONTAINERS=1 \
  CALC_FLOW_KAFKA_BOOTSTRAP=localhost:9092 \
  CALC_FLOW_PG_TEST_URL=postgresql://postgres:postgres@localhost:5432/postgres \
  CALC_FLOW_MYSQL_TEST_URL="${CALC_FLOW_MYSQL_TEST_URL:?set the MySQL test URL}" \
  CH_TEST_URL=http://localhost:8123 \
  uv run python scripts/run_rust_coverage.py
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps

# PyO3 package and Python adapters
uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests -q
uv run ruff check .
uv run ruff format --check .
uv run ruff format .

# Supply chain and release helpers
cargo audit --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177 --ignore RUSTSEC-2026-0235
cargo deny --locked check
python -m unittest discover -s scripts -p 'test_*.py' -t .
```

Keep Cargo, Maturin, uv, coverage, and release outputs under the repository
`target/` tree when working from a constrained mirror. Never leave a generated
`python/calc_flow/_native*.so` in source.

The Rust coverage gate includes the opt-in Kafka, PostgreSQL change data
capture (CDC), MySQL, and ClickHouse container tests in one llvm-cov profile set.
Start those four services before running `scripts/run_rust_coverage.py`; the
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

Pass `--no-run` to precompile the selected core, connector, and
PyO3 targets without executing tests. Run the normal command afterward with
the same build settings and target directory; it retains doctests and tests
that compile fixtures. Linux CI gives precompilation a 45-minute budget for cold
builds and normal execution a separate 30-minute budget, retaining three serial
PyO3 runs and the five-minute limit for each run.

Run informational benchmarks with:

```bash
CALC_FLOW_BENCHMARK_SCALE=overhead \
  JAX_PLATFORMS=cpu \
  uv run --extra benchmark pytest benchmarks --benchmark-only
```

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

- Target CPython 3.9 or newer for `calc-flow-python`. Use four spaces and
  double quotes, and retain `from __future__ import annotations`.
- Use built-in type syntax such as `list[str]`, `dict[str, object]`, and
  `A | B`.
- Prefer functions and frozen/slot data containers. Add a class only for
  identity, protocol, lifecycle, resource ownership, or explicitly stateful
  behavior.
- Never mutate caller-owned mappings, sequences, Arrow tables, arrays, or JSON
  values. Defensively copy before passing data to libraries that mutate.
- Keep pure CPU-local transforms synchronous. Await file, stream, process, and
  network I/O; blocking convenience methods must reject a running event loop.
  Calls on an existing `StreamSession` (`push`, `advance`, `finish`, `close`)
  are synchronous caller-thread engine calls with no I/O or cross-thread waits
  and may run inside an event loop. Session construction, including restore,
  and `checkpoint()` follow the blocking rule and reject a running event loop;
  checkpoint has no async alternative.
- Do not add compatibility shims for the removed Python v1 implementation.

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
  aligned epoch checkpoints, managed v4 manifest recovery for durable restart,
  per-output delivery proof, and deterministic metrics. Admitted single-source
  Rolling, Expression, and native array-sequence chains run inline in the source
  task without checkpoint or restore, retaining logical task identities and
  bounded sink backpressure; other graphs keep separate operator tasks. Control construction,
  task/coordinator/supervisor internals, and reaper ownership remain
  crate-private. The crate-root `StreamSession` instead drives an admitted
  plan (one dynamic input; rolling, expression, union, and native array-sequence
  nodes) synchronously
  on the caller's thread over session edges, with no tasks or queue
  backpressure. `StreamSession::new` retains the in-memory behavior;
  `new_with_checkpoints`, `restore`, and `checkpoint` add opt-in managed
  durable snapshots using v4 manifests with `job_kind=session`. See
  `docs/runtime-envelope.md`.
- `Port` declares name, `BatchKind`, required flag, and optional exact Arrow
  schema.
- `OperatorMetadata` owns shared graph metadata. `BatchOperator` and
  `StreamOperator` split finite processing from continuous processing and
  checkpoint lifecycle. `ExpressionOperator`, `SqlOperator`,
  `RollingOperator`, `CrossSectionOperator`, and `UnionOperator` implement both
  lifecycles. Finite union requires table ports with explicit exact schemas and
  `required=true`. `WindowAggregateOperator`, `StreamJoinOperator`, and `StreamAsofJoinOperator`
  also implement finite execution with run-local state. `SnapshotSqlOperator` is stream-only. External
  operators resolve through lifecycle-specific factories in `ProviderRegistry`.
- `OperatorMetadata: Send + Sync + Any` requires concrete operator types to
  be `'static`. Rolling/cross-section SideOutput exposes required `output`
  and `late` table ports. Late-derived paths allow only built-in single-input
  expression/SQL nodes before sinks, suppress Watermark/Idle, and retain
  first-in, first-out (FIFO) Barrier/EOF ordering and aligned recovery.
  Delivery remains per output.
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
  `LocalStateBackend` publishes immutable, checksum-verified segments.
  `CheckpointManifest` remains the strict v3 contract for direct legacy state
  transactions; managed runners and durable sessions use strict v4 manifests
  as their durable recovery truth.
- The old v2 `Source`/`Sink`, `MicroBatchRunner`, formed-batch push runner, and
  public `FileCheckpointStore` are removed. `FileProjectStore` remains the
  bounded atomic project-document store.

Apache DataFusion 55 is the sole table engine. Table operations accept one
expression/projection/filter node or one read-only `SELECT`/CTE SQL node.
Reject DDL, DML, utility statements, multiple statements, and table backend
selectors.

### Python application API and binding

- Public Python source lives under `python/calc_flow/`; native bindings live in
  `crates/calc-flow-python/`.
- `Batch.from_pyarrow` and `Batch.from_array` construct Python-facing native
  envelopes.
- Root `calc_flow` expressions, SQL, and `Program` are the application API.
  `pipe` applies a synchronous declaration function once; it does not serialize
  callbacks or run a Python pipeline per batch. Arrow schemas, named outputs,
  and SQL stages lower through one immutable expression graph to project-v3.
- SQL result schemas come from private native planning without reading rows.
  SQL output has a new row lineage and no inherited temporal ordering.
  Batch SQL supports multiple aliases; append stream SQL accepts one alias or
  the exact-schema multi-alias subset that appends rows without deduplication.
  Ordinary and grouped SQL aggregates emit a full cumulative snapshot after each
  batch by default. Rust `SqlOperator::set_aggregate_output_mode` and Python
  `changelog_sql` can instead select retract/upsert changes for a compact
  single-alias stream aggregate; project-v3 calls this `aggregate_output`.
  Eligible queries, including admitted grouped ordered LAST_VALUE, use compact
  state; unsupported aggregate shapes retain input and rerun over history.
  Other SQL runs per batch. SQL state has no fixed default cap;
  `sql_state_budget` limits cumulative input and `sql_compact_state_budget`
  independently requires and limits compact state per aggregate node. See
  `docs/streaming-guide.md` for admission, ordering and recovery boundaries.
  `snapshot_sql` is a separate stream-only complete-replacement query over
  latest initialized relations with exact schemas. In the current minimum
  viable product (MVP), its independent state budget accounts for candidates
  and checkpoint captures.
  Edge admission checks complete input and output messages against configured
  row/byte budgets and rejects oversized messages.
  Outputs currently connect directly to sinks and reject ExactlyOnce.
- `compute` and `collect` return Arrow tables using logical declaration names.
  Each convenience call owns a fresh batch plan. Async calls capture mappings
  and Batch references at entry; Arrow buffers remain shared and read-only.
  Internal schema normalization preserves caller metadata and `Batch.metadata`.
- `TableExpr.stream` and `Program.stream` translate logical bindings and own
  one native job through `async with` and `async for`. Results are Arrow tables
  or named `StreamOutput` events. Native state persists across batches, with
  bounded backpressure and awaited lifecycle cleanup. Event-time iterables default
  to validated nondecreasing arrival times and native watermarks at
  `max_seen - 1 microsecond`, published after every advancing batch (zero
  emission interval). `watermarks` selects existing policies by logical
  input name; supplied `SourceBinding` policies cannot be overridden. Inputs
  without event time retain per-batch output, except SQL aggregates with
  cumulative snapshots. `StreamRuntimeConfig.checkpointing=None` resolves to
  off for convenience streams and on for `StreamingRunner`. Ordinary iterables
  have no replay; opt-in temporary checkpoints do not provide durable restart or
  exactly-once application delivery.
- `python/calc_flow/session.py` implements `StreamSession` behind
  `TableExpr.session` and `Program.session`. Each session compiles one native
  plan with one dynamic input and maps watermark policies onto session
  watermarks; synchronous caller-thread `push`, `advance`, and `finish` return
  Arrow tables (programs: every declared output name), and `close` releases
  state without flushing. `checkpoints=None` retains in-memory behavior and
  ignores checkpoint configuration. Supplying the public
  `ManagedCheckpointRuntime` wrapper, unwrapped at the native boundary, enables
  durable v4 capture with effective `checkpointing=True`; `restore=True`
  requires storage and selects the highest complete session epoch, and
  `checkpoint() -> int` captures synchronously. Construction, restore, and
  checkpoint reject a running event loop; ordinary session calls retain their
  caller-thread contract. The PyO3 `_StreamSession` binding releases the global
  interpreter lock (GIL) per call and raises `RuntimeError` for busy, closed,
  finished, or failed sessions. See `docs/python-api.md`.
- `Program(name, mode="batch")` or `Program(name, mode="stream")` fixes the
  lifecycle at construction; legacy `engine="sql"|"streaming"` is an exclusive
  alias. Exactly one selector is required. Immutable `with_input` and `output`
  copies preserve it; `for_mode` creates an independent wrapper for another
  lifecycle. `Program.execute` dispatches to finite collection or owned streaming
  results. The same declaration graph has the same fingerprint across modes.
  `collect`/`collect_async` and `stream` reject the opposite mode.
- `Program.to_project` exports data-only native graphs without Python logical
  aliases or payloads. Reloaded projects and explicit runners use physical
  binding names; durable recovery uses explicit bindings and managed state.
- `with_late_output` returns immutable `LateOutputs(output, late)` references
  sharing one native rolling/cross-section state owner. Compilation rejects a
  Program unless both branches have explicit consumers; one owned iterator
  drains the named events. The marker supplies local
  lateness/policy and accepts one current stateful stage with existing-column
  operands and no upstream stateful stage. See `docs/streaming-guide.md`.
- Advanced functional `PipelineBuilder` and formula/SQL strings emit the same
  strict project-v3 graph and compile through Rust `Runtime`. Explicit runtime,
  cached-plan state, provider registration, and stream/checkpoint contracts remain
  available.
- Python scalar UDFs are trusted vectorized callbacks registered with exact
  Arrow types/provider/name/version/volatility and selected explicitly by
  nodes.
- NumPy/JAX are optional explicitly registered external providers. Their
  bounded AST evaluator must never use Python `eval`.
- `ProjectDocument` and the generated JSON Schema are the canonical data-only
  project contract. Never deserialize executable objects.
- Python runner/store adapters expose async methods and explicit blocking
  convenience methods.

## Tests

- Rust unit tests live beside source; integration tests live under
  `crates/calc-flow/tests/`. The `calc-flow-python` PyO3 binding has no
  separate Rust integration-test directory — its behavior is covered by inline
  unit tests and the Python suite under `python/tests/`.
- Python binding/adapter tests live under `python/tests/`. Define fixtures in
  focused test modules; do not create a shared `conftest.py`.
- Preserve `tests/fixtures/v1/` unchanged as historical parity evidence. It is
  not a v2 runtime or package path.
- Every behavior change starts with a focused failing test and records the
  expected failure before implementation.

## Release invariants

- Workspace crate and Python distribution versions move together as calendar
  versions (`YYYY.M.D` release dates, for example `2026.9.22`). The binding's Rust
  dependency is exact, and release tags are annotated
  `calc-flow-python-v<version>` at the current `main` head.
- `calc-flow-python` is the portable default. `calc-flow-python-v3` is a separate,
  explicitly installed, mutually exclusive distribution with the same version
  and `calc_flow` import package; install exactly one in a clean environment.
- Portable and v3 wheels contain only `calc_flow`, exactly one native module,
  metadata and a software bill of materials (SBOM), and license, with no runtime
  native-module selector.
- V3 wheels cover Linux, macOS, and Windows x86-64 with the same six Python tags
  and two abi3 tiers as portable wheels. Only v3 jobs set
  `target-cpu=x86-64-v3`; portable x86-64 and AArch64 builds remain unchanged.
  No v3 source distribution is published.
- Portable source distributions contain essential Rust/Python build sources.
  Frozen `src/calc_flow`, fixtures, and guidance directories are excluded unless
  a reviewed release exception documents why specific content is essential.
- The published crate includes license/source/examples/benches and excludes
  repository-only integration tests.
- Preserve pinned workflow actions, Maturin/Rust versions, audit policies, and
  artifact inspectors.

## Verification

Local verification covers the change and its directly affected behavior: use the
smallest focused unit/integration checks and necessary module compile, format,
lint, or type checks. Do not run full-repository, full-workspace, or all-surface
matrices locally by default. For documentation and agent guidance, check the
changed structure, synchronization, links/anchors, and diff; run only necessary
example checks that fit the requested scope, without building native code merely
to validate prose.

Full regression gates belong to GitHub CI. Routine benchmark measurements run
independently at 06:00 and 18:00 Asia/Shanghai every day. The complete commands
remain references for CI and explicitly scoped full verification. Keep the
combined Rust 90% line floor (including connector services); a skipped local
coverage run does not prove the gate.
Expand local full testing only for an explicit user request, reproduction or
diagnosis of a CI failure, or a clearly high-risk change without CI coverage.
State the specific reason and limited scope before using an exception. Local
performance measurements require a dedicated performance task, regression
diagnosis, or an explicit user request.

Feature performance measurements cover only paths directly affected by the
change and necessary controls or known regressions. Default to at most five
representative cases, using the smallest input that exercises the changed
behavior. Do not expand row counts, data types, modes, or checkpoint states into
a Cartesian matrix, or run a complete benchmark suite for ordinary feature
development. A broad measurement task requires an explicit user request.

Before execution, record each case's connection to the code change, baseline
and candidate, fixed sample count, and total wall-clock budget. The default
measurement budget is ten minutes across all selected cases, including fixture
creation, state restore, preflight, warmup, correctness checks, process startup,
and cleanup. Estimate and report builds separately. Reuse maintained harnesses
and compatible built artifacts; do not create a general measurement framework
for an individual feature. Keep the selected harness's correctness, resource,
paired-statistics, and provenance requirements.

Count each workload as a case; an atomic acceptance set is not one selection.
For example, `plan_end_to_end` defines a six-case acceptance set. A focused
subset reports per-case results; the set's aggregate and acceptance verdict
apply to complete-set measurements. Do not automatically complete the set to
satisfy its aggregate contract; complete-set measurements retain their scheduled
or explicitly requested broad scope.

Stop when the budget is exhausted or the comparison environment is unsuitable;
settle owned processes and report completed cases, unmeasured paths, and
uncertainty. Do not automatically expand the case set, sample count, or retry
until a favorable verdict appears. Inconclusive timings do not establish a
speedup, equivalence, or a confirmed regression. Apply the maintained regression
rule; do not invent a requirement that every confidence interval exclude the
regression threshold. Scheduled broad benchmarks remain independent of feature
development.

Use the smallest focused tests for test-driven development red/green and changed
refactors. Do not repeat an unchanged passing check; diagnose a recurring failure
before rerunning and report a blocker when it cannot be resolved.

After a commit/push or at review handoff, read at most one non-blocking CI status
snapshot. Do not wait, watch, poll, or sleep/retry unless the user or acceptance
criteria explicitly require the final CI result. Pending or absent CI can be
reported in a completed handoff; it is not green or merge-ready. Report failed,
cancelled, and inconclusive checks accurately. Required test, coverage, and
cross-platform checks block merge when failed or unresolved. Each Python package
distribution requires its artifact verification and respective installed-wheel
unit tests or clean-install smoke to pass before publication.
The final specialist review remains required; merging also requires explicit
authority and green required checks.

Confirm the generated contract has no unintended drift and check whitespace:

```bash
git diff --exit-code -- schemas/project-v3.schema.json
git diff --check
```

The Python release workflow builds and verifies the portable core wheels and
source distribution plus the separate v3 wheel matrix. It tests installed
portable Linux wheels across supported Python versions, checks v3 numeric vector
instructions and a clean v3 installation, and independently publishes the two
distributions on release tags after their respective gates. Benchmarks, soaks,
audits, crate packaging, and documentation checks do not run in that workflow.
Select local release verification under the same scope and exception rules above.
