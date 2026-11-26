# P1 StreamSession — specification

Date: 2026-11-15. Source baseline: `main` `39cab2fb` (product-identical to
`b9a46e12`; `39cab2fb` changed documentation only).
Branch: `feature/stream-session`. Plan:
[streaming core re-architecture](https://github.com/wegamekinglc/calc-flow/blob/4f83be5d5b12bbc2a93b1233f106033a487479df/docs/plans/2026-11-15-streaming-core-rearchitecture.md),
phase P1, first PR.
Status: revision 2, after [critique](../critiques/stream-session.md); ready
for implementation.

Revision 2 changes: synchronous Rust API on a session-owned current-thread
runtime (B4); admission narrowed to the plan's first graph (S1); `Ordered`
rejects rows at or before the current watermark (B1); source metadata stamping
(B2); runnable measurement protocol with ported #616 harness pieces (B3, S6);
user-approved running-loop exception with a rule amendment (B5); error table,
binding state machine, panic contract and test additions (S2–S10).

## 1. Objective

Add a public, synchronous, caller-thread push API over a compiled stream plan.
One `push` validates an input batch, runs it and an optional watermark through
every operator in topological order on the calling thread, and returns the
rows emitted by that call. There is no source task, sink task, progress
driver, Tokio↔asyncio callback bridge or per-edge task wakeup on this path.

This is the falsification step of the plan. The measured cost of one 1-row
warm dual-SMA append is ≈599 µs through the Python stream path and ≈6.1 µs for
the native rolling + projection dispatch alone (PR #616 artifacts under
`target/compiled-region-performance/`). The target for this PR is a Python
`push` p50 of ≤20 µs for that case (§8).

User decisions that apply: plan decision 1 approves this public push API and
reverses the removal of `StreamingRunner::step` (`crates/calc-flow/src/lib.rs:19`).
On 2026-11-15 the user approved a narrow running-event-loop exception for
calls on an existing session (§6.3). Plan decisions 2–6 are not implemented here.

## 2. Scope

In scope:

1. Rust core `StreamSession`, `SessionWatermark`, `SessionOutput`.
2. PyO3 binding `_StreamSession`.
3. Python `StreamSession`, `TableExpr.session`, `Program.session`.
4. AGENTS.md and `.claude/rules/code-style.md` amendment for the approved
   running-loop exception (§6.3).
5. Measurement harness: port the #616 warm-stream pieces and add a session
   driver (§8); run the protocol and commit the report.
6. Tests, rustdoc/doctests, `_native.pyi`, `__init__` export.
7. Documentation through `cf-doc-writer` after review (§9).

Out of scope (later P1 PRs or later phases): the NumPy path; plan decision 5
convenience-stream defaults; inlining single-source chains in the
`StreamingRunner` source task (supersedes draft PR #616); checkpoints in
sessions (P3); multiple dynamic inputs; static inputs; SQL, cross-section,
window, join, ASOF, snapshot SQL, DISTINCT, bounded ORDER, external providers,
native array sequence nodes and rolling indicator layouts (RSI/ATR/MACD);
`status()`/metrics; any operator-internal change.

## 3. Rust API

New module `crates/calc-flow/src/runtime/streaming/session.rs`, tests in
`runtime/streaming/session/tests.rs`. Crate-root exports next to
`StreamingRunner`: `StreamSession`, `SessionWatermark`, `SessionOutput`.

```rust
/// Caller-driven, in-memory execution of one compiled stream plan.
pub struct StreamSession { /* private */ }

impl StreamSession {
    /// Consumes a compiled plan and prepares every operator.
    pub fn new(
        plan: StreamExecutionPlan,
        config: &StreamRuntimeConfig,
        watermarks: BTreeMap<String, SessionWatermark>,
    ) -> Result<Self>;

    pub fn input_ids(&self) -> Vec<&str>;   // source binding IDs, plan order
    pub fn output_ids(&self) -> Vec<&str>;  // graph output IDs, plan order

    pub fn push(&mut self, input: &str, batch: Batch, watermark: Option<EventTime>)
        -> Result<SessionOutput>;
    pub fn advance(&mut self, input: &str, watermark: EventTime) -> Result<SessionOutput>;
    pub fn finish(&mut self) -> Result<SessionOutput>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SessionWatermark {
    /// Advances only through watermarks passed to `push` or `advance`
    /// (the runtime's source-provided mode).
    Explicit,
    /// Requires non-null event time that never decreases and stays after the
    /// current watermark; advances to `max(event_time) - 1 µs` after each push.
    Ordered { event_time_column: String },
    /// Advances to `max(event_time) - max_out_of_orderness` after each push.
    BoundedOutOfOrderness { event_time_column: String, max_out_of_orderness: Duration },
    /// Never advances; results finalize at `finish`.
    Disabled,
}

#[derive(Debug, Default)]
pub struct SessionOutput { /* BTreeMap<String, Vec<Batch>> */ }

impl SessionOutput {
    pub fn batches(&self, output: &str) -> &[Batch];   // empty slice when none
    pub fn is_empty(&self) -> bool;
    pub fn into_batches(self) -> BTreeMap<String, Vec<Batch>>;
}
```

### 3.1 Execution

`push`, `advance` and `finish` are synchronous. The session owns one
current-thread Tokio runtime built with `Builder::new_current_thread()
.enable_time()` (no I/O driver) at `new`, and drives each call with
`runtime.block_on(...)` on the calling thread. Operator futures, DataFusion
partition tasks and `yield_now` therefore run on the caller's thread; nothing
waits on another thread on the admitted path. Calling any of the three
methods while the current thread is inside a Tokio runtime context
(`tokio::runtime::Handle::try_current().is_ok()`) returns
`InvalidArgument { field: "session", message: "cannot run inside a Tokio runtime; call it from a blocking thread" }`
before any work. A synchronous call cannot be dropped midway, so no partial
drain survives a call.

Every public item has rustdoc. One doctest builds a rolling → projection plan
with `PipelineBuilder::compile_stream`, pushes two batches with explicit
watermarks and calls `finish`. Rustdoc states: sessions are in-memory and
non-durable; checkpoint fields of `StreamRuntimeConfig` are ignored, including
`checkpointing=true`; `Explicit` is the runtime's source-provided mode.

## 4. Semantics

### 4.1 Admission (checked in `new`)

Otherwise `new` returns `CalcFlowError::InvalidArgument` with the field path
shown:

| Condition                                                         | Field path                                   |
|-------------------------------------------------------------------|----------------------------------------------|
| not exactly one dynamic source binding                            | `session.inputs`                             |
| static inputs present                                             | `session.static_inputs`                      |
| native array sequence nodes present                               | `session.nodes.<id>`                         |
| node kind other than `Rolling`, `Expression`, `Union`             | `session.nodes.<id>`                         |
| `Rolling` with layout `ROLLING_INDICATOR_STATE_LAYOUT_VERSION` or `ROLLING_MACD_STATE_LAYOUT_VERSION` | `session.nodes.<id>` |
| an output requests `DeliveryGuarantee::ExactlyOnce`               | `session.requirements.delivery.<output>`     |
| a `watermarks` key that is not the input                          | `session.watermarks.<key>`                   |
| event-time column missing or not an Arrow timestamp               | `session.watermarks.<input>.event_time_column` |
| `max_out_of_orderness` zero or not representable in nanoseconds   | `session.watermarks.<input>.max_out_of_orderness` |
| `config.validate()` fails                                         | the config's own path                        |

Rejection messages for unsupported kinds name the kind and point to
`StreamingRunner`. Event-time resolution reuses `ResolvedEventTimeColumn`
rules (`progress/prepare.rs:475`) against the input route's expected schema;
`Ordered`/`BoundedOutOfOrderness` require a declared exact schema.

The job context is `StreamJobContext::new(allocate_job_id()?, plan.fingerprint(),
JsonMap::new(), None, CancellationToken::new()).with_checkpointing(false)`, the
same construction as `continuous.rs:1155-1163` with checkpointing off. Every
`OperatorTaskInputs` is built like `prepare_operator_task`
(`runner.rs:3526`) with `entity_work: None`, `sql_recovery: None`,
`checkpoint: None`, `restore: None`, released gates and a dropped-receiver-safe
entry-ack channel; each node gets `OperatorInputProgress::new(ingress names)`.
SQL budgets are applied with `set_sql_budgets` before decomposition, as
runners do (`runner.rs:1165`).

### 4.2 Push

Validation happens before any operator runs. A validation failure returns an
error and leaves the session usable and unchanged:

1. the session is open;
2. `input` is the session's input;
3. for **every** source route of the input (one binding may feed several
   nodes, `pipeline/stream.rs:375-381`): `batch` matches the route's expected
   kind and exact schema (`Port::validate`) and its stamped `EnvelopeCost`
   fits `config.edge_budget` (`EdgeBudgetExceeded`);
4. `Ordered`: across **every** record batch of the input, event time is
   non-null, nondecreasing, not below the maximum of earlier pushes, and
   strictly after the input's current effective watermark (a row at or before
   the watermark would be late for the default late policies,
   `docs/streaming-guide.md:1073-1077`). Comparison uses the declared unit
   as `i64`, like `_stream_inputs.py:201-218`; an empty batch does not
   advance anything;
5. an explicit `watermark` with `Disabled` is rejected; a watermark below the
   current one is rejected; an equal one is a no-op;
6. the generated candidate for `Ordered`/`BoundedOutOfOrderness` is computed
   on a clone of `GeneratedWatermarkState` (`progress/generated.rs:48-102`:
   nanosecond maximum, nulls ignored, `checked_sub`, `div_euclid(1000)`);
   an arithmetic failure is a validation error;
7. the next source sequence exists (overflow is a validation error).

Then the session stamps the batch exactly as the runner's source task does
(`sequenced_source_message`, `source_task.rs:1906`; extract or reuse it rather
than copying): `BatchMetadata::new(binding_id, sequence, attributes)`, sequence
starting at 0 and incremented per data push, attributes preserved, no cursor.
It enqueues `Data` on every route's edge and, when the effective watermark
increases, `Watermark(w)` after it. The effective watermark is
`max(generated candidate, explicit)` for `Ordered`/`BoundedOutOfOrderness`,
the explicit watermark for `Explicit`, and none for `Disabled`. The combination
of a generated and an explicit watermark is session-only semantics: the caller
declares completeness up to `w` (plan §4.1). Sequence, generator state and
watermark are committed only after a successful drain.

`advance` performs steps 1, 2, 5 and enqueues only the watermark. `finish`
enqueues `EndOfInput` on every route, drains, marks the session finished and
returns the flushed rows; a session with no pushes still finishes.

### 4.3 Drain and ordering

Nodes are processed in the plan's topological order
(`StreamRuntimePlanParts::nodes`, `pipeline/stream.rs:885`). For each node,
each ingress queue is emptied in FIFO order through `dispatch_message`
(`operator_task.rs:1473`), followed by `ingress_admission::acknowledge` (a
no-op here); when every ingress has seen end-of-input, the node runs
`finish_operator` once. Watermark aggregation, late-port suppression of
watermark/idle, output frontiers and `on_end` therefore follow the runner's
per-operator code. Running each node to quiescence in topological order is a
legal runner schedule; Union drains ingress A before B, one of the runner's
possible interleavings.

Per graph output, returned batches keep emission order; outputs have no
combined order. Watermark, idle and end-of-input messages reaching an output
are not returned. Session parity with the runner means equal concatenated
per-output row sequences (Union: equal per-branch subsequences or sorted rows).

### 4.4 Edges, budgets and memory

Add `OperatorEdgeSender::Session` and `OperatorEdgeReceiver::Session` in
`local_edge.rs` over one shared FIFO (`Arc<parking_lot::Mutex<VecDeque<StreamMessage>>>`)
with an edge ID and the configured budget. `send` applies the oversize rule
of `EdgeQueueCore::reject_oversize` (`edge_queue.rs`) and appends; it never
returns `Pending`. `budget()` reports the configured budget, so operators
chunk output exactly as on runner edges (`effective_output_budget`,
`operator_task.rs:2160`). The receiver has a synchronous `try_recv`.
`into_physical` rejects session edges. No runner path constructs them.

There is no queue-capacity backpressure: a queue holds everything its producer
emitted during that producer's turn in the pass, and the call returns only
after every queue is empty. A session therefore bounds memory per call by
the operators' own chunking, not per edge. The runtime-envelope bounded-FIFO
paragraph applies to runners and streams, not sessions; documentation says so.

### 4.5 Failure

Any error after validation marks the session failed and is returned; rows
emitted earlier in that call are discarded. Operator panics are contained with
`catch_unwind` around each dispatch and returned as
`CalcFlowError::Operator { node_id, message: format!("panicked: {}", panic_message(payload)) }`,
reusing `failure::panic_message` and its 1,024-byte bound. There is no task,
so `TaskPanicked` is not used; runtime-envelope documents the session variant.
Session errors are ordinary `CalcFlowError` values like batch `compute`
errors; they are not projected through the runner's `StreamingError`
redaction (the caller is in-process and owns the data). Later calls on a
failed session return
`InvalidArgument { field: "session", message: "failed earlier; create a new session" }`;
on a finished session, `... message: "already finished"`. Dropping a session
cancels its job token and drops its runtime.

### 4.6 Error table

Aligned for the binding and docs. Python lifecycle errors are raised by the
binding state machine before native code runs.

| Condition                                   | Rust                                    | Python                         |
|---------------------------------------------|-----------------------------------------|--------------------------------|
| schema differs from declared input schema   | (Python checks first) `Compile`         | `ValueError`                   |
| unknown input, watermark misuse, ordering   | `InvalidArgument`                       | `ConfigError`                  |
| batch over `edge_budget`                    | `EdgeBudgetExceeded`                    | `ExecutionError`               |
| called inside a Tokio runtime               | `InvalidArgument { field: "session" }`  | `RuntimeError`                 |
| operator failure or panic during a call     | `Operator`/`DataFusion`/other           | `to_py_err` class (`ExecutionError` family) |
| call on finished, failed, closed or busy    | `InvalidArgument { field: "session" }`  | `RuntimeError`                 |
| unsupported graph at construction           | `InvalidArgument`                       | `ConfigError`                  |

## 5. PyO3 binding

New `crates/calc-flow-python/src/session.rs`, class `_StreamSession`
(`module = "calc_flow._native"`, frozen pyclass with interior state),
registered in `lib.rs`:

```text
_StreamSession(plan: StreamExecutionPlan, config: dict, watermarks: dict[str, dict])
push(input: str, batch: object, watermark_micros: int | None) -> dict[str, list[pyarrow.RecordBatch]]
advance(input: str, watermark_micros: int) -> dict[str, list[pyarrow.RecordBatch]]
finish() -> dict[str, list[pyarrow.RecordBatch]]
close() -> None
input_ids, output_ids -> tuple[str, ...]
```

- The plan is consumed with `PyStreamExecutionPlan::take`; its Python owner
  stays alive with the session, as `_StreamingRunner` does.
- `config` uses `continuous::runtime_config`.
- `batch` accepts a native `Batch` (cloned inner), an object implementing
  `__arrow_c_array__` (record batch, imported with `pyo3_arrow::PyRecordBatch`)
  or `__arrow_c_stream__` (table, via `PyTable`), becoming a table `Batch`.
- State: `parking_lot::Mutex<SessionState>` with
  `Idle(Box<StreamSession>)`, `Busy`, `Closed`, `Failed(String)`,
  `Finished`. A call swaps `Idle` for `Busy`, releases the mutex, releases
  the GIL (`py.detach`), runs the native call, re-attaches, then stores the
  next state. `close()` during `Busy` stores `Closed`; the in-flight call
  then drops its session instead of restoring it. A native error after
  validation stores `Failed`; a validation error restores `Idle`. Lifecycle
  errors raise `RuntimeError` naming the state. The mutex is never held across
  `detach`. Inline unit tests cover every transition.
- Outputs are exported per record batch with `PyRecordBatch::into_pyarrow`
  (C data interface); no `RecordBatchReader`/`pa.table` round trip.
- A `#[cfg(test)]`-free fast path is not required beyond this; measure first.

## 6. Python API

New module `python/calc_flow/session.py`; export `StreamSession` from
`calc_flow`; add `_StreamSession` to `_native.pyi`.

```python
class StreamSession(Generic[T]):
    def push(self, inputs, /, *, watermark=None) -> T: ...
    def advance(self, watermark) -> T: ...
    def finish(self) -> T: ...
    def close(self) -> None: ...
    def __enter__(self) -> StreamSession[T]: ...
    def __exit__(self, *exc) -> None: ...   # close() without flushing

TableExpr.session(*, runtime=None, config=None, watermarks=None) -> StreamSession[pa.Table]
Program.session(*, runtime=None, config=None, watermarks=None) -> StreamSession[dict[str, pa.Table]]
```

### 6.1 Construction

- Rejects a running event loop with `_require_blocking("session")`
  (compilation is a blocking convenience call, like `collect_schema`).
- Compilation reuses `stream._compile_stream` with `portability=None`
  (portability requests remain a `stream()` feature); one fresh native plan
  per session; logical→physical input and output names come from it.
  `Program.session` requires stream mode like `Program.stream`.
- Static `Parameter` inputs and native array inputs are rejected with the
  logical name before compiling. Native admission errors carry physical node
  IDs; the docs say so.
- Output schemas for zero-row results come from `Program.collect_schema` on
  the same runtime, computed once at construction.

### 6.2 Calls

- Table sessions: `inputs` is one `pa.Table`, `pa.RecordBatch` or table
  `Batch`; `watermark` is a timezone-aware `datetime` or `Watermark`. The
  result is one `pa.Table`: `pa.Table.from_batches(batches)` when rows were
  emitted, else the cached empty table.
- Program sessions: `inputs` is a mapping by logical input name with the one
  dynamic input; `watermark` is a value or a one-entry mapping. The result maps
  every declared output name to a `pa.Table`; aliases of one physical output
  share batches.
- Input conversion matches ordinary iterables: `_table_batch` metadata
  normalization and the declared exact schema (`ValueError`). Budget and
  ordering checks are native.
- Watermark policies per logical input, as in `stream()`:
  `None` → `Ordered(event_time)` when the input declares event time, else
  `Disabled`; `SourceProvidedWatermarks()` → `Explicit`;
  `BoundedOutOfOrderness(col, delay, interval, idle_timeout)` →
  `BoundedOutOfOrderness(col, delay)` with `interval` ignored and
  `idle_timeout` required to be `None`; `DisabledWatermarks(idle_timeout)` →
  `Disabled` with `idle_timeout` required to be `None`. Docstrings say so.
- `config=None` uses `StreamRuntimeConfig()`; checkpoint fields are ignored.
- Late rows dropped by `late_policy="drop"` are not observable in P1 sessions
  (no `status()`); docs say so.

### 6.3 Running event loop (user-approved exception)

`push`, `advance`, `finish` and `close` may run inside a running event loop.
They execute the admitted operators on the calling thread with no I/O and no
cross-thread waits, so their cost is bounded by the caller's batch. The
docstring states that a large push blocks the loop for its duration. In this
PR, amend AGENTS.md (Python coding style) and `.claude/rules/code-style.md`
(Python) with this narrow rule: *calls on an existing `StreamSession`
(`push`, `advance`, `finish`, `close`) are synchronous caller-thread engine
calls with no I/O or cross-thread waits and may run inside an event loop;
session construction compiles and follows the blocking rule.* Any later
admission of an operator that waits on another thread must revisit the rule.

## 7. Tests (each starts red)

Rust, `runtime/streaming/session/tests.rs`, plans built with
`PipelineBuilder::compile_stream`. "Matches the runner" means equal
concatenated per-output row sequences against a `StreamingRunner` with
checkpointing off in the same test.

1. rolling → exact projection, one entity, three pushes with explicit
   watermarks: matches the runner.
2. `Ordered`: the latest timestamp stays open until a later push; `finish`
   flushes it; `advance` closes it without data.
3. B1: `push(watermark=T)` then a row at `T` is a validation error and the
   session stays usable; a row at `T + 1 µs` succeeds.
4. B2: rolling with `with_late_output` (side output) under `Explicit`
   watermarks: late rows, including `_cf_late_source` and `_cf_late_sequence`,
   match the runner.
5. Validation leaves the session usable: schema mismatch, oversize batch,
   watermark regression, decreasing `Ordered` time within a batch and across
   pushes, explicit watermark on `Disabled`, unknown input, empty batch accepted.
6. Operator failure (rolling late row, late policy error, `Explicit`
   watermark) fails the session; later calls return the failed-session error.
7. Admission rejects: two dynamic inputs, a static input, SQL, join, a
   rolling indicator layout, `ExactlyOnce`, an unknown watermark key, a
   non-timestamp event-time column, zero delay — each with its field path.
8. Fan-out: one input feeding two rolling nodes, two outputs and a Union of
   both branches: per-output rows match the runner (Union compared as sorted
   rows).
9. Internal edge `EdgeBudgetExceeded` after validation fails the session
   (an expression that widens rows beyond a small byte budget).
10. `finish` with no pushes; `finish` then `push` returns the finished error.
11. `BoundedOutOfOrderness` with nanosecond timestamps generates the same
    watermark sequence as `GeneratedWatermarkState`.
12. Calling `push` inside a Tokio runtime returns the documented error.
13. Panic containment through a `#[cfg(test)]` panic seam in the session
    dispatch wrapper: returns `Operator { message: "panicked: …" }` and fails
    the session.

PyO3 inline tests: binding state machine transitions (`Idle`/`Busy`/`Closed`/
`Failed`/`Finished`, close during busy).

Python, `python/tests/test_stream_session.py`:

1. `TableExpr.session` with `cf.ts` rolling mean under
   `SourceProvidedWatermarks` matches `TableExpr.stream` with the same policy
   and watermarks (compare concatenated rows after completion).
2. `Program.session` returns every declared output name, including zero-row
   tables with the declared schema.
3. Default `Ordered` keeps the latest timestamp open; `watermark=` closes it;
   pushing a row at that watermark raises `ConfigError` and the session still
   works.
4. Concurrent push: a registered scalar UDF blocks on a `threading.Event`
   while a second thread's push raises `RuntimeError`.
5. `close()` is idempotent; later calls raise `RuntimeError`; the context
   manager closes without flushing.
6. Schema mismatch raises `ValueError` and the session remains usable.
7. `push` works inside a running event loop; `TableExpr.session()` inside a
   running loop raises `RuntimeError`.
8. Static parameter input and `idle_timeout` policies are rejected.

Focused local commands:

```bash
cargo fmt --all --check
cargo test -p calc-flow --lib runtime::streaming::session
cargo test -p calc-flow --doc StreamSession
cargo clippy -p calc-flow -p calc-flow-python --lib --tests -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p calc-flow --no-deps
cargo test -p calc-flow-python --lib session
uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests/test_stream_session.py -q
uv run ruff check . && uv run ruff format --check .
git diff --exit-code -- schemas/project-v3.schema.json && git diff --check
```

## 8. Measurement protocol (pre-registered)

Question: how much of the 599 µs disappears with a caller-thread session.

### 8.1 Harness

Port from `origin/feature/compiled-stream-region` (draft #616) only the
harness pieces: warm-stream `exact_projection` and checkpoint-off support,
the `profile_warm_stream.py` flag, the TA-Lib bounded-history mode in
`rolling_indicator_comparison.py`, and their tests. Add a session driver to
the warm-stream worker and a compare option that assigns drivers per arm.
Both workers load the candidate checkout's harness; the baseline wheel runs the
runner driver, the candidate wheel runs the session driver. No new framework.

### 8.2 Configuration

- Graph and data: warm dual-SMA spread (`indicator="dual_sma_spread"`, window
  20, fast window 5), exact projection, history 4,096 rows, deterministic
  prices, per-append watermark at the last appended timestamp.
- Baseline build: reuse `target/compiled-region-performance/baseline-build`
  (`b9a46e12`, product-identical to `39cab2fb`). Candidate: a sealed release
  wheel of this branch from `scripts/profile_warm_stream.py build`.
- Runner driver: as the #616 measurement (checkpointing off,
  `SourceProvidedWatermarks`, forced GC). Timed boundary: unchanged warm-stream
  boundary (prepared Data/Watermark enqueue to sink table received).
- Session driver: the same prepared input object (table `Batch`) and
  watermark; timed boundary from immediately before `push(...)` to the
  returned `pa.Table`; forced GC before each sample like the runner. Candidate
  phase timers inside the same samples: Python argument handling and
  conversion, native call, result table assembly.

### 8.3 Cases (4) and samples

| Case | Workload              | Arms                                                        |
|------|-----------------------|-------------------------------------------------------------|
| 1    | 1 row × 1 entity      | baseline runner vs candidate session; TA-Lib reference arm  |
| 2    | 99 rows × 1 entity    | baseline runner vs candidate session                        |
| 3    | 6,400 rows × 64 ents  | baseline runner vs candidate session; TA-Lib reference arm  |
| 4    | 1 row × 1 entity      | baseline runner vs candidate runner (enum-variant control)  |

30 alternating AB/BA pairs per case in two persistent processes, seeded
bootstrap 95% CI of the median paired speedup, p50/p95 per arm. TA-Lib arms
are references, not paired arms. Every appended row matches the existing
oracle in every arm. Budget: 600 s wall-clock for all cases including startup,
warm-up, correctness checks and cleanup; builds reported separately. Stop when
exhausted; no added cases, samples or retries. Raw JSON and the Markdown
report go to `benchmarks/stream-session/p1-20261115.{json,md}`.

### 8.4 Verdicts

- Acceptance: case 1 candidate p50 ≤ 20 µs.
- Cases 1–3: "slower" means the speedup CI upper bound is below 1.0;
  a CI spanning 1.0 is inconclusive, not a regression.
- Case 4: a confirmed regression is a speedup CI upper bound below 0.95.
- If case 1 misses 20 µs, the phase timers locate the remaining cost; the PR
  reports it and stops for a decision (plan §7.2 stop rule) instead of
  expanding the measurement.

## 9. Documentation (via `cf-doc-writer`, after review)

`docs/python-api.md` ("Stream sessions" next to "Streaming results"),
`docs/streaming-guide.md` (session vs stream), `docs/rust-api.md`,
`docs/runtime-envelope.md` (session path: admission, no edge backpressure,
no checkpoint, panic variant, unredacted errors), `docs/api-reference.md`,
`docs/introduction.md` ("Choose how to declare and execute"), the AGENTS.md
Python architecture bullet, `CHANGELOG.md`, and the plan's execution record.
The `lib.rs:19` compile_fail doctest stays valid; its comment notes that
`StreamSession` is the supported push API.
