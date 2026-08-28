# Calc Flow design and architecture

Calc Flow is one Rust-native calculation engine with several entry points. The
Rust core owns semantics. Python is a PyO3 binding and adapter layer, connectors
are trusted implementations behind capability gates, and Studio is a separate
loopback application. None of those surfaces implements a second graph engine.

This document explains component ownership and end-to-end design. Use the
[getting-started guide](getting-started.md) for installation, the
[streaming guide](streaming-guide.md) for continuous jobs, and the
[API reference](api-reference.md) for exact public names.

## System map

```text
Rust application ───────────────┐
                               │
Python application ── PyO3 ────┼──> project/graph compiler ──> immutable plan
                               │                               │
Studio /api/v3 ── worker ──────┘                               ├──> batch executor
                                                               └──> streaming runner
                                                                        │
connectors <── trusted factories <───────────────────────────────────────┤
state/checkpoints <── managed runtime <──────────────────────────────────┘
```

| Component              | Owns                                                                 | Does not own                                      |
| ---------------------- | -------------------------------------------------------------------- | ------------------------------------------------- |
| `calc-flow`            | `Batch`, graph compile, DataFusion, plans, runners, state, manifests | Transport-specific clients or browser UI          |
| `calc-flow-python`     | PyO3 classes and async bridges                                       | Alternative execution semantics                   |
| `python/calc_flow`     | Functional builders, adapters, NumPy/JAX registration                | Serialized executable code                        |
| `calc-flow-connectors` | File, Kafka, PostgreSQL, ClickHouse, HTTP, WebSocket implementations | Graph compilation or job supervision              |
| `calc-flow-studio`     | Local process workers, resource limits, `/api/v3`, static assets     | Public-hosted multi-user service                  |
| React Studio           | Project editing, job controls, SSE observation                       | Direct access to connector secrets or checkpoints |

## Stable contracts

Four contracts connect the system:

1. `Batch` is the immutable data envelope passed through graphs and runners.
2. `ProjectSpec` format v3 is the strict data-only persisted configuration.
3. `BatchExecutionPlan` and `StreamExecutionPlan` are distinct compiled
   artifacts with deterministic fingerprints.
4. `CheckpointManifest` v3 is the durable truth for continuous recovery.

The graph configuration never contains a Python callable, Rust trait object,
import path, credential value, or serialized operator instance. A project
selects trusted registrations by exact provider, name, and version.

## Data model

A table `Batch` contains Arrow record batches. DataFusion is the only table
expression and SQL engine. An external `Batch` contains a payload owned by an
explicitly registered provider, such as a read-only NumPy or JAX array.
Table-to-array conversion is never implicit; `table_matmul` is the explicit
mixed-kind bridge.

Ports declare name, batch kind, whether an input is required, and optionally an
exact Arrow schema. Compilation rejects missing endpoints, incompatible kinds
or schemas, multiple writers to one input, invalid UDF references, cycles, and
ambiguous graph inputs or outputs before execution begins.

Caller-owned tables, arrays, mappings, sequences, and JSON values are treated
as read-only. Python adapters copy configuration at entry, and core plans own
immutable snapshots of registries and engine configuration.

## Batch path

```text
builder/project
    -> validation and deterministic graph compile
    -> one run-scoped DataFusion session when table nodes exist
    -> topological node execution
    -> named output batches + timings + DataFusion metrics
```

Each batch execution receives a fresh run context. It can carry copied JSON
settings, a cooperative cancellation token, and an absolute UTC deadline.
External-only plans do not initialize DataFusion. A mixed plan initializes one
session for its table nodes and keeps external-provider execution behind the
same cancellation, rollback, and timing boundaries.

## Streaming path

```text
stream plan + source/sink bindings + checkpoint root
    -> whole-job capability and delivery preflight
    -> bounded source, progress, operator, and sink tasks
    -> data + runtime-owned control on typed stream edges
    -> status, metrics, checkpoints, and one terminal outcome
```

Every edge carries one typed `StreamMessage`: immutable data, watermark,
barrier, idle, or end-of-input. Connectors may provide data, watermarks, and
idle observations. Barrier and end-of-input construction stays inside the
runtime, so an application cannot inject control traffic into a running job.

Rows and bytes are bounded independently on each edge. The source pauses at
the async send boundary when downstream work is slow; it does not accumulate
an unbounded application queue. One job-scoped progress driver owns watermark
generation, idle/reactivation, multi-source minimum progress, and deterministic
timer ordering.

`StreamingRunner` and `StreamingJob` are one-shot owners. Consuming ownership
makes connector close, task settlement, reaper cleanup, and terminal outcome
linearizable. Dropping the last Python job owner schedules cancellation and
cleanup instead of detaching background work.

## Stateful windows

`WindowAggregateOperator` supports fixed UTC tumbling and hopping windows.
Its `WindowSpec` names the event-time column, ordered group columns, geometry,
and ordered `count`, `sum`, `min`, `max`, or `avg` aggregates. It is stream-only.

The operator keeps deterministic incremental state keyed by window bounds and
group values. A watermark closes eligible windows; end-of-input closes the
remaining non-empty windows. Assignments whose window end is not later than
the current input watermark are late and are not applied. Session windows,
allowed-lateness updates, retractions, and early triggers remain unavailable.

## Stateful stream join

`StreamJoinOperator` is the bounded two-input stateful stream component. Its
`StreamJoinSpec` declares a two-input inner equi-Join over inclusive asymmetric
event-time bounds, explicit per-side state and per-batch match limits, and
output prefixes; the output schema is derived from the prefixed input columns.
Retained state is keyed and charged through a versioned logical encoding, so
admission decisions are deterministic across restarts.

Watermark progress evicts rows that can no longer match; null event times,
null keys, and rows strictly older than their own ingress watermark are never
retained. The operator checkpoints its delta-based state and an independent
output frontier, and exposes a payload-free per-node status through the job
status surface.

## Rolling windows

`RollingOperator` evaluates native lag, delta, and aggregate outputs over
entity-partitioned, event-time-ordered rows and runs in both batch and
stream graphs. Its `RollingSpec` declares ordered partition and sequence
keys, a non-null UTC `timestamp[us]` event-time column, lag/delta outputs
with positive row distances or count/sum/mean/min/max/variance/stddev and
covariance/correlation outputs over row-count or duration frames with
minimum-period gates, allowed lateness, and an envelope-scoped `error` or
metrics-recorded `drop` late-row policy.

Aggregates count valid samples — non-null, non-NaN values, with infinities
counting as numeric samples — and outputs over the same input column and
frame share one reversible accumulator group per entity. Floating sums
follow IEEE arithmetic, integer sums stay exact and checked through a wide
transient slide, and the statistical outputs read a West accumulator with
explicit infinity classification: a mean over one sign of infinity is that
infinity, over both signs it is NaN, and variance and standard deviation
over any infinity window are NaN behind the minimum-period and divisor null
gates. `min` and `max` read a monotonic extrema queue that preserves the
input type and orders floating samples by the IEEE total order; covariance
and correlation read a reversible West co-moment accumulator over
pairwise-valid positions — any infinity on either side reads NaN behind the
minimum-period and divisor gates, and only a finite zero-variance window
reads null for correlation.

Stream execution buffers rows until the input watermark passes each row's
event time plus the allowed lateness, then emits final rows in canonical
order; batch evaluation classifies no late rows. The operator checkpoints
its per-entity histories as an Arrow IPC segment with state version 1 — the
segment stores only history and buffered rows, and restore rebuilds every
accumulator by the same ordered fold — so a restored or reset operator
reproduces the same ordered output.

## Cross-section groups

`CrossSectionOperator` evaluates complete-group rank, percentile, z-score,
and demean outputs over exact-time or fixed-bucket groups and runs in both
batch and stream graphs. Its `CrossSectionSpec` declares the event-time
column, ordered entity and sequence keys, an optional partition key, the
grouping, per-output ordering choices with minimum samples, allowed
lateness, and a late-row policy; row identity is the event time, entity key,
and sequence key, and duplicates are rejected transactionally.

One micro-batch is never evidence of completeness: groups accumulate across
envelopes and close only when the input watermark reaches the group's
finality coordinate — the exact event time or the bucket end plus the
allowed lateness, with equality closing — or at end of input. A closed group
emits once in canonical order — groups by finality coordinate then key,
rows by event time, entity, and sequence — and releases its state and
identity index. Late rows follow the policy, with `error` rejecting the
whole envelope and `drop` discarding the row while recording metrics. Open
groups checkpoint at the aligned epoch cut as versioned Arrow IPC state —
per-segment configuration-hash and schema-fingerprint metadata with bounded
inline manifest fields — and restore reproduces the same ordered output,
watermark frontier, and metrics.

## Checkpoint transaction

One checkpoint creates a consistent epoch across the job:

1. sources reach an exact cut without splitting a connector-defined atomic
   unit;
2. runtime barriers follow the accepted source data;
3. multi-input operators align the epoch and stage versioned state segments;
4. transactional sinks pre-commit and return bounded evidence;
5. the runtime publishes the manifest last as the durable commit point;
6. sinks complete their commit and sources may acknowledge durable cursors.

Immutable state segments are checksum-addressed. A manifest binds the plan
fingerprint, participant identities, source positions, operator state,
watermark progress, and sink evidence. Recovery selects only a complete,
compatible manifest. A post-manifest sink failure is completed forward during
recovery; it is not rolled back behind durable truth.

A terminal checkpoint records ended sources and final operator output. Starting
the same completed lineage again returns the recovered terminal outcome without
reopening sources or emitting final windows twice. The executable behavior is
shown in [`08_streaming_recovery.py`](../examples/08_streaming_recovery.py).

## Delivery model

Delivery is proved per graph output, not claimed once for the whole process.
The runtime evaluates every reachable source, operator, edge policy, and sink
before opening a connector.

| Requested contract | Required route evidence                                           | Failure behavior                                    |
| ------------------ | ----------------------------------------------------------------- | --------------------------------------------------- |
| Best effort        | Explicit request; lossy or unreplayable sources are allowed       | Accepted data may be unavailable for replay         |
| At least once      | Lossless replayable sources and an ordinary or stronger sink      | A sink may observe duplicates after recovery        |
| Exactly once       | Exact replay, restorable operators, transactional/idempotent sink | Incompatible routes fail during whole-job preflight |

Status exposes both requested and effective delivery for every output. A
best-effort request is not silently upgraded, and an exactly-once request is
not silently weakened.

## Project and registry design

Project v3 stores graph nodes and edges, batch or stream runtime options,
connector bindings, format identities, named secret references, bounded state
settings, and delivery requests. JSON is canonical storage; YAML is a safe
import/export representation. Unknown fields and duplicate JSON keys fail.

Registries are the trust boundary:

- `UdfRegistry` resolves exact native DataFusion scalar UDF references;
- `ProviderRegistry` resolves lifecycle-specific external operator factories;
- `ConnectorRegistry` resolves transport and format factories;
- secret resolvers turn named references into values only when a connector
  opens.

Catalog and capability APIs expose safe metadata, not callback objects,
credential values, cursor payloads, or connector state.

## Python boundary

Python `PipelineBuilder` produces canonical project-v3 graph JSON and compiles
through the native `Runtime`. Blocking batch methods reject a running event
loop. Async methods let native work proceed without blocking Python and wait
for native cleanup before surfacing task cancellation.

Python scalar UDFs are trusted vectorized callbacks with exact Arrow types,
version, provider, and volatility. NumPy/JAX providers use a bounded allowlist
AST evaluator; they never call Python `eval`.

The Python continuous adapter validates connector method shapes before native
launch. It keeps connector callbacks on their owning event loop and provides
guarded blocking methods only outside an active loop.

## Studio boundary

Studio is a local application, not part of the core wheel. Its FastAPI backend
accepts only loopback hosting, persists projects, spawns bounded workers, and
owns job concurrency, memory, checkpoint-disk, cancellation, and shutdown
limits. `/api/v3/jobs` exposes admission and lifecycle controls. Server-sent
events provide reconnectable status and result observation.

The frontend edits the same project-v3 document used by Rust and Python. Its
checked OpenAPI file and generated TypeScript types must change together with
backend models.

## Failure, cancellation, and diagnostics

Public failures are typed. Streaming errors include safe category, job/epoch,
checkpoint phase, component kind/ID, diagnostic ID, and position fields where
applicable. Payload rows, cursor contents, secret values, filesystem paths,
connector internals, and panic messages are excluded from public projections.

Cancellation is cooperative at engine boundaries and owner-settled: the
terminal operation waits for connector close, task join, queue release, state
transaction settlement, and reaper cleanup. Cleanup failures remain secondary
diagnostics and do not overwrite the primary terminal cause.

## Extension choices

- Add table logic with `ExpressionOperator`, `SqlOperator`, or a trusted native
  scalar UDF.
- Add non-table bounded computation through a registered batch or stream
  operator factory.
- Add a transport through `calc-flow-connectors` and a capability descriptor;
  keep credentials behind secret references.
- Add local operational workflows through Studio without moving engine
  semantics into FastAPI or React.

The main design rule is simple: configuration is inert data, trusted code is
registered out of band, compilation freezes identity, and runners own every
stateful lifecycle.
