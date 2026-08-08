# Continuous Streaming 3.0 — M4 State and Window API Note

| Field             | Value                                                               |
| ----------------- | ------------------------------------------------------------------- |
| Status            | **Proposed — implementation contract**                              |
| Priority          | **P0 — companion to the M4 state/window specification**             |
| Baseline          | `main@6275599ca3bb872ab480b677f13bcd9698b144f0`                     |
| Milestone         | M4 — incremental state and final-only windows                       |
| Artifact slug     | `m4-state-window`                                                   |
| Controlling input | `specs/m4-state-window.md`, D1–D12 and AC-01–AC-58                  |
| Visibility        | additive M4 types only; public runner A6 remains unavailable        |
| Intended audience | Rust core, state, compiler, runtime, test, and benchmark owners     |

## 1. Purpose and compatibility boundary

This note translates the M4 specification into concrete Rust ownership,
module, type, and call-flow decisions. The specification controls semantics;
this note controls placement and implementation shape where multiple designs
could satisfy those semantics.

M4 is additive relative to the baseline:

- it adds state and window types approved by the original runtime API note;
- it does not remove or rename the current v2 `Checkpoint`, `CheckpointStore`,
  `FileCheckpointStore`, `MicroBatchRunner`, or public `StreamingRunner`;
- it does not expose the crate-private source-driven runner;
- it does not add Python, Studio, REST, OpenAPI, or project-schema fields.

The final M4 implementation commit may re-export the approved state and window
types from `lib.rs`. Earlier work-package commits keep incomplete types
crate-private. M5 performs the atomic v3 checkpoint-store and public runner
transition.

## 2. Module layout

The implementation uses this layout:

```text
crates/calc-flow/src/
  state/
    mod.rs          public re-exports and internal coordination values
    backend.rs      StateHandle and StateBackend
    manifest.rs     final v3 CheckpointManifest model and canonical validation
    segment.rs      inventory, segment operation, Arrow IPC envelope helpers
    local.rs        managed-root LocalStateBackend and lineage lease
  operator/
    window.rs       public window specification plus private execution state
  runtime/streaming/
    operator_task.rs
    metrics.rs
    runner.rs
```

M4 MUST NOT create `crates/calc-flow/src/checkpoint/model.rs`: Rust cannot use
that directory module while `src/checkpoint.rs` remains the baseline module.
M5 changes the checkpoint module layout and reuses `state::manifest` without a
second model.

## 3. State identity and manifest model

### 3.1 `StateHandle`

The public shape remains the M0-approved shape:

```rust
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateHandle {
    pub operator_id: String,
    pub epoch: Epoch,
    pub segment_id: String,
    pub relative_path: String,
    pub byte_len: u64,
    pub sha256: String,
}

impl StateHandle {
    pub fn new(
        operator_id: &str,
        epoch: Epoch,
        segment_id: &str,
        relative_path: &str,
        byte_len: u64,
        sha256: &str,
    ) -> Result<Self>;

    pub fn validate_for(
        &self,
        expected_operator: &str,
        expected_epoch: Epoch,
    ) -> Result<()>;
}
```

`Ord` is the canonical manifest order, not filesystem discovery order.
`relative_path` always names the committed subtree. Staging paths are backend
implementation details and never appear in a handle or manifest.

### 3.2 Manifest values

M4 implements the exact information model below. Connector-owned cursor and
pre-commit payloads remain strict bounded `JsonMap` values until their M6
connector types exist.

```rust
pub const MANIFEST_FORMAT_VERSION: u32 = 3;
pub const MAX_MANIFEST_DOCUMENT_BYTES: usize = 10 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryStatus {
    Final,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointManifest {
    format_version: u32,
    pipeline_name: String,
    pipeline_fingerprint: String,
    runtime_config_hash: String,
    epoch: Epoch,
    created_at: DateTime<Utc>,
    recovery_status: RecoveryStatus,
    sources: BTreeMap<String, SourceManifestEntry>,
    operators: BTreeMap<String, OperatorManifestEntry>,
    sinks: BTreeMap<String, SinkManifestEntry>,
    state_checksum: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceManifestEntry {
    pub cursor: Option<CursorManifestEntry>,
    pub identity_hash: String,
    pub sequence: u64,
    pub ended: bool,
    pub watermark_policy: SourceWatermarkManifestState,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CursorManifestEntry {
    pub order: String,
    pub payload: JsonMap,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorManifestEntry {
    pub progress: BTreeMap<String, OperatorIngressManifestEntry>,
    pub inline_metadata: JsonMap,
    pub segments: Vec<StateHandle>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorIngressManifestEntry {
    pub state: ManifestIngressState,
    pub watermark: Option<EventTime>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManifestIngressState {
    Active,
    Idle,
    Ended,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SinkManifestEntry {
    pub delivery: SinkDeliveryManifest,
    pub pre_commit: Option<JsonMap>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceWatermarkManifestState {
    SourceProvided {
        last_emitted_micros: Option<EventTime>,
        idle: bool,
    },
    BoundedOutOfOrderness {
        observed_max_micros: Option<EventTime>,
        last_emitted_micros: Option<EventTime>,
        idle: bool,
    },
    Disabled {
        idle: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionClass {
    Unbounded,
    Bounded,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SinkDeliveryManifest {
    Ordinary,
    EpochIdempotent {
        mechanism: String,
        retention: RetentionClass,
    },
    Transactional,
}
```

Construction is centralized so callers cannot provide a stale checksum:

```rust
impl CheckpointManifest {
    pub fn new(fields: CheckpointManifestFields) -> Result<Self>;
    pub fn validate(&self, expected: &ManifestExpectation<'_>) -> Result<()>;
    pub fn canonical_bytes(&self) -> Result<Vec<u8>>;
    pub fn recompute_state_checksum(&self) -> Result<String>;
}
```

`CheckpointManifestFields` omits `format_version` and `state_checksum`; `new`
fills both. Manifest fields are private and exposed through read-only accessors;
strict custom deserialization performs the same validation as `new` before a
value becomes observable. `ManifestExpectation` carries the pipeline name/fingerprint,
runtime-config hash, epoch, and exact stable source/operator/sink ID sets. M4
tests may construct it directly. M5 supplies it from the prepared job before
loading any segment or opening a source.

The checksum input is canonical JSON for exactly:

```json
{"operators": {...}, "sinks": {...}, "sources": {...}}
```

Keys use `BTreeMap` order and the three root fields stay in the shown order.
`created_at` is excluded from the state checksum but remains part of canonical
manifest bytes.

### 3.3 Strict JSON parsing

Manifest loading reuses the repository's bounded read, duplicate-key-aware
parse, depth validation, and canonical JSON helpers. Validation order is:

1. byte bound;
2. JSON object and duplicate-key validation;
3. format version;
4. strict typed deserialization with unknown-field rejection;
5. field and ID validation;
6. handle ownership/path validation;
7. expected-plan comparison;
8. state checksum.

No segment is opened before all eight steps succeed.

## 4. State backend and managed-root ownership

### 4.1 Backend contract and lineage session

```rust
#[async_trait]
pub trait StateBackend: Send + Sync {
    async fn open_lineage(&self, key: &StateLineageKey)
        -> Result<Box<dyn StateLineageBackend>>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateLineageKey {
    pipeline_name: String,
    pipeline_fingerprint: String,
}

impl StateLineageKey {
    pub fn new(pipeline_name: &str, pipeline_fingerprint: &str) -> Result<Self>;
    pub fn pipeline_name(&self) -> &str;
    pub fn pipeline_fingerprint(&self) -> &str;
}

#[async_trait]
pub trait StateLineageBackend: Send + Sync {
    fn identity_hash(&self) -> &str;
    async fn stage_segment(&self, handle: &StateHandle, bytes: &[u8]) -> Result<()>;
    async fn validate_segment(&self, handle: &StateHandle) -> Result<()>;
    async fn publish_segment(&self, handle: &StateHandle) -> Result<()>;
    async fn load_segment(&self, handle: &StateHandle) -> Result<Vec<u8>>;
    async fn collect_orphans(&self, retained: &[StateHandle]) -> Result<usize>;
}
```

This is a deliberate M4 correction to the original flat M0 sketch. Returning
a lineage-scoped session makes the cross-process lease unskippable for runtime
operations and gives the later `Arc<dyn StateBackend>` runner a backend-neutral
locking seam. `open_lineage` validates the key and acquires exclusivity before
returning; dropping the session releases the lease. A second session for the
same backend identity plus pipeline fingerprint fails with `Conflict`.

The methods have the following state machine:

```text
Absent --stage_segment--> Staged --validate_segment--> Validated
Validated --publish_segment--> Committed
Committed --load_segment--> Committed
```

Calling `publish_segment` for an absent or unvalidated handle fails. Repeating
`stage_segment` succeeds only when staged bytes have the exact expected length
and checksum. Repeating `publish_segment` succeeds only when the committed file
already matches the handle. These idempotent same-content cases support M5
retry; conflicting content always fails closed.

`load_segment` verifies length and checksum before returning bytes. The built-
in window restore then validates Arrow IPC and the expected state schema.

### 4.2 Local backend

```rust
#[derive(Clone, Debug)]
pub struct LocalStateBackend {
    root: Arc<ManagedStateRoot>,
}

impl LocalStateBackend {
    pub async fn new(root: impl AsRef<Path>) -> Result<Self>;
}

struct LocalStateLineageBackend {
    /* owns the cross-process advisory lock for its complete lifetime */
}
```

The backend creates and canonicalizes one root and then owns this layout:

```text
<root>/
  locks/<pipeline-sha256>.lock
  staging/<pipeline-sha256>/<epoch>/<operator-sha256>/<segment-sha256>.tmp
  committed/<pipeline-sha256>/<operator-sha256>/<epoch>-<segment-sha256>.arrow
```

The `.arrow` suffix describes built-in window state. A custom operator's opaque
bytes use `.segment`; the suffix is derived from validated operator metadata,
never from a caller path.

`LocalStateLineageBackend` is non-cloneable and owns the lease. Its internal
publication mutex serializes publication, retention, and compaction inside the
process. A portable advisory file lock supplies cross-process exclusivity.
Lock acquisition failure maps to `CalcFlowError::Conflict` before any staged
file is created. Because the workspace MSRV predates stable standard-library
file locking, WP2 may add one narrowly scoped, MSRV-compatible advisory-lock
dependency after `cargo audit` and `cargo deny` review; it must not add a
database or embedded KV dependency.

Every filesystem operation validates the canonical root and each managed
component with `symlink_metadata`. A symlink or unexpected file type fails.
The managed-root contract assumes external actors do not race mutations after
lease acquisition; M4 does not claim a hostile multi-user filesystem sandbox.

### 4.3 Private manifest commit harness

M4 adds a crate-private/test helper, not a public coordinator:

```rust
pub(crate) async fn commit_manifest_for_test(
    state: &dyn StateLineageBackend,
    manifest_root: &Path,
    manifest: &CheckpointManifest,
    staged: &BTreeMap<StateHandle, Vec<u8>>,
    fault: Option<CommitFaultPoint>,
) -> Result<()>;
```

It exists only to prove D3 crash outcomes and must not be used by the running
job in M4. M5 replaces it with coordinator-owned publication using the same
primitive ordering.

## 5. Segment inventory and compaction

### 5.1 Inventory values

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SegmentKind {
    Base,
    Delta,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SegmentDescriptor {
    pub kind: SegmentKind,
    pub state_layout_version: u32,
    pub schema_fingerprint: String,
    pub handle: StateHandle,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct StateInventory {
    pub segments: Vec<SegmentDescriptor>,
}
```

`StateInventory::validate` requires canonical `(epoch, segment_id)` order, at
most one base, no delta older than the base, unique handles/paths, one operator,
and one layout/schema fingerprint. It predicts the manifest contribution size
before publication.

### 5.2 Window segment rows

The built-in window segment is an Arrow IPC file with deterministic rows. The
logical columns are:

| Column                 | Arrow type                         | Meaning                                               |
| ---------------------- | ---------------------------------- | ----------------------------------------------------- |
| `_operation`           | `UInt8`                            | `0 = upsert`, `1 = tombstone`                         |
| `window_start`         | `Timestamp(Microsecond, "UTC")`    | exact key bound                                       |
| `window_end`           | `Timestamp(Microsecond, "UTC")`    | exact key bound                                       |
| `_stable_group_key`    | `LargeBinary`                      | deterministic D8 identity/order bytes                 |
| declared group columns | compiled logical types             | values used to reconstruct output                     |
| accumulator columns    | private compiled state types       | count/sum/min/max/avg intermediate values             |

Tombstone rows retain key columns and use null accumulator columns. Rows sort
by `(window_start, window_end, _stable_group_key, _operation)`. The stable key
is verified against the reconstructed declared group values during restore.

Accumulator columns use the aggregate declaration ordinal, never the user
output name, so path-like or duplicate physical names cannot enter the state
schema:

```text
_agg_0000_value
_agg_0000_count
_agg_0001_value
...
```

Count state is `UInt64`. Signed/unsigned/float sum state is respectively
`Int64`, `UInt64`, or `Float64`. Min/max state uses the compiled input logical
type. Average stores a `UInt64` count plus: signed integer sum as 16-byte
big-endian two's-complement `FixedSizeBinary(16)`; unsigned integer sum as
16-byte big-endian `FixedSizeBinary(16)`; or float sum as `Float64`. This exact
encoding makes widened integer averages portable without treating internal
accumulators as user-visible Decimal values.

The Arrow schema metadata carries:

- `calc_flow.state_layout_version`;
- `calc_flow.pipeline_fingerprint`;
- `calc_flow.operator_id`;
- `calc_flow.operator_configuration_hash`;
- `calc_flow.group_key_encoding = "g1"`.

Mismatch fails before any restored accumulator is installed.

### 5.3 Compaction call flow

```text
validated retained inventory
  -> blocking load and checksum verification
  -> Arrow IPC decode and per-segment validation
  -> ordered last-operation-wins fold into BTreeMap
  -> deterministic base Arrow IPC encode
  -> stage/validate/publish new immutable base
  -> candidate manifest references new base
  -> manifest publication (test harness in M4, coordinator in M5)
  -> later reachability collection
```

Compaction never deletes directly. It returns a replacement inventory; only a
later reachability pass may collect old handles after retained manifests stop
referencing them.

## 6. Window public specification

### 6.1 Public types

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregateFunction {
    Count,
    Sum,
    Min,
    Max,
    Avg,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AggregateSpec {
    pub function: AggregateFunction,
    pub column: String,
    pub output: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WindowGeometry {
    Tumbling { size_micros: u64 },
    Hopping { size_micros: u64, slide_micros: u64 },
}

pub const MAX_WINDOW_OVERLAP: u64 = 1_024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowSpec {
    pub event_time_column: String,
    pub group_by: Vec<String>,
    pub geometry: WindowGeometry,
    pub aggregates: Vec<AggregateSpec>,
}

pub struct WindowAggregateOperator {
    metadata: WindowOperatorMetadata,
    compiled: CompiledWindowSpec,
    state: WindowState,
}
```

Unlike the original illustrative API, serialized geometry stores exact
microseconds rather than `Duration`. Constructors still accept `Duration` and
reject non-microsecond values before constructing the serializable form:

```rust
impl WindowSpec {
    pub fn tumbling(event_time_column: &str, size: Duration) -> Result<Self>;
    pub fn hopping(
        event_time_column: &str,
        size: Duration,
        slide: Duration,
    ) -> Result<Self>;
    pub fn group_by<I, S>(self, columns: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>;
    pub fn aggregate(
        self,
        function: AggregateFunction,
        column: &str,
        output: &str,
    ) -> Result<Self>;
}

impl WindowAggregateOperator {
    pub fn new(name: &str, input_schema: SchemaRef, spec: WindowSpec) -> Result<Self>;
}
```

`new` fully validates and compiles the schema. No source is needed. It creates
one required `input` table port and one required `output` table port.

### 6.2 Compiler integration

`NodeOperator` adds `Window(WindowAggregateOperator)`. `CompiledStreamOperator`
adds the corresponding owned variant. Batch compilation explicitly rejects
`Window`, while stream validation accepts it.

`CompiledStreamOperator` also adds complete `checkpoint` and `restore`
dispatch for every variant. The current baseline dispatches only reset, data,
watermark, and end; leaving checkpoint/restore off this owned enum would make
the public trait methods unreachable from M5. M4 adds the dispatch seam and
unit tests but does not call it from the barrier-rejecting operator task.

`WindowAggregateOperator::configuration()` serializes every semantic field:

```json
{
  "kind": "window_aggregate",
  "state_layout_version": 1,
  "event_time_column": "event_time",
  "geometry": {"kind": "hopping", "size_micros": 60000000, "slide_micros": 10000000},
  "group_by": ["symbol"],
  "aggregates": [{"function": "sum", "column": "quantity", "output": "total_quantity"}],
  "group_key_encoding": "g1",
  "max_group_key_bytes": 65536,
  "null_event_time_policy": "drop"
}
```

This existing configuration path automatically feeds `graph_fingerprint`.
Runtime-only compaction thresholds do not enter the semantic fingerprint.

## 7. Window execution ownership

### 7.1 Compiled state

```rust
#[derive(Clone)]
struct CompiledWindowSpec {
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    event_time_index: usize,
    group_columns: Vec<CompiledGroupColumn>,
    aggregates: Vec<CompiledAggregate>,
    geometry: CompiledWindowGeometry,
    configuration_hash: String,
}

#[derive(Default)]
struct WindowState {
    accumulators: BTreeMap<WindowKey, AccumulatorRow>,
    dirty: BTreeSet<WindowKey>,
    emitted_pending_snapshot: BTreeSet<WindowKey>,
    restored_inventory: StateInventory,
    last_input_watermark: Option<EventTime>,
    next_output_sequence: u64,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct WindowKey {
    start: EventTime,
    end: EventTime,
    stable_group_key: Vec<u8>,
}
```

Mutation is confined to `WindowState`, which is owned by one operator task.
Input batches are read-only. Each handler builds a scratch delta and scratch
metric delta; it applies them only after all rows and aggregates succeed.

### 7.2 Data handler flow

```text
validate exact input schema
  -> read current WM_in from StreamOperatorContext
  -> for each RecordBatch and row in FIFO order
       -> null event time: scratch null counters
       -> checked EventTime conversion
       -> checked concrete assignment expansion
       -> stable group-key encoding and 64-KiB check
       -> partition assignments into closed and open
       -> scratch late metrics for closed assignments
       -> scratch aggregate updates for open assignments
  -> atomically install scratch accumulator/dirty changes
  -> atomically publish scratch metric changes
```

The implementation may use Arrow downcasts and scalar helpers, but it must not
mutate input arrays or rely on DataFusion private aggregate state.

### 7.3 Watermark and EOF flow

On `on_watermark(w)`:

1. select keys with `end <= w` that are not already in
   `emitted_pending_snapshot`;
2. finalize aggregates into Arrow arrays in canonical key order;
3. emit bounded `Batch` values through `StreamCollector`;
4. only after all emissions succeed, add the keys to
   `emitted_pending_snapshot`;
5. return; the runtime then forwards `Watermark(w)`.

The task passes its minimum effective output-edge budget through a
crate-private context accessor. If one close event exceeds the budget, the
operator chunks output into deterministic contiguous key ranges. Chunking does
not alter row order. One output row that alone exceeds the byte budget fails
before enqueue with the existing oversize-message error.

Each emitted chunk uses `BatchMetadata { source: operator_id, sequence:
next_output_sequence, attributes: {} }`. The sequence starts at zero for a new
lineage, increments with checked arithmetic per successfully emitted chunk,
and is captured/restored with operator state. Failure after a partial chunk
send terminates the job; recovery from the previous manifest reuses the same
deterministic sequence range.

`on_end` uses the same path with all remaining non-empty keys and is
idempotent within one operator instance.

### 7.4 Snapshot and restore flow

`checkpoint(epoch)`:

1. freezes the current dirty and emitted-pending sets;
2. returns bounded inline metadata containing layout/configuration hashes,
   last input watermark, metric totals, and segment descriptors only;
3. hands off already prepared Arrow IPC delta bytes under deterministic
   segment IDs;
4. after successful local capture, replaces dirty state with a fresh owned set
   and converts emitted-pending keys into tombstone intent for the captured
   delta.

The operation is allowed to mutate live operator bookkeeping. If later durable
publication fails, M5 terminates the job; it never continues from the mutated
state under an uncommitted epoch.

`restore(snapshot)` validates every inline field and segment before replacing
the complete `WindowState`. It never merges with existing state. Repeating the
same restore is idempotent. `reset()` clears every accumulator, dirty/emitted
set, inventory, watermark, and metric value.

M4 integration tests feed snapshot bytes through `LocalStateBackend` and a
validated `StateInventory`; the running M4 operator task does not execute a
barrier or durable restore.

## 8. Persistent metric wiring

The baseline `StreamOperatorContext` creates an isolated recorder on every
construction. M4 changes the internal constructor to receive a shared sink:

```rust
pub struct StreamOperatorContext<'a> {
    job: &'a StreamJobContext,
    operator_id: &'a str,
    input_watermark: Option<EventTime>,
    output_budget: EdgeBudget,
    late_metrics: Arc<dyn LateMetricSink>,
}

pub(crate) trait LateMetricSink: Send + Sync {
    fn record(&self, delta: LateMetricDelta) -> Result<()>;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct LateMetricDelta {
    late_rows: u64,
    affected_batches: u64,
    max_lateness_micros: Option<u64>,
    null_event_time_rows: u64,
    null_event_time_batches: u64,
}
```

The existing public `StreamOperatorContext::new(job, operator_id,
input_watermark)` remains source-compatible and installs the default budget
plus an isolated recorder for direct operator callers. Runtime code uses a new
`pub(crate) for_task(...)` constructor with the effective output budget and
task-owned metric sink. Thus persistent runtime ownership does not break the
already public helper constructor.

`OperatorProgress` owns the live metric state and `MetricsRecorder` mirrors it
into `OperatorMetrics` for one stable node ID. Context creation in data,
watermark, and EOF handlers receives the same task-owned sink.

To preserve the existing public method, `record_late_rows` converts its inputs
to `LateMetricDelta`. A new crate-private method records null-time deltas.
Window handlers calculate one combined delta and call the sink only after the
state transaction succeeds.

`OperatorProgressSnapshot`, `OperatorStatus`, and `OperatorMetrics` add the
five D11 values. They remain crate-private until the post-M5 status API gate.

## 9. Error mapping

M4 reuses existing public error families unless a stable recovery classifier
requires a new variant later in M5:

| Failure                                                  | Error family                              |
| -------------------------------------------------------- | ----------------------------------------- |
| invalid window/aggregate/handle argument                 | `InvalidArgument` with exact field path   |
| invalid graph schema or unsupported combination          | `Compile`                                 |
| invalid manifest/segment/state layout/checksum           | `Format` or `CheckpointMismatch`          |
| state root lock conflict                                 | `Conflict`                                |
| filesystem operation                                     | `Io` with managed path and source         |
| aggregate/window row failure during execution            | `Operator` with node ID and field context |
| impossible inventory/order/counter invariant             | `Internal`                                |

Errors never contain row payloads, group-key bytes, cursor payloads, or secret
values. Filesystem errors may contain only the validated managed path.

## 10. Test placement and named evidence

Pure private tests remain beside their modules:

- `state/manifest.rs`: canonical parse/validation/checksum tests;
- `state/segment.rs`: inventory, Arrow IPC, fold, and compaction properties;
- `state/local.rs`: managed path and operation-state tests using `TempDir`;
- `operator/window.rs`: scalar, encoding, assignment, and accumulator units;
- `runtime/streaming/operator_task.rs`: shared context metrics and
  handler-before-forward ordering.

Public and cross-module behavior lives in focused integration tests:

```text
crates/calc-flow/tests/state_backend.rs
crates/calc-flow/tests/local_state.rs
crates/calc-flow/tests/window_compile.rs
crates/calc-flow/tests/window_tumbling.rs
crates/calc-flow/tests/window_hopping.rs
crates/calc-flow/tests/window_properties.rs
```

Every AC-01–AC-58 receives one named test or an evidence-row explanation.
Tests use paused Tokio time and observable state, never sleeps. Property tests
record their seed on failure.

## 11. Benchmark and soak interfaces

Criterion cases are separate so state costs are attributable:

```text
state_incremental_write/<dirty_keys>
state_full_restore/<live_keys>
state_compaction/<segments>/<live_keys>
window_tumbling/<rows>/<groups>
window_hopping/<rows>/<groups>/<overlap>
```

Each new case has a matching baseline/control where comparison is meaningful.
The implementation evidence records sample count, confidence interval, and
same-machine base/head SHAs. A single point estimate cannot approve the 5%
gate.

The M4 soak extends the existing exact 1,200-second two-source slow-sink
harness with a window node and enough distinct keys to exercise dirty flush
and bounded state. It retains:

- ten-second cadence;
- exactly 120 samples;
- 300-second/30-sample warm-up for RSS trend;
- two sources and a deliberately slow sink;
- queue, task, receipt, timer, state-segment, and live-key terminal checks;
- zero missing logical input and deterministic expected final aggregate.

The soak does not invoke M5 barriers. It may trigger the M4 private state
snapshot/compaction harness at deterministic logical boundaries.

## 12. Implementation commit sequence

The implementation PR should use these narrow imperative commits:

1. `Add M4 manifest and state handle model`
2. `Implement local immutable state backend`
3. `Add retained-state compaction`
4. `Add window aggregate compile contract`
5. `Implement tumbling window aggregation`
6. `Implement hopping and late assignment handling`
7. `Persist window state and metrics`
8. `Record M4 verification evidence`

Each commit must build and pass its focused tests. Public re-exports land only
when their complete behavior and rustdoc examples are present. The final
implementation diff is reviewed against this API note and the controlling M4
specification, not against illustrative historical file lists.
