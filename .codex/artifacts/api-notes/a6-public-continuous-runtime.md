# Public Continuous Streaming A6 - API Note

Artifact slug: `a6-public-continuous-runtime`.

## Control and Status

- Controlling input:
  `.codex/artifacts/specs/a6-public-continuous-runtime.md` on PR #113 source
  head `5cea87097f37ba579d3985a96e18d856f41c14f9`, SHA-256
  `d2730be9dd135ad95e57db052b87e515ec722d468b4c4bc9b34dab26b9735948`.
- Baseline: `main@aa3bbf0b40aef74898a59b6d0d0028c59a2d6993`, including
  merged PR #108 (`d4a85bcecbac0eefa8ff1c5bf75a5b54fe23a447`) and PR #110
  (`aa3bbf0b40aef74898a59b6d0d0028c59a2d6993`).
- Status: proposed public surface, ready for a new independent critique.
- Superseded input: both earlier specs, the API note at
  `a4d7441f760cbea33a1657923bc0378adb610fb7c452ccefe3e287e60b2fcec3`,
  both Block critiques, the old-baseline API note at
  `f5305d1786b5b388397c544317beb2ce28ae4cb489988fe0a072ab9600ff31e7`,
  and its zero-blocker critique are historical review inputs only. They do not
  control any signature or behavior below.
- Scope of this artifact: public Rust, PyO3, Python, Studio persistence wiring,
  errors, examples, and the atomic repository cutover. This note changes no
  runtime behavior, public export, generated file, or test.

The specification controls behavior. This note fixes the exact public names,
signatures, defaults, projections, messages, and validation targets needed to
implement that behavior without another design choice.

## Audiences

- Rust users: one source-driven runner, one owning job, async connector traits,
  one single-root local state/manifest namespace owner, one safe streaming
  error boundary, exact replay capability, and data-only observation types.
- Python users: a native-backed `StreamExecutionPlan`, async-only connector
  lifecycle, one-shot runner/job wrappers, guarded blocking twins, and a
  synchronous `status()` that is safe inside an event loop.
- Studio clients: no REST or OpenAPI change. Existing `/api/v2` checkpoint
  summaries and deletion remain stable while their backend persistence becomes
  Studio-private.

## Surface Today

At the baseline, `crates/calc-flow/src/lib.rs` exports the following legacy
surface:

```rust
pub use checkpoint::{
    CHECKPOINT_FORMAT_VERSION, Checkpoint, CheckpointStore,
    FileCheckpointStore, MAX_CHECKPOINT_DOCUMENT_BYTES,
};
pub use io::{BatchingSource, Sink, Source, SourceItem};
pub use runtime::{MicroBatchRunner, SinkRouter, StreamingRunner};
```

The public `StreamingRunner` is push-based over `BatchExecutionPlan`; it has
`step`, `reset`, and `plan_snapshot`. Python projects only expose
`ExecutionPlan`, `_MicroBatchRunner`, `_StreamingRunner`, and
`_FileCheckpointStore`. Python source and sink callbacks may be synchronous,
`status()` does not exist, and `StreamExecutionPlan` is not bound.

The M5 source-driven runner, job, source/sink traits, capability proof,
checkpoint coordinator, status, reaper, and manifest transaction are
crate-private. PRs #108/#110 add manual FIFO coordination, source-bound
cursors, once-sampled source descriptors, and durable capability identity.
Important current-main details that this public API must preserve or tighten
are:

- `ContinuousRunner::start(&self)` permits a reusable runner core instead of a
  consuming public start;
- `CheckpointRuntimeSpec` accepts a state backend and manifest root but only
  the state lineage is leased;
- the existing `CalcFlowError` variants can retain paths, raw I/O sources,
  panic payloads, and connector text across `Display`, `Debug`, and
  `source()`;
- `ManifestPublication::Installed { parent_synced: false, .. }` can already be
  selected, so it is not an absent publication;
- #108 resolves manual waiters only at `CheckpointAdvancement::Completed`,
  after durable publication and every expected sink commit acknowledgement;
- #110 binds configured, restored, and emitted cursors to the external source
  binding ID, samples schema/watermark/replay/delivery/bounds once before
  open, and fingerprints that descriptor into durable source identity;
- the private compatibility fallback can still infer exact positioning from
  `SourceCapabilities.replayable: bool`; Public A6 must remove that fallback;
- `TransactionalStreamSink::recover` receives the complete manifest;
- private status includes cursors, task details, and an owner-drop counter that
  cannot be read after the only public owner has been dropped.

## Proposed Surface

### Public replacement inventory

The A6 implementation is one breaking replacement. It has no aliases,
overloads, or compatibility wrappers for removed v2 runner/checkpoint names.

| Surface                   | Remove                                                                                                               | Add or retain                                                                                                                       |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Rust checkpoint exports   | `Checkpoint`, `CHECKPOINT_FORMAT_VERSION`, `MAX_CHECKPOINT_DOCUMENT_BYTES`, `CheckpointStore`, `FileCheckpointStore` | Keep manifest v3 and standalone state APIs; add single-root local `ManagedCheckpointRuntime`                                        |
| Rust connector exports    | `Source`, `SourceItem`, `BatchingSource`, `Sink`, `SinkRouter`                                                       | identity-bound `Cursor`, frozen `SourceCapabilities`, `SourceEvent`, `StreamSource`, bindings, async sink traits, scoped recovery   |
| Rust runner exports       | `MicroBatchRunner`; push runner methods `step`, `reset`, `plan_snapshot`                                             | one-shot `StreamingRunner::start(self)` and non-cloneable `StreamingJob`                                                            |
| PyO3 registration         | `_MicroBatchRunner`, legacy `_StreamingRunner`, `_FileCheckpointStore`                                               | `StreamExecutionPlan`, single-directory `_ManagedCheckpointRuntime`, replacement runner/job, safe error projections                 |
| Python public API         | `Source`, `MicroBatchRunner`, public `FileCheckpointStore`, push runner methods                                      | async continuous protocols/bindings, single-directory checkpoint owner, stream plan, runner/job, status/outcome/config/error types  |
| Studio public contract    | direct import/construction of public `FileCheckpointStore`                                                           | private v2 checkpoint-document persistence; unchanged `create_app()` and unchanged `/api/v2` OpenAPI                                |
| Repository consumers      | legacy examples, benchmarks, active docs, and release assertions                                                     | continuous recovery examples, public-runner benchmarks, A6 docs, removed-symbol and package/startup smoke assertions                |

`BatchExecutionPlan` and its Python `ExecutionPlan` wrapper remain available.
`StreamExecutionPlan` keeps that exact name in Rust, PyO3, and Python; it is
never renamed to the ambiguous `ExecutionPlan`.

### Rust plan and configuration

Existing Rust compilation signatures remain the stream plan entry point:

```rust
impl PipelineBuilder {
    pub fn compile_stream(
        self,
        udfs: &UdfRegistrySnapshot,
        requirements: &StreamRequirements,
    ) -> Result<StreamExecutionPlan>;
}

#[derive(Clone, Debug, Default)]
pub struct StreamRequirements {
    pub delivery: BTreeMap<String, DeliveryGuarantee>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryGuarantee {
    AtLeastOnce,
    ExactlyOnce,
}

#[derive(Clone, Copy, Debug)]
pub struct StreamRuntimeConfig {
    pub checkpoint_interval: Duration, // default 60 seconds
    pub checkpoint_timeout: Duration,  // default 10 minutes
    pub edge_budget: EdgeBudget,       // default 10_000 rows / 64 MiB
    pub retained_epochs: usize,        // default 2; must be positive
}
```

Outputs absent from `StreamRequirements.delivery` request at-least-once.
Durations must be exact non-negative microseconds; interval, timeout, edge
limits, and retention are positive where the baseline validator requires it.
Runtime tuning contributes to `runtime_config_changed` but never changes the
semantic lineage identity or rejects recovery.

### Rust source data and exact replay positioning

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Cursor {
    source_id: Option<String>,
    order: Vec<u8>,
    payload: JsonMap,
}

impl Cursor {
    /// Constructs a cursor already owned by one external source binding ID.
    pub fn new(
        source_id: impl Into<String>,
        order: Vec<u8>,
        payload: JsonMap,
    ) -> Result<Self>;
    /// Constructs a cursor whose owner is assigned by runner admission.
    pub fn unbound(order: Vec<u8>, payload: JsonMap) -> Result<Self>;
    pub fn source_id(&self) -> Option<&str>;
    pub fn order(&self) -> &[u8];
    pub fn payload(&self) -> &JsonMap;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayPositioning {
    ExactPauseReportAndSeek,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeWatermarkCapability {
    NeverEmits,
    EmitsNative,
    RuntimeToggleable,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceDeliveryCapability {
    Lossless,
    Lossy,
}

#[derive(Clone, Debug)]
pub enum SourceSchema {
    Exact(SchemaRef),
    DynamicOrUnknown,
}

#[derive(Clone, Debug)]
pub struct SourceCapabilities {
    pub replay_positioning: ReplayPositioning,
    pub delivery: SourceDeliveryCapability,
    pub max_batch_rows: usize,
    pub max_batch_bytes: usize,
    pub schema: SourceSchema,
    pub native_watermarks: NativeWatermarkCapability,
}

#[derive(Clone, Debug)]
pub enum SourceEvent {
    Data { batch: Batch, cursor: Cursor },
    Watermark(EventTime),
    Idle,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WatermarkPolicy {
    SourceProvided,
    BoundedOutOfOrderness {
        event_time_column: String,
        max_out_of_orderness: Duration,
        emit_interval: Duration,
        idle_timeout: Option<Duration>,
    },
    Disabled {
        idle_timeout: Option<Duration>,
    },
}

impl Default for WatermarkPolicy {
    fn default() -> Self { Self::SourceProvided }
}

#[async_trait]
pub trait StreamSource: Send {
    fn capabilities(&self) -> SourceCapabilities;
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()>;
    async fn next(&mut self) -> Result<Option<SourceEvent>>;
    async fn close(&mut self) -> Result<()>;
}

pub struct SourceBinding { /* owned connector and immutable policy */ }

impl SourceBinding {
    pub fn new(source: impl StreamSource + 'static) -> Self;
    #[must_use]
    pub fn with_watermark_policy(self, policy: WatermarkPolicy) -> Self;
}
```

`Cursor::new` creates an explicitly owned cursor and `Cursor::unbound` creates
an unbound cursor, matching the post-#110 private value shape. Both validate a
non-empty order key of at most 16 KiB and a bounded strict JSON payload; `new`
additionally validates the portable source ID. Runner preflight/admission
binds an unbound configured or emitted cursor to the exact external input key.
A cursor already owned by a different source fails before connector open when
configured, or before
progress/data admission when emitted. Recovery reconstructs an owned cursor
from the enclosing manifest source participant; the manifest entry does not
duplicate the ID. A cursor passed to `StreamSource::open` is therefore always
owned and `source_id()` is `Some`.

The lowercase-hex manifest order encoding remains v3. Cursor order/payload is
available only to the connector and managed manifest; source ownership is
used only for validation. Order, payload, and restored offsets are never
projected into status, errors, debug output, logs, metrics, or public
diagnostics.

`ReplayPositioning` replaces every public `replayable` boolean. Capabilities
are sampled exactly once in whole-job preflight before `open`. The runtime
combines `SourceCapabilities` with the normalized binding watermark policy,
freezes schema, native-watermark capability, replay positioning, delivery,
and positive row/byte bounds, and fingerprints that complete descriptor into
the prepared-job and durable per-source identities. Recovery rejects drift in
any field. `ExactPauseReportAndSeek` plus `Lossless` is a declared protocol
and connector conformance obligation: a source that ignores the owned cursor
passed to `open` fails conformance and cannot qualify any exactly-once output.
`SourceDeliveryCapability::Lossy` can qualify only at-least-once outputs.

The source ID is not stored in `SourceBinding`; it is the exact external input
key in the runner's `BTreeMap`. Runtime-owned source sequence numbers begin at
zero on a new lineage and restore from the selected manifest. A source never
supplies a sequence or constructs a barrier.

### Rust sink bindings and scoped recovery

```rust
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SinkDelivery {
    Ordinary,
    EpochIdempotent {
        mechanism: String,
        retention: RetentionClass,
    },
    Transactional,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SinkRecovery {
    epoch: Epoch,
    terminal: bool,
    delivery: SinkDelivery,
    pre_commit: JsonMap,
}

impl SinkRecovery {
    pub const fn epoch(&self) -> Epoch;
    pub const fn terminal(&self) -> bool;
    pub const fn delivery(&self) -> &SinkDelivery;
    pub const fn pre_commit(&self) -> &JsonMap;
}

#[async_trait]
pub trait StreamSink: Send {
    async fn open(&mut self) -> Result<()>;
    async fn write(&mut self, batch: &Batch) -> Result<()>;
    async fn close(&mut self) -> Result<()>;
}

#[async_trait]
pub trait TransactionalStreamSink: Send {
    async fn open(&mut self) -> Result<()>;
    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()>;
    async fn write(&mut self, batch: &Batch) -> Result<()>;
    async fn pre_commit(&mut self, epoch: Epoch) -> Result<JsonMap>;
    async fn commit(&mut self, epoch: Epoch, pre_commit: &JsonMap) -> Result<()>;
    async fn abort(
        &mut self,
        epoch: Epoch,
        pre_commit: Option<&JsonMap>,
    ) -> Result<()>;
    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()>;
    async fn close(&mut self) -> Result<()>;
}

pub struct SinkBinding { /* stable sink ID, connector, frozen delivery */ }

impl SinkBinding {
    pub fn ordinary(
        sink_id: &str,
        sink: impl StreamSink + 'static,
    ) -> Result<Self>;

    pub fn transactional(
        sink_id: &str,
        sink: impl TransactionalStreamSink + 'static,
    ) -> Result<Self>;

    pub fn epoch_idempotent(
        sink_id: &str,
        sink: impl TransactionalStreamSink + 'static,
        mechanism: &str,
        retention: RetentionClass,
    ) -> Result<Self>;

    pub fn sink_id(&self) -> &str;
    pub fn delivery(&self) -> &SinkDelivery;
}
```

Transactional capability is proven by the complete trait, never by a boolean.
Epoch-idempotent bindings use that same complete lifecycle and add portable
mechanism/retention evidence. Bounded retention never qualifies as unqualified
exactly-once.

`SinkRecovery` has no public constructor. The engine creates a fresh value per
sink from the selected manifest. It contains exactly the selected epoch,
terminal flag, that sink's frozen delivery evidence, and that sink's own
pre-commit map. It does not contain a sink ID, another sink's evidence, source
cursor/progress, operator state/handles/checksums, manifest bytes/path, or
managed roots. Connector mutation is impossible because all fields are
private and accessors borrow engine-owned defensive values.

The runner's outer sink map is keyed by exact plan output ID. Each value is a
stable ordered `Vec<SinkBinding>`. Sink IDs are globally unique across the
whole job, not merely within one output.

### Complete state/manifest namespace owner

```rust
pub struct ManagedCheckpointRuntime {
    /* immutable lexical local root; non-Clone */
}

impl ManagedCheckpointRuntime {
    /// Captures one local managed root. Directory creation,
    /// canonicalization, child derivation, and locking occur in `start`.
    pub fn new(managed_root: impl Into<PathBuf>) -> Result<Self>;
}
```

There is no `load`, `save`, `publish`, `delete`, `ignore_existing`, or
`reset_lineage` method. There is also no `local` alias, two-root overload,
backend-taking constructor, `From<LocalStateBackend>`, builder escape hatch,
or accessor for an internally created backend. The standalone public
`StateBackend` and `LocalStateBackend` APIs remain available outside this A6
owner but cannot be supplied to it.

`new` performs only `PathBuf` conversion, empty-path rejection, and immutable
lexical capture. It creates no directory, canonicalizes no path, and acquires
no lease. Async start creates and canonicalizes the managed root `R`, derives
the fixed non-overlapping children `R/state` and `R/manifests`, constructs the
`LocalStateBackend` internally for `R/state`, and binds canonical `R`, pipeline
name, and semantic fingerprint into the managed namespace identity. Callers
can name neither child independently.

Before manifest scan, connector open, state mutation, or task spawn, start
acquires both:

1. one cross-process exclusive lease covering the complete canonical managed
   root `R`; and
2. the engine-created local backend's
   `StateBackend::open_lineage(StateLineageKey)` lease under `R/state`.

If either acquisition fails, start releases everything already acquired
before returning the safe conflict. Two jobs using the same canonical root
conflict before lifecycle work, including distinct lexical paths, symlink
aliases, and separate processes. Equal, ancestor, and descendant state/
manifest roots are impossible by construction because only the fixed sibling
children exist. Both the complete-root and lineage leases remain owned through
terminal connector cleanup and are released only after all conforming work has
settled or the reaper completes.

### One-shot runner and sole owning job

```rust
pub struct StreamingRunner { /* non-Clone, unstarted owner */ }

impl StreamingRunner {
    pub fn new(
        plan: StreamExecutionPlan,
        sources: BTreeMap<String, SourceBinding>,
        sinks: BTreeMap<String, Vec<SinkBinding>>,
        checkpoints: ManagedCheckpointRuntime,
    ) -> Result<Self>;

    #[must_use]
    pub fn with_runtime_config(
        self,
        config: StreamRuntimeConfig,
    ) -> Result<Self>;

    pub async fn start(self) -> Result<StreamingJob>;
}

pub struct StreamingJob { /* non-Clone, sole lifecycle owner */ }

impl StreamingJob {
    pub fn id(&self) -> u64;
    pub fn status(&self) -> JobStatus;
    pub async fn trigger_checkpoint(&self) -> Result<Epoch>;
    pub async fn shutdown(&self) -> JobOutcome;
    pub async fn cancel(&self) -> JobOutcome;
    pub async fn wait(&self) -> JobOutcome;
}
```

`StreamingRunner::new` is pure: it consumes and defensively owns the exact
plan/binding/checkpoint values, validates data-only shapes, and applies the
default runtime config. It performs no filesystem I/O, lease acquisition,
connector call, task spawn, or state mutation. The functional
`with_runtime_config` step validates and returns a new owner.

`start(self)` consumes the Rust runner and can publish at most one job. It
returns only after complete pure preflight, complete-root and lineage leases,
all-candidate manifest selection, reset/restore, sink-scoped recovery, async
connector open, task registration, and the live-data gate are ready. Before
that point there is no source poll, sink write, or operator data callback.
Dropping a pending start requests cancellation, settles every begun conforming
connector/task, releases both leases, and never makes a job observable.

The returned `StreamingJob` is the only public owner. It transitively owns all
tasks, connector values, queue endpoints, cancellation state, state/manifest
data, complete-root/lineage leases, manifest transaction, and cleanup/reaper
responsibility. It is not `Clone`. Method futures may share observer state
without creating another owner.

`wait`, `shutdown`, and `cancel` observe the same immutable terminal outcome.
Dropping a wait future detaches only that observer. Once shutdown/cancel has
recorded its transition, dropping its future does not revoke it; another call
reattaches. Dropping a manual-checkpoint future after acceptance does not
dequeue the request.

Dropping the owning job is non-blocking. It records abandonment, requests
cancellation, and transfers complete cleanup to a self-contained runtime
reaper. It emits exactly one safe warning event named
`abandoned_streaming_job`. No `JobStatus` field named
`abandoned_job_drops` exists; process-internal accounting is not public job
status because the only job handle has already gone.

### Rust lifecycle, outcome, status, and error values

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Running,
    Draining,
    Completed,
    Cancelled,
    Failed,
    RecoveryRequired,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalCause {
    NaturalEnd,
    GracefulShutdown,
    ExplicitCancel,
    DeadlineExceeded,
    Failure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingErrorCategory {
    Validation,
    Compile,
    Conflict,
    Cancelled,
    CheckpointTimeout,
    CheckpointMismatch,
    CheckpointPublicationUnknown,
    Io,
    Operator,
    Connector,
    TaskPanicked,
    Internal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentKind {
    Job,
    Edge,
    Source,
    Operator,
    Sink,
    Checkpoint,
}

#[derive(Clone, Debug, Eq, Error, PartialEq, Serialize)]
#[error("{message}")]
pub struct StreamingError {
    category: StreamingErrorCategory,
    message: String,
    job_id: Option<u64>,
    epoch: Option<Epoch>,
    checkpoint_phase: Option<CheckpointPhase>,
    component_kind: Option<ComponentKind>,
    component_id: Option<String>,
    diagnostic_id: Option<u64>,
    position: u32,
}

impl StreamingError {
    pub const fn category(&self) -> StreamingErrorCategory;
    pub fn message(&self) -> &str;
    pub const fn job_id(&self) -> Option<u64>;
    pub const fn epoch(&self) -> Option<Epoch>;
    pub const fn checkpoint_phase(&self) -> Option<CheckpointPhase>;
    pub const fn component_kind(&self) -> Option<ComponentKind>;
    pub fn component_id(&self) -> Option<&str>;
    pub const fn diagnostic_id(&self) -> Option<u64>;
    pub const fn position(&self) -> u32;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JobOutcome {
    pub state: JobState,
    pub cause: TerminalCause,
    pub completed_epoch: Option<Epoch>,
    pub errors: Vec<StreamingError>,
}
```

`StreamingError` has no public constructor, mutable field, `Deserialize`
implementation, raw-cause field, generic metadata map, or extension payload.
Only the crate-private A6 conversion seam can create one. Its derived
`Display` is exactly the safe engine-authored `message`; normal and alternate
`Debug` contain only the fields above; its `std::error::Error::source()` is
`None` because no field is marked as a source. Serialization is allowlist-only
and uses the snake-case enum projections shown above.

The existing non-exhaustive outer error gains exactly this variant:

```rust
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CalcFlowError {
    // Existing batch, project, store, and provider variants remain.
    #[error("{0}")]
    Streaming(#[from] StreamingError),
}
```

The outer `Display` is the same safe message. Its `Debug` contains only the
safe inner value, and its sole `source()` is that `StreamingError`, whose own
source is `None`. No public `From` implementation converts an arbitrary
`CalcFlowError`, `std::io::Error`, panic payload, connector value, or Python
exception into `StreamingError`; all raw-to-safe mapping stays crate-private.

Every `Result<T>` on `ManagedCheckpointRuntime`, `StreamingRunner`, and
`StreamingJob` below uses the repository alias `calc_flow::Result<T>` and has
one failure shape: `Err(CalcFlowError::Streaming(error))`. This applies to
`ManagedCheckpointRuntime::new`, `StreamingRunner::new`,
`with_runtime_config`, `start`, and `StreamingJob::trigger_checkpoint`.
`id` and `status` are infallible; `shutdown`, `cancel`, and `wait` always
return the immutable `JobOutcome`, whose `errors` already contain only safe
`StreamingError` values. Connector trait methods may receive existing core
errors from application implementations, but the runtime converts and drops
their raw values exactly once before any runner/job observer, outcome, log,
diagnostic, metric, or PyO3 boundary.

The crate-private conversion seam uses this exhaustive mapping and never
falls back to formatting the source error:

| Internal failure context                               | `StreamingErrorCategory`        |
| ------------------------------------------------------ | ------------------------------- |
| A6 argument, route, or capability validation           | `Validation`                    |
| stream-plan compile or frozen-plan incompatibility     | `Compile`                       |
| complete managed-root or lineage lease conflict        | `Conflict`                      |
| explicit or cooperative lifecycle cancellation         | `Cancelled`                     |
| checkpoint deadline before the D5 completed point      | `CheckpointTimeout`             |
| candidate, identity, checksum, or manifest mismatch    | `CheckpointMismatch`            |
| installed manifest with unknown parent-sync durability | `CheckpointPublicationUnknown`  |
| engine-owned filesystem or state/manifest I/O          | `Io`                            |
| operator callback or execution failure                 | `Operator`                      |
| source or sink open/lifecycle callback failure         | `Connector`                     |
| contained runtime task panic                           | `TaskPanicked`                  |
| unclassified engine invariant failure                  | `Internal`                      |

The last four `JobState` variants are terminal and immutable. Failure/panic,
explicit cancel, deadline, then graceful/natural completion is the terminal
precedence when triggers are observed at one decision point. `JobOutcome`
always has a terminal state. Its error vector is deterministically ordered by
`position`; it carries safe engine-generated values only.

The status allowlist is represented by these exact public fields:

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputDeliveryStatus {
    pub requested: DeliveryGuarantee,
    pub effective: DeliveryGuarantee,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgeStatus {
    pub current_envelopes: usize,
    pub current_rows: usize,
    pub current_bytes: usize,
    pub high_water_envelopes: usize,
    pub high_water_rows: usize,
    pub high_water_bytes: usize,
    pub blocked_sends: u64,
    pub blocked_duration: Duration,
    pub envelope_limit: usize,
    pub row_limit: usize,
    pub byte_limit: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceStatus {
    pub replay_positioning: ReplayPositioning,
    pub delivery: SourceDeliveryCapability,
    pub max_batch_rows: usize,
    pub max_batch_bytes: usize,
    pub next_sequence: Option<u64>,
    pub ended: bool,
    pub polls: u64,
    pub data_batches: u64,
    pub data_rows: u64,
    pub data_bytes: u64,
    pub fanned_out_batches: u64,
    pub fanned_out_rows: u64,
    pub fanned_out_bytes: u64,
    pub errors: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperatorStatus {
    pub input_batches: u64,
    pub input_rows: u64,
    pub input_bytes: u64,
    pub fanned_out_batches: u64,
    pub fanned_out_rows: u64,
    pub fanned_out_bytes: u64,
    pub processing_duration: Duration,
    pub errors: u64,
    pub ended: bool,
    pub late_rows: u64,
    pub late_affected_batches: u64,
    pub max_lateness: Option<Duration>,
    pub null_event_time_rows: u64,
    pub null_event_time_batches: u64,
    pub datafusion_runtime_created: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SinkStatus {
    pub output_id: String,
    pub effective_delivery: SinkDelivery,
    pub delivered_batches: u64,
    pub delivered_rows: u64,
    pub delivered_bytes: u64,
    pub write_duration: Duration,
    pub errors: u64,
    pub ended: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointPhase {
    Requested,
    SourcesCut,
    OperatorsSnapshotted,
    SinksPrecommitted,
    ManifestInstalled,
    ManifestDurable,
    SinksCommitted,
    Completed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointStatus {
    pub current_epoch: Option<Epoch>,
    pub phase: Option<CheckpointPhase>,
    pub terminal: bool,
    pub source_acknowledgements: usize,
    pub expected_sources: usize,
    pub operator_acknowledgements: usize,
    pub expected_operators: usize,
    pub sink_precommit_acknowledgements: usize,
    pub expected_sink_precommits: usize,
    pub sink_commit_acknowledgements: usize,
    pub expected_sink_commits: usize,
    pub elapsed: Option<Duration>,
    pub last_completed_epoch: Option<Epoch>,
    pub installed_unknown_epoch: Option<Epoch>,
    pub failure_category: Option<StreamingErrorCategory>,
    pub runtime_config_changed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JobStatus {
    pub job_id: u64,
    pub state: JobState,
    pub terminal_cause: Option<TerminalCause>,
    pub delivery: BTreeMap<String, OutputDeliveryStatus>,
    pub task_count: usize,
    pub task_errors: u64,
    pub metrics_overflowed: bool,
    pub edges: BTreeMap<String, EdgeStatus>,
    pub sources: BTreeMap<String, SourceStatus>,
    pub operators: BTreeMap<String, OperatorStatus>,
    pub sinks: BTreeMap<String, SinkStatus>,
    pub checkpoint: CheckpointStatus,
}
```

All maps are `BTreeMap`s keyed by stable IDs. `status()` clones this
data-only snapshot, never performs I/O, never blocks on a connector, and never
fails. Internal task IDs/names, private launch IDs, cursors, state handles,
manifest fields, connector payloads, and future internal metrics are absent.
Unknown internal fields do not flow into public status automatically.

`blocked_duration`, `processing_duration`, `write_duration`,
`max_lateness`, and checkpoint `elapsed` are Rust `Duration`s. Their Python
dictionary projection uses non-negative integer microseconds. Although the
baseline `EdgeBudget.max_rows` enforces both envelope and row capacity, status
names the two observed/configured quantities separately.

### Checkpoint publication and manual completion

The public method does not return a publication enum. Publication retains its
three states, but manual success additionally requires the existing #108
`Completed` advancement:

| Internal publication/completion result                   | Public manual observer                                                                                                                   | Checkpoint status                                                                 | Sink intent                                                                       |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| absent                                                   | `CalcFlowError::Streaming` with the mapped safe category; never returns epoch `E`                                                        | completed epoch unchanged; no installed-unknown epoch                             | prepared intent may abort; recovery cannot select `E`                             |
| installed, durability unknown                            | `CalcFlowError::Streaming(error)` where category is `CheckpointPublicationUnknown` and `error.epoch() == Some(E)`; Python typed subclass | `recovery_required`; `installed_unknown_epoch = E`; completed epoch unchanged     | neither commit nor abort; preserve intent for next-start reconciliation           |
| durable, but a sink commit fails before `Completed`      | `CalcFlowError::Streaming` with the mapped safe category; never returns epoch `E`                                                        | completed epoch unchanged; job may become `recovery_required`                     | durable intent remains selectable and is completed forward on the next start      |
| durable and every expected sink commit is acknowledged   | returns `E`                                                                                                                              | `last_completed_epoch = E`; installed-unknown epoch cleared after `Completed`     | every expected sink commit is acknowledged; later maintenance cannot revoke `E`   |

Manual, periodic, and terminal requests share one bounded, FIFO,
single-flight coordinator. Requests never coalesce. Every accepted request
owns a distinct increasing epoch. A manual request waits behind an active
periodic request; terminal completion is ordered after the active epoch and
every earlier accepted manual request. Queue submission backpressures at the
configured bound.

`trigger_checkpoint()` returns `E` only after all referenced state segments
are durable, all sinks have pre-committed, the expected canonical manifest
bytes have been installed, the manifest parent directory has been
synchronized, and every expected sink commit has been acknowledged. This is
the existing #108 `CheckpointAdvancement::Completed` seam; Public A6 adds no
second queue, epoch allocator, waiter registry, or earlier public completion
point. It does not wait for later retention, orphan collection, or compaction.
An immediate crash after the return can select `E`, and no sink commit for
`E` remains unacknowledged.

If rename installed the expected bytes but parent sync failed, the observer
receives the distinct indeterminate-publication error. The job does not
advance `last_completed_epoch`, commit, or abort. It converges to
`recovery_required`. A next process may select `E` if it persisted and fully
validates, or select the highest remaining wholly valid epoch if it did not.
Neither path may claim that `E` was absent at the original failure point.

A failure before install is absent. A sink commit failure after durability but
before `Completed` fails the observer and preserves forward-recoverable intent;
it is not a successful manual checkpoint. A maintenance failure after
`Completed` does not revoke a previously returned epoch and is visible through
later status and the terminal outcome. Dropping a manual observer loses only
its result and leaves the accepted request in FIFO order.

The indeterminate-publication shape is therefore:

```rust
Err(CalcFlowError::Streaming(error))
    if error.category() == StreamingErrorCategory::CheckpointPublicationUnknown
        && error.epoch() == Some(epoch)
```

Its stable category is `checkpoint_publication_unknown`. The display text is
exactly `checkpoint epoch <E> was installed but publication durability is
unknown`; it contains no path, manifest bytes, client error, or connector
value. There is no direct `CalcFlowError::CheckpointPublicationUnknown`
variant because that would violate the single A6 wrapper boundary.

### Automatic recovery and completed terminal manifests

Start has no resume flag. With no completed manifest, it resets operators,
opens sources at their configured beginning, starts runtime sequences at zero,
and assigns the first checkpoint epoch one. With completed manifest `R`, it
restores state/progress and each sink's scoped recovery, seeks every non-ended
source to its exact cursor, and assigns the next epoch `R + 1`.

The selector validates every canonical candidate in deterministic ascending
epoch order. One unreadable, corrupt, or semantically mismatched candidate at
any lower or higher epoch poisons the lineage. It never skips the bad
candidate, falls back to an older valid candidate, or silently starts empty.
If every candidate validates, a sparse lineage selects its highest epoch.
Temporary and orphan files never select recovery.

A terminal selected manifest runs sink-scoped recovery and returns a completed
job outcome without polling a source or re-emitting operator end output. A
terminal/failed job is never restarted in place; a new consumed runner creates
a new job. A caller wanting a fresh lineage supplies a fresh managed namespace.

### Capability proof

Whole-job preflight first validates exact source-map equality, exact output
coverage, globally unique sink IDs, stable participant identities, source
batch bounds, and all connector capability values. It binds unowned cursors to
their source-map key and rejects foreign configured cursors. It then samples
capabilities once before open, normalizes the watermark policy, freezes and
fingerprints the complete descriptor, and derives a proof separately for each
requested output over only its reachable subgraph. Recovery rejects descriptor
drift before connector open; live admission binds unowned emitted cursors and
rejects foreign ones before progress or data can advance.

Exactly-once for output `O` requires:

- every reachable source declares `ExactPauseReportAndSeek` and `Lossless`,
  honors emitted owned cursors on reopen in connector conformance, and fits
  its first-hop bounds;
- durable progress and the resolved watermark policy (default
  `SourceProvided`) have the exact M5 projection;
- every reachable stateful operator and selected UDF is deterministic;
- every reachable edge is bounded and lossless; and
- every routed sink is transactional or has an approved epoch-idempotent
  mechanism with sufficient/unbounded retention.

The requested/effective pair is frozen into status and manifest evidence. An
incompatible component on a disjoint output does not weaken output `O`, and an
at-least-once output is never silently upgraded. An adversarial source that
declares exact positioning but ignores the restore cursor must fail conformance
and cannot appear in an effective exactly-once proof.

### Redaction boundary

`StreamingError`, `JobOutcome`, logs, metrics labels, and diagnostic records
use only stable categories, engine-authored messages, stable component
kind/ID, job/epoch/phase, counts/durations, safe diagnostic ID, and deterministic
ordering. Connector/Python failures are sanitized at the boundary.

The following never appear in public status, formatted errors or source
chains, `Debug`, logs, metrics labels, Python exceptions, or diagnostic
records: batches/row values/metadata attributes; cursor bytes/payload/offsets;
pre-commit maps; state bytes/handles/segment IDs/checksums; manifest bytes or
paths; managed roots; connector options/credentials/URLs/query strings/raw
client errors; callback representations; UDF source/callables; panic payloads;
or private task/launch/runtime identity values.

The managed manifest remains engine-owned recovery data. It may contain only
the v3 schema-approved bounded recovery fields. It never contains raw batches,
connector options/credentials/errors, callback representations, UDF source,
or panic payloads, and it is never passed whole to a connector.

### Python plan and immutable configuration

`python/calc_flow/pipeline.py` keeps `ExecutionPlan` and `compile()` for batch
compatibility and adds this stream-specific path:

```python
class DeliveryGuarantee(StrEnum):
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


@dataclass(frozen=True, slots=True)
class StreamRequirements:
    delivery: Mapping[str, DeliveryGuarantee] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EdgeBudget:
    max_rows: int = 10_000
    max_bytes: int = 64 << 20


@dataclass(frozen=True, slots=True)
class StreamRuntimeConfig:
    checkpoint_interval: timedelta = timedelta(seconds=60)
    checkpoint_timeout: timedelta = timedelta(minutes=10)
    edge_budget: EdgeBudget = EdgeBudget()
    retained_epochs: int = 2


@dataclass(frozen=True, slots=True)
class StreamExecutionPlan:
    _inner: _native.StreamExecutionPlan = field(repr=False)

    @property
    def name(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...

    @property
    def requirements(self) -> StreamRequirements: ...

    @property
    def source_binding_ids(self) -> tuple[str, ...]: ...

    @property
    def sink_binding_ids(self) -> tuple[str, ...]: ...


class Runtime:
    def compile_stream_project(
        self,
        project_json: str,
        *,
        requirements: StreamRequirements | None = None,
    ) -> StreamExecutionPlan: ...


class PipelineBuilder:
    def compile_stream(
        self,
        *,
        requirements: StreamRequirements | None = None,
        runtime: Runtime | None = None,
    ) -> StreamExecutionPlan: ...
```

`None` requirements means all outputs request at-least-once. The adapters copy
every input mapping; the frozen values retain no caller-owned mutable mapping.
`timedelta` values must be exact microseconds and convert once to checked u64
microseconds. `compile()` continues to return the batch `ExecutionPlan`; there
is no execution mode switch on one plan type.

### Python connector values and async-only protocols

```python
type JSONValue = (
    None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
)


class ReplayPositioning(StrEnum):
    EXACT_PAUSE_REPORT_AND_SEEK = "exact_pause_report_and_seek"
    UNSUPPORTED = "unsupported"


class NativeWatermarkCapability(StrEnum):
    NEVER_EMITS = "never_emits"
    EMITS_NATIVE = "emits_native"
    RUNTIME_TOGGLEABLE = "runtime_toggleable"
    UNKNOWN = "unknown"


class SourceDeliveryCapability(StrEnum):
    LOSSLESS = "lossless"
    LOSSY = "lossy"


@dataclass(frozen=True, slots=True)
class SourceProvidedWatermarks:
    pass


@dataclass(frozen=True, slots=True)
class BoundedOutOfOrderness:
    event_time_column: str
    max_out_of_orderness: timedelta
    emit_interval: timedelta
    idle_timeout: timedelta | None = None


@dataclass(frozen=True, slots=True)
class DisabledWatermarks:
    idle_timeout: timedelta | None = None


type WatermarkPolicy = (
    SourceProvidedWatermarks | BoundedOutOfOrderness | DisabledWatermarks
)


@dataclass(frozen=True, slots=True)
class OrdinaryDelivery:
    pass


@dataclass(frozen=True, slots=True)
class EpochIdempotentDelivery:
    mechanism: str
    retention: Literal["bounded", "unbounded"]


@dataclass(frozen=True, slots=True)
class TransactionalDelivery:
    pass


type SinkDelivery = (
    OrdinaryDelivery | EpochIdempotentDelivery | TransactionalDelivery
)


@dataclass(frozen=True, slots=True)
class Cursor:
    order: bytes
    payload: Mapping[str, JSONValue]
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class SourceCapabilities:
    replay_positioning: ReplayPositioning
    delivery: SourceDeliveryCapability
    max_batch_rows: int
    max_batch_bytes: int
    schema: pa.Schema | None = None
    native_watermarks: NativeWatermarkCapability = (
        NativeWatermarkCapability.EMITS_NATIVE
    )


@dataclass(frozen=True, slots=True)
class Data:
    batch: Batch
    cursor: Cursor


@dataclass(frozen=True, slots=True)
class Watermark:
    at: datetime


@dataclass(frozen=True, slots=True)
class Idle:
    pass


type SourceEvent = Data | Watermark | Idle


class StreamSource(Protocol):
    def capabilities(self) -> SourceCapabilities: ...
    async def open(self, cursor: Cursor | None) -> None: ...
    async def next(self) -> SourceEvent | None: ...
    async def close(self) -> None: ...


class StreamSink(Protocol):
    async def open(self) -> None: ...
    async def write(self, batch: Batch) -> None: ...
    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SinkRecovery:
    epoch: int
    terminal: bool
    delivery: SinkDelivery
    pre_commit: Mapping[str, JSONValue]


class TransactionalStreamSink(Protocol):
    async def open(self) -> None: ...
    async def begin_epoch(self, epoch: int) -> None: ...
    async def write(self, batch: Batch) -> None: ...
    async def pre_commit(self, epoch: int) -> Mapping[str, JSONValue]: ...
    async def commit(
        self,
        epoch: int,
        pre_commit: Mapping[str, JSONValue],
    ) -> None: ...
    async def abort(
        self,
        epoch: int,
        pre_commit: Mapping[str, JSONValue] | None,
    ) -> None: ...
    async def recover(self, recovery: SinkRecovery) -> None: ...
    async def close(self) -> None: ...


class SourceBinding:
    def __init__(
        self,
        source: StreamSource,
        *,
        watermark_policy: WatermarkPolicy | None = None,
    ) -> None: ...


class SinkBinding:
    @classmethod
    def ordinary(cls, sink_id: str, sink: StreamSink) -> SinkBinding: ...

    @classmethod
    def transactional(
        cls,
        sink_id: str,
        sink: TransactionalStreamSink,
    ) -> SinkBinding: ...

    @classmethod
    def epoch_idempotent(
        cls,
        sink_id: str,
        sink: TransactionalStreamSink,
        *,
        mechanism: str,
        retention: Literal["bounded", "unbounded"],
    ) -> SinkBinding: ...
```

The `SourceDeliveryCapability`, three `WatermarkPolicy`, and three
`SinkDelivery` variants are the frozen Python projections of the corresponding
Rust values. `watermark_policy=None` means `SourceProvidedWatermarks()`, not
disabled. `Watermark.at` must be timezone-aware and is normalized to an exact
UTC microsecond.

`Cursor(..., source_id=None)` is unbound. Preflight/admission creates a fresh
frozen cursor bound to the exact source-map key; an explicitly named different
`source_id` is rejected before open or live progress. A cursor delivered to
`StreamSource.open` always has a non-`None` `source_id`, including recovery
from a v3 manifest whose enclosing source participant supplies the identity.
The adapter never mutates the connector's frozen value. `SourceCapabilities`
is called once before open; its delivery, replay, schema, native-watermark, and
row/byte-bound values plus the normalized binding watermark policy are copied,
frozen, and fingerprinted into durable identity. No Python `replayable` boolean
or private fallback is accepted.

Adapter construction validates all required connector methods with
`inspect.iscoroutinefunction` before invoking any connector method or entering
native launch. Every source `open`/`next`/`close`, ordinary sink
`open`/`write`/`close`, and transactional sink `open`/`begin_epoch`/`write`/
`pre_commit`/`commit`/`abort`/`recover`/`close` must be declared with
`async def`. Returning an awaitable from an ordinary synchronous `def` does
not qualify. Only `capabilities()` and other pure data accessors are
synchronous.

Accepted connector coroutines must not perform blocking Python/native work on
the event-loop or runtime thread and must cooperate with cancellation at await
points. The engine guarantees bounded ownership cleanup for conforming
connectors; it does not claim to preempt a callback that violates this public
contract.

The adapter defensively copies cursor payloads, pre-commit mappings,
capability values, and sink recovery before/after every Python boundary.
Python mutation cannot alter the engine-owned manifest or another callback's
view. A full manifest never becomes a Python object supplied to a connector.

### Python managed checkpoint owner, runner, and job

```python
class ManagedCheckpointRuntime:
    def __init__(
        self,
        directory: os.PathLike[str] | str,
        /,
    ) -> None: ...


class StreamingRunner:
    def __init__(
        self,
        plan: StreamExecutionPlan,
        sources: Mapping[str, SourceBinding],
        sinks: Mapping[str, Sequence[SinkBinding]],
        checkpoints: ManagedCheckpointRuntime,
        *,
        config: StreamRuntimeConfig | None = None,
    ) -> None: ...

    async def start_async(self) -> StreamingJob: ...
    def start(self) -> StreamingJob: ...


class StreamingJob:
    @property
    def id(self) -> int: ...

    def status(self) -> JobStatus: ...

    async def trigger_checkpoint_async(self) -> int: ...
    def trigger_checkpoint(self) -> int: ...

    async def shutdown_async(self) -> JobOutcome: ...
    def shutdown(self) -> JobOutcome: ...

    async def cancel_async(self) -> JobOutcome: ...
    def cancel(self) -> JobOutcome: ...

    async def wait_async(self) -> JobOutcome: ...
    def wait(self) -> JobOutcome: ...
```

`ManagedCheckpointRuntime.__init__` performs only path-like conversion,
lexical validation, and immutable path capture. It performs no directory
creation, canonicalization, scan, or locking. That I/O begins only inside
native async start, or after `start()` has passed its event-loop check. The
adapter accepts no keyword for a state or manifest child and no native/backend
object. Native start derives `<directory>/state` and
`<directory>/manifests`; neither internal path is exposed back to Python.

Python cannot express a move, so runner consumption is enforced at runtime.
Once native launch begins, the runner is consumed whether start returns a job,
raises, or its awaiter is cancelled. A second attempt raises builtin
`RuntimeError` with exactly:

```text
streaming runner has already been consumed by start()
```

An event-loop rejection happens before launch and does not consume the runner.
A pure validation failure raised by the constructor creates no native launch.
There is no runner-level shutdown, restart, reset, step, plan snapshot, or
second start method.

`status()` is the only status operation. It is synchronous, CPU-local, and
permitted inside an active event loop. It returns a fresh defensive dictionary
every call. There is no `status_async()`.

The blocking `start`, `trigger_checkpoint`, `shutdown`, `cancel`, and `wait`
methods check for a running event loop before any other work and raise builtin
`RuntimeError` with exactly:

```text
<method>() cannot run inside an event loop; use <method>_async()
```

The rule does not apply to the `id` property or `status()`.

Async cancellation has this exact projection:

| Cancelled Python awaiter                    | Required native effect                                                                       |
| ------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `start_async()`                             | cancel and reap the provisional launch; never expose a half-started job                      |
| `wait_async()`                              | detach only the observer; do not affect the job                                              |
| `trigger_checkpoint_async()`                | keep an accepted request queued; detach only the observer                                    |
| `shutdown_async()` or `cancel_async()`      | preserve the recorded transition and convergence; a later call reattaches                    |

If native terminal completion linearized before Python cancellation, the
native result/error wins. Otherwise cleanup or observer detachment is
established before `asyncio.CancelledError` propagates. After observed
terminal completion, a conforming connector leaves no pending Python task,
native await lease, callback root, connector, complete-root/lineage lease, or runtime
worker.

Python GC of the sole `StreamingJob` has the same non-blocking cancellation,
reaper transfer, and one safe `abandoned_streaming_job` warning as Rust drop.
Callers needing cleanup proof await a lifecycle method instead of relying on
GC.

### Python status and outcome typing

All enum-like strings below use the Rust snake-case serialization. Every
duration field is a non-negative integer ending in `_micros`.

```python
type StreamingErrorCategoryValue = Literal[
    "validation",
    "compile",
    "conflict",
    "cancelled",
    "checkpoint_timeout",
    "checkpoint_mismatch",
    "checkpoint_publication_unknown",
    "io",
    "operator",
    "connector",
    "task_panicked",
    "internal",
]


class OutputDeliveryStatus(TypedDict):
    requested: Literal["at_least_once", "exactly_once"]
    effective: Literal["at_least_once", "exactly_once"]


class EdgeStatus(TypedDict):
    current_envelopes: int
    current_rows: int
    current_bytes: int
    high_water_envelopes: int
    high_water_rows: int
    high_water_bytes: int
    blocked_sends: int
    blocked_duration_micros: int
    envelope_limit: int
    row_limit: int
    byte_limit: int


class SourceStatus(TypedDict):
    replay_positioning: Literal[
        "exact_pause_report_and_seek",
        "unsupported",
    ]
    delivery: Literal["lossless", "lossy"]
    max_batch_rows: int
    max_batch_bytes: int
    next_sequence: int | None
    ended: bool
    polls: int
    data_batches: int
    data_rows: int
    data_bytes: int
    fanned_out_batches: int
    fanned_out_rows: int
    fanned_out_bytes: int
    errors: int


class OperatorStatus(TypedDict):
    input_batches: int
    input_rows: int
    input_bytes: int
    fanned_out_batches: int
    fanned_out_rows: int
    fanned_out_bytes: int
    processing_duration_micros: int
    errors: int
    ended: bool
    late_rows: int
    late_affected_batches: int
    max_lateness_micros: int | None
    null_event_time_rows: int
    null_event_time_batches: int
    datafusion_runtime_created: bool


class OrdinaryDeliveryValue(TypedDict):
    kind: Literal["ordinary"]


class EpochIdempotentDeliveryValue(TypedDict):
    kind: Literal["epoch_idempotent"]
    mechanism: str
    retention: Literal["bounded", "unbounded"]


class TransactionalDeliveryValue(TypedDict):
    kind: Literal["transactional"]


type SinkDeliveryValue = (
    OrdinaryDeliveryValue
    | EpochIdempotentDeliveryValue
    | TransactionalDeliveryValue
)


class SinkStatus(TypedDict):
    output_id: str
    effective_delivery: SinkDeliveryValue
    delivered_batches: int
    delivered_rows: int
    delivered_bytes: int
    write_duration_micros: int
    errors: int
    ended: bool


class CheckpointStatus(TypedDict):
    current_epoch: int | None
    phase: str | None
    terminal: bool
    source_acknowledgements: int
    expected_sources: int
    operator_acknowledgements: int
    expected_operators: int
    sink_precommit_acknowledgements: int
    expected_sink_precommits: int
    sink_commit_acknowledgements: int
    expected_sink_commits: int
    elapsed_micros: int | None
    last_completed_epoch: int | None
    installed_unknown_epoch: int | None
    failure_category: str | None
    runtime_config_changed: bool


class JobStatus(TypedDict):
    job_id: int
    state: Literal[
        "running",
        "draining",
        "completed",
        "cancelled",
        "failed",
        "recovery_required",
    ]
    terminal_cause: str | None
    delivery: dict[str, OutputDeliveryStatus]
    task_count: int
    task_errors: int
    metrics_overflowed: bool
    edges: dict[str, EdgeStatus]
    sources: dict[str, SourceStatus]
    operators: dict[str, OperatorStatus]
    sinks: dict[str, SinkStatus]
    checkpoint: CheckpointStatus


class StreamingErrorValue(TypedDict):
    category: StreamingErrorCategoryValue
    message: str
    job_id: int | None
    epoch: int | None
    checkpoint_phase: str | None
    component_kind: str | None
    component_id: str | None
    diagnostic_id: int | None
    position: int


class JobOutcomeValue(TypedDict):
    state: str
    cause: str
    completed_epoch: int | None
    errors: tuple[StreamingErrorValue, ...]


@dataclass(frozen=True, slots=True)
class StreamingError:
    category: StreamingErrorCategoryValue
    message: str
    job_id: int | None
    epoch: int | None
    checkpoint_phase: str | None
    component_kind: str | None
    component_id: str | None
    diagnostic_id: int | None
    position: int


@dataclass(frozen=True, slots=True)
class JobOutcome:
    state: str
    cause: str
    completed_epoch: int | None
    errors: tuple[StreamingError, ...]
```

The dicts under `delivery`, `edges`, `sources`, `operators`, and `sinks` are
inserted in stable-ID order and are new objects on every call. `JobStatus`
contains no cursor, raw error, private task map, or `abandoned_job_drops` key.

Every failed native A6 owner/runner/job call receives only the Rust
`CalcFlowError::Streaming` value and projects it into this read-only public
exception surface:

```python
class StreamingRuntimeError(CalcFlowError):
    @property
    def category(self) -> StreamingErrorCategoryValue: ...
    @property
    def message(self) -> str: ...
    @property
    def job_id(self) -> int | None: ...
    @property
    def epoch(self) -> int | None: ...
    @property
    def checkpoint_phase(self) -> str | None: ...
    @property
    def component_kind(self) -> str | None: ...
    @property
    def component_id(self) -> str | None: ...
    @property
    def diagnostic_id(self) -> int | None: ...
    @property
    def position(self) -> int: ...


class CheckpointPublicationUnknownError(StreamingRuntimeError):
    @property
    def epoch(self) -> int: ...
```

All categories use `StreamingRuntimeError`; only
`checkpoint_publication_unknown` uses the narrower subclass because callers
must distinguish an installed indeterminate epoch without parsing text. Both
classes are exported from `calc_flow.errors` and top-level `calc_flow`.
Their `str()` equals safe `message`. The PyO3 converter copies the nine safe
fields, discards the Rust wrapper, and explicitly clears `__cause__` and
`__context__` to `None` immediately before propagation. It never calls the
existing general `to_py_err` branch that preserves `CalcFlowError::Io` or its
`PyOSError` cause, and never copies a callback `repr`, exception string,
traceback, path, or panic payload.

### `_native.pyi` and PyO3 registration

The checked-in stub removes all three legacy native classes and registers the
replacement names with these exact supported signatures. Adapter-only helper
types may be private, but no extra public lifecycle method may be registered.

```python
class StreamingRuntimeError(CalcFlowError):
    @property
    def category(self) -> StreamingErrorCategoryValue: ...
    @property
    def message(self) -> str: ...
    @property
    def job_id(self) -> int | None: ...
    @property
    def epoch(self) -> int | None: ...
    @property
    def checkpoint_phase(self) -> str | None: ...
    @property
    def component_kind(self) -> str | None: ...
    @property
    def component_id(self) -> str | None: ...
    @property
    def diagnostic_id(self) -> int | None: ...
    @property
    def position(self) -> int: ...


class CheckpointPublicationUnknownError(StreamingRuntimeError):
    @property
    def epoch(self) -> int: ...


class Runtime:
    def compile_stream_project(
        self,
        project_json: str,
        requirements: Mapping[str, str],
    ) -> StreamExecutionPlan: ...


@final
class StreamExecutionPlan:
    @property
    def name(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...

    @property
    def requirements(self) -> dict[str, str]: ...

    @property
    def source_binding_ids(self) -> tuple[str, ...]: ...

    @property
    def sink_binding_ids(self) -> tuple[str, ...]: ...


@final
class _ManagedCheckpointRuntime:
    def __init__(self, directory: str, /) -> None: ...


@final
class _StreamingRunner:
    def __init__(
        self,
        plan: StreamExecutionPlan,
        sources: Mapping[str, object],
        sinks: Mapping[str, Sequence[object]],
        checkpoints: _ManagedCheckpointRuntime,
        config: Mapping[str, object],
    ) -> None: ...

    def start_async(self) -> Awaitable[_StreamingJob]: ...


@final
class _StreamingJob:
    @property
    def id(self) -> int: ...

    def status(self) -> JobStatus: ...
    def trigger_checkpoint_async(self) -> Awaitable[int]: ...
    def shutdown_async(self) -> Awaitable[JobOutcomeValue]: ...
    def cancel_async(self) -> Awaitable[JobOutcomeValue]: ...
    def wait_async(self) -> Awaitable[JobOutcomeValue]: ...


```

The Python adapters own argument copying, async-shape validation, guarded
blocking twins, and frozen value construction. PyO3 owns the native
runner/job/checkpoint lifetime and calls the Rust engine. Neither layer
implements checkpoint selection, capability derivation, delivery semantics,
or terminal arbitration independently.

Registration and stub parity tests assert that `_MicroBatchRunner`, the old
method-bearing `_StreamingRunner`, and `_FileCheckpointStore` are absent; the
replacement `_StreamingRunner` has `start_async` and none of `step_async`,
`reset_async`, `plan_snapshot_async`, or runner `shutdown_async`. They also
assert that `_ManagedCheckpointRuntime` accepts exactly one positional string,
both safe exception classes are registered/exported, and every A6 native
failure has no Python cause or context.

### Studio REST stability and private persistence

Public Studio API signatures are unchanged:

```python
def create_app(
    *,
    project_directory: str | Path = ".calc-flow-projects",
    checkpoint_directory: str | Path = ".calc-flow-checkpoints",
    project_store: ProjectStoreProtocol | None = None,
    checkpoint_store: CheckpointStoreProtocol | None = None,
    runtime: RuntimeProtocol | None = None,
    run_manager: RunManagerProtocol | None = None,
    frontend_directory: str | Path | None = None,
) -> FastAPI: ...


class CheckpointStoreProtocol(Protocol):
    async def load(self, pipeline_name: str) -> dict[str, object] | None: ...
    async def delete(self, pipeline_name: str) -> None: ...
```

The implementation replaces the top-level `calc_flow.FileCheckpointStore`
import with a private Studio checkpoint-document store under
`web-ui/backend/src/calc_flow_studio/`. That store preserves only the
existing v2 document behavior needed by Studio's GET/DELETE routes. It is not
exported by `calc_flow`, is not an alias for the removed symbol, and is not
accepted by `ManagedCheckpointRuntime`.

These REST contracts and `CheckpointSummary` remain byte-for-byte stable:

```http
GET    /api/v2/projects/{project_id}/checkpoint
DELETE /api/v2/projects/{project_id}/checkpoint
```

`web-ui/openapi.json` and generated `web-ui/src/api/schema.d.ts` must have no
A6 route/model diff. Default `create_app()`, injected-store tests, checkpoint
summary/delete behavior, Studio package import, and installed-wheel startup
must all remain green after the public Python legacy symbol disappears.

## Why This Shape

- A consuming `start(self)` makes the ownership transfer visible in Rust and
  leaves one non-cloneable job as the only lifecycle owner. A reusable runner
  or a runner-level shutdown would reintroduce two competing owners.
- `ManagedCheckpointRuntime` accepts one local root and derives both children
  behind one complete-root lease. A backend input, a second root, or even a
  two-root local convenience would make overlap and namespace identity
  unverifiable after type erasure.
- One defaulted `with_runtime_config` step keeps the common Rust constructor at
  four related required inputs without hiding runtime tuning in an argument
  bag. The Python keyword-only `config=None` projects the same default.
- `ReplayPositioning` is an enum carried through identity/status/manifest; a
  boolean cannot distinguish exact pause/report/seek from an informal claim
  and cannot state the conformance obligation.
- `SourceCapabilities` is one complete once-sampled descriptor, including
  lossless/lossy delivery and declared bounds. Retaining #110's private
  `replayable` fallback or resampling schema/watermark/delivery independently
  would let preflight, durable identity, status, and recovery disagree.
- `Cursor` retains an optional source owner so an unbound connector value can
  be bound once while an already foreign value fails closed. Omitting the
  owner from memory would recreate the cross-source recovery/admission bug
  that #110 removed; duplicating it inside the v3 cursor entry would change the
  manifest wire format and is therefore rejected.
- `SinkRecovery` is an engine-created scoped capability. Passing a full
  manifest would expose cross-participant cursor, state, and sink evidence and
  let a connector mutate a caller-owned recovery document.
- Manual checkpoint success is an epoch only. The installed-unknown branch is
  a distinct typed error and status field; treating it as absent contradicts
  the baseline's selectable post-rename state. Returning success before parent
  sync overstates durability; returning after parent sync but before all sink
  commit acknowledgements would contradict #108's `Completed` waiter seam.
- Connector lifecycle is async-only in Python so native cancellation and GC
  can settle one owned coroutine graph. Supporting synchronous callbacks would
  make the promised cleanup boundary unachievable.
- `status()` remains synchronous because it is a pure cloned snapshot. Adding
  `status_async()` or rejecting an event loop would introduce fake asynchrony
  and make the cheapest observation path harder to use.
- Owner drop uses one warning instead of a per-job drop counter because no
  public handle remains to read that counter. Callers who need proof use an
  awaited lifecycle method.
- One `CalcFlowError::Streaming(StreamingError)` variant makes every fallible
  A6 lifecycle cross the same safe conversion seam. Reusing raw-bearing
  `Io`, `Operator`, `ExternalProvider`, or `TaskPanicked` variants would leak
  through `Debug`, `source()`, or Python exception chaining even when the
  display string looked sanitized.
- Studio persistence stays private and the REST contract stays unchanged.
  Replacing existing `/api/v2` routes or their document model would exceed the
  controlling spec and turn a public runtime cut into an unrelated Studio API
  migration.
- The repository cut is atomic because a partial cut cannot simultaneously
  satisfy Rust compilation, native registration, Python imports/stubs, Studio
  startup, shipped examples, benchmark collection, documentation, and package
  smoke tests.

## Error Cases

All messages are engine-generated and contain only the safe coordinates shown.
For every category row, Rust returns `CalcFlowError::Streaming`; Python raises
`StreamingRuntimeError`, except that publication-unknown uses its named
subclass. Backticks delimit values and are not part of the message.

| Input or transition violation                    | Safe category / Python projection                                           | Exact message or message pattern                                                                |
| ------------------------------------------------ | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| managed directory is empty                       | `validation` / `StreamingRuntimeError`                                      | `managed checkpoint directory must not be empty`                                                |
| caller supplies a backend or two roots           | signature rejection; no runtime error                                       | no such Rust/Python/PyO3 signature                                                              |
| source map misses input `orders`                 | `validation` / `StreamingRuntimeError`                                      | `source bindings are missing external input "orders"`                                           |
| source map contains unknown `extra`              | `validation` / `StreamingRuntimeError`                                      | `source binding "extra" does not match a compiled external input`                               |
| output `result` has no sink route                | `validation` / `StreamingRuntimeError`                                      | `sink bindings are missing graph output "result"`                                               |
| sink ID `archive` occurs on two outputs          | `validation` / `StreamingRuntimeError`                                      | `sink ID "archive" is configured for more than one graph output`                                |
| source bound is zero                             | `validation` / `StreamingRuntimeError`                                      | `source "orders" max_batch_rows must be greater than zero`                                      |
| source batch bound exceeds first edge            | `validation` / `StreamingRuntimeError`                                      | `source "orders" maximum batch exceeds edge "<stable_edge_id>" budget`                          |
| cursor is owned by source `other`                | `validation` / `StreamingRuntimeError`                                      | `source "orders" received a foreign cursor owned by "other"`                                    |
| exact output reaches unsupported source          | `compile` / `StreamingRuntimeError`                                         | `output "result" requires exactly_once but source "orders" lacks exact_pause_report_and_seek`   |
| exact output reaches lossy source                | `compile` / `StreamingRuntimeError`                                         | `output "result" requires exactly_once but source "orders" is lossy`                            |
| exact source ignores restored cursor             | `validation` / `StreamingRuntimeError`                                      | `source "orders" declared exact_pause_report_and_seek but did not honor the restore cursor`     |
| exact output reaches volatile operator/UDF       | `compile` / `StreamingRuntimeError`                                         | `output "result" requires exactly_once but operator "price" is not deterministic`               |
| exact output reaches ordinary sink               | `compile` / `StreamingRuntimeError`                                         | `output "result" requires exactly_once but sink "preview" is ordinary`                          |
| epoch-idempotent retention is bounded            | `compile` / `StreamingRuntimeError`                                         | `output "result" requires exactly_once but sink "archive" has bounded retention`                |
| complete managed-root lease is held              | `conflict` / `StreamingRuntimeError`                                        | `managed checkpoint directory is already leased for pipeline "<pipeline_name>"`                 |
| derived state lineage lease is held              | `conflict` / `StreamingRuntimeError`                                        | `state lineage "<pipeline_name>" is already leased`                                             |
| local root initialization fails                  | `io` / `StreamingRuntimeError`                                              | `managed checkpoint storage initialization failed`                                              |
| any canonical manifest candidate is invalid      | `checkpoint_mismatch` / `StreamingRuntimeError`                             | `checkpoint lineage contains an invalid manifest candidate at epoch <E>`                        |
| checkpoint request times out before `Completed`  | `checkpoint_timeout` / `StreamingRuntimeError`                              | `checkpoint epoch <E> exceeded the configured timeout`                                          |
| manifest installed but parent sync failed        | `checkpoint_publication_unknown` / `CheckpointPublicationUnknownError`      | `checkpoint epoch <E> was installed but publication durability is unknown`                      |
| sink commit fails before `Completed`             | `connector` / `StreamingRuntimeError`                                       | `sink "archive" commit failed for checkpoint epoch <E>`                                         |
| Python source lifecycle method is synchronous    | builtin `TypeError`                                                         | `source "orders" open() must be declared with async def`                                        |
| Python transactional sink method is synchronous  | builtin `TypeError`                                                         | `sink "archive" recover() must be declared with async def`                                      |
| Python blocking method runs in event loop        | builtin `RuntimeError`                                                      | `<method>() cannot run inside an event loop; use <method>_async()`                              |
| consumed Python runner starts again              | builtin `RuntimeError`                                                      | `streaming runner has already been consumed by start()`                                         |
| connector raises with a secret/raw client error  | `connector` / `StreamingRuntimeError`                                       | `source "orders" open failed` or `sink "archive" write failed`; raw text is omitted             |
| contained task panics with a secret payload      | `task_panicked` / `StreamingRuntimeError`                                   | `streaming task failed`                                                                         |
| sole job owner is dropped                        | warning event, not status/error                                             | event `abandoned_streaming_job`; only allowlisted job/state/count coordinates                   |

Duplicate/missing/unknown/cross-output IDs, capability failures, managed-root
validation, and both lease conflicts occur before lifecycle work. Sync Python
method rejection occurs before the method is called. The event-loop check
occurs before consumed-state or argument validation for each blocking method.

## Example

The Rust shipped example replaces `micro_batch_recovery.rs` with a
20-50-line public continuous recovery example. Application connector
implementations are defined in focused support so the happy path stays short:

```rust
use std::collections::BTreeMap;

use calc_flow::{
    DeliveryGuarantee, ManagedCheckpointRuntime, PipelineBuilder,
    SinkBinding, SourceBinding, StreamRequirements, StreamingRunner,
    UdfRegistry,
};

#[tokio::main]
async fn main() -> calc_flow::Result<()> {
    let mut requirements = StreamRequirements::default();
    requirements.delivery.insert(
        "output".into(),
        DeliveryGuarantee::ExactlyOnce,
    );
    let plan = PipelineBuilder::new("orders")?
        .add_node("total", total_operator()?)?
        .compile_stream(&UdfRegistry::new().snapshot(), &requirements)?;
    let checkpoints = ManagedCheckpointRuntime::new(".calc-flow-continuous")?;
    let runner = StreamingRunner::new(
        plan,
        BTreeMap::from([("input".into(), SourceBinding::new(order_source()))]),
        BTreeMap::from([(
            "output".into(),
            vec![SinkBinding::transactional("archive", archive_sink())?],
        )]),
        checkpoints,
    )?;
    let job = runner.start().await?;
    let completed_epoch = job.trigger_checkpoint().await?;
    let outcome = job.shutdown().await;
    assert!(outcome.completed_epoch >= Some(completed_epoch));
    Ok(())
}
```

The Python shipped example replaces `04_micro_batch_recovery.py`. It shows the
async-only lifecycle, exact replay position, synchronous-in-loop status, and
explicit terminal wait:

```python
import asyncio

from calc_flow import PipelineBuilder
from calc_flow.runtime import (
    Cursor,
    Data,
    ManagedCheckpointRuntime,
    ReplayPositioning,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    StreamingRunner,
)


class Orders:
    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            delivery=SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=100,
            max_batch_bytes=1 << 20,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self.offset = 0 if cursor is None else int(cursor.payload["offset"])

    async def next(self) -> Data | None:
        if self.offset == 3:
            return None
        batch = make_batch(self.offset)
        self.offset += 1
        cursor = Cursor(
            self.offset.to_bytes(8, "big"),
            {"offset": self.offset},
            source_id="input",
        )
        return Data(batch, cursor)

    async def close(self) -> None:
        pass


async def main() -> None:
    plan = PipelineBuilder("orders").expression("total", "total = a + b").compile_stream()
    runner = StreamingRunner(
        plan,
        {"input": SourceBinding(Orders())},
        {"output": [ordinary_sink_binding()]},
        ManagedCheckpointRuntime(".calc-flow-continuous"),
    )
    job = await runner.start_async()
    assert job.status()["state"] == "running"
    outcome = await job.wait_async()
    assert outcome.state == "completed"


asyncio.run(main())
```

The exactly-once example variant changes requirements and uses
`SinkBinding.transactional`; the ordinary version intentionally reports
at-least-once. Connector conformance examples reopen a new process with the
persisted cursor and prove no gap or duplicate.

## Atomic Repository Cutover

The implementation is accepted only as one mergeable revision in which every
row below is complete. A PR may contain multiple commits for review, but no
commit intended to land independently may publish a mixed old/new boundary.

| Area                            | Required replacement and smoke proof                                                                                                                                  |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `crates/calc-flow/src/lib.rs`   | remove v2 checkpoint/source/sink/runner exports; export every A6 Rust type above; keep manifest v3, state, batch plan, and project v2                                 |
| Rust modules/tests              | delete legacy runtime/checkpoint implementation or make it non-public only where Studio does not consume it; compile new example and negative removed-symbol fixtures |
| PyO3 module registration        | remove `_MicroBatchRunner`, old `_StreamingRunner`, `_FileCheckpointStore`; register exact replacement plan/checkpoint/runner/job/error surface                       |
| `python/calc_flow/_native.pyi`  | match runtime registration and supported native call signatures exactly; no stale class or method                                                                     |
| Python adapters and `__all__`   | export A6 values/protocols/runner/job plus safe exception types; remove `MicroBatchRunner`, public `FileCheckpointStore`, and old source/runner names                 |
| Studio backend                  | replace direct public store import with private v2 document persistence; preserve default `create_app()`, injected protocol, routes, documents, and wheel startup     |
| OpenAPI/generated TypeScript    | `web-ui/openapi.json` and `web-ui/src/api/schema.d.ts` have no A6 diff; frontend build/test remains green                                                             |
| Rust/Python examples            | replace micro-batch recovery examples with public continuous recovery; signature/compile/run checks use only A6 names                                                 |
| Benchmarks                      | replace `FileCheckpointStore` Python cases with managed checkpoint/public-runner cases or remove obsolete scenarios; benchmark discovery and common-edge harness pass |
| Active docs                     | update README, introduction, API reference, Rust/Python API, runtime envelope, example inventories, release/migration notes, and plan header                          |
| Release and package smoke       | core crate/wheel and Studio wheel import/start; removed names fail; new names import; `create_app()` and checkpoint routes work after clean installation              |

Historical v1 evidence under `tests/fixtures/v1/` remains byte-for-byte
unchanged. Historical `docs/v1-final-api.md`, archived `docs/superpowers/`,
and the v0.2 design/migration record may name removed symbols as history, but
must not be imported, compiled, discovered as an example/benchmark, or used by
release smoke. Every active code consumer and current-surface claim must be
migrated.

The plan header is updated in the same documentation cut to exactly:

```text
M0-M5 internal complete; Public A6 pending
```

It is changed to public-complete only after implementation, all evidence, and
zero-blocker review exist.

The A6-01 delivery PR is documentation-only and precedes that implementation
cut. After the six-axis critique reaches zero blockers,
`cf-implementer`/`cf-reviewer` assemble the final spec, API note, critique,
plan-status line, and related documentation updates on the shared baseline;
run the documentation and artifact validation gates; commit and push the
branch; and open a draft PR whose title contains `DAL-103` and whose body
references GitHub issue `#94`. That PR must contain no runtime behavior,
public export, generated-contract, or test implementation change. Creating it
records the approved design chain; it does not claim merge, runtime
implementation, or issue acceptance.

## Validation Mapping

### Six independent critique axes

The next critique must report each axis independently and may return zero
blockers only when every named proof below exists.

| Critique axis          | Required positive proof                                                                                                                             | Required adversarial proof                                                                                                        |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| ownership              | consuming Rust `start(self)`; non-Clone job; one-root owner; launch/job/GC/drop convergence; complete-root and lineage leases released after settle | dropped start/job, cancelled Python gate, same-root in-process/cross-process/symlink conflicts, backend/two-root compile failures |
| checkpoint completion  | manual success at #108 `Completed` after parent sync and all sink commit acks; immediate-crash recovery of returned epoch                           | real directory-sync failure; partial commit fails observer; no completed advance; both installed-unknown crash outcomes           |
| recovery               | empty lineage, `R + 1`, all-valid sparse highest, terminal resume, cross-language restart                                                           | corrupt-lower/valid-higher and valid-lower/corrupt-higher both poison; sink canaries see only their own recovery                  |
| capability proof       | cursor plus frozen descriptor parity across Rust/Python/status/identity/manifest; reachable-only requested/effective derivation                     | foreign cursor; descriptor drift; boolean fallback; ignored restore cursor; lossy source; volatile UDF; bounded sink retention    |
| redaction              | every fallible A6 lifecycle returns only the safe wrapper; exact D7/error/log/manifest censuses                                                     | raw variant rejected; constructor/start/checkpoint/outcome and Python cause/context canaries; two-sink isolation                  |
| cutover                | crate/native/Python/Studio/example/benchmark/docs/release smoke all agree on one revision                                                           | removed imports/methods fail; default Studio startup/routes work without the removed public store; generated API remains clean    |

### Focused test targets

| Contract                                  | Proposed evidence location and assertion                                                                                                                                                       |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rust signatures/ownership                 | `crates/calc-flow/tests/public_continuous_runtime.rs`: compile-pass `start(self)` use; compile-fail reuse and Clone fixtures; job sole owner                                                   |
| single-root owner signatures              | Rust compile-fail rejects backend/two-root constructors; PyO3 registration, `inspect.signature`, adapter, top-level export, and stub accept one local directory only                           |
| pure construction and gated start         | inline runner tests: no I/O/lifecycle at construction; every launch cancellation gate settles tasks/connectors and both root/lineage leases                                                    |
| complete namespace lease                  | inline state/runner plus subprocess tests: one-root derived children, same-root cross-process/symlink conflicts, backend/two-root signature rejection                                          |
| new run/resume/terminal                   | public runner tests: empty epoch one, selected `R` then `R + 1`, terminal no-poll/no-reemit, every-candidate fail-closed                                                                       |
| checkpoint FIFO and completion            | public/inline coordinator tests: #108 bounded FIFO reuse, periodic/manual/terminal ordering, dropped observer, `Completed` return seam, partial-commit failure                                 |
| installed-unknown                         | real filesystem parent-sync failure and restart tests: typed epoch, recovery-required, no commit/abort, persisted and lost outcomes                                                            |
| cursor and frozen source identity         | Rust/Python configured/restored/emitted binding, foreign rejection, once-sampled schema/watermark/replay/delivery/bounds fingerprint, restart-drift matrix                                     |
| exact positioning/capability              | public connector conformance: no `replayable` fallback, lossless enum/identity/status/manifest parity, cursor-ignoring adversary, per-output reachability                                      |
| sink-scoped recovery                      | two-sink Rust canary and `python/tests/test_continuous_runtime.py`: no cursor/operator/root/other-sink visibility, defensive copies                                                            |
| Python async lifecycle/cancellation       | `python/tests/test_continuous_runtime.py`: reject every sync method before call; all four awaiter-cancel rows; GC; no callback root/lease/task left                                            |
| Python synchronous status                 | event-loop test calls `job.status()` successfully, sees a fresh sorted dict, proves no `status_async` and no `abandoned_job_drops`                                                             |
| safe wrapper signatures and mapping       | compile/runtime tests cover constructor, runner configuration/start, and checkpoint; every error is exactly `CalcFlowError::Streaming` with the exhaustive category/coordinate mapping         |
| outward and manifest redaction            | canary census over `Display`, normal/alternate `Debug`, complete `source()` chains, outcome, status, logs, metrics, diagnostics, safe serialization, and strict manifest fields                |
| Python safe exception projection          | constructor/preflight, local I/O, connector, operator, panic, timeout, pre-install I/O, and installed-unknown cases preserve safe fields with no text/traceback/`__cause__`/`__context__` leak |
| Studio contract                           | `web-ui/backend/tests/test_app.py`: default import/`create_app()`, injected store, existing v2 GET/DELETE documents; OpenAPI and schema generated-file no-diff                                 |
| examples and benchmarks                   | Rust example compile/run; every Python example signature/run; benchmark collection; public common-edge performance comparison                                                                  |
| removed-symbol census                     | Rust compile-fail, native `hasattr`, Python import/`__all__`/stub, active repository consumer scan, release inspector                                                                          |
| package/release smoke                     | clean core wheel/sdist/crate/Studio wheel inspection and install; core/Studio smoke, default app startup, examples, benchmark discovery                                                        |
| cross-surface durability                  | fresh Rust-to-Python and Python-to-Rust processes share only managed filesystem evidence and verify bytes/epochs/seek/output/idempotency/zero ownership                                        |
| long-run ownership                        | exact final-head 20-minute, three-process-generation, 120-sample public runner soak                                                                                                            |

### Named 48-case M5 fault matrix

Enumeration is not evidence. Every ID below executes at the exact final public
runner head and records selected epoch; restored cursor/watermark/idle/end/
sequence; operator/window state; ordinary/transactional output; missing and
duplicate counts; reachable/temporary artifacts; terminal state/error;
task/connector/complete-root-and-lineage-lease cleanup; and manual observer
classification.

| Boundary             | I/O case                           | Panic case                            | Cancel case                            | Restart case                            |
| -------------------- | ---------------------------------- | ------------------------------------- | -------------------------------------- | --------------------------------------- |
| Source admission     | `m5_fault/source_admission/io`     | `m5_fault/source_admission/panic`     | `m5_fault/source_admission/cancel`     | `m5_fault/source_admission/restart`     |
| Source cut           | `m5_fault/source_cut/io`           | `m5_fault/source_cut/panic`           | `m5_fault/source_cut/cancel`           | `m5_fault/source_cut/restart`           |
| Partial alignment    | `m5_fault/partial_alignment/io`    | `m5_fault/partial_alignment/panic`    | `m5_fault/partial_alignment/cancel`    | `m5_fault/partial_alignment/restart`    |
| State stage          | `m5_fault/state_stage/io`          | `m5_fault/state_stage/panic`          | `m5_fault/state_stage/cancel`          | `m5_fault/state_stage/restart`          |
| Sink pre-commit      | `m5_fault/sink_pre_commit/io`      | `m5_fault/sink_pre_commit/panic`      | `m5_fault/sink_pre_commit/cancel`      | `m5_fault/sink_pre_commit/restart`      |
| Manifest write       | `m5_fault/manifest_write/io`       | `m5_fault/manifest_write/panic`       | `m5_fault/manifest_write/cancel`       | `m5_fault/manifest_write/restart`       |
| Manifest rename      | `m5_fault/manifest_rename/io`      | `m5_fault/manifest_rename/panic`      | `m5_fault/manifest_rename/cancel`      | `m5_fault/manifest_rename/restart`      |
| Manifest parent sync | `m5_fault/manifest_parent_sync/io` | `m5_fault/manifest_parent_sync/panic` | `m5_fault/manifest_parent_sync/cancel` | `m5_fault/manifest_parent_sync/restart` |
| Partial sink commit  | `m5_fault/partial_sink_commit/io`  | `m5_fault/partial_sink_commit/panic`  | `m5_fault/partial_sink_commit/cancel`  | `m5_fault/partial_sink_commit/restart`  |
| Completed commit     | `m5_fault/completed_commit/io`     | `m5_fault/completed_commit/panic`     | `m5_fault/completed_commit/cancel`     | `m5_fault/completed_commit/restart`     |
| Retention            | `m5_fault/retention/io`            | `m5_fault/retention/panic`            | `m5_fault/retention/cancel`            | `m5_fault/retention/restart`            |
| Compaction           | `m5_fault/compaction/io`           | `m5_fault/compaction/panic`           | `m5_fault/compaction/cancel`           | `m5_fault/compaction/restart`           |

Manifest-rename and parent-sync cases explicitly cover absent,
installed-unknown, and completed plus both post-crash outcomes of
installed-unknown. Partial-sink-commit cases must fail the manual observer and
preserve forward recovery; only `Completed` may return the epoch. The
parent-sync installed-unknown case uses a real directory-sync failure, not
only a post-sync test hook.

### Cross-surface and repository gates

The cross-language restart E2E runs both directions in fresh processes with
one deterministic connector protocol and identical participant IDs. It
verifies manifest bytes, selected/next epoch, exact source seek, final window
output, sink commit idempotency, and zero live resources. Only manifests,
state segments, and sink evidence cross the process boundary.

The standard soak launches exactly three child OS generations for global
ranges `0..40`, `40..80`, and `80..120`. It records exactly 120 ten-second
samples over 1,200 measured seconds and checks bounded queues/tasks/state/
manifests/RSS, restart continuity, no missing/duplicate transactional output,
no temporary artifact, and terminal zero ownership. Evidence pins exact
commit and executable hashes; any implementation push invalidates it.

Every applicable command group in `AGENTS.md` must pass on the exact final
head, including:

```text
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
uv run python scripts/run_rust_tests.py
cargo llvm-cov --workspace --all-features --fail-under-lines 90
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests -q
uv run ruff check .
uv run ruff format --check .
cd web-ui/backend && uv run --project . --extra dev pytest --cov=calc_flow_studio
cd web-ui && npm ci && npm run sync:api && npm run build && npm test
cd web-ui && npm run test:e2e && npm audit --omit=dev
python -m unittest scripts.test_run_rust_tests scripts.test_inspect_wheel scripts.test_release_config
```

Release acceptance additionally builds and inspects the core wheel, source
distribution, crate, and Studio wheel; installs them in clean environments;
runs core/Studio smoke; compiles/runs examples; collects benchmarks; and
proves:

```text
git diff --exit-code -- schemas/project-v2.schema.json web-ui/openapi.json web-ui/src/api/schema.d.ts
git diff --check
```

The public implementation is compared with baseline
`aa3bbf0b40aef74898a59b6d0d0028c59a2d6993` using the existing common-edge
harness. A greater-than-5% regression needs statistically conclusive same-host
evidence and explicit approval. A6 adds no data-path task, queue, batch copy,
Arrow materialization, or per-batch serialization relative to the M5 private
path.

## Open Questions

None. This surface now matches the controlling spec hash and
`main@aa3bbf0b…`: it preserves the single-root owner and safe error boundary,
projects #108's `Completed` manual-checkpoint seam, and exposes #110's
identity-bound cursor plus once-sampled lossless/lossy source descriptor with
no `replayable` fallback. It is ready for `cf-critic` to review independently
on ownership, checkpoint completion, recovery, capability proof, redaction,
and cutover. It must not move to implementation until that review reports zero
blockers on all six axes against this API-note hash and the exact PR head.
