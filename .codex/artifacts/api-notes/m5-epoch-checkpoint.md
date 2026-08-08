# Continuous Streaming 3.0 — M5 Epoch Checkpoint API Note

| Field             | Value                                                                    |
| ----------------- | ------------------------------------------------------------------------ |
| Status            | **Proposed — implementation contract**                                   |
| Priority          | **P0 — companion to the M5 epoch checkpoint specification**              |
| Baseline          | `main@a5fc2c395e347041f8d16384be99af7e23d2ebff`                          |
| Milestone         | M5 — epoch coordination, recovery, and exactly-once                      |
| Artifact slug     | `m5-epoch-checkpoint`                                                    |
| Controlling input | `specs/m5-epoch-checkpoint.md`, D1–D12 and AC-01–AC-48                   |
| Visibility        | crate-private runtime additions; existing public v2 and M4 APIs remain   |
| Intended audience | Rust runtime, state, connector, compiler, test, and benchmark owners     |

## 1. Purpose and compatibility boundary

This note translates the M5 specification into concrete Rust ownership,
module, type, channel, and call-flow decisions. The specification controls
semantics; this note controls implementation shape where multiple designs
could satisfy those semantics.

M5 is additive relative to the baseline:

- it reuses the public M4 `CheckpointManifest`, `StateHandle`,
  `StateBackend`, and `LocalStateBackend` types;
- it adds crate-private runtime coordination and sink lifecycle types;
- it keeps public v2 checkpoint stores and push runners unchanged;
- it does not expose the crate-private source-driven job or checkpoint control;
- it adds no Python, Studio, REST, OpenAPI, project-schema, or production
  connector surface.

The final M5 integration commit may reorganize private modules, but it MUST
not publish incomplete coordinator or transaction types through `lib.rs`.

## 2. Module layout and ownership

The intended implementation layout is:

```text
crates/calc-flow/src/
  state/
    manifest.rs       existing final v3 model; identity/diagnostic split
    local.rs          existing segment backend; shared durable primitives
    transaction.rs    production manifest publish/select/retain transaction
  runtime/streaming/
    checkpoint/
      mod.rs          private re-exports
      command.rs      bounded coordinator/participant messages
      coordinator.rs  one-in-flight epoch state machine
      recovery.rs     manifest selection and gated restore orchestration
      transaction.rs  sink commit/abort/recover sequencing
    progress/
      driver.rs       exact cut and barrier fan-out through edge owner
      snapshot.rs     bounded durable semantic projection helpers
    source_task.rs    pause/cut/ack participation
    operator_task.rs  alignment, snapshot, restore participation
    sink_task.rs      ordinary and transactional execution paths
    job.rs            preflight capability proof and validated bindings
    runner.rs         private owner wiring and terminal checkpoint
    metrics.rs        bounded checkpoint/recovery measurements
```

If Rust module constraints make `state/transaction.rs` unsuitable, the same
production transaction MAY live under `runtime/streaming/checkpoint/store.rs`.
Its ownership rules and API remain unchanged. M5 MUST NOT create a duplicate
manifest model under `checkpoint/model.rs`.

The private runner owns, in drop order:

1. immutable validated plan and prepared capability proof;
2. lineage-exclusive state session and manifest transaction;
3. restored participant state and paused connector bindings;
4. checkpoint coordinator and bounded command/ack channels;
5. progress owner, source, operator, and sink tasks;
6. supervisor, cancellation token, reaper, status, and metrics.

## 3. Manifest transaction

### 3.1 Identity and diagnostic validation

`ManifestExpectation` is split so operational configuration is not semantic
identity:

```rust
pub struct ManifestExpectation<'a> {
    pub pipeline_name: &'a str,
    pub pipeline_fingerprint: &'a str,
    pub epoch: Epoch,
    pub source_ids: &'a BTreeSet<String>,
    pub operator_ids: &'a BTreeSet<String>,
    pub sink_ids: &'a BTreeSet<String>,
}

pub(crate) struct ManifestDiagnostics<'a> {
    pub runtime_config_hash: &'a str,
}

pub(crate) struct ManifestValidation {
    pub runtime_config_changed: bool,
}
```

`CheckpointManifest::validate` retains strict version, checksum, semantic
identity, epoch, participant, and handle validation. A private comparison
returns `ManifestValidation` after semantic validation. It never changes or
rewrites loaded manifest bytes.

If changing the public field set would be unnecessarily disruptive, M5 MAY
retain `runtime_config_hash` in `ManifestExpectation` but MUST remove it from
the failing identity predicate and return the same diagnostic result through a
private method. Tests, not layout preference, enforce AC-05.

### 3.2 Production transaction boundary

The runner receives an opened lineage transaction rather than a raw manifest
directory:

```rust
pub(crate) struct ManifestTransaction {
    lineage: Arc<dyn StateLineageBackend>,
    manifest_root: PathBuf,
    retention: CheckpointRetention,
}

impl ManifestTransaction {
    pub(crate) async fn select_latest(
        &self,
        expected: &PreparedManifestIdentity,
    ) -> Result<Option<SelectedManifest>>;

    pub(crate) async fn publish(
        &self,
        prepared: PreparedEpochManifest,
    ) -> Result<PublishedManifest>;

    pub(crate) async fn load_operator_segments(
        &self,
        manifest: &CheckpointManifest,
    ) -> Result<BTreeMap<String, Vec<LoadedStateSegment>>>;

    pub(crate) async fn retain(
        &self,
        latest: &PublishedManifest,
        in_flight: Option<Epoch>,
    ) -> Result<RetentionReport>;
}
```

Constructors validate roots and acquire the existing lineage lease before an
instance is exposed. Return values are owned, immutable, and contain no open
file handles. Filesystem work and large hashing/serialization use asynchronous
I/O or an owned blocking worker. Publication reuses the exact M4 sequence:
stage, sync, validate, publish segments, validate manifest, durable temporary
manifest, atomic rename, parent sync.

`select_latest` enumerates bounded candidate names, rejects unexpected file
types and links, parses every candidate strictly, validates semantic identity
and segments, and chooses the highest valid completed epoch. A corrupt higher
candidate is reported as a deterministic recovery error rather than silently
falling back when it claims the same lineage; unrelated non-candidate files
also fail closed.

## 4. Durable progress projection

The durable projection is assembled directly into existing manifest entries:

```rust
pub(crate) struct DurableSourceProgress {
    pub cursor: Option<CursorManifestEntry>,
    pub identity_hash: String,
    pub next_sequence: u64,
    pub ended: bool,
    pub watermark_policy: SourceWatermarkManifestState,
}

pub(crate) struct DurableOperatorProgress {
    pub ingresses: BTreeMap<String, OperatorIngressManifestEntry>,
}
```

There is deliberately no serializable equivalent of
`StreamProgressDriverSnapshot`. The private conversion boundary reads only
settled semantic values at a coordinator-owned cut. It rejects pending
receipts or an unsettled driver instead of persisting them.

Restore constructs a fresh driver through normal `prepare_stream_job`, then
installs validated semantic state before task spawn. It allocates a new trace
generation and receipt counter, rebases idle/generated-watermark timers from
the new logical origin, and publishes a fresh bounded status snapshot. Full
configured delays are used; elapsed wall time during downtime is not inferred.

## 5. Coordinator messages and state

### 5.1 Commands and acknowledgements

All message enums are crate-private, bounded, payload-conscious, and carry the
stable participant ID explicitly:

```rust
pub(crate) enum CheckpointRequest {
    Periodic,
    Terminal { source_cut: BTreeMap<String, SourceCut> },
}

pub(crate) enum SourceCheckpointCommand {
    Cut { epoch: Epoch },
    Cancel { epoch: Epoch },
}

pub(crate) enum CheckpointAck {
    Source(SourceCheckpointAck),
    Operator(OperatorCheckpointAck),
    Sink(SinkCheckpointAck),
    TerminalOperator(TerminalOperatorAck),
    TerminalSink(TerminalSinkAck),
}

pub(crate) struct SourceCheckpointAck {
    pub source_id: String,
    pub epoch: Epoch,
    pub progress: DurableSourceProgress,
}

pub(crate) struct OperatorCheckpointAck {
    pub operator_id: String,
    pub epoch: Epoch,
    pub progress: DurableOperatorProgress,
    pub snapshot: OperatorStateSnapshot,
}

pub(crate) struct SinkCheckpointAck {
    pub output_id: String,
    pub sink_id: String,
    pub epoch: Epoch,
    pub delivery: SinkDeliveryManifest,
    pub pre_commit: Option<JsonMap>,
}
```

The coordinator canonicalizes acknowledgements into `BTreeMap`s. Identical
duplicates are idempotent; conflicting duplicates fail. Connector-owned JSON
is validated for size, depth, keys, and canonical form before it enters state.
Status and errors include IDs and phases but never log cursor or pre-commit
payloads.

### 5.2 Epoch state machine

```rust
pub(crate) struct CheckpointCoordinator {
    next_epoch: Epoch,
    in_flight: Option<CheckpointEpochState>,
    expected: ParticipantSet,
    requests: BoundedFifo<CheckpointRequest>,
    timeout: Duration,
    transaction: Arc<ManifestTransaction>,
}

pub(crate) enum CheckpointPhase {
    Requested,
    SourcesCut,
    OperatorsSnapshotted,
    SinksPrecommitted,
    ManifestDurable,
    SinksCommitted,
    Completed,
}
```

The actual bounded FIFO uses Tokio channels, not a new unbounded container.
`next_epoch` is allocated with `checked_add(1)`. Phase transitions validate
the complete expected participant set and use a deadline derived from the
validated runtime configuration. The coordinator task is supervised like all
other private runtime tasks and cancels siblings on failure.

## 6. Source cut through the progress owner

`LiveProgressCoordinator` already owns source-output senders. M5 adds one
serialized operation to that owner:

```rust
pub(crate) async fn checkpoint_cut(
    &self,
    epoch: Epoch,
    sources: BTreeMap<BindingIdentity, DurableSourceProgress>,
    cancellation: &CancellationToken,
) -> Result<Vec<SourceCheckpointAck>>;
```

The concrete signature MAY split request registration from acknowledgement,
but edge ownership MUST remain here. The operation acquires the same serial
drive boundary used for progress emission, proves every pre-cut submission is
settled, dispatches any ready progress first, then sends the barrier through
the owned routes. It sends no barrier for ended bindings and returns their
terminal cut out of band.

Each source task has a bounded command receiver. On `Cut(E)` it disables poll,
finishes its at-most-one admitted/prefetched item under the existing cursor
contract, supplies observed cursor and policy state to the progress owner, and
waits for cut completion. Only then does it promote observed progress to
durable and resume polling.

Tests use a progress-owner pause hook, not direct edge injection, to prove no
data, generated watermark, idle, or end can overtake the barrier.

## 7. Operator alignment and snapshot

Each operator ingress adds:

```rust
struct OperatorIngress {
    receiver: EdgeReceiver,
    saw_explicit_eof: bool,
    barrier: IngressBarrierState,
}

enum IngressBarrierState {
    Open,
    Blocked(Epoch),
    Ended,
}
```

`receive_ready` filters out `Blocked` and `Ended` ingresses. If every live
ingress is blocked for expected epoch E, the task invokes a separate
`complete_alignment(E)` path. It must never drain a blocked channel into an
auxiliary buffer.

`complete_alignment` calls the compiled operator checkpoint dispatch, validates
snapshot identity and bounds, sends the operator ack, forwards the barrier
through normal bounded output senders, and resets live ingresses to `Open`.
The forward occurs after local state capture/staging succeeds and before the
global manifest completes. Any error leaves ingresses owned by the failing
task and cancels the job; it does not attempt to resume data processing.

Restore calls compiled operator `reset`, then `restore` with validated inline
metadata and loaded segments, then installs ingress progress. No operator
handler is invoked before all operators restore successfully.

## 8. Sink binding and transaction lifecycle

### 8.1 Private binding variants

The validated job distinguishes ordinary and checkpoint-aware sinks:

```rust
pub(crate) enum ValidatedSinkBinding {
    Ordinary(ValidatedOrdinarySink),
    Transactional(ValidatedTransactionalSink),
    EpochIdempotent(ValidatedEpochIdempotentSink),
}

#[async_trait]
pub(crate) trait TransactionalStreamSink: Send {
    async fn open(&mut self, context: SinkOpenContext) -> Result<()>;
    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()>;
    async fn write(&mut self, batch: Batch) -> Result<()>;
    async fn pre_commit(&mut self, epoch: Epoch) -> Result<JsonMap>;
    async fn commit(&mut self, epoch: Epoch, state: &JsonMap) -> Result<()>;
    async fn abort(&mut self, epoch: Epoch, state: Option<&JsonMap>) -> Result<()>;
    async fn recover(&mut self, manifest: &CheckpointManifest) -> Result<()>;
    async fn close(&mut self) -> Result<()>;
}
```

This trait is a private M5 execution seam, not the public M6 connector API.
The implementation may adapt existing sink ownership rather than duplicate
open/write/close. It MUST preserve asynchronous lifecycle calls, batch
immutability, and cancellation.

### 8.2 Data-plane and commit-plane split

The sink task owns write, begin, and pre-commit. It sends pre-commit metadata
to the coordinator and continues with the next epoch only after a bounded
coordinator command authorizes it. External commit/abort MAY execute in the
same sink task to preserve single-owner mutation; the coordinator sends
`ManifestDurable(E)` or `Abort(E)` rather than calling the sink concurrently.

After manifest durability, a commit error fails the live job but preserves
the manifest and transaction for forward recovery. Before manifest durability,
failure sends abort to every participant that prepared E. Abort errors are
reported but never cause publication or broaden cleanup.

A filesystem-backed deterministic transactional sink lives under focused test
support and writes epoch-staged files plus an idempotent committed marker. It
is sufficient for protocol and fault evidence but is not exported or described
as the production file connector promised by M6.

## 9. Capability derivation

Preflight builds one deterministic proof per plan output:

```rust
pub(crate) struct OutputDeliveryProof {
    pub output_id: String,
    pub requested: DeliveryGuarantee,
    pub reachable_sources: BTreeSet<String>,
    pub reachable_operators: BTreeSet<String>,
    pub sink_mechanisms: BTreeMap<String, SinkDeliveryManifest>,
}
```

The compiler already rejects volatile UDFs for exactly-once. Runtime preflight
adds source replay/seek, progress projection, operator snapshot/restore, edge
losslessness, sink mechanism, and idempotency retention checks using the
consumed immutable plan. The error field is
`requirements.delivery.<output_id>` and the message names the first stable
component ID and missing capability. Validation completes before any `open`,
`reset`, `restore`, lease-mutating transaction, or spawn.

## 10. Normal checkpoint call flow

For periodic epoch E:

```text
coordinator allocates E and sends Cut(E) to all sources
sources stop poll and submit their settled observed positions
progress owner drains ready pre-cut emissions and fans out Barrier(E)
sources promote observed positions to durable and acknowledge E
operators block arrived ingresses and continue other ingresses
operators align, checkpoint/stage state, acknowledge, forward Barrier(E)
sinks pre-commit after all pre-barrier data and acknowledge E
coordinator validates the complete canonical acknowledgement set
manifest transaction publishes state and manifest last
coordinator sends ManifestDurable(E) to every transactional sink owner
sinks commit idempotently and acknowledge completion
coordinator records completed E, runs retention, and services next request
```

Retention failure after manifest publication is a job failure but does not
invalidate E. Recovery still selects E. Status distinguishes commit success
from later maintenance failure without adding another durable truth.

## 11. Recovery call flow

```text
compile and preflight immutable plan
open lineage-exclusive manifest transaction
select and validate latest completed manifest
load and validate all referenced operator segments
construct fresh progress driver and install semantic projection
reset and restore every operator while task gates remain closed
open sources paused; seek non-ended bindings to exact cursors
open sinks; recover manifest-described epoch idempotently
if manifest is terminal: close resources and return completed
otherwise allocate manifest epoch + 1, spawn tasks, and release gates
```

An empty lineage starts at epoch one with reset operators and newly opened
sources/sinks. An invalid lineage never falls back to empty state.

## 12. Terminal checkpoint call flow

```text
all sources report final cursor and end through the progress owner
operators consume end, emit final-only outputs, and report terminal state
sinks consume final output and end but remain recoverable
coordinator allocates the next terminal epoch out of band
operators capture post-on_end state for that epoch
sinks pre-commit final output for that epoch
coordinator publishes the final manifest
sinks commit idempotently
job settles every task and returns completed
```

No `StreamMessage::barrier` is created after an edge has observed end. A
terminal manifest restore executes only sink recovery and resource closure;
it never calls source poll or operator `on_end` again.

The manifest model gains no terminal boolean. Recovery classifies an epoch as
terminal only when the exact prepared source set is present and every source
has `ended = true`, and the exact operator ingress sets are present and every
ingress state is `Ended`. A mixed ended/non-ended terminal shape fails
validation. `RecoveryStatus::Final` still denotes a complete recoverable
manifest and is not used as the job-terminal discriminator.

## 13. Failure, cancellation, and observability

All failures become one deterministic private job error with phase, epoch, and
stable participant ID. Error sources preserve the underlying I/O/operator/sink
cause without serializing connector payload. Coordinator timeout, task panic,
edge closure, and cancellation follow the supervisor's existing first-failure
policy.

The coordinator owns cleanup intent, while resources remain single-owner:

- source tasks close sources;
- operator tasks drop owned operator state;
- sink tasks abort only pre-manifest transactions and close sinks;
- manifest transaction removes only validated unreachable artifacts;
- runner reaper waits for all supervised and blocking work.

Metrics use existing checked recorder patterns. Epoch and participant IDs are
not unbounded labels. Private status snapshots expose counts and phases only.

## 14. Test placement and named evidence

Crate-private coordination tests live beside their implementation under
`runtime/streaming`. Public M4 manifest/state regression tests remain under
`crates/calc-flow/tests/`. Focused test modules cover:

- `state::transaction`: crash-boundary publication, strict selection,
  diagnostics-only runtime hash, retention, and corrupt candidates;
- `progress::driver`: exact source cut and timer non-overtaking;
- `source_task`: paused cursor cut, idle, ended, cancel, and reopen behavior;
- `operator_task`: all two-input arrival orders, blocked ingress, snapshot
  failure, future/regressed epoch, and boundedness;
- `sink_task`/`checkpoint::transaction`: pre-commit ordering, abort, partial
  multi-sink commit, idempotent recover, and terminal output;
- `runner`: gated recovery order, terminal manifest, failure cleanup, and task
  zero;
- fault support: deterministic cancellation, I/O, panic, and restart at every
  D12 boundary.

Every behavior change begins with one focused failing test and records its
expected failure in the implementation PR. Tests use paused Tokio time where
possible and explicit gates rather than sleeps.

## 15. Benchmark and soak interfaces

Criterion pairs compare the same baseline and candidate payloads for:

- no-checkpoint steady-state data path;
- barrier cut/fan-out with one and two sources;
- aligned two-input pass-through and window operators;
- dirty-window state stage plus manifest publication;
- cold restore with retained deltas and compacted base state;
- one and multiple transactional sink commits.

Noise reporting includes raw samples, median, confidence interval, and the
exact baseline/candidate SHAs. A regression above 5% blocks merge unless an
approved explanation and follow-up threshold are recorded.

The soak command is implemented as a reproducible repository script or
ignored test with a `--duration 20m` default. It emits machine-readable
one-minute samples and a final summary containing raw SHA, restart schedule,
source/output counts, duplicates, missing values, epochs, task/queue/state/
manifest bounds, and terminal resource counts. Future continuous-streaming
soaks use the same 20-minute standard unless a later approved plan changes it.

## 16. Implementation commit sequence

Recommended narrow commits on the M5 integration branch are:

1. `Implement production manifest transactions`
2. `Add durable streaming progress restore`
3. `Coordinate source checkpoint cuts`
4. `Align operator checkpoint barriers`
5. `Add transactional sink checkpoints`
6. `Recover complete checkpointed jobs`
7. `Checkpoint terminal streaming output`
8. `Add M5 fault and soak evidence`

Each commit keeps the relevant focused tests green. Stacked review PRs target
the integration branch; the final implementation PR to `main` contains the
complete protocol and exact-head evidence.
