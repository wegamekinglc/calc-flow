# Continuous Streaming 3.0 — M3 Delta API Note

| Field             | Value                                                             |
| ----------------- | ----------------------------------------------------------------- |
| Status            | **Proposed — implementation contract, fourth revision**           |
| Priority          | **P0 — companion to the M3 delta specification**                  |
| Baseline          | `main@88243e9565795cd0bce01a155f75e735f4b47728`                   |
| Milestone         | M3 — event-time progress and watermark coordination               |
| Visibility        | crate-private only; no public v2, Python, REST, or OpenAPI change |
| Controlling input | `m3-delta-spec.md`, D1–D9, FR-001–FR-079, AC-01–AC-85            |
| Review input      | `m3-delta-critique.md`, round 3                                   |
| Intended audience | runtime, connector, compiler, test, implementation, review owners |

## 1. Purpose, authority, and compatibility rule

This note translates the controlling fourth-revision M3 delta specification into implementable internal Rust boundaries. It is normative for M3-owned information content, visibility, ownership, ordering, admission, fencing, terminal transitions, receipt settlement, replay, and phase-correct error behavior. Existing baseline types such as the logical graph, compiled stream, record batch, source registry, output sink, cancellation token, channel, and public error family retain their baseline names and public contracts.

Precedence is:

```text
m3-delta-spec.md
> this API note
> compatible baseline implementation details
```

Normative and evidentiary sources are:

1. `m3-delta-spec.md`, the controlling M3 delta;
2. `docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md`;
3. the repository's controlling continuous-streaming runtime specification;
4. PR #83's final M2 implementation, acceptance evidence, review resolution, CI/Codacy evidence, and 20-minute soak;
5. baseline commit `88243e9565795cd0bce01a155f75e735f4b47728`;
6. round-two critique findings B-R2-01, B-R2-02, and N-R2-01;
7. round-three critique findings B-R3-01 and N-R3-01.

All new items below are `pub(crate)` at most and are never re-exported. Private fields and constructors are intentional. A mechanically equivalent crate-private representation is allowed only when the implementation PR maps it field-for-field to this note and preserves every ownership and linearization rule.

## 2. Public-surface and milestone firewall

### 2.1 Zero-public-change declaration

```text
public Rust API after M3 == public Rust API at baseline
Python surface after M3 == Python surface at baseline
REST/OpenAPI after M3 == REST/OpenAPI at baseline
public connector trait/open after M3 == baseline
```

M3 MUST NOT add a public error variant, connector hook, `open()` parameter, runner, binding type, Python object, REST route, or OpenAPI schema. It reuses `calc_flow::Result<T>` and the existing public error container.

`compile_stream()` remains the pure graph/operator compiler. Source binding, schema, capability, queue/fence configuration, and watermark policy enter only the crate-private whole-job preparation boundary.

### 2.2 Milestone firewall

- M3 owns crate-private progress preparation, generated/native watermark normalization, idleness, multi-input progress, complete admission/drain/terminal/settlement execution tracing, completion receipts, and transient exact-coordinate replay.
- M3 never classifies, counts, tags, splits, or drops a row or assignment as late and exposes no late-row metric.
- M4 first forms concrete window assignments and applies final-only drop only when `assignment.window_end <= WM_in`.
- M5 owns versioned durable encoding, checkpoint barriers, manifests, process restart, deadline rebasing, and exactly-once recovery.
- Public source-driven runner A6 remains post-M5.
- Allowed lateness, side output, and retraction remain Continuous Streaming 3.0 non-goals.

### 2.3 Decision-to-API register

| Decision | Binding API decision |
| -------- | -------------------- |
| D1 | `DriverEmission::ForwardData` carries accepted data unchanged; no late branch, counter, tag, or error exists. |
| D2 | `CapturedLogicalCoordinate`, exact paused upstreams, the completed `ProgressExecutionTrace` prefix, and a consuming `ProgressReplayRequest` define same-process replay; the drain projection alone is insufficient. |
| D3 | unchanged `compile_stream()` is followed by side-effect-free `prepare_stream_job`; all policy/schema/capability/config failures precede open/spawn/timer/pull. |
| D4 | scratch evaluation follows `ReadyKey` order only until the first error; the whole selected snapshot commits or fails atomically, with complete receipt settlement. |
| D5 | one job-scoped driver owns logical time, admission/trace/gate/settlement coordinates, drain epochs, inbox fences, ready keys, timers, progress, status, and terminal cleanup. |
| D6 | a complete 4 × 3 capability matrix produces exactly one normalized mode and never silently merges native/generated watermarks. |
| D7 | every M3 addition remains crate-private and A6 remains post-M5. |
| D8 | one serialized admission/inbox protocol owns accepted envelopes, closes a committed-End gate, extracts the post-fence terminal tail, and settles every sender exactly once before exit. |
| D9 | one complete execution-trace state machine prevalidates and consumes admission, drain, terminal, settlement, and phase-failure records before governed effects; pre-key errors never fabricate `ReadyKey`. |

## 3. Internal module layout

The implementation SHOULD use existing stream/runtime modules. Equivalent placement is allowed, but no M3 file may import an M4 window executor.

```text
crates/calc-flow/src/
  stream/
    watermark_policy.rs    policy, descriptors, normalization
    prepare.rs             whole-job preparation and fingerprints
  runtime/
    progress.rs            one driver, admission, drain arbitration
    progress/
      aggregate.rs         InputProgress and MultiInputProgress
      trace.rs             complete execution trace and drain projection
      generated.rs         generated watermark state
      snapshot.rs          transient capture, exact restore/replay
      status.rs            crate-private status and bounded counters
```

There is no dependency on `operator/window.rs`.

## 4. Shared internal values, checked allocators, and ordering

### 4.1 Identity, time, and semantic counters

```rust
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct BindingIdentity(Arc<str>);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct BindingOrdinal(u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct LocalSequence(u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GlobalSequence(u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct TimerSequence(u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct TimerGeneration(u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct IdleEpoch(u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct ReceiptSequence(u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct InboxSequence(u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct FenceSequence(u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct DrainEpoch(u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct TraceRecordOrdinal(u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct TracePosition(u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct AdmissionAttemptOrdinal(u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct AdmissionGateGeneration(u64);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct GateCloseOrdinal(u64);

/// Canonical trace order only; never receipt send/wakeup order.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct SettlementOrdinal(u64);

/// Canonical nanoseconds from the Arrow timestamp epoch.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct EventTime(i128);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct LogicalInstant(u128);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ClockTraceFingerprint([u8; 32]);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DriverClockCoordinate {
    trace: ClockTraceFingerprint,
    instant: LogicalInstant,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PreparedJobFingerprint([u8; 32]);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct NormalizedConfigFingerprint([u8; 32]);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct RuntimeFenceConfigFingerprint([u8; 32]);

#[derive(Clone, Debug, Eq, PartialEq)]
struct CheckedSemanticAllocator {
    next: u64,
    error_path: &'static str,
}

impl CheckedSemanticAllocator {
    /// Returns a value only when its successor is representable.
    fn checked_peek_and_successor(&self) -> crate::Result<(u64, u64)>;

    fn allocate(&mut self) -> crate::Result<u64> {
        let (value, successor) = self.checked_peek_and_successor()?;
        self.next = successor;
        Ok(value)
    }
}
```

`TraceRecordOrdinal`, `TracePosition`, `AdmissionAttemptOrdinal`, `AdmissionGateGeneration`, `GateCloseOrdinal`, `SettlementOrdinal`, `ReceiptSequence`, `InboxSequence`, `FenceSequence`, `DrainEpoch`, ordinal, local/global sequence, timer generation/sequence, and idle epoch all use this checked rule. They never wrap, saturate, reuse a value, or allocate a value without a representable successor. Admission probes every required successor first and commits all governed allocator updates together; failure consumes none. Drain and terminal preparation similarly probe all required coordinates before committing any governed effect.

Event-time unit conversion, watermark subtraction, and logical deadline addition/multiplication use checked arithmetic. Observational counters alone may saturate and cannot affect semantics.

### 4.2 Driver clock and ready order

```rust
pub(crate) trait DriverLogicalClock: Send + Sync + 'static {
    fn coordinate(&self) -> DriverClockCoordinate;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum ReadyClass {
    InputOrControl = 0,
    WatermarkTimer = 1,
    IdleTimer = 2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReadyKey {
    logical_instant: LogicalInstant,
    class: ReadyClass,
    binding_ordinal: BindingOrdinal,
    local_sequence: LocalSequence,
}

impl Ord for ReadyKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.logical_instant,
            self.class,
            self.binding_ordinal,
            self.local_sequence,
        )
            .cmp(&(
                other.logical_instant,
                other.class,
                other.binding_ordinal,
                other.local_sequence,
            ))
    }
}

impl PartialOrd for ReadyKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
```

Only the driver creates `ReadyKey`. All selected input/control and due timers in one drain use the one driver clock coordinate captured for that epoch. Timer generation and timer sequence are absent from semantic ordering; generation only proves stale/current identity.

### 4.3 Runtime queue/fence configuration

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FenceSelectionPolicy {
    AllVisible,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StreamProgressRuntimeConfig {
    per_binding_inbox_capacity: NonZeroUsize,
    fence_selection: FenceSelectionPolicy,
}
```

M3 supports `AllVisible`: one fence includes every already-accepted envelope through the inclusive upper sequence observed when that binding is frozen. Capacity and selection policy are semantic because they can change drain grouping. They are normalized during preparation, covered by both the prepared-job and runtime-fence fingerprints, stored in the prepared job/snapshot, and compared exactly on restore. `StreamProgressDriver::new` and `restore` accept no free capacity or selection-policy parameter.

Output backpressure and natural scheduler timing are not replayed as wall-clock facts. Any effect they have on semantic grouping is represented only by the resulting recorded drain epochs and fences.

## 5. Complete progress execution trace and replay vocabulary

### 5.1 Accepted identity, gate state, and drain projection values

```rust
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct AcceptedEnvelopeIdentity {
    binding: BindingIdentity,
    binding_ordinal: BindingOrdinal,
    admission_attempt: AdmissionAttemptOrdinal,
    receipt_sequence: ReceiptSequence,
    inbox_sequence: InboxSequence,
    upstream_position: RawUpstreamPosition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InboxUpperFence {
    Empty,
    Inclusive(InboxSequence),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InboxFenceCoordinate {
    drain_epoch: DrainEpoch,
    binding_ordinal: BindingOrdinal,
    fence_sequence: FenceSequence,
    upper: InboxUpperFence,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct TimerIdentity {
    binding_ordinal: BindingOrdinal,
    kind: TimerKind,
    deadline: LogicalInstant,
    generation: TimerGeneration,
    timer_sequence: TimerSequence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RawIngressEventKind {
    Data,
    ConnectorWatermark,
    ConnectorIdle,
    EndOfInput,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SelectedRawTraceIdentity {
    accepted_identity: AcceptedEnvelopeIdentity,
    event_kind: RawIngressEventKind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReadyItemIdentity {
    Raw(SelectedRawTraceIdentity),
    Timer(TimerIdentity),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReadyKeyRange {
    Empty,
    Inclusive { first: ReadyKey, last: ReadyKey },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AdmissionGateSnapshot {
    state: AdmissionGateState,
    generation: AdmissionGateGeneration,
    close: Option<AdmissionGateCloseCoordinate>,
}
```

An accepted identity is globally equality-complete for M3: it contains the binding identity/ordinal, admission attempt, receipt, binding-local inbox sequence, and exact upstream cursor/control identity. An inbox fence exists for every prepared binding in every drain epoch. `Empty` selects no raw envelope; `Inclusive(s)` selects every queued envelope through `s`. Above-fence accepted work waits for a later drain unless an End/cancel/fatal transition extracts it.

### 5.2 Admission attempts

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionDecisionRecord {
    Accepted {
        accepted: AcceptedEnvelopeIdentity,
    },
    ImmediateRejected {
        error: StableErrorRecord,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AdmissionAttemptRecord {
    trace_record_ordinal: TraceRecordOrdinal,
    trace_position: TracePosition,
    attempt_ordinal: AdmissionAttemptOrdinal,
    binding: BindingIdentity,
    binding_ordinal: BindingOrdinal,
    event_kind: RawIngressEventKind,
    upstream_position: RawUpstreamPosition,
    observed_gate: AdmissionGateSnapshot,
    decision: AdmissionDecisionRecord,
}
```

Every call to submit gets exactly one checked attempt ordinal before gate observation and exactly one admission record before its result is visible. `Accepted` owns a receipt/inbox identity and later exactly one settlement. `ImmediateRejected` owns neither receipt/inbox identity nor settlement. If the attempt ordinal or trace-record ordinal itself is exhausted, no attempt record can be allocated; the call fails as a pre-key `DriverPhaseError` at the last exact coordinate and accepts nothing.

### 5.3 Drain records are a projection, not the replay contract

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DrainEpochOutcomeRecord {
    Committed,
    SelectedItemFailed(SelectedItemErrorRecord),
    DriverPhaseFailed(DriverPhaseErrorRecord),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DrainEpochRecord {
    trace_record_ordinal: TraceRecordOrdinal,
    trace_position: TracePosition,
    epoch: DrainEpoch,
    driver_clock: DriverClockCoordinate,
    inbox_fences: Vec<InboxFenceCoordinate>,
    selected_items_in_ready_order: Vec<ReadyItemIdentity>,
    selected_key_range: ReadyKeyRange,
    due_timers_in_ready_order: Vec<TimerIdentity>,
    outcome: DrainEpochOutcomeRecord,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DrainFenceTrace {
    records: Vec<DrainEpochRecord>,
}
```

Each drain records all bindings' fences, selected identities/order/key range, due timers, required empty epochs, and exactly one outcome. A `DriverPhaseFailed` drain has an epoch but no selected-item key. If epoch allocation itself failed, no drain record or invented epoch is created; the full trace may instead end with the last representable phase-failure coordinate.

`DrainFenceTrace` is only `ProgressExecutionTrace::drain_projection()`. It deliberately omits admission, terminal/gate-close, settlement, and non-drain phase-failure records and MUST NOT be named as sufficient for deterministic receipt replay.

### 5.4 Terminal and gate-close records

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TerminalTransitionCause {
    EndCommit,
    Cancellation,
    Fatal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct AdmissionGateCloseCoordinate {
    close_ordinal: GateCloseOrdinal,
    cause: TerminalTransitionCause,
    old_generation: AdmissionGateGeneration,
    new_generation: AdmissionGateGeneration,
    closed_state: AdmissionGateState,
    next_inbox_sequence_cut: InboxSequence,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BindingGateTransitionRecord {
    binding: BindingIdentity,
    binding_ordinal: BindingOrdinal,
    close: AdmissionGateCloseCoordinate,
    extracted_tail: Vec<AcceptedEnvelopeIdentity>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TerminalTransitionRecord {
    trace_record_ordinal: TraceRecordOrdinal,
    trace_position: TracePosition,
    cause: TerminalTransitionCause,
    driver_clock: DriverClockCoordinate,
    owning_drain: Option<DrainEpoch>,
    transitions_in_binding_order: Vec<BindingGateTransitionRecord>,
}
```

`EndCommit` has an owning drain and normally one affected binding; cancel/fatal may have no drain and include every open binding in binding-ordinal order. Each close record freezes the old/new generation, stable cause/state, checked close ordinal, `next_inbox_sequence` cut, and exact accepted tail extracted at that cut. Tail identities are canonically ordered by accepted identity. Replay never rediscovers this boundary from scheduler timing.

### 5.5 Exactly-one settlement records

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SettlementOwner {
    Drain(DrainEpoch),
    TerminalTransition(TraceRecordOrdinal),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SettlementDispositionRecord {
    CommitSuccess,
    TransactionError(SelectedItemErrorRecord),
    PostEndTailReject,
    Cancelled,
    Fatal(DriverFailureRecord),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum StableSettlementOutcome {
    Success {
        committed_upstream_position: RawUpstreamPosition,
    },
    Failure(StableErrorRecord),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SettlementRecord {
    trace_record_ordinal: TraceRecordOrdinal,
    trace_position: TracePosition,
    settlement_ordinal: SettlementOrdinal,
    accepted: AcceptedEnvelopeIdentity,
    owner: SettlementOwner,
    disposition: SettlementDispositionRecord,
    outcome: StableSettlementOutcome,
}
```

Each accepted identity appears in exactly one `SettlementRecord`; immediate rejections appear in none. Within one settlement batch records are canonical by `(binding_ordinal, inbox_sequence, receipt_sequence)`. `SettlementOrdinal` is a checked canonical trace coordinate only. It is not receipt send/wakeup order, and a dropped receiver never changes the recorded disposition/outcome.

### 5.6 Full trace and drain projection

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProgressTraceRecord {
    Admission(AdmissionAttemptRecord),
    Drain(DrainEpochRecord),
    Terminal(TerminalTransitionRecord),
    Settlement(SettlementRecord),
    DriverPhaseFailure(DriverPhaseFailureTraceRecord),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProgressExecutionTrace {
    records: Vec<ProgressTraceRecord>,
}

impl ProgressExecutionTrace {
    pub(crate) fn drain_projection(&self) -> DrainFenceTrace;
}
```

`ProgressExecutionTrace` is ordered, lossless, in-memory, and the only trace sufficient for M3 receipt replay. Every record has a checked trace ordinal/position. It covers each admission decision, drain/fence/outcome, End/cancel/fatal gate transition and close cut, exactly-one settlement, and recordable driver-phase failure. The drain projection remains useful for progress debugging but is never a replay substitute.

### 5.7 Complete-plan validation and sequential consumption

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RawAttemptTraceIdentity {
    binding: BindingIdentity,
    event_kind: RawIngressEventKind,
    upstream_position: RawUpstreamPosition,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProgressReplayRequest {
    completed_prefix: ProgressExecutionTrace,
    raw_attempts: Arc<[RawAttemptTraceIdentity]>,
    logical_clock_trace: Arc<[DriverClockCoordinate]>,
    expected_suffix: ProgressExecutionTrace,
    declared_completion_position: TracePosition,
}

pub(crate) struct PrevalidatedProgressReplay {
    prefix: ProgressExecutionTrace,
    expected_suffix: Arc<[ProgressTraceRecord]>,
    raw_attempts: Arc<[RawAttemptTraceIdentity]>,
    logical_clock_trace: Arc<[DriverClockCoordinate]>,
    completion_position: TracePosition,
}

struct ProgressReplayCursor {
    expected_suffix: Arc<[ProgressTraceRecord]>,
    next_record_index: usize,
    next_trace_position: TracePosition,
}

impl ProgressReplayRequest {
    pub(crate) fn prevalidate(
        self,
        captured: &CapturedLogicalCoordinate,
    ) -> Result<PrevalidatedProgressReplay, DriverPhaseError>;
}

impl ProgressReplayCursor {
    fn consume_admission(
        &mut self,
        actual: &AdmissionAttemptRecord,
    ) -> Result<(), DriverPhaseError>;

    fn consume_drain(
        &mut self,
        actual: &DrainEpochRecord,
    ) -> Result<(), DriverPhaseError>;

    fn consume_terminal(
        &mut self,
        actual: &TerminalTransitionRecord,
    ) -> Result<(), DriverPhaseError>;

    fn consume_settlements(
        &mut self,
        canonical_actual: &[SettlementRecord],
    ) -> Result<(), DriverPhaseError>;

    fn consume_phase_failure(
        &mut self,
        actual: &DriverPhaseFailureTraceRecord,
    ) -> Result<(), DriverPhaseError>;

    fn finish(self) -> Result<(), DriverPhaseError>;
}
```

`prevalidate` runs before driver construction, timer installation, source resume, admission, receipt, cursor, state, or emission effect. It validates exact prefix/position, contiguous trace/attempt/gate/settlement coordinates, raw-attempt/admission one-to-one order, accepted-identity uniqueness, exactly one settlement per acceptance and none per rejection, valid settlement owners, disjoint selected/tail/cancel/fatal membership, terminal generations/cuts/tails, complete logical-clock coordinates, and a suffix ending exactly at `declared_completion_position`. Missing, extra, reordered, wrong-owner, too-short, too-long, and structurally mismatched plans fail here when detectable.

During replay, the shared trace machine consumes the next exact record before its governed effect becomes visible:

- admission record before enqueue/receipt return or immediate rejection;
- drain record before epoch state/timer/cursor/emission commit;
- terminal record while holding the gate commit guard and before gate closure/tail extraction;
- canonical settlement records before any receipt send;
- phase-failure record before fatal cleanup effects.

For an End/cancel/fatal race, replay waits without exposing closure until every tail identity recorded as accepted has arrived. It then reacquires the gate guard, rejects any unexpected accepted identity/cut/generation, consumes terminal and settlement records, and commits. `finish` rejects any extra expected suffix; an unexpected actual record fails at the current trace position. Effects from earlier completely validated records remain committed, but a mismatched record and its owning epoch/transition expose none of their governed effects.

The deterministic claim is exactly:

```text
same prepared job/config + exact captured coordinate
+ same ordered raw input/control attempts
+ same driver logical-clock trace
+ same post-capture ProgressExecutionTrace
=> item-identical admission decisions, emissions, receipt dispositions/results,
   selected-item and driver-phase errors, gate transitions, terminal outcome
```

The same raw attempts and clock trace with another full execution trace are a different logical execution. M3 promises neither acceptance/settlement grouping nor outcome equality across them. In particular, `DrainFenceTrace` equality alone does not distinguish “accepted End tail then failed receipt” from “immediate post-End rejection.”

## 6. Watermark policy, descriptors, and preparation

### 6.1 Raw and normalized policy

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum WatermarkPolicy {
    SourceProvided,
    BoundedOutOfOrderness {
        event_time_column: Arc<str>,
        max_out_of_orderness: Duration,
        emit_interval: Duration,
        idle_timeout: Option<Duration>,
    },
    Disabled {
        idle_timeout: Option<Duration>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ArrowTimestampUnit {
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedEventTimeColumn {
    name: Arc<str>,
    index: usize,
    unit: ArrowTimestampUnit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeWatermarkDirective {
    LeaveEnabled,
    EnableThroughExistingPrivateRoute,
    DisableThroughExistingPrivateRoute,
    AlreadyDisabled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NormalizedWatermarkMode {
    SourceProvided {
        native_directive: NativeWatermarkDirective,
    },
    Generated {
        event_time: ResolvedEventTimeColumn,
        max_out_of_orderness: Duration,
        emit_interval: Duration,
        idle_timeout: Option<Duration>,
        native_directive: NativeWatermarkDirective,
    },
    Disabled {
        idle_timeout: Option<Duration>,
        native_directive: NativeWatermarkDirective,
    },
}
```

Every applicable duration is positive and representable. `Generated` accepts only a statically known Arrow timestamp in second/millisecond/microsecond/nanosecond units. `SourceProvided` schedules no generated timer. `Generated` and `Disabled` reject connector watermarks at runtime. No policy fallback or native/generated merge exists.

### 6.2 Side-effect-free descriptor catalog

```rust
#[derive(Clone, Debug)]
pub(crate) enum DeclaredSchema {
    Known(SchemaRef),
    DynamicOrUnknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeWatermarkCapability {
    NeverEmits,
    EmitsNative,
    RuntimeToggleable,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReplayPositioningCapability {
    ExactPauseReportAndSeek,
    Unsupported,
}

#[derive(Clone, Debug)]
pub(crate) struct ExistingPrivateToggleRoute {
    route_id: Arc<str>,
}

#[derive(Clone, Debug)]
pub(crate) struct SourceDescriptor {
    binding: BindingIdentity,
    declared_schema: DeclaredSchema,
    native_watermarks: NativeWatermarkCapability,
    replay_positioning: ReplayPositioningCapability,
    existing_toggle_route: Option<ExistingPrivateToggleRoute>,
}

#[derive(Clone, Debug)]
pub(crate) struct SourceDescriptorCatalog {
    by_binding: std::collections::BTreeMap<BindingIdentity, SourceDescriptor>,
}

impl SourceDescriptorCatalog {
    pub(crate) fn collect(
        bindings: &[SourceBindingSpec],
        registry: &SourceRegistry,
    ) -> crate::Result<Self>;
}
```

Descriptor collection may inspect immutable configuration already in memory. It performs no connector open, I/O, pull, task spawn, timer registration, or mutation. `RuntimeToggleable` is legal only when the baseline already has the named private route. `ExactPauseReportAndSeek` is advertised only for a source that can pause, report equality-comparable cursor/control coordinates, and seek both exactly in the same process.

### 6.3 Capability matrix

| Descriptor capability | `SourceProvided` | `Generated` | `Disabled` |
| --------------------- | ---------------- | ----------- | ---------- |
| `NeverEmits` | reject | allow, `AlreadyDisabled` | allow, `AlreadyDisabled` |
| `EmitsNative` | allow, `LeaveEnabled` | reject | reject |
| `RuntimeToggleable` | allow only with existing private enable route | allow only with existing private disable route | allow only with existing private disable route |
| `Unknown` | reject | reject | reject |

The implementation uses one exhaustive table/match. A rejected cell is a preflight error before side effects.

### 6.4 Prepared job and whole-job preflight

```rust
#[derive(Clone, Debug)]
pub(crate) struct SourceBindingSpec {
    identity: BindingIdentity,
    source_node: SourceNodeId,
    watermark_policy: WatermarkPolicy,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedSourceBinding {
    identity: BindingIdentity,
    binding_table_index: usize,
    source_node: SourceNodeId,
    declared_schema: DeclaredSchema,
    normalized_watermark: NormalizedWatermarkMode,
    normalized_config_fingerprint: NormalizedConfigFingerprint,
    replay_positioning: ReplayPositioningCapability,
    existing_toggle_route: Option<ExistingPrivateToggleRoute>,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedStreamJob {
    compiled: CompiledStream,
    bindings: Arc<[PreparedSourceBinding]>,
    runtime_progress_config: StreamProgressRuntimeConfig,
    runtime_fence_config_fingerprint: RuntimeFenceConfigFingerprint,
    fingerprint: PreparedJobFingerprint,
}

pub(crate) fn prepare_stream_job(
    graph: &LogicalStreamGraph,
    bindings: &[SourceBindingSpec],
    runtime_progress_config: StreamProgressRuntimeConfig,
    registry: &SourceRegistry,
) -> crate::Result<PreparedStreamJob>;
```

Preparation performs, in order:

1. unchanged `compile_stream(graph)`;
2. deterministic compiled source-node ordering;
3. descriptor collection without side effects;
4. binding identity/cardinality/order validation;
5. duration, schema, column, capability, toggle-route, and replay capability normalization;
6. capacity/fence policy validation;
7. complete normalized binding, runtime-fence, and prepared-job fingerprints;
8. immutable prepared-job construction.

The prepared fingerprint covers compiled graph identity/order, ordered binding identities/config fingerprints, source capability decisions, exact per-binding inbox capacity, and exact fence selection. `PreparedStreamJob` contains no opened connector, live task, receiver, timer, or mutable runtime state. Every failure occurs before connector open, private-toggle application, task spawn, timer registration, or pull.

## 7. Stable internal and phase-correct errors

M3 adds no public error variant. It uses a cloneable stable payload plus two disjoint internal identity domains. Only selected-item evaluation after complete key assignment may carry a `ReadyKey`.

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProgressFailureKind {
    InvalidPolicy,
    InvalidSchema,
    CapabilityConflict,
    ProtocolViolation,
    SnapshotBoundary,
    SnapshotMismatch,
    ReplayMismatch,
    RestoreUnsupported,
    Arithmetic,
    CounterExhaustion,
    Cancelled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProgressFailure {
    kind: ProgressFailureKind,
    path: Arc<str>,
    reason: Arc<str>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StableErrorRecord {
    kind: ProgressFailureKind,
    path: Arc<str>,
    reason: Arc<str>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SelectedItemError {
    first_error_key: ReadyKey,
    failure: ProgressFailure,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SelectedItemErrorRecord {
    first_error_key: ReadyKey,
    error: StableErrorRecord,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DriverFailurePhase {
    AdmissionAttemptAllocation,
    AdmissionDecision,
    DrainEpochAllocation,
    FenceAllocation,
    InboxFreeze,
    ReadyKeyConstruction,
    ReplayPlanValidation,
    ReplayRecordValidation,
    GateClosePlanning,
    SettlementPlanning,
    SnapshotValidation,
    TerminalCleanup,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DriverPhaseCoordinate {
    Admission {
        last_attempt: Option<AdmissionAttemptOrdinal>,
        binding: BindingIdentity,
        gate_generation: Option<AdmissionGateGeneration>,
    },
    Drain {
        epoch: Option<DrainEpoch>,
        binding: Option<BindingOrdinal>,
        fence: Option<FenceSequence>,
    },
    Trace {
        last_record: Option<TraceRecordOrdinal>,
        position: TracePosition,
    },
    GateClose {
        binding: BindingOrdinal,
        generation: AdmissionGateGeneration,
        last_close: Option<GateCloseOrdinal>,
    },
    Counter {
        stable_path: Arc<str>,
        last_value: Option<u64>,
    },
    Snapshot {
        stable_path: Arc<str>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DriverPhaseError {
    phase: DriverFailurePhase,
    coordinate: DriverPhaseCoordinate,
    failure: ProgressFailure,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DriverPhaseErrorRecord {
    phase: DriverFailurePhase,
    coordinate: DriverPhaseCoordinate,
    error: StableErrorRecord,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DriverPhaseFailureTraceRecord {
    trace_record_ordinal: TraceRecordOrdinal,
    trace_position: TracePosition,
    error: DriverPhaseErrorRecord,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DriverFailureRecord {
    SelectedItem(SelectedItemErrorRecord),
    DriverPhase(DriverPhaseErrorRecord),
}

impl ProgressFailure {
    pub(crate) fn stable_record(&self) -> StableErrorRecord;

    pub(crate) fn into_existing_error(self) -> crate::Error;
}

impl SelectedItemError {
    pub(crate) fn into_existing_error(self) -> crate::Error;
}

impl DriverPhaseError {
    pub(crate) fn into_existing_error(self) -> crate::Error;
}
```

Required stable paths include:

```text
sources.<binding>.watermark_policy.event_time_column
sources.<binding>.watermark_policy.max_out_of_orderness
sources.<binding>.watermark_policy.emit_interval
sources.<binding>.watermark_policy.idle_timeout
sources.<binding>.capabilities.native_watermarks
runtime.nodes.<node>.ingress.<ingress>.progress
runtime.progress.admission.<binding>.post_end
runtime.progress.admission.<binding>.cancelled
runtime.progress.snapshot.boundary
runtime.progress.snapshot.prepared_job_fingerprint
runtime.progress.snapshot.coordinate.driver_logical_instant
runtime.progress.snapshot.coordinate.next_drain_epoch
runtime.progress.snapshot.coordinate.next_receipt_sequence
runtime.progress.snapshot.coordinate.next_trace_record_ordinal
runtime.progress.snapshot.coordinate.next_admission_attempt_ordinal
runtime.progress.snapshot.coordinate.next_gate_close_ordinal
runtime.progress.snapshot.coordinate.next_settlement_ordinal
runtime.progress.snapshot.coordinate.progress_execution_trace
runtime.progress.snapshot.coordinate.consumed_trace_position
runtime.progress.snapshot.coordinate.bindings.<binding>.upstream_delivery_replay_cursor
runtime.progress.snapshot.coordinate.bindings.<binding>.control_frontier
runtime.progress.snapshot.coordinate.bindings.<binding>.admission_gate_generation
runtime.progress.snapshot.coordinate.bindings.<binding>.admission_gate_state
runtime.progress.snapshot.coordinate.bindings.<binding>.admission_gate_close_ordinal
runtime.progress.snapshot.coordinate.bindings.<binding>.admission_gate_close_cut
runtime.progress.snapshot.coordinate.bindings.<binding>.next_inbox_sequence
runtime.progress.snapshot.coordinate.bindings.<binding>.next_fence_sequence
runtime.progress.snapshot.coordinate.bindings.<binding>.last_completed_fence
runtime.progress.snapshot.bindings.<binding>.normalized_config_fingerprint
runtime.progress.drains.<epoch>.first_error
runtime.progress.driver_phase.<phase>.<coordinate>
runtime.progress.counters.<counter>
runtime.progress.timers.deadline
```

The reason contains no secret or arbitrary connector payload. `SelectedItemError` is created only while iterating a fully keyed sorted snapshot; the first encountered failure owns the lowest failing key, suffix semantic hooks do not run, and every selected raw sender receives that stable error.

Admission/attempt allocation, drain/fence allocation, inbox freeze, local/timer key construction, replay structure/record validation, gate-close planning, trace/settlement allocation, and other pre-key failures return `DriverPhaseError`. It has an exact available coordinate and no `ReadyKey`, sentinel key, or item attribution. It commits no governed effect and triggers recorded fatal cleanup plus exactly-once settlement of every accepted envelope. If trace-record or drain-epoch exhaustion prevents a record/epoch, the last exact coordinate identifies the fatal boundary and replay beyond it is unsupported.

Terminal-tail and future-submit failures share the stable post-End category while retaining settlement/admission context. Cancellation is stable `Cancelled`. All forms map into the existing public error family only at the runtime boundary.

## 8. Per-binding and multi-input progress

### 8.1 Generated watermark and ingress activity

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IngressActivity {
    Active { watermark: Option<EventTime> },
    Idle { watermark: Option<EventTime> },
    Ended { final_watermark: Option<EventTime> },
}

#[derive(Clone, Debug)]
pub(crate) struct GeneratedWatermarkState {
    event_time: ResolvedEventTimeColumn,
    max_out_of_orderness: Duration,
    emit_interval: Duration,
    max_observed_event_time: Option<EventTime>,
    last_generated_watermark: Option<EventTime>,
}

impl GeneratedWatermarkState {
    fn observe_batch(&mut self, batch: &RecordBatch) -> crate::Result<()>;

    fn candidate(&self) -> crate::Result<Option<EventTime>>;

    fn accept_timer_candidate(&mut self) -> crate::Result<Option<EventTime>>;
}
```

`observe_batch` selects the maximum representable non-null timestamp. Empty or null-only input changes neither maximum nor watermark. Candidate arithmetic is checked. Only a current eligible generated timer calls `accept_timer_candidate`, and only strict advancement emits.

### 8.2 Binding state and evaluation result

```rust
#[derive(Clone, Debug)]
pub(crate) struct InputProgress {
    identity: BindingIdentity,
    ordinal: BindingOrdinal,
    mode: NormalizedWatermarkMode,
    activity: IngressActivity,
    last_connector_watermark: Option<EventTime>,
    generated: Option<GeneratedWatermarkState>,
    effective_watermark: Option<EventTime>,
    committed_upstream_position: RawUpstreamPosition,
    watermark_timer: Option<CurrentTimer>,
    idle_timer: Option<CurrentTimer>,
    next_local_sequence: CheckedSemanticAllocator,
    next_timer_sequence: CheckedSemanticAllocator,
    next_watermark_generation: CheckedSemanticAllocator,
    next_idle_generation: CheckedSemanticAllocator,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AggregateInput {
    Data,
    Watermark(EventTime),
    Idle,
    EndOfInput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DriverEventContext {
    current_instant: LogicalInstant,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TimerIntent {
    RearmIdleFromQualifyingActivity,
    ContinueGeneratedCadence {
        expired_phase_deadline: LogicalInstant,
    },
    Unregister(TimerKind),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct InputEvaluation {
    reactivated: bool,
    forward_data: bool,
    aggregate_input: Option<AggregateInput>,
    timer_intents: Vec<TimerIntent>,
    stale_timer_suppressed: bool,
    end_committed_if_transaction_succeeds: bool,
}

impl InputProgress {
    fn evaluate_raw(
        &mut self,
        event: &RawIngressEvent,
        context: DriverEventContext,
    ) -> crate::Result<InputEvaluation>;

    fn evaluate_internal_timer(
        &mut self,
        timer: &ReadyTimer,
        context: DriverEventContext,
    ) -> crate::Result<InputEvaluation>;
}
```

The driver alone builds `DriverEventContext`. An input API never accepts `now`. `InputEvaluation` contains semantic timer intents, not timer coordinates.

Data or a mode-legal connector watermark reactivates an idle ingress before event semantics, clears the aggregate idle latch, and rearms applicable idle timeout from the same driver instant. An illegal connector watermark errors before reactivation. Source watermark equality is suppressed; regression is a protocol error. Explicit connector `Idle` unregisters the idle timer but leaves generated cadence running. A current generated timer never reactivates an idle binding.

End preserves final effective watermark and stages timer removal. Any later selected connector/source event, including repeated End, is a protocol error. Only a driver-internal timer proven stale by generation is effect-free after End.

### 8.3 Multi-input aggregate

```rust
#[derive(Clone, Debug)]
pub(crate) struct MultiInputProgress {
    ingresses: std::collections::BTreeMap<BindingOrdinal, InputAggregateState>,
    last_emitted_watermark: Option<EventTime>,
    idle_latched: bool,
    idle_epoch: IdleEpoch,
    terminal: bool,
    next_global_sequence: CheckedSemanticAllocator,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InputAggregateState {
    activity: IngressActivity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProgressEmissionKind {
    Watermark(EventTime),
    Idle,
    EndOfInput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProgressEmission {
    sequence: GlobalSequence,
    kind: ProgressEmissionKind,
}

impl MultiInputProgress {
    fn evaluate(
        &mut self,
        ordinal: BindingOrdinal,
        input: AggregateInput,
    ) -> crate::Result<Vec<ProgressEmission>>;
}
```

The exact aggregate rules are:

- `Active(None)` participates and blocks a known minimum;
- `Active(Some(w))` participates with `w`;
- `Idle` and `Ended` are excluded from the active minimum;
- a watermark emits only when every active ingress is known and the minimum strictly advances;
- a non-empty all-live-idle set checked-increments the epoch and emits one plain `Idle` only when the latch is clear;
- repeated Idle in the same epoch emits nothing;
- Data or legal source watermark reactivates and clears the latch before aggregation;
- End permanently excludes an ingress; an end-induced watermark is buffered before terminal;
- final End buffers exactly one `EndOfInput`, no `Idle`, no MAX/sentinel watermark.

Global sequence and idle epoch exhaustion fail the full drain transaction. None of these paths compares event time to a watermark for lateness.

## 9. Admission inbox, raw envelopes, and exactly-once receipts

### 9.1 Raw submission and owned completion sender

```rust
#[derive(Debug)]
pub(crate) enum RawIngressEvent {
    Data(RecordBatch),
    ConnectorWatermark(EventTime),
    ConnectorIdle,
    EndOfInput,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum RawUpstreamPosition {
    Exact {
        delivery_replay_cursor: ReplayCursor,
        control_frontier: ControlFrontier,
    },
    Unavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CommittedRawInput {
    binding: BindingIdentity,
    accepted_identity: AcceptedEnvelopeIdentity,
    upstream_position: RawUpstreamPosition,
}

pub(crate) struct RawCommitReceipt {
    settled: OneShotReceiver<crate::Result<CommittedRawInput>>,
}

impl RawCommitReceipt {
    pub(crate) async fn wait_settled(self) -> crate::Result<CommittedRawInput>;
}

struct RawIngressEnvelope {
    identity: AcceptedEnvelopeIdentity,
    binding: BindingIdentity,
    event: RawIngressEvent,
    upstream_position: RawUpstreamPosition,
    settlement: OneShotSender<crate::Result<CommittedRawInput>>,
}

#[derive(Clone)]
pub(crate) struct RawIngressSender {
    admission: Arc<RawAdmissionInbox>,
}

impl RawIngressSender {
    pub(crate) async fn submit(
        &self,
        binding: BindingIdentity,
        event: RawIngressEvent,
        upstream_position: RawUpstreamPosition,
    ) -> crate::Result<RawCommitReceipt>;
}
```

`submit` is the only adapter-facing surface. It has no logical time, ordinal, sequence, fence, timer, class, or ready-key parameter. It creates a one-shot pair, transfers the sender into `RawIngressEnvelope`, and returns the receiver only after admission atomically accepts the envelope. A rejected submission returns an immediate `Err` and creates no accepted envelope/receipt.

The one-shot payload is itself `crate::Result<CommittedRawInput>`. Success is sent only after semantic state, timer heap/wake, upstream frontier/cursor acknowledgement, and completed fence coordinate commit. Failure is sent for transaction failure, terminal-tail rejection, cancellation, or fatal cleanup. A dropped receiver makes `send` fail, but the single send attempt still completes driver ownership and never rolls back semantic state.

### 9.2 Serialized gate and per-binding inbox state

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionGateState {
    Open,
    ClosedAfterEnd,
    ClosedCancelled,
    ClosedFatal,
}

struct BindingAdmissionState {
    gate: AdmissionGateSnapshot,
    next_inbox_sequence: CheckedSemanticAllocator,
    next_fence_sequence: CheckedSemanticAllocator,
    last_completed_fence: Option<InboxFenceCoordinate>,
    accepted: std::collections::VecDeque<RawIngressEnvelope>,
}

enum TraceExecutionMode {
    Record,
    Replay(ProgressReplayCursor),
}

struct ProgressTraceState {
    completed: ProgressExecutionTrace,
    mode: TraceExecutionMode,
    next_trace_record: CheckedSemanticAllocator,
    consumed_trace_position: CheckedSemanticAllocator,
    next_settlement_ordinal: CheckedSemanticAllocator,
}

struct AdmissionState {
    next_admission_attempt: CheckedSemanticAllocator,
    next_receipt_sequence: CheckedSemanticAllocator,
    next_gate_close_ordinal: CheckedSemanticAllocator,
    bindings: std::collections::BTreeMap<BindingOrdinal, BindingAdmissionState>,
    trace: ProgressTraceState,
    pending_fatal: Option<DriverPhaseError>,
}

struct RawAdmissionInbox {
    config: StreamProgressRuntimeConfig,
    ordinal_by_identity: Arc<
        std::collections::BTreeMap<BindingIdentity, BindingOrdinal>,
    >,
    state: Mutex<AdmissionState>,
    capacity_available: std::collections::BTreeMap<
        BindingOrdinal,
        AsyncCapacitySignal,
    >,
    driver_wake: CentralDriverWake,
}

struct FrozenAdmissionEpoch {
    epoch: DrainEpoch,
    fences: Vec<InboxFenceCoordinate>,
    selected_raw: Vec<RawIngressEnvelope>,
}

struct AdmissionCommitGuard<'a> {
    state: MutexGuard<'a, AdmissionState>,
}

struct PreparedTerminalTransition {
    record: TerminalTransitionRecord,
    extracted: Vec<RawIngressEnvelope>,
    settlements: Vec<PendingReceiptSettlement>,
}

impl RawAdmissionInbox {
    async fn accept_raw(
        &self,
        binding: BindingIdentity,
        event: RawIngressEvent,
        upstream_position: RawUpstreamPosition,
    ) -> crate::Result<RawCommitReceipt>;

    fn freeze_record_epoch(
        &self,
        epoch: DrainEpoch,
        ordinals: &[BindingOrdinal],
    ) -> Result<FrozenAdmissionEpoch, DriverPhaseError>;

    fn freeze_replay_epoch(
        &self,
        expected: &DrainEpochRecord,
        ordinals: &[BindingOrdinal],
    ) -> Result<FrozenAdmissionEpoch, DriverPhaseError>;

    async fn prepare_terminal_transition(
        &self,
        cause: TerminalTransitionCause,
        driver_clock: DriverClockCoordinate,
        owning_drain: Option<DrainEpoch>,
        selected_fences: &[InboxFenceCoordinate],
        failure: Option<DriverFailureRecord>,
    ) -> Result<PreparedTerminalTransition, DriverPhaseError>;

    fn begin_commit(&self) -> AdmissionCommitGuard<'_>;

    fn trace_snapshot(&self) -> ProgressTraceSnapshot;
}

impl AdmissionCommitGuard<'_> {
    /// Infallible after preparation. Commits the already validated close cuts,
    /// exact extracted membership, terminal record, and settlement records.
    fn commit_terminal_transition(
        &mut self,
        prepared: PreparedTerminalTransition,
    ) -> Vec<CommittedReceiptSettlement>;

    fn commit_completed_fence(&mut self, fence: InboxFenceCoordinate);
}
```

The concrete implementation may use the existing M2 mutex/channel/semaphore primitives, but it MUST preserve this serialized state machine. Capacity acquisition never bypasses the locked gate check.

Admission performs one atomic operation under the state lock:

1. resolve the immutable binding ordinal;
2. probe and allocate the stable admission-attempt ordinal before reading the gate;
3. snapshot gate generation/state/close coordinate;
4. decide `Accepted` or `ImmediateRejected`;
5. for acceptance, probe receipt/inbox successors and capacity without consuming a partial subset;
6. record mode appends, or replay mode consumes, the exact `AdmissionAttemptRecord` before the result is visible;
7. only then acceptance consumes receipt/inbox coordinates, transfers the sender, enqueues, and wakes the driver; rejection returns its stable error without an accepted identity.

If the gate is `ClosedAfterEnd`, submission records the observed closed generation/cause and immediately returns the stable post-End error without receipt/inbox allocation or enqueue. Cancelled/fatal gates similarly reject. If attempt/trace/receipt/inbox allocation fails, the call accepts nothing and reports `DriverPhaseError` with the exact admission/counter/trace coordinate. The inbox schedules recorded fatal cleanup, which transitions every gate and settles all earlier accepted envelopes.

`freeze_record_epoch` receives a driver-checked epoch and atomically probes every per-binding fence successor before changing any. It records all bindings, uses `Empty` or the exact inclusive upper inbox sequence, removes precisely the eligible envelopes, and consumes fence coordinates together. `freeze_replay_epoch` consumes the next expected drain record before extraction and rejects a missing/extra/reordered fence, identity, timer, or key coordinate as pre-key `DriverPhaseError`.

`prepare_terminal_transition` is the single End/cancel/fatal gate protocol. In replay it inspects the expected terminal record, waits without exposing closure until every identity recorded as accepted is present, reacquires the gate lock, rejects unexpected identities or a different generation/cut, validates the canonical settlement records, and returns a prepared value. In record mode it checked-allocates trace/gate-generation/gate-close/settlement coordinates and stages the actual exact membership. `commit_terminal_transition` is then infallible: it changes gates, extracts precisely that membership, commits terminal/settlement trace records, and returns sender-owning committed settlements.

### 9.3 Receipt settlement plans

```rust
struct PendingReceiptSettlement {
    identity: AcceptedEnvelopeIdentity,
    sender: OneShotSender<crate::Result<CommittedRawInput>>,
    record: SettlementRecord,
    result: Result<CommittedRawInput, ProgressFailure>,
}

struct CommittedReceiptSettlement {
    identity: AcceptedEnvelopeIdentity,
    sender: OneShotSender<crate::Result<CommittedRawInput>>,
    result: Result<CommittedRawInput, ProgressFailure>,
    committed_trace_position: TracePosition,
}

impl CommittedReceiptSettlement {
    /// Legal only after the exact SettlementRecord was appended/consumed.
    fn send(self) {
        let result = self.result.map_err(ProgressFailure::into_existing_error);
        let _receiver_may_have_been_dropped = self.sender.send(result);
    }
}
```

There is no sender-less acknowledgement vector and no pending-ack side map. The sender moves exactly once from accepted envelope to selected transaction, terminal-tail plan, cancel/fatal cleanup, `PendingReceiptSettlement`, and then `CommittedReceiptSettlement::send`. The latter can be constructed only while the matching exact settlement record is appended or consumed under trace/gate ownership. Rust ownership prevents duplicate settlement. A commit/error/cancel path that cannot account for a sender and trace record is invalid.

## 10. Driver-owned timers and cadence

```rust
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum TimerKind {
    Watermark,
    Idle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CurrentTimer {
    deadline: LogicalInstant,
    generation: TimerGeneration,
    timer_sequence: TimerSequence,
    ready_local_sequence: LocalSequence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LogicalTimer {
    binding_ordinal: BindingOrdinal,
    kind: TimerKind,
    deadline: LogicalInstant,
    generation: TimerGeneration,
    timer_sequence: TimerSequence,
    ready_local_sequence: LocalSequence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct TimerHeapKey {
    deadline: LogicalInstant,
    binding_ordinal: BindingOrdinal,
    timer_sequence: TimerSequence,
}

#[derive(Clone, Debug)]
struct ReadyTimer {
    identity: TimerIdentity,
    ready_local_sequence: LocalSequence,
}

struct TimerAllocator;

impl TimerAllocator {
    fn checked_add_delay(
        instant: LogicalInstant,
        delay: Duration,
    ) -> crate::Result<LogicalInstant>;

    fn checked_next_phase_deadline(
        expired_deadline: LogicalInstant,
        current_instant: LogicalInstant,
        emit_interval: Duration,
    ) -> crate::Result<LogicalInstant>;

    fn arm_initial_generated(
        scratch: &mut DriverScratch,
        binding: BindingOrdinal,
        running_instant: LogicalInstant,
    ) -> crate::Result<()>;

    fn resolve_intents(
        scratch: &mut DriverScratch,
        binding: BindingOrdinal,
        current_instant: LogicalInstant,
        intents: &[TimerIntent],
    ) -> crate::Result<()>;
}
```

Only `TimerAllocator` inside the single driver allocates/replaces/unregisters timers. The first generated deadline is running instant plus `emit_interval`. Recurrence is phase-anchored to the expired scheduled deadline `D`; at current instant `t`, one semantic expiry is evaluated and the next deadline is the smallest checked `D + n * interval > t`. Missed ticks never create a burst. No-observation and non-advancing candidates still schedule the next phase tick.

Idle deadline is qualifying activity instant plus timeout. Data, a mode-legal connector watermark, or reactivation qualifies. Explicit Idle/current idle expiry unregisters the idle timer. End unregisters both current timers. Cancel/terminal/fatal cleanup clears current and stale heap entries plus the one central wake. Restore copies exact timers and registers exactly one wake at the heap minimum; it never recomputes/rebases a deadline.

## 11. Transaction structures and first-error contract

### 11.1 Ready payload and driver emissions

```rust
#[derive(Debug)]
enum ReadyPayload {
    Raw(RawIngressEnvelope),
    Timer(ReadyTimer),
}

#[derive(Debug)]
struct ReadyStep {
    key: ReadyKey,
    identity: ReadyItemIdentity,
    payload: ReadyPayload,
}

#[derive(Debug)]
pub(crate) enum DriverEmission {
    ForwardData {
        binding: BindingOrdinal,
        batch: RecordBatch,
    },
    Progress(ProgressEmission),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BindingEndCommit {
    binding: BindingOrdinal,
    selected_fence: InboxFenceCoordinate,
}
```

`ForwardData` is data-neutral. It carries no late classification or metric action.

### 11.2 Scratch, commit, and failure ownership

```rust
#[derive(Clone)]
struct DriverProgressState {
    current_clock: DriverClockCoordinate,
    inputs: Vec<InputProgress>,
    aggregate: MultiInputProgress,
    timer_heap: std::collections::BTreeMap<TimerHeapKey, LogicalTimer>,
    status_counters: ProgressCounters,
}

struct FrozenReadySnapshot {
    epoch: DrainEpoch,
    clock: DriverClockCoordinate,
    fences: Vec<InboxFenceCoordinate>,
    steps: Vec<ReadyStep>,
    drain_record: DrainEpochRecord,
}

struct DriverScratch {
    state: DriverProgressState,
    buffered_emissions: Vec<DriverEmission>,
    buffered_timer_mutations: Vec<TimerMutation>,
    selected_receipts: Vec<PendingReceiptSettlement>,
    end_commits: Vec<BindingEndCommit>,
}

struct PreparedDrainCommit {
    next_state: DriverProgressState,
    emissions: Vec<DriverEmission>,
    timer_mutations: Vec<TimerMutation>,
    selected_receipts: Vec<PendingReceiptSettlement>,
    end_commits: Vec<BindingEndCommit>,
    drain_record: DrainEpochRecord,
    next_central_wake: Option<LogicalInstant>,
}

struct PreparedDrainFailure {
    error: SelectedItemError,
    selected_receipts: Vec<PendingReceiptSettlement>,
    drain_record: DrainEpochRecord,
}

enum PreparedDrain {
    Commit(PreparedDrainCommit),
    Fail(PreparedDrainFailure),
}

pub(crate) struct CommittedDrain {
    emissions: Vec<DriverEmission>,
    committed_inputs: Vec<CommittedRawInput>,
    terminal_tail_failures: usize,
}

enum DrainOutcome {
    NoArbitration,
    Committed(CommittedDrain),
    FatalSelected(SelectedItemError),
    FatalPhase(DriverPhaseError),
}
```

Evaluation consumes the frozen snapshot and therefore cannot lose a selected sender. For each selected raw envelope, scratch initially owns its sender. On success it stages a `CommitSuccess` settlement record/result; on the first failure it rewrites every selected settlement—including the unevaluated suffix—to the same `TransactionError(SelectedItemErrorRecord)`. `PreparedDrainFailure` returns all senders to the driver.

The driver evaluates sorted steps only until the first error. The first encountered error is deterministic because `ReadyKey` is total. No suffix semantic validation/evaluation hook runs. Nevertheless, every selected raw envelope is failure-settled with the same first error. A valid prefix never commits independently. Allocation/freeze/key/trace/gate failures do not enter `PreparedDrainFailure`; they return `DriverPhaseError` before selected-item evaluation.

## 12. Job-scoped driver, commit linearization, and cleanup

### 12.1 Driver API

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DriverPhase {
    Prepared,
    RunningQuiescent,
    ApplyingDrain,
    Cancelling,
    Cancelled,
    FatalCleanup,
    TerminalCleanup,
    Terminal,
}

pub(crate) struct StreamProgressDriver<C: DriverLogicalClock> {
    prepared_job: Arc<PreparedStreamJob>,
    clock: C,
    ordinal_by_identity: std::collections::BTreeMap<BindingIdentity, BindingOrdinal>,
    ordered_ordinals: Arc<[BindingOrdinal]>,
    next_binding_ordinal: CheckedSemanticAllocator,
    next_drain_epoch: CheckedSemanticAllocator,
    admission: Arc<RawAdmissionInbox>,
    central_wake: CentralLogicalWake,
    state: DriverProgressState,
    phase: DriverPhase,
}

impl<C: DriverLogicalClock> StreamProgressDriver<C> {
    pub(crate) fn new(
        prepared_job: Arc<PreparedStreamJob>,
        clock: C,
    ) -> crate::Result<(Self, RawIngressSender)>;

    pub(crate) fn start_running(&mut self) -> crate::Result<()>;

    pub(crate) async fn run(
        self,
        output: DriverOutputSink,
        cancellation: CancellationToken,
    ) -> crate::Result<()>;

    pub(crate) fn status(&self) -> StreamProgressStatus;

    fn freeze_ready_set(
        &mut self,
        trigger: DrainTrigger,
    ) -> Result<Option<FrozenReadySnapshot>, DriverPhaseError>;

    fn evaluate_transaction(
        &self,
        frozen: FrozenReadySnapshot,
    ) -> PreparedDrain;

    async fn commit_transaction(
        &mut self,
        prepared: PreparedDrainCommit,
    ) -> Result<CommittedDrain, DriverPhaseError>;

    async fn fail_selected_transaction(
        &mut self,
        prepared: PreparedDrainFailure,
    ) -> SelectedItemError;

    async fn cancel_and_settle_all(
        &mut self,
        failure: ProgressFailure,
    ) -> Result<(), DriverPhaseError>;

    async fn fatal_cleanup_and_settle_all(
        &mut self,
        failure: DriverFailureRecord,
    ) -> Result<(), DriverPhaseError>;

    async fn drain_ready(&mut self, trigger: DrainTrigger) -> DrainOutcome;
}
```

`new` uses only the runtime config already fingerprinted in `PreparedStreamJob`. It checked-allocates ordinals from immutable table order before opening admission. `start_running` reads its own clock, stages all initial generated timers, commits only if all deadlines/allocations succeed, installs the single heap-minimum wake, and opens admission. There is no per-binding semantic sleeper.

### 12.2 Freeze and replay

On a semantic trigger, `freeze_ready_set`:

1. requires `RunningQuiescent` and reads one driver clock coordinate;
2. atomically probes trace/drain-epoch and all inbox-fence successors;
3. in record mode stages every binding's exact visible upper sequence; in replay mode validates the next expected drain coordinate before extraction;
4. assigns raw ordinal/local sequence and `InputOrControl` keys using checked construction;
5. gathers due internal timers and assigns checked timer keys at the same logical instant;
6. sorts the fixed snapshot by `ReadyKey`;
7. stages selected raw/timer identities, inclusive key range or Empty, due timers, and every inbox fence;
8. validates the complete expected drain record shape before selected-item evaluation.

Work accepted above a fence waits for a later drain except the successful-End terminal tail. A required empty epoch is returned and traced. A no-op poll that never enters arbitration returns `NoArbitration` and consumes no epoch.

If trace/drain/fence/local/timer allocation, inbox freeze, key construction, or replay validation fails, `freeze_ready_set` returns `DriverPhaseError` with phase and exact available coordinate. It never fabricates `ReadyKey`. The driver then records/validates fatal transition and settlement membership when trace capacity permits, closes admission, takes all selected/unselected accepted envelopes, settles each exactly once, clears timers/wake, and exits without the failing phase's state, emission, timer, cursor, admission, receipt, or fence effect.

### 12.3 Success commit and final-End tail

`commit_transaction` may fail only with a pre-commit `DriverPhaseError` while validating trace/gate/settlement coordinates. It performs this serialized sequence:

1. stage the committed drain record and selected success settlement records;
2. if End is present, asynchronously prepare the exact terminal record, gate generations/close cuts, accepted tail membership, and post-End settlement records; replay waits for recorded accepted tails and rejects unexpected ones without exposing closure;
3. acquire `AdmissionCommitGuard`, excluding concurrent submit linearization;
4. validate/consume or append the drain, terminal, and canonical settlement records in exact linearization order;
5. atomically replace live semantic state, timer heap/wake, counters, cursor/frontier, completed fences, End gate state, and exact extracted tail membership;
6. release the guard with sender-owning `CommittedReceiptSettlement` values whose records are already committed;
7. send selected successes and tail failures exactly once; cross-receipt wake order is unspecified;
8. only after every send attempt, return the complete buffered emission batch.

The gate closure in step 5 is the End linearization point. Every envelope racing End is exactly one of:

- accepted at or below the frozen fence and selected;
- accepted above the fence before closure, extracted as terminal tail and failed without another drain;
- submitted after closure, immediately rejected without an accepted identity.

It cannot be lost between categories. End-induced watermark remains before `EndOfInput`. Terminal tail settlement produces no data/progress emission and no upstream cursor acknowledgement.

If the commit makes all ingresses ended, the driver enters `TerminalCleanup` and reaches `Terminal` only after the terminal and settlement trace suffix is consumed, every selected success and terminal-tail failure send has been attempted, every gate is closed, all accepted queues are empty, current/stale timers and the central wake are empty, and no detached progress work remains.

### 12.4 Failure, cancellation, receiver drop, and exhaustion

`fail_selected_transaction` discards scratch effects, validates/appends the selected-item drain outcome plus canonical transaction-error settlements, and settlement-attempts every selected sender once with the stable lowest-key error. It commits no semantic state, emission, timer, cursor acknowledgement, or completed fence. It then invokes fatal cleanup with `DriverFailureRecord::SelectedItem`; cleanup records exact gate generations/cuts, selected/unselected accepted membership, fatal settlements, and clears timers/wake before exit.

Cancellation uses `prepare_terminal_transition(Cancellation, ...)` even when no drain exists. It records or consumes exact gate closes/cuts and accepted membership, commits canonical cancellation settlement records, sends each once, clears timers/wake, verifies the expected trace suffix, and exits with no later emission. Fatal cleanup uses the same protocol with `Fatal` and the selected-item or driver-phase cause.

A dropped receiver affects only the result of `OneShotSender::send`. The sender is still consumed, commit is not rolled back, the trace does not change, no retry occurs, and no work remains owned.

Attempt/trace/receipt/inbox exhaustion during admission accepts nothing from the failing call and consumes no partial allocation. Drain/fence/local-key/trace validation exhaustion is `DriverPhaseError`, not a selected-item error; selected evaluation counter/deadline exhaustion is `SelectedItemError` only when it occurs after the relevant key is complete. Gate-generation/close/settlement exhaustion also returns `DriverPhaseError`. Every case enters fatal cleanup and exactly-once settles all previously accepted work; no path can wrap, fabricate a key, partially commit, or strand a sender.

## 13. Complete transient snapshot, restore, and replay API

### 13.1 Opaque upstream coordinates

```rust
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct ReplayCursor {
    source_kind: Arc<str>,
    opaque_equality_token: Arc<[u8]>,
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct ControlFrontier {
    source_kind: Arc<str>,
    opaque_equality_token: Arc<[u8]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PositionedBindingFrontier {
    binding_identity: BindingIdentity,
    upstream_delivery_replay_cursor: ReplayCursor,
    control_frontier: ControlFrontier,
}

pub(crate) struct PausedExactUpstreams {
    binding_frontiers: Vec<PositionedBindingFrontier>,
    pause_guard: ExactPauseGuard,
}
```

Only an existing crate-private replay-position coordinator can mint `PausedExactUpstreams`. It proves every represented source remains paused and exactly positioned. A binding lacking `ExactPauseReportAndSeek` deterministically rejects restore-capable capture. Row counts, wall time, latest offsets, or best-effort seek are never accepted.

### 13.2 Snapshot information content

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LogicalTimerSnapshot {
    deadline: LogicalInstant,
    binding_ordinal: BindingOrdinal,
    kind: TimerKind,
    generation: TimerGeneration,
    timer_sequence: TimerSequence,
    ready_local_sequence: LocalSequence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum UnsettledAcceptedEnvelopeCount {
    Zero,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BindingReplayFrontier {
    binding_identity: BindingIdentity,
    driver_assigned_ordinal: BindingOrdinal,
    upstream_delivery_replay_cursor: ReplayCursor,
    control_frontier: ControlFrontier,
    admission_gate: AdmissionGateSnapshot,
    next_local_sequence: u64,
    next_timer_sequence: u64,
    next_watermark_generation: u64,
    next_idle_generation: u64,
    next_inbox_sequence: u64,
    next_fence_sequence: u64,
    last_completed_fence: Option<InboxFenceCoordinate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CapturedLogicalCoordinate {
    driver_clock: DriverClockCoordinate,
    binding_frontiers: Vec<BindingReplayFrontier>,
    runtime_fence_config_fingerprint: RuntimeFenceConfigFingerprint,
    next_binding_ordinal: u64,
    next_global_sequence: u64,
    next_receipt_sequence: u64,
    next_drain_epoch: u64,
    next_trace_record_ordinal: u64,
    consumed_trace_position: TracePosition,
    next_admission_attempt_ordinal: u64,
    next_gate_close_ordinal: u64,
    next_settlement_ordinal: u64,
    completed_progress_execution_trace: ProgressExecutionTrace,
    scheduled_timers: Vec<LogicalTimerSnapshot>,
    unsettled_accepted_envelopes: UnsettledAcceptedEnvelopeCount,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BindingProgressSnapshot {
    identity: BindingIdentity,
    ordinal: BindingOrdinal,
    normalized_config_fingerprint: NormalizedConfigFingerprint,
    upstream_delivery_replay_cursor: ReplayCursor,
    control_frontier: ControlFrontier,
    admission_gate: AdmissionGateSnapshot,
    activity: IngressActivity,
    last_connector_watermark: Option<EventTime>,
    generated_max_observed_event_time: Option<EventTime>,
    last_generated_watermark: Option<EventTime>,
    effective_watermark: Option<EventTime>,
    current_watermark_timer: Option<LogicalTimerSnapshot>,
    current_idle_timer: Option<LogicalTimerSnapshot>,
    next_local_sequence: u64,
    next_timer_sequence: u64,
    next_watermark_generation: u64,
    next_idle_generation: u64,
    next_inbox_sequence: u64,
    next_fence_sequence: u64,
    last_completed_fence: Option<InboxFenceCoordinate>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct InputProgressSnapshot {
    ordinal: BindingOrdinal,
    activity: IngressActivity,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MultiInputProgressSnapshot {
    ingresses: Vec<InputProgressSnapshot>,
    last_emitted_aggregate_watermark: Option<EventTime>,
    idle_latched: bool,
    idle_epoch: IdleEpoch,
    terminal: bool,
    next_global_sequence: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StreamProgressSnapshot {
    prepared_job_fingerprint: PreparedJobFingerprint,
    coordinate: CapturedLogicalCoordinate,
    bindings: Vec<BindingProgressSnapshot>,
    aggregate: MultiInputProgressSnapshot,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProgressTraceSnapshot {
    completed: ProgressExecutionTrace,
    next_trace_record_ordinal: u64,
    consumed_trace_position: TracePosition,
    next_admission_attempt_ordinal: u64,
    next_gate_close_ordinal: u64,
    next_settlement_ordinal: u64,
}
```

The information content is complete. `completed_progress_execution_trace` is the lossless prefix containing admissions, drains/empty epochs, terminal transitions/gate cuts, settlements, and recordable phase failures. `consumed_trace_position`, next trace/admission/gate-close/settlement/drain/receipt coordinates, each gate generation/state/close coordinate, each inbox/fence/local/global/timer coordinate, every last-completed fence, and exact current/stale timer heap are retained. `Option` distinguishes absent/current timers. Duplicated binding/aggregate/coordinate fields compare exactly; restore never chooses one conflicting copy. The drain projection may be recomputed but is not separately sufficient.

Restore-capable capture requires `UnsettledAcceptedEnvelopeCount::Zero`. That means every accepted envelope sender has already been settlement-attempted, the admission queues contain none, no scratch transition exists, the preceding emission batch has returned, and the driver is before the next fence. Capture rejects otherwise.

### 13.3 Restore request and validated token

```rust
pub(crate) struct RestoreRequest<C: DriverLogicalClock> {
    prepared_job: Arc<PreparedStreamJob>,
    clock: C,
    snapshot: StreamProgressSnapshot,
    positioned_upstreams: PausedExactUpstreams,
    replay: ProgressReplayRequest,
}

pub(crate) struct ValidatedProgressRestore<C: DriverLogicalClock> {
    prepared_job: Arc<PreparedStreamJob>,
    clock: C,
    snapshot: StreamProgressSnapshot,
    positioned_upstreams: PausedExactUpstreams,
    replay: PrevalidatedProgressReplay,
}

impl<C: DriverLogicalClock> StreamProgressDriver<C> {
    pub(crate) fn capture_snapshot(
        &self,
        paused_upstreams: &PausedExactUpstreams,
    ) -> crate::Result<StreamProgressSnapshot>;

    pub(crate) fn validate_restore(
        request: RestoreRequest<C>,
    ) -> crate::Result<ValidatedProgressRestore<C>>;

    pub(crate) fn restore(
        validated: ValidatedProgressRestore<C>,
    ) -> crate::Result<(Self, RawIngressSender)>;
}
```

There is no free restore-time `inbox_capacity` or fence policy. `restore` uses the exactly fingerprinted prepared config. `validate_restore` consumes the pause guard and replay request and checks, before driver construction/timer installation/source resume/task spawn/emission:

1. exact prepared-job and runtime-fence fingerprints;
2. exact binding count, identity, ordinal, order, and normalized config;
3. exact paused upstream delivery/replay cursor and control frontier for every binding;
4. exact driver clock trace and instant;
5. exact admission-gate state, generation, close cause/ordinal, and next-inbox cut;
6. exact next ordinal/global/receipt/drain/trace-record/admission-attempt/gate-close/settlement/local/timer/generation/inbox/fence coordinate;
7. exact last-completed per-inbox fences;
8. exact completed `ProgressExecutionTrace` prefix/consumed position and equality with the prevalidated replay prefix;
9. exact current and stale scheduled timers, heap order, deadlines, generation, timer/local sequence;
10. exact aggregate state, ingress state, latch, idle epoch, terminal state, and watermark invariants;
11. proof of zero accepted/unsettled envelopes and non-exhausted next coordinates;
12. complete replay-plan structure: one ordered admission per raw attempt, one settlement per acceptance and none per rejection, valid owners/cuts/tails, contiguous coordinates, and an exact complete suffix.

Any missing or unequal field fails at its stable path with all upstreams still paused and no mutation. Same-trace earlier/later clock, different clock trace, missing/extra/reordered/short/long admission, drain, terminal, settlement, phase-failure, or empty-drain record; a different gate generation/cut/tail; wrong settlement owner; different fence/key/timer; or unconsumed suffix is an exact `DriverPhaseError` mismatch.

`restore` copies state, heap, all allocators, gate snapshots, complete trace prefix/position, last fences, and prevalidated replay cursor exactly. It registers one central wake at the exact heap minimum and never calls deadline arithmetic. Upstreams remain paused until installation succeeds. A terminal snapshot restores with closed gates/cuts, empty queues/heap/wake, zero unsettled senders, and no future emission.

Snapshot and replay values are in-memory only: no serde, bytes, file, manifest, version tag, cross-process transfer, process restart, deadline rebase, or durable/exactly-once claim. The required phrase is “snapshot-ready and deterministically replayable progress state.”

## 14. Status and metrics

```rust
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ProgressCounters {
    stale_timers_suppressed: u64,
    watermarks_advanced: u64,
    equal_watermarks_suppressed: u64,
    idle_transitions: u64,
    reactivations: u64,
    drains_committed: u64,
    drains_failed: u64,
    receipt_settlement_attempts: u64,
    dropped_receivers: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProgressModeName {
    SourceProvided,
    Generated,
    Disabled,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BindingProgressStatus {
    identity: BindingIdentity,
    ordinal: BindingOrdinal,
    mode: ProgressModeName,
    activity: IngressActivity,
    admission_gate: AdmissionGateSnapshot,
    last_connector_watermark: Option<EventTime>,
    generated_max_observed_event_time: Option<EventTime>,
    last_generated_watermark: Option<EventTime>,
    effective_watermark: Option<EventTime>,
    watermark_deadline: Option<LogicalInstant>,
    idle_deadline: Option<LogicalInstant>,
    last_completed_fence: Option<InboxFenceCoordinate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StreamProgressStatus {
    bindings: Vec<BindingProgressStatus>,
    last_aggregate_watermark: Option<EventTime>,
    live_ingresses: usize,
    active_ingresses: usize,
    idle_ingresses: usize,
    ended_ingresses: usize,
    idle_epoch: IdleEpoch,
    idle_latched: bool,
    next_drain_epoch: DrainEpoch,
    next_admission_attempt: AdmissionAttemptOrdinal,
    next_gate_close: GateCloseOrdinal,
    next_settlement: SettlementOrdinal,
    next_trace_record: TraceRecordOrdinal,
    consumed_trace_position: TracePosition,
    completed_trace_records: usize,
    unsettled_receipts: usize,
    timer_heap_size: usize,
    terminal: bool,
    counters: ProgressCounters,
}
```

Status is an immutable copy of driver-owned state and allocates nothing. Metric labels use only bounded enums. Raw binding names, timestamps, columns, cursors, and connector payloads are not labels. Observational counters saturate and never feed semantics.

M3 exposes no late-row/late-assignment field, counter, tag, log category, or error. In particular, `late_rows_observed`, `late_rows_dropped`, and aliases are forbidden.

## 15. Construction, lifecycle, and exact flows

### 15.1 Fresh construction

```text
compile + descriptor collection + whole-job preflight/config fingerprint
-> construct driver with checked ordinals, checked admission allocators, closed gates
-> connector open
-> apply already-existing private native-WM directive if needed
-> start_running at driver clock: atomically install initial generated timers
-> open gates, install one central wake, start one driver task
-> first pull
```

A preflight failure happens before connector open. The existing private toggle route must finish before task spawn/timer/pull. No public `open()` change exists.

### 15.2 Generated watermark flow

1. preflight resolves known Arrow timestamp column and disables native emission if required;
2. `start_running` arms `t + emit_interval` with checked arithmetic;
3. raw Data consumes its exact admission-attempt record, then is accepted with driver/inbox-owned receipt and inbox sequence;
4. a drain fence selects it; the driver stamps one `ReadyKey` at the drain instant;
5. scratch reactivates if needed, observes non-null max, forwards data unchanged, and rearms idle from that instant;
6. one due generated timer evaluates one candidate after same-instant input/control;
7. strict advancement emits; regardless, one next phase-aligned timer is staged;
8. successful commit records/consumes the exact `CommitSuccess` settlement before sending the receipt, after state/timer/frontier/fence commit.

### 15.3 Source watermark, idle, and reactivation

Source-provided mode has no generated timer. A legal connector watermark reactivates before validation/aggregation, suppresses equality, errors on regression, and can advance the active minimum. Data behaves similarly for reactivation. All-live-idle emits one `Idle` per checked epoch; repeats are latched. Reactivation clears the latch. Effective and aggregate watermarks never regress.

### 15.4 First-error transaction flow

1. the driver successfully allocates the drain/fences, freezes exact membership, constructs every key, and validates the expected drain record;
2. scratch owns every selected raw sender and stages all effects;
3. steps are evaluated in total order only until the first deterministic error;
4. no suffix semantic hook runs;
5. scratch effects are discarded;
6. the selected-failure drain outcome and canonical transaction-error settlement records are appended/consumed;
7. every selected raw sender, including unevaluated suffix, receives one send attempt with that first error;
8. fatal transition/cuts and fatal settlement records are appended/consumed for every unselected accepted sender;
9. every unselected accepted sender is sent its recorded fatal result;
10. no state, emission, timer, cursor, or completed-fence coordinate commits.

This first-key flow applies only to failures encountered after complete key assignment, such as selected post-End input, illegal watermark, or selected-item arithmetic/counter failure. Admission, drain/fence allocation, inbox freeze, key construction, replay validation, terminal planning, trace/settlement allocation, and their overflows are `DriverPhaseError` flows with phase coordinate and no `ReadyKey`; they expose none of the failing phase's effects and then perform complete recorded fatal cleanup.

### 15.5 Final End and admission race

Suppose the final End is selected through inbox sequence `s` and Data is accepted at `s + 1` before commit:

1. scratch End buffers any end-induced Watermark then `EndOfInput`;
2. Data admission records `Accepted` with its attempt/receipt/inbox identity before End closure;
3. replay waits until that exact recorded identity exists; record mode observes the actual identity;
4. commit acquires the admission guard and validates/stages `TerminalTransitionRecord` with old/new generation, close ordinal, cut `s + 2`, and exact extracted identity;
5. the gate becomes `ClosedAfterEnd`, the exact tail is extracted, and drain/terminal/settlement records plus semantic/timer/frontier/fence state commit atomically;
6. selected End receipt is sent success after its record commits;
7. the `s + 1` Data receipt is sent one recorded post-End failure, with no emission/cursor acknowledgement and no later drain;
8. a future submit consumes an `ImmediateRejected` admission record observing the closed generation/cause, with no accepted identity or settlement record;
9. all-ended exits only after the expected trace suffix is consumed, every settlement attempt completes, and gates/queues/heap/wake/detached work are empty.

If End and Data were both below one earlier fence, Data encounters scratch `Ended`, becomes the first selected error if first failing by key, and the whole drain fails. If End closes before the Data attempt, Data is immediately rejected. Accepted tail, same-drain transaction error, and immediate rejection are distinct `ProgressExecutionTrace` executions even when their drain projection is equal.

### 15.6 Exact transient replay

1. reach a receipt-quiescent boundary and pause/position every replay-capable upstream exactly;
2. capture job/config/clock/frontier/gate/allocator/timer/aggregate state plus full execution-trace prefix/position;
3. construct `ProgressReplayRequest` with the ordered raw attempts, logical-clock trace, exact prefix, complete expected suffix, and declared completion position;
4. prevalidate trace length/structure/coordinates, one admission per attempt, one settlement per acceptance and none per rejection, owners, cuts/tails, and complete suffix before any runtime side effect;
5. restore exact state and begin at the captured trace/admission/gate/settlement/drain/inbox/fence coordinates;
6. consume each admission record before its result, each drain record before epoch effects, each terminal record before gate closure, and each canonical settlement record before receipt send;
7. validate required empty epochs, cancel/fatal races without a drain, immediate rejections, phase failures, and end-of-trace completion;
8. obtain item-identical admission decisions, emissions, receipt dispositions/results, phase-correct errors, gate transitions, and terminal result.

Using another full execution trace intentionally has no identical admission/settlement/grouping promise. Equal `DrainFenceTrace` projections alone are insufficient.

### 15.7 Cancellation, fatal overflow, and receiver drop

Cancellation and fatal exhaustion stage/consume exact terminal records with cause/clock/gate generations/close ordinals/cuts/tails, then stage/consume canonical settlement records, close gates, atomically take accepted queues, send every result once, clear timer heap/wake, and only then exit. Admission/trace counter overflow accepts nothing from the failing call and still drives fatal cleanup for earlier accepted work when record capacity permits. A receiver dropped at any point changes only the one-shot send result; recorded disposition and semantic cleanup remain unchanged.

## 16. Compatibility, errors, and adversarial defenses

### 16.1 Compatibility table

| Surface / behavior | Baseline to M3 change | Compatibility result |
| ------------------ | --------------------- | -------------------- |
| Public Rust exports | none | source/binary API baseline unchanged |
| Connector trait / `open()` | none | connector public obligations unchanged |
| `compile_stream()` | no binding/catalog/config parameter | remains pure and binding-agnostic |
| Python | none | no migration |
| REST / OpenAPI | none | no route/schema migration |
| Serialized user config | none | M3 policy/config stays internal |
| Runtime construction | internal preflight and one progress driver | internal call-site migration only |
| Backpressure/receipts | existing M2 bounded path gains exact internal settlement ownership | M2 lifecycle suite must remain green |
| Snapshot | crate-private in-memory exact coordinate and full execution trace | not durable/cross-version |
| Late data | no M3 behavior or metric | M4 owns assignment-level drop |
| Window/state | none | no M4 dependency |
| Public A6 | none | remains post-M5 |

### 16.2 Required error outcomes

| Case | Required outcome |
| ---- | ---------------- |
| unknown/dynamic schema in Generated | binding-qualified preflight failure before open |
| missing/unsupported event-time column | stable column path before open |
| invalid duration or policy/capability cell | preflight failure; no fallback |
| connector WM in Generated/Disabled | first-key protocol error; no reactivation/partial effect |
| source WM regression | first-key protocol error; equality only suppresses |
| selected input after scratch/committed End | whole finite snapshot fails with first error and all selected receipts fail |
| accepted post-fence tail after committed End | atomic extraction and one post-End failure per sender; no next drain |
| submit after committed End | immediate post-End rejection; no accepted identity |
| snapshot non-quiescent/unsettled | capture rejected |
| job/config/cursor/frontier/clock/allocator/gate/timer/trace mismatch | restore/replay `DriverPhaseError` before governed effects |
| missing/extra/reordered/short/long execution trace | complete-plan or per-record `DriverPhaseError`; no mismatched-record effect |
| different full execution trace | explicit trace inequality; no deterministic-outcome claim across traces |
| dropped receipt receiver | ignore send failure; retain commit/error semantics; no retry |
| pre-key counter/trace/gate overflow | `DriverPhaseError`, no fabricated key/partial effect, settle every accepted sender before exit |
| selected-item counter/deadline overflow | keyed error only after complete key assignment; atomic selected failure and cleanup |
| cancellation | stable cancellation failure for every accepted sender, clean exit |

### 16.3 Adversarial defense table

| Adversarial condition | Defense |
| --------------------- | ------- |
| adapter forges time/key/fence/ordinal/sequence | raw submit surface cannot express them; constructors remain private |
| arrival after current fence | deferred, except committed-End tail is extracted and failed during commit |
| input and both timers at one instant | `InputOrControl < WatermarkTimer < IdleTimer`, then ordinal/local sequence |
| delayed watermark wake crosses many ticks | one expiry and smallest future phase deadline; no burst |
| same-instant activity and old idle timer | input wins, generation makes old timer stale |
| valid prefix then invalid suffix | first-error short-circuit; entire selected snapshot fails; all selected receipts settle with same error |
| multiple invalid selected items | lowest `ReadyKey` error wins; no suffix semantic hook |
| End/Data have different natural batching | full execution trace captures admission/fence/settlement; different trace is inequality |
| final End races accepted tail | recorded attempt and gate cut; replay waits for exact tail then serialized close/extract |
| submit after End | admission record observes closed generation/cause and rejects without receipt/inbox/settlement |
| cancel/fatal races admission | terminal record fixes all close cuts/tails; settlement records fix every accepted outcome |
| missing/extra settlement or suffix | prevalidation/ordered cursor rejects before governed send/transition effect |
| receiver disappears | one send attempt is consumed and ignored; no rollback/retry |
| all-ended tries to exit early | terminal cleanup proves every accepted sender attempted and all gates/queues/timers/wakes empty |
| trace/admission/gate/receipt/inbox/fence/epoch overflow | checked probe, phase-correct error, no wrap/partial effect, fatal settlement cleanup |
| source cannot exact-pause/seek | restore-capable capture deterministically unsupported |
| output backpressure differs | only the resulting full execution trace is semantic; wall-clock delay is not replayed |
| old/equal/new row time relative to WM | all accepted data remains unclassified and unchanged |

## 17. D1–D9 and FR-001–FR-079 mapping

### 17.1 Decision closure checklist

| Decision | Concrete closure |
| -------- | ---------------- |
| D1 | data-neutral `ForwardData`; no M3 late state/metric; M4 predicate remains `window_end <= WM_in`. |
| D2 | complete in-memory coordinate plus exact raw-attempt/clock/full-execution trace; different full trace is explicitly another execution; no M5 claim. |
| D3 | pure compiler plus side-effect-free whole-job preflight with exact config fingerprint. |
| D4 | full activity/idle/end state machine and first-key atomic scratch transaction. |
| D5 | one driver owns time, full execution trace, fences, ready order, timers, checked coordinates, settlement, and cleanup. |
| D6 | exhaustive capability matrix and runtime illegal-native checks. |
| D7 | all additions crate-private; public/API bindings frozen; A6 post-M5. |
| D8 | accepted envelopes own senders; End commit atomically closes/extracts; cancel/fatal/all-ended settle all. |
| D9 | complete trace records every admission/terminal/settlement; ordered replay consumes records before effects; pre-key errors use `DriverPhaseError`. |

### 17.2 Functional requirement mapping

| FR | Concrete API seam / obligation |
| -- | ------------------------------ |
| FR-001 | unchanged `compile_stream()` is graph-only; `prepare_stream_job` is separate. |
| FR-002 | preparation is immutable/side-effect-free and precedes runtime construction. |
| FR-003 | `SourceDescriptor` carries schema state and capability without open. |
| FR-004 | prepared table order is stable; driver alone checked-allocates ordinals. |
| FR-005 | normalization validates every applicable duration and representation. |
| FR-006 | `ResolvedEventTimeColumn` requires known schema and supported timestamp unit. |
| FR-007 | the 4 × 3 table yields exactly one normalized mode. |
| FR-008 | binding/job fingerprints include exact capacity and fence selection. |
| FR-009 | `ProgressFailure.path` is stable and binding-qualified. |
| FR-010 | prepared job owns no opened connector/live task/timer/receiver. |
| FR-011 | `ArrowTimestampUnit` covers second/milli/micro/nano. |
| FR-012 | `observe_batch` retains the maximum representable non-null event time. |
| FR-013 | empty/null-only batches do not advance generated progress. |
| FR-014 | unit conversion and bounded-OOO subtraction are checked. |
| FR-015 | only an eligible current WM timer can emit strict generated advancement. |
| FR-016 | source regression errors; equality suppresses. |
| FR-017 | per-input and aggregate guards never regress effective watermark. |
| FR-018 | Generated/Disabled reject connector WM; SourceProvided has no generation. |
| FR-019 | idle rearm uses the qualifying driver instant only. |
| FR-020 | generation-proven stale timers alone are effect-free. |
| FR-021 | Idle retains effective watermark for state/status/snapshot. |
| FR-022 | Data/legal WM reactivates before event semantics. |
| FR-023 | `Active(None)` blocks a known aggregate minimum. |
| FR-024 | aggregate emits only a strictly advancing known active minimum. |
| FR-025 | latch plus checked `IdleEpoch` yields one Idle per epoch. |
| FR-026 | all-ended yields one End, no Idle/MAX. |
| FR-027 | end-induced Watermark is buffered before End. |
| FR-028 | selected post-End raw event is first-key atomic failure; terminal tails use End extraction. |
| FR-029 | one `StreamProgressDriver` owns all semantic progress/order state. |
| FR-030 | checked drain epoch/fences freeze exact raw membership; driver creates keys and records trace. |
| FR-031 | generation validates stale entries; ordinal/local sequence break ties. |
| FR-032 | capture requires receipt-quiescent phase. |
| FR-033 | coordinate contains full execution-trace prefix/position and every trace/admission/gate/settlement/drain/fence allocator coordinate. |
| FR-034 | binding snapshot contains identity/order/config/frontier/activity/gate generation/state/close cut/WM/timers/allocators/fence. |
| FR-035 | driver snapshot contains aggregate state, all next coordinates, full execution trace, and zero unsettled proof. |
| FR-036 | restore compares every fingerprint, trace position, gate generation/cut, allocation, fence, frontier, and timer before effects. |
| FR-037 | restore copies pending timer state and sequences exactly. |
| FR-038 | same ordered raw attempts + clock + full `ProgressExecutionTrace` produces item-identical admissions/receipts/errors/gates/terminal result. |
| FR-039 | snapshot/replay types have no stable serialization or persistence. |
| FR-040 | cancel/fatal/terminal record exact cause/cuts/tails/settlements, close gates, clear heap/wake, and prevent later emission. |
| FR-041 | accepted data is forwarded unchanged without WM-based lateness. |
| FR-042 | status/error/metric types expose no late-row observation/drop. |
| FR-043 | module plan has no M4 window dependency. |
| FR-044 | all additions remain crate-private; public/Python/REST/OpenAPI unchanged. |
| FR-045 | raw submit cannot carry time/key/ordinal/sequence/timer/fence. |
| FR-046 | all selected input uses the one driver drain instant; no input method takes `now`. |
| FR-047 | initial generated deadline is checked `running + interval`. |
| FR-048 | recurrence is prior-deadline phase anchored and missed ticks coalesce once. |
| FR-049 | idle deadline is checked `qualifying activity instant + timeout`. |
| FR-050 | driver-private timer allocator alone mutates timer lifecycle. |
| FR-051 | after complete key assignment, sorted evaluation short-circuits at first selected-item error; scratch owns effects/senders. |
| FR-052 | keyed selected failure commits zero and fails every selected sender; pre-key failures follow phase-error API. |
| FR-053 | non-exact source rejects restore-capable capture. |
| FR-054 | paused exact upstream token precedes restore validation/installation. |
| FR-055 | trace/cursor/admission/gate/settlement/receipt/inbox/fence/drain/order/timer/epoch counters all use checked successors. |
| FR-056 | exhaustion uses existing error container, zero governed partial effect, and complete accepted settlement. |
| FR-057 | full trace retains drain projection plus every admission/terminal/settlement/phase-failure lifecycle record. |
| FR-058 | replay consumes exact ordered attempts/clock/full execution trace; different full trace carries no outcome promise. |
| FR-059 | admission checked-allocates attempt first, records observed gate and accepted/rejected decision before visibility. |
| FR-060 | matching settlement record commits before receipt send; success follows state/frontier/fence commit and error acks no cursor. |
| FR-061 | dropped receiver changes only send observation and retains no driver work. |
| FR-062 | End terminal record fixes cause/clock/generation/close ordinal/cut/exact tail; guard closes/extracts without another drain. |
| FR-063 | future post-End attempt records closed generation/cause and rejects without receipt/inbox/enqueue/settlement. |
| FR-064 | all-ended exits after terminal/settlement suffix consumption, all send attempts, and empty gates/queues/timers/wake/work. |
| FR-065 | cancellation records exact transition/cuts/tails and settlements even without a drain, then cleans and exits. |
| FR-066 | lowest failing selected `ReadyKey` wins; no suffix hook; rule never applies to pre-key phase failures. |
| FR-067 | §§2, 17, and 21 preserve prior fixes and M3/M4/M5/A6 firewall. |
| FR-068 | `AdmissionAttemptRecord` is one-for-one with attempts and records ordinal/binding/gate/accepted-or-rejected outcome. |
| FR-069 | `TerminalTransitionRecord` losslessly records End/cancel/fatal cause, clock, close coordinates/cuts, and exact tails. |
| FR-070 | `SettlementRecord` gives every accepted identity exactly one owned stable disposition; rejections have none. |
| FR-071 | settlement canonical trace order is not receipt send/wakeup order; receiver drop changes no disposition. |
| FR-072 | prevalidated replay consumes admission/drain/terminal/settlement records before their governed effects. |
| FR-073 | snapshot round-trips complete execution-trace/gate/next-coordinate state. |
| FR-074 | trace/position/attempt/generation/close/settlement counters use checked allocation and fatal settlement cleanup. |
| FR-075 | `DriverPhaseError` covers every pre-key phase with exact coordinate and no fabricated `ReadyKey`. |
| FR-076 | replay prevalidation proves one admission per attempt, one settlement per acceptance, valid owners/cuts, and complete suffix. |
| FR-077 | terminal replay waits for exact recorded accepted tails and validates cut/batch under driver/gate ownership before effects. |
| FR-078 | fatal/cancel trace records include cause/clock/gates/cuts/selected+unselected membership and settlements. |
| FR-079 | full-trace/phase-error refinements preserve all D1–D8, AC-01–75, M2, no-late, and milestone firewalls. |

Mapping count: **D1–D9 = 9/9; FR-001–FR-079 = 79/79.**

## 18. AC-01–AC-85 API evidence mapping

Every criterion requires behavioral evidence. Type-existence tests alone do not satisfy it. Equivalent test names require an explicit PR mapping.

| AC | Required named test | API behavior under test |
| -- | ------------------- | ----------------------- |
| AC-01 | `compile_stream_remains_binding_agnostic` | unchanged compiler has no binding/catalog/config dependency. |
| AC-02 | `preflight_failure_has_no_runtime_side_effects` | failure precedes open/spawn/timer/pull. |
| AC-03 | `generated_policy_rejects_unknown_schema_before_open` | dynamic/unknown Generated rejects. |
| AC-04 | `generated_policy_reports_missing_column_path` | missing column returns exact binding path. |
| AC-05 | `generated_policy_rejects_unsupported_event_time_type` | only four timestamp units resolve. |
| AC-06 | `watermark_policy_rejects_invalid_durations` | zero/unrepresentable durations reject. |
| AC-07 | `watermark_policy_capability_matrix_is_exhaustive` | all 12 matrix cells have one outcome. |
| AC-08 | `runtime_toggleable_requires_existing_private_hook` | toggleable requires existing route. |
| AC-09 | `watermark_modes_never_silently_merge` | one producer only; illegal native input errors. |
| AC-10 | `generated_watermark_supports_all_timestamp_units` | checked unit normalization. |
| AC-11 | `generated_watermark_uses_non_null_max` | maximum non-null selection. |
| AC-12 | `empty_or_null_batch_does_not_advance_watermark` | progress-neutral empty/null input. |
| AC-13 | `generated_watermark_checks_event_time_arithmetic` | conversion/subtraction controlled failure. |
| AC-14 | `generated_watermark_never_regresses_or_duplicates` | strict generated advancement. |
| AC-15 | `source_watermark_is_monotonic` | regression error/equality suppression. |
| AC-16 | `illegal_connector_watermark_is_rejected` | illegal native WM errors before reactivation. |
| AC-17 | `queued_data_precedes_timers_at_same_deadline` | input/control class wins. |
| AC-18 | `watermark_timer_precedes_idle_timer_across_bindings` | WM timer class wins. |
| AC-19 | `driver_owned_ordinals_and_sequences_break_timer_ties` | driver-only stable tie break. |
| AC-20 | `timer_generation_is_validation_only` | generation absent from `ReadyKey`. |
| AC-21 | `new_ready_work_waits_for_next_arbitration_snapshot` | above-fence deferral and End-tail exception. |
| AC-22 | `active_ingress_without_watermark_holds_progress` | `Active(None)` blocks. |
| AC-23 | `multi_input_watermark_is_active_minimum` | known active minimum. |
| AC-24 | `idle_and_ended_ingresses_are_excluded_from_minimum` | Idle/Ended exclusion. |
| AC-25 | `multi_input_emits_idle_once_per_idle_epoch` | one checked epoch emission. |
| AC-26 | `repeated_idle_is_idempotent_within_epoch` | latch suppression. |
| AC-27 | `data_reactivation_precedes_processing` | data reactivation order. |
| AC-28 | `legal_watermark_reactivation_precedes_aggregation` | legal WM reactivation order. |
| AC-29 | `reactivation_starts_a_new_idle_epoch` | latch clear then one later epoch. |
| AC-30 | `reactivation_cannot_regress_aggregate_watermark` | strict aggregate monotonicity. |
| AC-31 | `final_watermark_advancement_precedes_end` | end-induced ordering. |
| AC-32 | `all_ended_emits_no_idle_or_sentinel_watermark` | one End only. |
| AC-33 | `post_end_input_aborts_whole_ready_snapshot_atomically` | selected post-End first error, zero effects, all selected failure receipts. |
| AC-34 | `progress_snapshot_requires_quiescent_boundary` | no capture with scratch/unsettled sender. |
| AC-35 | `progress_snapshot_roundtrips_complete_logical_coordinate` | full job/clock/frontier/gate/trace/timer/allocator/fence state. |
| AC-36 | `progress_restore_rejects_prepared_job_mismatch` | job mismatch before construction. |
| AC-37 | `progress_restore_rejects_binding_identity_mismatch` | identity/ordinal/order mismatch. |
| AC-38 | `progress_restore_rejects_normalized_config_mismatch` | config/fence fingerprint mismatch. |
| AC-39 | `progress_restore_requires_exact_captured_coordinate` | exact clock/frontier/gate/trace/sequence/fence/timer equality. |
| AC-40 | `progress_replay_is_deterministic_from_exact_coordinate` | same attempts/clock/full execution trace replay 100 times with all admissions/receipts/errors/gates/terminal identical. |
| AC-41 | `cancel_unregisters_all_progress_timers` | cancel records cuts/tails/settlements, closes gates, clears timers/wake, no later emission. |
| AC-42 | `terminal_end_leaves_no_progress_work` | terminal waits for trace suffix, settlements, and empty gates/queues/heap/wake. |
| AC-43 | `m3_does_not_classify_or_drop_late_rows` | old/equal/new event-time data forwards unchanged. |
| AC-44 | `m3_observability_has_no_late_row_metric` | no late field/metric/log/error. |
| AC-45 | `m3_builds_without_window_operator` | no M4 dependency. |
| AC-46 | `m3_preserves_public_rust_api` | public Rust baseline diff empty. |
| AC-47 | `m3_preserves_external_surface_baselines` | Python/REST/OpenAPI diff empty. |
| AC-48 | `m2_runtime_regression_suite_remains_green` | M2 lifecycle/backpressure/cancel/terminal unchanged. |
| AC-49 | `adapter_cannot_supply_or_forge_ready_key` | raw signature cannot express semantic fields. |
| AC-50 | `input_control_uses_driver_owned_logical_instant` | one driver instant, no caller `now`. |
| AC-51 | `first_watermark_deadline_is_running_instant_plus_interval` | exact checked initial deadline. |
| AC-52 | `watermark_cadence_is_phase_anchored` | recurrence from prior scheduled deadline. |
| AC-53 | `missed_watermark_ticks_coalesce_deterministically` | one expiry and smallest next phase. |
| AC-54 | `idle_deadline_uses_last_driver_activity_instant` | exact qualifying activity anchor. |
| AC-55 | `timer_lifecycle_is_driver_owned` | private allocator exclusively owns timer lifecycle. |
| AC-56 | `ready_snapshot_protocol_error_discards_valid_prefix` | first-error short-circuit and zero prefix visibility. |
| AC-57 | `ready_snapshot_success_commits_atomically` | sender-owning commit settles only after state/timer/frontier/fence commit. |
| AC-58 | `post_end_stale_internal_timer_is_effect_free` | stale-generation-only exception. |
| AC-59 | `progress_snapshot_rejects_non_replayable_source` | non-exact source cannot create restore-capable snapshot. |
| AC-60 | `progress_restore_requires_paused_exact_upstream_position` | guarded exact source position before install. |
| AC-61 | `sequence_exhaustion_aborts_without_mutation` | local/global exhaustion no wrap/partial commit/unsettled work. |
| AC-62 | `timer_counter_exhaustion_aborts_without_mutation` | timer generation/sequence fatal atomicity. |
| AC-63 | `idle_epoch_exhaustion_aborts_without_mutation` | idle epoch fatal atomicity. |
| AC-64 | `ordinal_or_deadline_exhaustion_is_atomic_fatal_error` | pre-admission ordinal or scratch deadline failure and cleanup. |
| AC-65 | `progress_replay_reproduces_full_progress_execution_trace` | admissions, drains/empties, terminals/cuts, settlements, and phase failures consume exactly. |
| AC-66 | `same_raw_and_clock_with_different_fences_is_trace_inequality` | any different admission/fence/terminal/settlement trace has no cross-trace promise. |
| AC-67 | `commit_receipt_succeeds_only_after_atomic_commit` | actual sender is retained and success follows semantic/fence commit. |
| AC-68 | `failed_drain_settles_all_selected_with_first_error` | lowest key error, no suffix hook, every selected sender failed once. |
| AC-69 | `final_end_atomically_rejects_post_fence_tail` | recorded attempt/cut/tail links exact accepted final-End failure without next drain. |
| AC-70 | `submit_after_end_is_rejected_immediately` | rejection record observes closed generation/cause and allocates/enqueues no accepted identity. |
| AC-71 | `dropped_receipt_receiver_does_not_rollback_commit` | ignored send failure, one attempt, no retained work. |
| AC-72 | `all_ended_waits_for_settlement_and_cleanup` | terminal/settlement records consumed and all gates/timers/wakes clean before exit. |
| AC-73 | `cancel_settles_every_accepted_receipt` | exact cancellation race cut and every selected/unselected settlement record/send before exit. |
| AC-74 | `receipt_fence_epoch_and_inbox_counter_overflow_is_atomic` | checked trace/admission/gate/receipt/fence/epoch allocation and complete fatal settlement. |
| AC-75 | `m3_delta_prior_decisions_and_scope_firewall_remain_intact` | original conflicts, old blocker repairs, no-late, M4/M5/A6 firewall. |
| AC-76 | `progress_replay_reproduces_terminal_tail_linearization` | exact tail attempt, gate generation/ordinal/cut, extraction, and post-End settlement replay. |
| AC-77 | `progress_replay_reproduces_fatal_cleanup_settlements` | accepted-versus-rejected fatal race, gate cuts/tails, and fatal settlements replay. |
| AC-78 | `progress_replay_reproduces_cancellation_settlements` | cancellation race and exact settlement membership replay without requiring a drain. |
| AC-79 | `progress_replay_reproduces_immediate_rejection` | post-End rejection replays attempt/gate/cause/reason with no accepted identity/settlement. |
| AC-80 | `progress_replay_reproduces_settlement_dispositions` | all five dispositions replay by accepted identity; receiver drop changes none. |
| AC-81 | `progress_replay_rejects_missing_or_extra_settlement_record` | missing/extra/reordered/wrong-owner lifecycle record fails before governed effect. |
| AC-82 | `progress_replay_requires_complete_expected_trace_suffix` | short/long/missing/extra attempt or unconsumed suffix fails complete-plan validation. |
| AC-83 | `progress_snapshot_roundtrips_execution_trace_coordinate` | exact trace prefix/position, next trace/attempt/close/settlement, and gate snapshots round-trip. |
| AC-84 | `execution_trace_admission_and_gate_counter_overflow_is_atomic` | trace/cursor/attempt/generation/close/settlement overflow is checked, phase-atomic, settlement-complete. |
| AC-85 | `pre_key_driver_phase_error_has_no_ready_key` | admission/freeze/fence/drain/key/trace failure has exact phase coordinate, no key, zero effect, complete cleanup. |

Mapping count: **AC-01–AC-85 = 85/85; named tests = 85/85.**

Additional refinement tests remain useful and do not replace an AC:

| Seam | Named test |
| ---- | ---------- |
| generated timer without candidate | `generated_watermark_rearms_without_candidate` |
| generated suppressed candidate | `generated_watermark_rearms_after_suppressed_candidate` |
| explicit connector Idle | `connector_idle_unregisters_idle_timer_and_preserves_wm_phase` |
| restore central wake | `restore_registers_only_exact_heap_minimum_wake` |
| exact earlier/later clock rejection | `restore_rejects_same_trace_earlier_or_later_instant` |
| required empty replay epoch | `replay_consumes_required_empty_drain_epoch` |
| admission allocation atomicity | `receipt_and_inbox_allocation_are_one_atomic_admission` |
| fatal cleanup queued tail | `fatal_cleanup_settles_selected_and_unselected_once` |
| observational saturation | `observational_counter_saturates_without_semantic_effect` |

## 19. Verification and merge-gate obligations

| Gate | Required evidence |
| ---- | ----------------- |
| AC evidence | explicit named mapping and behavioral result for AC-01–AC-85. |
| deterministic ordering | fake/paused logical clock; no semantic assertion based on real sleep. |
| deterministic stress | at least 100 seeds with ordered attempts, clock, full execution trace, exact config, admissions, gates/cuts, settlements, phase failures, and terminal result. |
| exact transient replay | exact coordinate plus same complete trace replayed 100 times with item-identical admissions, emissions, receipts, errors, gates, and terminal result. |
| different-trace boundary | same attempts/clock with different admission/fence/terminal/settlement trace is explicit inequality. |
| receipt ownership | success, first-error, End tail, immediate reject, cancel/fatal races, dropped receiver, all-ended, and all settlement dispositions. |
| trace completeness | missing/extra/reordered/wrong-owner/short/long/unconsumed records fail before the mismatched governed effect. |
| transaction atomicity | valid-prefix/fatal-suffix proves keyed first error; pre-key matrix proves `DriverPhaseError` without fabricated key. |
| exhaustion | trace/cursor/attempt/gate/settlement/order/timer/receipt/inbox/fence/epoch/deadline counters never wrap or strand work. |
| M2 safety | complete lifecycle, bounded-backpressure, cancellation, receipt, and terminal regression matrix remains green. |
| soak | standardized 20-minute run with admissions, drains/fences/timers, gate cuts, per-path settlements, max unsettled, phase errors, terminal/cancel/fatal and resources. |
| API firewall | public Rust, Python, REST, and OpenAPI baseline diffs are empty. |
| scope firewall | no M4 window dependency, M5 durability, or post-M5 A6 exposure. |
| automated review | CI green; no unresolved introduced/exposed Codacy issue; Copilot/human threads resolved; designated approval. |
| document gate | reconciled spec/API/critique and `BLOCKS REMAINING: 0`. |

The 20-minute duration is the default M3 and future soak standard unless a later controlling specification replaces it.

## 20. Non-goals at the API boundary

M3 defines none of the following:

- late-row/late-assignment classification, counting, tagging, split, or drop;
- `LateDataPolicy` execution;
- row-level `event_time < watermark` or `<= watermark` decisions;
- window assignment or `operator/window.rs` integration;
- allowed lateness, side output, or retraction;
- snapshot serde/bytes/version/manifest/checkpoint barrier;
- process restart, crash/durable recovery, deadline rebase, or exactly-once recovery;
- public watermark policy/capability/status/runner/Python/REST/OpenAPI additions.

M4 may consume aggregate progress only after a concrete assignment exists and may drop only `assignment.window_end <= WM_in`. M5 may replace this in-memory representation under a new durable version/migration contract. A6 remains post-M5.

## 21. Round-three critique closure and critic handoff

### 21.1 B-R3-01 — outside-selected lifecycle replay

Closed in place by §§5, 9, 12, 13, and 15:

- `ProgressExecutionTrace` is the only trace named as sufficient; `DrainFenceTrace` is explicitly a drain-only projection;
- `AdmissionAttemptRecord` fixes attempt order, observed gate generation/state, and accepted identity or immediate rejection;
- `TerminalTransitionRecord` fixes End/cancel/fatal cause, logical clock, gate generations/close ordinals/cuts, and exact extracted tails;
- `SettlementRecord` fixes exactly one disposition/outcome for every accepted identity and none for a rejection;
- `ProgressReplayRequest::prevalidate` rejects structurally missing/extra/reordered/short/long/wrong-owner plans before runtime construction;
- `ProgressReplayCursor` consumes admission, drain, terminal, settlement, and phase-failure records before their governed effect;
- terminal replay waits for recorded accepted tails, rejects unexpected identities/cuts, and does not rediscover the boundary from scheduler timing;
- snapshot/restore carries the full prefix/position and every next trace/admission/gate/settlement/drain/inbox/fence coordinate;
- AC-40, AC-65–AC-66, and AC-76–AC-84 cover deterministic full-trace replay and lifecycle races.

### 21.2 N-R3-01 — phase-correct error identity

Closed in place by §§7, 11–12, 15.4, and 16:

- `SelectedItemError` exists only after every selected item has a complete `ReadyKey` and identifies the lowest failing key;
- `DriverPhaseError` identifies admission/freeze/fence/drain/key/trace/gate/settlement failure by exact phase coordinate with no key field;
- drain outcome has distinct selected-item and driver-phase forms, and epoch-allocation exhaustion invents neither epoch nor key;
- pre-key failures expose zero governed effects and enter complete recorded fatal settlement cleanup;
- AC-74, AC-84, and AC-85 exercise counter boundaries and no-fabricated-key behavior.

### 21.3 Prior repairs retained

The fourth revision retains:

1. raw-only submission and driver-owned time/key/order;
2. exact capacity/fence fingerprint with no free restore override;
3. sender ownership through success, selected failure, End tail, cancel/fatal, and receiver drop;
4. End close-and-extract and immediate future rejection;
5. first selected error, no suffix semantic hook, and zero-prefix visibility;
6. phase-anchored/coalesced generated timers and driver-owned idle deadlines;
7. exact paused upstream/frontier/clock/timer restore;
8. stale-internal-timer-only post-End suppression and checked exhaustion;
9. pure compiler and preflight-before-open;
10. no public v2/Python/REST/OpenAPI change or M3 late metric;
11. M4 `window_end <= WM_in`, M5 durability, and post-M5 A6 boundaries.

There are no intentionally blocking open questions in this API note.

## 22. Definition of API-note completion

This API note is complete when implementation can prepare an immutable side-effect-free job; normalize exactly one watermark mode; accept only raw attempts into one driver-owned checked admission protocol; prevalidate and consume the complete admission/drain/terminal/settlement execution trace before governed effects; distinguish selected-item errors from pre-key driver-phase errors; move every accepted sender to one recorded disposition and one send attempt; replay End/cancel/fatal races exactly; capture and restore the complete in-memory trace/gate/allocation coordinate; expose bounded crate-private status with no late metric; satisfy D1–D9, FR-001–FR-079, and AC-01–AC-85; and change no public surface.
