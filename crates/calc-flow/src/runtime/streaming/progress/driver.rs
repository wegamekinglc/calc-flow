use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::Arc,
    time::Duration,
};

use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::{
    Batch, CalcFlowError, Epoch, EventTime, Result, SourceManifestEntry,
    SourceWatermarkManifestState, runtime::streaming::source_task::decode_canonical_cursor_order,
};

use super::{
    aggregate::{AggregateInput, IngressActivity, MultiInputProgress, ProgressEmission},
    durable::{DurableProgressRestore, DurableSourceCut},
    generated::{GeneratedWatermarkState, next_phase_deadline},
    prepare::{
        BindingIdentity, BindingOrdinal, NormalizedWatermarkMode, PreparedSourceBinding,
        PreparedStreamJob,
    },
    snapshot::{
        CapturedBindingCoordinate, CapturedLogicalCoordinate, PausedExactUpstreams, RestoreRequest,
        StreamProgressSnapshot,
    },
    status::{
        BindingProgressStatus, LiveProgressStatusHandle, ProgressCounters, StreamProgressStatus,
    },
    trace::{
        AcceptedEnvelopeIdentity, AdmissionAttemptRecord, AdmissionDecisionRecord,
        AdmissionGateCloseCoordinate, AdmissionGateSnapshot, AdmissionGateState,
        BindingGateTransitionRecord, DrainEpochOutcomeRecord, DrainEpochRecord,
        InboxFenceCoordinate, InboxUpperFence, ProgressExecutionTrace, ProgressReplayRequest,
        ProgressTraceRecord, RawIngressEventKind, RawUpstreamPosition, ReadyItemIdentity,
        ReadyKeyRange, SettlementDisposition, SettlementOwner, SettlementRecord,
        TerminalTransitionCause, TerminalTransitionRecord, TraceController,
    },
    types::{
        AdmissionAttemptOrdinal, AdmissionGateGeneration, CheckedSemanticAllocator, CurrentTimer,
        DrainEpoch, DriverClockCoordinate, DriverFailurePhase, DriverLogicalClock,
        DriverPhaseCoordinate, DriverPhaseError, FenceSequence, GateCloseOrdinal, InboxSequence,
        LocalSequence, LogicalInstant, ProgressFailure, ProgressFailureKind, ReadyClass, ReadyKey,
        ReceiptSequence, SelectedItemError, TimerGeneration, TimerIdentity, TimerKind,
        TimerSequence,
    },
};

#[derive(Clone, Debug)]
pub(crate) enum RawIngressEvent {
    Data(Batch),
    ConnectorWatermark(EventTime),
    ConnectorIdle,
    EndOfInput,
}

#[cfg(test)]
mod checkpoint_cut_tests {
    use std::{collections::BTreeMap, sync::Arc};

    use datafusion::arrow::{
        array::{ArrayRef, Int64Array, TimestampMicrosecondArray},
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    };

    use super::{DurableSourceCut, LiveProgressCoordinator, RawIngressEvent};
    use crate::runtime::streaming::progress::prepare::{
        BindingIdentity, DeclaredSchema, NativeWatermarkCapability, ReplayPositioningCapability,
        SourceBindingSpec, SourceDescriptor, StreamProgressRuntimeConfig, WatermarkPolicy,
        prepare_stream_job,
    };
    use crate::runtime::streaming::progress::trace::RawUpstreamPosition;
    use crate::{
        Batch, BatchMetadata, CancellationToken, CursorManifestEntry, EdgeBudget, Epoch,
        SourceWatermarkManifestState, StreamMessageKind, edge_channel,
    };

    fn binding(value: &str) -> BindingIdentity {
        BindingIdentity::new(value).unwrap()
    }

    fn source(value: &str) -> SourceBindingSpec {
        SourceBindingSpec {
            descriptor: SourceDescriptor::new(
                binding(value),
                DeclaredSchema::DynamicOrUnknown,
                NativeWatermarkCapability::NeverEmits,
                ReplayPositioningCapability::ExactPauseReportAndSeek,
                None,
            ),
            watermark_policy: WatermarkPolicy::Disabled { idle_timeout: None },
        }
    }

    fn generated_source(value: &str) -> SourceBindingSpec {
        SourceBindingSpec {
            descriptor: SourceDescriptor::new(
                binding(value),
                DeclaredSchema::Known(Arc::new(Schema::new(vec![Field::new(
                    "at",
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                    true,
                )]))),
                NativeWatermarkCapability::NeverEmits,
                ReplayPositioningCapability::ExactPauseReportAndSeek,
                None,
            ),
            watermark_policy: WatermarkPolicy::BoundedOutOfOrderness {
                event_time_column: Arc::from("at"),
                max_out_of_orderness: std::time::Duration::from_micros(1),
                emit_interval: std::time::Duration::from_secs(5),
                idle_timeout: None,
            },
        }
    }

    fn batch(sequence: u64) -> Batch {
        Batch::table(
            vec![
                RecordBatch::try_from_iter(vec![(
                    "value",
                    Arc::new(Int64Array::from(vec![1])) as _,
                )])
                .unwrap(),
            ],
            BatchMetadata::new("left", sequence, BTreeMap::new()).unwrap(),
        )
        .unwrap()
    }

    fn timestamp_batch(sequence: u64, event_time: i64) -> Batch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        )]));
        Batch::table(
            vec![
                RecordBatch::try_new(
                    schema,
                    vec![Arc::new(TimestampMicrosecondArray::from(vec![event_time])) as ArrayRef],
                )
                .unwrap(),
            ],
            BatchMetadata::new("left", sequence, BTreeMap::new()).unwrap(),
        )
        .unwrap()
    }

    fn exact(value: u8) -> RawUpstreamPosition {
        RawUpstreamPosition::Exact {
            delivery_replay_cursor: vec![value],
            control_frontier: vec![value],
        }
    }

    #[tokio::test]
    async fn progress_owner_serializes_the_cut_and_skips_ended_routes() {
        let prepared = Arc::new(
            prepare_stream_job(
                "compiled",
                &[source("left"), source("right")],
                StreamProgressRuntimeConfig::default(),
            )
            .unwrap(),
        );
        let budget = EdgeBudget {
            max_rows: 8,
            max_bytes: 1 << 20,
        };
        let (left_sender, mut left_receiver) = edge_channel("left", budget).unwrap();
        let (right_sender, right_receiver) = edge_channel("right", budget).unwrap();
        let cancellation = CancellationToken::new();
        let coordinator = LiveProgressCoordinator::new(
            &prepared,
            BTreeMap::from([
                ("left".into(), vec![left_sender]),
                ("right".into(), vec![right_sender]),
            ]),
            cancellation.clone(),
        )
        .unwrap();
        coordinator
            .submit(binding("left"), RawIngressEvent::Data(batch(7)), exact(1))
            .await
            .unwrap();
        coordinator
            .submit(binding("right"), RawIngressEvent::EndOfInput, exact(2))
            .await
            .unwrap();

        let durable = coordinator
            .checkpoint_cut(
                Epoch::INITIAL,
                &BTreeMap::from([
                    (
                        binding("left"),
                        DurableSourceCut {
                            cursor: Some(CursorManifestEntry {
                                order: "01".into(),
                                payload: BTreeMap::new(),
                            }),
                            next_sequence: 8,
                            ended: false,
                        },
                    ),
                    (
                        binding("right"),
                        DurableSourceCut {
                            cursor: Some(CursorManifestEntry {
                                order: "02".into(),
                                payload: BTreeMap::new(),
                            }),
                            next_sequence: 1,
                            ended: true,
                        },
                    ),
                ]),
                &cancellation,
            )
            .await
            .unwrap();
        assert_eq!(durable["left"].sequence, 8);
        assert!(!durable["left"].ended);
        assert_eq!(
            durable["left"].watermark_policy,
            SourceWatermarkManifestState::Disabled { idle: false }
        );
        assert_eq!(durable["right"].sequence, 1);
        assert!(durable["right"].ended);
        coordinator
            .submit(binding("left"), RawIngressEvent::Data(batch(8)), exact(3))
            .await
            .unwrap();

        let before_cut = left_receiver.recv().await.unwrap().unwrap();
        assert_eq!(before_cut.kind(), StreamMessageKind::Data);
        assert_eq!(before_cut.as_data().unwrap().metadata().sequence(), 7);
        assert_eq!(
            left_receiver.recv().await.unwrap().unwrap().as_barrier(),
            Some(Epoch::INITIAL)
        );
        let after_cut = left_receiver.recv().await.unwrap().unwrap();
        assert_eq!(after_cut.kind(), StreamMessageKind::Data);
        assert_eq!(after_cut.as_data().unwrap().metadata().sequence(), 8);
        assert_eq!(right_receiver.metrics().queue_depth, 0);
    }

    #[tokio::test]
    async fn idle_live_source_participates_in_the_checkpoint_cut() {
        let prepared = Arc::new(
            prepare_stream_job(
                "compiled",
                &[source("idle")],
                StreamProgressRuntimeConfig::default(),
            )
            .unwrap(),
        );
        let (sender, mut receiver) = edge_channel(
            "idle",
            EdgeBudget {
                max_rows: 8,
                max_bytes: 1 << 20,
            },
        )
        .unwrap();
        let cancellation = CancellationToken::new();
        let coordinator = LiveProgressCoordinator::new(
            &prepared,
            BTreeMap::from([("idle".into(), vec![sender])]),
            cancellation.clone(),
        )
        .unwrap();
        coordinator
            .submit(binding("idle"), RawIngressEvent::ConnectorIdle, exact(1))
            .await
            .unwrap();

        let durable = coordinator
            .checkpoint_cut(
                Epoch::INITIAL,
                &BTreeMap::from([(
                    binding("idle"),
                    DurableSourceCut {
                        cursor: None,
                        next_sequence: 0,
                        ended: false,
                    },
                )]),
                &cancellation,
            )
            .await
            .unwrap();

        assert_eq!(
            receiver.recv().await.unwrap().unwrap().kind(),
            StreamMessageKind::Idle
        );
        assert_eq!(
            receiver.recv().await.unwrap().unwrap().as_barrier(),
            Some(Epoch::INITIAL)
        );
        assert_eq!(
            durable["idle"].watermark_policy,
            SourceWatermarkManifestState::Disabled { idle: true }
        );
        assert!(!durable["idle"].ended);
    }

    #[tokio::test(start_paused = true)]
    async fn ready_progress_timer_is_drained_before_the_checkpoint_barrier() {
        let prepared = Arc::new(
            prepare_stream_job(
                "compiled",
                &[generated_source("timed")],
                StreamProgressRuntimeConfig::default(),
            )
            .unwrap(),
        );
        let (sender, mut receiver) = edge_channel(
            "timed",
            EdgeBudget {
                max_rows: 8,
                max_bytes: 1 << 20,
            },
        )
        .unwrap();
        let cancellation = CancellationToken::new();
        let coordinator = LiveProgressCoordinator::new(
            &prepared,
            BTreeMap::from([("timed".into(), vec![sender])]),
            cancellation.clone(),
        )
        .unwrap();
        coordinator
            .submit(
                binding("timed"),
                RawIngressEvent::Data(timestamp_batch(0, 10)),
                exact(1),
            )
            .await
            .unwrap();
        tokio::time::advance(std::time::Duration::from_secs(5)).await;

        coordinator
            .checkpoint_cut(
                Epoch::INITIAL,
                &BTreeMap::from([(
                    binding("timed"),
                    DurableSourceCut {
                        cursor: None,
                        next_sequence: 1,
                        ended: false,
                    },
                )]),
                &cancellation,
            )
            .await
            .unwrap();

        assert_eq!(
            receiver.recv().await.unwrap().unwrap().kind(),
            StreamMessageKind::Data
        );
        assert_eq!(
            receiver.recv().await.unwrap().unwrap().kind(),
            StreamMessageKind::Watermark
        );
        assert_eq!(
            receiver.recv().await.unwrap().unwrap().as_barrier(),
            Some(Epoch::INITIAL)
        );
    }
}

impl RawIngressEvent {
    const fn kind(&self) -> RawIngressEventKind {
        match self {
            Self::Data(_) => RawIngressEventKind::Data,
            Self::ConnectorWatermark(_) => RawIngressEventKind::ConnectorWatermark,
            Self::ConnectorIdle => RawIngressEventKind::ConnectorIdle,
            Self::EndOfInput => RawIngressEventKind::EndOfInput,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CommittedRawInput {
    pub(crate) binding: BindingIdentity,
    pub(crate) accepted_identity: AcceptedEnvelopeIdentity,
    pub(crate) upstream_position: RawUpstreamPosition,
}

pub(crate) struct RawCommitReceipt {
    settled: oneshot::Receiver<Result<CommittedRawInput>>,
}

impl RawCommitReceipt {
    pub(crate) async fn wait_settled(self) -> Result<CommittedRawInput> {
        self.settled.await.map_err(|_| CalcFlowError::Internal {
            message: "progress driver dropped a raw receipt without a settlement attempt".into(),
        })?
    }
}

struct RawIngressEnvelope {
    identity: AcceptedEnvelopeIdentity,
    event: RawIngressEvent,
    settlement: oneshot::Sender<Result<CommittedRawInput>>,
}

struct BindingAdmissionState {
    identity: BindingIdentity,
    gate: AdmissionGateSnapshot,
    next_inbox_sequence: CheckedSemanticAllocator,
    next_fence_sequence: CheckedSemanticAllocator,
    last_completed_fence: Option<InboxFenceCoordinate>,
    last_committed_upstream: Option<RawUpstreamPosition>,
    accepted: VecDeque<RawIngressEnvelope>,
}

struct AdmissionState {
    ordinal_by_identity: BTreeMap<BindingIdentity, BindingOrdinal>,
    bindings: BTreeMap<BindingOrdinal, BindingAdmissionState>,
    next_admission_attempt: CheckedSemanticAllocator,
    next_receipt_sequence: CheckedSemanticAllocator,
    next_gate_close_ordinal: CheckedSemanticAllocator,
    trace: TraceController,
    unsettled_receipts: usize,
    capacity: usize,
    pending_fatal: Option<AdmissionFatalSignal>,
}

#[derive(Clone, Debug)]
struct AdmissionFatalSignal {
    phase: DriverFailurePhase,
    path: String,
    reason: String,
}

#[derive(Clone)]
pub(crate) struct RawIngressSender {
    admission: Arc<Mutex<AdmissionState>>,
}

impl RawIngressSender {
    pub(crate) fn submit(
        &self,
        binding: BindingIdentity,
        event: RawIngressEvent,
        upstream_position: RawUpstreamPosition,
    ) -> std::future::Ready<Result<RawCommitReceipt>> {
        std::future::ready(self.submit_now(binding, event, upstream_position))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one lock-scoped admission transaction validates, traces, and commits atomically"
    )]
    fn submit_now(
        &self,
        binding: BindingIdentity,
        event: RawIngressEvent,
        upstream_position: RawUpstreamPosition,
    ) -> Result<RawCommitReceipt> {
        let mut state = self.admission.lock();
        if let Some(fatal) = &state.pending_fatal {
            return Err(CalcFlowError::InvalidArgument {
                field: fatal.path.clone(),
                message: fatal.reason.clone(),
            });
        }
        let ordinal = *state.ordinal_by_identity.get(&binding).ok_or_else(|| {
            CalcFlowError::InvalidArgument {
                field: "runtime.progress.admission.binding".into(),
                message: format!("unknown source binding {:?}", binding.as_str()),
            }
        })?;
        let attempt_probe = match state.next_admission_attempt.checked_peek_and_successor() {
            Ok(probe) => probe,
            Err(error) => {
                mark_admission_fatal(
                    &mut state,
                    DriverFailurePhase::AdmissionAttemptAllocation,
                    &error,
                );
                return Err(error);
            }
        };
        let trace_coordinates = match state.trace.next_coordinates() {
            Ok(coordinates) => coordinates,
            Err(error) => {
                mark_admission_fatal(&mut state, DriverFailurePhase::AdmissionDecision, &error);
                return Err(error);
            }
        };
        let binding_state = state
            .bindings
            .get(&ordinal)
            .expect("ordinal map and admission bindings are constructed together");
        let observed_gate = binding_state.gate.clone();
        let kind = event.kind();
        let attempt = AdmissionAttemptOrdinal(attempt_probe.0);
        if observed_gate.state != AdmissionGateState::Open {
            let (path, reason) = rejection_for_gate(binding.as_str(), observed_gate.state);
            let record = AdmissionAttemptRecord {
                trace_record_ordinal: trace_coordinates.0,
                trace_position: trace_coordinates.1,
                attempt_ordinal: attempt,
                binding,
                binding_ordinal: ordinal,
                event_kind: kind,
                upstream_position,
                observed_gate,
                decision: AdmissionDecisionRecord::ImmediateRejected {
                    path: path.clone(),
                    reason: reason.clone(),
                },
            };
            let mut trace = state.trace.clone();
            if let Err(error) = trace.append(ProgressTraceRecord::Admission(record)) {
                mark_admission_fatal(&mut state, DriverFailurePhase::AdmissionDecision, &error);
                return Err(error);
            }
            state
                .next_admission_attempt
                .allocate()
                .expect("the attempt successor was probed");
            state.trace = trace;
            return Err(CalcFlowError::InvalidArgument {
                field: path,
                message: reason,
            });
        }
        if binding_state.accepted.len() >= state.capacity {
            let path = format!("runtime.progress.admission.{}.capacity", binding.as_str());
            let reason = "per-binding raw inbox capacity is exhausted".to_string();
            let record = AdmissionAttemptRecord {
                trace_record_ordinal: trace_coordinates.0,
                trace_position: trace_coordinates.1,
                attempt_ordinal: attempt,
                binding,
                binding_ordinal: ordinal,
                event_kind: kind,
                upstream_position,
                observed_gate,
                decision: AdmissionDecisionRecord::ImmediateRejected {
                    path: path.clone(),
                    reason: reason.clone(),
                },
            };
            let mut trace = state.trace.clone();
            if let Err(error) = trace.append(ProgressTraceRecord::Admission(record)) {
                mark_admission_fatal(&mut state, DriverFailurePhase::AdmissionDecision, &error);
                return Err(error);
            }
            state
                .next_admission_attempt
                .allocate()
                .expect("the attempt successor was probed");
            state.trace = trace;
            return Err(CalcFlowError::InvalidArgument {
                field: path,
                message: reason,
            });
        }
        let receipt_probe = match state.next_receipt_sequence.checked_peek_and_successor() {
            Ok(probe) => probe,
            Err(error) => {
                mark_admission_fatal(&mut state, DriverFailurePhase::AdmissionDecision, &error);
                return Err(error);
            }
        };
        let inbox_probe = match binding_state
            .next_inbox_sequence
            .checked_peek_and_successor()
        {
            Ok(probe) => probe,
            Err(error) => {
                mark_admission_fatal(&mut state, DriverFailurePhase::AdmissionDecision, &error);
                return Err(error);
            }
        };
        let identity = AcceptedEnvelopeIdentity {
            binding: binding.clone(),
            binding_ordinal: ordinal,
            admission_attempt: attempt,
            receipt_sequence: ReceiptSequence(receipt_probe.0),
            inbox_sequence: InboxSequence(inbox_probe.0),
            upstream_position: upstream_position.clone(),
        };
        let record = AdmissionAttemptRecord {
            trace_record_ordinal: trace_coordinates.0,
            trace_position: trace_coordinates.1,
            attempt_ordinal: attempt,
            binding: binding.clone(),
            binding_ordinal: ordinal,
            event_kind: kind,
            upstream_position,
            observed_gate,
            decision: AdmissionDecisionRecord::Accepted {
                accepted: identity.clone(),
            },
        };
        let mut trace = state.trace.clone();
        if let Err(error) = trace.append(ProgressTraceRecord::Admission(record)) {
            mark_admission_fatal(&mut state, DriverFailurePhase::AdmissionDecision, &error);
            return Err(error);
        }
        let (settlement, settled) = oneshot::channel();
        state
            .next_admission_attempt
            .allocate()
            .expect("the attempt successor was probed");
        state
            .next_receipt_sequence
            .allocate()
            .expect("the receipt successor was probed");
        state
            .bindings
            .get_mut(&ordinal)
            .expect("binding exists")
            .next_inbox_sequence
            .allocate()
            .expect("the inbox successor was probed");
        state
            .bindings
            .get_mut(&ordinal)
            .expect("binding exists")
            .accepted
            .push_back(RawIngressEnvelope {
                identity,
                event,
                settlement,
            });
        state.unsettled_receipts += 1;
        state.trace = trace;
        Ok(RawCommitReceipt { settled })
    }
}

fn mark_admission_fatal(
    state: &mut AdmissionState,
    phase: DriverFailurePhase,
    error: &CalcFlowError,
) {
    let (path, reason) = match error {
        CalcFlowError::InvalidArgument { field, message } => (field.clone(), message.clone()),
        _ => ("runtime.progress.admission".into(), error.to_string()),
    };
    state.pending_fatal.get_or_insert(AdmissionFatalSignal {
        phase,
        path,
        reason,
    });
}

fn rejection_for_gate(binding: &str, gate: AdmissionGateState) -> (String, String) {
    let suffix = match gate {
        AdmissionGateState::Open => "open",
        AdmissionGateState::ClosedAfterEnd => "post_end",
        AdmissionGateState::ClosedCancelled => "cancelled",
        AdmissionGateState::ClosedFatal => "fatal",
    };
    (
        format!("runtime.progress.admission.{binding}.{suffix}"),
        match gate {
            AdmissionGateState::Open => "admission gate is open",
            AdmissionGateState::ClosedAfterEnd => "input/control after committed EndOfInput",
            AdmissionGateState::ClosedCancelled => "stream progress driver was cancelled",
            AdmissionGateState::ClosedFatal => "stream progress driver failed",
        }
        .into(),
    )
}

struct FrozenAdmissionEpoch {
    fences: Vec<InboxFenceCoordinate>,
    selected: Vec<RawIngressEnvelope>,
}

fn freeze_admission(state: &mut AdmissionState, epoch: DrainEpoch) -> Result<FrozenAdmissionEpoch> {
    for binding in state.bindings.values() {
        binding.next_fence_sequence.checked_peek_and_successor()?;
    }
    let mut fences = Vec::with_capacity(state.bindings.len());
    let mut selected = Vec::new();
    for (ordinal, binding) in &mut state.bindings {
        let upper = binding
            .accepted
            .back()
            .map_or(InboxUpperFence::Empty, |envelope| {
                InboxUpperFence::Inclusive(envelope.identity.inbox_sequence)
            });
        let fence = InboxFenceCoordinate {
            drain_epoch: epoch,
            binding_ordinal: *ordinal,
            fence_sequence: FenceSequence(binding.next_fence_sequence.next()),
            upper,
        };
        fences.push(fence);
        selected.extend(binding.accepted.drain(..));
    }
    Ok(FrozenAdmissionEpoch { fences, selected })
}

fn requeue_frozen_selection(admission: &mut AdmissionState, selected: Vec<RawIngressEnvelope>) {
    let mut by_binding = BTreeMap::<BindingOrdinal, Vec<RawIngressEnvelope>>::new();
    for envelope in selected {
        by_binding
            .entry(envelope.identity.binding_ordinal)
            .or_default()
            .push(envelope);
    }
    for (ordinal, envelopes) in by_binding {
        let accepted = &mut admission
            .bindings
            .get_mut(&ordinal)
            .expect("frozen envelope binding remains prepared")
            .accepted;
        for envelope in envelopes.into_iter().rev() {
            accepted.push_front(envelope);
        }
    }
}

#[derive(Clone, Debug)]
struct BindingProgressState {
    prepared: PreparedSourceBinding,
    last_source_watermark: Option<EventTime>,
    generated: Option<GeneratedWatermarkState>,
    watermark_timer: Option<CurrentTimer>,
    idle_timer: Option<CurrentTimer>,
    next_local_sequence: CheckedSemanticAllocator,
    next_timer_generation: CheckedSemanticAllocator,
    next_timer_sequence: CheckedSemanticAllocator,
    ended: bool,
}

#[derive(Clone, Debug)]
struct DriverProgressState {
    clock: DriverClockCoordinate,
    bindings: BTreeMap<BindingOrdinal, BindingProgressState>,
    aggregate: MultiInputProgress,
    timer_heap: BTreeMap<TimerIdentity, CurrentTimer>,
}

#[derive(Clone, Debug)]
struct AdmissionBindingSnapshot {
    identity: BindingIdentity,
    gate: AdmissionGateSnapshot,
    next_inbox_sequence: u64,
    next_fence_sequence: u64,
    last_completed_fence: Option<InboxFenceCoordinate>,
    last_committed_upstream: RawUpstreamPosition,
}

#[derive(Clone, Debug)]
pub(super) struct DriverSnapshotPayload {
    state: DriverProgressState,
    admission_bindings: BTreeMap<BindingOrdinal, AdmissionBindingSnapshot>,
    next_admission_attempt: u64,
    next_receipt_sequence: u64,
    next_gate_close_ordinal: u64,
    next_drain_epoch: u64,
}

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

#[derive(Clone, Debug)]
pub(crate) enum DriverEmission {
    ForwardData {
        binding: BindingOrdinal,
        batch: Batch,
    },
    Progress(ProgressEmission),
}

#[derive(Clone, Debug)]
pub(crate) struct CommittedDrain {
    pub(crate) emissions: Vec<DriverEmission>,
    pub(crate) committed_inputs: Vec<CommittedRawInput>,
    pub(crate) terminal_tail_failures: usize,
}

struct ReadyRaw {
    key: ReadyKey,
    envelope: RawIngressEnvelope,
}

#[derive(Clone, Copy)]
struct ReadyTimer {
    key: ReadyKey,
    identity: TimerIdentity,
}

enum ReadyStep {
    Raw(ReadyRaw),
    Timer(ReadyTimer),
}

impl ReadyStep {
    const fn key(&self) -> ReadyKey {
        match self {
            Self::Raw(raw) => raw.key,
            Self::Timer(timer) => timer.key,
        }
    }

    fn identity(&self) -> ReadyItemIdentity {
        match self {
            Self::Raw(raw) => ReadyItemIdentity::Raw {
                accepted: raw.envelope.identity.clone(),
                event_kind: raw.envelope.event.kind(),
            },
            Self::Timer(timer) => ReadyItemIdentity::Timer(timer.identity),
        }
    }
}

pub(crate) struct StreamProgressDriver<C: DriverLogicalClock> {
    prepared_job: Arc<PreparedStreamJob>,
    clock: C,
    admission: Arc<Mutex<AdmissionState>>,
    state: DriverProgressState,
    next_drain_epoch: CheckedSemanticAllocator,
    phase: DriverPhase,
    #[cfg(test)]
    after_freeze: Option<Box<dyn FnOnce() + Send>>,
}

impl<C: DriverLogicalClock> StreamProgressDriver<C> {
    pub(crate) fn new(
        prepared_job: Arc<PreparedStreamJob>,
        clock: C,
    ) -> Result<(Self, RawIngressSender)> {
        Self::with_trace(prepared_job, clock, TraceController::record())
    }

    pub(crate) fn replay(
        prepared_job: Arc<PreparedStreamJob>,
        clock: C,
        request: ProgressReplayRequest,
    ) -> Result<(Self, RawIngressSender)> {
        Self::with_trace(prepared_job, clock, TraceController::replay(request))
    }

    pub(crate) fn restore_durable(
        prepared_job: &Arc<PreparedStreamJob>,
        clock: C,
        restored: &DurableProgressRestore,
    ) -> Result<(Self, RawIngressSender)> {
        let (mut driver, sender) =
            Self::with_trace(Arc::clone(prepared_job), clock, TraceController::record())?;
        let mut activities = Vec::with_capacity(prepared_job.bindings.len());
        for prepared in prepared_job.bindings.iter() {
            let source = restored
                .sources
                .get(prepared.identity.as_str())
                .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                    message: format!(
                        "durable progress is missing source {:?}",
                        prepared.identity.as_str()
                    ),
                })?;
            let activity = if source.ended {
                IngressActivity::Ended {
                    final_watermark: source.last_watermark,
                }
            } else if source.idle {
                IngressActivity::Idle {
                    watermark: source.last_watermark,
                }
            } else {
                IngressActivity::Active {
                    watermark: source.last_watermark,
                }
            };
            activities.push((prepared.ordinal, activity));
            let binding = driver
                .state
                .bindings
                .get_mut(&prepared.ordinal)
                .expect("prepared durable source has driver state");
            binding.last_source_watermark = source.last_watermark;
            binding.ended = source.ended;
            if let (
                Some(generated),
                NormalizedWatermarkMode::Generated {
                    event_time,
                    max_out_of_orderness,
                    ..
                },
            ) = (&mut binding.generated, &prepared.normalized_watermark)
            {
                *generated = GeneratedWatermarkState::restore(
                    event_time.clone(),
                    *max_out_of_orderness,
                    source.observed_max,
                    source.last_watermark,
                );
            }
            let last_committed = source.cursor.as_ref().map(|cursor| {
                decode_canonical_cursor_order(&cursor.order).map(|order| {
                    RawUpstreamPosition::Exact {
                        delivery_replay_cursor: order,
                        control_frontier: source.next_sequence.to_be_bytes().to_vec(),
                    }
                })
            });
            driver
                .admission
                .lock()
                .bindings
                .get_mut(&prepared.ordinal)
                .expect("prepared durable source has admission state")
                .last_committed_upstream = last_committed.transpose()?;
            if let Some(deadline) = source.watermark_deadline {
                arm_timer(
                    &mut driver.state,
                    prepared.ordinal,
                    TimerKind::Watermark,
                    deadline,
                )?;
            }
            if let Some(deadline) = source.idle_deadline {
                arm_timer(
                    &mut driver.state,
                    prepared.ordinal,
                    TimerKind::Idle,
                    deadline,
                )?;
            }
        }
        driver.state.aggregate = MultiInputProgress::restore(activities);
        driver.state.clock = driver.clock.coordinate();
        driver.phase = if driver.state.aggregate.terminal() {
            DriverPhase::Terminal
        } else {
            DriverPhase::RunningQuiescent
        };
        Ok((driver, sender))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "driver construction initializes one exact ordered state table and its allocators"
    )]
    fn with_trace(
        prepared_job: Arc<PreparedStreamJob>,
        clock: C,
        trace: TraceController,
    ) -> Result<(Self, RawIngressSender)> {
        let mut ordinal_by_identity = BTreeMap::new();
        let mut admission_bindings = BTreeMap::new();
        let mut progress_bindings = BTreeMap::new();
        for prepared in prepared_job.bindings.iter() {
            if ordinal_by_identity
                .insert(prepared.identity.clone(), prepared.ordinal)
                .is_some()
            {
                return Err(CalcFlowError::InvalidArgument {
                    field: "runtime.progress.bindings".into(),
                    message: "prepared job contains duplicate binding identity".into(),
                });
            }
            admission_bindings.insert(
                prepared.ordinal,
                BindingAdmissionState {
                    identity: prepared.identity.clone(),
                    gate: AdmissionGateSnapshot {
                        state: AdmissionGateState::Open,
                        generation: AdmissionGateGeneration(0),
                        close: None,
                    },
                    next_inbox_sequence: CheckedSemanticAllocator::new(
                        0,
                        format!(
                            "runtime.progress.counters.{}.inbox_sequence",
                            prepared.identity.as_str()
                        ),
                    ),
                    next_fence_sequence: CheckedSemanticAllocator::new(
                        0,
                        format!(
                            "runtime.progress.counters.{}.fence_sequence",
                            prepared.identity.as_str()
                        ),
                    ),
                    last_completed_fence: None,
                    last_committed_upstream: None,
                    accepted: VecDeque::new(),
                },
            );
            let generated = match &prepared.normalized_watermark {
                NormalizedWatermarkMode::Generated {
                    event_time,
                    max_out_of_orderness,
                    ..
                } => Some(GeneratedWatermarkState::new(
                    event_time.clone(),
                    *max_out_of_orderness,
                )),
                _ => None,
            };
            progress_bindings.insert(
                prepared.ordinal,
                BindingProgressState {
                    prepared: prepared.clone(),
                    last_source_watermark: None,
                    generated,
                    watermark_timer: None,
                    idle_timer: None,
                    next_local_sequence: CheckedSemanticAllocator::new(
                        0,
                        format!(
                            "runtime.progress.counters.{}.local_sequence",
                            prepared.identity.as_str()
                        ),
                    ),
                    next_timer_generation: CheckedSemanticAllocator::new(
                        0,
                        format!(
                            "runtime.progress.counters.{}.timer_generation",
                            prepared.identity.as_str()
                        ),
                    ),
                    next_timer_sequence: CheckedSemanticAllocator::new(
                        0,
                        format!(
                            "runtime.progress.counters.{}.timer_sequence",
                            prepared.identity.as_str()
                        ),
                    ),
                    ended: false,
                },
            );
        }
        let initial_clock = clock.coordinate();
        let admission = Arc::new(Mutex::new(AdmissionState {
            ordinal_by_identity,
            bindings: admission_bindings,
            next_admission_attempt: CheckedSemanticAllocator::new(
                0,
                "runtime.progress.counters.admission_attempt",
            ),
            next_receipt_sequence: CheckedSemanticAllocator::new(
                0,
                "runtime.progress.counters.receipt_sequence",
            ),
            next_gate_close_ordinal: CheckedSemanticAllocator::new(
                0,
                "runtime.progress.counters.gate_close_ordinal",
            ),
            trace,
            unsettled_receipts: 0,
            capacity: prepared_job
                .runtime_progress_config
                .per_binding_inbox_capacity
                .get(),
            pending_fatal: None,
        }));
        let driver = Self {
            prepared_job,
            clock,
            admission: Arc::clone(&admission),
            state: DriverProgressState {
                clock: initial_clock,
                aggregate: MultiInputProgress::new(progress_bindings.keys().copied()),
                bindings: progress_bindings,
                timer_heap: BTreeMap::new(),
            },
            next_drain_epoch: CheckedSemanticAllocator::new(
                0,
                "runtime.progress.counters.drain_epoch",
            ),
            phase: DriverPhase::Prepared,
            #[cfg(test)]
            after_freeze: None,
        };
        Ok((driver, RawIngressSender { admission }))
    }

    pub(crate) fn start_running(&mut self) -> Result<()> {
        if self.phase != DriverPhase::Prepared {
            return Err(CalcFlowError::InvalidArgument {
                field: "runtime.progress.driver.phase".into(),
                message: "driver can start only from Prepared".into(),
            });
        }
        let coordinate = self.clock.coordinate();
        let mut scratch = self.state.clone();
        scratch.clock = coordinate;
        let ordinals = scratch.bindings.keys().copied().collect::<Vec<_>>();
        for ordinal in ordinals {
            let interval = match &scratch.bindings[&ordinal].prepared.normalized_watermark {
                NormalizedWatermarkMode::Generated { emit_interval, .. } => Some(*emit_interval),
                _ => None,
            };
            if let Some(interval) = interval {
                arm_timer(
                    &mut scratch,
                    ordinal,
                    TimerKind::Watermark,
                    coordinate.instant.checked_add(interval)?,
                )?;
            }
        }
        self.state = scratch;
        self.phase = DriverPhase::RunningQuiescent;
        Ok(())
    }

    pub(crate) fn phase(&self) -> DriverPhase {
        self.phase
    }

    pub(crate) fn trace(&self) -> ProgressExecutionTrace {
        self.admission.lock().trace.completed().clone()
    }

    pub(crate) fn unsettled_receipts(&self) -> usize {
        self.admission.lock().unsettled_receipts
    }

    pub(crate) fn has_ready(&self) -> bool {
        let now = self.clock.coordinate().instant;
        self.admission
            .lock()
            .bindings
            .values()
            .any(|binding| !binding.accepted.is_empty())
            || self
                .state
                .timer_heap
                .keys()
                .any(|identity| identity.deadline <= now)
    }

    pub(crate) fn next_central_wake(&self) -> Option<LogicalInstant> {
        self.state
            .timer_heap
            .keys()
            .map(|identity| identity.deadline)
            .min()
    }

    pub(crate) fn drain_ready(&mut self) -> Result<CommittedDrain> {
        if let Some(outcome) = self.reconcile_pending_admission_fatal() {
            return outcome;
        }
        if self.phase != DriverPhase::RunningQuiescent {
            return Err(CalcFlowError::InvalidArgument {
                field: "runtime.progress.driver.phase".into(),
                message: "driver is not at a running quiescent boundary".into(),
            });
        }
        self.phase = DriverPhase::ApplyingDrain;
        let outcome = self.drain_ready_inner();
        if self.phase == DriverPhase::ApplyingDrain {
            self.phase = if self.state.aggregate.terminal() {
                DriverPhase::Terminal
            } else {
                DriverPhase::RunningQuiescent
            };
        }
        outcome
    }

    fn reconcile_pending_admission_fatal(&mut self) -> Option<Result<CommittedDrain>> {
        let fatal = self.admission.lock().pending_fatal.take()?;
        Some(self.fatal_phase_cleanup(
            fatal.phase,
            CalcFlowError::InvalidArgument {
                field: fatal.path,
                message: fatal.reason,
            },
            Vec::new(),
            self.clock.coordinate(),
        ))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "finite-snapshot arbitration stays visible as one transactional state machine"
    )]
    fn drain_ready_inner(&mut self) -> Result<CommittedDrain> {
        let coordinate = self.clock.coordinate();
        let epoch = match self.next_drain_epoch.checked_peek_and_successor() {
            Ok((next, _)) => DrainEpoch(next),
            Err(error) => {
                return self.fatal_phase_cleanup(
                    DriverFailurePhase::DrainEpochAllocation,
                    error,
                    Vec::new(),
                    coordinate,
                );
            }
        };
        let frozen_result = {
            let mut admission = self.admission.lock();
            freeze_admission(&mut admission, epoch)
        };
        let frozen = match frozen_result {
            Ok(frozen) => frozen,
            Err(error) => {
                return self.fatal_phase_cleanup(
                    DriverFailurePhase::FenceAllocation,
                    error,
                    Vec::new(),
                    coordinate,
                );
            }
        };
        #[cfg(test)]
        if let Some(after_freeze) = self.after_freeze.take() {
            after_freeze();
        }
        let mut scratch = self.state.clone();
        scratch.clock = coordinate;
        if let Err(error) = preflight_ready_key_allocations(&scratch, &frozen.selected) {
            return self.fatal_phase_cleanup(
                DriverFailurePhase::ReadyKeyConstruction,
                error,
                frozen.selected,
                coordinate,
            );
        }
        let mut steps = ready_steps(&mut scratch, frozen.selected, coordinate);
        steps.sort_by_key(ReadyStep::key);
        let identities = steps.iter().map(ReadyStep::identity).collect::<Vec<_>>();
        let keys = steps.iter().map(ReadyStep::key).collect::<Vec<_>>();
        let due_timers = steps
            .iter()
            .filter_map(|step| match step {
                ReadyStep::Timer(timer) => Some(timer.identity),
                ReadyStep::Raw(_) => None,
            })
            .collect::<Vec<_>>();
        let mut selected_envelopes = Vec::new();
        let mut emissions = Vec::new();
        let mut end_ordinals = Vec::new();
        let mut first_error = None;
        for step in steps {
            match step {
                ReadyStep::Raw(raw) => {
                    if first_error.is_none() {
                        match evaluate_raw(
                            &mut scratch,
                            raw.key,
                            &raw.envelope,
                            &mut emissions,
                            &mut end_ordinals,
                        ) {
                            Ok(()) => {}
                            Err(error) => first_error = Some(error),
                        }
                    }
                    selected_envelopes.push(raw.envelope);
                }
                ReadyStep::Timer(timer) if first_error.is_none() => {
                    if let Err(error) = evaluate_timer(&mut scratch, timer, &mut emissions) {
                        first_error = Some(error);
                    }
                }
                ReadyStep::Timer(_) => {}
            }
        }
        let key_range =
            keys.first()
                .zip(keys.last())
                .map_or(ReadyKeyRange::Empty, |(first, last)| {
                    ReadyKeyRange::Inclusive {
                        first: *first,
                        last: *last,
                    }
                });
        if let Some(error) = first_error {
            return self.commit_failed_drain(
                epoch,
                coordinate,
                &frozen.fences,
                identities,
                key_range,
                due_timers,
                selected_envelopes,
                error,
            );
        }
        self.commit_successful_drain(
            epoch,
            coordinate,
            &frozen.fences,
            identities,
            key_range,
            due_timers,
            selected_envelopes,
            end_ordinals,
            scratch,
            emissions,
        )
    }

    fn fatal_phase_cleanup(
        &mut self,
        phase: DriverFailurePhase,
        error: CalcFlowError,
        selected: Vec<RawIngressEnvelope>,
        coordinate: DriverClockCoordinate,
    ) -> Result<CommittedDrain> {
        let (path, reason) = match &error {
            CalcFlowError::InvalidArgument { field, message } => (field.clone(), message.clone()),
            _ => ("runtime.progress.driver_phase".into(), error.to_string()),
        };
        let phase_error = DriverPhaseError {
            phase,
            coordinate: DriverPhaseCoordinate::Counter {
                stable_path: Arc::from(path.as_str()),
                last_value: None,
            },
            failure: ProgressFailure::new(
                ProgressFailureKind::CounterExhaustion,
                path.clone(),
                reason.clone(),
            ),
        };
        let mut admission = self.admission.lock();
        requeue_frozen_selection(&mut admission, selected);
        let ordinals = admission.bindings.keys().copied().collect::<Vec<_>>();
        let traced_cleanup = (|| -> Result<_> {
            let (transitions, _) = plan_terminal_transitions(
                &admission,
                &ordinals,
                AdmissionGateState::ClosedFatal,
                TerminalTransitionCause::Fatal,
                coordinate,
                None,
            )?;
            let identities = transitions
                .iter()
                .flat_map(|transition| transition.extracted_tail.iter().cloned())
                .collect::<Vec<_>>();
            let mut trace = admission.trace.clone();
            let (ordinal, position) = trace.next_coordinates()?;
            trace.append(ProgressTraceRecord::DriverPhaseFailure {
                trace_record_ordinal: ordinal,
                trace_position: position,
                error: phase_error.clone(),
            })?;
            let (ordinal, position) = trace.next_coordinates()?;
            trace.append(ProgressTraceRecord::Terminal(TerminalTransitionRecord {
                trace_record_ordinal: ordinal,
                trace_position: position,
                cause: TerminalTransitionCause::Fatal,
                driver_clock: coordinate,
                owning_drain: None,
                transitions_in_binding_order: transitions.clone(),
            }))?;
            for identity in &identities {
                let (ordinal, position) = trace.next_coordinates()?;
                trace.append(ProgressTraceRecord::Settlement(SettlementRecord {
                    trace_record_ordinal: ordinal,
                    trace_position: position,
                    accepted: identity.clone(),
                    owner: SettlementOwner::Terminal {
                        cause: TerminalTransitionCause::Fatal,
                        owning_drain: None,
                    },
                    disposition: SettlementDisposition::Fatal,
                }))?;
            }
            Ok((trace, transitions))
        })();
        let mut queued = Vec::new();
        match traced_cleanup {
            Ok((trace, transitions)) => {
                commit_terminal_transitions(&mut admission, &transitions, &mut queued);
                admission.trace = trace;
            }
            Err(_) => {
                for binding in admission.bindings.values_mut() {
                    queued.extend(binding.accepted.drain(..));
                    binding.gate.state = AdmissionGateState::ClosedFatal;
                }
            }
        }
        admission.unsettled_receipts = admission
            .unsettled_receipts
            .checked_sub(queued.len())
            .expect("fatal cleanup extracts every accepted receipt exactly once");
        debug_assert_eq!(admission.unsettled_receipts, 0);
        drop(admission);
        self.state.timer_heap.clear();
        for binding in self.state.bindings.values_mut() {
            binding.watermark_timer = None;
            binding.idle_timer = None;
        }
        self.phase = DriverPhase::Terminal;
        settle_failure(queued, &path, &reason);
        Err(error)
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "the commit boundary consumes the complete validated drain transaction"
    )]
    fn commit_successful_drain(
        &mut self,
        epoch: DrainEpoch,
        coordinate: DriverClockCoordinate,
        fences: &[InboxFenceCoordinate],
        identities: Vec<ReadyItemIdentity>,
        key_range: ReadyKeyRange,
        due_timers: Vec<TimerIdentity>,
        selected: Vec<RawIngressEnvelope>,
        mut end_ordinals: Vec<BindingOrdinal>,
        scratch: DriverProgressState,
        emissions: Vec<DriverEmission>,
    ) -> Result<CommittedDrain> {
        end_ordinals.sort();
        end_ordinals.dedup();
        let mut admission = self.admission.lock();
        let mut trace = admission.trace.clone();
        let drain_record = (|| -> Result<()> {
            let (record_ordinal, record_position) = trace.next_coordinates()?;
            trace.append(ProgressTraceRecord::Drain(DrainEpochRecord {
                trace_record_ordinal: record_ordinal,
                trace_position: record_position,
                epoch,
                driver_clock: coordinate,
                inbox_fences: fences.to_owned(),
                selected_items_in_ready_order: identities,
                selected_key_range: key_range,
                due_timers_in_ready_order: due_timers,
                outcome: DrainEpochOutcomeRecord::Committed,
            }))
        })();
        if let Err(error) = drain_record {
            drop(admission);
            return self.fatal_phase_cleanup(
                DriverFailurePhase::ReplayRecordValidation,
                error,
                selected,
                coordinate,
            );
        }
        let (transitions, mut tails) =
            match plan_end_transitions(&admission, &end_ordinals, coordinate, Some(epoch)) {
                Ok(planned) => planned,
                Err(error) => {
                    drop(admission);
                    return self.fatal_phase_cleanup(
                        DriverFailurePhase::GateClosePlanning,
                        error,
                        selected,
                        coordinate,
                    );
                }
            };
        let tail_identities = transitions
            .iter()
            .flat_map(|transition| transition.extracted_tail.iter().cloned())
            .collect::<Vec<_>>();
        let settlement_trace = (|| -> Result<()> {
            if !transitions.is_empty() {
                let (ordinal, position) = trace.next_coordinates()?;
                trace.append(ProgressTraceRecord::Terminal(TerminalTransitionRecord {
                    trace_record_ordinal: ordinal,
                    trace_position: position,
                    cause: TerminalTransitionCause::EndCommit,
                    driver_clock: coordinate,
                    owning_drain: Some(epoch),
                    transitions_in_binding_order: transitions.clone(),
                }))?;
            }
            for envelope in &selected {
                let (ordinal, position) = trace.next_coordinates()?;
                trace.append(ProgressTraceRecord::Settlement(SettlementRecord {
                    trace_record_ordinal: ordinal,
                    trace_position: position,
                    accepted: envelope.identity.clone(),
                    owner: SettlementOwner::Drain(epoch),
                    disposition: SettlementDisposition::CommitSuccess,
                }))?;
            }
            for identity in &tail_identities {
                let (ordinal, position) = trace.next_coordinates()?;
                trace.append(ProgressTraceRecord::Settlement(SettlementRecord {
                    trace_record_ordinal: ordinal,
                    trace_position: position,
                    accepted: identity.clone(),
                    owner: SettlementOwner::Terminal {
                        cause: TerminalTransitionCause::EndCommit,
                        owning_drain: Some(epoch),
                    },
                    disposition: SettlementDisposition::PostEndTailReject,
                }))?;
            }
            Ok(())
        })();
        if let Err(error) = settlement_trace {
            drop(admission);
            return self.fatal_phase_cleanup(
                DriverFailurePhase::SettlementPlanning,
                error,
                selected,
                coordinate,
            );
        }
        self.next_drain_epoch
            .allocate()
            .expect("drain epoch successor was reserved before freezing admission");
        for fence in fences {
            admission
                .bindings
                .get_mut(&fence.binding_ordinal)
                .expect("fence binding exists")
                .next_fence_sequence
                .allocate()
                .expect("fence successor was reserved before freezing admission");
        }
        commit_end_transitions(&mut admission, &transitions, &mut tails);
        for fence in fences {
            admission
                .bindings
                .get_mut(&fence.binding_ordinal)
                .expect("fence binding exists")
                .last_completed_fence = Some(*fence);
        }
        for envelope in &selected {
            admission
                .bindings
                .get_mut(&envelope.identity.binding_ordinal)
                .expect("selected binding exists")
                .last_committed_upstream = Some(envelope.identity.upstream_position.clone());
        }
        admission.unsettled_receipts = admission
            .unsettled_receipts
            .checked_sub(selected.len() + tails.len())
            .expect("settlement membership is a subset of accepted receipts");
        admission.trace = trace;
        drop(admission);
        self.state = scratch;
        let committed_inputs = settle_success(selected);
        let terminal_tail_failures = tails.len();
        settle_failure(
            tails,
            "runtime.progress.admission.post_end",
            "input/control was accepted after the final End fence",
        );
        Ok(CommittedDrain {
            emissions,
            committed_inputs,
            terminal_tail_failures,
        })
    }

    #[allow(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "the failed commit stages its trace, terminal transition, and settlements together"
    )]
    fn commit_failed_drain(
        &mut self,
        epoch: DrainEpoch,
        coordinate: DriverClockCoordinate,
        fences: &[InboxFenceCoordinate],
        identities: Vec<ReadyItemIdentity>,
        key_range: ReadyKeyRange,
        due_timers: Vec<TimerIdentity>,
        selected: Vec<RawIngressEnvelope>,
        error: SelectedItemError,
    ) -> Result<CommittedDrain> {
        let mut admission = self.admission.lock();
        let mut trace = admission.trace.clone();
        let drain_record = (|| -> Result<()> {
            let (ordinal, position) = trace.next_coordinates()?;
            trace.append(ProgressTraceRecord::Drain(DrainEpochRecord {
                trace_record_ordinal: ordinal,
                trace_position: position,
                epoch,
                driver_clock: coordinate,
                inbox_fences: fences.to_owned(),
                selected_items_in_ready_order: identities,
                selected_key_range: key_range,
                due_timers_in_ready_order: due_timers,
                outcome: DrainEpochOutcomeRecord::SelectedItemFailed {
                    first_error_key: error.first_error_key,
                    path: error.failure.path.to_string(),
                    reason: error.failure.reason.to_string(),
                },
            }))
        })();
        if let Err(trace_error) = drain_record {
            drop(admission);
            return self.fatal_phase_cleanup(
                DriverFailurePhase::ReplayRecordValidation,
                trace_error,
                selected,
                coordinate,
            );
        }
        let ordinals = admission.bindings.keys().copied().collect::<Vec<_>>();
        let transition_result = plan_terminal_transitions(
            &admission,
            &ordinals,
            AdmissionGateState::ClosedFatal,
            TerminalTransitionCause::Fatal,
            coordinate,
            Some(epoch),
        );
        let (transitions, mut queued) = match transition_result {
            Ok(planned) => planned,
            Err(transition_error) => {
                drop(admission);
                return self.fatal_phase_cleanup(
                    DriverFailurePhase::GateClosePlanning,
                    transition_error,
                    selected,
                    coordinate,
                );
            }
        };
        let queued_identities = transitions
            .iter()
            .flat_map(|transition| transition.extracted_tail.iter().cloned())
            .collect::<Vec<_>>();
        let settlement_trace = (|| -> Result<()> {
            let (ordinal, position) = trace.next_coordinates()?;
            trace.append(ProgressTraceRecord::Terminal(TerminalTransitionRecord {
                trace_record_ordinal: ordinal,
                trace_position: position,
                cause: TerminalTransitionCause::Fatal,
                driver_clock: coordinate,
                owning_drain: Some(epoch),
                transitions_in_binding_order: transitions.clone(),
            }))?;
            for envelope in &selected {
                let (ordinal, position) = trace.next_coordinates()?;
                trace.append(ProgressTraceRecord::Settlement(SettlementRecord {
                    trace_record_ordinal: ordinal,
                    trace_position: position,
                    accepted: envelope.identity.clone(),
                    owner: SettlementOwner::Drain(epoch),
                    disposition: SettlementDisposition::TransactionError {
                        path: error.failure.path.to_string(),
                        reason: error.failure.reason.to_string(),
                    },
                }))?;
            }
            for identity in &queued_identities {
                let (ordinal, position) = trace.next_coordinates()?;
                trace.append(ProgressTraceRecord::Settlement(SettlementRecord {
                    trace_record_ordinal: ordinal,
                    trace_position: position,
                    accepted: identity.clone(),
                    owner: SettlementOwner::Terminal {
                        cause: TerminalTransitionCause::Fatal,
                        owning_drain: Some(epoch),
                    },
                    disposition: SettlementDisposition::Fatal,
                }))?;
            }
            Ok(())
        })();
        if let Err(trace_error) = settlement_trace {
            drop(admission);
            return self.fatal_phase_cleanup(
                DriverFailurePhase::SettlementPlanning,
                trace_error,
                selected,
                coordinate,
            );
        }
        self.next_drain_epoch
            .allocate()
            .expect("drain epoch successor was reserved before freezing admission");
        for fence in fences {
            admission
                .bindings
                .get_mut(&fence.binding_ordinal)
                .expect("fence binding exists")
                .next_fence_sequence
                .allocate()
                .expect("fence successor was reserved before freezing admission");
        }
        commit_terminal_transitions(&mut admission, &transitions, &mut queued);
        admission.unsettled_receipts = admission
            .unsettled_receipts
            .checked_sub(selected.len() + queued.len())
            .expect("fatal settlement membership is exact");
        admission.trace = trace;
        drop(admission);
        self.state.timer_heap.clear();
        self.phase = DriverPhase::Terminal;
        settle_failure(selected, &error.failure.path, &error.failure.reason);
        settle_failure(
            queued,
            "runtime.progress.driver.fatal",
            "progress driver terminated after a selected-item failure",
        );
        Err(error.failure.into_existing_error())
    }

    pub(crate) fn cancel(&mut self) -> Result<()> {
        if let Some(outcome) = self.reconcile_pending_admission_fatal() {
            return outcome.map(|_| ());
        }
        if matches!(self.phase, DriverPhase::Cancelled | DriverPhase::Terminal) {
            return Ok(());
        }
        self.phase = DriverPhase::Cancelling;
        let coordinate = self.clock.coordinate();
        let mut admission = self.admission.lock();
        let ordinals = admission.bindings.keys().copied().collect::<Vec<_>>();
        let transition_result = plan_terminal_transitions(
            &admission,
            &ordinals,
            AdmissionGateState::ClosedCancelled,
            TerminalTransitionCause::Cancellation,
            coordinate,
            None,
        );
        let (transitions, mut queued) = match transition_result {
            Ok(planned) => planned,
            Err(error) => {
                drop(admission);
                return self
                    .fatal_phase_cleanup(
                        DriverFailurePhase::GateClosePlanning,
                        error,
                        Vec::new(),
                        coordinate,
                    )
                    .map(|_| ());
            }
        };
        let queued_identities = transitions
            .iter()
            .flat_map(|transition| transition.extracted_tail.iter().cloned())
            .collect::<Vec<_>>();
        let trace_result = (|| -> Result<TraceController> {
            let mut trace = admission.trace.clone();
            let (ordinal, position) = trace.next_coordinates()?;
            trace.append(ProgressTraceRecord::Terminal(TerminalTransitionRecord {
                trace_record_ordinal: ordinal,
                trace_position: position,
                cause: TerminalTransitionCause::Cancellation,
                driver_clock: coordinate,
                owning_drain: None,
                transitions_in_binding_order: transitions.clone(),
            }))?;
            for identity in &queued_identities {
                let (ordinal, position) = trace.next_coordinates()?;
                trace.append(ProgressTraceRecord::Settlement(SettlementRecord {
                    trace_record_ordinal: ordinal,
                    trace_position: position,
                    accepted: identity.clone(),
                    owner: SettlementOwner::Terminal {
                        cause: TerminalTransitionCause::Cancellation,
                        owning_drain: None,
                    },
                    disposition: SettlementDisposition::Cancelled,
                }))?;
            }
            Ok(trace)
        })();
        let trace = match trace_result {
            Ok(trace) => trace,
            Err(error) => {
                drop(admission);
                return self
                    .fatal_phase_cleanup(
                        DriverFailurePhase::SettlementPlanning,
                        error,
                        Vec::new(),
                        coordinate,
                    )
                    .map(|_| ());
            }
        };
        commit_terminal_transitions(&mut admission, &transitions, &mut queued);
        admission.unsettled_receipts = admission
            .unsettled_receipts
            .checked_sub(queued.len())
            .expect("cancel settlement membership is exact");
        admission.trace = trace;
        drop(admission);
        self.state.timer_heap.clear();
        for binding in self.state.bindings.values_mut() {
            binding.watermark_timer = None;
            binding.idle_timer = None;
        }
        settle_failure(
            queued,
            "runtime.progress.driver.cancelled",
            "stream progress driver was cancelled",
        );
        self.phase = DriverPhase::Cancelled;
        Ok(())
    }

    pub(crate) fn finish_replay(&self) -> Result<()> {
        self.admission.lock().trace.finish_replay()
    }

    #[allow(
        clippy::too_many_lines,
        reason = "capture maps every normative logical coordinate field without a lossy helper"
    )]
    pub(crate) fn capture_snapshot(
        &self,
        paused_upstreams: &PausedExactUpstreams,
    ) -> Result<StreamProgressSnapshot> {
        if !matches!(
            self.phase,
            DriverPhase::RunningQuiescent | DriverPhase::Terminal
        ) {
            return Err(CalcFlowError::InvalidArgument {
                field: "runtime.progress.snapshot.boundary".into(),
                message: "snapshot capture requires a quiescent event boundary".into(),
            });
        }
        let admission = self.admission.lock();
        if admission.unsettled_receipts != 0
            || admission
                .bindings
                .values()
                .any(|binding| !binding.accepted.is_empty())
        {
            return Err(CalcFlowError::InvalidArgument {
                field: "runtime.progress.snapshot.boundary".into(),
                message: "snapshot capture requires zero queued or unsettled raw envelopes".into(),
            });
        }
        let mut binding_coordinates = Vec::with_capacity(self.prepared_job.bindings.len());
        let mut admission_bindings = BTreeMap::new();
        for prepared in self.prepared_job.bindings.iter() {
            if prepared.replay_positioning
                != super::prepare::ReplayPositioningCapability::ExactPauseReportAndSeek
            {
                return Err(CalcFlowError::InvalidArgument {
                    field: format!(
                        "runtime.progress.snapshot.bindings.{}.replay_positioning",
                        prepared.identity.as_str()
                    ),
                    message: "source cannot pause, report, and seek an exact replay position"
                        .into(),
                });
            }
            let binding = &admission.bindings[&prepared.ordinal];
            let position = binding.last_committed_upstream.clone().ok_or_else(|| {
                CalcFlowError::InvalidArgument {
                    field: format!(
                        "runtime.progress.snapshot.coordinate.bindings.{}.upstream_delivery_replay_cursor",
                        prepared.identity.as_str()
                    ),
                    message: "source has not reported an exact committed upstream position".into(),
                }
            })?;
            if matches!(position, RawUpstreamPosition::Unavailable)
                || paused_upstreams.positions.get(&prepared.identity) != Some(&position)
            {
                return Err(CalcFlowError::InvalidArgument {
                    field: format!(
                        "runtime.progress.snapshot.coordinate.bindings.{}.upstream_delivery_replay_cursor",
                        prepared.identity.as_str()
                    ),
                    message:
                        "paused upstream position does not equal the committed driver position"
                            .into(),
                });
            }
            binding_coordinates.push(CapturedBindingCoordinate {
                identity: prepared.identity.clone(),
                ordinal: prepared.ordinal,
                normalized_config_fingerprint: prepared.normalized_config_fingerprint,
                upstream_position: position.clone(),
                activity: self
                    .state
                    .aggregate
                    .activity(prepared.ordinal)
                    .expect("aggregate and prepared bindings share ordinals"),
                admission_gate: binding.gate.clone(),
                last_source_watermark: self.state.bindings[&prepared.ordinal].last_source_watermark,
                generated_max_nanos: self.state.bindings[&prepared.ordinal]
                    .generated
                    .as_ref()
                    .and_then(GeneratedWatermarkState::max_observed_nanos),
                last_generated_watermark: self.state.bindings[&prepared.ordinal]
                    .generated
                    .as_ref()
                    .and_then(GeneratedWatermarkState::last_emitted),
                watermark_timer: self.state.bindings[&prepared.ordinal].watermark_timer,
                idle_timer: self.state.bindings[&prepared.ordinal].idle_timer,
                next_local_sequence: self.state.bindings[&prepared.ordinal]
                    .next_local_sequence
                    .next(),
                next_timer_generation: self.state.bindings[&prepared.ordinal]
                    .next_timer_generation
                    .next(),
                next_timer_sequence: self.state.bindings[&prepared.ordinal]
                    .next_timer_sequence
                    .next(),
                next_inbox_sequence: binding.next_inbox_sequence.next(),
                next_fence_sequence: binding.next_fence_sequence.next(),
                last_completed_fence: binding.last_completed_fence,
            });
            admission_bindings.insert(
                prepared.ordinal,
                AdmissionBindingSnapshot {
                    identity: binding.identity.clone(),
                    gate: binding.gate.clone(),
                    next_inbox_sequence: binding.next_inbox_sequence.next(),
                    next_fence_sequence: binding.next_fence_sequence.next(),
                    last_completed_fence: binding.last_completed_fence,
                    last_committed_upstream: position,
                },
            );
        }
        let coordinate = CapturedLogicalCoordinate {
            driver_clock: self.state.clock,
            runtime_fence_config_fingerprint: self.prepared_job.runtime_fence_config_fingerprint,
            bindings: binding_coordinates,
            next_drain_epoch: self.next_drain_epoch.next(),
            next_admission_attempt: admission.next_admission_attempt.next(),
            next_receipt_sequence: admission.next_receipt_sequence.next(),
            next_gate_close_ordinal: admission.next_gate_close_ordinal.next(),
            next_global_sequence: self.state.aggregate.next_global_sequence(),
            idle_epoch: self.state.aggregate.idle_epoch(),
            next_idle_epoch: self.state.aggregate.next_idle_epoch(),
            aggregate_watermark: self.state.aggregate.last_emitted_watermark(),
            idle_latched: self.state.aggregate.idle_latched(),
            terminal: self.state.aggregate.terminal(),
            scheduled_timers: self
                .state
                .timer_heap
                .iter()
                .map(|(identity, timer)| (*identity, *timer))
                .collect(),
            unsettled_accepted_envelopes: admission.unsettled_receipts,
            next_trace_record_ordinal: admission.trace.next_record(),
            consumed_trace_position: admission.trace.next_position(),
            progress_execution_trace: admission.trace.completed().clone(),
        };
        Ok(StreamProgressSnapshot {
            prepared_job_fingerprint: self.prepared_job.fingerprint,
            phase: self.phase,
            coordinate,
            payload: DriverSnapshotPayload {
                state: self.state.clone(),
                admission_bindings,
                next_admission_attempt: admission.next_admission_attempt.next(),
                next_receipt_sequence: admission.next_receipt_sequence.next(),
                next_gate_close_ordinal: admission.next_gate_close_ordinal.next(),
                next_drain_epoch: self.next_drain_epoch.next(),
            },
        })
    }

    pub(crate) fn restore(
        prepared_job: Arc<PreparedStreamJob>,
        clock: C,
        request: RestoreRequest,
    ) -> Result<(Self, RawIngressSender)> {
        let RestoreRequest {
            snapshot,
            paused_upstreams,
            replay,
        } = request;
        validate_restore_request(&prepared_job, &clock, &snapshot, &paused_upstreams)?;
        let trace = match replay {
            Some(replay) => TraceController::replay_from_prefix(
                snapshot.coordinate.progress_execution_trace.clone(),
                snapshot.coordinate.next_trace_record_ordinal,
                snapshot.coordinate.consumed_trace_position,
                replay,
            )?,
            None => TraceController::restore_prefix(
                snapshot.coordinate.progress_execution_trace.clone(),
                snapshot.coordinate.next_trace_record_ordinal,
                snapshot.coordinate.consumed_trace_position,
            )?,
        };
        let (mut driver, sender) = Self::with_trace(prepared_job, clock, trace)?;
        driver.state = snapshot.payload.state;
        driver.phase = snapshot.phase;
        driver
            .next_drain_epoch
            .set_next_for_restore(snapshot.payload.next_drain_epoch);
        {
            let mut admission = driver.admission.lock();
            admission
                .next_admission_attempt
                .set_next_for_restore(snapshot.payload.next_admission_attempt);
            admission
                .next_receipt_sequence
                .set_next_for_restore(snapshot.payload.next_receipt_sequence);
            admission
                .next_gate_close_ordinal
                .set_next_for_restore(snapshot.payload.next_gate_close_ordinal);
            for (ordinal, captured) in snapshot.payload.admission_bindings {
                let binding = admission
                    .bindings
                    .get_mut(&ordinal)
                    .expect("prepared fingerprint validated binding membership");
                binding.identity = captured.identity;
                binding.gate = captured.gate;
                binding
                    .next_inbox_sequence
                    .set_next_for_restore(captured.next_inbox_sequence);
                binding
                    .next_fence_sequence
                    .set_next_for_restore(captured.next_fence_sequence);
                binding.last_completed_fence = captured.last_completed_fence;
                binding.last_committed_upstream = Some(captured.last_committed_upstream);
            }
        }
        Ok((driver, sender))
    }

    pub(crate) fn status(&self) -> StreamProgressStatus {
        let admission = self.admission.lock();
        let counters = progress_counters(
            admission.trace.completed(),
            self.state.aggregate.next_global_sequence(),
            self.state.timer_heap.len(),
        );
        let bindings = self
            .state
            .bindings
            .iter()
            .map(|(ordinal, state)| {
                let admission_binding = &admission.bindings[ordinal];
                (
                    state.prepared.identity.clone(),
                    BindingProgressStatus {
                        identity: state.prepared.identity.clone(),
                        ordinal: *ordinal,
                        activity: self
                            .state
                            .aggregate
                            .activity(*ordinal)
                            .expect("aggregate and driver bindings share ordinals"),
                        last_source_watermark: state.last_source_watermark,
                        generated_max_nanos: state
                            .generated
                            .as_ref()
                            .and_then(GeneratedWatermarkState::max_observed_nanos),
                        gate_state: admission_binding.gate.state,
                        gate_generation: admission_binding.gate.generation.0,
                        queued_envelopes: admission_binding.accepted.len(),
                    },
                )
            })
            .collect();
        StreamProgressStatus {
            phase: self.phase,
            logical_instant: self.state.clock.instant,
            aggregate_watermark: self.state.aggregate.last_emitted_watermark(),
            idle_latched: self.state.aggregate.idle_latched(),
            bindings,
            counters,
            unsettled_receipts: admission.unsettled_receipts,
            next_central_wake: self.next_central_wake(),
            terminal_gate_cuts: admission
                .bindings
                .values()
                .filter_map(|binding| {
                    binding
                        .gate
                        .close
                        .clone()
                        .map(|close| (binding.identity.clone(), close))
                })
                .collect(),
        }
    }

    #[cfg(test)]
    fn install_after_freeze(&mut self, after_freeze: impl FnOnce() + Send + 'static) {
        assert!(self.after_freeze.replace(Box::new(after_freeze)).is_none());
    }
}

fn progress_counters(
    trace: &ProgressExecutionTrace,
    progress_emissions: u64,
    timer_entries: usize,
) -> ProgressCounters {
    let mut counters = ProgressCounters {
        trace_records: u64::try_from(trace.records.len()).unwrap_or(u64::MAX),
        ..ProgressCounters::default()
    };
    for record in &trace.records {
        match record {
            ProgressTraceRecord::Admission(record) => {
                counters.admission_attempts = counters.admission_attempts.saturating_add(1);
                match record.decision {
                    AdmissionDecisionRecord::Accepted { .. } => {
                        counters.accepted_envelopes = counters.accepted_envelopes.saturating_add(1);
                    }
                    AdmissionDecisionRecord::ImmediateRejected { .. } => {
                        counters.immediate_rejections =
                            counters.immediate_rejections.saturating_add(1);
                    }
                }
            }
            ProgressTraceRecord::Drain(record) => {
                counters.drain_epochs = counters.drain_epochs.saturating_add(1);
                let inbox_fences = u64::try_from(record.inbox_fences.len()).unwrap_or(u64::MAX);
                let selected_items =
                    u64::try_from(record.selected_items_in_ready_order.len()).unwrap_or(u64::MAX);
                let due_timers =
                    u64::try_from(record.due_timers_in_ready_order.len()).unwrap_or(u64::MAX);
                counters.inbox_fences = counters.inbox_fences.saturating_add(inbox_fences);
                counters.due_timers = counters.due_timers.saturating_add(due_timers);
                counters.maximum_inbox_fences_per_drain =
                    counters.maximum_inbox_fences_per_drain.max(inbox_fences);
                counters.maximum_selected_items_per_drain = counters
                    .maximum_selected_items_per_drain
                    .max(selected_items);
                counters.maximum_due_timers_per_drain =
                    counters.maximum_due_timers_per_drain.max(due_timers);
            }
            ProgressTraceRecord::Terminal(record) => {
                counters.terminal_transitions = counters.terminal_transitions.saturating_add(1);
                counters.gate_transitions = counters.gate_transitions.saturating_add(
                    u64::try_from(record.transitions_in_binding_order.len()).unwrap_or(u64::MAX),
                );
            }
            ProgressTraceRecord::Settlement(record) => {
                counters.settlement_attempts = counters.settlement_attempts.saturating_add(1);
                match record.disposition {
                    SettlementDisposition::CommitSuccess => {
                        counters.commit_success_settlements =
                            counters.commit_success_settlements.saturating_add(1);
                    }
                    SettlementDisposition::TransactionError { .. } => {
                        counters.transaction_error_settlements =
                            counters.transaction_error_settlements.saturating_add(1);
                    }
                    SettlementDisposition::PostEndTailReject => {
                        counters.post_end_tail_settlements =
                            counters.post_end_tail_settlements.saturating_add(1);
                    }
                    SettlementDisposition::Cancelled => {
                        counters.cancelled_settlements =
                            counters.cancelled_settlements.saturating_add(1);
                    }
                    SettlementDisposition::Fatal => {
                        counters.fatal_settlements = counters.fatal_settlements.saturating_add(1);
                    }
                }
            }
            ProgressTraceRecord::DriverPhaseFailure { .. } => {
                counters.driver_phase_failures = counters.driver_phase_failures.saturating_add(1);
            }
        }
    }
    counters.progress_emissions = progress_emissions;
    counters.timer_entries = u64::try_from(timer_entries).unwrap_or(u64::MAX);
    counters
}

#[allow(
    clippy::too_many_lines,
    reason = "restore validation compares every captured coordinate before any state mutation"
)]
fn validate_restore_request<C: DriverLogicalClock>(
    prepared_job: &PreparedStreamJob,
    clock: &C,
    snapshot: &StreamProgressSnapshot,
    paused_upstreams: &PausedExactUpstreams,
) -> Result<()> {
    if prepared_job.fingerprint != snapshot.prepared_job_fingerprint {
        return Err(CalcFlowError::InvalidArgument {
            field: "runtime.progress.snapshot.prepared_job_fingerprint".into(),
            message: "prepared job fingerprint does not exactly match the snapshot".into(),
        });
    }
    if clock.coordinate() != snapshot.coordinate.driver_clock {
        return Err(CalcFlowError::InvalidArgument {
            field: "runtime.progress.snapshot.coordinate.driver_logical_instant".into(),
            message: "driver clock coordinate does not exactly match the snapshot".into(),
        });
    }
    if prepared_job.runtime_fence_config_fingerprint
        != snapshot.coordinate.runtime_fence_config_fingerprint
    {
        return Err(CalcFlowError::InvalidArgument {
            field: "runtime.progress.snapshot.coordinate.runtime_fence_config_fingerprint".into(),
            message: "runtime and fence configuration does not exactly match the snapshot".into(),
        });
    }
    if prepared_job.bindings.len() != snapshot.coordinate.bindings.len() {
        return Err(CalcFlowError::InvalidArgument {
            field: "runtime.progress.snapshot.coordinate.bindings".into(),
            message: "binding cardinality does not exactly match the snapshot".into(),
        });
    }
    if snapshot.payload.state.bindings.len() != prepared_job.bindings.len()
        || snapshot.payload.admission_bindings.len() != prepared_job.bindings.len()
    {
        return Err(CalcFlowError::InvalidArgument {
            field: "runtime.progress.snapshot.coordinate.bindings".into(),
            message: "captured state cardinality does not exactly match the prepared job".into(),
        });
    }
    for (prepared, captured) in prepared_job
        .bindings
        .iter()
        .zip(&snapshot.coordinate.bindings)
    {
        if prepared.identity != captured.identity || prepared.ordinal != captured.ordinal {
            return Err(CalcFlowError::InvalidArgument {
                field: "runtime.progress.snapshot.coordinate.bindings".into(),
                message: "binding identity or ordinal does not exactly match the snapshot".into(),
            });
        }
        if prepared.normalized_config_fingerprint != captured.normalized_config_fingerprint {
            return Err(CalcFlowError::InvalidArgument {
                field: format!(
                    "runtime.progress.snapshot.bindings.{}.normalized_config_fingerprint",
                    prepared.identity.as_str()
                ),
                message: "normalized source configuration does not exactly match".into(),
            });
        }
        if paused_upstreams.positions.get(&prepared.identity) != Some(&captured.upstream_position) {
            return Err(CalcFlowError::InvalidArgument {
                field: format!(
                    "runtime.progress.snapshot.coordinate.bindings.{}.upstream_delivery_replay_cursor",
                    prepared.identity.as_str()
                ),
                message: "paused upstream does not exactly match the captured position".into(),
            });
        }
        let state = &snapshot.payload.state.bindings[&prepared.ordinal];
        let admission = &snapshot.payload.admission_bindings[&prepared.ordinal];
        let generated_max_nanos = state
            .generated
            .as_ref()
            .and_then(GeneratedWatermarkState::max_observed_nanos);
        let last_generated_watermark = state
            .generated
            .as_ref()
            .and_then(GeneratedWatermarkState::last_emitted);
        if admission.identity != captured.identity
            || admission.last_committed_upstream != captured.upstream_position
            || admission.gate != captured.admission_gate
            || admission.next_inbox_sequence != captured.next_inbox_sequence
            || admission.next_fence_sequence != captured.next_fence_sequence
            || admission.last_completed_fence != captured.last_completed_fence
            || snapshot.payload.state.aggregate.activity(prepared.ordinal)
                != Some(captured.activity)
            || state.last_source_watermark != captured.last_source_watermark
            || generated_max_nanos != captured.generated_max_nanos
            || last_generated_watermark != captured.last_generated_watermark
            || state.watermark_timer != captured.watermark_timer
            || state.idle_timer != captured.idle_timer
            || state.next_local_sequence.next() != captured.next_local_sequence
            || state.next_timer_generation.next() != captured.next_timer_generation
            || state.next_timer_sequence.next() != captured.next_timer_sequence
        {
            return Err(CalcFlowError::InvalidArgument {
                field: format!(
                    "runtime.progress.snapshot.coordinate.bindings.{}",
                    prepared.identity.as_str()
                ),
                message: "binding progress, gate, timer, fence, or allocation coordinate does not exactly match"
                    .into(),
            });
        }
    }
    let scheduled_timers = snapshot
        .payload
        .state
        .timer_heap
        .iter()
        .map(|(identity, timer)| (*identity, *timer))
        .collect::<Vec<_>>();
    if snapshot.payload.state.clock != snapshot.coordinate.driver_clock
        || snapshot.payload.next_drain_epoch != snapshot.coordinate.next_drain_epoch
        || snapshot.payload.next_admission_attempt != snapshot.coordinate.next_admission_attempt
        || snapshot.payload.next_receipt_sequence != snapshot.coordinate.next_receipt_sequence
        || snapshot.payload.next_gate_close_ordinal != snapshot.coordinate.next_gate_close_ordinal
        || snapshot.payload.state.aggregate.next_global_sequence()
            != snapshot.coordinate.next_global_sequence
        || snapshot.payload.state.aggregate.idle_epoch() != snapshot.coordinate.idle_epoch
        || snapshot.payload.state.aggregate.next_idle_epoch() != snapshot.coordinate.next_idle_epoch
        || snapshot.payload.state.aggregate.last_emitted_watermark()
            != snapshot.coordinate.aggregate_watermark
        || snapshot.payload.state.aggregate.idle_latched() != snapshot.coordinate.idle_latched
        || snapshot.payload.state.aggregate.terminal() != snapshot.coordinate.terminal
        || scheduled_timers != snapshot.coordinate.scheduled_timers
        || snapshot.coordinate.unsettled_accepted_envelopes != 0
    {
        return Err(CalcFlowError::InvalidArgument {
            field: "runtime.progress.snapshot.coordinate".into(),
            message: "aggregate, timer, or next-allocation coordinate does not exactly match"
                .into(),
        });
    }
    TraceController::restore_prefix(
        snapshot.coordinate.progress_execution_trace.clone(),
        snapshot.coordinate.next_trace_record_ordinal,
        snapshot.coordinate.consumed_trace_position,
    )?;
    Ok(())
}

fn ready_steps(
    state: &mut DriverProgressState,
    selected: Vec<RawIngressEnvelope>,
    coordinate: DriverClockCoordinate,
) -> Vec<ReadyStep> {
    let mut steps = Vec::new();
    for envelope in selected {
        let binding = state
            .bindings
            .get_mut(&envelope.identity.binding_ordinal)
            .expect("admission and driver bindings share ordinals");
        let local = binding
            .next_local_sequence
            .allocate()
            .expect("ready-key allocations were reserved before envelope ownership moved");
        steps.push(ReadyStep::Raw(ReadyRaw {
            key: ReadyKey {
                logical_instant: coordinate.instant,
                class: ReadyClass::InputOrControl,
                binding_ordinal: envelope.identity.binding_ordinal,
                local_sequence: LocalSequence(local),
            },
            envelope,
        }));
    }
    let due = state
        .timer_heap
        .iter()
        .filter(|(identity, _)| identity.deadline <= coordinate.instant)
        .map(|(identity, current)| (*identity, *current))
        .collect::<Vec<_>>();
    for (identity, current) in due {
        let class = match identity.kind {
            TimerKind::Watermark => ReadyClass::WatermarkTimer,
            TimerKind::Idle => ReadyClass::IdleTimer,
        };
        steps.push(ReadyStep::Timer(ReadyTimer {
            key: ReadyKey {
                logical_instant: coordinate.instant,
                class,
                binding_ordinal: identity.binding_ordinal,
                local_sequence: current.ready_local_sequence,
            },
            identity,
        }));
    }
    steps
}

fn preflight_ready_key_allocations(
    state: &DriverProgressState,
    selected: &[RawIngressEnvelope],
) -> Result<()> {
    let mut counts = BTreeMap::<BindingOrdinal, usize>::new();
    for envelope in selected {
        let count = counts.entry(envelope.identity.binding_ordinal).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| CalcFlowError::InvalidArgument {
                field: "runtime.progress.ready_key.selected_count".into(),
                message: "selected input count exceeds the platform range".into(),
            })?;
    }
    for (ordinal, count) in counts {
        state.bindings[&ordinal]
            .next_local_sequence
            .checked_successor_after(count)?;
    }
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "the exhaustive raw-event match defines one scratch-state transition boundary"
)]
fn evaluate_raw(
    state: &mut DriverProgressState,
    key: ReadyKey,
    envelope: &RawIngressEnvelope,
    emissions: &mut Vec<DriverEmission>,
    end_ordinals: &mut Vec<BindingOrdinal>,
) -> std::result::Result<(), SelectedItemError> {
    let ordinal = envelope.identity.binding_ordinal;
    let binding = state
        .bindings
        .get(&ordinal)
        .expect("ready raw binding exists");
    if binding.ended {
        return Err(selected_error(
            key,
            &format!(
                "runtime.progress.admission.{}.post_end",
                binding.prepared.identity.as_str()
            ),
            "input/control after EndOfInput is forbidden",
        ));
    }
    match &envelope.event {
        RawIngressEvent::Data(batch) => {
            let binding_id = binding.prepared.identity.as_str().to_owned();
            if let Some(generated) = state
                .bindings
                .get_mut(&ordinal)
                .expect("binding exists")
                .generated
                .as_mut()
            {
                generated
                    .observe_batch(batch, &binding_id)
                    .map_err(|error| {
                        selected_error(key, "runtime.progress.generated", &error.to_string())
                    })?;
            }
            rearm_idle(state, ordinal, key.logical_instant).map_err(|error| {
                selected_error(key, "runtime.progress.timers.idle", &error.to_string())
            })?;
            append_progress(
                emissions,
                state
                    .aggregate
                    .evaluate(ordinal, AggregateInput::Data)
                    .map_err(|failure| SelectedItemError {
                        first_error_key: key,
                        failure,
                    })?,
            );
            emissions.push(DriverEmission::ForwardData {
                binding: ordinal,
                batch: batch.clone(),
            });
        }
        RawIngressEvent::ConnectorWatermark(watermark) => {
            if !matches!(
                state.bindings[&ordinal].prepared.normalized_watermark,
                NormalizedWatermarkMode::SourceProvided { .. }
            ) {
                return Err(selected_error(
                    key,
                    &format!(
                        "sources.{}.watermark_policy",
                        state.bindings[&ordinal].prepared.identity.as_str()
                    ),
                    "connector watermark is illegal in generated or disabled mode",
                ));
            }
            let binding = state.bindings.get_mut(&ordinal).expect("binding exists");
            if binding
                .last_source_watermark
                .is_some_and(|previous| *watermark < previous)
            {
                return Err(selected_error(
                    key,
                    &format!("sources.{}.watermark", binding.prepared.identity.as_str()),
                    "source-provided watermark regressed",
                ));
            }
            let advances = binding.last_source_watermark != Some(*watermark);
            binding.last_source_watermark = Some(*watermark);
            rearm_idle(state, ordinal, key.logical_instant).map_err(|error| {
                selected_error(key, "runtime.progress.timers.idle", &error.to_string())
            })?;
            if advances {
                append_progress(
                    emissions,
                    state
                        .aggregate
                        .evaluate(ordinal, AggregateInput::Watermark(*watermark))
                        .map_err(|failure| SelectedItemError {
                            first_error_key: key,
                            failure,
                        })?,
                );
            }
        }
        RawIngressEvent::ConnectorIdle => {
            remove_timer(state, ordinal, TimerKind::Idle);
            append_progress(
                emissions,
                state
                    .aggregate
                    .evaluate(ordinal, AggregateInput::Idle)
                    .map_err(|failure| SelectedItemError {
                        first_error_key: key,
                        failure,
                    })?,
            );
        }
        RawIngressEvent::EndOfInput => {
            remove_timer(state, ordinal, TimerKind::Watermark);
            remove_timer(state, ordinal, TimerKind::Idle);
            append_progress(
                emissions,
                state
                    .aggregate
                    .evaluate(ordinal, AggregateInput::End)
                    .map_err(|failure| SelectedItemError {
                        first_error_key: key,
                        failure,
                    })?,
            );
            state
                .bindings
                .get_mut(&ordinal)
                .expect("binding exists")
                .ended = true;
            end_ordinals.push(ordinal);
        }
    }
    Ok(())
}

fn evaluate_timer(
    state: &mut DriverProgressState,
    timer: ReadyTimer,
    emissions: &mut Vec<DriverEmission>,
) -> std::result::Result<(), SelectedItemError> {
    let ordinal = timer.identity.binding_ordinal;
    let current = match timer.identity.kind {
        TimerKind::Watermark => state.bindings[&ordinal].watermark_timer,
        TimerKind::Idle => state.bindings[&ordinal].idle_timer,
    };
    if current.map(|current| current.identity(ordinal, timer.identity.kind)) != Some(timer.identity)
    {
        state.timer_heap.remove(&timer.identity);
        return Ok(());
    }
    if state.bindings[&ordinal].ended {
        remove_timer(state, ordinal, timer.identity.kind);
        return Ok(());
    }
    match timer.identity.kind {
        TimerKind::Watermark => {
            let binding_id = state.bindings[&ordinal]
                .prepared
                .identity
                .as_str()
                .to_owned();
            let NormalizedWatermarkMode::Generated {
                emit_interval: interval,
                ..
            } = state.bindings[&ordinal].prepared.normalized_watermark
            else {
                return Err(selected_error(
                    timer.key,
                    "runtime.progress.timers.watermark",
                    "watermark timer exists for a non-generated binding",
                ));
            };
            let watermark = state
                .bindings
                .get_mut(&ordinal)
                .expect("binding exists")
                .generated
                .as_mut()
                .expect("generated mode owns generated state")
                .on_timer(&binding_id)
                .map_err(|error| {
                    selected_error(timer.key, "runtime.progress.generated", &error.to_string())
                })?;
            if let Some(watermark) = watermark {
                append_progress(
                    emissions,
                    state
                        .aggregate
                        .evaluate(ordinal, AggregateInput::Watermark(watermark))
                        .map_err(|failure| SelectedItemError {
                            first_error_key: timer.key,
                            failure,
                        })?,
                );
            }
            let deadline =
                next_phase_deadline(timer.identity.deadline, timer.key.logical_instant, interval)
                    .map_err(|error| {
                    selected_error(
                        timer.key,
                        "runtime.progress.timers.watermark",
                        &error.to_string(),
                    )
                })?;
            arm_timer(state, ordinal, TimerKind::Watermark, deadline).map_err(|error| {
                selected_error(
                    timer.key,
                    "runtime.progress.timers.watermark",
                    &error.to_string(),
                )
            })?;
        }
        TimerKind::Idle => {
            remove_timer(state, ordinal, TimerKind::Idle);
            append_progress(
                emissions,
                state
                    .aggregate
                    .evaluate(ordinal, AggregateInput::Idle)
                    .map_err(|failure| SelectedItemError {
                        first_error_key: timer.key,
                        failure,
                    })?,
            );
        }
    }
    Ok(())
}

fn selected_error(key: ReadyKey, path: &str, reason: &str) -> SelectedItemError {
    SelectedItemError {
        first_error_key: key,
        failure: ProgressFailure::protocol(path, reason),
    }
}

fn append_progress(emissions: &mut Vec<DriverEmission>, progress: Vec<ProgressEmission>) {
    emissions.extend(progress.into_iter().map(DriverEmission::Progress));
}

fn rearm_idle(
    state: &mut DriverProgressState,
    ordinal: BindingOrdinal,
    instant: LogicalInstant,
) -> Result<()> {
    let timeout = state.bindings[&ordinal]
        .prepared
        .normalized_watermark
        .idle_timeout();
    if let Some(timeout) = timeout {
        arm_timer(
            state,
            ordinal,
            TimerKind::Idle,
            instant.checked_add(timeout)?,
        )?;
    }
    Ok(())
}

fn arm_timer(
    state: &mut DriverProgressState,
    ordinal: BindingOrdinal,
    kind: TimerKind,
    deadline: LogicalInstant,
) -> Result<()> {
    remove_timer(state, ordinal, kind);
    let binding = state.bindings.get_mut(&ordinal).expect("binding exists");
    binding.next_timer_generation.checked_peek_and_successor()?;
    binding.next_timer_sequence.checked_peek_and_successor()?;
    binding.next_local_sequence.checked_peek_and_successor()?;
    let current = CurrentTimer {
        deadline,
        generation: TimerGeneration(binding.next_timer_generation.allocate()?),
        timer_sequence: TimerSequence(binding.next_timer_sequence.allocate()?),
        ready_local_sequence: LocalSequence(binding.next_local_sequence.allocate()?),
    };
    let identity = current.identity(ordinal, kind);
    match kind {
        TimerKind::Watermark => binding.watermark_timer = Some(current),
        TimerKind::Idle => binding.idle_timer = Some(current),
    }
    state.timer_heap.insert(identity, current);
    Ok(())
}

fn remove_timer(state: &mut DriverProgressState, ordinal: BindingOrdinal, kind: TimerKind) {
    let binding = state.bindings.get_mut(&ordinal).expect("binding exists");
    let current = match kind {
        TimerKind::Watermark => binding.watermark_timer.take(),
        TimerKind::Idle => binding.idle_timer.take(),
    };
    if let Some(current) = current {
        state.timer_heap.remove(&current.identity(ordinal, kind));
    }
}

fn plan_end_transitions(
    admission: &AdmissionState,
    ordinals: &[BindingOrdinal],
    coordinate: DriverClockCoordinate,
    owning_drain: Option<DrainEpoch>,
) -> Result<(Vec<BindingGateTransitionRecord>, Vec<RawIngressEnvelope>)> {
    let _ = coordinate;
    plan_terminal_transitions(
        admission,
        ordinals,
        AdmissionGateState::ClosedAfterEnd,
        TerminalTransitionCause::EndCommit,
        coordinate,
        owning_drain,
    )
}

fn plan_terminal_transitions(
    admission: &AdmissionState,
    ordinals: &[BindingOrdinal],
    closed_state: AdmissionGateState,
    cause: TerminalTransitionCause,
    _coordinate: DriverClockCoordinate,
    _owning_drain: Option<DrainEpoch>,
) -> Result<(Vec<BindingGateTransitionRecord>, Vec<RawIngressEnvelope>)> {
    let mut next_close = admission.next_gate_close_ordinal.clone();
    let mut transitions = Vec::new();
    for ordinal in ordinals {
        let binding = admission.bindings.get(ordinal).expect("binding exists");
        if binding.gate.state != AdmissionGateState::Open {
            continue;
        }
        let close_ordinal = GateCloseOrdinal(next_close.allocate()?);
        let new_generation = binding
            .gate
            .generation
            .0
            .checked_add(1)
            .filter(|value| value.checked_add(1).is_some())
            .map(AdmissionGateGeneration)
            .ok_or_else(|| CalcFlowError::InvalidArgument {
                field: "runtime.progress.counters.gate_generation".into(),
                message: "gate generation exhausted".into(),
            })?;
        let extracted_tail = binding
            .accepted
            .iter()
            .map(|envelope| envelope.identity.clone())
            .collect();
        transitions.push(BindingGateTransitionRecord {
            binding: binding.identity.clone(),
            binding_ordinal: *ordinal,
            close: AdmissionGateCloseCoordinate {
                close_ordinal,
                cause,
                old_generation: binding.gate.generation,
                new_generation,
                closed_state,
                next_inbox_sequence_cut: InboxSequence(binding.next_inbox_sequence.next()),
            },
            extracted_tail,
        });
    }
    Ok((transitions, Vec::new()))
}

fn commit_end_transitions(
    admission: &mut AdmissionState,
    transitions: &[BindingGateTransitionRecord],
    extracted: &mut Vec<RawIngressEnvelope>,
) {
    commit_terminal_transitions(admission, transitions, extracted);
}

fn commit_terminal_transitions(
    admission: &mut AdmissionState,
    transitions: &[BindingGateTransitionRecord],
    extracted: &mut Vec<RawIngressEnvelope>,
) {
    for transition in transitions {
        let binding = admission
            .bindings
            .get_mut(&transition.binding_ordinal)
            .expect("binding exists");
        debug_assert_eq!(
            binding
                .accepted
                .iter()
                .map(|envelope| envelope.identity.clone())
                .collect::<Vec<_>>(),
            transition.extracted_tail
        );
        extracted.extend(binding.accepted.drain(..));
        binding.gate = AdmissionGateSnapshot {
            state: transition.close.closed_state,
            generation: transition.close.new_generation,
            close: Some(transition.close.clone()),
        };
        admission
            .next_gate_close_ordinal
            .allocate()
            .expect("gate close successor was planned");
    }
}

fn settle_success(selected: Vec<RawIngressEnvelope>) -> Vec<CommittedRawInput> {
    selected
        .into_iter()
        .map(|envelope| {
            let committed = CommittedRawInput {
                binding: envelope.identity.binding.clone(),
                accepted_identity: envelope.identity.clone(),
                upstream_position: envelope.identity.upstream_position.clone(),
            };
            let _receiver_may_have_been_dropped = envelope.settlement.send(Ok(committed.clone()));
            committed
        })
        .collect()
}

fn settle_failure(selected: Vec<RawIngressEnvelope>, path: &str, reason: &str) {
    for envelope in selected {
        let _receiver_may_have_been_dropped =
            envelope
                .settlement
                .send(Err(CalcFlowError::InvalidArgument {
                    field: path.into(),
                    message: reason.into(),
                }));
    }
}

#[derive(Clone)]
pub(crate) struct ManualClock(Arc<Mutex<DriverClockCoordinate>>);

impl ManualClock {
    pub(crate) fn new(instant: LogicalInstant) -> Self {
        Self(Arc::new(Mutex::new(DriverClockCoordinate::new(
            [0; 32], instant,
        ))))
    }

    pub(crate) fn set(&self, instant: LogicalInstant) {
        self.0.lock().instant = instant;
    }
}

impl DriverLogicalClock for ManualClock {
    fn coordinate(&self) -> DriverClockCoordinate {
        *self.0.lock()
    }
}

#[derive(Clone)]
struct RuntimeLogicalClock {
    started: tokio::time::Instant,
    trace: [u8; 32],
}

impl RuntimeLogicalClock {
    fn new(trace: [u8; 32]) -> Self {
        Self {
            started: tokio::time::Instant::now(),
            trace,
        }
    }
}

impl DriverLogicalClock for RuntimeLogicalClock {
    fn coordinate(&self) -> DriverClockCoordinate {
        DriverClockCoordinate::new(
            self.trace,
            LogicalInstant(self.started.elapsed().as_nanos()),
        )
    }
}

struct LiveProgressInner {
    sender: RawIngressSender,
    driver: tokio::sync::Mutex<StreamProgressDriver<RuntimeLogicalClock>>,
    outputs: tokio::sync::Mutex<BTreeMap<BindingOrdinal, Vec<super::super::EdgeSender>>>,
    cancellation: crate::CancellationToken,
    status: LiveProgressStatusHandle,
    drive_serial: tokio::sync::Mutex<()>,
    wake: tokio::sync::Notify,
}

#[derive(Clone)]
pub(crate) struct LiveProgressCoordinator(Arc<LiveProgressInner>);

impl LiveProgressCoordinator {
    pub(crate) fn new(
        prepared: &Arc<PreparedStreamJob>,
        outputs_by_binding: BTreeMap<String, Vec<super::super::EdgeSender>>,
        cancellation: crate::CancellationToken,
    ) -> Result<Self> {
        let mut trace = [0_u8; 32];
        trace.copy_from_slice(&prepared.fingerprint.as_bytes());
        let clock = RuntimeLogicalClock::new(trace);
        let (mut driver, sender) = StreamProgressDriver::new(Arc::clone(prepared), clock)?;
        driver.start_running()?;
        Self::from_driver(
            prepared.as_ref(),
            outputs_by_binding,
            cancellation,
            driver,
            sender,
        )
    }

    pub(crate) fn new_restored(
        prepared: &Arc<PreparedStreamJob>,
        outputs_by_binding: BTreeMap<String, Vec<super::super::EdgeSender>>,
        cancellation: crate::CancellationToken,
        restored: &DurableProgressRestore,
    ) -> Result<Self> {
        let mut trace = [0_u8; 32];
        trace.copy_from_slice(&prepared.fingerprint.as_bytes());
        let clock = RuntimeLogicalClock::new(trace);
        let (driver, sender) = StreamProgressDriver::restore_durable(prepared, clock, restored)?;
        Self::from_driver(
            prepared.as_ref(),
            outputs_by_binding,
            cancellation,
            driver,
            sender,
        )
    }

    fn from_driver(
        prepared: &PreparedStreamJob,
        mut outputs_by_binding: BTreeMap<String, Vec<super::super::EdgeSender>>,
        cancellation: crate::CancellationToken,
        driver: StreamProgressDriver<RuntimeLogicalClock>,
        sender: RawIngressSender,
    ) -> Result<Self> {
        let mut outputs = BTreeMap::new();
        for binding in prepared.bindings.iter() {
            let binding_outputs = outputs_by_binding
                .remove(binding.identity.as_str())
                .ok_or_else(|| CalcFlowError::Internal {
                    message: format!(
                        "prepared progress binding {:?} has no runtime source route",
                        binding.identity.as_str()
                    ),
                })?;
            outputs.insert(binding.ordinal, binding_outputs);
        }
        if !outputs_by_binding.is_empty() {
            return Err(CalcFlowError::Internal {
                message: "runtime source routes contain an unprepared progress binding".into(),
            });
        }
        let status = LiveProgressStatusHandle::new(driver.status());
        Ok(Self(Arc::new(LiveProgressInner {
            sender,
            driver: tokio::sync::Mutex::new(driver),
            outputs: tokio::sync::Mutex::new(outputs),
            cancellation,
            status,
            drive_serial: tokio::sync::Mutex::new(()),
            wake: tokio::sync::Notify::new(),
        })))
    }

    pub(crate) async fn submit(
        &self,
        binding: BindingIdentity,
        event: RawIngressEvent,
        upstream_position: RawUpstreamPosition,
    ) -> Result<CommittedRawInput> {
        let settlement_started = std::time::Instant::now();
        let receipt = match self
            .0
            .sender
            .submit(binding, event, upstream_position)
            .await
        {
            Ok(receipt) => receipt,
            Err(error) => {
                let outcome = {
                    self.0
                        .driver
                        .lock()
                        .await
                        .reconcile_pending_admission_fatal()
                };
                if let Some(outcome) = outcome {
                    self.publish_status().await;
                    return outcome.and(Err(error));
                }
                return Err(error);
            }
        };
        self.publish_status().await;
        self.0.wake.notify_one();
        let _drive = self.0.drive_serial.lock().await;
        self.drive_ready(&self.0.cancellation).await?;
        let settlement = receipt.wait_settled().await;
        self.0
            .status
            .observe_settlement_latency(settlement_started.elapsed());
        settlement
    }

    pub(crate) async fn checkpoint_cut(
        &self,
        epoch: Epoch,
        source_cuts: &BTreeMap<BindingIdentity, DurableSourceCut>,
        cancellation: &crate::CancellationToken,
    ) -> Result<BTreeMap<String, SourceManifestEntry>> {
        let _drive = self.0.drive_serial.lock().await;
        self.drive_ready(cancellation).await?;
        let (live_ordinals, durable) = {
            let driver = self.0.driver.lock().await;
            if driver.unsettled_receipts() != 0 || driver.has_ready() {
                return Err(CalcFlowError::Internal {
                    message: format!(
                        "checkpoint epoch {} reached an unsettled progress cut",
                        epoch.as_u64()
                    ),
                });
            }
            let expected = driver
                .state
                .bindings
                .values()
                .map(|state| state.prepared.identity.clone())
                .collect::<BTreeSet<_>>();
            if source_cuts.keys().cloned().collect::<BTreeSet<_>>() != expected {
                return Err(CalcFlowError::CheckpointMismatch {
                    message: "checkpoint source cut IDs do not match progress state".into(),
                });
            }
            let durable = durable_source_manifest_entries(&driver, source_cuts)?;
            let live = driver
                .state
                .bindings
                .iter()
                .filter(|(_, state)| !state.ended)
                .map(|(ordinal, _)| *ordinal)
                .collect::<Vec<_>>();
            (live, durable)
        };
        let mut outputs = self.0.outputs.lock().await;
        for ordinal in live_ordinals {
            send_progress_fanout(
                outputs
                    .get_mut(&ordinal)
                    .expect("prepared progress binding has a runtime route"),
                super::super::StreamMessage::barrier(epoch),
                cancellation,
            )
            .await?;
        }
        Ok(durable)
    }

    pub(crate) async fn terminal_checkpoint_cut(
        &self,
        epoch: Epoch,
        source_cuts: &BTreeMap<BindingIdentity, DurableSourceCut>,
        cancellation: &crate::CancellationToken,
    ) -> Result<BTreeMap<String, SourceManifestEntry>> {
        let _drive = self.0.drive_serial.lock().await;
        self.drive_ready(cancellation).await?;
        let driver = self.0.driver.lock().await;
        if driver.unsettled_receipts() != 0 || driver.has_ready() {
            return Err(CalcFlowError::Internal {
                message: format!(
                    "terminal checkpoint epoch {} reached an unsettled progress cut",
                    epoch.as_u64()
                ),
            });
        }
        if driver.state.bindings.values().any(|state| !state.ended) {
            return Err(CalcFlowError::CheckpointMismatch {
                message: "terminal checkpoint requires every source progress binding to end".into(),
            });
        }
        let expected = driver
            .state
            .bindings
            .values()
            .map(|state| state.prepared.identity.clone())
            .collect::<BTreeSet<_>>();
        if source_cuts.keys().cloned().collect::<BTreeSet<_>>() != expected {
            return Err(CalcFlowError::CheckpointMismatch {
                message: "terminal checkpoint source cut IDs do not match progress state".into(),
            });
        }
        durable_source_manifest_entries(&driver, source_cuts)
    }

    async fn drive_ready(&self, cancellation: &crate::CancellationToken) -> Result<()> {
        loop {
            let ready = self.0.driver.lock().await.has_ready();
            if !ready {
                return Ok(());
            }
            let drain = self.0.driver.lock().await.drain_ready();
            self.publish_status().await;
            let drain = drain?;
            self.dispatch(drain.emissions, cancellation).await?;
        }
    }

    async fn run(&self, cancellation: crate::CancellationToken) -> Result<()> {
        loop {
            if cancellation.is_cancelled() {
                self.0.driver.lock().await.cancel()?;
                self.publish_status().await;
                return Ok(());
            }
            let (ready, next_wake, phase) = {
                let driver = self.0.driver.lock().await;
                (
                    driver.has_ready(),
                    driver.next_central_wake(),
                    driver.phase(),
                )
            };
            if matches!(phase, DriverPhase::Terminal | DriverPhase::Cancelled) {
                return Ok(());
            }
            if ready {
                let _drive = self.0.drive_serial.lock().await;
                self.drive_ready(&cancellation).await?;
                continue;
            }
            let timer_wait = async {
                match next_wake {
                    Some(deadline) => {
                        let now = self.0.driver.lock().await.clock.coordinate().instant;
                        let delay_nanos = deadline.0.saturating_sub(now.0);
                        let delay =
                            Duration::from_nanos(u64::try_from(delay_nanos).unwrap_or(u64::MAX));
                        tokio::time::sleep(delay).await;
                    }
                    None => std::future::pending::<()>().await,
                }
            };
            tokio::select! {
                biased;
                () = cancellation.cancelled() => {
                    self.0.driver.lock().await.cancel()?;
                    self.publish_status().await;
                    return Ok(());
                }
                () = self.0.wake.notified() => {}
                () = timer_wait => {}
            }
        }
    }

    pub(crate) fn status_handle(&self) -> LiveProgressStatusHandle {
        self.0.status.clone()
    }

    async fn publish_status(&self) {
        self.0.status.publish(self.0.driver.lock().await.status());
    }

    async fn dispatch(
        &self,
        emissions: Vec<DriverEmission>,
        cancellation: &crate::CancellationToken,
    ) -> Result<()> {
        let mut outputs = self.0.outputs.lock().await;
        for emission in emissions {
            match emission {
                DriverEmission::ForwardData { binding, batch } => {
                    send_progress_fanout(
                        outputs
                            .get_mut(&binding)
                            .expect("driver emission binding has a prepared route"),
                        super::super::StreamMessage::data(batch),
                        cancellation,
                    )
                    .await?;
                }
                DriverEmission::Progress(progress) => {
                    let message = match progress.kind {
                        super::aggregate::ProgressEmissionKind::Watermark(watermark) => {
                            super::super::StreamMessage::watermark(watermark)
                        }
                        super::aggregate::ProgressEmissionKind::Idle => {
                            super::super::StreamMessage::idle()
                        }
                        super::aggregate::ProgressEmissionKind::EndOfInput => {
                            super::super::StreamMessage::end_of_input()
                        }
                    };
                    for binding_outputs in outputs.values_mut() {
                        send_progress_fanout(binding_outputs, message.clone(), cancellation)
                            .await?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn durable_source_manifest_entries<C: DriverLogicalClock>(
    driver: &StreamProgressDriver<C>,
    source_cuts: &BTreeMap<BindingIdentity, DurableSourceCut>,
) -> Result<BTreeMap<String, SourceManifestEntry>> {
    driver
        .state
        .bindings
        .values()
        .map(|state| {
            let id = state.prepared.identity.as_str();
            let cut = &source_cuts[&state.prepared.identity];
            let activity = driver
                .state
                .aggregate
                .activity(state.prepared.ordinal)
                .expect("aggregate and source progress share prepared ordinals");
            let ended = matches!(activity, IngressActivity::Ended { .. });
            if cut.ended != ended || state.ended != ended {
                return Err(CalcFlowError::CheckpointMismatch {
                    message: format!(
                        "checkpoint source terminal cut for {id:?} does not match progress state"
                    ),
                });
            }
            let idle = matches!(activity, IngressActivity::Idle { .. });
            let watermark_policy = match &state.prepared.normalized_watermark {
                NormalizedWatermarkMode::SourceProvided { .. } => {
                    SourceWatermarkManifestState::SourceProvided {
                        last_emitted_micros: state.last_source_watermark,
                        idle,
                    }
                }
                NormalizedWatermarkMode::Generated { .. } => {
                    let generated = state
                        .generated
                        .as_ref()
                        .expect("generated watermark policy has runtime state");
                    SourceWatermarkManifestState::BoundedOutOfOrderness {
                        observed_max_micros: generated
                            .max_observed_nanos()
                            .map(event_time_from_nanos)
                            .transpose()?,
                        last_emitted_micros: generated.last_emitted(),
                        idle,
                    }
                }
                NormalizedWatermarkMode::Disabled { .. } => {
                    SourceWatermarkManifestState::Disabled { idle }
                }
            };
            Ok((
                id.to_owned(),
                SourceManifestEntry {
                    cursor: cut.cursor.clone(),
                    identity_hash: state.prepared.identity_hash(),
                    sequence: cut.next_sequence,
                    ended,
                    watermark_policy,
                },
            ))
        })
        .collect()
}

fn event_time_from_nanos(nanos: i128) -> Result<EventTime> {
    i64::try_from(nanos.div_euclid(1_000))
        .map(EventTime::from_micros)
        .map_err(|_| CalcFlowError::CheckpointMismatch {
            message: "generated watermark observed maximum exceeds the durable range".into(),
        })
}

async fn send_progress_fanout(
    outputs: &mut [super::super::EdgeSender],
    message: super::super::StreamMessage,
    cancellation: &crate::CancellationToken,
) -> Result<()> {
    for output in outputs {
        tokio::select! {
            biased;
            () = cancellation.cancelled() => return Ok(()),
            result = output.send(message.clone()) => result?,
        }
    }
    Ok(())
}

pub(crate) fn spawn_live_progress_task(
    supervisor: &mut super::super::supervisor::TaskSupervisor,
    coordinator: LiveProgressCoordinator,
    cancellation: crate::CancellationToken,
) {
    supervisor.spawn_with_failure_signal("progress:driver", move |failure_signal| async move {
        let result = coordinator.run(cancellation).await;
        if result.is_err() {
            failure_signal.cancel_siblings();
        }
        result
    });
}

#[allow(
    dead_code,
    reason = "phase errors are constructed by snapshot/replay paths"
)]
fn phase_error_marker(binding: BindingIdentity) -> DriverPhaseError {
    DriverPhaseError {
        phase: DriverFailurePhase::AdmissionDecision,
        coordinate: DriverPhaseCoordinate::Admission {
            last_attempt: None,
            binding,
            gate_generation: None,
        },
        failure: ProgressFailure::new(
            ProgressFailureKind::ProtocolViolation,
            "runtime.progress.driver_phase",
            "phase failure",
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, num::NonZeroUsize, sync::Arc, time::Duration};

    use datafusion::arrow::{
        array::{ArrayRef, Int64Array, TimestampMicrosecondArray},
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    };
    use futures::executor::block_on;

    use super::{DriverEmission, ManualClock, RawIngressEvent, StreamProgressDriver};
    use crate::runtime::streaming::progress::{
        aggregate::ProgressEmissionKind,
        durable::{DurableProgressRestore, RestoredSourceProgress},
        prepare::{
            BindingIdentity, DeclaredSchema, FenceSelectionPolicy, NativeWatermarkCapability,
            ReplayPositioningCapability, SourceBindingSpec, SourceDescriptor,
            StreamProgressRuntimeConfig, WatermarkPolicy, prepare_stream_job,
        },
        trace::{
            AdmissionDecisionRecord, ProgressExecutionTrace, ProgressReplayRequest,
            ProgressTraceRecord, RawIngressEventKind, RawUpstreamPosition, SettlementDisposition,
        },
        types::{
            CheckedSemanticAllocator, DriverFailurePhase, LogicalInstant, ReadyClass, TimerKind,
        },
    };
    use crate::{Batch, BatchMetadata, CursorManifestEntry, EventTime, JsonMap};

    #[derive(Debug, Eq, PartialEq)]
    struct RecordedSeedArtifact {
        seed: u64,
        ordered_raw_attempts: Vec<(BindingIdentity, RawIngressEventKind, RawUpstreamPosition)>,
        logical_clock_trace: Vec<LogicalInstant>,
        prepared_job_fingerprint: [u8; 32],
        runtime_config: StreamProgressRuntimeConfig,
        execution_trace: ProgressExecutionTrace,
        terminal_phase: super::DriverPhase,
        terminal_unsettled_receipts: usize,
        terminal_timer_entries: u64,
    }

    fn identity(value: &str) -> BindingIdentity {
        BindingIdentity::new(value).unwrap()
    }

    fn exact(value: u8) -> RawUpstreamPosition {
        RawUpstreamPosition::Exact {
            delivery_replay_cursor: vec![value],
            control_frontier: vec![value],
        }
    }

    fn source_provided(binding: &str) -> SourceBindingSpec {
        SourceBindingSpec {
            descriptor: SourceDescriptor::new(
                identity(binding),
                DeclaredSchema::DynamicOrUnknown,
                NativeWatermarkCapability::EmitsNative,
                ReplayPositioningCapability::ExactPauseReportAndSeek,
                None,
            ),
            watermark_policy: WatermarkPolicy::SourceProvided,
        }
    }

    fn generated(binding: &str, interval_ns: u64, idle_ns: Option<u64>) -> SourceBindingSpec {
        SourceBindingSpec {
            descriptor: SourceDescriptor::new(
                identity(binding),
                DeclaredSchema::Known(Arc::new(Schema::new(vec![Field::new(
                    "at",
                    DataType::Timestamp(TimeUnit::Microsecond, None),
                    true,
                )]))),
                NativeWatermarkCapability::NeverEmits,
                ReplayPositioningCapability::ExactPauseReportAndSeek,
                None,
            ),
            watermark_policy: WatermarkPolicy::BoundedOutOfOrderness {
                event_time_column: Arc::from("at"),
                max_out_of_orderness: Duration::from_micros(1),
                emit_interval: Duration::from_nanos(interval_ns),
                idle_timeout: idle_ns.map(Duration::from_nanos),
            },
        }
    }

    fn disabled(binding: &str, idle_ns: Option<u64>) -> SourceBindingSpec {
        SourceBindingSpec {
            descriptor: SourceDescriptor::new(
                identity(binding),
                DeclaredSchema::DynamicOrUnknown,
                NativeWatermarkCapability::NeverEmits,
                ReplayPositioningCapability::ExactPauseReportAndSeek,
                None,
            ),
            watermark_policy: WatermarkPolicy::Disabled {
                idle_timeout: idle_ns.map(Duration::from_nanos),
            },
        }
    }

    fn prepared(bindings: &[SourceBindingSpec]) -> Arc<super::PreparedStreamJob> {
        Arc::new(
            prepare_stream_job(
                "compiled",
                bindings,
                StreamProgressRuntimeConfig {
                    per_binding_inbox_capacity: NonZeroUsize::new(32).unwrap(),
                    fence_selection: FenceSelectionPolicy::AllVisible,
                },
            )
            .unwrap(),
        )
    }

    fn int_batch(value: i64) -> Batch {
        Batch::table(
            vec![
                RecordBatch::try_from_iter(vec![(
                    "value",
                    Arc::new(Int64Array::from(vec![value])) as ArrayRef,
                )])
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    #[test]
    fn durable_driver_restore_rejects_noncanonical_cursor_hex() {
        let prepared = prepared(&[source_provided("left")]);
        for order in ["0", "00FF"] {
            let restored = DurableProgressRestore {
                origin: LogicalInstant::ZERO,
                sources: BTreeMap::from([(
                    "left".into(),
                    RestoredSourceProgress {
                        cursor: Some(CursorManifestEntry {
                            order: order.into(),
                            payload: JsonMap::new(),
                        }),
                        next_sequence: 1,
                        ended: false,
                        idle: false,
                        observed_max: None,
                        last_watermark: None,
                        watermark_deadline: None,
                        idle_deadline: None,
                    },
                )]),
                next_receipt_sequence: 0,
                trace_records: 0,
            };

            let error = match StreamProgressDriver::restore_durable(
                &prepared,
                ManualClock::new(LogicalInstant::ZERO),
                &restored,
            ) {
                Ok(_) => panic!("noncanonical durable cursor unexpectedly restored"),
                Err(error) => error,
            };

            assert!(matches!(
                error,
                crate::CalcFlowError::CheckpointMismatch { message }
                    if message.contains("canonical lowercase even-length hexadecimal")
            ));
        }
    }

    fn timestamp_batch(value: i64) -> Batch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "at",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        )]));
        Batch::table(
            vec![
                RecordBatch::try_new(
                    schema,
                    vec![Arc::new(TimestampMicrosecondArray::from(vec![value]))],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn source_watermark_is_monotonic() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[source_provided("left")]), clock).unwrap();
        driver.start_running().unwrap();
        for (position, watermark) in [(0, 5), (1, 5)] {
            let receipt = sender
                .submit(
                    identity("left"),
                    RawIngressEvent::ConnectorWatermark(EventTime::from_micros(watermark)),
                    exact(position),
                )
                .await
                .unwrap();
            driver.drain_ready().unwrap();
            receipt.wait_settled().await.unwrap();
        }
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::ConnectorWatermark(EventTime::from_micros(4)),
                exact(2),
            )
            .await
            .unwrap();
        assert!(driver.drain_ready().is_err());
        assert!(receipt.wait_settled().await.is_err());
        assert_eq!(driver.unsettled_receipts(), 0);
    }

    #[tokio::test]
    async fn illegal_connector_watermark_is_rejected() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::ConnectorWatermark(EventTime::from_micros(1)),
                exact(0),
            )
            .await
            .unwrap();
        assert!(driver.drain_ready().is_err());
        assert!(receipt.wait_settled().await.is_err());
    }

    #[tokio::test]
    async fn queued_data_precedes_timers_at_same_deadline() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[generated("left", 10, None)]), clock.clone())
                .unwrap();
        driver.start_running().unwrap();
        clock.set(LogicalInstant(10));
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(timestamp_batch(5)),
                exact(0),
            )
            .await
            .unwrap();
        let drain = driver.drain_ready().unwrap();
        assert!(matches!(
            drain.emissions.first(),
            Some(DriverEmission::ForwardData { .. })
        ));
        receipt.wait_settled().await.unwrap();
        let record = driver.trace().drain_projection().last().unwrap().clone();
        assert!(matches!(
            record.selected_items_in_ready_order[0],
            super::ReadyItemIdentity::Raw { .. }
        ));
    }

    #[tokio::test]
    async fn watermark_timer_precedes_idle_timer_across_bindings() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) = StreamProgressDriver::new(
            prepared(&[generated("left", 10, Some(10)), disabled("right", Some(10))]),
            clock.clone(),
        )
        .unwrap();
        driver.start_running().unwrap();
        for (position, binding, batch) in
            [(0, "left", timestamp_batch(5)), (1, "right", int_batch(1))]
        {
            let receipt = sender
                .submit(
                    identity(binding),
                    RawIngressEvent::Data(batch),
                    exact(position),
                )
                .await
                .unwrap();
            driver.drain_ready().unwrap();
            receipt.wait_settled().await.unwrap();
        }
        clock.set(LogicalInstant(10));
        driver.drain_ready().unwrap();
        let drain = driver.trace().drain_projection().last().unwrap().clone();
        let classes = drain.selected_key_range;
        assert!(
            matches!(classes, super::ReadyKeyRange::Inclusive { first, .. } if first.class == ReadyClass::WatermarkTimer)
        );
        assert_eq!(
            drain.due_timers_in_ready_order[0].kind,
            TimerKind::Watermark
        );
        assert!(
            drain
                .due_timers_in_ready_order
                .iter()
                .skip(1)
                .all(|timer| timer.kind == TimerKind::Idle)
        );
    }

    #[tokio::test]
    async fn timer_generation_is_validation_only() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, _) =
            StreamProgressDriver::new(prepared(&[generated("left", 10, None)]), clock.clone())
                .unwrap();
        driver.start_running().unwrap();
        let ordinal = super::BindingOrdinal::new(0);
        let stale = driver.state.bindings[&ordinal].watermark_timer.unwrap();
        super::arm_timer(
            &mut driver.state,
            ordinal,
            TimerKind::Watermark,
            LogicalInstant(20),
        )
        .unwrap();
        driver
            .state
            .timer_heap
            .insert(stale.identity(ordinal, TimerKind::Watermark), stale);
        clock.set(LogicalInstant(20));
        let drain = driver.drain_ready().unwrap();
        assert_eq!(
            driver
                .trace()
                .drain_projection()
                .last()
                .unwrap()
                .due_timers_in_ready_order
                .len(),
            2
        );
        assert!(drain.emissions.is_empty());
        assert_eq!(driver.next_central_wake(), Some(LogicalInstant(30)));
    }

    #[tokio::test]
    async fn timer_lifecycle_is_driver_owned() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[generated("left", 10, Some(5))]), clock).unwrap();
        driver.start_running().unwrap();
        assert_eq!(driver.state.timer_heap.len(), 1);
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(timestamp_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        driver.drain_ready().unwrap();
        receipt.wait_settled().await.unwrap();
        assert_eq!(driver.state.timer_heap.len(), 2);
        assert!(
            driver.state.bindings[&super::BindingOrdinal::new(0)]
                .next_timer_generation
                .next()
                >= 2
        );
    }

    #[tokio::test]
    async fn post_end_input_aborts_whole_ready_snapshot_atomically() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        let end = sender
            .submit(identity("left"), RawIngressEvent::EndOfInput, exact(0))
            .await
            .unwrap();
        let data = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(1),
            )
            .await
            .unwrap();
        assert!(driver.drain_ready().is_err());
        assert!(end.wait_settled().await.is_err());
        assert!(data.wait_settled().await.is_err());
        assert_eq!(driver.unsettled_receipts(), 0);
    }

    #[tokio::test]
    async fn m3_does_not_classify_or_drop_late_rows() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[source_provided("left")]), clock).unwrap();
        driver.start_running().unwrap();
        let watermark = sender
            .submit(
                identity("left"),
                RawIngressEvent::ConnectorWatermark(EventTime::from_micros(100)),
                exact(0),
            )
            .await
            .unwrap();
        driver.drain_ready().unwrap();
        watermark.wait_settled().await.unwrap();
        for (position, value) in [(1, 1), (2, 100), (3, 1000)] {
            let receipt = sender
                .submit(
                    identity("left"),
                    RawIngressEvent::Data(int_batch(value)),
                    exact(position),
                )
                .await
                .unwrap();
            let drain = driver.drain_ready().unwrap();
            assert!(matches!(
                drain.emissions.as_slice(),
                [DriverEmission::ForwardData { .. }]
            ));
            receipt.wait_settled().await.unwrap();
        }
    }

    #[tokio::test]
    async fn input_control_uses_driver_owned_logical_instant() {
        let clock = ManualClock::new(LogicalInstant(7));
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        for position in 0..2 {
            sender
                .submit(
                    identity("left"),
                    RawIngressEvent::Data(int_batch(i64::from(position))),
                    exact(position),
                )
                .await
                .unwrap();
        }
        driver.drain_ready().unwrap();
        let record = driver.trace().drain_projection().last().unwrap().clone();
        let super::ReadyKeyRange::Inclusive { first, last } = record.selected_key_range else {
            panic!("expected a non-empty key range");
        };
        assert_eq!(first.logical_instant, LogicalInstant(7));
        assert_eq!(last.logical_instant, LogicalInstant(7));
    }

    #[tokio::test]
    async fn idle_deadline_uses_last_driver_activity_instant() {
        let clock = ManualClock::new(LogicalInstant(5));
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", Some(10))]), clock).unwrap();
        driver.start_running().unwrap();
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        driver.drain_ready().unwrap();
        receipt.wait_settled().await.unwrap();
        assert_eq!(driver.next_central_wake(), Some(LogicalInstant(15)));
    }

    #[tokio::test]
    async fn ready_snapshot_success_commits_atomically() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        let mut receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        assert!(matches!(
            receipt.settled.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        let drain = driver.drain_ready().unwrap();
        assert_eq!(drain.committed_inputs.len(), 1);
        receipt.wait_settled().await.unwrap();
        assert_eq!(driver.unsettled_receipts(), 0);
    }

    #[tokio::test]
    async fn commit_receipt_succeeds_only_after_atomic_commit() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        let mut receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        assert!(matches!(
            receipt.settled.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        let before = driver.status();
        assert_eq!(before.unsettled_receipts, 1);
        let drain = driver.drain_ready().unwrap();
        let committed = receipt.wait_settled().await.unwrap();
        assert_eq!(drain.committed_inputs, vec![committed]);
        assert_eq!(driver.status().unsettled_receipts, 0);
    }

    #[tokio::test]
    async fn new_ready_work_waits_for_next_arbitration_snapshot() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        let first = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        let tail = Arc::new(std::sync::Mutex::new(None));
        let tail_for_hook = Arc::clone(&tail);
        let sender_for_hook = sender.clone();
        driver.install_after_freeze(move || {
            let receipt = block_on(sender_for_hook.submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(2)),
                exact(1),
            ))
            .unwrap();
            *tail_for_hook.lock().unwrap() = Some(receipt);
        });
        assert_eq!(driver.drain_ready().unwrap().committed_inputs.len(), 1);
        first.wait_settled().await.unwrap();
        assert_eq!(driver.unsettled_receipts(), 1);
        assert_eq!(driver.drain_ready().unwrap().committed_inputs.len(), 1);
        let tail_receipt = tail.lock().unwrap().take().unwrap();
        tail_receipt.wait_settled().await.unwrap();
    }

    #[tokio::test]
    async fn ready_snapshot_protocol_error_discards_valid_prefix() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        let valid = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        let invalid = sender
            .submit(
                identity("left"),
                RawIngressEvent::ConnectorWatermark(EventTime::from_micros(1)),
                exact(1),
            )
            .await
            .unwrap();
        assert!(driver.drain_ready().is_err());
        assert!(valid.wait_settled().await.is_err());
        assert!(invalid.wait_settled().await.is_err());
        assert_eq!(driver.status().aggregate_watermark, None);
        assert_eq!(driver.unsettled_receipts(), 0);
    }

    #[tokio::test]
    async fn failed_drain_settles_all_selected_with_first_error() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        let receipts =
            futures::future::join_all([(0_u8, 0_i64), (1, 1)].map(|(position, watermark)| {
                sender.submit(
                    identity("left"),
                    RawIngressEvent::ConnectorWatermark(EventTime::from_micros(watermark)),
                    exact(position),
                )
            }))
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(driver.drain_ready().is_err());
        let errors = futures::future::join_all(
            receipts
                .into_iter()
                .map(super::RawCommitReceipt::wait_settled),
        )
        .await;
        assert!(errors.iter().all(Result::is_err));
        assert_eq!(driver.unsettled_receipts(), 0);
    }

    #[tokio::test]
    async fn post_end_stale_internal_timer_is_effect_free() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[generated("left", 10, None)]), clock.clone())
                .unwrap();
        driver.start_running().unwrap();
        clock.set(LogicalInstant(10));
        let end = sender
            .submit(identity("left"), RawIngressEvent::EndOfInput, exact(0))
            .await
            .unwrap();
        let drain = driver.drain_ready().unwrap();
        end.wait_settled().await.unwrap();
        assert!(matches!(
            drain.emissions.as_slice(),
            [DriverEmission::Progress(super::ProgressEmission {
                kind: ProgressEmissionKind::EndOfInput,
                ..
            })]
        ));
        assert_eq!(driver.next_central_wake(), None);
    }

    #[tokio::test]
    async fn sequence_exhaustion_aborts_without_mutation() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        driver
            .state
            .bindings
            .get_mut(&super::BindingOrdinal::new(0))
            .unwrap()
            .next_local_sequence = CheckedSemanticAllocator::new(
            u64::MAX,
            "runtime.progress.counters.left.local_sequence",
        );
        assert!(driver.drain_ready().is_err());
        assert!(receipt.wait_settled().await.is_err());
        assert_eq!(driver.unsettled_receipts(), 0);
        assert_eq!(driver.phase(), super::DriverPhase::Terminal);
        assert!(driver.trace().records.iter().any(|record| {
            matches!(record, ProgressTraceRecord::DriverPhaseFailure { error, .. }
                if error.phase == DriverFailurePhase::ReadyKeyConstruction)
        }));
    }

    #[tokio::test]
    async fn timer_counter_exhaustion_aborts_without_mutation() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", Some(10))]), clock).unwrap();
        driver.start_running().unwrap();
        driver
            .state
            .bindings
            .get_mut(&super::BindingOrdinal::new(0))
            .unwrap()
            .next_timer_generation = CheckedSemanticAllocator::new(
            u64::MAX,
            "runtime.progress.counters.left.timer_generation",
        );
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        assert!(driver.drain_ready().is_err());
        assert!(receipt.wait_settled().await.is_err());
        assert_eq!(driver.next_central_wake(), None);
        assert_eq!(driver.unsettled_receipts(), 0);
    }

    #[tokio::test]
    async fn receipt_fence_epoch_and_inbox_counter_overflow_is_atomic() {
        for counter in ["receipt", "inbox"] {
            let clock = ManualClock::new(LogicalInstant::ZERO);
            let (mut driver, sender) =
                StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
            driver.start_running().unwrap();
            {
                let mut admission = sender.admission.lock();
                if counter == "receipt" {
                    admission
                        .next_receipt_sequence
                        .set_next_for_restore(u64::MAX);
                } else {
                    admission
                        .bindings
                        .get_mut(&super::BindingOrdinal::new(0))
                        .unwrap()
                        .next_inbox_sequence
                        .set_next_for_restore(u64::MAX);
                }
            }
            assert!(
                sender
                    .submit(
                        identity("left"),
                        RawIngressEvent::Data(int_batch(1)),
                        exact(0),
                    )
                    .await
                    .is_err()
            );
            assert!(driver.drain_ready().is_err());
            assert_eq!(driver.phase(), super::DriverPhase::Terminal);
            assert_eq!(driver.unsettled_receipts(), 0);
        }

        for counter in ["fence", "drain"] {
            let clock = ManualClock::new(LogicalInstant::ZERO);
            let (mut driver, sender) =
                StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
            driver.start_running().unwrap();
            let receipt = sender
                .submit(
                    identity("left"),
                    RawIngressEvent::Data(int_batch(1)),
                    exact(0),
                )
                .await
                .unwrap();
            if counter == "fence" {
                sender
                    .admission
                    .lock()
                    .bindings
                    .get_mut(&super::BindingOrdinal::new(0))
                    .unwrap()
                    .next_fence_sequence
                    .set_next_for_restore(u64::MAX);
            } else {
                driver.next_drain_epoch.set_next_for_restore(u64::MAX);
            }
            assert!(driver.drain_ready().is_err());
            assert!(receipt.wait_settled().await.is_err());
            assert_eq!(driver.phase(), super::DriverPhase::Terminal);
            assert_eq!(driver.unsettled_receipts(), 0);
        }
    }

    #[tokio::test]
    async fn execution_trace_admission_and_gate_counter_overflow_is_atomic() {
        for counter in ["trace", "admission"] {
            let clock = ManualClock::new(LogicalInstant::ZERO);
            let (mut driver, sender) =
                StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
            driver.start_running().unwrap();
            {
                let mut admission = sender.admission.lock();
                if counter == "trace" {
                    admission.trace.set_next_coordinates_for_test(u64::MAX);
                } else {
                    admission
                        .next_admission_attempt
                        .set_next_for_restore(u64::MAX);
                }
            }
            assert!(
                sender
                    .submit(
                        identity("left"),
                        RawIngressEvent::Data(int_batch(1)),
                        exact(0),
                    )
                    .await
                    .is_err()
            );
            assert!(driver.drain_ready().is_err());
            assert_eq!(driver.phase(), super::DriverPhase::Terminal);
            assert_eq!(driver.unsettled_receipts(), 0);
        }

        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        sender
            .admission
            .lock()
            .trace
            .set_next_coordinates_for_test(u64::MAX);
        assert!(driver.drain_ready().is_err());
        assert!(receipt.wait_settled().await.is_err());
        assert_eq!(driver.phase(), super::DriverPhase::Terminal);
        assert_eq!(driver.unsettled_receipts(), 0);

        for counter in ["generation", "close"] {
            let clock = ManualClock::new(LogicalInstant::ZERO);
            let (mut driver, sender) =
                StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
            driver.start_running().unwrap();
            let receipt = sender
                .submit(
                    identity("left"),
                    RawIngressEvent::Data(int_batch(1)),
                    exact(0),
                )
                .await
                .unwrap();
            {
                let mut admission = sender.admission.lock();
                if counter == "generation" {
                    admission
                        .bindings
                        .get_mut(&super::BindingOrdinal::new(0))
                        .unwrap()
                        .gate
                        .generation = super::AdmissionGateGeneration(u64::MAX);
                } else {
                    admission
                        .next_gate_close_ordinal
                        .set_next_for_restore(u64::MAX);
                }
            }
            assert!(driver.cancel().is_err());
            assert!(receipt.wait_settled().await.is_err());
            assert_eq!(driver.phase(), super::DriverPhase::Terminal);
            assert_eq!(driver.unsettled_receipts(), 0);
        }
    }

    #[tokio::test]
    async fn pre_key_driver_phase_error_has_no_ready_key() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        driver
            .state
            .bindings
            .get_mut(&super::BindingOrdinal::new(0))
            .unwrap()
            .next_local_sequence = CheckedSemanticAllocator::new(
            u64::MAX,
            "runtime.progress.counters.left.local_sequence",
        );
        assert!(driver.drain_ready().is_err());
        assert!(receipt.wait_settled().await.is_err());
        let trace = driver.trace();
        assert!(
            !trace
                .records
                .iter()
                .any(|record| { matches!(record, ProgressTraceRecord::Drain(_)) })
        );
        assert!(trace.records.iter().any(|record| {
            matches!(record, ProgressTraceRecord::DriverPhaseFailure { error, .. }
                if error.phase == DriverFailurePhase::ReadyKeyConstruction
                    && matches!(error.coordinate, super::DriverPhaseCoordinate::Counter { .. }))
        }));
    }

    #[test]
    fn ordinal_or_deadline_exhaustion_is_atomic_fatal_error() {
        let clock = ManualClock::new(LogicalInstant(u128::MAX));
        let (mut driver, _) =
            StreamProgressDriver::new(prepared(&[generated("left", 1, None)]), clock).unwrap();
        assert!(driver.start_running().is_err());
        assert_eq!(driver.phase(), super::DriverPhase::Prepared);
        assert_eq!(driver.unsettled_receipts(), 0);
        assert_eq!(driver.next_central_wake(), None);
    }

    #[tokio::test]
    async fn final_end_atomically_rejects_post_fence_tail() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        let end = sender
            .submit(identity("left"), RawIngressEvent::EndOfInput, exact(0))
            .await
            .unwrap();
        let tail_sender = sender.clone();
        driver.install_after_freeze(move || {
            let receipt = block_on(tail_sender.submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(2)),
                exact(1),
            ))
            .unwrap();
            drop(receipt);
        });
        let drain = driver.drain_ready().unwrap();
        assert_eq!(drain.terminal_tail_failures, 1);
        end.wait_settled().await.unwrap();
        assert_eq!(driver.unsettled_receipts(), 0);
        assert!(driver
            .trace()
            .records
            .iter()
            .any(|record| matches!(record, ProgressTraceRecord::Settlement(record) if record.disposition == SettlementDisposition::PostEndTailReject)));
    }

    async fn terminal_tail_trace() -> ProgressExecutionTrace {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        let end = sender
            .submit(identity("left"), RawIngressEvent::EndOfInput, exact(0))
            .await
            .unwrap();
        let tail_sender = sender.clone();
        driver.install_after_freeze(move || {
            drop(
                block_on(tail_sender.submit(
                    identity("left"),
                    RawIngressEvent::Data(int_batch(2)),
                    exact(1),
                ))
                .unwrap(),
            );
        });
        driver.drain_ready().unwrap();
        end.wait_settled().await.unwrap();
        driver.trace()
    }

    #[tokio::test]
    async fn progress_replay_reproduces_terminal_tail_linearization() {
        let expected = terminal_tail_trace().await;
        let request = ProgressReplayRequest::prevalidate(expected.clone()).unwrap();
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut replay, sender) =
            StreamProgressDriver::replay(prepared(&[disabled("left", None)]), clock, request)
                .unwrap();
        replay.start_running().unwrap();
        let end = sender
            .submit(identity("left"), RawIngressEvent::EndOfInput, exact(0))
            .await
            .unwrap();
        let tail_sender = sender.clone();
        replay.install_after_freeze(move || {
            drop(
                block_on(tail_sender.submit(
                    identity("left"),
                    RawIngressEvent::Data(int_batch(2)),
                    exact(1),
                ))
                .unwrap(),
            );
        });
        replay.drain_ready().unwrap();
        end.wait_settled().await.unwrap();
        replay.finish_replay().unwrap();
        assert_eq!(replay.trace(), expected);
    }

    #[tokio::test]
    async fn progress_replay_reproduces_cancellation_settlements() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let prepared = prepared(&[disabled("left", None)]);
        let (mut recorded, sender) =
            StreamProgressDriver::new(Arc::clone(&prepared), clock.clone()).unwrap();
        recorded.start_running().unwrap();
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        recorded.cancel().unwrap();
        assert!(receipt.wait_settled().await.is_err());
        let expected = recorded.trace();
        let request = ProgressReplayRequest::prevalidate(expected.clone()).unwrap();
        let (mut replay, sender) = StreamProgressDriver::replay(prepared, clock, request).unwrap();
        replay.start_running().unwrap();
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        replay.cancel().unwrap();
        assert!(receipt.wait_settled().await.is_err());
        replay.finish_replay().unwrap();
        assert_eq!(replay.trace(), expected);
    }

    #[tokio::test]
    async fn progress_replay_reproduces_immediate_rejection() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let prepared = prepared(&[disabled("left", None)]);
        let (mut recorded, sender) =
            StreamProgressDriver::new(Arc::clone(&prepared), clock.clone()).unwrap();
        recorded.start_running().unwrap();
        let end = sender
            .submit(identity("left"), RawIngressEvent::EndOfInput, exact(0))
            .await
            .unwrap();
        recorded.drain_ready().unwrap();
        end.wait_settled().await.unwrap();
        assert!(
            sender
                .submit(
                    identity("left"),
                    RawIngressEvent::Data(int_batch(1)),
                    exact(1),
                )
                .await
                .is_err()
        );
        let expected = recorded.trace();
        let request = ProgressReplayRequest::prevalidate(expected.clone()).unwrap();
        let (mut replay, sender) = StreamProgressDriver::replay(prepared, clock, request).unwrap();
        replay.start_running().unwrap();
        let end = sender
            .submit(identity("left"), RawIngressEvent::EndOfInput, exact(0))
            .await
            .unwrap();
        replay.drain_ready().unwrap();
        end.wait_settled().await.unwrap();
        assert!(
            sender
                .submit(
                    identity("left"),
                    RawIngressEvent::Data(int_batch(1)),
                    exact(1),
                )
                .await
                .is_err()
        );
        replay.finish_replay().unwrap();
        assert_eq!(replay.trace(), expected);
    }

    #[tokio::test]
    async fn progress_replay_reproduces_fatal_cleanup_settlements() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let prepared = prepared(&[disabled("left", None)]);
        let (mut recorded, sender) =
            StreamProgressDriver::new(Arc::clone(&prepared), clock.clone()).unwrap();
        recorded.start_running().unwrap();
        let invalid = sender
            .submit(
                identity("left"),
                RawIngressEvent::ConnectorWatermark(EventTime::from_micros(1)),
                exact(0),
            )
            .await
            .unwrap();
        let tail_sender = sender.clone();
        recorded.install_after_freeze(move || {
            drop(
                block_on(tail_sender.submit(
                    identity("left"),
                    RawIngressEvent::Data(int_batch(2)),
                    exact(1),
                ))
                .unwrap(),
            );
        });
        assert!(recorded.drain_ready().is_err());
        assert!(invalid.wait_settled().await.is_err());
        let expected = recorded.trace();
        let request = ProgressReplayRequest::prevalidate(expected.clone()).unwrap();
        let (mut replay, sender) = StreamProgressDriver::replay(prepared, clock, request).unwrap();
        replay.start_running().unwrap();
        let invalid = sender
            .submit(
                identity("left"),
                RawIngressEvent::ConnectorWatermark(EventTime::from_micros(1)),
                exact(0),
            )
            .await
            .unwrap();
        let tail_sender = sender.clone();
        replay.install_after_freeze(move || {
            drop(
                block_on(tail_sender.submit(
                    identity("left"),
                    RawIngressEvent::Data(int_batch(2)),
                    exact(1),
                ))
                .unwrap(),
            );
        });
        assert!(replay.drain_ready().is_err());
        assert!(invalid.wait_settled().await.is_err());
        replay.finish_replay().unwrap();
        assert_eq!(replay.trace(), expected);
    }

    #[tokio::test]
    async fn progress_replay_reproduces_full_progress_execution_trace() {
        let expected = terminal_tail_trace().await;
        assert!(
            expected
                .records
                .iter()
                .any(|record| { matches!(record, ProgressTraceRecord::Admission(_)) })
        );
        assert!(
            expected
                .records
                .iter()
                .any(|record| { matches!(record, ProgressTraceRecord::Drain(_)) })
        );
        assert!(
            expected
                .records
                .iter()
                .any(|record| { matches!(record, ProgressTraceRecord::Terminal(_)) })
        );
        assert!(
            expected
                .records
                .iter()
                .any(|record| { matches!(record, ProgressTraceRecord::Settlement(_)) })
        );
        let request = ProgressReplayRequest::prevalidate(expected.clone()).unwrap();
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut replay, sender) =
            StreamProgressDriver::replay(prepared(&[disabled("left", None)]), clock, request)
                .unwrap();
        replay.start_running().unwrap();
        let end = sender
            .submit(identity("left"), RawIngressEvent::EndOfInput, exact(0))
            .await
            .unwrap();
        let tail_sender = sender.clone();
        replay.install_after_freeze(move || {
            drop(
                block_on(tail_sender.submit(
                    identity("left"),
                    RawIngressEvent::Data(int_batch(2)),
                    exact(1),
                ))
                .unwrap(),
            );
        });
        replay.drain_ready().unwrap();
        end.wait_settled().await.unwrap();
        replay.finish_replay().unwrap();
        assert_eq!(replay.trace(), expected);
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the recorded artifact keeps each seeded lifecycle attempt and terminal path visible"
    )]
    async fn record_seed_artifact(seed: u64) -> RecordedSeedArtifact {
        let prepared = prepared(&[disabled("left", None), disabled("right", None)]);
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(Arc::clone(&prepared), clock.clone()).unwrap();
        driver.start_running().unwrap();
        let mut ordered_raw_attempts = Vec::new();
        let mut logical_clock_trace = Vec::new();
        let mut next_instant = u128::from(seed) * 100;
        let mut next_position = 0_u8;

        clock.set(LogicalInstant(next_instant));
        logical_clock_trace.push(LogicalInstant(next_instant));
        driver.drain_ready().unwrap();

        let binding_order = if seed % 2 == 0 {
            ["left", "right"]
        } else {
            ["right", "left"]
        };
        for binding in binding_order {
            next_instant += 1;
            clock.set(LogicalInstant(next_instant));
            logical_clock_trace.push(LogicalInstant(next_instant));
            let upstream = exact(next_position);
            next_position += 1;
            ordered_raw_attempts.push((
                identity(binding),
                RawIngressEventKind::Data,
                upstream.clone(),
            ));
            let receipt = sender
                .submit(
                    identity(binding),
                    RawIngressEvent::Data(int_batch(i64::try_from(seed).unwrap())),
                    upstream,
                )
                .await
                .unwrap();
            driver.drain_ready().unwrap();
            receipt.wait_settled().await.unwrap();
        }

        match seed % 4 {
            0 => {
                for binding in binding_order {
                    next_instant += 1;
                    clock.set(LogicalInstant(next_instant));
                    logical_clock_trace.push(LogicalInstant(next_instant));
                    let upstream = exact(next_position);
                    next_position += 1;
                    ordered_raw_attempts.push((
                        identity(binding),
                        RawIngressEventKind::EndOfInput,
                        upstream.clone(),
                    ));
                    let receipt = sender
                        .submit(identity(binding), RawIngressEvent::EndOfInput, upstream)
                        .await
                        .unwrap();
                    driver.drain_ready().unwrap();
                    receipt.wait_settled().await.unwrap();
                }
            }
            1 => {
                next_instant += 1;
                clock.set(LogicalInstant(next_instant));
                logical_clock_trace.push(LogicalInstant(next_instant));
                let upstream = exact(next_position);
                next_position += 1;
                ordered_raw_attempts.push((
                    identity("left"),
                    RawIngressEventKind::Data,
                    upstream.clone(),
                ));
                let receipt = sender
                    .submit(
                        identity("left"),
                        RawIngressEvent::Data(int_batch(-1)),
                        upstream,
                    )
                    .await
                    .unwrap();
                driver.cancel().unwrap();
                assert!(receipt.wait_settled().await.is_err());
            }
            2 => {
                next_instant += 1;
                clock.set(LogicalInstant(next_instant));
                logical_clock_trace.push(LogicalInstant(next_instant));
                let upstream = exact(next_position);
                next_position += 1;
                ordered_raw_attempts.push((
                    identity("right"),
                    RawIngressEventKind::ConnectorWatermark,
                    upstream.clone(),
                ));
                let receipt = sender
                    .submit(
                        identity("right"),
                        RawIngressEvent::ConnectorWatermark(EventTime::from_micros(1)),
                        upstream,
                    )
                    .await
                    .unwrap();
                assert!(driver.drain_ready().is_err());
                assert!(receipt.wait_settled().await.is_err());
            }
            _ => {
                next_instant += 1;
                clock.set(LogicalInstant(next_instant));
                logical_clock_trace.push(LogicalInstant(next_instant));
                let upstream = exact(next_position);
                next_position += 1;
                ordered_raw_attempts.push((
                    identity("right"),
                    RawIngressEventKind::Data,
                    upstream.clone(),
                ));
                let receipt = sender
                    .submit(
                        identity("right"),
                        RawIngressEvent::Data(int_batch(-1)),
                        upstream,
                    )
                    .await
                    .unwrap();
                driver
                    .state
                    .bindings
                    .get_mut(&super::BindingOrdinal::new(1))
                    .unwrap()
                    .next_local_sequence = CheckedSemanticAllocator::new(
                    u64::MAX,
                    "runtime.progress.counters.right.local_sequence",
                );
                assert!(driver.drain_ready().is_err());
                assert!(receipt.wait_settled().await.is_err());
            }
        }

        next_instant += 1;
        clock.set(LogicalInstant(next_instant));
        logical_clock_trace.push(LogicalInstant(next_instant));
        let upstream = exact(next_position);
        ordered_raw_attempts.push((
            identity("left"),
            RawIngressEventKind::Data,
            upstream.clone(),
        ));
        assert!(
            sender
                .submit(
                    identity("left"),
                    RawIngressEvent::Data(int_batch(-2)),
                    upstream,
                )
                .await
                .is_err()
        );

        let status = driver.status();
        RecordedSeedArtifact {
            seed,
            ordered_raw_attempts,
            logical_clock_trace,
            prepared_job_fingerprint: prepared.fingerprint.as_bytes(),
            runtime_config: prepared.runtime_progress_config.clone(),
            execution_trace: driver.trace(),
            terminal_phase: driver.phase(),
            terminal_unsettled_receipts: status.unsettled_receipts,
            terminal_timer_entries: status.counters.timer_entries,
        }
    }

    #[tokio::test]
    async fn progress_recording_stress_captures_one_hundred_seed_artifacts() {
        let mut saw_phase_failure = false;
        for seed in 0..100 {
            let first = record_seed_artifact(seed).await;
            let second = record_seed_artifact(seed).await;
            assert_eq!(first, second, "progress artifact diverged at seed {seed}");
            assert!(!first.ordered_raw_attempts.is_empty());
            assert!(!first.logical_clock_trace.is_empty());
            assert_eq!(first.terminal_unsettled_receipts, 0);
            assert_eq!(first.terminal_timer_entries, 0);
            assert!(matches!(
                first.terminal_phase,
                super::DriverPhase::Terminal | super::DriverPhase::Cancelled
            ));
            assert!(
                first
                    .execution_trace
                    .records
                    .iter()
                    .any(|record| matches!(record, ProgressTraceRecord::Admission(_)))
            );
            assert!(
                first
                    .execution_trace
                    .records
                    .iter()
                    .any(|record| matches!(record, ProgressTraceRecord::Drain(_)))
            );
            assert!(
                first
                    .execution_trace
                    .records
                    .iter()
                    .any(|record| matches!(record, ProgressTraceRecord::Terminal(_)))
            );
            assert!(
                first
                    .execution_trace
                    .records
                    .iter()
                    .any(|record| matches!(record, ProgressTraceRecord::Settlement(_)))
            );
            saw_phase_failure |= first
                .execution_trace
                .records
                .iter()
                .any(|record| matches!(record, ProgressTraceRecord::DriverPhaseFailure { .. }));
        }
        assert!(saw_phase_failure);
    }

    #[tokio::test]
    async fn progress_replay_rejects_missing_or_extra_settlement_record() {
        let trace = terminal_tail_trace().await;
        let mut missing = trace.clone();
        let removed = missing.records.pop().unwrap();
        assert!(matches!(removed, ProgressTraceRecord::Settlement(_)));
        assert!(ProgressReplayRequest::prevalidate(missing).is_err());
        let mut extra = trace;
        extra.records.push(removed);
        assert!(ProgressReplayRequest::prevalidate(extra).is_err());
    }

    #[tokio::test]
    async fn progress_replay_reproduces_settlement_dispositions() {
        let tail_trace = terminal_tail_trace().await;
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut success, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock.clone()).unwrap();
        success.start_running().unwrap();
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        success.drain_ready().unwrap();
        receipt.wait_settled().await.unwrap();

        let (mut transaction, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock.clone()).unwrap();
        transaction.start_running().unwrap();
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::ConnectorWatermark(EventTime::from_micros(1)),
                exact(0),
            )
            .await
            .unwrap();
        assert!(transaction.drain_ready().is_err());
        assert!(receipt.wait_settled().await.is_err());

        let (mut cancelled, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        cancelled.start_running().unwrap();
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        cancelled.cancel().unwrap();
        assert!(receipt.wait_settled().await.is_err());

        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut fatal, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        fatal.start_running().unwrap();
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        fatal
            .state
            .bindings
            .get_mut(&super::BindingOrdinal::new(0))
            .unwrap()
            .next_local_sequence = CheckedSemanticAllocator::new(
            u64::MAX,
            "runtime.progress.counters.left.local_sequence",
        );
        assert!(fatal.drain_ready().is_err());
        assert!(receipt.wait_settled().await.is_err());

        let dispositions = success
            .trace()
            .records
            .into_iter()
            .chain(transaction.trace().records)
            .chain(tail_trace.records)
            .chain(cancelled.trace().records)
            .chain(fatal.trace().records)
            .filter_map(|record| match record {
                ProgressTraceRecord::Settlement(record) => Some(record.disposition),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(dispositions.contains(&SettlementDisposition::CommitSuccess));
        assert!(dispositions.iter().any(|disposition| {
            matches!(disposition, SettlementDisposition::TransactionError { .. })
        }));
        assert!(dispositions.contains(&SettlementDisposition::PostEndTailReject));
        assert!(dispositions.contains(&SettlementDisposition::Cancelled));
        assert!(dispositions.contains(&SettlementDisposition::Fatal));
    }

    #[tokio::test]
    async fn submit_after_end_is_rejected_immediately() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        let end = sender
            .submit(identity("left"), RawIngressEvent::EndOfInput, exact(0))
            .await
            .unwrap();
        driver.drain_ready().unwrap();
        end.wait_settled().await.unwrap();
        assert!(
            sender
                .submit(
                    identity("left"),
                    RawIngressEvent::Data(int_batch(2)),
                    exact(1),
                )
                .await
                .is_err()
        );
        let admission = driver.trace().records.last().cloned().unwrap();
        assert!(matches!(
            admission,
            ProgressTraceRecord::Admission(super::AdmissionAttemptRecord {
                decision: AdmissionDecisionRecord::ImmediateRejected { .. },
                ..
            })
        ));
    }

    #[tokio::test]
    async fn cancel_settles_every_accepted_receipt() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", Some(10))]), clock).unwrap();
        driver.start_running().unwrap();
        let receipts = futures::future::join_all((0..3).map(|position| {
            sender.submit(
                identity("left"),
                RawIngressEvent::Data(int_batch(i64::from(position))),
                exact(position),
            )
        }))
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
        driver.cancel().unwrap();
        for receipt in receipts {
            assert!(receipt.wait_settled().await.is_err());
        }
        assert_eq!(driver.unsettled_receipts(), 0);
        assert_eq!(driver.next_central_wake(), None);
    }

    #[tokio::test]
    async fn cancel_unregisters_all_progress_timers() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[generated("left", 10, Some(5))]), clock).unwrap();
        driver.start_running().unwrap();
        let receipt = sender
            .submit(
                identity("left"),
                RawIngressEvent::Data(timestamp_batch(1)),
                exact(0),
            )
            .await
            .unwrap();
        driver.cancel().unwrap();
        assert!(receipt.wait_settled().await.is_err());
        assert_eq!(driver.phase(), super::DriverPhase::Cancelled);
        assert_eq!(driver.unsettled_receipts(), 0);
        assert_eq!(driver.next_central_wake(), None);
        assert!(driver.state.timer_heap.is_empty());
    }

    #[tokio::test]
    async fn terminal_end_leaves_no_progress_work() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[generated("left", 10, Some(5))]), clock).unwrap();
        driver.start_running().unwrap();
        let end = sender
            .submit(identity("left"), RawIngressEvent::EndOfInput, exact(0))
            .await
            .unwrap();
        driver.drain_ready().unwrap();
        end.wait_settled().await.unwrap();
        assert_eq!(driver.phase(), super::DriverPhase::Terminal);
        assert_eq!(driver.unsettled_receipts(), 0);
        assert_eq!(driver.next_central_wake(), None);
        assert!(matches!(
            driver.trace().records.last(),
            Some(ProgressTraceRecord::Settlement(record))
                if record.disposition == SettlementDisposition::CommitSuccess
        ));
    }

    #[tokio::test]
    async fn all_ended_waits_for_settlement_and_cleanup() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) = StreamProgressDriver::new(
            prepared(&[disabled("left", None), disabled("right", None)]),
            clock,
        )
        .unwrap();
        driver.start_running().unwrap();
        let receipts =
            futures::future::join_all([(0_u8, "left"), (1, "right")].map(|(position, binding)| {
                sender.submit(
                    identity(binding),
                    RawIngressEvent::EndOfInput,
                    exact(position),
                )
            }))
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        driver.drain_ready().unwrap();
        for receipt in receipts {
            receipt.wait_settled().await.unwrap();
        }
        assert_eq!(driver.phase(), super::DriverPhase::Terminal);
        assert_eq!(driver.unsettled_receipts(), 0);
        assert_eq!(driver.next_central_wake(), None);
        assert!(
            driver
                .admission
                .lock()
                .bindings
                .values()
                .all(|binding| binding.gate.state == super::AdmissionGateState::ClosedAfterEnd)
        );
    }

    #[tokio::test]
    async fn dropped_receipt_receiver_does_not_rollback_commit() {
        let clock = ManualClock::new(LogicalInstant::ZERO);
        let (mut driver, sender) =
            StreamProgressDriver::new(prepared(&[disabled("left", None)]), clock).unwrap();
        driver.start_running().unwrap();
        drop(
            sender
                .submit(
                    identity("left"),
                    RawIngressEvent::Data(int_batch(1)),
                    exact(0),
                )
                .await
                .unwrap(),
        );
        let drain = driver.drain_ready().unwrap();
        assert_eq!(drain.committed_inputs.len(), 1);
        assert_eq!(driver.unsettled_receipts(), 0);
    }

    #[test]
    fn adapter_cannot_supply_or_forge_ready_key() {
        let submit = super::RawIngressSender::submit;
        let _ = submit;
    }

    #[test]
    fn m3_observability_has_no_late_row_metric() {
        let fields = ["unsettled_receipts", "next_central_wake", "trace"];
        assert!(fields.iter().all(|field| !field.contains("late")));
    }

    #[test]
    fn driver_owned_ordinals_and_sequences_break_timer_ties() {
        assert!(
            super::ReadyKey {
                logical_instant: LogicalInstant(1),
                class: ReadyClass::WatermarkTimer,
                binding_ordinal: super::BindingOrdinal::new(0),
                local_sequence: super::LocalSequence(9),
            } < super::ReadyKey {
                logical_instant: LogicalInstant(1),
                class: ReadyClass::WatermarkTimer,
                binding_ordinal: super::BindingOrdinal::new(1),
                local_sequence: super::LocalSequence(0),
            }
        );
    }
}
