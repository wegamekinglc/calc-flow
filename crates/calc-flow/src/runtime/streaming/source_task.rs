#[cfg(test)]
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{panic::AssertUnwindSafe, sync::Arc};

use async_trait::async_trait;
use futures::FutureExt;
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::{mpsc, watch};

use super::{
    EdgeSender, EnvelopeCost, StreamJobContext, StreamMessage,
    context::wait_for_task_gate,
    metrics::MetricsRecorder,
    progress::{
        BindingIdentity, DeclaredSchema, ExistingPrivateToggleRoute, LiveProgressCoordinator,
        NativeWatermarkCapability, PreparedSourceBinding, RawIngressEvent, RawUpstreamPosition,
        ReplayPositioningCapability, SourceBindingSpec, SourceDescriptor, WatermarkPolicy,
    },
    supervisor::{TaskFailureSignal, TaskId, TaskSupervisor, panic_message},
};
use crate::{Batch, BatchMetadata, CalcFlowError, EventTime, JsonMap, Result, canonical_json};

const MAX_CURSOR_ORDER_BYTES: usize = 16 * 1024;

/// Source-defined position with a bytewise order key and opaque JSON payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Cursor {
    order: Vec<u8>,
    payload: JsonMap,
}

impl Cursor {
    pub(crate) fn new(order: Vec<u8>, payload: JsonMap) -> Result<Self> {
        if order.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "cursor.order".into(),
                message: "must not be empty".into(),
            });
        }
        if order.len() > MAX_CURSOR_ORDER_BYTES {
            return Err(CalcFlowError::InvalidArgument {
                field: "cursor.order".into(),
                message: format!("must not exceed {MAX_CURSOR_ORDER_BYTES} bytes"),
            });
        }
        let payload_value = Value::Object(payload.clone().into_iter().collect());
        canonical_json(&payload_value).map_err(|error| CalcFlowError::InvalidArgument {
            field: "cursor.payload".into(),
            message: error.to_string(),
        })?;
        Ok(Self { order, payload })
    }

    fn is_after(&self, previous: &Self) -> bool {
        self.order > previous.order
    }

    pub(crate) fn order(&self) -> &[u8] {
        &self.order
    }
}

/// Events a connector source can return; barriers remain runtime-only.
#[derive(Clone, Debug)]
pub(crate) enum SourceEvent {
    Data { batch: Batch, cursor: Cursor },
    Watermark(EventTime),
    Idle,
}

/// Capabilities sampled before a source is opened.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SourceCapabilities {
    pub(crate) replayable: bool,
    pub(crate) max_batch_rows: usize,
    pub(crate) max_batch_bytes: usize,
}

/// Internal source lifecycle contract for M2 runtime completion.
#[async_trait]
pub(crate) trait StreamSource: Send {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()>;

    /// May block indefinitely and need not be cancellation-safe for reuse.
    /// The runtime keeps exactly one call in flight, drops it only at
    /// teardown, never polls this instance again, and then calls `close`.
    async fn next(&mut self) -> Result<Option<SourceEvent>>;

    /// Must tolerate a `next` future dropped during teardown (spec D3.5).
    async fn close(&mut self) -> Result<()>;

    fn capabilities(&self) -> SourceCapabilities;

    fn declared_schema(&self) -> DeclaredSchema {
        DeclaredSchema::DynamicOrUnknown
    }

    fn native_watermark_capability(&self) -> NativeWatermarkCapability {
        NativeWatermarkCapability::EmitsNative
    }

    fn replay_positioning_capability(&self) -> Option<ReplayPositioningCapability> {
        None
    }

    fn existing_private_watermark_toggle(&self) -> Option<ExistingPrivateToggleRoute> {
        None
    }
}

/// Test-only oracle populated at the FR10A slot-commit linearization point.
/// Each job injects its own instance, so parallel tests and reused source IDs
/// cannot share observations accidentally.
#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct AcceptedSequenceRecorder(Arc<Mutex<BTreeMap<String, Vec<u64>>>>);

#[cfg(test)]
impl AcceptedSequenceRecorder {
    pub(crate) fn record_committed_for_test(&self, source_id: &str, sequence: u64) {
        self.0
            .lock()
            .entry(source_id.to_owned())
            .or_default()
            .push(sequence);
    }

    pub(crate) fn snapshot(&self) -> BTreeMap<String, Vec<u64>> {
        self.0.lock().clone()
    }
}

/// One validated source binding. Binding identity is assigned by the job.
pub(crate) struct SourceBinding {
    source: Box<dyn StreamSource>,
    capabilities: Option<SourceCapabilities>,
    resume_cursor: Option<Cursor>,
    next_sequence: u64,
    open_began: bool,
    watermark_policy: WatermarkPolicy,
    prepared_progress: Option<PreparedSourceBinding>,
    #[cfg(test)]
    accepted_sequence_recorder: Option<AcceptedSequenceRecorder>,
}

impl SourceBinding {
    #[allow(
        clippy::unnecessary_wraps,
        reason = "preserve the validated-binding constructor contract while capability validation moves to whole-job preflight"
    )]
    pub(crate) fn new(
        source: Box<dyn StreamSource>,
        resume_cursor: Option<Cursor>,
        next_sequence: u64,
    ) -> Result<Self> {
        Ok(Self {
            source,
            capabilities: None,
            resume_cursor,
            next_sequence,
            open_began: false,
            watermark_policy: WatermarkPolicy::default(),
            prepared_progress: None,
            #[cfg(test)]
            accepted_sequence_recorder: None,
        })
    }

    pub(crate) fn with_watermark_policy(mut self, policy: WatermarkPolicy) -> Self {
        self.watermark_policy = policy;
        self
    }

    #[cfg(test)]
    pub(crate) fn with_accepted_sequence_recorder(
        mut self,
        recorder: AcceptedSequenceRecorder,
    ) -> Self {
        self.accepted_sequence_recorder = Some(recorder);
        self
    }

    pub(crate) fn sample_capabilities_once(&mut self) -> SourceCapabilities {
        *self
            .capabilities
            .get_or_insert_with(|| self.source.capabilities())
    }

    pub(crate) fn progress_spec(&self, binding_id: &str) -> Result<SourceBindingSpec> {
        let identity = BindingIdentity::new(binding_id)?;
        Ok(SourceBindingSpec {
            descriptor: SourceDescriptor::new(
                identity,
                self.source.declared_schema(),
                self.source.native_watermark_capability(),
                self.source
                    .replay_positioning_capability()
                    .unwrap_or_else(|| {
                        if self
                            .capabilities
                            .expect("progress descriptors are collected after capability sampling")
                            .replayable
                        {
                            ReplayPositioningCapability::ExactPauseReportAndSeek
                        } else {
                            ReplayPositioningCapability::Unsupported
                        }
                    }),
                self.source.existing_private_watermark_toggle(),
            ),
            watermark_policy: self.watermark_policy.clone(),
        })
    }

    pub(crate) fn install_prepared_progress(&mut self, prepared: PreparedSourceBinding) {
        self.prepared_progress = Some(prepared);
    }

    pub(crate) async fn open(&mut self) -> Result<()> {
        self.open_began = true;
        self.source.open(self.resume_cursor.clone()).await
    }

    pub(crate) async fn close(&mut self) -> Result<()> {
        self.source.close().await
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SourceProgressSnapshot {
    pub(crate) replayable: bool,
    pub(crate) latest_observed_cursor: Option<Cursor>,
    pub(crate) durable_cursor: Option<Cursor>,
    pub(crate) next_sequence: Option<u64>,
    pub(crate) ended: bool,
}

/// Separates volatile observed progress from checkpoint-durable progress.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SourceAcceptState {
    Polling,
    Slotted,
    Draining { retain_slot: bool },
    Closed,
}

struct SourceAcceptance {
    state: Mutex<SourceAcceptState>,
    drain: crate::CancellationToken,
    pump_closed: AtomicBool,
    pump_operation_failed: AtomicBool,
    close_failures: Mutex<Vec<CalcFlowError>>,
    #[cfg(test)]
    accepted_sequence: Mutex<Option<TestAcceptedSequence>>,
}

#[cfg(test)]
struct TestAcceptedSequence {
    recorder: AcceptedSequenceRecorder,
    source_id: String,
    next_sequence: Option<u64>,
}

impl SourceAcceptance {
    fn new() -> Self {
        Self {
            state: Mutex::new(SourceAcceptState::Polling),
            drain: crate::CancellationToken::new(),
            pump_closed: AtomicBool::new(false),
            pump_operation_failed: AtomicBool::new(false),
            close_failures: Mutex::new(Vec::new()),
            #[cfg(test)]
            accepted_sequence: Mutex::new(None),
        }
    }

    #[cfg(test)]
    fn install_accepted_sequence_recorder(
        &mut self,
        recorder: Option<AcceptedSequenceRecorder>,
        source_id: &str,
        next_sequence: u64,
    ) {
        *self.accepted_sequence.get_mut() = recorder.map(|recorder| TestAcceptedSequence {
            recorder,
            source_id: source_id.to_owned(),
            next_sequence: Some(next_sequence),
        });
    }

    fn request_drain(&self) {
        let mut state = self.state.lock();
        *state = match *state {
            SourceAcceptState::Polling => SourceAcceptState::Draining { retain_slot: false },
            SourceAcceptState::Slotted => SourceAcceptState::Draining { retain_slot: true },
            SourceAcceptState::Draining { retain_slot } => {
                SourceAcceptState::Draining { retain_slot }
            }
            SourceAcceptState::Closed => SourceAcceptState::Closed,
        };
        if !matches!(*state, SourceAcceptState::Closed) {
            self.drain.cancel();
        }
    }

    fn commit_slot(&self) -> bool {
        let mut state = self.state.lock();
        match *state {
            SourceAcceptState::Polling => {
                *state = SourceAcceptState::Slotted;
                true
            }
            SourceAcceptState::Slotted
            | SourceAcceptState::Draining { .. }
            | SourceAcceptState::Closed => false,
        }
    }

    #[cfg(test)]
    fn commit_event_slot(&self, event: &SourceEvent) -> bool {
        let mut state = self.state.lock();
        if !matches!(*state, SourceAcceptState::Polling) {
            return false;
        }
        *state = SourceAcceptState::Slotted;
        if matches!(event, SourceEvent::Data { .. })
            && let Some(accepted) = self.accepted_sequence.lock().as_mut()
            && let Some(sequence) = accepted.next_sequence
        {
            accepted
                .recorder
                .record_committed_for_test(&accepted.source_id, sequence);
            accepted.next_sequence = sequence.checked_add(1);
        }
        true
    }

    fn dequeue_slot(&self) {
        let mut state = self.state.lock();
        *state = match *state {
            SourceAcceptState::Slotted => SourceAcceptState::Polling,
            SourceAcceptState::Draining { retain_slot: true } => {
                SourceAcceptState::Draining { retain_slot: false }
            }
            state => state,
        };
    }

    fn is_draining(&self) -> bool {
        matches!(*self.state.lock(), SourceAcceptState::Draining { .. })
    }

    fn mark_pump_closed(&self) {
        self.pump_closed.store(true, Ordering::Release);
    }

    fn mark_pump_operation_failed(&self) {
        self.pump_operation_failed.store(true, Ordering::Release);
    }

    fn record_close_failure(&self, error: CalcFlowError) {
        self.close_failures.lock().push(error);
    }

    fn mark_closed(&self) {
        *self.state.lock() = SourceAcceptState::Closed;
    }
}

#[derive(Clone)]
pub(crate) struct SourceProgress {
    snapshot: Arc<Mutex<SourceProgressSnapshot>>,
    acceptance: Arc<SourceAcceptance>,
}

pub(crate) struct SourceCloseFailures {
    pub(crate) pump_operation_failed: bool,
    pub(crate) errors: Vec<CalcFlowError>,
}

impl std::fmt::Debug for SourceProgress {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SourceProgress")
            .field("snapshot", &*self.snapshot.lock())
            .field("acceptance", &*self.acceptance.state.lock())
            .finish()
    }
}

impl SourceProgress {
    pub(crate) fn snapshot(&self) -> SourceProgressSnapshot {
        self.snapshot.lock().clone()
    }

    pub(crate) fn request_drain(&self) {
        self.acceptance.request_drain();
    }

    pub(crate) fn accept_state(&self) -> SourceAcceptState {
        *self.acceptance.state.lock()
    }

    pub(crate) fn take_close_failures(&self) -> SourceCloseFailures {
        SourceCloseFailures {
            pump_operation_failed: self
                .acceptance
                .pump_operation_failed
                .load(Ordering::Acquire),
            errors: std::mem::take(&mut *self.acceptance.close_failures.lock()),
        }
    }
}

enum PumpEvent {
    Event(SourceEvent),
    End,
}

enum PumpCompletion {
    Cancelled,
    Draining,
    Ended,
}

struct SourceTaskInputs {
    binding_id: String,
    first_sequence: u64,
    resume_cursor: Option<Cursor>,
    outputs: Vec<EdgeSender>,
    slot: mpsc::Receiver<PumpEvent>,
    progress: SourceProgress,
    acceptance: Arc<SourceAcceptance>,
    cancellation: crate::CancellationToken,
    launch_cancel: crate::CancellationToken,
    metrics: MetricsRecorder,
    live_progress: Option<LiveSourceProgress>,
}

struct LiveSourceProgress {
    coordinator: LiveProgressCoordinator,
    binding: BindingIdentity,
}

struct SourcePumpInputs {
    source: Box<dyn StreamSource>,
    resume_cursor: Option<Cursor>,
    open_began: bool,
    slot: mpsc::Sender<PumpEvent>,
    cancellation: crate::CancellationToken,
    launch_cancel: crate::CancellationToken,
    acceptance: Arc<SourceAcceptance>,
    data_gate: watch::Receiver<bool>,
    binding_id: String,
    metrics: MetricsRecorder,
}

/// Registers the pump and source task as two owned supervised units.
pub(crate) fn spawn_source_tasks(
    supervisor: &mut TaskSupervisor,
    context: &StreamJobContext,
    binding_id: &str,
    binding: SourceBinding,
    outputs: Vec<EdgeSender>,
) -> Result<SourceProgress> {
    let (data_gate, data_gate_rx) = watch::channel(true);
    let progress = spawn_source_tasks_gated(
        supervisor,
        context,
        binding_id,
        binding,
        outputs,
        data_gate_rx,
        crate::CancellationToken::new(),
    )?;
    drop(data_gate);
    Ok(progress)
}

/// Registers a source whose connector may already have been opened by the
/// runner launch phase. Polling remains parked until the data gate opens.
pub(crate) fn spawn_source_tasks_gated(
    supervisor: &mut TaskSupervisor,
    context: &StreamJobContext,
    binding_id: &str,
    binding: SourceBinding,
    outputs: Vec<EdgeSender>,
    data_gate: watch::Receiver<bool>,
    launch_cancel: crate::CancellationToken,
) -> Result<SourceProgress> {
    spawn_source_tasks_gated_with_metrics(
        supervisor,
        context,
        binding_id,
        binding,
        outputs,
        data_gate,
        launch_cancel,
        MetricsRecorder::default(),
    )
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the private spawn boundary mirrors the source task lifecycle dependencies"
)]
pub(crate) fn spawn_source_tasks_gated_with_metrics(
    supervisor: &mut TaskSupervisor,
    context: &StreamJobContext,
    binding_id: &str,
    binding: SourceBinding,
    outputs: Vec<EdgeSender>,
    data_gate: watch::Receiver<bool>,
    launch_cancel: crate::CancellationToken,
    metrics: MetricsRecorder,
) -> Result<SourceProgress> {
    spawn_source_tasks_gated_with_optional_progress(
        supervisor,
        context,
        binding_id,
        binding,
        outputs,
        data_gate,
        launch_cancel,
        metrics,
        None,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "the private spawn boundary mirrors the source task lifecycle dependencies"
)]
pub(crate) fn spawn_source_tasks_gated_with_live_progress(
    supervisor: &mut TaskSupervisor,
    context: &StreamJobContext,
    binding_id: &str,
    binding: SourceBinding,
    data_gate: watch::Receiver<bool>,
    launch_cancel: crate::CancellationToken,
    metrics: MetricsRecorder,
    live_progress: LiveProgressCoordinator,
) -> Result<SourceProgress> {
    // Whole-job preflight already validated this binding's sampled maximum
    // batch size against its first-hop edge before those outputs moved into
    // the live progress coordinator (job::validate_source_budget).
    spawn_source_tasks_gated_with_optional_progress(
        supervisor,
        context,
        binding_id,
        binding,
        Vec::new(),
        data_gate,
        launch_cancel,
        metrics,
        Some(live_progress),
    )
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the private spawn boundary mirrors the source task lifecycle dependencies"
)]
fn spawn_source_tasks_gated_with_optional_progress(
    supervisor: &mut TaskSupervisor,
    context: &StreamJobContext,
    binding_id: &str,
    mut binding: SourceBinding,
    outputs: Vec<EdgeSender>,
    data_gate: watch::Receiver<bool>,
    launch_cancel: crate::CancellationToken,
    metrics: MetricsRecorder,
    live_progress: Option<LiveProgressCoordinator>,
) -> Result<SourceProgress> {
    let source_context = context.for_source(binding_id)?;
    let binding_id = source_context.scope_id().to_owned();
    if live_progress.is_none() {
        validate_source_outputs(&binding_id, &outputs)?;
    }
    let capabilities = binding.sample_capabilities_once();
    validate_source_capabilities(&binding_id, capabilities)?;
    validate_source_edge_budgets(&binding_id, capabilities, &outputs)?;
    let progress_binding = take_live_progress_binding(&mut binding, live_progress.is_some())?;
    let live_progress = live_progress
        .zip(progress_binding)
        .map(|(coordinator, binding)| LiveSourceProgress {
            coordinator,
            binding,
        });

    Ok(spawn_validated_source_tasks(
        supervisor,
        &source_context,
        binding_id,
        binding,
        capabilities,
        outputs,
        data_gate,
        launch_cancel,
        metrics,
        live_progress,
    ))
}

fn take_live_progress_binding(
    binding: &mut SourceBinding,
    live_progress_enabled: bool,
) -> Result<Option<BindingIdentity>> {
    if !live_progress_enabled {
        return Ok(None);
    }
    binding
        .prepared_progress
        .take()
        .map(|prepared| prepared.identity)
        .map(Some)
        .ok_or_else(|| CalcFlowError::Internal {
            message: "live source binding is missing prepared progress identity".into(),
        })
}

fn validate_source_outputs(binding_id: &str, outputs: &[EdgeSender]) -> Result<()> {
    if outputs.is_empty() {
        return Err(CalcFlowError::InvalidArgument {
            field: format!("sources.{binding_id}.outputs"),
            message: "must contain at least one edge".into(),
        });
    }
    Ok(())
}

pub(super) fn validate_source_capabilities(
    binding_id: &str,
    capabilities: SourceCapabilities,
) -> Result<()> {
    if capabilities.max_batch_rows == 0 {
        return Err(CalcFlowError::InvalidArgument {
            field: format!("sources.{binding_id}.capabilities.max_batch_rows"),
            message: "must be greater than zero".into(),
        });
    }
    if capabilities.max_batch_bytes == 0 {
        return Err(CalcFlowError::InvalidArgument {
            field: format!("sources.{binding_id}.capabilities.max_batch_bytes"),
            message: "must be greater than zero".into(),
        });
    }
    Ok(())
}

fn validate_source_edge_budgets(
    binding_id: &str,
    capabilities: SourceCapabilities,
    outputs: &[EdgeSender],
) -> Result<()> {
    for output in outputs {
        let budget = output.budget();
        if capabilities.max_batch_rows > budget.max_rows
            || capabilities.max_batch_bytes > budget.max_bytes
        {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("sources.{binding_id}.capabilities"),
                message: format!(
                    "maximum batch ({}, {} bytes) exceeds edge {:?} budget ({}, {} bytes)",
                    capabilities.max_batch_rows,
                    capabilities.max_batch_bytes,
                    output.edge(),
                    budget.max_rows,
                    budget.max_bytes
                ),
            });
        }
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "the private spawn boundary owns the already-validated source lifecycle dependencies"
)]
fn spawn_validated_source_tasks(
    supervisor: &mut TaskSupervisor,
    source_context: &super::context::StreamTaskContext,
    binding_id: String,
    binding: SourceBinding,
    capabilities: SourceCapabilities,
    outputs: Vec<EdgeSender>,
    data_gate: watch::Receiver<bool>,
    launch_cancel: crate::CancellationToken,
    metrics: MetricsRecorder,
    live_progress: Option<LiveSourceProgress>,
) -> SourceProgress {
    let SourceBinding {
        source,
        capabilities: _,
        resume_cursor,
        next_sequence,
        open_began,
        watermark_policy: _,
        prepared_progress: _,
        #[cfg(test)]
        accepted_sequence_recorder,
    } = binding;
    #[cfg(test)]
    let source_acceptance = {
        let mut acceptance = SourceAcceptance::new();
        acceptance.install_accepted_sequence_recorder(
            accepted_sequence_recorder,
            &binding_id,
            next_sequence,
        );
        acceptance
    };
    #[cfg(not(test))]
    let source_acceptance = SourceAcceptance::new();
    let acceptance = Arc::new(source_acceptance);
    let progress = SourceProgress {
        snapshot: Arc::new(Mutex::new(SourceProgressSnapshot {
            replayable: capabilities.replayable,
            latest_observed_cursor: None,
            durable_cursor: resume_cursor.clone(),
            next_sequence: Some(next_sequence),
            ended: false,
        })),
        acceptance: Arc::clone(&acceptance),
    };
    let (slot_tx, slot_rx) = mpsc::channel(1);
    let cancellation = source_context.job().cancellation().clone();
    let pump_cancellation = cancellation.clone();
    let pump_resume_cursor = resume_cursor.clone();
    let pump_acceptance = Arc::clone(&acceptance);
    let pump_launch_cancel = launch_cancel.clone();
    let pump_binding_id = binding_id.clone();
    let pump_metrics = metrics.clone();
    supervisor.spawn_with_failure_signal(
        format!("source:{binding_id}:pump"),
        move |failure_signal| async move {
            let task_id = failure_signal.task_id();
            run_source_pump(
                SourcePumpInputs {
                    source,
                    resume_cursor: pump_resume_cursor,
                    open_began,
                    slot: slot_tx,
                    cancellation: pump_cancellation,
                    launch_cancel: pump_launch_cancel,
                    acceptance: pump_acceptance,
                    data_gate,
                    binding_id: pump_binding_id,
                    metrics: pump_metrics,
                },
                task_id,
                move || failure_signal.cancel_siblings(),
            )
            .await
        },
    );
    let task_progress = progress.clone();
    supervisor.spawn_with_failure_signal(
        format!("source:{binding_id}:task"),
        move |failure_signal| async move {
            run_source_task(
                SourceTaskInputs {
                    binding_id,
                    first_sequence: next_sequence,
                    resume_cursor,
                    outputs,
                    slot: slot_rx,
                    progress: task_progress,
                    acceptance,
                    cancellation,
                    launch_cancel,
                    metrics,
                    live_progress,
                },
                failure_signal,
            )
            .await
        },
    );
    progress
}

#[allow(
    clippy::too_many_lines,
    reason = "source operation, panic containment, and exactly-once close form one serialized resource-owning lifecycle"
)]
async fn run_source_pump(
    inputs: SourcePumpInputs,
    task_id: TaskId,
    on_error: impl FnOnce(),
) -> Result<()> {
    let SourcePumpInputs {
        mut source,
        resume_cursor,
        open_began,
        slot,
        cancellation,
        launch_cancel,
        acceptance,
        mut data_gate,
        binding_id,
        metrics,
    } = inputs;
    let operation = run_source_pump_operation(
        &mut source,
        resume_cursor,
        open_began,
        &slot,
        &cancellation,
        &launch_cancel,
        &acceptance,
        &mut data_gate,
        &binding_id,
        &metrics,
        task_id,
    )
    .await;

    if operation.is_err() {
        acceptance.mark_pump_operation_failed();
        on_error();
    }
    let close_failed = match AssertUnwindSafe(source.close()).catch_unwind().await {
        Ok(Ok(())) => false,
        Ok(Err(error)) => {
            acceptance.record_close_failure(error);
            true
        }
        Err(payload) => {
            acceptance.record_close_failure(CalcFlowError::TaskPanicked {
                task_id: task_id.as_u64(),
                message: panic_message(payload.as_ref()),
            });
            true
        }
    };
    acceptance.mark_pump_closed();
    match operation {
        Ok(PumpCompletion::Cancelled | PumpCompletion::Draining) if close_failed => {
            Err(source_close_task_failed(&binding_id))
        }
        Ok(PumpCompletion::Cancelled | PumpCompletion::Draining) => Ok(()),
        Ok(PumpCompletion::Ended) => {
            if close_failed {
                return Err(source_close_task_failed(&binding_id));
            }
            if !acceptance.commit_slot() {
                return Ok(());
            }
            tokio::select! {
                biased;
                () = launch_cancel.cancelled() => Ok(()),
                () = cancellation.cancelled() => Ok(()),
                result = slot.send(PumpEvent::End) => result.map_err(|_| CalcFlowError::Internal {
                    message: "source prefetch slot closed before end-of-input delivery".into(),
                }),
            }
        }
        Err(error) => Err(error),
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the serialized pump operation receives the source's owned lifecycle resources"
)]
async fn run_source_pump_operation(
    source: &mut Box<dyn StreamSource>,
    resume_cursor: Option<Cursor>,
    open_began: bool,
    slot: &mpsc::Sender<PumpEvent>,
    cancellation: &crate::CancellationToken,
    launch_cancel: &crate::CancellationToken,
    acceptance: &SourceAcceptance,
    data_gate: &mut watch::Receiver<bool>,
    binding_id: &str,
    metrics: &MetricsRecorder,
    task_id: TaskId,
) -> Result<PumpCompletion> {
    if let Some(completion) = open_source(
        source,
        resume_cursor,
        open_began,
        cancellation,
        launch_cancel,
        task_id,
    )
    .await?
    {
        return Ok(completion);
    }
    if !wait_for_task_gate(
        data_gate,
        launch_cancel,
        cancellation,
        "source data gate closed before release",
    )
    .await?
    {
        return Ok(PumpCompletion::Cancelled);
    }
    poll_source_events(
        source,
        slot,
        cancellation,
        launch_cancel,
        acceptance,
        binding_id,
        metrics,
        task_id,
    )
    .await
}

async fn open_source(
    source: &mut Box<dyn StreamSource>,
    resume_cursor: Option<Cursor>,
    open_began: bool,
    cancellation: &crate::CancellationToken,
    launch_cancel: &crate::CancellationToken,
    task_id: TaskId,
) -> Result<Option<PumpCompletion>> {
    if open_began {
        return Ok(None);
    }
    tokio::select! {
        biased;
        () = launch_cancel.cancelled() => Ok(Some(PumpCompletion::Cancelled)),
        () = cancellation.cancelled() => Ok(Some(PumpCompletion::Cancelled)),
        result = AssertUnwindSafe(source.open(resume_cursor)).catch_unwind() => {
            match result {
                Ok(result) => result.map(|()| None),
                Err(payload) => Err(CalcFlowError::TaskPanicked {
                    task_id: task_id.as_u64(),
                    message: panic_message(payload.as_ref()),
                }),
            }
        },
    }
}

enum PumpReservation<'a> {
    Permit(mpsc::Permit<'a, PumpEvent>),
    Cancelled,
    Draining,
}

async fn reserve_pump_slot<'a>(
    slot: &'a mpsc::Sender<PumpEvent>,
    cancellation: &crate::CancellationToken,
    launch_cancel: &crate::CancellationToken,
    acceptance: &SourceAcceptance,
) -> Result<PumpReservation<'a>> {
    tokio::select! {
        biased;
        () = launch_cancel.cancelled() => Ok(PumpReservation::Cancelled),
        () = cancellation.cancelled() => Ok(PumpReservation::Cancelled),
        () = acceptance.drain.cancelled() => Ok(PumpReservation::Draining),
        permit = slot.reserve() => permit
            .map(PumpReservation::Permit)
            .map_err(|_| CalcFlowError::Internal {
                message: "source prefetch slot closed before pump convergence".into(),
            }),
    }
}

enum NextSourceEvent {
    Event(Option<SourceEvent>),
    Complete(PumpCompletion),
}

async fn next_source_event(
    source: &mut Box<dyn StreamSource>,
    cancellation: &crate::CancellationToken,
    launch_cancel: &crate::CancellationToken,
    acceptance: &SourceAcceptance,
    task_id: TaskId,
) -> Result<NextSourceEvent> {
    tokio::select! {
        biased;
        () = launch_cancel.cancelled() => Ok(NextSourceEvent::Complete(PumpCompletion::Cancelled)),
        () = cancellation.cancelled() => Ok(NextSourceEvent::Complete(PumpCompletion::Cancelled)),
        () = acceptance.drain.cancelled() => Ok(NextSourceEvent::Complete(PumpCompletion::Draining)),
        event = AssertUnwindSafe(source.next()).catch_unwind() => match event {
            Ok(event) => event.map(NextSourceEvent::Event),
            Err(payload) => Err(CalcFlowError::TaskPanicked {
                task_id: task_id.as_u64(),
                message: panic_message(payload.as_ref()),
            }),
        },
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the poll loop coordinates one source with its slot and cancellation boundaries"
)]
async fn poll_source_events(
    source: &mut Box<dyn StreamSource>,
    slot: &mpsc::Sender<PumpEvent>,
    cancellation: &crate::CancellationToken,
    launch_cancel: &crate::CancellationToken,
    acceptance: &SourceAcceptance,
    binding_id: &str,
    metrics: &MetricsRecorder,
    task_id: TaskId,
) -> Result<PumpCompletion> {
    loop {
        if acceptance.is_draining() {
            return Ok(PumpCompletion::Draining);
        }
        let permit = match reserve_pump_slot(slot, cancellation, launch_cancel, acceptance).await? {
            PumpReservation::Permit(permit) => permit,
            PumpReservation::Cancelled => return Ok(PumpCompletion::Cancelled),
            PumpReservation::Draining => return Ok(PumpCompletion::Draining),
        };
        metrics.record_source_poll(binding_id)?;
        let event =
            match next_source_event(source, cancellation, launch_cancel, acceptance, task_id)
                .await?
            {
                NextSourceEvent::Event(event) => event,
                NextSourceEvent::Complete(completion) => return Ok(completion),
            };
        let Some(event) = event else {
            return Ok(PumpCompletion::Ended);
        };
        #[cfg(test)]
        let committed = acceptance.commit_event_slot(&event);
        #[cfg(not(test))]
        let committed = acceptance.commit_slot();
        if !committed {
            return Ok(PumpCompletion::Draining);
        }
        permit.send(PumpEvent::Event(event));
    }
}

fn source_close_task_failed(binding_id: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!(
            "source task {binding_id:?} failed during close; inspect its private failure records"
        ),
    }
}

async fn run_source_task(
    mut inputs: SourceTaskInputs,
    failure_signal: TaskFailureSignal,
) -> Result<()> {
    let operation = run_source_task_loop(&mut inputs).await;
    if operation.is_err() {
        failure_signal.cancel_siblings();
    }
    operation
}

#[allow(
    clippy::too_many_lines,
    reason = "the source loop keeps ordering validation, fan-out, and terminal handling atomic"
)]
async fn run_source_task_loop(inputs: &mut SourceTaskInputs) -> Result<()> {
    let mut order = SourceOrderState {
        next_sequence: Some(inputs.first_sequence),
        last_cursor: inputs.resume_cursor.clone(),
        last_watermark: None,
    };
    loop {
        let event = match receive_pump_event(inputs).await {
            PumpReceive::Cancelled => return Ok(()),
            PumpReceive::Closed => return handle_closed_pump(inputs).await,
            PumpReceive::Event(event) => event,
        };
        inputs.acceptance.dequeue_slot();
        if process_pump_event(inputs, event, &mut order).await? == SourceLoopStep::Complete {
            return Ok(());
        }
    }
}

struct SourceOrderState {
    next_sequence: Option<u64>,
    last_cursor: Option<Cursor>,
    last_watermark: Option<EventTime>,
}

enum PumpReceive {
    Event(PumpEvent),
    Closed,
    Cancelled,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum SourceLoopStep {
    Continue,
    Complete,
}

async fn receive_pump_event(inputs: &mut SourceTaskInputs) -> PumpReceive {
    tokio::select! {
        biased;
        () = inputs.launch_cancel.cancelled() => PumpReceive::Cancelled,
        () = inputs.cancellation.cancelled() => PumpReceive::Cancelled,
        event = inputs.slot.recv() => event.map_or(PumpReceive::Closed, PumpReceive::Event),
    }
}

async fn handle_closed_pump(inputs: &mut SourceTaskInputs) -> Result<()> {
    if inputs.cancellation.is_cancelled() {
        inputs.acceptance.mark_closed();
        return Ok(());
    }
    if !inputs.acceptance.is_draining() || !inputs.acceptance.pump_closed.load(Ordering::Acquire) {
        return Err(CalcFlowError::Internal {
            message: format!(
                "source {:?} pump ended without end-of-input",
                inputs.binding_id
            ),
        });
    }
    finish_source_input(inputs).await
}

async fn process_pump_event(
    inputs: &mut SourceTaskInputs,
    event: PumpEvent,
    order: &mut SourceOrderState,
) -> Result<SourceLoopStep> {
    match event {
        PumpEvent::Event(SourceEvent::Data { batch, cursor }) => {
            process_source_data(inputs, batch, cursor, order).await
        }
        PumpEvent::Event(SourceEvent::Watermark(watermark)) => {
            process_source_watermark(inputs, watermark, order).await
        }
        PumpEvent::Event(SourceEvent::Idle) => {
            if let Some(progress) = &inputs.live_progress {
                progress
                    .coordinator
                    .submit(
                        progress.binding.clone(),
                        RawIngressEvent::ConnectorIdle,
                        raw_upstream_position(order.last_cursor.as_ref(), order.next_sequence),
                    )
                    .await?;
                return Ok(SourceLoopStep::Continue);
            }
            let sent = send_fanout(
                &mut inputs.outputs,
                StreamMessage::idle(),
                &inputs.cancellation,
            )
            .await?;
            Ok(if sent {
                SourceLoopStep::Continue
            } else {
                SourceLoopStep::Complete
            })
        }
        PumpEvent::End => finish_source_input(inputs)
            .await
            .map(|()| SourceLoopStep::Complete),
    }
}

async fn process_source_data(
    inputs: &mut SourceTaskInputs,
    batch: Batch,
    cursor: Cursor,
    order: &mut SourceOrderState,
) -> Result<SourceLoopStep> {
    validate_source_cursor(&inputs.binding_id, order.last_cursor.as_ref(), &cursor)?;
    let (sequence, message, cost) =
        sequenced_source_message(&inputs.binding_id, order.next_sequence, &batch)?;
    if let Some(progress) = &inputs.live_progress {
        inputs
            .metrics
            .record_source_data(&inputs.binding_id, cost, sequence)?;
        let sequenced = message
            .as_data()
            .expect("sequenced source message always carries data")
            .clone();
        progress
            .coordinator
            .submit(
                progress.binding.clone(),
                RawIngressEvent::Data(sequenced),
                data_upstream_position(&cursor, sequence)?,
            )
            .await?;
        inputs
            .metrics
            .record_source_output(&inputs.binding_id, cost)?;
        record_source_progress(inputs, order, cursor, sequence);
        return Ok(SourceLoopStep::Continue);
    }
    if !send_source_data(inputs, message, cost, sequence).await? {
        return Ok(SourceLoopStep::Complete);
    }
    record_source_progress(inputs, order, cursor, sequence);
    Ok(SourceLoopStep::Continue)
}

fn sequenced_source_message(
    binding_id: &str,
    next_sequence: Option<u64>,
    batch: &Batch,
) -> Result<(u64, StreamMessage, EnvelopeCost)> {
    let sequence = next_sequence.ok_or_else(|| CalcFlowError::Internal {
        message: format!("source binding {binding_id:?} sequence is exhausted"),
    })?;
    let metadata = BatchMetadata::new(binding_id, sequence, batch.metadata().attributes().clone())?;
    let message = StreamMessage::data(batch.with_metadata(metadata));
    let cost = EnvelopeCost::of_message(&message)?;
    Ok((sequence, message, cost))
}

async fn send_source_data(
    inputs: &mut SourceTaskInputs,
    message: StreamMessage,
    cost: EnvelopeCost,
    sequence: u64,
) -> Result<bool> {
    inputs
        .metrics
        .record_source_data(&inputs.binding_id, cost, sequence)?;
    if !send_fanout(&mut inputs.outputs, message, &inputs.cancellation).await? {
        return Ok(false);
    }
    inputs
        .metrics
        .record_source_output(&inputs.binding_id, cost)?;
    Ok(true)
}

fn record_source_progress(
    inputs: &SourceTaskInputs,
    order: &mut SourceOrderState,
    cursor: Cursor,
    sequence: u64,
) {
    order.next_sequence = sequence.checked_add(1);
    order.last_cursor = Some(cursor.clone());
    let mut snapshot = inputs.progress.snapshot.lock();
    snapshot.latest_observed_cursor = Some(cursor);
    snapshot.next_sequence = order.next_sequence;
}

fn validate_source_cursor(
    binding_id: &str,
    previous: Option<&Cursor>,
    cursor: &Cursor,
) -> Result<()> {
    if previous.is_some_and(|previous| !cursor.is_after(previous)) {
        return Err(CalcFlowError::InvalidArgument {
            field: format!("sources.{binding_id}.cursor"),
            message: format!(
                "source binding {binding_id:?} emitted a repeated or regressed cursor"
            ),
        });
    }
    Ok(())
}

async fn process_source_watermark(
    inputs: &mut SourceTaskInputs,
    watermark: EventTime,
    order: &mut SourceOrderState,
) -> Result<SourceLoopStep> {
    if let Some(progress) = &inputs.live_progress {
        progress
            .coordinator
            .submit(
                progress.binding.clone(),
                RawIngressEvent::ConnectorWatermark(watermark),
                raw_upstream_position(order.last_cursor.as_ref(), order.next_sequence),
            )
            .await?;
        order.last_watermark = Some(watermark);
        return Ok(SourceLoopStep::Continue);
    }
    if order
        .last_watermark
        .is_some_and(|previous| watermark < previous)
    {
        return Err(CalcFlowError::InvalidArgument {
            field: format!("sources.{}.watermark", inputs.binding_id),
            message: format!(
                "source binding {:?} regressed its watermark",
                inputs.binding_id
            ),
        });
    }
    let sent = send_fanout(
        &mut inputs.outputs,
        StreamMessage::watermark(watermark),
        &inputs.cancellation,
    )
    .await?;
    order.last_watermark = Some(watermark);
    Ok(if sent {
        SourceLoopStep::Continue
    } else {
        SourceLoopStep::Complete
    })
}

async fn finish_source_input(inputs: &mut SourceTaskInputs) -> Result<()> {
    if let Some(progress) = &inputs.live_progress {
        let snapshot = inputs.progress.snapshot.lock().clone();
        progress
            .coordinator
            .submit(
                progress.binding.clone(),
                RawIngressEvent::EndOfInput,
                end_upstream_position(&snapshot),
            )
            .await?;
        inputs.metrics.record_source_end(&inputs.binding_id)?;
        inputs.progress.snapshot.lock().ended = true;
        inputs.acceptance.mark_closed();
        return Ok(());
    }
    if send_fanout(
        &mut inputs.outputs,
        StreamMessage::end_of_input(),
        &inputs.cancellation,
    )
    .await?
    {
        inputs.metrics.record_source_end(&inputs.binding_id)?;
        inputs.progress.snapshot.lock().ended = true;
    }
    inputs.acceptance.mark_closed();
    Ok(())
}

fn end_upstream_position(snapshot: &SourceProgressSnapshot) -> RawUpstreamPosition {
    raw_upstream_position(
        snapshot
            .latest_observed_cursor
            .as_ref()
            .or(snapshot.durable_cursor.as_ref()),
        snapshot.next_sequence,
    )
}

fn data_upstream_position(cursor: &Cursor, sequence: u64) -> Result<RawUpstreamPosition> {
    let control_frontier =
        sequence
            .checked_add(1)
            .ok_or_else(|| CalcFlowError::InvalidArgument {
                field: "runtime.progress.source.control_frontier".into(),
                message: "data control frontier exhausted before progress admission".into(),
            })?;
    Ok(raw_upstream_position(Some(cursor), Some(control_frontier)))
}

fn raw_upstream_position(
    cursor: Option<&Cursor>,
    control_sequence: Option<u64>,
) -> RawUpstreamPosition {
    match (cursor, control_sequence) {
        (Some(cursor), Some(control_sequence)) => RawUpstreamPosition::Exact {
            delivery_replay_cursor: cursor.order().to_vec(),
            control_frontier: control_sequence.to_be_bytes().to_vec(),
        },
        _ => RawUpstreamPosition::Unavailable,
    }
}

async fn send_fanout(
    outputs: &mut [EdgeSender],
    message: StreamMessage,
    cancellation: &crate::CancellationToken,
) -> Result<bool> {
    for output in outputs {
        let sent = tokio::select! {
            biased;
            () = cancellation.cancelled() => return Ok(false),
            result = output.send(message.clone()) => result,
        };
        sent?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use std::{
        any::Any,
        collections::{BTreeMap, VecDeque},
        future::{Future, pending, poll_fn},
        pin::Pin,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        task::Poll,
    };

    use async_trait::async_trait;
    use futures::task::AtomicWaker;
    use parking_lot::Mutex;
    use serde_json::json;
    use tokio::sync::{Notify, mpsc, oneshot, watch};

    use super::{
        AcceptedSequenceRecorder, Cursor, SourceAcceptState, SourceAcceptance, SourceBinding,
        SourceCapabilities, SourceEvent, SourceProgressSnapshot, SourcePumpInputs, StreamSource,
        data_upstream_position, end_upstream_position, run_source_pump, spawn_source_tasks,
        spawn_source_tasks_gated_with_metrics, take_live_progress_binding,
    };
    use crate::{
        Batch, BatchMetadata, CalcFlowError, CancellationToken, EdgeBudget, EdgeReceiver,
        EventTime, ExternalPayload, JsonMap, Result, StreamJobContext, StreamMessage,
        StreamMessageKind, edge_channel,
        runtime::streaming::{
            metrics::MetricsRecorder,
            progress::RawUpstreamPosition,
            supervisor::{TaskId, TaskSupervisor},
        },
    };

    #[derive(Debug)]
    struct TestPayload;

    #[test]
    fn live_progress_requires_prepared_binding_before_spawn() {
        let mut binding =
            SourceBinding::new(Box::new(StepSource::new(std::iter::empty())), None, 0).unwrap();

        let error = take_live_progress_binding(&mut binding, true).unwrap_err();

        assert!(matches!(
            error,
            CalcFlowError::Internal { message }
                if message == "live source binding is missing prepared progress identity"
        ));
    }

    #[test]
    fn resumed_source_eof_uses_the_durable_upstream_position() {
        let snapshot = SourceProgressSnapshot {
            replayable: true,
            latest_observed_cursor: None,
            durable_cursor: Some(cursor(4)),
            next_sequence: Some(7),
            ended: false,
        };

        assert_eq!(
            end_upstream_position(&snapshot),
            RawUpstreamPosition::Exact {
                delivery_replay_cursor: vec![4],
                control_frontier: 7_u64.to_be_bytes().to_vec(),
            }
        );
    }

    #[test]
    fn accepted_data_advances_the_upstream_control_frontier() {
        assert_eq!(
            data_upstream_position(&cursor(4), 7).unwrap(),
            RawUpstreamPosition::Exact {
                delivery_replay_cursor: vec![4],
                control_frontier: 8_u64.to_be_bytes().to_vec(),
            }
        );
    }

    #[test]
    fn data_frontier_exhaustion_fails_before_progress_admission() {
        let error = data_upstream_position(&cursor(4), u64::MAX).unwrap_err();
        assert!(error.to_string().contains("control frontier"));
    }

    impl ExternalPayload for TestPayload {
        fn backend(&self) -> &'static str {
            "m2-source-test"
        }

        fn len(&self) -> usize {
            1
        }

        fn estimated_bytes(&self) -> usize {
            1
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn batch(attribute: u64) -> Batch {
        Batch::external(
            Arc::new(TestPayload),
            BatchMetadata::new(
                "connector-owned",
                99,
                BTreeMap::from([("attribute".into(), json!(attribute))]),
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn cursor(position: u8) -> Cursor {
        Cursor::new(
            vec![position],
            BTreeMap::from([("position".into(), json!(position))]),
        )
        .unwrap()
    }

    fn context(cancellation: CancellationToken) -> StreamJobContext {
        StreamJobContext::new(7, "fingerprint", JsonMap::new(), None, cancellation)
    }

    #[test]
    fn accepted_sequence_recorder_observes_only_the_winning_slot_commit() {
        let recorder = AcceptedSequenceRecorder::default();
        let isolated = AcceptedSequenceRecorder::default();
        let mut acceptance = SourceAcceptance::new();
        acceptance.install_accepted_sequence_recorder(Some(recorder.clone()), "left", 7);
        let event = SourceEvent::Data {
            batch: batch(7),
            cursor: cursor(7),
        };

        assert!(acceptance.commit_event_slot(&event));
        acceptance.request_drain();
        assert!(!acceptance.commit_event_slot(&event));

        assert_eq!(
            recorder.snapshot(),
            BTreeMap::from([("left".into(), vec![7])])
        );
        assert!(isolated.snapshot().is_empty());
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum OpenObservation {
        NotOpened,
        Opened(Option<Cursor>),
    }

    struct StepSource {
        events: VecDeque<Result<Option<SourceEvent>>>,
        capabilities: SourceCapabilities,
        next_calls: Option<mpsc::UnboundedSender<usize>>,
        next_count: usize,
        opened_at: Arc<Mutex<OpenObservation>>,
        closed: Arc<AtomicBool>,
    }

    struct ControlledNextState {
        next_calls: AtomicUsize,
        next_entered: AtomicBool,
        ready: AtomicBool,
        value: Mutex<Option<SourceEvent>>,
        waker: AtomicWaker,
        dropped_polls: AtomicUsize,
        close_started: AtomicBool,
        block_close: AtomicBool,
        close_release: Notify,
        closed: AtomicBool,
    }

    impl ControlledNextState {
        fn new(block_close: bool) -> Arc<Self> {
            Arc::new(Self {
                next_calls: AtomicUsize::new(0),
                next_entered: AtomicBool::new(false),
                ready: AtomicBool::new(false),
                value: Mutex::new(None),
                waker: AtomicWaker::new(),
                dropped_polls: AtomicUsize::new(0),
                close_started: AtomicBool::new(false),
                block_close: AtomicBool::new(block_close),
                close_release: Notify::new(),
                closed: AtomicBool::new(false),
            })
        }

        fn release(&self, value: Option<SourceEvent>) {
            *self.value.lock() = value;
            self.ready.store(true, Ordering::Release);
            self.waker.wake();
        }
    }

    struct ControlledNextFuture {
        state: Arc<ControlledNextState>,
        completed: bool,
    }

    impl Future for ControlledNextFuture {
        type Output = Result<Option<SourceEvent>>;

        fn poll(
            mut self: Pin<&mut Self>,
            context: &mut std::task::Context<'_>,
        ) -> Poll<Self::Output> {
            self.state.next_entered.store(true, Ordering::Release);
            self.state.waker.register(context.waker());
            if self.state.ready.swap(false, Ordering::AcqRel) {
                let value = self.state.value.lock().take();
                self.completed = true;
                Poll::Ready(Ok(value))
            } else {
                Poll::Pending
            }
        }
    }

    impl Drop for ControlledNextFuture {
        fn drop(&mut self) {
            if !self.completed {
                self.state.dropped_polls.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    struct ControlledSource(Arc<ControlledNextState>);

    #[async_trait]
    impl StreamSource for ControlledSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            self.0.next_calls.fetch_add(1, Ordering::SeqCst);
            ControlledNextFuture {
                state: Arc::clone(&self.0),
                completed: false,
            }
            .await
        }

        async fn close(&mut self) -> Result<()> {
            self.0.close_started.store(true, Ordering::Release);
            if self.0.block_close.load(Ordering::Acquire) {
                self.0.close_release.notified().await;
            }
            self.0.closed.store(true, Ordering::Release);
            Ok(())
        }

        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replayable: true,
                max_batch_rows: 1,
                max_batch_bytes: 1,
            }
        }
    }

    impl StepSource {
        fn new(events: impl IntoIterator<Item = Result<Option<SourceEvent>>>) -> Self {
            Self {
                events: events.into_iter().collect(),
                capabilities: SourceCapabilities {
                    replayable: true,
                    max_batch_rows: 1,
                    max_batch_bytes: 1,
                },
                next_calls: None,
                next_count: 0,
                opened_at: Arc::new(Mutex::new(OpenObservation::NotOpened)),
                closed: Arc::new(AtomicBool::new(false)),
            }
        }

        fn with_next_calls(mut self, next_calls: mpsc::UnboundedSender<usize>) -> Self {
            self.next_calls = Some(next_calls);
            self
        }

        fn with_capabilities(mut self, capabilities: SourceCapabilities) -> Self {
            self.capabilities = capabilities;
            self
        }
    }

    #[async_trait]
    impl StreamSource for StepSource {
        async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
            *self.opened_at.lock() = OpenObservation::Opened(cursor);
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            self.next_count += 1;
            if let Some(next_calls) = &self.next_calls {
                next_calls.send(self.next_count).unwrap();
            }
            self.events.pop_front().unwrap_or(Ok(None))
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn capabilities(&self) -> SourceCapabilities {
            self.capabilities
        }
    }

    async fn assert_pending<T>(future: &mut (impl Future<Output = T> + Unpin)) {
        poll_fn(|context| match Pin::new(&mut *future).poll(context) {
            Poll::Pending => Poll::Ready(()),
            Poll::Ready(_) => panic!("future unexpectedly completed"),
        })
        .await;
    }

    #[test]
    fn zero_source_capability_is_rejected_before_open() {
        let source = StepSource::new([]).with_capabilities(SourceCapabilities {
            replayable: true,
            max_batch_rows: 0,
            max_batch_bytes: 1,
        });
        let opened_at = Arc::clone(&source.opened_at);

        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let (sender, _receiver) = edge_channel(
            "source->output",
            EdgeBudget {
                max_rows: 1,
                max_bytes: 1,
            },
        )
        .unwrap();

        let result = spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "input",
            binding,
            vec![sender],
        );

        assert!(matches!(
            result,
            Err(CalcFlowError::InvalidArgument { ref field, .. })
                if field == "sources.input.capabilities.max_batch_rows"
        ));
        assert_eq!(*opened_at.lock(), OpenObservation::NotOpened);
    }

    #[test]
    fn downstream_budget_is_rejected_before_open_or_task_registration() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let source = StepSource::new([]).with_capabilities(SourceCapabilities {
            replayable: true,
            max_batch_rows: 2,
            max_batch_bytes: 1,
        });
        let opened_at = Arc::clone(&source.opened_at);
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, _receiver) = edge_channel(
            "source->too-small",
            EdgeBudget {
                max_rows: 1,
                max_bytes: 1,
            },
        )
        .unwrap();

        let result = spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "input",
            binding,
            vec![sender],
        );

        assert!(matches!(
            result,
            Err(CalcFlowError::InvalidArgument { ref field, .. })
                if field == "sources.input.capabilities"
        ));
        assert_eq!(*opened_at.lock(), OpenObservation::NotOpened);
        assert_eq!(supervisor.task_count(), 0);
    }

    #[tokio::test]
    async fn data_metadata_cursor_fanout_and_end_are_ordered() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let source = StepSource::new([
            Ok(Some(SourceEvent::Data {
                batch: batch(10),
                cursor: cursor(1),
            })),
            Ok(Some(SourceEvent::Data {
                batch: batch(20),
                cursor: cursor(2),
            })),
            Ok(None),
        ]);
        let opened_at = Arc::clone(&source.opened_at);
        let closed = Arc::clone(&source.closed);
        let resume = cursor(0);
        let binding = SourceBinding::new(Box::new(source), Some(resume.clone()), 5).unwrap();
        let (left_tx, mut left_rx) = edge_channel("source->left", EdgeBudget::default()).unwrap();
        let (right_tx, mut right_rx) =
            edge_channel("source->right", EdgeBudget::default()).unwrap();

        let progress = spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "orders",
            binding,
            vec![left_tx, right_tx],
        )
        .unwrap();
        let report = supervisor.join_all().await;

        assert!(report.errors.is_empty());
        assert_eq!(*opened_at.lock(), OpenObservation::Opened(Some(resume)));
        assert!(closed.load(Ordering::SeqCst));
        for receiver in [&mut left_rx, &mut right_rx] {
            for (sequence, attribute) in [(5, 10), (6, 20)] {
                let message = receiver.recv().await.unwrap().unwrap();
                let data = message.as_data().unwrap();
                assert_eq!(data.metadata().source(), "orders");
                assert_eq!(data.metadata().sequence(), sequence);
                assert_eq!(data.metadata().attributes()["attribute"], json!(attribute));
            }
            assert_eq!(
                receiver.recv().await.unwrap().unwrap().kind(),
                StreamMessageKind::EndOfInput
            );
            assert!(receiver.recv().await.unwrap().is_none());
        }
        let snapshot = progress.snapshot();
        assert_eq!(snapshot.latest_observed_cursor, Some(cursor(2)));
        assert_eq!(snapshot.durable_cursor, Some(cursor(0)));
        assert_eq!(snapshot.next_sequence, Some(7));
        assert!(snapshot.ended);
    }

    #[tokio::test]
    async fn source_fanout_keeps_the_first_branch_prefix_and_exact_edge_metrics() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let source = StepSource::new([Ok(Some(SourceEvent::Data {
            batch: batch(10),
            cursor: cursor(1),
        }))]);
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let metrics = MetricsRecorder::new(
            [
                ("source->left".into(), EdgeBudget::new(1, 1).unwrap()),
                ("source->right".into(), EdgeBudget::new(1, 1).unwrap()),
            ],
            ["input".into()],
            [],
            [],
        );
        let budget = EdgeBudget {
            max_rows: 1,
            max_bytes: 1,
        };
        let (left_tx, mut left_rx) = crate::runtime::streaming::channel::edge_channel_with_metrics(
            "source->left",
            budget,
            metrics.clone(),
        )
        .unwrap();
        let (right_tx, mut right_rx) =
            crate::runtime::streaming::channel::edge_channel_with_metrics(
                "source->right",
                budget,
                metrics.clone(),
            )
            .unwrap();
        right_rx.close();
        let (_data_gate, data_gate) = watch::channel(true);

        spawn_source_tasks_gated_with_metrics(
            &mut supervisor,
            &context(cancellation),
            "input",
            binding,
            vec![left_tx, right_tx],
            data_gate,
            CancellationToken::new(),
            metrics.clone(),
        )
        .unwrap();

        let report = supervisor.join_all().await;

        assert_eq!(report.errors.len(), 1);
        assert!(matches!(
            report.errors[0].error,
            CalcFlowError::EdgeClosed { ref edge } if edge == "source->right"
        ));
        let prefix = left_rx.recv().await.unwrap().unwrap();
        assert_eq!(prefix.as_data().unwrap().metadata().sequence(), 0);
        assert!(left_rx.recv().await.unwrap().is_none());
        assert!(right_rx.recv().await.unwrap().is_none());
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.edges["source->left"].input_batches, 1);
        assert_eq!(snapshot.edges["source->right"].input_batches, 0);
        assert_eq!(snapshot.sources["input"].data_batches, 1);
        assert_eq!(snapshot.sources["input"].fully_fanned_out_batches, 0);
    }

    #[tokio::test]
    async fn full_edge_stops_source_polling_after_the_single_prefetch_slot() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let (calls_tx, mut calls_rx) = mpsc::unbounded_channel();
        let source = StepSource::new((1..=4).map(|position| {
            Ok(Some(SourceEvent::Data {
                batch: batch(position.into()),
                cursor: cursor(position),
            }))
        }))
        .with_next_calls(calls_tx);
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, mut receiver) = edge_channel(
            "source->slow",
            EdgeBudget {
                max_rows: 1,
                max_bytes: 1,
            },
        )
        .unwrap();

        spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "input",
            binding,
            vec![sender],
        )
        .unwrap();

        assert_eq!(calls_rx.recv().await, Some(1));
        assert_eq!(calls_rx.recv().await, Some(2));
        assert_eq!(calls_rx.recv().await, Some(3));
        let mut fourth_call = Box::pin(calls_rx.recv());
        assert_pending(&mut fourth_call).await;
        drop(fourth_call);

        assert_eq!(
            receiver
                .recv()
                .await
                .unwrap()
                .unwrap()
                .as_data()
                .unwrap()
                .metadata()
                .sequence(),
            0
        );
        assert_eq!(calls_rx.recv().await, Some(4));

        supervisor.cancel();
        assert!(supervisor.join_all().await.errors.is_empty());
    }

    #[tokio::test]
    async fn graceful_drain_includes_the_committed_slot_and_pending_edge_send() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let (calls_tx, mut calls_rx) = mpsc::unbounded_channel();
        let source = StepSource::new((1..=4).map(|position| {
            Ok(Some(SourceEvent::Data {
                batch: batch(position.into()),
                cursor: cursor(position),
            }))
        }))
        .with_next_calls(calls_tx);
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, mut receiver) = edge_channel(
            "source->draining",
            EdgeBudget {
                max_rows: 1,
                max_bytes: 1,
            },
        )
        .unwrap();
        let progress = spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "input",
            binding,
            vec![sender],
        )
        .unwrap();
        for expected in 1..=3 {
            assert_eq!(calls_rx.recv().await, Some(expected));
        }

        progress.request_drain();

        for sequence in 0..=2 {
            assert_eq!(
                receiver
                    .recv()
                    .await
                    .unwrap()
                    .unwrap()
                    .as_data()
                    .unwrap()
                    .metadata()
                    .sequence(),
                sequence
            );
        }
        assert_eq!(
            receiver.recv().await.unwrap().unwrap().kind(),
            StreamMessageKind::EndOfInput
        );
        assert!(receiver.recv().await.unwrap().is_none());
        assert!(supervisor.join_all().await.errors.is_empty());
        assert_eq!(
            calls_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        );
        assert!(progress.snapshot().ended);
    }

    #[test]
    fn drain_cut_and_slot_commit_have_one_serialized_winner() {
        let cut_wins = SourceAcceptance::new();
        cut_wins.request_drain();
        assert!(!cut_wins.commit_slot());
        assert_eq!(
            *cut_wins.state.lock(),
            SourceAcceptState::Draining { retain_slot: false }
        );

        let commit_wins = SourceAcceptance::new();
        assert!(commit_wins.commit_slot());
        commit_wins.request_drain();
        assert_eq!(
            *commit_wins.state.lock(),
            SourceAcceptState::Draining { retain_slot: true }
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum UncommittedCut {
        BeforePoll,
        PendingPoll,
        ReadyDataBeforeRuntimePoll,
        ReadyNoneBeforeRuntimePoll,
        NoneObservedWhileClosePending,
    }

    async fn wait_for_cut(state: &ControlledNextState, cut: UncommittedCut) {
        let observed = match cut {
            UncommittedCut::BeforePoll => return,
            UncommittedCut::PendingPoll
            | UncommittedCut::ReadyDataBeforeRuntimePoll
            | UncommittedCut::ReadyNoneBeforeRuntimePoll => &state.next_entered,
            UncommittedCut::NoneObservedWhileClosePending => &state.close_started,
        };
        for _ in 0..100 {
            if observed.load(Ordering::Acquire) {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(observed.load(Ordering::Acquire), "{cut:?}");
    }

    async fn collect_kinds(receiver: &mut EdgeReceiver) -> Vec<StreamMessageKind> {
        let mut kinds = Vec::new();
        while let Some(message) = receiver.recv().await.unwrap() {
            kinds.push(message.kind());
        }
        kinds
    }

    async fn assert_uncommitted_cut(terminal_is_cancel: bool, cut: UncommittedCut) {
        let cancellation = CancellationToken::new();
        let state =
            ControlledNextState::new(matches!(cut, UncommittedCut::NoneObservedWhileClosePending));
        if matches!(cut, UncommittedCut::NoneObservedWhileClosePending) {
            state.release(None);
        }
        let binding =
            SourceBinding::new(Box::new(ControlledSource(Arc::clone(&state))), None, 0).unwrap();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let (sender, mut receiver) = edge_channel(
            "source->cut",
            EdgeBudget {
                max_rows: 1,
                max_bytes: 1,
            },
        )
        .unwrap();
        let data_initially_open = !matches!(cut, UncommittedCut::BeforePoll);
        let (data_gate, data_gate_rx) = watch::channel(data_initially_open);
        let progress = super::spawn_source_tasks_gated(
            &mut supervisor,
            &context(cancellation.clone()),
            "input",
            binding,
            vec![sender],
            data_gate_rx,
            CancellationToken::new(),
        )
        .unwrap();

        wait_for_cut(&state, cut).await;
        match cut {
            UncommittedCut::ReadyDataBeforeRuntimePoll => state.release(Some(SourceEvent::Data {
                batch: batch(1),
                cursor: cursor(1),
            })),
            UncommittedCut::ReadyNoneBeforeRuntimePoll => state.release(None),
            _ => {}
        }

        if terminal_is_cancel {
            cancellation.cancel();
        } else {
            progress.request_drain();
        }
        if !data_initially_open {
            data_gate.send(true).unwrap();
        }
        if matches!(cut, UncommittedCut::NoneObservedWhileClosePending) {
            state.close_release.notify_waiters();
        }

        let report = supervisor.join_all().await;
        assert!(report.errors.is_empty(), "{cut:?}: {report:?}");
        let kinds = collect_kinds(&mut receiver).await;
        if terminal_is_cancel {
            assert!(
                !kinds.contains(&StreamMessageKind::Data),
                "uncommitted data escaped explicit cancel at {cut:?}: {kinds:?}"
            );
        } else {
            assert_eq!(kinds, [StreamMessageKind::EndOfInput], "{cut:?}");
            assert_eq!(
                progress.accept_state(),
                SourceAcceptState::Closed,
                "{cut:?}"
            );
        }
        assert!(state.closed.load(Ordering::Acquire), "{cut:?}");
        assert_eq!(supervisor.task_count(), 0, "{cut:?}");
        let expected_drops = usize::from(matches!(
            cut,
            UncommittedCut::PendingPoll
                | UncommittedCut::ReadyDataBeforeRuntimePoll
                | UncommittedCut::ReadyNoneBeforeRuntimePoll
        ));
        assert_eq!(
            state.dropped_polls.load(Ordering::SeqCst),
            expected_drops,
            "{cut:?}"
        );
    }

    #[tokio::test]
    async fn uncommitted_drain_and_cancel_cut_matrix_is_deterministic() {
        for terminal_is_cancel in [false, true] {
            for cut in [
                UncommittedCut::BeforePoll,
                UncommittedCut::PendingPoll,
                UncommittedCut::ReadyDataBeforeRuntimePoll,
                UncommittedCut::ReadyNoneBeforeRuntimePoll,
                UncommittedCut::NoneObservedWhileClosePending,
            ] {
                assert_uncommitted_cut(terminal_is_cancel, cut).await;
            }
        }
    }

    #[tokio::test]
    async fn terminal_after_edge_enqueue_drains_only_for_graceful_shutdown() {
        for terminal_is_cancel in [false, true] {
            let cancellation = CancellationToken::new();
            let state = ControlledNextState::new(false);
            state.release(Some(SourceEvent::Data {
                batch: batch(7),
                cursor: cursor(1),
            }));
            let binding =
                SourceBinding::new(Box::new(ControlledSource(Arc::clone(&state))), None, 0)
                    .unwrap();
            let mut supervisor = TaskSupervisor::new(cancellation.clone());
            let (sender, mut receiver) = edge_channel(
                "source->enqueued",
                EdgeBudget {
                    max_rows: 1,
                    max_bytes: 1,
                },
            )
            .unwrap();
            let progress = spawn_source_tasks(
                &mut supervisor,
                &context(cancellation.clone()),
                "input",
                binding,
                vec![sender],
            )
            .unwrap();
            let first = receiver.recv().await.unwrap().unwrap();
            assert_eq!(first.kind(), StreamMessageKind::Data);
            for _ in 0..100 {
                if state.next_calls.load(Ordering::SeqCst) >= 2 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert_eq!(state.next_calls.load(Ordering::SeqCst), 2);

            if terminal_is_cancel {
                cancellation.cancel();
            } else {
                progress.request_drain();
            }

            let report = supervisor.join_all().await;
            assert!(report.errors.is_empty(), "{report:?}");
            let mut kinds = Vec::new();
            while let Some(message) = receiver.recv().await.unwrap() {
                kinds.push(message.kind());
            }
            if terminal_is_cancel {
                assert!(kinds.is_empty());
            } else {
                assert_eq!(kinds, [StreamMessageKind::EndOfInput]);
                assert_eq!(progress.accept_state(), SourceAcceptState::Closed);
            }
            assert!(state.closed.load(Ordering::Acquire));
            assert_eq!(supervisor.task_count(), 0);
        }
    }

    #[tokio::test]
    async fn terminal_while_edge_send_is_pending_is_drain_or_discard_by_cause() {
        for terminal_is_cancel in [false, true] {
            let cancellation = CancellationToken::new();
            let state = ControlledNextState::new(false);
            state.release(Some(SourceEvent::Data {
                batch: batch(8),
                cursor: cursor(1),
            }));
            let binding =
                SourceBinding::new(Box::new(ControlledSource(Arc::clone(&state))), None, 0)
                    .unwrap();
            let mut supervisor = TaskSupervisor::new(cancellation.clone());
            let (mut sender, mut receiver) = edge_channel(
                "source->pending-send",
                EdgeBudget {
                    max_rows: 1,
                    max_bytes: 1,
                },
            )
            .unwrap();
            sender.send(StreamMessage::data(batch(99))).await.unwrap();
            let progress = spawn_source_tasks(
                &mut supervisor,
                &context(cancellation.clone()),
                "input",
                binding,
                vec![sender],
            )
            .unwrap();
            for _ in 0..100 {
                if state.next_calls.load(Ordering::SeqCst) >= 2 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert_eq!(state.next_calls.load(Ordering::SeqCst), 2);

            if terminal_is_cancel {
                cancellation.cancel();
            } else {
                progress.request_drain();
            }
            let prefill = receiver.recv().await.unwrap().unwrap();
            assert_eq!(prefill.as_data().unwrap().metadata().sequence(), 99);
            let mut join = Box::pin(supervisor.join_all());
            let mut delivered = Vec::new();
            let report = loop {
                tokio::select! {
                    report = &mut join => break report,
                    message = receiver.recv() => match message.unwrap() {
                        Some(message) => delivered.push(message),
                        None => break (&mut join).await,
                    },
                }
            };
            drop(join);
            assert!(report.errors.is_empty(), "{report:?}");
            while let Some(message) = receiver.recv().await.unwrap() {
                delivered.push(message);
            }
            if terminal_is_cancel {
                assert!(delivered.is_empty());
            } else {
                assert_eq!(delivered.len(), 2);
                assert_eq!(delivered[0].as_data().unwrap().metadata().sequence(), 0);
                assert_eq!(delivered[1].kind(), StreamMessageKind::EndOfInput);
                assert_eq!(progress.accept_state(), SourceAcceptState::Closed);
            }
            assert!(state.closed.load(Ordering::Acquire));
            assert_eq!(supervisor.task_count(), 0);
        }
    }

    struct PollGuard(Arc<AtomicBool>);

    impl Drop for PollGuard {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    struct BlockingOpenSource {
        open_poll_dropped: Arc<AtomicBool>,
        closed: Arc<AtomicBool>,
    }

    #[async_trait]
    impl StreamSource for BlockingOpenSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            let _guard = PollGuard(Arc::clone(&self.open_poll_dropped));
            pending().await
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            panic!("next must not run before open completes");
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replayable: true,
                max_batch_rows: 1,
                max_batch_bytes: 1,
            }
        }
    }

    #[tokio::test]
    async fn cancel_drops_a_blocked_open_then_closes_source() {
        let cancellation = CancellationToken::new();
        let open_poll_dropped = Arc::new(AtomicBool::new(false));
        let closed = Arc::new(AtomicBool::new(false));
        let source = BlockingOpenSource {
            open_poll_dropped: Arc::clone(&open_poll_dropped),
            closed: Arc::clone(&closed),
        };
        let (slot_tx, _slot_rx) = mpsc::channel(1);
        let mut pump = Box::pin(run_source_pump(
            SourcePumpInputs {
                source: Box::new(source),
                resume_cursor: None,
                open_began: false,
                slot: slot_tx,
                cancellation: cancellation.clone(),
                launch_cancel: CancellationToken::new(),
                acceptance: Arc::new(SourceAcceptance::new()),
                data_gate: watch::channel(true).1,
                binding_id: "source".into(),
                metrics: MetricsRecorder::default(),
            },
            TaskId::new(0),
            || {},
        ));
        assert_pending(&mut pump).await;

        cancellation.cancel();
        let result = poll_fn(|context| match pump.as_mut().poll(context) {
            Poll::Ready(result) => Poll::Ready(result),
            Poll::Pending => panic!("source pump did not observe cancellation during open"),
        })
        .await;

        assert!(result.is_ok());
        assert!(open_poll_dropped.load(Ordering::SeqCst));
        assert!(closed.load(Ordering::SeqCst));
    }

    struct BlockingSource {
        started: Arc<Mutex<Option<oneshot::Sender<()>>>>,
        poll_dropped: Arc<AtomicBool>,
        closed: Arc<AtomicBool>,
    }

    #[async_trait]
    impl StreamSource for BlockingSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            if let Some(started) = self.started.lock().take() {
                let _ = started.send(());
            }
            let _guard = PollGuard(Arc::clone(&self.poll_dropped));
            pending().await
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replayable: true,
                max_batch_rows: 1,
                max_batch_bytes: 1,
            }
        }
    }

    #[tokio::test]
    async fn cancel_drops_a_blocked_next_poll_then_closes_and_joins_source() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let (started_tx, started_rx) = oneshot::channel();
        let poll_dropped = Arc::new(AtomicBool::new(false));
        let closed = Arc::new(AtomicBool::new(false));
        let source = BlockingSource {
            started: Arc::new(Mutex::new(Some(started_tx))),
            poll_dropped: Arc::clone(&poll_dropped),
            closed: Arc::clone(&closed),
        };
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, _receiver) = edge_channel("source->sink", EdgeBudget::default()).unwrap();
        spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "quiet",
            binding,
            vec![sender],
        )
        .unwrap();

        started_rx.await.unwrap();
        supervisor.cancel();
        let report = supervisor.join_all().await;

        assert!(report.errors.is_empty());
        assert!(poll_dropped.load(Ordering::SeqCst));
        assert!(closed.load(Ordering::SeqCst));
        assert_eq!(supervisor.task_count(), 0);
    }

    #[tokio::test]
    async fn graceful_drain_drops_an_uncommitted_pending_poll_and_emits_eof() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let (started_tx, started_rx) = oneshot::channel();
        let poll_dropped = Arc::new(AtomicBool::new(false));
        let closed = Arc::new(AtomicBool::new(false));
        let source = BlockingSource {
            started: Arc::new(Mutex::new(Some(started_tx))),
            poll_dropped: Arc::clone(&poll_dropped),
            closed: Arc::clone(&closed),
        };
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, mut receiver) = edge_channel("source->drain", EdgeBudget::default()).unwrap();
        let progress = spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "quiet",
            binding,
            vec![sender],
        )
        .unwrap();

        started_rx.await.unwrap();
        progress.request_drain();

        assert_eq!(
            receiver.recv().await.unwrap().unwrap().kind(),
            StreamMessageKind::EndOfInput
        );
        assert!(receiver.recv().await.unwrap().is_none());
        assert!(supervisor.join_all().await.errors.is_empty());
        assert!(poll_dropped.load(Ordering::SeqCst));
        assert!(closed.load(Ordering::SeqCst));
        assert_eq!(progress.accept_state(), SourceAcceptState::Closed);
    }

    struct ErrorThenBlockingCloseSource {
        close_started: Option<oneshot::Sender<()>>,
        close_release: Option<oneshot::Receiver<()>>,
    }

    #[async_trait]
    impl StreamSource for ErrorThenBlockingCloseSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            Err(CalcFlowError::Internal {
                message: "source operation failed".into(),
            })
        }

        async fn close(&mut self) -> Result<()> {
            self.close_started.take().unwrap().send(()).unwrap();
            self.close_release.take().unwrap().await.unwrap();
            Ok(())
        }

        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replayable: true,
                max_batch_rows: 1,
                max_batch_bytes: 1,
            }
        }
    }

    #[tokio::test]
    async fn source_error_cancels_siblings_before_waiting_for_close() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let (close_started_tx, close_started_rx) = oneshot::channel();
        let (close_release_tx, close_release_rx) = oneshot::channel();
        let source = ErrorThenBlockingCloseSource {
            close_started: Some(close_started_tx),
            close_release: Some(close_release_rx),
        };
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, _receiver) = edge_channel("source->sink", EdgeBudget::default()).unwrap();
        spawn_source_tasks(
            &mut supervisor,
            &context(cancellation.clone()),
            "broken",
            binding,
            vec![sender],
        )
        .unwrap();
        let (sibling_finished_tx, sibling_finished_rx) = oneshot::channel();
        let sibling_cancellation = cancellation.clone();
        supervisor.spawn("sibling", async move {
            sibling_cancellation.cancelled().await;
            sibling_finished_tx.send(()).unwrap();
            Ok(())
        });

        close_started_rx.await.unwrap();
        assert!(
            cancellation.is_cancelled(),
            "source failure must trigger cancellation before close completes"
        );
        sibling_finished_rx.await.unwrap();
        let mut join = Box::pin(supervisor.join_all());
        assert_pending(&mut join).await;
        close_release_tx.send(()).unwrap();
        let report = join.await;

        assert_eq!(report.errors.len(), 1);
        assert!(matches!(
            &report.errors[0].error,
            CalcFlowError::Internal { message } if message == "source operation failed"
        ));
    }

    #[tokio::test]
    async fn cursor_regression_fails_before_the_offending_batch_is_enqueued() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let source = StepSource::new([
            Ok(Some(SourceEvent::Data {
                batch: batch(1),
                cursor: cursor(2),
            })),
            Ok(Some(SourceEvent::Data {
                batch: batch(2),
                cursor: cursor(1),
            })),
        ]);
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, mut receiver) = edge_channel("source->sink", EdgeBudget::default()).unwrap();
        spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "orders",
            binding,
            vec![sender],
        )
        .unwrap();

        let report = supervisor.join_all().await;

        assert_eq!(report.errors.len(), 1);
        assert!(matches!(
            &report.errors[0].error,
            CalcFlowError::InvalidArgument { field, .. } if field == "sources.orders.cursor"
        ));
        let first = receiver.recv().await.unwrap().unwrap();
        assert_eq!(first.as_data().unwrap().metadata().sequence(), 0);
        assert!(receiver.recv().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn watermark_regression_names_the_binding_and_stays_off_the_edge() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let source = StepSource::new([
            Ok(Some(SourceEvent::Watermark(EventTime::from_micros(5)))),
            Ok(Some(SourceEvent::Watermark(EventTime::from_micros(4)))),
        ]);
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, mut receiver) = edge_channel("source->sink", EdgeBudget::default()).unwrap();
        spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "orders",
            binding,
            vec![sender],
        )
        .unwrap();

        let report = supervisor.join_all().await;

        assert_eq!(report.errors.len(), 1);
        assert!(matches!(
            &report.errors[0].error,
            CalcFlowError::InvalidArgument { field, .. }
                if field == "sources.orders.watermark"
        ));
        assert_eq!(
            receiver.recv().await.unwrap().unwrap().kind(),
            StreamMessageKind::Watermark
        );
        assert!(receiver.recv().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn first_cursor_after_recovery_must_advance_past_the_resume_cursor() {
        let cancellation = CancellationToken::new();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let resume = cursor(2);
        let source = StepSource::new([Ok(Some(SourceEvent::Data {
            batch: batch(1),
            cursor: resume.clone(),
        }))]);
        let binding = SourceBinding::new(Box::new(source), Some(resume), 7).unwrap();
        let (sender, mut receiver) = edge_channel("source->sink", EdgeBudget::default()).unwrap();
        spawn_source_tasks(
            &mut supervisor,
            &context(cancellation),
            "input",
            binding,
            vec![sender],
        )
        .unwrap();

        let report = supervisor.join_all().await;

        assert_eq!(report.errors.len(), 1);
        assert!(matches!(
            &report.errors[0].error,
            CalcFlowError::InvalidArgument { field, .. } if field == "sources.input.cursor"
        ));
        assert!(receiver.recv().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn source_error_cancels_and_joins_all_supervised_siblings() {
        let cancellation = CancellationToken::new();
        let sibling_finished = Arc::new(AtomicBool::new(false));
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        let source = StepSource::new([Err(CalcFlowError::Internal {
            message: "source read failed".into(),
        })]);
        let closed = Arc::clone(&source.closed);
        let binding = SourceBinding::new(Box::new(source), None, 0).unwrap();
        let (sender, _receiver) = edge_channel("source->sink", EdgeBudget::default()).unwrap();
        spawn_source_tasks(
            &mut supervisor,
            &context(cancellation.clone()),
            "broken",
            binding,
            vec![sender],
        )
        .unwrap();
        let sibling_finished_in_task = Arc::clone(&sibling_finished);
        supervisor.spawn("sibling", async move {
            cancellation.cancelled().await;
            sibling_finished_in_task.store(true, Ordering::SeqCst);
            Ok(())
        });

        let report = supervisor.join_all().await;

        assert_eq!(report.errors.len(), 1);
        assert!(matches!(
            &report.errors[0].error,
            CalcFlowError::Internal { message } if message == "source read failed"
        ));
        assert!(closed.load(Ordering::SeqCst));
        assert!(sibling_finished.load(Ordering::SeqCst));
        assert_eq!(supervisor.task_count(), 0);
    }
}
