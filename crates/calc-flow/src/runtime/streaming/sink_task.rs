use std::{
    cell::Cell,
    collections::BTreeMap,
    future::Future,
    io::{self, Write},
    panic::{self, AssertUnwindSafe},
    sync::{Arc, Once},
};

use futures::FutureExt;
use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};

use super::{
    EdgeReceiver, EnvelopeCost, StreamMessage, StreamMessageKind,
    context::{StreamTaskContext, wait_for_task_gate},
    job::ValidatedOrdinarySink,
    metrics::{MetricsRecorder, sink_metric_id},
    supervisor::{TaskFailureSignal, TaskId, TaskSupervisor, panic_message},
};
use crate::{
    CalcFlowError, CancellationToken, Epoch, JsonMap, Result, SinkDeliveryManifest,
    SinkManifestEntry, canonical_json,
};

const MAX_SINK_PRECOMMIT_BYTES: usize = 64 * 1024;

thread_local! {
    static REDACT_SENSITIVE_SINK_PANIC: Cell<bool> = const { Cell::new(false) };
}

static INSTALL_SENSITIVE_SINK_PANIC_HOOK: Once = Once::new();

fn install_sensitive_sink_panic_hook() {
    // Keep one permanent delegating hook. Swapping a process-global hook around
    // each connector call would suppress unrelated panics on concurrent tasks.
    INSTALL_SENSITIVE_SINK_PANIC_HOOK.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |panic| {
            let redact = REDACT_SENSITIVE_SINK_PANIC
                .try_with(Cell::get)
                .unwrap_or(false);
            if redact {
                write_redacted_sensitive_sink_panic(panic);
            } else {
                previous(panic);
            }
        }));
    });
}

fn write_redacted_sensitive_sink_panic(panic: &panic::PanicHookInfo<'_>) {
    let mut stderr = io::stderr().lock();
    if let Some(location) = panic.location() {
        let _ = writeln!(
            stderr,
            "transactional sink panicked at {}:{}:{} (payload redacted)",
            location.file(),
            location.line(),
            location.column()
        );
    } else {
        let _ = writeln!(
            stderr,
            "transactional sink panicked at an unknown location (payload redacted)"
        );
    }
}

struct SensitiveSinkPanicGuard {
    previous: bool,
}

impl SensitiveSinkPanicGuard {
    fn enter() -> Self {
        let previous = REDACT_SENSITIVE_SINK_PANIC.with(|redact| redact.replace(true));
        Self { previous }
    }
}

impl Drop for SensitiveSinkPanicGuard {
    fn drop(&mut self) {
        REDACT_SENSITIVE_SINK_PANIC.with(|redact| redact.set(self.previous));
    }
}

async fn catch_sensitive_sink_unwind<F>(future: F) -> std::thread::Result<F::Output>
where
    F: Future,
{
    install_sensitive_sink_panic_hook();
    let future = AssertUnwindSafe(future).catch_unwind();
    futures::pin_mut!(future);
    futures::future::poll_fn(|context| {
        // Tokio may move the future between worker threads, so scope each poll
        // instead of relying on a task-wide thread-local value.
        let _guard = SensitiveSinkPanicGuard::enter();
        future.as_mut().poll(context)
    })
    .await
}

pub(crate) struct SinkCheckpointAck {
    pub(crate) output_id: String,
    pub(crate) epoch: Epoch,
    pub(crate) sinks: BTreeMap<String, SinkManifestEntry>,
}

pub(crate) struct SinkFinalizeAck {
    pub(crate) output_id: String,
    pub(crate) epoch: Epoch,
}

pub(crate) enum SinkCheckpointCommand {
    ManifestDurable(Epoch),
    Preserve(Epoch),
    Terminal(Epoch),
    TerminalManifestDurable(Epoch),
    Abort(Epoch),
}

pub(crate) struct SinkCheckpointPort {
    pub(crate) initial_epoch: Epoch,
    pub(crate) acknowledgements: mpsc::Sender<SinkCheckpointAck>,
    pub(crate) commands: mpsc::Receiver<SinkCheckpointCommand>,
    pub(crate) finalizations: mpsc::Sender<SinkFinalizeAck>,
    pub(crate) terminal_ready: Option<mpsc::Sender<String>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SinkFailurePhase {
    Write,
    Checkpoint,
    Close,
}

#[derive(Debug)]
pub(crate) struct SinkTaskFailure {
    pub(crate) output_id: String,
    pub(crate) sink_id: String,
    pub(crate) phase: SinkFailurePhase,
    pub(crate) error: CalcFlowError,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SinkProgressSnapshot {
    pub(crate) delivered_batches: u64,
    pub(crate) delivered_rows: u64,
    pub(crate) delivered_bytes: u64,
    pub(crate) ended: bool,
}

#[derive(Default)]
struct SinkProgressState {
    snapshot: SinkProgressSnapshot,
    failures: Vec<SinkTaskFailure>,
}

#[derive(Clone, Default)]
pub(crate) struct SinkProgress(Arc<Mutex<SinkProgressState>>);

impl SinkProgress {
    pub(crate) fn snapshot(&self) -> SinkProgressSnapshot {
        self.0.lock().snapshot.clone()
    }

    pub(crate) fn take_failures(&self) -> Vec<SinkTaskFailure> {
        std::mem::take(&mut self.0.lock().failures)
    }

    fn record_delivery(&self, rows: usize, bytes: usize) -> Result<()> {
        let mut state = self.0.lock();
        let (batches, rows, bytes) = next_delivery_totals(&state.snapshot, rows, bytes)?;
        state.snapshot.delivered_batches = batches;
        state.snapshot.delivered_rows = rows;
        state.snapshot.delivered_bytes = bytes;
        Ok(())
    }

    fn record_failure(&self, failure: SinkTaskFailure) {
        self.0.lock().failures.push(failure);
    }

    fn mark_ended(&self) {
        self.0.lock().snapshot.ended = true;
    }
}

fn next_delivery_totals(
    snapshot: &SinkProgressSnapshot,
    rows: usize,
    bytes: usize,
) -> Result<(u64, u64, u64)> {
    let rows = delivery_increment(rows, "delivered_rows")?;
    let bytes = delivery_increment(bytes, "delivered_bytes")?;
    Ok((
        next_delivery_counter(snapshot.delivered_batches, 1, "delivered_batches")?,
        next_delivery_counter(snapshot.delivered_rows, rows, "delivered_rows")?,
        next_delivery_counter(snapshot.delivered_bytes, bytes, "delivered_bytes")?,
    ))
}

fn delivery_increment(value: usize, counter: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| counter_overflow(counter))
}

fn next_delivery_counter(current: u64, increment: u64, counter: &str) -> Result<u64> {
    current
        .checked_add(increment)
        .ok_or_else(|| counter_overflow(counter))
}

fn counter_overflow(counter: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("sink {counter} counter overflowed"),
    }
}

#[derive(Default)]
pub(super) struct SinkEpochOwner {
    abortable_epoch: Option<Epoch>,
}

#[cfg(test)]
pub(super) type SinkCommitFaultHook = Arc<dyn Fn() -> Result<()> + Send + Sync>;

impl SinkEpochOwner {
    fn begin(&mut self, epoch: Epoch) {
        self.abortable_epoch = Some(epoch);
    }

    fn preserve(&mut self, epoch: Epoch) {
        self.settle(epoch);
    }

    fn settle(&mut self, epoch: Epoch) {
        if self.abortable_epoch == Some(epoch) {
            self.abortable_epoch = None;
        }
    }

    const fn abortable_epoch(&self) -> Option<Epoch> {
        self.abortable_epoch
    }
}

pub(crate) struct SinkTaskInputs {
    pub(crate) output_id: String,
    pub(crate) pipeline_name: Option<String>,
    pub(crate) sinks: Vec<ValidatedOrdinarySink>,
    pub(crate) input: EdgeReceiver,
    pub(crate) context: StreamTaskContext,
    pub(crate) progress: SinkProgress,
    pub(crate) metrics: MetricsRecorder,
    pub(crate) data_gate: watch::Receiver<bool>,
    pub(crate) launch_cancel: CancellationToken,
    pub(crate) checkpoint: Option<SinkCheckpointPort>,
    pub(super) epoch_owner: SinkEpochOwner,
    #[cfg(test)]
    pub(super) sink_commit_fault: Option<SinkCommitFaultHook>,
}

pub(crate) fn spawn_sink_task(supervisor: &mut TaskSupervisor, inputs: SinkTaskInputs) -> TaskId {
    let task_name = format!("sink:{}", inputs.output_id);
    supervisor.spawn_with_failure_signal(task_name, move |failure_signal| async move {
        run_sink_task(inputs, failure_signal).await
    })
}

async fn run_sink_task(
    mut inputs: SinkTaskInputs,
    failure_signal: TaskFailureSignal,
) -> Result<()> {
    let closed_message = format!(
        "sink {:?} data gate closed before release",
        inputs.output_id
    );
    if !wait_for_task_gate(
        &mut inputs.data_gate,
        &inputs.launch_cancel,
        inputs.context.job().cancellation(),
        &closed_message,
    )
    .await?
    {
        let close_failed = close_all(&mut inputs, &failure_signal).await;
        return if close_failed {
            failure_signal.cancel_siblings();
            Err(task_failed(&inputs.output_id))
        } else {
            Ok(())
        };
    }
    if let Some(epoch) = inputs
        .checkpoint
        .as_ref()
        .map(|checkpoint| checkpoint.initial_epoch)
    {
        inputs.epoch_owner.begin(epoch);
    }
    if let Err((sink_id, error)) =
        begin_checkpoint_epoch(&mut inputs, failure_signal.task_id()).await
    {
        record_sink_failure(&inputs, sink_id, SinkFailurePhase::Checkpoint, error);
        if let Some(epoch) = inputs
            .checkpoint
            .as_ref()
            .map(|checkpoint| checkpoint.initial_epoch)
        {
            abort_all(
                &mut inputs,
                epoch,
                &BTreeMap::new(),
                failure_signal.task_id(),
            )
            .await;
            inputs.epoch_owner.settle(epoch);
        }
        failure_signal.cancel_siblings();
        let _ = close_all(&mut inputs, &failure_signal).await;
        return Err(task_failed(&inputs.output_id));
    }
    let failed = run_sink_loop(&mut inputs, &failure_signal).await;
    if let Some(epoch) = inputs.epoch_owner.abortable_epoch() {
        abort_all(
            &mut inputs,
            epoch,
            &BTreeMap::new(),
            failure_signal.task_id(),
        )
        .await;
        inputs.epoch_owner.settle(epoch);
    }
    let close_failed = close_all(&mut inputs, &failure_signal).await;
    if failed || close_failed {
        failure_signal.cancel_siblings();
        Err(task_failed(&inputs.output_id))
    } else {
        Ok(())
    }
}

fn task_failed(output_id: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!(
            "ordinary sink task {output_id:?} failed; inspect its private failure records"
        ),
    }
}

async fn run_sink_loop(inputs: &mut SinkTaskInputs, failure_signal: &TaskFailureSignal) -> bool {
    loop {
        match next_sink_step(inputs, failure_signal.task_id()).await {
            SinkLoopStep::Continue => {}
            SinkLoopStep::Complete | SinkLoopStep::Cancelled => return false,
            SinkLoopStep::Failed { sink_id, error } => {
                record_sink_failure(inputs, sink_id, SinkFailurePhase::Write, error);
                failure_signal.cancel_siblings();
                return true;
            }
            SinkLoopStep::CheckpointFailed { sink_id, error } => {
                record_sink_failure(inputs, sink_id, SinkFailurePhase::Checkpoint, error);
                failure_signal.cancel_siblings();
                return true;
            }
        }
    }
}

enum SinkLoopStep {
    Continue,
    Complete,
    Cancelled,
    Failed {
        sink_id: String,
        error: CalcFlowError,
    },
    CheckpointFailed {
        sink_id: String,
        error: CalcFlowError,
    },
}

async fn next_sink_step(inputs: &mut SinkTaskInputs, task_id: TaskId) -> SinkLoopStep {
    match receive_sink_message(inputs).await {
        Ok(Some(message)) => process_sink_message(inputs, message, task_id).await,
        Ok(None) => SinkLoopStep::Cancelled,
        Err(error) => SinkLoopStep::Failed {
            sink_id: inputs.output_id.clone(),
            error,
        },
    }
}

async fn receive_sink_message(inputs: &mut SinkTaskInputs) -> Result<Option<StreamMessage>> {
    let received = tokio::select! {
        biased;
        () = inputs.context.job().cancellation().cancelled() => return Ok(None),
        result = inputs.input.recv() => result,
    }?;
    match received {
        Some(message) => Ok(Some(message)),
        None if inputs.context.job().cancellation().is_cancelled() => Ok(None),
        None => Err(CalcFlowError::EdgeClosed {
            edge: inputs.input.edge().into(),
        }),
    }
}

async fn process_sink_message(
    inputs: &mut SinkTaskInputs,
    message: StreamMessage,
    task_id: TaskId,
) -> SinkLoopStep {
    match message.kind() {
        StreamMessageKind::Data => deliver_data(inputs, &message, task_id).await,
        StreamMessageKind::Watermark | StreamMessageKind::Idle => SinkLoopStep::Continue,
        StreamMessageKind::EndOfInput => {
            inputs.progress.mark_ended();
            process_terminal_checkpoint(inputs, task_id).await
        }
        StreamMessageKind::Barrier => process_checkpoint_barrier(inputs, &message, task_id).await,
    }
}

async fn deliver_data(
    inputs: &mut SinkTaskInputs,
    message: &StreamMessage,
    task_id: TaskId,
) -> SinkLoopStep {
    let batch = message.as_data().expect("data kind always has a batch");
    let cost = match EnvelopeCost::of_message(message) {
        Ok(cost) => cost,
        Err(error) => {
            return SinkLoopStep::Failed {
                sink_id: inputs.output_id.clone(),
                error,
            };
        }
    };
    for sink in &mut inputs.sinks {
        match write_sink(
            sink,
            batch,
            cost,
            &inputs.output_id,
            &inputs.metrics,
            inputs.context.job().cancellation(),
            task_id,
        )
        .await
        {
            Ok(true) => {}
            Ok(false) => return SinkLoopStep::Cancelled,
            Err(error) => {
                return SinkLoopStep::Failed {
                    sink_id: sink.sink_id.to_string(),
                    error,
                };
            }
        }
    }
    match inputs.progress.record_delivery(cost.rows(), cost.bytes()) {
        Ok(()) => SinkLoopStep::Continue,
        Err(error) => SinkLoopStep::Failed {
            sink_id: inputs.output_id.clone(),
            error,
        },
    }
}

async fn write_sink(
    sink: &mut ValidatedOrdinarySink,
    batch: &crate::Batch,
    cost: EnvelopeCost,
    output_id: &str,
    metrics: &MetricsRecorder,
    cancellation: &CancellationToken,
    task_id: TaskId,
) -> Result<bool> {
    let timer = metrics.timer();
    let result = tokio::select! {
        biased;
        () = cancellation.cancelled() => return Ok(false),
        result = AssertUnwindSafe(sink.binding.write(batch)).catch_unwind() => result,
    };
    match result {
        Ok(result) => result?,
        Err(payload) => {
            return Err(CalcFlowError::TaskPanicked {
                task_id: task_id.as_u64(),
                message: panic_message(payload.as_ref()),
            });
        }
    }
    metrics.record_sink_delivery(
        &sink_metric_id(output_id, sink.sink_id.as_str()),
        cost,
        &timer,
    )?;
    Ok(true)
}

fn record_sink_failure(
    inputs: &SinkTaskInputs,
    sink_id: String,
    phase: SinkFailurePhase,
    error: CalcFlowError,
) {
    inputs.progress.record_failure(SinkTaskFailure {
        output_id: inputs.output_id.clone(),
        sink_id,
        phase,
        error,
    });
}

async fn begin_checkpoint_epoch(
    inputs: &mut SinkTaskInputs,
    task_id: TaskId,
) -> std::result::Result<(), (String, CalcFlowError)> {
    let Some(checkpoint) = &inputs.checkpoint else {
        return Ok(());
    };
    let epoch = checkpoint.initial_epoch;
    for sink in &mut inputs.sinks {
        if sink.binding.is_ordinary() {
            continue;
        }
        let result = AssertUnwindSafe(sink.binding.begin_epoch(epoch))
            .catch_unwind()
            .await;
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => return Err((sink.sink_id.to_string(), error)),
            Err(payload) => {
                return Err((
                    sink.sink_id.to_string(),
                    CalcFlowError::TaskPanicked {
                        task_id: task_id.as_u64(),
                        message: panic_message(payload.as_ref()),
                    },
                ));
            }
        }
    }
    Ok(())
}

async fn process_checkpoint_barrier(
    inputs: &mut SinkTaskInputs,
    message: &StreamMessage,
    task_id: TaskId,
) -> SinkLoopStep {
    let Some(expected_epoch) = inputs
        .checkpoint
        .as_ref()
        .map(|checkpoint| checkpoint.initial_epoch)
    else {
        return SinkLoopStep::CheckpointFailed {
            sink_id: inputs.output_id.clone(),
            error: CalcFlowError::InvalidArgument {
                field: format!("sinks.{}", inputs.output_id),
                message: "barrier delivery is unavailable before M5".into(),
            },
        };
    };
    let epoch = message
        .as_barrier()
        .expect("barrier kind always carries an epoch");
    if epoch != expected_epoch {
        return SinkLoopStep::CheckpointFailed {
            sink_id: inputs.output_id.clone(),
            error: CalcFlowError::CheckpointMismatch {
                message: format!(
                    "sink output {:?} expected epoch {} but received {}",
                    inputs.output_id,
                    expected_epoch.as_u64(),
                    epoch.as_u64()
                ),
            },
        };
    }
    let prepared = match pre_commit_all(inputs, epoch, task_id).await {
        Ok(prepared) => prepared,
        Err((sink_id, error, prepared)) => {
            abort_all(inputs, epoch, &prepared, task_id).await;
            inputs.epoch_owner.settle(epoch);
            return SinkLoopStep::CheckpointFailed { sink_id, error };
        }
    };
    let acknowledgement = SinkCheckpointAck {
        output_id: inputs.output_id.clone(),
        epoch,
        sinks: prepared.clone(),
    };
    let checkpoint = inputs
        .checkpoint
        .as_mut()
        .expect("checkpoint-enabled sink retains its port");
    let sent = checkpoint
        .acknowledgements
        .send(acknowledgement)
        .await
        .is_ok();
    if !sent {
        abort_all(inputs, epoch, &prepared, task_id).await;
        inputs.epoch_owner.settle(epoch);
        return SinkLoopStep::Cancelled;
    }
    let command = checkpoint.commands.recv().await;
    apply_checkpoint_command(inputs, epoch, &prepared, command, task_id, false).await
}

async fn process_terminal_checkpoint(inputs: &mut SinkTaskInputs, task_id: TaskId) -> SinkLoopStep {
    let Some(ready) = inputs
        .checkpoint
        .as_ref()
        .and_then(|checkpoint| checkpoint.terminal_ready.clone())
    else {
        return SinkLoopStep::Complete;
    };
    let cancellation = inputs.context.job().cancellation().clone();
    let ready_sent = tokio::select! {
        biased;
        () = cancellation.cancelled() => false,
        result = ready.send(inputs.output_id.clone()) => result.is_ok(),
    };
    if !ready_sent {
        return SinkLoopStep::Cancelled;
    }
    let command = {
        let checkpoint = inputs
            .checkpoint
            .as_mut()
            .expect("terminal sink retains its checkpoint port");
        checkpoint.commands.recv().await
    };
    let Some(SinkCheckpointCommand::Terminal(epoch)) = command else {
        if cancellation.is_cancelled() {
            return SinkLoopStep::Cancelled;
        }
        return SinkLoopStep::CheckpointFailed {
            sink_id: inputs.output_id.clone(),
            error: checkpoint_channel_closed(&inputs.output_id),
        };
    };
    if inputs
        .checkpoint
        .as_ref()
        .is_none_or(|checkpoint| checkpoint.initial_epoch != epoch)
    {
        return SinkLoopStep::CheckpointFailed {
            sink_id: inputs.output_id.clone(),
            error: CalcFlowError::CheckpointMismatch {
                message: format!(
                    "sink output {:?} received an unexpected terminal epoch {}",
                    inputs.output_id,
                    epoch.as_u64()
                ),
            },
        };
    }
    let prepared = match pre_commit_all(inputs, epoch, task_id).await {
        Ok(prepared) => prepared,
        Err((sink_id, error, prepared)) => {
            abort_all(inputs, epoch, &prepared, task_id).await;
            inputs.epoch_owner.settle(epoch);
            return SinkLoopStep::CheckpointFailed { sink_id, error };
        }
    };
    let acknowledgement = SinkCheckpointAck {
        output_id: inputs.output_id.clone(),
        epoch,
        sinks: prepared.clone(),
    };
    let sent = {
        let checkpoint = inputs
            .checkpoint
            .as_mut()
            .expect("terminal sink retains its checkpoint port");
        checkpoint
            .acknowledgements
            .send(acknowledgement)
            .await
            .is_ok()
    };
    if !sent {
        abort_all(inputs, epoch, &prepared, task_id).await;
        inputs.epoch_owner.settle(epoch);
        return SinkLoopStep::Cancelled;
    }
    let command = {
        let checkpoint = inputs
            .checkpoint
            .as_mut()
            .expect("terminal sink retains its checkpoint port");
        checkpoint.commands.recv().await
    };
    apply_checkpoint_command(inputs, epoch, &prepared, command, task_id, true).await
}

async fn apply_checkpoint_command(
    inputs: &mut SinkTaskInputs,
    epoch: Epoch,
    prepared: &BTreeMap<String, SinkManifestEntry>,
    command: Option<SinkCheckpointCommand>,
    task_id: TaskId,
    terminal: bool,
) -> SinkLoopStep {
    let Some(command) = command else {
        abort_all(inputs, epoch, prepared, task_id).await;
        inputs.epoch_owner.settle(epoch);
        return SinkLoopStep::Cancelled;
    };
    if sink_command_epoch(&command) != epoch {
        return fail_checkpoint_command(inputs, epoch, prepared, task_id).await;
    }
    match command {
        SinkCheckpointCommand::ManifestDurable(_) => {
            inputs.epoch_owner.preserve(epoch);
            finalize_if_terminal_mode(inputs, epoch, prepared, task_id, terminal, false).await
        }
        SinkCheckpointCommand::TerminalManifestDurable(_) => {
            inputs.epoch_owner.preserve(epoch);
            finalize_if_terminal_mode(inputs, epoch, prepared, task_id, terminal, true).await
        }
        SinkCheckpointCommand::Abort(_) => {
            abort_all(inputs, epoch, prepared, task_id).await;
            inputs.epoch_owner.settle(epoch);
            SinkLoopStep::CheckpointFailed {
                sink_id: inputs.output_id.clone(),
                error: CalcFlowError::Internal {
                    message: format!("checkpoint epoch {} was aborted", epoch.as_u64()),
                },
            }
        }
        SinkCheckpointCommand::Preserve(_) => {
            inputs.epoch_owner.preserve(epoch);
            preserve_checkpoint(terminal)
        }
        SinkCheckpointCommand::Terminal(_) => {
            fail_checkpoint_command(inputs, epoch, prepared, task_id).await
        }
    }
}

async fn finalize_if_terminal_mode(
    inputs: &mut SinkTaskInputs,
    epoch: Epoch,
    prepared: &BTreeMap<String, SinkManifestEntry>,
    task_id: TaskId,
    terminal: bool,
    expected_terminal: bool,
) -> SinkLoopStep {
    if terminal == expected_terminal {
        finalize_checkpoint(inputs, epoch, prepared, task_id, terminal).await
    } else {
        fail_preserved_checkpoint_command(inputs, epoch)
    }
}

fn preserve_checkpoint(terminal: bool) -> SinkLoopStep {
    if terminal {
        SinkLoopStep::Complete
    } else {
        SinkLoopStep::Cancelled
    }
}

const fn sink_command_epoch(command: &SinkCheckpointCommand) -> Epoch {
    match command {
        SinkCheckpointCommand::ManifestDurable(epoch)
        | SinkCheckpointCommand::Preserve(epoch)
        | SinkCheckpointCommand::Terminal(epoch)
        | SinkCheckpointCommand::TerminalManifestDurable(epoch)
        | SinkCheckpointCommand::Abort(epoch) => *epoch,
    }
}

async fn fail_checkpoint_command(
    inputs: &mut SinkTaskInputs,
    epoch: Epoch,
    prepared: &BTreeMap<String, SinkManifestEntry>,
    task_id: TaskId,
) -> SinkLoopStep {
    abort_all(inputs, epoch, prepared, task_id).await;
    inputs.epoch_owner.settle(epoch);
    SinkLoopStep::CheckpointFailed {
        sink_id: inputs.output_id.clone(),
        error: CalcFlowError::CheckpointMismatch {
            message: format!(
                "sink output {:?} received an out-of-phase command",
                inputs.output_id
            ),
        },
    }
}

fn fail_preserved_checkpoint_command(inputs: &SinkTaskInputs, epoch: Epoch) -> SinkLoopStep {
    SinkLoopStep::CheckpointFailed {
        sink_id: inputs.output_id.clone(),
        error: CalcFlowError::CheckpointMismatch {
            message: format!(
                "sink output {:?} received an out-of-phase durable command for epoch {}",
                inputs.output_id,
                epoch.as_u64()
            ),
        },
    }
}

async fn finalize_checkpoint(
    inputs: &mut SinkTaskInputs,
    epoch: Epoch,
    prepared: &BTreeMap<String, SinkManifestEntry>,
    task_id: TaskId,
    terminal: bool,
) -> SinkLoopStep {
    if let Err(step) = commit_checkpoint_sinks(inputs, epoch, prepared, task_id).await {
        return step;
    }
    if let Err(step) = send_checkpoint_finalization(inputs, epoch).await {
        return step;
    }
    if terminal {
        return SinkLoopStep::Complete;
    }
    if inputs.context.job().cancellation().is_cancelled() {
        return SinkLoopStep::Cancelled;
    }
    begin_next_checkpoint(inputs, epoch, task_id).await
}

async fn commit_checkpoint_sinks(
    inputs: &mut SinkTaskInputs,
    epoch: Epoch,
    prepared: &BTreeMap<String, SinkManifestEntry>,
    task_id: TaskId,
) -> std::result::Result<(), SinkLoopStep> {
    commit_all(inputs, epoch, prepared, task_id)
        .await
        .map_err(|(sink_id, error)| {
            if inputs.context.job().cancellation().is_cancelled()
                && matches!(&error, CalcFlowError::Cancelled { .. })
            {
                SinkLoopStep::Cancelled
            } else {
                SinkLoopStep::CheckpointFailed {
                    error: durable_commit_recovery_required(inputs, &sink_id, epoch),
                    sink_id,
                }
            }
        })
}

fn durable_commit_recovery_required(
    inputs: &SinkTaskInputs,
    sink_id: &str,
    epoch: Epoch,
) -> CalcFlowError {
    CalcFlowError::RecoveryRequired {
        pipeline_name: inputs
            .pipeline_name
            .clone()
            .expect("durable sink commit requires a checkpoint pipeline identity"),
        message: format!(
            "sink {sink_id:?} did not acknowledge durable epoch {}; restart must recover it",
            epoch.as_u64()
        ),
    }
}

async fn send_checkpoint_finalization(
    inputs: &mut SinkTaskInputs,
    epoch: Epoch,
) -> std::result::Result<(), SinkLoopStep> {
    let output_id = inputs.output_id.clone();
    let finalization = SinkFinalizeAck {
        output_id: output_id.clone(),
        epoch,
    };
    let cancellation = inputs.context.job().cancellation().clone();
    let checkpoint = inputs
        .checkpoint
        .as_mut()
        .expect("checkpoint-enabled sink retains its port");
    if send_finalization(&checkpoint.finalizations, finalization, &cancellation).await {
        return Ok(());
    }
    Err(finalization_send_failure(
        output_id,
        &cancellation,
        checkpoint.finalizations.is_closed(),
    ))
}

async fn send_finalization(
    finalizations: &mpsc::Sender<SinkFinalizeAck>,
    finalization: SinkFinalizeAck,
    cancellation: &CancellationToken,
) -> bool {
    match finalizations.try_send(finalization) {
        Ok(()) => true,
        Err(mpsc::error::TrySendError::Full(finalization)) => tokio::select! {
            biased;
            () = cancellation.cancelled() => false,
            result = finalizations.send(finalization) => result.is_ok(),
        },
        Err(mpsc::error::TrySendError::Closed(_)) => false,
    }
}

fn finalization_send_failure(
    output_id: String,
    cancellation: &CancellationToken,
    channel_closed: bool,
) -> SinkLoopStep {
    if cancellation.is_cancelled() && channel_closed {
        SinkLoopStep::Cancelled
    } else {
        SinkLoopStep::CheckpointFailed {
            error: checkpoint_channel_closed(&output_id),
            sink_id: output_id,
        }
    }
}

async fn begin_next_checkpoint(
    inputs: &mut SinkTaskInputs,
    epoch: Epoch,
    task_id: TaskId,
) -> SinkLoopStep {
    let next_epoch = match epoch.next() {
        Ok(epoch) => epoch,
        Err(error) => {
            return SinkLoopStep::CheckpointFailed {
                sink_id: inputs.output_id.clone(),
                error,
            };
        }
    };
    let checkpoint = inputs
        .checkpoint
        .as_mut()
        .expect("checkpoint-enabled sink retains its port");
    checkpoint.initial_epoch = next_epoch;
    inputs.epoch_owner.begin(next_epoch);
    match begin_checkpoint_epoch(inputs, task_id).await {
        Ok(()) => SinkLoopStep::Continue,
        Err((sink_id, error)) => SinkLoopStep::CheckpointFailed { sink_id, error },
    }
}

async fn pre_commit_all(
    inputs: &mut SinkTaskInputs,
    epoch: Epoch,
    task_id: TaskId,
) -> std::result::Result<
    BTreeMap<String, SinkManifestEntry>,
    (String, CalcFlowError, BTreeMap<String, SinkManifestEntry>),
> {
    let mut prepared = BTreeMap::new();
    for sink in &mut inputs.sinks {
        if sink.binding.is_ordinary() {
            prepared.insert(
                sink.sink_id.to_string(),
                SinkManifestEntry {
                    delivery: SinkDeliveryManifest::Ordinary,
                    pre_commit: None,
                },
            );
            continue;
        }
        let result = AssertUnwindSafe(sink.binding.pre_commit(epoch))
            .catch_unwind()
            .await;
        let metadata = match result {
            Ok(Ok(metadata)) => metadata,
            Ok(Err(error)) => return Err((sink.sink_id.to_string(), error, prepared)),
            Err(payload) => {
                return Err((
                    sink.sink_id.to_string(),
                    CalcFlowError::TaskPanicked {
                        task_id: task_id.as_u64(),
                        message: panic_message(payload.as_ref()),
                    },
                    prepared,
                ));
            }
        };
        if let Err(error) = validate_pre_commit_metadata(sink.sink_id.as_str(), &metadata) {
            return Err((sink.sink_id.to_string(), error, prepared));
        }
        prepared.insert(
            sink.sink_id.to_string(),
            SinkManifestEntry {
                delivery: sink.binding.capability().into_manifest(),
                pre_commit: Some(metadata),
            },
        );
    }
    Ok(prepared)
}

fn validate_pre_commit_metadata(sink_id: &str, metadata: &JsonMap) -> Result<()> {
    let value = serde_json::Value::Object(
        metadata
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    );
    let bytes = canonical_json(&value)?;
    if bytes.len() > MAX_SINK_PRECOMMIT_BYTES {
        return Err(CalcFlowError::InvalidArgument {
            field: format!("sinks.{sink_id}.pre_commit"),
            message: format!("must not exceed {MAX_SINK_PRECOMMIT_BYTES} canonical JSON bytes"),
        });
    }
    Ok(())
}

async fn commit_all(
    inputs: &mut SinkTaskInputs,
    epoch: Epoch,
    prepared: &BTreeMap<String, SinkManifestEntry>,
    task_id: TaskId,
) -> std::result::Result<(), (String, CalcFlowError)> {
    #[cfg(not(test))]
    let _ = task_id;
    #[cfg(test)]
    let transactional_sink_count = inputs
        .sinks
        .iter()
        .filter(|sink| !sink.binding.is_ordinary())
        .count();
    #[cfg(test)]
    let mut committed_sink_count = 0_usize;
    for sink in &mut inputs.sinks {
        if sink.binding.is_ordinary() {
            continue;
        }
        let state = prepared[sink.sink_id.as_str()]
            .pre_commit
            .as_ref()
            .expect("transactional pre-commit metadata is present");
        let result = catch_sensitive_sink_unwind(sink.binding.commit(epoch, state)).await;
        match result {
            Ok(Ok(())) => {
                #[cfg(test)]
                {
                    committed_sink_count += 1;
                    if committed_sink_count < transactional_sink_count
                        && let Some(inject) = &inputs.sink_commit_fault
                    {
                        let result = panic::catch_unwind(AssertUnwindSafe(|| inject()));
                        match result {
                            Ok(Ok(())) => {}
                            Ok(Err(error)) => return Err((sink.sink_id.to_string(), error)),
                            Err(payload) => {
                                return Err((
                                    sink.sink_id.to_string(),
                                    CalcFlowError::TaskPanicked {
                                        task_id: task_id.as_u64(),
                                        message: panic_message(payload.as_ref()),
                                    },
                                ));
                            }
                        }
                    }
                }
            }
            Ok(Err(error)) => return Err((sink.sink_id.to_string(), error)),
            Err(_) => {
                return Err((
                    sink.sink_id.to_string(),
                    CalcFlowError::Internal {
                        message: format!("sink commit panicked for epoch {}", epoch.as_u64()),
                    },
                ));
            }
        }
    }
    Ok(())
}

async fn abort_all(
    inputs: &mut SinkTaskInputs,
    epoch: Epoch,
    prepared: &BTreeMap<String, SinkManifestEntry>,
    _task_id: TaskId,
) {
    let output_id = inputs.output_id.clone();
    let progress = inputs.progress.clone();
    for sink in &mut inputs.sinks {
        if sink.binding.is_ordinary() {
            continue;
        }
        let state = prepared
            .get(sink.sink_id.as_str())
            .and_then(|entry| entry.pre_commit.as_ref());
        let result = catch_sensitive_sink_unwind(sink.binding.abort(epoch, state)).await;
        let error = match result {
            Ok(Ok(())) => None,
            Ok(Err(_)) | Err(_) => Some(CalcFlowError::Internal {
                message: format!("sink abort failed for epoch {}", epoch.as_u64()),
            }),
        };
        if let Some(error) = error {
            progress.record_failure(SinkTaskFailure {
                output_id: output_id.clone(),
                sink_id: sink.sink_id.to_string(),
                phase: SinkFailurePhase::Checkpoint,
                error,
            });
        }
    }
}

fn checkpoint_channel_closed(output_id: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("sink output {output_id:?} checkpoint channel closed"),
    }
}

pub(crate) async fn recover_transactional_sinks(
    sinks: &mut [ValidatedOrdinarySink],
    manifest: &crate::CheckpointManifest,
) -> Result<()> {
    for sink in sinks {
        let entry = manifest.sinks().get(sink.sink_id.as_str()).ok_or_else(|| {
            CalcFlowError::CheckpointMismatch {
                message: format!(
                    "checkpoint manifest is missing sink {:?}",
                    sink.sink_id.as_str()
                ),
            }
        })?;
        if !validate_recovery_entry(sink, entry)? {
            continue;
        }
        let recovery = catch_sensitive_sink_unwind(sink.binding.recover(manifest)).await;
        if !matches!(recovery, Ok(Ok(()))) {
            return Err(CalcFlowError::RecoveryRequired {
                pipeline_name: manifest.pipeline_name().into(),
                message: format!(
                    "sink {:?} did not recover durable epoch {}; retry recovery before allocating a new epoch",
                    sink.sink_id.as_str(),
                    manifest.epoch().as_u64()
                ),
            });
        }
    }
    Ok(())
}

fn validate_recovery_entry(
    sink: &ValidatedOrdinarySink,
    entry: &SinkManifestEntry,
) -> Result<bool> {
    let expected_delivery = sink.binding.capability().into_manifest();
    if entry.delivery != expected_delivery {
        return Err(CalcFlowError::CheckpointMismatch {
            message: format!(
                "checkpoint manifest sink {:?} delivery evidence does not match the prepared binding",
                sink.sink_id.as_str()
            ),
        });
    }
    if matches!(expected_delivery, SinkDeliveryManifest::Ordinary) {
        if entry.pre_commit.is_some() {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "checkpoint manifest ordinary sink {:?} carries transactional state",
                    sink.sink_id.as_str()
                ),
            });
        }
        return Ok(false);
    }
    if entry.pre_commit.is_none() {
        return Err(CalcFlowError::CheckpointMismatch {
            message: format!(
                "checkpoint manifest sink {:?} is not a prepared transaction",
                sink.sink_id.as_str()
            ),
        });
    }
    Ok(true)
}

async fn close_all(inputs: &mut SinkTaskInputs, failure_signal: &TaskFailureSignal) -> bool {
    let mut indexes = (0..inputs.sinks.len()).collect::<Vec<_>>();
    indexes.sort_by(|left, right| {
        inputs.sinks[*left]
            .sink_id
            .cmp(&inputs.sinks[*right].sink_id)
    });
    let mut failed = false;
    for index in indexes {
        let sink = &mut inputs.sinks[index];
        let result = AssertUnwindSafe(sink.binding.close()).catch_unwind().await;
        let error = match result {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error),
            Err(payload) => Some(CalcFlowError::TaskPanicked {
                task_id: failure_signal.task_id().as_u64(),
                message: panic_message(payload.as_ref()),
            }),
        };
        if let Some(error) = error {
            failed = true;
            inputs.progress.record_failure(SinkTaskFailure {
                output_id: inputs.output_id.clone(),
                sink_id: sink.sink_id.to_string(),
                phase: SinkFailurePhase::Close,
                error,
            });
        }
    }
    failed
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        process::Command,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    use async_trait::async_trait;
    use chrono::{TimeZone, Utc};
    use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
    use parking_lot::Mutex;
    use tokio::sync::{Notify, mpsc, watch};

    use super::{
        SinkCheckpointCommand, SinkCheckpointPort, SinkEpochOwner, SinkFailurePhase,
        SinkFinalizeAck, SinkProgress, SinkTaskInputs, recover_transactional_sinks,
        spawn_sink_task, validate_pre_commit_metadata,
    };
    use crate::{
        Batch, BatchMetadata, CalcFlowError, CancellationToken, EdgeBudget, JsonMap, Result,
        RetentionClass, SinkDeliveryManifest, StreamJobContext, StreamMessage, edge_channel,
        runtime::streaming::{
            job::{
                OrdinarySinkBinding, OrdinaryStreamSink, TransactionalStreamSink,
                ValidatedOrdinarySink,
            },
            supervisor::TaskSupervisor,
        },
    };

    fn batch(sequence: u64) -> Batch {
        let record = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![i64::try_from(sequence).unwrap()])) as _,
        )])
        .unwrap();
        Batch::table(
            vec![record],
            BatchMetadata::new("source", sequence, BTreeMap::new()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn pre_commit_metadata_is_canonical_and_bounded_before_acknowledgement() {
        validate_pre_commit_metadata(
            "sink",
            &BTreeMap::from([("prepared".into(), serde_json::json!(1))]),
        )
        .unwrap();
        let error = validate_pre_commit_metadata(
            "sink",
            &BTreeMap::from([("oversized".into(), serde_json::json!("x".repeat(64 * 1024)))]),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. }
                if field == "sinks.sink.pre_commit"
        ));
    }

    struct RecordingSink {
        id: String,
        log: Arc<Mutex<Vec<String>>>,
        fail_write: Option<u64>,
        fail_close: bool,
        write_gate: Option<Arc<Notify>>,
        closes: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl OrdinaryStreamSink for RecordingSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, batch: &Batch) -> Result<()> {
            let sequence = batch.metadata().sequence();
            self.log.lock().push(format!("{}:{sequence}", self.id));
            if let Some(gate) = &self.write_gate {
                gate.notified().await;
            }
            if self.fail_write == Some(sequence) {
                Err(CalcFlowError::Internal {
                    message: format!("{} write", self.id),
                })
            } else {
                Ok(())
            }
        }

        async fn close(&mut self) -> Result<()> {
            self.closes.fetch_add(1, Ordering::SeqCst);
            self.log.lock().push(format!("{}:close", self.id));
            if self.fail_close {
                Err(CalcFlowError::Internal {
                    message: format!("{} close", self.id),
                })
            } else {
                Ok(())
            }
        }
    }

    struct RecordingTransactionalSink {
        id: String,
        log: Arc<Mutex<Vec<String>>>,
        closes: Arc<AtomicUsize>,
    }

    struct RecoveryTransactionalSink {
        id: String,
        fail_commit: bool,
        committed: Arc<Mutex<std::collections::BTreeSet<String>>>,
        log: Arc<Mutex<Vec<String>>>,
    }

    const PRECOMMIT_SENTINEL: &str = "precommit-secret-sentinel";
    const PUBLIC_PANIC_SENTINEL: &str = "unrelated-public-panic-sentinel";
    const REDACTED_PANIC_DIAGNOSTIC: &str = "transactional sink panicked at ";
    const SENSITIVE_PANIC_CHILD: &str = "CALC_FLOW_SENSITIVE_SINK_PANIC_CHILD";

    struct LeakingLifecycleSink {
        panic_on_abort: bool,
        panic_on_commit: bool,
        panic_on_recover: bool,
    }

    #[async_trait]
    impl TransactionalStreamSink for LeakingLifecycleSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn pre_commit(&mut self, _epoch: crate::Epoch) -> Result<JsonMap> {
            Ok(BTreeMap::from([(
                "token".into(),
                serde_json::json!(PRECOMMIT_SENTINEL),
            )]))
        }

        async fn commit(&mut self, _epoch: crate::Epoch, _state: &JsonMap) -> Result<()> {
            assert!(
                !self.panic_on_commit,
                "commit panicked for {PRECOMMIT_SENTINEL}"
            );
            Ok(())
        }

        async fn abort(&mut self, _epoch: crate::Epoch, state: Option<&JsonMap>) -> Result<()> {
            assert!(
                !self.panic_on_abort,
                "abort panicked for {PRECOMMIT_SENTINEL}"
            );
            Err(CalcFlowError::Internal {
                message: format!("abort failed for {state:?}"),
            })
        }

        async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
            assert!(
                !self.panic_on_recover,
                "recover panicked for {PRECOMMIT_SENTINEL}"
            );
            Err(CalcFlowError::Internal {
                message: format!("recover failed for {:?}", manifest.sinks()),
            })
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl TransactionalStreamSink for RecoveryTransactionalSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn begin_epoch(&mut self, _epoch: crate::Epoch) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn pre_commit(&mut self, epoch: crate::Epoch) -> Result<JsonMap> {
            Ok(BTreeMap::from([(
                "prepared".into(),
                serde_json::json!(epoch.as_u64()),
            )]))
        }

        async fn commit(&mut self, epoch: crate::Epoch, _state: &JsonMap) -> Result<()> {
            self.log
                .lock()
                .push(format!("{}:commit:{}", self.id, epoch.as_u64()));
            if self.fail_commit {
                return Err(CalcFlowError::Internal {
                    message: format!("{} commit failed", self.id),
                });
            }
            self.committed.lock().insert(self.id.clone());
            Ok(())
        }

        async fn abort(&mut self, epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
            self.log
                .lock()
                .push(format!("{}:abort:{}", self.id, epoch.as_u64()));
            Ok(())
        }

        async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
            assert!(manifest.sinks().contains_key(&self.id));
            self.log
                .lock()
                .push(format!("{}:recover:{}", self.id, manifest.epoch().as_u64()));
            self.committed.lock().insert(self.id.clone());
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl TransactionalStreamSink for RecordingTransactionalSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn begin_epoch(&mut self, epoch: crate::Epoch) -> Result<()> {
            self.log
                .lock()
                .push(format!("{}:begin:{}", self.id, epoch.as_u64()));
            Ok(())
        }

        async fn write(&mut self, batch: &Batch) -> Result<()> {
            self.log
                .lock()
                .push(format!("{}:write:{}", self.id, batch.metadata().sequence()));
            Ok(())
        }

        async fn pre_commit(&mut self, epoch: crate::Epoch) -> Result<JsonMap> {
            self.log
                .lock()
                .push(format!("{}:precommit:{}", self.id, epoch.as_u64()));
            Ok(BTreeMap::from([(
                "prepared".into(),
                serde_json::json!(epoch.as_u64()),
            )]))
        }

        async fn commit(&mut self, epoch: crate::Epoch, state: &JsonMap) -> Result<()> {
            assert_eq!(state["prepared"], serde_json::json!(epoch.as_u64()));
            self.log
                .lock()
                .push(format!("{}:commit:{}", self.id, epoch.as_u64()));
            Ok(())
        }

        async fn abort(&mut self, epoch: crate::Epoch, _state: Option<&JsonMap>) -> Result<()> {
            self.log
                .lock()
                .push(format!("{}:abort:{}", self.id, epoch.as_u64()));
            Ok(())
        }

        async fn recover(&mut self, _manifest: &crate::CheckpointManifest) -> Result<()> {
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            self.closes.fetch_add(1, Ordering::SeqCst);
            self.log.lock().push(format!("{}:close", self.id));
            Ok(())
        }
    }

    struct Harness {
        supervisor: TaskSupervisor,
        cancellation: CancellationToken,
        sender: crate::EdgeSender,
        data: watch::Sender<bool>,
        progress: SinkProgress,
    }

    fn validated_sink(
        id: &str,
        log: &Arc<Mutex<Vec<String>>>,
        fail_write: Option<u64>,
        fail_close: bool,
        write_gate: Option<Arc<Notify>>,
        closes: &Arc<AtomicUsize>,
    ) -> ValidatedOrdinarySink {
        ValidatedOrdinarySink {
            sink_id: id.into(),
            binding: OrdinarySinkBinding::new(Box::new(RecordingSink {
                id: id.into(),
                log: Arc::clone(log),
                fail_write,
                fail_close,
                write_gate,
                closes: Arc::clone(closes),
            })),
        }
    }

    fn harness(sinks: Vec<ValidatedOrdinarySink>) -> Harness {
        harness_with_checkpoint(sinks, None)
    }

    fn harness_with_checkpoint(
        sinks: Vec<ValidatedOrdinarySink>,
        checkpoint: Option<SinkCheckpointPort>,
    ) -> Harness {
        let cancellation = CancellationToken::new();
        let context =
            StreamJobContext::new(9, "fingerprint", JsonMap::new(), None, cancellation.clone());
        let (sender, receiver) = edge_channel(
            "node.output->sink",
            EdgeBudget {
                max_rows: 64,
                max_bytes: 1 << 20,
            },
        )
        .unwrap();
        let (data, data_rx) = watch::channel(false);
        let progress = SinkProgress::default();
        let mut supervisor = TaskSupervisor::new(cancellation.clone());
        spawn_sink_task(
            &mut supervisor,
            SinkTaskInputs {
                output_id: "output".into(),
                pipeline_name: checkpoint.as_ref().map(|_| "pipeline".into()),
                sinks,
                input: receiver,
                context: context.for_sink("output").unwrap(),
                progress: progress.clone(),
                metrics: super::MetricsRecorder::default(),
                data_gate: data_rx,
                launch_cancel: CancellationToken::new(),
                checkpoint,
                epoch_owner: SinkEpochOwner::default(),
                sink_commit_fault: None,
            },
        );
        Harness {
            supervisor,
            cancellation,
            sender,
            data,
            progress,
        }
    }

    fn validated_transactional_sink(
        id: &str,
        log: &Arc<Mutex<Vec<String>>>,
        closes: &Arc<AtomicUsize>,
    ) -> ValidatedOrdinarySink {
        ValidatedOrdinarySink {
            sink_id: id.into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(RecordingTransactionalSink {
                id: id.into(),
                log: Arc::clone(log),
                closes: Arc::clone(closes),
            })),
        }
    }

    fn validated_epoch_idempotent_sink(
        id: &str,
        log: &Arc<Mutex<Vec<String>>>,
        closes: &Arc<AtomicUsize>,
    ) -> ValidatedOrdinarySink {
        ValidatedOrdinarySink {
            sink_id: id.into(),
            binding: OrdinarySinkBinding::new_epoch_idempotent(
                Box::new(RecordingTransactionalSink {
                    id: id.into(),
                    log: Arc::clone(log),
                    closes: Arc::clone(closes),
                }),
                "epoch-ledger",
                RetentionClass::Unbounded,
            )
            .unwrap(),
        }
    }

    #[tokio::test]
    async fn transactional_barrier_precommits_before_manifest_and_commits_after_durable_command() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let (ack_tx, mut ack_rx) = mpsc::channel(1);
        let (command_tx, command_rx) = mpsc::channel(1);
        let (finalize_tx, mut finalize_rx) = mpsc::channel(1);
        let mut harness = harness_with_checkpoint(
            vec![validated_transactional_sink("tx", &log, &closes)],
            Some(SinkCheckpointPort {
                initial_epoch: crate::Epoch::INITIAL,
                acknowledgements: ack_tx,
                commands: command_rx,
                finalizations: finalize_tx,
                terminal_ready: None,
            }),
        );
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::data(batch(0)))
            .await
            .unwrap();
        harness
            .sender
            .send(StreamMessage::barrier(crate::Epoch::INITIAL))
            .await
            .unwrap();

        let acknowledgement = ack_rx.recv().await.unwrap();
        assert_eq!(acknowledgement.output_id, "output");
        assert_eq!(acknowledgement.epoch, crate::Epoch::INITIAL);
        assert_eq!(
            acknowledgement.sinks["tx"].pre_commit.as_ref().unwrap()["prepared"],
            serde_json::json!(1)
        );
        assert_eq!(
            &log.lock()[..],
            ["tx:begin:1", "tx:write:0", "tx:precommit:1"]
        );

        command_tx
            .send(SinkCheckpointCommand::ManifestDurable(
                crate::Epoch::INITIAL,
            ))
            .await
            .unwrap();
        let finalized = finalize_rx.recv().await.unwrap();
        assert_eq!(finalized.epoch, crate::Epoch::INITIAL);
        assert_eq!(
            &log.lock()[..],
            [
                "tx:begin:1",
                "tx:write:0",
                "tx:precommit:1",
                "tx:commit:1",
                "tx:begin:2"
            ]
        );

        harness.cancellation.cancel();
        assert!(harness.supervisor.join_all().await.errors.is_empty());
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn epoch_idempotent_barrier_persists_declared_delivery_evidence() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let (ack_tx, mut ack_rx) = mpsc::channel(1);
        let (command_tx, command_rx) = mpsc::channel(1);
        let (finalize_tx, mut finalize_rx) = mpsc::channel(1);
        let mut harness = harness_with_checkpoint(
            vec![validated_epoch_idempotent_sink(
                "deduplicated",
                &log,
                &closes,
            )],
            Some(SinkCheckpointPort {
                initial_epoch: crate::Epoch::INITIAL,
                acknowledgements: ack_tx,
                commands: command_rx,
                finalizations: finalize_tx,
                terminal_ready: None,
            }),
        );
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::barrier(crate::Epoch::INITIAL))
            .await
            .unwrap();

        let acknowledgement = ack_rx.recv().await.unwrap();
        assert_eq!(
            acknowledgement.sinks["deduplicated"].delivery,
            SinkDeliveryManifest::EpochIdempotent {
                mechanism: "epoch-ledger".into(),
                retention: RetentionClass::Unbounded,
            }
        );

        command_tx
            .send(SinkCheckpointCommand::ManifestDurable(
                crate::Epoch::INITIAL,
            ))
            .await
            .unwrap();
        assert_eq!(
            finalize_rx.recv().await.unwrap().epoch,
            crate::Epoch::INITIAL
        );
        harness.cancellation.cancel();
        assert!(harness.supervisor.join_all().await.errors.is_empty());
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn pre_manifest_abort_aborts_every_prepared_sink_and_never_commits() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let (ack_tx, mut ack_rx) = mpsc::channel(1);
        let (command_tx, command_rx) = mpsc::channel(1);
        let (finalize_tx, mut finalize_rx) = mpsc::channel(1);
        let mut harness = harness_with_checkpoint(
            ["a", "b"]
                .map(|id| validated_transactional_sink(id, &log, &closes))
                .into(),
            Some(SinkCheckpointPort {
                initial_epoch: crate::Epoch::INITIAL,
                acknowledgements: ack_tx,
                commands: command_rx,
                finalizations: finalize_tx,
                terminal_ready: None,
            }),
        );
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::barrier(crate::Epoch::INITIAL))
            .await
            .unwrap();
        assert_eq!(ack_rx.recv().await.unwrap().sinks.len(), 2);

        command_tx
            .send(SinkCheckpointCommand::Abort(crate::Epoch::INITIAL))
            .await
            .unwrap();
        let report = harness.supervisor.join_all().await;

        assert_eq!(report.errors.len(), 1);
        assert!(finalize_rx.recv().await.is_none());
        let log = log.lock();
        assert!(log.contains(&"a:abort:1".into()));
        assert!(log.contains(&"b:abort:1".into()));
        assert!(!log.iter().any(|entry| entry.contains(":commit:")));
        assert_eq!(closes.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn pre_commit_metadata_never_leaks_through_abort_diagnostics() {
        let (ack_tx, mut ack_rx) = mpsc::channel(1);
        let (command_tx, command_rx) = mpsc::channel(1);
        let (finalize_tx, _finalize_rx) = mpsc::channel(1);
        let mut harness = harness_with_checkpoint(
            vec![ValidatedOrdinarySink {
                sink_id: "redacted".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(LeakingLifecycleSink {
                    panic_on_abort: false,
                    panic_on_commit: false,
                    panic_on_recover: false,
                })),
            }],
            Some(SinkCheckpointPort {
                initial_epoch: crate::Epoch::INITIAL,
                acknowledgements: ack_tx,
                commands: command_rx,
                finalizations: finalize_tx,
                terminal_ready: None,
            }),
        );
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::barrier(crate::Epoch::INITIAL))
            .await
            .unwrap();
        let acknowledgement = ack_rx.recv().await.unwrap();
        assert_eq!(
            acknowledgement.sinks["redacted"]
                .pre_commit
                .as_ref()
                .unwrap()["token"],
            serde_json::json!(PRECOMMIT_SENTINEL)
        );

        command_tx
            .send(SinkCheckpointCommand::Abort(crate::Epoch::INITIAL))
            .await
            .unwrap();
        let _ = harness.supervisor.join_all().await;

        let diagnostics = format!("{:?}", harness.progress.take_failures());
        assert!(!diagnostics.contains(PRECOMMIT_SENTINEL));
        assert!(diagnostics.contains("sink abort failed for epoch 1"));
    }

    #[tokio::test]
    async fn cancellation_aborts_the_active_transactional_epoch_before_a_barrier() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let (ack_tx, _ack_rx) = mpsc::channel(1);
        let (_command_tx, command_rx) = mpsc::channel(1);
        let (finalize_tx, _finalize_rx) = mpsc::channel(1);
        let mut harness = harness_with_checkpoint(
            vec![validated_transactional_sink("tx", &log, &closes)],
            Some(SinkCheckpointPort {
                initial_epoch: crate::Epoch::INITIAL,
                acknowledgements: ack_tx,
                commands: command_rx,
                finalizations: finalize_tx,
                terminal_ready: None,
            }),
        );
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::data(batch(0)))
            .await
            .unwrap();
        while !log.lock().contains(&"tx:write:0".into()) {
            tokio::task::yield_now().await;
        }

        harness.cancellation.cancel();
        let report = harness.supervisor.join_all().await;

        assert!(report.errors.is_empty());
        assert_eq!(
            &log.lock()[..],
            ["tx:begin:1", "tx:write:0", "tx:abort:1", "tx:close"]
        );
    }

    #[tokio::test]
    async fn sibling_failure_aborts_the_active_transactional_epoch_before_a_barrier() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let (ack_tx, _ack_rx) = mpsc::channel(1);
        let (_command_tx, command_rx) = mpsc::channel(1);
        let (finalize_tx, _finalize_rx) = mpsc::channel(1);
        let mut harness = harness_with_checkpoint(
            vec![validated_transactional_sink("tx", &log, &closes)],
            Some(SinkCheckpointPort {
                initial_epoch: crate::Epoch::INITIAL,
                acknowledgements: ack_tx,
                commands: command_rx,
                finalizations: finalize_tx,
                terminal_ready: None,
            }),
        );
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::data(batch(0)))
            .await
            .unwrap();
        while !log.lock().contains(&"tx:write:0".into()) {
            tokio::task::yield_now().await;
        }
        harness.supervisor.spawn("failing-sibling", async {
            Err(CalcFlowError::Internal {
                message: "sibling failed".into(),
            })
        });

        let report = harness.supervisor.join_all().await;

        assert_eq!(report.primary_error_count, 1);
        assert_eq!(
            &log.lock()[..],
            ["tx:begin:1", "tx:write:0", "tx:abort:1", "tx:close"]
        );
    }

    #[tokio::test]
    async fn out_of_phase_pre_manifest_command_aborts_prepared_state() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let (ack_tx, mut ack_rx) = mpsc::channel(1);
        let (command_tx, command_rx) = mpsc::channel(1);
        let (finalize_tx, _finalize_rx) = mpsc::channel(1);
        let mut harness = harness_with_checkpoint(
            vec![validated_transactional_sink("tx", &log, &closes)],
            Some(SinkCheckpointPort {
                initial_epoch: crate::Epoch::INITIAL,
                acknowledgements: ack_tx,
                commands: command_rx,
                finalizations: finalize_tx,
                terminal_ready: None,
            }),
        );
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::barrier(crate::Epoch::INITIAL))
            .await
            .unwrap();
        ack_rx.recv().await.unwrap();
        command_tx
            .send(SinkCheckpointCommand::ManifestDurable(
                crate::Epoch::INITIAL.next().unwrap(),
            ))
            .await
            .unwrap();

        let report = harness.supervisor.join_all().await;

        assert_eq!(report.errors.len(), 1);
        assert!(log.lock().contains(&"tx:abort:1".into()));
        assert!(!log.lock().iter().any(|entry| entry.contains(":commit:")));
    }

    #[tokio::test]
    async fn wrong_terminal_durable_commands_preserve_prepared_epochs_without_abort() {
        let nonterminal_log = Arc::new(Mutex::new(Vec::new()));
        let nonterminal_closes = Arc::new(AtomicUsize::new(0));
        let (ack_tx, mut ack_rx) = mpsc::channel(1);
        let (command_tx, command_rx) = mpsc::channel(1);
        let (finalize_tx, mut finalize_rx) = mpsc::channel(1);
        let mut nonterminal = harness_with_checkpoint(
            vec![validated_transactional_sink(
                "tx",
                &nonterminal_log,
                &nonterminal_closes,
            )],
            Some(SinkCheckpointPort {
                initial_epoch: crate::Epoch::INITIAL,
                acknowledgements: ack_tx,
                commands: command_rx,
                finalizations: finalize_tx,
                terminal_ready: None,
            }),
        );
        nonterminal.data.send(true).unwrap();
        nonterminal
            .sender
            .send(StreamMessage::barrier(crate::Epoch::INITIAL))
            .await
            .unwrap();
        ack_rx.recv().await.unwrap();
        command_tx
            .send(SinkCheckpointCommand::TerminalManifestDurable(
                crate::Epoch::INITIAL,
            ))
            .await
            .unwrap();

        let report = nonterminal.supervisor.join_all().await;
        assert_eq!(report.errors.len(), 1);
        assert!(finalize_rx.recv().await.is_none());
        assert!(
            !nonterminal_log
                .lock()
                .iter()
                .any(|entry| entry.contains(":abort:") || entry.contains(":commit:"))
        );

        let terminal_log = Arc::new(Mutex::new(Vec::new()));
        let terminal_closes = Arc::new(AtomicUsize::new(0));
        let (ack_tx, mut ack_rx) = mpsc::channel(1);
        let (command_tx, command_rx) = mpsc::channel(2);
        let (finalize_tx, mut finalize_rx) = mpsc::channel(1);
        let (terminal_ready_tx, mut terminal_ready_rx) = mpsc::channel(1);
        let mut terminal = harness_with_checkpoint(
            vec![validated_transactional_sink(
                "tx",
                &terminal_log,
                &terminal_closes,
            )],
            Some(SinkCheckpointPort {
                initial_epoch: crate::Epoch::INITIAL,
                acknowledgements: ack_tx,
                commands: command_rx,
                finalizations: finalize_tx,
                terminal_ready: Some(terminal_ready_tx),
            }),
        );
        terminal.data.send(true).unwrap();
        terminal
            .sender
            .send(StreamMessage::end_of_input())
            .await
            .unwrap();
        assert_eq!(terminal_ready_rx.recv().await.as_deref(), Some("output"));
        command_tx
            .send(SinkCheckpointCommand::Terminal(crate::Epoch::INITIAL))
            .await
            .unwrap();
        ack_rx.recv().await.unwrap();
        command_tx
            .send(SinkCheckpointCommand::ManifestDurable(
                crate::Epoch::INITIAL,
            ))
            .await
            .unwrap();

        let report = terminal.supervisor.join_all().await;
        assert_eq!(report.errors.len(), 1);
        assert!(finalize_rx.recv().await.is_none());
        assert!(
            !terminal_log
                .lock()
                .iter()
                .any(|entry| entry.contains(":abort:") || entry.contains(":commit:"))
        );
    }

    #[tokio::test]
    async fn post_manifest_partial_multi_sink_commit_recovers_forward_without_abort() {
        let committed = Arc::new(Mutex::new(std::collections::BTreeSet::new()));
        let log = Arc::new(Mutex::new(Vec::new()));
        let live_sink = |id: &str, fail_commit: bool| ValidatedOrdinarySink {
            sink_id: id.into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(RecoveryTransactionalSink {
                id: id.into(),
                fail_commit,
                committed: Arc::clone(&committed),
                log: Arc::clone(&log),
            })),
        };
        let (ack_tx, mut ack_rx) = mpsc::channel(1);
        let (command_tx, command_rx) = mpsc::channel(1);
        let (finalize_tx, mut finalize_rx) = mpsc::channel(1);
        let mut harness = harness_with_checkpoint(
            vec![live_sink("a", false), live_sink("b", true)],
            Some(SinkCheckpointPort {
                initial_epoch: crate::Epoch::INITIAL,
                acknowledgements: ack_tx,
                commands: command_rx,
                finalizations: finalize_tx,
                terminal_ready: None,
            }),
        );
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::barrier(crate::Epoch::INITIAL))
            .await
            .unwrap();
        let acknowledgement = ack_rx.recv().await.unwrap();
        command_tx
            .send(SinkCheckpointCommand::ManifestDurable(
                crate::Epoch::INITIAL,
            ))
            .await
            .unwrap();

        let report = harness.supervisor.join_all().await;

        assert_eq!(report.errors.len(), 1);
        let failures = harness.progress.take_failures();
        assert!(matches!(
            &failures[0].error,
            CalcFlowError::RecoveryRequired {
                pipeline_name,
                message,
            } if pipeline_name == "pipeline"
                && message.contains("sink \"b\"")
                && message.contains("epoch 1")
        ));
        assert!(finalize_rx.recv().await.is_none());
        assert_eq!(
            &*committed.lock(),
            &std::collections::BTreeSet::from(["a".into()])
        );
        assert!(!log.lock().iter().any(|entry| entry.contains(":abort:")));
        let manifest = crate::CheckpointManifest::new(crate::CheckpointManifestFields {
            pipeline_name: "pipeline".into(),
            pipeline_fingerprint:
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
            runtime_config_hash: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
                .into(),
            epoch: crate::Epoch::INITIAL,
            created_at: Utc.with_ymd_and_hms(2026, 8, 9, 8, 0, 0).unwrap(),
            recovery_status: crate::RecoveryStatus::Final,
            sources: BTreeMap::new(),
            operators: BTreeMap::new(),
            sinks: acknowledgement.sinks,
        })
        .unwrap();
        let recovery_sink = |id: &str| ValidatedOrdinarySink {
            sink_id: id.into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(RecoveryTransactionalSink {
                id: id.into(),
                fail_commit: false,
                committed: Arc::clone(&committed),
                log: Arc::clone(&log),
            })),
        };
        let mut recovery_sinks = vec![recovery_sink("a"), recovery_sink("b")];

        recover_transactional_sinks(&mut recovery_sinks, &manifest)
            .await
            .unwrap();

        assert_eq!(
            &*committed.lock(),
            &std::collections::BTreeSet::from(["a".into(), "b".into()])
        );
        assert!(log.lock().contains(&"a:recover:1".into()));
        assert!(log.lock().contains(&"b:recover:1".into()));
    }

    #[tokio::test]
    async fn recovery_accepts_matching_epoch_idempotent_delivery_evidence() {
        let committed = Arc::new(Mutex::new(std::collections::BTreeSet::new()));
        let log = Arc::new(Mutex::new(Vec::new()));
        let manifest = crate::CheckpointManifest::new(crate::CheckpointManifestFields {
            pipeline_name: "pipeline".into(),
            pipeline_fingerprint:
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
            runtime_config_hash: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
                .into(),
            epoch: crate::Epoch::INITIAL,
            created_at: Utc.with_ymd_and_hms(2026, 8, 9, 8, 0, 0).unwrap(),
            recovery_status: crate::RecoveryStatus::Final,
            sources: BTreeMap::new(),
            operators: BTreeMap::new(),
            sinks: BTreeMap::from([(
                "deduplicated".into(),
                crate::SinkManifestEntry {
                    delivery: SinkDeliveryManifest::EpochIdempotent {
                        mechanism: "epoch-ledger".into(),
                        retention: RetentionClass::Unbounded,
                    },
                    pre_commit: Some(BTreeMap::from([("prepared".into(), serde_json::json!(1))])),
                },
            )]),
        })
        .unwrap();
        let mut sinks = vec![ValidatedOrdinarySink {
            sink_id: "deduplicated".into(),
            binding: OrdinarySinkBinding::new_epoch_idempotent(
                Box::new(RecoveryTransactionalSink {
                    id: "deduplicated".into(),
                    fail_commit: false,
                    committed,
                    log: Arc::clone(&log),
                }),
                "epoch-ledger",
                RetentionClass::Unbounded,
            )
            .unwrap(),
        }];

        recover_transactional_sinks(&mut sinks, &manifest)
            .await
            .unwrap();

        assert_eq!(&log.lock()[..], ["deduplicated:recover:1"]);
    }

    #[tokio::test]
    async fn recovery_missing_ordinary_sink_uses_capability_neutral_diagnostic() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let manifest = crate::CheckpointManifest::new(crate::CheckpointManifestFields {
            pipeline_name: "pipeline".into(),
            pipeline_fingerprint:
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
            runtime_config_hash: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
                .into(),
            epoch: crate::Epoch::INITIAL,
            created_at: Utc.with_ymd_and_hms(2026, 8, 9, 8, 0, 0).unwrap(),
            recovery_status: crate::RecoveryStatus::Final,
            sources: BTreeMap::new(),
            operators: BTreeMap::new(),
            sinks: BTreeMap::new(),
        })
        .unwrap();
        let mut sinks = vec![validated_sink("ordinary", &log, None, false, None, &closes)];

        let error = recover_transactional_sinks(&mut sinks, &manifest)
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            CalcFlowError::CheckpointMismatch { ref message }
                if message == "checkpoint manifest is missing sink \"ordinary\""
        ));
    }

    #[tokio::test]
    async fn pre_commit_metadata_never_leaks_through_recovery_diagnostics() {
        let manifest = crate::CheckpointManifest::new(crate::CheckpointManifestFields {
            pipeline_name: "pipeline".into(),
            pipeline_fingerprint:
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
            runtime_config_hash: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
                .into(),
            epoch: crate::Epoch::INITIAL,
            created_at: Utc.with_ymd_and_hms(2026, 8, 9, 8, 0, 0).unwrap(),
            recovery_status: crate::RecoveryStatus::Final,
            sources: BTreeMap::new(),
            operators: BTreeMap::new(),
            sinks: BTreeMap::from([(
                "redacted".into(),
                crate::SinkManifestEntry {
                    delivery: SinkDeliveryManifest::Transactional,
                    pre_commit: Some(BTreeMap::from([(
                        "token".into(),
                        serde_json::json!(PRECOMMIT_SENTINEL),
                    )])),
                },
            )]),
        })
        .unwrap();
        let mut sinks = vec![ValidatedOrdinarySink {
            sink_id: "redacted".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(LeakingLifecycleSink {
                panic_on_abort: false,
                panic_on_commit: false,
                panic_on_recover: false,
            })),
        }];

        let error = recover_transactional_sinks(&mut sinks, &manifest)
            .await
            .unwrap_err();
        let diagnostic = format!("{error:?}");

        assert!(!diagnostic.contains(PRECOMMIT_SENTINEL));
        assert!(matches!(
            error,
            CalcFlowError::RecoveryRequired {
                ref pipeline_name,
                ref message,
            } if pipeline_name == "pipeline"
                && message.contains("sink \"redacted\"")
                && message.contains("epoch 1")
        ));
    }

    #[tokio::test]
    async fn sensitive_lifecycle_panic_payloads_never_reach_process_stderr() {
        if std::env::var_os(SENSITIVE_PANIC_CHILD).is_none() {
            let cargo = std::path::Path::new(env!("CARGO"));
            assert!(
                cargo.is_absolute(),
                "Cargo must provide its test harness with an absolute executable path"
            );
            let output = Command::new(cargo)
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .args([
                    "test",
                    "--quiet",
                    "--package",
                    env!("CARGO_PKG_NAME"),
                    "--lib",
                    "runtime::streaming::sink_task::tests::sensitive_lifecycle_panic_payloads_never_reach_process_stderr",
                    "--",
                    "--exact",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(SENSITIVE_PANIC_CHILD, "1")
                .env("RUST_BACKTRACE", "0")
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            assert!(
                output.status.success(),
                "child test failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            assert!(
                stderr.contains(PUBLIC_PANIC_SENTINEL),
                "child did not exercise the process panic hook: {stderr}"
            );
            assert_eq!(
                stderr.matches(REDACTED_PANIC_DIAGNOSTIC).count(),
                3,
                "each sensitive lifecycle panic must emit a payload-free diagnostic: {stderr}"
            );
            assert!(
                stderr.contains("sink_task.rs:"),
                "redacted panic diagnostic omitted its source location: {stderr}"
            );
            assert!(!stdout.contains(PRECOMMIT_SENTINEL), "{stdout}");
            assert!(!stderr.contains(PRECOMMIT_SENTINEL), "{stderr}");
            return;
        }

        exercise_commit_panic_redaction().await;
        exercise_abort_panic_redaction().await;
        exercise_recovery_panic_redaction().await;

        let unrelated = std::panic::catch_unwind(|| panic!("{PUBLIC_PANIC_SENTINEL}"));
        assert!(unrelated.is_err());
    }

    async fn exercise_commit_panic_redaction() {
        let (ack_tx, mut ack_rx) = mpsc::channel(1);
        let (command_tx, command_rx) = mpsc::channel(1);
        let (finalize_tx, _finalize_rx) = mpsc::channel(1);
        let mut harness = harness_with_checkpoint(
            vec![ValidatedOrdinarySink {
                sink_id: "redacted".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(LeakingLifecycleSink {
                    panic_on_abort: false,
                    panic_on_commit: true,
                    panic_on_recover: false,
                })),
            }],
            Some(SinkCheckpointPort {
                initial_epoch: crate::Epoch::INITIAL,
                acknowledgements: ack_tx,
                commands: command_rx,
                finalizations: finalize_tx,
                terminal_ready: None,
            }),
        );
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::barrier(crate::Epoch::INITIAL))
            .await
            .unwrap();
        ack_rx.recv().await.unwrap();
        command_tx
            .send(SinkCheckpointCommand::ManifestDurable(
                crate::Epoch::INITIAL,
            ))
            .await
            .unwrap();

        let report = harness.supervisor.join_all().await;
        let diagnostics = format!("{:?}", harness.progress.take_failures());

        assert_eq!(report.errors.len(), 1);
        assert!(!diagnostics.contains(PRECOMMIT_SENTINEL));
        assert!(diagnostics.contains("restart must recover"));
    }

    async fn exercise_abort_panic_redaction() {
        let (ack_tx, mut ack_rx) = mpsc::channel(1);
        let (command_tx, command_rx) = mpsc::channel(1);
        let (finalize_tx, _finalize_rx) = mpsc::channel(1);
        let mut harness = harness_with_checkpoint(
            vec![ValidatedOrdinarySink {
                sink_id: "redacted".into(),
                binding: OrdinarySinkBinding::new_transactional(Box::new(LeakingLifecycleSink {
                    panic_on_abort: true,
                    panic_on_commit: false,
                    panic_on_recover: false,
                })),
            }],
            Some(SinkCheckpointPort {
                initial_epoch: crate::Epoch::INITIAL,
                acknowledgements: ack_tx,
                commands: command_rx,
                finalizations: finalize_tx,
                terminal_ready: None,
            }),
        );
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::barrier(crate::Epoch::INITIAL))
            .await
            .unwrap();
        ack_rx.recv().await.unwrap();
        command_tx
            .send(SinkCheckpointCommand::Abort(crate::Epoch::INITIAL))
            .await
            .unwrap();

        let report = harness.supervisor.join_all().await;
        let diagnostics = format!("{:?}", harness.progress.take_failures());

        assert_eq!(report.errors.len(), 1);
        assert!(!diagnostics.contains(PRECOMMIT_SENTINEL));
        assert!(diagnostics.contains("sink abort failed for epoch 1"));
    }

    async fn exercise_recovery_panic_redaction() {
        let manifest = crate::CheckpointManifest::new(crate::CheckpointManifestFields {
            pipeline_name: "pipeline".into(),
            pipeline_fingerprint:
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
            runtime_config_hash: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
                .into(),
            epoch: crate::Epoch::INITIAL,
            created_at: Utc.with_ymd_and_hms(2026, 8, 9, 8, 0, 0).unwrap(),
            recovery_status: crate::RecoveryStatus::Final,
            sources: BTreeMap::new(),
            operators: BTreeMap::new(),
            sinks: BTreeMap::from([(
                "redacted".into(),
                crate::SinkManifestEntry {
                    delivery: SinkDeliveryManifest::Transactional,
                    pre_commit: Some(BTreeMap::from([(
                        "token".into(),
                        serde_json::json!(PRECOMMIT_SENTINEL),
                    )])),
                },
            )]),
        })
        .unwrap();
        let mut sinks = vec![ValidatedOrdinarySink {
            sink_id: "redacted".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(LeakingLifecycleSink {
                panic_on_abort: false,
                panic_on_commit: false,
                panic_on_recover: true,
            })),
        }];

        let error = recover_transactional_sinks(&mut sinks, &manifest)
            .await
            .unwrap_err();
        let diagnostic = format!("{error:?}");

        assert!(!diagnostic.contains(PRECOMMIT_SENTINEL));
        assert!(matches!(
            error,
            CalcFlowError::RecoveryRequired {
                ref pipeline_name,
                ref message,
            } if pipeline_name == "pipeline"
                && message.contains("sink \"redacted\"")
                && message.contains("epoch 1")
        ));
    }

    #[tokio::test]
    async fn recovery_rejects_changed_epoch_idempotent_evidence_before_connector() {
        let committed = Arc::new(Mutex::new(std::collections::BTreeSet::new()));
        let log = Arc::new(Mutex::new(Vec::new()));
        let manifest = crate::CheckpointManifest::new(crate::CheckpointManifestFields {
            pipeline_name: "pipeline".into(),
            pipeline_fingerprint:
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
            runtime_config_hash: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
                .into(),
            epoch: crate::Epoch::INITIAL,
            created_at: Utc.with_ymd_and_hms(2026, 8, 9, 8, 0, 0).unwrap(),
            recovery_status: crate::RecoveryStatus::Final,
            sources: BTreeMap::new(),
            operators: BTreeMap::new(),
            sinks: BTreeMap::from([(
                "deduplicated".into(),
                crate::SinkManifestEntry {
                    delivery: SinkDeliveryManifest::EpochIdempotent {
                        mechanism: "other-ledger".into(),
                        retention: RetentionClass::Unbounded,
                    },
                    pre_commit: Some(BTreeMap::from([("prepared".into(), serde_json::json!(1))])),
                },
            )]),
        })
        .unwrap();
        let mut sinks = vec![ValidatedOrdinarySink {
            sink_id: "deduplicated".into(),
            binding: OrdinarySinkBinding::new_epoch_idempotent(
                Box::new(RecoveryTransactionalSink {
                    id: "deduplicated".into(),
                    fail_commit: false,
                    committed,
                    log: Arc::clone(&log),
                }),
                "epoch-ledger",
                RetentionClass::Unbounded,
            )
            .unwrap(),
        }];

        let error = recover_transactional_sinks(&mut sinks, &manifest)
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            CalcFlowError::CheckpointMismatch { ref message }
                if message.contains("deduplicated")
                    && message.contains("delivery evidence")
        ));
        assert!(log.lock().is_empty());
    }

    #[tokio::test]
    async fn cancellation_unblocks_full_post_manifest_finalization_channel_without_abort() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let (ack_tx, mut ack_rx) = mpsc::channel(1);
        let (command_tx, command_rx) = mpsc::channel(1);
        let (finalize_tx, _finalize_rx) = mpsc::channel(1);
        finalize_tx
            .send(SinkFinalizeAck {
                output_id: "occupied".into(),
                epoch: crate::Epoch::INITIAL,
            })
            .await
            .unwrap();
        let mut harness = harness_with_checkpoint(
            vec![validated_transactional_sink("tx", &log, &closes)],
            Some(SinkCheckpointPort {
                initial_epoch: crate::Epoch::INITIAL,
                acknowledgements: ack_tx,
                commands: command_rx,
                finalizations: finalize_tx,
                terminal_ready: None,
            }),
        );
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::barrier(crate::Epoch::INITIAL))
            .await
            .unwrap();
        ack_rx.recv().await.unwrap();
        command_tx
            .send(SinkCheckpointCommand::ManifestDurable(
                crate::Epoch::INITIAL,
            ))
            .await
            .unwrap();
        while !log.lock().contains(&"tx:commit:1".into()) {
            tokio::task::yield_now().await;
        }

        harness.cancellation.cancel();
        let report =
            tokio::time::timeout(Duration::from_millis(100), harness.supervisor.join_all())
                .await
                .expect("cancellation must unblock finalization send");

        assert_eq!(report.errors.len(), 1);
        assert!(!log.lock().iter().any(|entry| entry.contains(":abort:")));
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn queued_manifest_durable_command_wins_over_same_turn_cancellation() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let (ack_tx, mut ack_rx) = mpsc::channel(1);
        let (command_tx, command_rx) = mpsc::channel(1);
        let (finalize_tx, _finalize_rx) = mpsc::channel(1);
        let mut harness = harness_with_checkpoint(
            vec![validated_transactional_sink("tx", &log, &closes)],
            Some(SinkCheckpointPort {
                initial_epoch: crate::Epoch::INITIAL,
                acknowledgements: ack_tx,
                commands: command_rx,
                finalizations: finalize_tx,
                terminal_ready: None,
            }),
        );
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::barrier(crate::Epoch::INITIAL))
            .await
            .unwrap();
        ack_rx.recv().await.unwrap();

        command_tx
            .send(SinkCheckpointCommand::ManifestDurable(
                crate::Epoch::INITIAL,
            ))
            .await
            .unwrap();
        harness.cancellation.cancel();

        let report = harness.supervisor.join_all().await;
        assert!(log.lock().contains(&"tx:commit:1".into()));
        assert!(!log.lock().iter().any(|entry| entry.contains(":abort:")));
        assert!(report.errors.is_empty());
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn configured_sink_order_finishes_batch_before_next_batch() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let sinks = ["one", "two", "three"]
            .map(|id| validated_sink(id, &log, None, false, None, &closes))
            .into();
        let mut harness = harness(sinks);
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::data(batch(0)))
            .await
            .unwrap();
        harness
            .sender
            .send(StreamMessage::data(batch(1)))
            .await
            .unwrap();
        harness
            .sender
            .send(StreamMessage::end_of_input())
            .await
            .unwrap();
        let report = harness.supervisor.join_all().await;
        assert!(report.errors.is_empty());
        assert_eq!(
            &log.lock()[..6],
            ["one:0", "two:0", "three:0", "one:1", "two:1", "three:1"]
        );
        assert_eq!(closes.load(Ordering::SeqCst), 3);
        assert_eq!(harness.progress.snapshot().delivered_batches, 2);
        assert!(harness.progress.snapshot().ended);
    }

    #[tokio::test]
    async fn sink_two_write_failure_stops_sink_three_and_close_failures_are_stable_secondaries() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let sinks = vec![
            validated_sink("one", &log, None, true, None, &closes),
            validated_sink("two", &log, Some(0), false, None, &closes),
            validated_sink("three", &log, None, true, None, &closes),
        ];
        let mut harness = harness(sinks);
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::data(batch(0)))
            .await
            .unwrap();
        let report = harness.supervisor.join_all().await;
        assert_eq!(report.errors.len(), 1);
        assert_eq!(&log.lock()[..2], ["one:0", "two:0"]);
        assert!(!log.lock().contains(&"three:0".to_owned()));
        assert_eq!(closes.load(Ordering::SeqCst), 3);
        let failures = harness.progress.take_failures();
        assert_eq!(failures.len(), 3);
        assert_eq!(failures[0].sink_id, "two");
        assert_eq!(failures[0].phase, SinkFailurePhase::Write);
        assert_eq!(failures[1].sink_id, "one");
        assert_eq!(failures[2].sink_id, "three");
        assert!(
            failures[1..]
                .iter()
                .all(|failure| failure.phase == SinkFailurePhase::Close)
        );
    }

    #[tokio::test]
    async fn cancel_drops_inflight_write_without_drain_then_closes_and_joins() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Notify::new());
        let sinks = vec![validated_sink(
            "slow",
            &log,
            None,
            false,
            Some(gate),
            &closes,
        )];
        let mut harness = harness(sinks);
        harness.data.send(true).unwrap();
        harness
            .sender
            .send(StreamMessage::data(batch(0)))
            .await
            .unwrap();
        while log.lock().is_empty() {
            tokio::task::yield_now().await;
        }
        harness.cancellation.cancel();
        let report = harness.supervisor.join_all().await;
        assert!(report.errors.is_empty());
        assert_eq!(closes.load(Ordering::SeqCst), 1);
        assert_eq!(harness.progress.snapshot().delivered_batches, 0);
        assert!(!harness.progress.snapshot().ended);
    }

    #[tokio::test]
    async fn premature_input_close_is_edge_closed_and_not_natural_end() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let sinks = vec![validated_sink("one", &log, None, false, None, &closes)];
        let mut harness = harness(sinks);
        harness.data.send(true).unwrap();
        drop(harness.sender);
        let report = harness.supervisor.join_all().await;
        assert_eq!(report.errors.len(), 1);
        let failures = harness.progress.take_failures();
        assert_eq!(failures.len(), 1);
        assert!(
            matches!(failures[0].error, CalcFlowError::EdgeClosed { ref edge } if edge == "node.output->sink")
        );
        assert!(!harness.progress.snapshot().ended);
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn convergence_marking_turns_closed_sink_ingress_into_a_wakeup() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let closes = Arc::new(AtomicUsize::new(0));
        let sinks = vec![validated_sink("one", &log, None, false, None, &closes)];
        let mut harness = harness(sinks);
        harness.data.send(true).unwrap();

        harness.cancellation.cancel();
        drop(harness.sender);

        let report = harness.supervisor.join_all().await;
        assert!(report.errors.is_empty(), "{report:?}");
        assert!(harness.progress.take_failures().is_empty());
        assert!(!harness.progress.snapshot().ended);
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }
}
