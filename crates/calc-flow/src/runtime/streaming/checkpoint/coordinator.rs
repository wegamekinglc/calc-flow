use std::{
    collections::BTreeMap,
    collections::BTreeSet,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use parking_lot::Mutex;

use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
    time::Instant,
};

use crate::{CalcFlowError, CancellationToken, Epoch, Result};

pub(crate) enum CheckpointRequest {
    Periodic,
    Terminal,
    Manual(oneshot::Sender<Result<Epoch>>),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum CheckpointPhase {
    Requested,
    SourcesCut,
    OperatorsSnapshotted,
    SinksPrecommitted,
    ManifestDurable,
    SinksCommitted,
    Completed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointEvent {
    Started(Epoch),
    PhaseAdvanced(Epoch, CheckpointPhase),
    ReadyToPublish(Epoch),
    Completed(Epoch),
    Failed(Epoch, String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ParticipantSet {
    pub(crate) sources: BTreeSet<String>,
    pub(crate) operators: BTreeSet<String>,
    pub(crate) sinks: BTreeSet<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AckKind {
    Source,
    Operator,
    SinkPrecommit,
    SinkCommit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CheckpointAck {
    kind: AckKind,
    participant_id: String,
    epoch: Epoch,
    canonical_digest: String,
}

impl CheckpointAck {
    pub(crate) fn source(id: &str, epoch: Epoch, canonical_digest: &str) -> Self {
        Self::new(AckKind::Source, id, epoch, canonical_digest)
    }

    pub(crate) fn operator(id: &str, epoch: Epoch, canonical_digest: &str) -> Self {
        Self::new(AckKind::Operator, id, epoch, canonical_digest)
    }

    pub(crate) fn sink_precommit(id: &str, epoch: Epoch, canonical_digest: &str) -> Self {
        Self::new(AckKind::SinkPrecommit, id, epoch, canonical_digest)
    }

    pub(crate) fn sink_commit(id: &str, epoch: Epoch) -> Self {
        Self::new(AckKind::SinkCommit, id, epoch, "committed")
    }

    fn new(kind: AckKind, id: &str, epoch: Epoch, canonical_digest: &str) -> Self {
        Self {
            kind,
            participant_id: id.into(),
            epoch,
            canonical_digest: canonical_digest.into(),
        }
    }
}

enum CoordinatorCommand {
    Ack(CheckpointAck),
    ManifestDurable(Epoch),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ManualCheckpointFailure {
    Cancelled,
    Failed {
        category: ManualCheckpointFailureCategory,
        epoch: Option<Epoch>,
        phase: Option<CheckpointPhase>,
    },
    RecoveryRequired {
        pipeline_name: String,
        message: String,
    },
    SinkCommit {
        sink_id: String,
        epoch: Epoch,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ManualCheckpointFailureCategory {
    Io,
    Timeout,
    Protocol,
    Internal,
}

impl ManualCheckpointFailure {
    fn into_error(self) -> CalcFlowError {
        match self {
            Self::Cancelled => checkpoint_cancelled(),
            Self::Failed {
                category,
                epoch,
                phase,
            } => super::super::projection::manual_checkpoint_failure_error(category, epoch, phase),
            Self::RecoveryRequired {
                pipeline_name,
                message,
            } => CalcFlowError::RecoveryRequired {
                pipeline_name,
                message,
            },
            Self::SinkCommit { sink_id, epoch } => {
                super::super::projection::manual_sink_commit_failure_error(&sink_id, epoch)
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct CheckpointCoordinatorHandle {
    requests: mpsc::Sender<CheckpointRequest>,
    commands: mpsc::Sender<CoordinatorCommand>,
    cancellation: CancellationToken,
    termination: Arc<Mutex<Option<ManualCheckpointFailure>>>,
}

impl CheckpointCoordinatorHandle {
    pub(crate) async fn request(&self, request: CheckpointRequest) -> Result<()> {
        self.requests
            .send(request)
            .await
            .map_err(|_| coordinator_closed())
    }

    pub(crate) async fn request_manual(&self) -> Result<ManualCheckpointWaiter> {
        let (result, receiver) = oneshot::channel();
        self.request(CheckpointRequest::Manual(result)).await?;
        Ok(ManualCheckpointWaiter { receiver })
    }

    pub(crate) async fn ack(&self, ack: CheckpointAck) -> Result<()> {
        self.commands
            .send(CoordinatorCommand::Ack(ack))
            .await
            .map_err(|_| coordinator_closed())
    }

    pub(crate) async fn manifest_durable(&self, epoch: Epoch) -> Result<()> {
        self.commands
            .send(CoordinatorCommand::ManifestDurable(epoch))
            .await
            .map_err(|_| coordinator_closed())
    }

    pub(crate) fn terminate(&self, failure: ManualCheckpointFailure) {
        let mut termination = self.termination.lock();
        if termination.is_none() {
            *termination = Some(failure);
        }
        drop(termination);
        self.cancellation.cancel();
    }
}

pub(crate) struct ManualCheckpointWaiter {
    receiver: oneshot::Receiver<Result<Epoch>>,
}

impl Future for ManualCheckpointWaiter {
    type Output = Result<Epoch>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.receiver).poll(context) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(coordinator_closed())),
        }
    }
}

pub(crate) fn spawn_checkpoint_coordinator(
    expected: ParticipantSet,
    next_epoch: Epoch,
    channel_capacity: usize,
    timeout: Duration,
    cancellation: CancellationToken,
) -> Result<(
    CheckpointCoordinatorHandle,
    mpsc::Receiver<CheckpointEvent>,
    JoinHandle<Result<()>>,
)> {
    if channel_capacity == 0 {
        return Err(CalcFlowError::InvalidArgument {
            field: "runtime.checkpoint.channel_capacity".into(),
            message: "must be positive".into(),
        });
    }
    if timeout.is_zero() {
        return Err(CalcFlowError::InvalidArgument {
            field: "runtime.checkpoint.timeout".into(),
            message: "must be positive".into(),
        });
    }
    validate_participants(&expected)?;
    let (request_tx, request_rx) = mpsc::channel(channel_capacity);
    let (command_tx, command_rx) = mpsc::channel(channel_capacity);
    let (event_tx, event_rx) = mpsc::channel(channel_capacity);
    let termination = Arc::new(Mutex::new(None));
    let handle = CheckpointCoordinatorHandle {
        requests: request_tx,
        commands: command_tx,
        cancellation: cancellation.clone(),
        termination: Arc::clone(&termination),
    };
    let task = tokio::spawn(run_coordinator(
        expected,
        next_epoch,
        timeout,
        request_rx,
        command_rx,
        event_tx,
        cancellation,
        termination,
    ));
    Ok((handle, event_rx, task))
}

struct EpochState {
    epoch: Epoch,
    manual_result: Option<oneshot::Sender<Result<Epoch>>>,
    phase: CheckpointPhase,
    deadline: Instant,
    source_acks: BTreeMap<String, CheckpointAck>,
    operator_acks: BTreeMap<String, CheckpointAck>,
    sink_precommits: BTreeMap<String, CheckpointAck>,
    sink_commits: BTreeMap<String, CheckpointAck>,
}

struct CoordinatorFailure {
    error: CalcFlowError,
    category: ManualCheckpointFailureCategory,
}

#[allow(
    clippy::too_many_arguments,
    reason = "the owned coordinator task receives every bounded channel and immutable policy"
)]
async fn run_coordinator(
    expected: ParticipantSet,
    mut next_epoch: Epoch,
    timeout: Duration,
    mut requests: mpsc::Receiver<CheckpointRequest>,
    mut commands: mpsc::Receiver<CoordinatorCommand>,
    events: mpsc::Sender<CheckpointEvent>,
    cancellation: CancellationToken,
    termination: Arc<Mutex<Option<ManualCheckpointFailure>>>,
) -> Result<()> {
    loop {
        let Some(request) = receive_request(&mut requests, &cancellation, &termination).await?
        else {
            return Ok(());
        };
        let (mut state, following_epoch) = begin_epoch(request, next_epoch, timeout)?;
        next_epoch = following_epoch;
        send_event(
            &events,
            CheckpointEvent::Started(state.epoch),
            &cancellation,
        )
        .await?;
        loop {
            let command = match receive_command(&mut commands, &state, &events, &cancellation).await
            {
                Ok(Some(command)) => command,
                Ok(None) => {
                    let failure = termination_failure(&termination);
                    fail_epoch_manual(&mut state, failure.clone());
                    fail_queued_manuals(&mut requests, failure).await;
                    return Ok(());
                }
                Err(error) => {
                    let failure = checkpoint_failed(error.category, state.epoch, state.phase);
                    fail_epoch_manual(&mut state, failure.clone());
                    fail_queued_manuals(&mut requests, failure).await;
                    return Err(error.error);
                }
            };
            match apply_or_fail_protocol(&mut state, &expected, command, &events, &cancellation)
                .await
            {
                Ok(true) => break,
                Ok(false) => {}
                Err(error) => {
                    let failure = checkpoint_failed(error.category, state.epoch, state.phase);
                    fail_epoch_manual(&mut state, failure.clone());
                    fail_queued_manuals(&mut requests, failure).await;
                    return Err(error.error);
                }
            }
        }
    }
}

async fn receive_request(
    requests: &mut mpsc::Receiver<CheckpointRequest>,
    cancellation: &CancellationToken,
    termination: &Arc<Mutex<Option<ManualCheckpointFailure>>>,
) -> Result<Option<CheckpointRequest>> {
    tokio::select! {
        biased;
        () = cancellation.cancelled() => {
            fail_queued_manuals(requests, termination_failure(termination)).await;
            Ok(None)
        },
        request = requests.recv() => request.map(Some).ok_or_else(coordinator_closed),
    }
}

fn fail_epoch_manual(state: &mut EpochState, failure: ManualCheckpointFailure) {
    if let Some(result) = state.manual_result.take() {
        let _ = result.send(Err(failure.into_error()));
    }
}

async fn fail_queued_manuals(
    requests: &mut mpsc::Receiver<CheckpointRequest>,
    failure: ManualCheckpointFailure,
) {
    requests.close();
    while let Some(request) = requests.recv().await {
        if let CheckpointRequest::Manual(result) = request {
            let _ = result.send(Err(failure.clone().into_error()));
        }
    }
}

fn termination_failure(
    termination: &Arc<Mutex<Option<ManualCheckpointFailure>>>,
) -> ManualCheckpointFailure {
    termination
        .lock()
        .clone()
        .unwrap_or(ManualCheckpointFailure::Cancelled)
}

fn begin_epoch(
    request: CheckpointRequest,
    epoch: Epoch,
    timeout: Duration,
) -> Result<(EpochState, Epoch)> {
    let next_epoch = epoch.next()?;
    let manual_result = match request {
        CheckpointRequest::Periodic | CheckpointRequest::Terminal => None,
        CheckpointRequest::Manual(result) => Some(result),
    };
    Ok((
        EpochState {
            epoch,
            manual_result,
            phase: CheckpointPhase::Requested,
            deadline: Instant::now() + timeout,
            source_acks: BTreeMap::new(),
            operator_acks: BTreeMap::new(),
            sink_precommits: BTreeMap::new(),
            sink_commits: BTreeMap::new(),
        },
        next_epoch,
    ))
}

async fn receive_command(
    commands: &mut mpsc::Receiver<CoordinatorCommand>,
    state: &EpochState,
    events: &mpsc::Sender<CheckpointEvent>,
    cancellation: &CancellationToken,
) -> std::result::Result<Option<CoordinatorCommand>, CoordinatorFailure> {
    tokio::select! {
        biased;
        () = cancellation.cancelled() => Ok(None),
        () = tokio::time::sleep_until(state.deadline) => {
            send_event(
                events,
                CheckpointEvent::Failed(state.epoch, "timeout".into()),
                cancellation,
            )
            .await
            .map_err(|error| CoordinatorFailure {
                error,
                category: ManualCheckpointFailureCategory::Internal,
            })?;
            cancellation.cancel();
            Err(CoordinatorFailure {
                error: CalcFlowError::Internal {
                    message: format!("checkpoint epoch {} timed out", state.epoch.as_u64()),
                },
                category: ManualCheckpointFailureCategory::Timeout,
            })
        }
        command = commands.recv() => command.map(Some).ok_or_else(|| CoordinatorFailure {
            error: coordinator_closed(),
            category: ManualCheckpointFailureCategory::Internal,
        }),
    }
}

async fn apply_or_fail_protocol(
    state: &mut EpochState,
    expected: &ParticipantSet,
    command: CoordinatorCommand,
    events: &mpsc::Sender<CheckpointEvent>,
    cancellation: &CancellationToken,
) -> std::result::Result<bool, CoordinatorFailure> {
    match apply_command(state, expected, command, events, cancellation).await {
        Ok(completed) => Ok(completed),
        Err(error) => {
            send_event(
                events,
                CheckpointEvent::Failed(state.epoch, "protocol".into()),
                cancellation,
            )
            .await
            .map_err(|send_error| CoordinatorFailure {
                error: send_error,
                category: ManualCheckpointFailureCategory::Internal,
            })?;
            cancellation.cancel();
            Err(CoordinatorFailure {
                error,
                category: ManualCheckpointFailureCategory::Protocol,
            })
        }
    }
}

async fn apply_command(
    state: &mut EpochState,
    expected: &ParticipantSet,
    command: CoordinatorCommand,
    events: &mpsc::Sender<CheckpointEvent>,
    cancellation: &CancellationToken,
) -> Result<bool> {
    match command {
        CoordinatorCommand::Ack(ack) => {
            validate_ack(state, expected, &ack)?;
            if is_identical_duplicate(state, &ack)? {
                return Ok(false);
            }
            record_ack(state, ack);
            advance_after_ack(state, expected, events, cancellation).await
        }
        CoordinatorCommand::ManifestDurable(epoch) => {
            if epoch != state.epoch || state.phase != CheckpointPhase::SinksPrecommitted {
                return Err(protocol_error(
                    state.epoch,
                    "manifest durable command is out of phase",
                ));
            }
            state.phase = CheckpointPhase::ManifestDurable;
            send_event(
                events,
                CheckpointEvent::PhaseAdvanced(state.epoch, state.phase),
                cancellation,
            )
            .await?;
            Ok(false)
        }
    }
}

fn validate_ack(state: &EpochState, expected: &ParticipantSet, ack: &CheckpointAck) -> Result<()> {
    if ack.epoch != state.epoch {
        return Err(protocol_error(state.epoch, "ack epoch does not match"));
    }
    if !valid_ack_digest(&ack.canonical_digest) {
        return Err(protocol_error(state.epoch, "ack digest is not bounded"));
    }
    if ack.kind == AckKind::SinkCommit && state.phase != CheckpointPhase::ManifestDurable {
        return Err(protocol_error(
            state.epoch,
            "sink commit ack precedes manifest durability",
        ));
    }
    if !is_expected_participant(expected, ack) {
        return Err(protocol_error(
            state.epoch,
            &format!(
                "{} ack participant {:?} is foreign",
                ack_kind_name(ack.kind),
                ack.participant_id
            ),
        ));
    }
    Ok(())
}

fn valid_ack_digest(digest: &str) -> bool {
    !digest.is_empty() && digest.len() <= 64 * 1024 && !digest.contains('\0')
}

fn is_expected_participant(expected: &ParticipantSet, ack: &CheckpointAck) -> bool {
    match ack.kind {
        AckKind::Source => expected.sources.contains(&ack.participant_id),
        AckKind::Operator => expected.operators.contains(&ack.participant_id),
        AckKind::SinkPrecommit | AckKind::SinkCommit => {
            expected.sinks.contains(&ack.participant_id)
        }
    }
}

const fn ack_kind_name(kind: AckKind) -> &'static str {
    match kind {
        AckKind::Source => "source",
        AckKind::Operator => "operator",
        AckKind::SinkPrecommit => "sink precommit",
        AckKind::SinkCommit => "sink commit",
    }
}

fn is_identical_duplicate(state: &EpochState, ack: &CheckpointAck) -> Result<bool> {
    let previous = ack_map(state, ack.kind).get(&ack.participant_id);
    match previous {
        Some(previous) if previous == ack => Ok(true),
        Some(_) => Err(protocol_error(state.epoch, "conflicting duplicate ack")),
        None => Ok(false),
    }
}

fn ack_map(state: &EpochState, kind: AckKind) -> &BTreeMap<String, CheckpointAck> {
    match kind {
        AckKind::Source => &state.source_acks,
        AckKind::Operator => &state.operator_acks,
        AckKind::SinkPrecommit => &state.sink_precommits,
        AckKind::SinkCommit => &state.sink_commits,
    }
}

fn record_ack(state: &mut EpochState, ack: CheckpointAck) {
    let map = match ack.kind {
        AckKind::Source => &mut state.source_acks,
        AckKind::Operator => &mut state.operator_acks,
        AckKind::SinkPrecommit => &mut state.sink_precommits,
        AckKind::SinkCommit => &mut state.sink_commits,
    };
    map.insert(ack.participant_id.clone(), ack);
}

async fn advance_after_ack(
    state: &mut EpochState,
    expected: &ParticipantSet,
    events: &mpsc::Sender<CheckpointEvent>,
    cancellation: &CancellationToken,
) -> Result<bool> {
    loop {
        match next_advancement(state, expected) {
            CheckpointAdvancement::Phase(next) => {
                state.phase = next;
                send_event(
                    events,
                    CheckpointEvent::PhaseAdvanced(state.epoch, next),
                    cancellation,
                )
                .await?;
            }
            CheckpointAdvancement::ReadyToPublish => {
                state.phase = CheckpointPhase::SinksPrecommitted;
                send_event(
                    events,
                    CheckpointEvent::ReadyToPublish(state.epoch),
                    cancellation,
                )
                .await?;
                return Ok(false);
            }
            CheckpointAdvancement::Completed => {
                state.phase = CheckpointPhase::Completed;
                send_event(
                    events,
                    CheckpointEvent::Completed(state.epoch),
                    cancellation,
                )
                .await?;
                if let Some(result) = state.manual_result.take() {
                    let _ = result.send(Ok(state.epoch));
                }
                return Ok(true);
            }
            CheckpointAdvancement::Pending => return Ok(false),
        }
    }
}

enum CheckpointAdvancement {
    Phase(CheckpointPhase),
    ReadyToPublish,
    Completed,
    Pending,
}

fn next_advancement(state: &EpochState, expected: &ParticipantSet) -> CheckpointAdvancement {
    match state.phase {
        CheckpointPhase::Requested if state.source_acks.len() == expected.sources.len() => {
            CheckpointAdvancement::Phase(CheckpointPhase::SourcesCut)
        }
        CheckpointPhase::SourcesCut if state.operator_acks.len() == expected.operators.len() => {
            CheckpointAdvancement::Phase(CheckpointPhase::OperatorsSnapshotted)
        }
        CheckpointPhase::OperatorsSnapshotted
            if state.sink_precommits.len() == expected.sinks.len() =>
        {
            CheckpointAdvancement::ReadyToPublish
        }
        CheckpointPhase::ManifestDurable if state.sink_commits.len() == expected.sinks.len() => {
            CheckpointAdvancement::Completed
        }
        _ => CheckpointAdvancement::Pending,
    }
}

async fn send_event(
    events: &mpsc::Sender<CheckpointEvent>,
    event: CheckpointEvent,
    cancellation: &CancellationToken,
) -> Result<()> {
    tokio::select! {
        biased;
        () = cancellation.cancelled() => Ok(()),
        result = events.send(event) => result.map_err(|_| coordinator_closed()),
    }
}

fn validate_participants(expected: &ParticipantSet) -> Result<()> {
    if expected.sources.is_empty() || expected.operators.is_empty() || expected.sinks.is_empty() {
        return Err(CalcFlowError::InvalidArgument {
            field: "runtime.checkpoint.participants".into(),
            message: "source, operator, and sink participant sets must be non-empty".into(),
        });
    }
    Ok(())
}

fn protocol_error(epoch: Epoch, message: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("checkpoint epoch {}: {message}", epoch.as_u64()),
    }
}

fn coordinator_closed() -> CalcFlowError {
    CalcFlowError::Internal {
        message: "checkpoint coordinator channel closed".into(),
    }
}

fn checkpoint_cancelled() -> CalcFlowError {
    CalcFlowError::Cancelled {
        run_id: "checkpoint".into(),
    }
}

fn checkpoint_failed(
    category: ManualCheckpointFailureCategory,
    epoch: Epoch,
    phase: CheckpointPhase,
) -> ManualCheckpointFailure {
    ManualCheckpointFailure::Failed {
        category,
        epoch: Some(epoch),
        phase: Some(phase),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, time::Duration};

    use super::{
        CheckpointAck, CheckpointEvent, CheckpointPhase, CheckpointRequest, ParticipantSet,
        spawn_checkpoint_coordinator,
    };
    use crate::{CalcFlowError, CancellationToken, Epoch};

    fn participants() -> ParticipantSet {
        ParticipantSet {
            sources: BTreeSet::from(["source".into()]),
            operators: BTreeSet::from(["operator".into()]),
            sinks: BTreeSet::from(["sink".into()]),
        }
    }

    async fn complete_epoch(
        handle: &super::CheckpointCoordinatorHandle,
        events: &mut tokio::sync::mpsc::Receiver<CheckpointEvent>,
        epoch: Epoch,
    ) {
        handle
            .ack(CheckpointAck::source("source", epoch, "source-state"))
            .await
            .unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::PhaseAdvanced(epoch, CheckpointPhase::SourcesCut)
        );
        handle
            .ack(CheckpointAck::operator("operator", epoch, "operator-state"))
            .await
            .unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::PhaseAdvanced(epoch, CheckpointPhase::OperatorsSnapshotted)
        );
        handle
            .ack(CheckpointAck::sink_precommit("sink", epoch, "sink-state"))
            .await
            .unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::ReadyToPublish(epoch)
        );
        handle.manifest_durable(epoch).await.unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::PhaseAdvanced(epoch, CheckpointPhase::ManifestDurable)
        );
        handle
            .ack(CheckpointAck::sink_commit("sink", epoch))
            .await
            .unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Completed(epoch)
        );
    }

    #[tokio::test]
    async fn manual_periodic_and_manual_requests_share_one_fifo_epoch_allocator() {
        let cancellation = CancellationToken::new();
        let (handle, mut events, task) = spawn_checkpoint_coordinator(
            participants(),
            Epoch::INITIAL,
            4,
            Duration::from_secs(30),
            cancellation.clone(),
        )
        .unwrap();
        let first = handle.request_manual().await.unwrap();
        handle.request(CheckpointRequest::Periodic).await.unwrap();
        let third = handle.request_manual().await.unwrap();

        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(Epoch::INITIAL)
        );
        complete_epoch(&handle, &mut events, Epoch::INITIAL).await;
        assert_eq!(first.await.unwrap(), Epoch::INITIAL);

        let second_epoch = Epoch::INITIAL.next().unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(second_epoch)
        );
        complete_epoch(&handle, &mut events, second_epoch).await;

        let third_epoch = second_epoch.next().unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(third_epoch)
        );
        complete_epoch(&handle, &mut events, third_epoch).await;
        assert_eq!(third.await.unwrap(), third_epoch);

        cancellation.cancel();
        task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn cancellation_fails_inflight_and_queued_manual_waiters_deterministically() {
        let cancellation = CancellationToken::new();
        let (handle, mut events, task) = spawn_checkpoint_coordinator(
            participants(),
            Epoch::INITIAL,
            4,
            Duration::from_secs(30),
            cancellation.clone(),
        )
        .unwrap();
        let inflight = handle.request_manual().await.unwrap();
        let queued = handle.request_manual().await.unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(Epoch::INITIAL)
        );

        cancellation.cancel();

        assert!(matches!(
            inflight.await,
            Err(CalcFlowError::Cancelled { ref run_id }) if run_id == "checkpoint"
        ));
        assert!(matches!(
            queued.await,
            Err(CalcFlowError::Cancelled { ref run_id }) if run_id == "checkpoint"
        ));
        task.await.unwrap().unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn timeout_fails_manual_waiter_with_epoch_context() {
        let cancellation = CancellationToken::new();
        let (handle, mut events, task) = spawn_checkpoint_coordinator(
            participants(),
            Epoch::INITIAL,
            2,
            Duration::from_secs(5),
            cancellation.clone(),
        )
        .unwrap();
        let waiter = handle.request_manual().await.unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(Epoch::INITIAL)
        );

        tokio::time::advance(Duration::from_secs(5)).await;
        tokio::task::yield_now().await;

        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Failed(Epoch::INITIAL, "timeout".into())
        );
        let CalcFlowError::Streaming(error) = waiter.await.unwrap_err() else {
            panic!("manual timeout must use the safe streaming boundary");
        };
        assert_eq!(
            error.category(),
            crate::runtime::streaming::projection::StreamingErrorCategory::CheckpointTimeout
        );
        assert_eq!(error.epoch(), Some(Epoch::INITIAL));
        assert!(task.await.unwrap().is_err());
        assert!(cancellation.is_cancelled());
    }

    #[tokio::test]
    async fn dropped_manual_waiter_does_not_dequeue_accepted_request() {
        let cancellation = CancellationToken::new();
        let (handle, mut events, task) = spawn_checkpoint_coordinator(
            participants(),
            Epoch::INITIAL,
            4,
            Duration::from_secs(30),
            cancellation.clone(),
        )
        .unwrap();
        let dropped = handle.request_manual().await.unwrap();
        let retained = handle.request_manual().await.unwrap();
        drop(dropped);

        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(Epoch::INITIAL)
        );
        complete_epoch(&handle, &mut events, Epoch::INITIAL).await;
        let retained_epoch = Epoch::INITIAL.next().unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(retained_epoch)
        );
        complete_epoch(&handle, &mut events, retained_epoch).await;
        assert_eq!(retained.await.unwrap(), retained_epoch);

        cancellation.cancel();
        task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn bounded_requests_allocate_single_flight_fifo_epochs() {
        let cancellation = CancellationToken::new();
        let (handle, mut events, task) = spawn_checkpoint_coordinator(
            participants(),
            Epoch::INITIAL,
            2,
            Duration::from_secs(30),
            cancellation.clone(),
        )
        .unwrap();
        handle.request(CheckpointRequest::Periodic).await.unwrap();
        handle.request(CheckpointRequest::Periodic).await.unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(Epoch::INITIAL)
        );
        handle
            .ack(CheckpointAck::source(
                "source",
                Epoch::INITIAL,
                "source-state",
            ))
            .await
            .unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::PhaseAdvanced(Epoch::INITIAL, CheckpointPhase::SourcesCut)
        );
        handle
            .ack(CheckpointAck::operator(
                "operator",
                Epoch::INITIAL,
                "operator-state",
            ))
            .await
            .unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::PhaseAdvanced(Epoch::INITIAL, CheckpointPhase::OperatorsSnapshotted,)
        );
        handle
            .ack(CheckpointAck::sink_precommit(
                "sink",
                Epoch::INITIAL,
                "sink-state",
            ))
            .await
            .unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::ReadyToPublish(Epoch::INITIAL)
        );
        handle.manifest_durable(Epoch::INITIAL).await.unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::PhaseAdvanced(Epoch::INITIAL, CheckpointPhase::ManifestDurable)
        );
        handle
            .ack(CheckpointAck::sink_commit("sink", Epoch::INITIAL))
            .await
            .unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Completed(Epoch::INITIAL)
        );
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(Epoch::INITIAL.next().unwrap())
        );

        cancellation.cancel();
        task.await.unwrap().unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn missing_source_ack_times_out_the_epoch_and_cancels_the_job() {
        let cancellation = CancellationToken::new();
        let (handle, mut events, task) = spawn_checkpoint_coordinator(
            participants(),
            Epoch::INITIAL,
            2,
            Duration::from_secs(5),
            cancellation.clone(),
        )
        .unwrap();
        handle.request(CheckpointRequest::Periodic).await.unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(Epoch::INITIAL)
        );

        tokio::time::advance(Duration::from_secs(5)).await;
        tokio::task::yield_now().await;

        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Failed(Epoch::INITIAL, "timeout".into())
        );
        assert_eq!(
            task.await.unwrap().unwrap_err().to_string(),
            "internal invariant failed: checkpoint epoch 1 timed out"
        );
        assert!(cancellation.is_cancelled());
    }

    #[tokio::test]
    async fn conflicting_duplicate_ack_fails_the_epoch_and_cancels_the_job() {
        let cancellation = CancellationToken::new();
        let (handle, mut events, task) = spawn_checkpoint_coordinator(
            participants(),
            Epoch::INITIAL,
            2,
            Duration::from_secs(30),
            cancellation.clone(),
        )
        .unwrap();
        handle.request(CheckpointRequest::Periodic).await.unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(Epoch::INITIAL)
        );
        let acknowledged = CheckpointAck::source("source", Epoch::INITIAL, "source-state");
        handle.ack(acknowledged.clone()).await.unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::PhaseAdvanced(Epoch::INITIAL, CheckpointPhase::SourcesCut)
        );
        handle.ack(acknowledged).await.unwrap();
        handle
            .ack(CheckpointAck::source(
                "source",
                Epoch::INITIAL,
                "conflicting-state",
            ))
            .await
            .unwrap();

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), events.recv())
                .await
                .unwrap()
                .unwrap(),
            CheckpointEvent::Failed(Epoch::INITIAL, "protocol".into())
        );
        assert!(task.await.unwrap().is_err());
        assert!(cancellation.is_cancelled());
    }

    #[tokio::test]
    async fn foreign_operator_ack_fails_the_epoch_with_participant_context() {
        let cancellation = CancellationToken::new();
        let (handle, mut events, task) = spawn_checkpoint_coordinator(
            participants(),
            Epoch::INITIAL,
            2,
            Duration::from_secs(30),
            cancellation.clone(),
        )
        .unwrap();
        handle.request(CheckpointRequest::Periodic).await.unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(Epoch::INITIAL)
        );

        handle
            .ack(CheckpointAck::operator(
                "intruder",
                Epoch::INITIAL,
                "foreign-state",
            ))
            .await
            .unwrap();

        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Failed(Epoch::INITIAL, "protocol".into())
        );
        let error = task.await.unwrap().unwrap_err().to_string();
        assert!(error.contains("operator"));
        assert!(error.contains("intruder"));
        assert!(cancellation.is_cancelled());
    }

    #[tokio::test]
    async fn early_downstream_ack_advances_after_its_prerequisite_cut() {
        let cancellation = CancellationToken::new();
        let (handle, mut events, task) = spawn_checkpoint_coordinator(
            participants(),
            Epoch::INITIAL,
            4,
            Duration::from_secs(30),
            cancellation.clone(),
        )
        .unwrap();
        handle.request(CheckpointRequest::Periodic).await.unwrap();
        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::Started(Epoch::INITIAL)
        );
        handle
            .ack(CheckpointAck::operator(
                "operator",
                Epoch::INITIAL,
                "operator-state",
            ))
            .await
            .unwrap();
        handle
            .ack(CheckpointAck::source(
                "source",
                Epoch::INITIAL,
                "source-state",
            ))
            .await
            .unwrap();

        assert_eq!(
            events.recv().await.unwrap(),
            CheckpointEvent::PhaseAdvanced(Epoch::INITIAL, CheckpointPhase::SourcesCut)
        );
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), events.recv())
                .await
                .unwrap()
                .unwrap(),
            CheckpointEvent::PhaseAdvanced(Epoch::INITIAL, CheckpointPhase::OperatorsSnapshotted)
        );

        cancellation.cancel();
        task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn sink_commit_before_manifest_durability_fails_closed() {
        let cancellation = CancellationToken::new();
        let (handle, mut events, task) = spawn_checkpoint_coordinator(
            participants(),
            Epoch::INITIAL,
            4,
            Duration::from_secs(30),
            cancellation.clone(),
        )
        .unwrap();
        handle.request(CheckpointRequest::Periodic).await.unwrap();
        assert!(matches!(
            events.recv().await,
            Some(CheckpointEvent::Started(_))
        ));
        handle
            .ack(CheckpointAck::source(
                "source",
                Epoch::INITIAL,
                "source-state",
            ))
            .await
            .unwrap();
        handle
            .ack(CheckpointAck::operator(
                "operator",
                Epoch::INITIAL,
                "operator-state",
            ))
            .await
            .unwrap();
        handle
            .ack(CheckpointAck::sink_precommit(
                "sink",
                Epoch::INITIAL,
                "sink-state",
            ))
            .await
            .unwrap();
        for _ in 0..3 {
            events.recv().await.unwrap();
        }

        handle
            .ack(CheckpointAck::sink_commit("sink", Epoch::INITIAL))
            .await
            .unwrap();

        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), events.recv())
                .await
                .unwrap()
                .unwrap(),
            CheckpointEvent::Failed(Epoch::INITIAL, "protocol".into())
        );
        assert!(task.await.unwrap().is_err());
        assert!(cancellation.is_cancelled());
    }
}
