use std::{collections::BTreeMap, collections::BTreeSet, time::Duration};

use tokio::{sync::mpsc, task::JoinHandle, time::Instant};

use crate::{CalcFlowError, CancellationToken, Epoch, Result};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CheckpointRequest {
    Periodic,
    Terminal,
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

#[derive(Clone)]
pub(crate) struct CheckpointCoordinatorHandle {
    requests: mpsc::Sender<CheckpointRequest>,
    commands: mpsc::Sender<CoordinatorCommand>,
}

impl CheckpointCoordinatorHandle {
    pub(crate) async fn request(&self, request: CheckpointRequest) -> Result<()> {
        self.requests
            .send(request)
            .await
            .map_err(|_| coordinator_closed())
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
    let handle = CheckpointCoordinatorHandle {
        requests: request_tx,
        commands: command_tx,
    };
    let task = tokio::spawn(run_coordinator(
        expected,
        next_epoch,
        timeout,
        request_rx,
        command_rx,
        event_tx,
        cancellation,
    ));
    Ok((handle, event_rx, task))
}

struct EpochState {
    request: CheckpointRequest,
    epoch: Epoch,
    phase: CheckpointPhase,
    deadline: Instant,
    source_acks: BTreeMap<String, CheckpointAck>,
    operator_acks: BTreeMap<String, CheckpointAck>,
    sink_precommits: BTreeMap<String, CheckpointAck>,
    sink_commits: BTreeMap<String, CheckpointAck>,
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
) -> Result<()> {
    let mut in_flight = None;
    loop {
        if in_flight.is_none() {
            let request = tokio::select! {
                biased;
                () = cancellation.cancelled() => return Ok(()),
                request = requests.recv() => request.ok_or_else(coordinator_closed)?,
            };
            let allocated = next_epoch;
            next_epoch = next_epoch.next()?;
            in_flight = Some(EpochState {
                request,
                epoch: allocated,
                phase: CheckpointPhase::Requested,
                deadline: Instant::now() + timeout,
                source_acks: BTreeMap::new(),
                operator_acks: BTreeMap::new(),
                sink_precommits: BTreeMap::new(),
                sink_commits: BTreeMap::new(),
            });
            send_event(&events, CheckpointEvent::Started(allocated), &cancellation).await?;
            continue;
        }

        let state = in_flight.as_mut().expect("in-flight state was checked");
        let command = tokio::select! {
            biased;
            () = cancellation.cancelled() => return Ok(()),
            () = tokio::time::sleep_until(state.deadline) => {
                let epoch = state.epoch;
                let failure = CheckpointEvent::Failed(epoch, "timeout".into());
                send_event(&events, failure, &cancellation).await?;
                cancellation.cancel();
                return Err(CalcFlowError::Internal {
                    message: format!("checkpoint epoch {} timed out", epoch.as_u64()),
                });
            }
            command = commands.recv() => command.ok_or_else(coordinator_closed)?,
        };
        let completed = match apply_command(state, &expected, command, &events, &cancellation).await
        {
            Ok(completed) => completed,
            Err(error) => {
                let epoch = state.epoch;
                send_event(
                    &events,
                    CheckpointEvent::Failed(epoch, "protocol".into()),
                    &cancellation,
                )
                .await?;
                cancellation.cancel();
                return Err(error);
            }
        };
        if completed {
            in_flight = None;
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
        return Err(protocol_error(
            state.epoch,
            "acknowledgement epoch does not match",
        ));
    }
    if ack.canonical_digest.is_empty()
        || ack.canonical_digest.len() > 64 * 1024
        || ack.canonical_digest.contains('\0')
    {
        return Err(protocol_error(
            state.epoch,
            "acknowledgement digest is not bounded",
        ));
    }
    if ack.kind == AckKind::SinkCommit && state.phase != CheckpointPhase::ManifestDurable {
        return Err(protocol_error(
            state.epoch,
            "sink commit acknowledgement precedes manifest durability",
        ));
    }
    let known = match ack.kind {
        AckKind::Source => expected.sources.contains(&ack.participant_id),
        AckKind::Operator => expected.operators.contains(&ack.participant_id),
        AckKind::SinkPrecommit | AckKind::SinkCommit => {
            expected.sinks.contains(&ack.participant_id)
        }
    };
    if !known {
        let kind = match ack.kind {
            AckKind::Source => "source",
            AckKind::Operator => "operator",
            AckKind::SinkPrecommit => "sink precommit",
            AckKind::SinkCommit => "sink commit",
        };
        return Err(protocol_error(
            state.epoch,
            &format!(
                "{kind} acknowledgement participant {:?} is foreign",
                ack.participant_id
            ),
        ));
    }
    Ok(())
}

fn is_identical_duplicate(state: &EpochState, ack: &CheckpointAck) -> Result<bool> {
    let previous = ack_map(state, ack.kind).get(&ack.participant_id);
    match previous {
        Some(previous) if previous == ack => Ok(true),
        Some(_) => Err(protocol_error(
            state.epoch,
            "conflicting duplicate acknowledgement",
        )),
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
        let next = match state.phase {
            CheckpointPhase::Requested if state.source_acks.len() == expected.sources.len() => {
                Some(CheckpointPhase::SourcesCut)
            }
            CheckpointPhase::SourcesCut
                if state.operator_acks.len() == expected.operators.len() =>
            {
                Some(CheckpointPhase::OperatorsSnapshotted)
            }
            CheckpointPhase::OperatorsSnapshotted
                if state.sink_precommits.len() == expected.sinks.len() =>
            {
                state.phase = CheckpointPhase::SinksPrecommitted;
                send_event(
                    events,
                    CheckpointEvent::ReadyToPublish(state.epoch),
                    cancellation,
                )
                .await?;
                return Ok(false);
            }
            CheckpointPhase::ManifestDurable
                if state.sink_commits.len() == expected.sinks.len() =>
            {
                state.phase = CheckpointPhase::Completed;
                send_event(
                    events,
                    CheckpointEvent::Completed(state.epoch),
                    cancellation,
                )
                .await?;
                return Ok(true);
            }
            _ => None,
        };
        let Some(next) = next else {
            return Ok(false);
        };
        state.phase = next;
        send_event(
            events,
            CheckpointEvent::PhaseAdvanced(state.epoch, next),
            cancellation,
        )
        .await?;
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

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, time::Duration};

    use super::{
        CheckpointAck, CheckpointEvent, CheckpointPhase, CheckpointRequest, ParticipantSet,
        spawn_checkpoint_coordinator,
    };
    use crate::{CancellationToken, Epoch};

    fn participants() -> ParticipantSet {
        ParticipantSet {
            sources: BTreeSet::from(["source".into()]),
            operators: BTreeSet::from(["operator".into()]),
            sinks: BTreeSet::from(["sink".into()]),
        }
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
