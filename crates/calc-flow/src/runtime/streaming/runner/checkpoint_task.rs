//! Managed checkpoint task and durable settlement.

use super::*;

#[cfg(test)]
mod tests;

pub(super) struct LiveCheckpointTaskInputs {
    pub(super) checkpoint: OpenedCheckpointRuntime,
    pub(super) channels: LiveCheckpointChannels,
    pub(super) live_progress: LiveProgressCoordinator,
    pub(super) sources: BTreeMap<String, SourceProgress>,
    pub(super) sinks: BTreeMap<String, SinkProgress>,
    pub(super) cancellation: CancellationToken,
    pub(super) metrics: MetricsRecorder,
    pub(super) core: Arc<JobCore>,
}

struct ManualCheckpointRegistration {
    core: Arc<JobCore>,
    coordinator: CheckpointCoordinatorHandle,
}

impl ManualCheckpointRegistration {
    fn install(core: Arc<JobCore>, coordinator: CheckpointCoordinatorHandle) -> Self {
        let previous = core.manual_checkpoint.lock().replace(coordinator.clone());
        debug_assert!(previous.is_none());
        core.changed.notify_waiters();
        Self { core, coordinator }
    }
}

impl Drop for ManualCheckpointRegistration {
    fn drop(&mut self) {
        self.core.manual_checkpoint.lock().take();
        self.core.changed.notify_waiters();
        self.coordinator.terminate(ManualCheckpointFailure::Failed {
            category: ManualCheckpointFailureCategory::Internal,
            epoch: None,
            phase: None,
        });
    }
}

/// The durable side of one checkpoint advances in this order. A failed step
/// retains the last completed phase for recovery diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DurableSettlementPhase {
    Published,
    SinksCommanded,
    CoordinatorDurable,
    SourceAcked,
}

impl DurableSettlementPhase {
    pub(super) fn advance(
        &mut self,
        expected: Self,
        next: Self,
        epoch: Epoch,
    ) -> crate::Result<()> {
        let adjacent = matches!(
            (*self, next),
            (Self::Published, Self::SinksCommanded)
                | (Self::SinksCommanded, Self::CoordinatorDurable)
                | (Self::CoordinatorDurable, Self::SourceAcked)
        );
        if *self != expected || !adjacent {
            return Err(checkpoint_protocol_error(
                epoch,
                "durable manifest settlement advanced out of order",
            ));
        }
        *self = next;
        Ok(())
    }
}

pub(super) struct DurableSettlementRequest<'a> {
    pub(super) epoch: Epoch,
    pub(super) terminal: bool,
    pub(super) acknowledgement_timeout: Duration,
    pub(super) phase: &'a mut DurableSettlementPhase,
}

#[derive(Default)]
struct EpochManifestAssembly {
    epoch: Option<Epoch>,
    terminal: bool,
    settlement_phase: Option<DurableSettlementPhase>,
    manifest_installed_unknown: bool,
    deferred_publication_error: Option<CalcFlowError>,
    sources: BTreeMap<String, SourceManifestEntry>,
    operators: BTreeMap<String, OperatorManifestEntry>,
    working_states: BTreeMap<String, Arc<crate::state::WorkingStatePins>>,
    operator_checkpoint_credits:
        BTreeMap<String, super::super::operator_task::OperatorCheckpointCredit>,
    sink_outputs: BTreeMap<String, BTreeMap<String, SinkManifestEntry>>,
    finalized_sink_outputs: BTreeSet<String>,
    timed_phase: Option<CheckpointPhase>,
    phase_timer: Option<MetricsTimer>,
    checkpoint_timer: Option<MetricsTimer>,
}

impl EpochManifestAssembly {
    const fn manifest_durable(&self) -> bool {
        self.settlement_phase.is_some()
    }

    fn start(&mut self, epoch: Epoch, terminal: bool) -> crate::Result<()> {
        if self.epoch.replace(epoch).is_some() {
            return Err(checkpoint_protocol_error(
                epoch,
                "started while another manifest assembly was active",
            ));
        }
        self.sources.clear();
        self.operators.clear();
        self.working_states.clear();
        self.operator_checkpoint_credits.clear();
        self.sink_outputs.clear();
        self.finalized_sink_outputs.clear();
        self.terminal = terminal;
        self.settlement_phase = None;
        self.manifest_installed_unknown = false;
        self.deferred_publication_error = None;
        Ok(())
    }

    fn expect_epoch(&self, epoch: Epoch) -> crate::Result<()> {
        if self.epoch == Some(epoch) {
            Ok(())
        } else {
            Err(checkpoint_protocol_error(
                epoch,
                "ack does not match the active manifest assembly",
            ))
        }
    }

    fn promote_terminal(&mut self, epoch: Epoch) -> crate::Result<()> {
        self.expect_epoch(epoch)?;
        self.terminal = true;
        Ok(())
    }

    fn complete(&mut self, epoch: Epoch) -> crate::Result<()> {
        self.expect_epoch(epoch)?;
        self.epoch = None;
        self.settlement_phase = None;
        self.manifest_installed_unknown = false;
        self.working_states.clear();
        Ok(())
    }

    fn start_metrics(&mut self, metrics: &MetricsRecorder, terminal: bool) -> crate::Result<()> {
        metrics.record_checkpoint_requested(terminal)?;
        self.timed_phase = Some(CheckpointPhase::Requested);
        self.phase_timer = Some(metrics.timer());
        self.checkpoint_timer = Some(metrics.timer());
        Ok(())
    }

    fn advance_metrics(
        &mut self,
        metrics: &MetricsRecorder,
        next: CheckpointPhase,
    ) -> crate::Result<()> {
        if let (Some(phase), Some(timer)) = (self.timed_phase, self.phase_timer.as_ref()) {
            metrics
                .record_checkpoint_phase(phase, timer.elapsed("checkpoint", "phase_duration")?)?;
        }
        self.timed_phase = Some(next);
        self.phase_timer = Some(metrics.timer());
        Ok(())
    }

    fn complete_metrics(&mut self, metrics: &MetricsRecorder, terminal: bool) -> crate::Result<()> {
        self.advance_metrics(metrics, CheckpointPhase::SinksCommitted)?;
        let elapsed = self
            .checkpoint_timer
            .as_ref()
            .ok_or_else(|| CalcFlowError::Internal {
                message: "checkpoint completion omitted its metrics timer".into(),
            })?
            .elapsed("checkpoint", "total_duration")?;
        metrics.record_checkpoint_completed(terminal, elapsed)?;
        self.timed_phase = None;
        self.phase_timer = None;
        self.checkpoint_timer = None;
        Ok(())
    }

    fn fail_metrics(&mut self, metrics: &MetricsRecorder, terminal: bool) -> crate::Result<()> {
        if let (Some(phase), Some(timer)) = (self.timed_phase, self.phase_timer.as_ref()) {
            metrics
                .record_checkpoint_phase(phase, timer.elapsed("checkpoint", "phase_duration")?)?;
        }
        metrics.record_checkpoint_failed(terminal)?;
        self.timed_phase = None;
        self.phase_timer = None;
        self.checkpoint_timer = None;
        Ok(())
    }

    fn cancel_metrics(&mut self) {
        self.timed_phase = None;
        self.phase_timer = None;
        self.checkpoint_timer = None;
    }
}

// Extracted from runner.rs; this task owns the existing epoch event loop.
// #lizard forgives
#[allow(
    clippy::too_many_lines,
    reason = "the checkpoint task is the single owner of epoch events, acks, and manifest publication"
)]
pub(super) async fn run_live_checkpoint_task(
    inputs: LiveCheckpointTaskInputs,
) -> crate::Result<()> {
    let LiveCheckpointTaskInputs {
        checkpoint,
        mut channels,
        live_progress,
        sources,
        sinks,
        cancellation,
        metrics,
        core,
    } = inputs;
    let participants = ParticipantSet {
        sources: checkpoint.identity.source_ids.clone(),
        operators: checkpoint.identity.operator_ids.clone(),
        sinks: channels.sink_commands.keys().cloned().collect(),
    };
    let expected_operators = participants.operators.clone();
    let expected_sinks = participants.sinks.clone();
    checkpoint.status.set_expected(
        participants.sources.len(),
        participants.operators.len(),
        participants.sinks.len(),
    );
    let capacity = participants
        .sources
        .len()
        .saturating_add(participants.operators.len())
        .saturating_add(participants.sinks.len())
        .max(4);
    let coordinator_cancellation = CancellationToken::new();
    let (coordinator, mut events, coordinator_task) = spawn_checkpoint_coordinator(
        participants,
        checkpoint.next_epoch,
        capacity,
        checkpoint.config.checkpoint_timeout,
        coordinator_cancellation.clone(),
    )?;
    let _manual_checkpoint =
        ManualCheckpointRegistration::install(Arc::clone(&core), coordinator.clone());
    let mut interval = tokio::time::interval(checkpoint.config.checkpoint_interval);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    interval.tick().await;
    let mut request_active = false;
    let mut terminal_request_active = false;
    let mut assembly = EpochManifestAssembly::default();
    let mut terminal_source_cuts = None;
    let mut terminal_operators = BTreeSet::new();
    let mut terminal_sinks = BTreeSet::new();
    let terminal_sources = {
        let terminal_sources = sources.clone();
        let terminal_cancellation = cancellation.clone();
        async move { wait_for_terminal_source_cuts(&terminal_sources, &terminal_cancellation).await }
    };
    tokio::pin!(terminal_sources);
    let mut terminal_sources_observed = false;
    let result = loop {
        tokio::select! {
            biased;
            () = cancellation.cancelled(), if !assembly.manifest_durable() => break Ok(()),
            event = events.recv() => {
                let Some(event) = event else {
                    break Err(CalcFlowError::Internal {
                        message: "checkpoint coordinator event channel closed".into(),
                    });
                };
                match handle_checkpoint_event(
                    event,
                    &coordinator,
                    &checkpoint,
                    &live_progress,
                    &sources,
                    &channels.sink_commands,
                    &channels.operator_commands,
                    &cancellation,
                    &metrics,
                    &mut assembly,
                    &mut request_active,
                    &mut terminal_request_active,
                    &mut terminal_source_cuts,
                ).await {
                    Ok(true) => break Ok(()),
                    Ok(false) => {
                        if cancellation.is_cancelled() {
                            continue;
                        }
                        if let Err(error) = maybe_request_terminal_checkpoint(
                            &coordinator,
                            &expected_operators,
                            &expected_sinks,
                            &terminal_operators,
                            &terminal_sinks,
                            terminal_source_cuts.is_some(),
                            &mut request_active,
                            &mut terminal_request_active,
                        ).await {
                            break Err(error);
                        }
                    }
                    Err(error) => break Err(error),
                }
            }
            terminal = &mut terminal_sources,
                if !terminal_sources_observed && !assembly.manifest_durable() => {
                match terminal {
                    Ok(mut cuts) => {
                        if let Err(error) = add_restored_ended_source_cuts(&checkpoint, &mut cuts) {
                            break Err(error);
                        }
                        terminal_source_cuts = Some(cuts);
                        terminal_sources_observed = true;
                    }
                    Err(error) => break Err(error),
                }
                if let Err(error) = maybe_request_terminal_checkpoint(
                    &coordinator,
                    &expected_operators,
                    &expected_sinks,
                    &terminal_operators,
                    &terminal_sinks,
                    terminal_source_cuts.is_some(),
                    &mut request_active,
                    &mut terminal_request_active,
                ).await {
                    break Err(error);
                }
            }
            ready = channels.operator_terminal_ready.recv(), if !assembly.manifest_durable() => {
                let Some(ready) = ready else {
                    break Err(checkpoint_channel_closed("operator terminal readiness"));
                };
                if !expected_operators.contains(&ready) {
                    break Err(CalcFlowError::CheckpointMismatch {
                        message: format!("foreign terminal operator {ready:?}"),
                    });
                }
                terminal_operators.insert(ready);
                if let Err(error) = maybe_request_terminal_checkpoint(
                    &coordinator,
                    &expected_operators,
                    &expected_sinks,
                    &terminal_operators,
                    &terminal_sinks,
                    terminal_source_cuts.is_some(),
                    &mut request_active,
                    &mut terminal_request_active,
                ).await {
                    break Err(error);
                }
            }
            ready = channels.sink_terminal_ready.recv(), if !assembly.manifest_durable() => {
                let Some(ready) = ready else {
                    break Err(checkpoint_channel_closed("sink terminal readiness"));
                };
                if !expected_sinks.contains(&ready) {
                    break Err(CalcFlowError::CheckpointMismatch {
                        message: format!("foreign terminal sink output {ready:?}"),
                    });
                }
                terminal_sinks.insert(ready);
                if let Err(error) = maybe_request_terminal_checkpoint(
                    &coordinator,
                    &expected_operators,
                    &expected_sinks,
                    &terminal_operators,
                    &terminal_sinks,
                    terminal_source_cuts.is_some(),
                    &mut request_active,
                    &mut terminal_request_active,
                ).await {
                    break Err(error);
                }
            }
            ack = channels.operator_acks.recv(),
                if !assembly.manifest_durable() => {
                let Some(ack) = ack else {
                    break Err(checkpoint_channel_closed("operator acks"));
                };
                if let Err(error) = accept_operator_ack(
                    ack,
                    &coordinator,
                    &mut assembly,
                    &checkpoint.status,
                ).await {
                    break Err(error);
                }
            }
            ack = channels.sink_acks.recv(),
                if !assembly.manifest_durable() => {
                let Some(ack) = ack else {
                    break Err(checkpoint_channel_closed("sink acks"));
                };
                if let Err(error) = accept_sink_ack(
                    ack,
                    &coordinator,
                    &mut assembly,
                    &checkpoint.status,
                ).await {
                    break Err(error);
                }
                #[cfg(test)]
                if let Err(error) =
                    checkpoint.inject_fault(CheckpointFaultPoint::SinkPreCommit, &cancellation)
                {
                    break Err(error);
                }
            }
            finalization = channels.sink_finalizations.recv(),
                if assembly.finalized_sink_outputs != expected_sinks => {
                let Some(finalization) = finalization else {
                    if cancellation.is_cancelled() {
                        break Ok(());
                    }
                    break Err(checkpoint_channel_closed("sink finalizations"));
                };
                if let Err(error) = accept_sink_finalization(
                    finalization,
                    &coordinator,
                    &mut assembly,
                    &checkpoint.status,
                ).await {
                    break Err(error);
                }
            }
            _ = interval.tick(), if !request_active && !terminal_sources_observed => {
                if let Err(error) = coordinator.request(CheckpointRequest::Periodic).await {
                    break Err(error);
                }
                request_active = true;
            }
        }
    };
    #[cfg(test)]
    super::super::soak::diagnostics::driver_finished(
        result.is_err(),
        &checkpoint.status.snapshot(),
        &expected_operators,
        assembly.operators.keys(),
    );
    let publication_unknown = assembly.manifest_installed_unknown;
    if !assembly.manifest_durable()
        && !publication_unknown
        && assembly.finalized_sink_outputs.is_empty()
        && let Some(epoch) = assembly.epoch
    {
        notify_sink_abort(&channels.sink_commands, epoch).await;
    }
    let sink_commit_incomplete =
        assembly.manifest_durable() && assembly.finalized_sink_outputs != expected_sinks;
    let settlement_incomplete = assembly.manifest_durable()
        && assembly.settlement_phase != Some(DurableSettlementPhase::SourceAcked);
    let sink_completion_note = if sink_commit_incomplete {
        "; sink completion was not observed"
    } else {
        ""
    };
    let sink_commit_failure = sink_commit_incomplete
        .then(|| {
            sinks
                .values()
                .find_map(SinkProgress::checkpoint_failure_sink_id)
        })
        .flatten();
    let result = match result {
        _ if publication_unknown => Err(CalcFlowError::RecoveryRequired {
            pipeline_name: checkpoint.identity.pipeline_name.clone(),
            message: format!(
                "checkpoint epoch {} was installed but publication durability is unknown",
                assembly
                    .epoch
                    .expect("indeterminate publication retains its active epoch")
                    .as_u64()
            ),
        }),
        Err(error) if settlement_incomplete => Err(CalcFlowError::RecoveryRequired {
            pipeline_name: checkpoint.identity.pipeline_name.clone(),
            message: format!(
                "checkpoint manifest settlement stopped at {:?}: {error}{sink_completion_note}",
                assembly.settlement_phase
            ),
        }),
        Err(error) if sink_commit_incomplete => Err(CalcFlowError::RecoveryRequired {
            pipeline_name: checkpoint.identity.pipeline_name.clone(),
            message: format!(
                "checkpoint manifest is durable but sink commit did not complete: {error}"
            ),
        }),
        Ok(()) if sink_commit_incomplete => Err(CalcFlowError::RecoveryRequired {
            pipeline_name: checkpoint.identity.pipeline_name.clone(),
            message: "checkpoint manifest is durable but sink commit completion was not observed"
                .into(),
        }),
        result => result,
    };
    let manual_failure = if let (Some(sink_id), Some(epoch)) = (sink_commit_failure, assembly.epoch)
    {
        ManualCheckpointFailure::SinkCommit { sink_id, epoch }
    } else {
        match &result {
            _ if core.operation_cancel_requested.load(Ordering::Acquire)
                && !assembly.manifest_durable()
                && !publication_unknown =>
            {
                ManualCheckpointFailure::Cancelled
            }
            Err(CalcFlowError::RecoveryRequired {
                pipeline_name,
                message,
            }) => ManualCheckpointFailure::RecoveryRequired {
                pipeline_name: pipeline_name.clone(),
                message: message.clone(),
            },
            Err(error) => ManualCheckpointFailure::Failed {
                category: if matches!(error, CalcFlowError::Io { .. }) {
                    ManualCheckpointFailureCategory::Io
                } else {
                    ManualCheckpointFailureCategory::Internal
                },
                epoch: assembly.epoch,
                phase: checkpoint.status.snapshot().phase,
            },
            Ok(()) if cancellation.is_cancelled() => ManualCheckpointFailure::Cancelled,
            Ok(()) => ManualCheckpointFailure::Failed {
                category: ManualCheckpointFailureCategory::Internal,
                epoch: assembly.epoch,
                phase: checkpoint.status.snapshot().phase,
            },
        }
    };
    match &manual_failure {
        ManualCheckpointFailure::Failed { category, .. } if result.is_err() => {
            let category = match category {
                ManualCheckpointFailureCategory::Timeout => CheckpointFailureCategory::Timeout,
                ManualCheckpointFailureCategory::Protocol => CheckpointFailureCategory::Protocol,
                ManualCheckpointFailureCategory::Io => CheckpointFailureCategory::Io,
                ManualCheckpointFailureCategory::Internal => CheckpointFailureCategory::Runtime,
            };
            checkpoint.status.fail_if_unset(category);
        }
        ManualCheckpointFailure::Cancelled
            if core.operation_cancel_requested.load(Ordering::Acquire) =>
        {
            checkpoint.status.cancel();
        }
        ManualCheckpointFailure::Cancelled => {
            checkpoint
                .status
                .fail_if_unset(CheckpointFailureCategory::Runtime);
        }
        // Natural completion still rejects unfinished manual requests below.
        ManualCheckpointFailure::Failed { .. }
        | ManualCheckpointFailure::RecoveryRequired { .. }
        | ManualCheckpointFailure::SinkCommit { .. } => {}
    }
    coordinator.terminate(manual_failure);
    drop(coordinator);
    let result = match (result, coordinator_task.await) {
        (Err(error), _) => Err(error),
        (Ok(()), Ok(result)) => result,
        (Ok(()), Err(error)) => Err(CalcFlowError::Internal {
            message: format!("checkpoint coordinator task join failed: {error}"),
        }),
    };
    if result.is_err() {
        assembly.fail_metrics(&metrics, assembly.terminal)?;
    } else if cancellation.is_cancelled() {
        assembly.cancel_metrics();
    }
    result
}

// Extracted from runner.rs; this transition preserves checkpoint event order.
// #lizard forgives
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "one event transition consumes the coordinator-owned checkpoint dependencies"
)]
async fn handle_checkpoint_event(
    event: CheckpointEvent,
    coordinator: &CheckpointCoordinatorHandle,
    checkpoint: &OpenedCheckpointRuntime,
    live_progress: &LiveProgressCoordinator,
    sources: &BTreeMap<String, SourceProgress>,
    sink_commands: &BTreeMap<String, mpsc::Sender<SinkCheckpointCommand>>,
    operator_commands: &BTreeMap<String, mpsc::Sender<OperatorCheckpointCommand>>,
    cancellation: &CancellationToken,
    metrics: &MetricsRecorder,
    assembly: &mut EpochManifestAssembly,
    request_active: &mut bool,
    terminal_request_active: &mut bool,
    terminal_source_cuts: &mut Option<
        BTreeMap<super::super::progress::BindingIdentity, DurableSourceCut>,
    >,
) -> crate::Result<bool> {
    #[cfg(test)]
    super::super::soak::diagnostics::checkpoint_event(&event);
    match event {
        CheckpointEvent::Started(epoch) => {
            checkpoint.status.start(epoch, *terminal_request_active);
            assembly.start(epoch, *terminal_request_active)?;
            assembly.start_metrics(metrics, *terminal_request_active)?;
            #[cfg(test)]
            if checkpoint.inject_fault(CheckpointFaultPoint::SourceAdmission, cancellation)? {
                return Ok(false);
            }
            #[cfg(test)]
            checkpoint.pause_after_started().await;
            let durable = if *terminal_request_active {
                let cuts = terminal_source_cuts.take().ok_or_else(|| {
                    checkpoint_protocol_error(epoch, "terminal source cuts are missing")
                })?;
                let durable = live_progress
                    .terminal_checkpoint_cut(epoch, &cuts, cancellation)
                    .await?;
                notify_terminal_checkpoint(operator_commands, sink_commands, epoch).await?;
                durable
            } else {
                let mut durable_cuts =
                    pause_checkpoint_sources(sources, epoch, cancellation).await?;
                if let Err(error) = add_restored_ended_source_cuts(checkpoint, &mut durable_cuts) {
                    abort_checkpoint_sources(sources, epoch);
                    return Err(error);
                }
                let promote_terminal = source_cuts_are_terminal(&durable_cuts);
                let durable_result = if promote_terminal {
                    let promotion = checkpoint
                        .status
                        .promote_terminal(epoch)
                        .and_then(|()| assembly.promote_terminal(epoch))
                        .and_then(|()| metrics.record_checkpoint_promoted_terminal());
                    if let Err(error) = promotion {
                        abort_checkpoint_sources(sources, epoch);
                        return Err(error);
                    }
                    *terminal_request_active = true;
                    match live_progress
                        .terminal_checkpoint_cut(epoch, &durable_cuts, cancellation)
                        .await
                    {
                        Ok(durable) => {
                            notify_terminal_checkpoint(operator_commands, sink_commands, epoch)
                                .await
                                .map(|()| durable)
                        }
                        Err(error) => Err(error),
                    }
                } else {
                    live_progress
                        .checkpoint_cut(epoch, &durable_cuts, cancellation)
                        .await
                };
                let durable = match durable_result {
                    Ok(durable) => durable,
                    Err(error) => {
                        abort_checkpoint_sources(sources, epoch);
                        return Err(error);
                    }
                };
                for source in sources.values() {
                    source.commit_checkpoint(epoch)?;
                }
                durable
            };
            assembly.sources = durable;
            #[cfg(test)]
            if checkpoint.inject_fault(CheckpointFaultPoint::SourceCut, cancellation)? {
                return Ok(false);
            }
            for (source_id, entry) in &assembly.sources {
                coordinator
                    .ack(CheckpointAck::source(
                        source_id,
                        epoch,
                        &checkpoint_digest(entry)?,
                    ))
                    .await?;
            }
            checkpoint
                .status
                .acknowledge_sources(epoch, assembly.sources.len());
        }
        CheckpointEvent::ReadyToPublish(epoch) => {
            publish_epoch_manifest(
                checkpoint,
                coordinator,
                sources,
                sink_commands,
                cancellation,
                metrics,
                assembly,
                epoch,
            )
            .await?;
        }
        CheckpointEvent::Completed(epoch) => {
            #[cfg(test)]
            if checkpoint.inject_fault(CheckpointFaultPoint::CompletedCommit, cancellation)? {
                assembly.settlement_phase = None;
                return Ok(false);
            }
            if let Some(error) = assembly.deferred_publication_error.take() {
                return Err(error);
            }
            let terminal = assembly.terminal;
            assembly.complete(epoch)?;
            assembly.complete_metrics(metrics, terminal)?;
            retain_completed_epoch(checkpoint, cancellation, metrics, epoch).await?;
            *request_active = false;
            *terminal_request_active = false;
            if terminal {
                return Ok(true);
            }
        }
        CheckpointEvent::Failed(epoch, phase) => {
            checkpoint.status.fail(if phase == "timeout" {
                CheckpointFailureCategory::Timeout
            } else {
                CheckpointFailureCategory::Protocol
            });
            return Err(checkpoint_protocol_error(
                epoch,
                &format!("coordinator failed during {phase}"),
            ));
        }
        CheckpointEvent::PhaseAdvanced(epoch, phase) => {
            // Publication records this phase synchronously because settlement
            // can block on source acknowledgement before this event is read.
            if phase != CheckpointPhase::ManifestDurable || !assembly.manifest_durable() {
                checkpoint.status.advance(epoch, phase);
                assembly.advance_metrics(metrics, phase)?;
            }
        }
    }
    Ok(false)
}

pub(super) fn source_cuts_are_terminal(
    cuts: &BTreeMap<super::super::progress::BindingIdentity, DurableSourceCut>,
) -> bool {
    !cuts.is_empty() && cuts.values().all(|cut| cut.ended)
}

async fn notify_terminal_checkpoint(
    operator_commands: &BTreeMap<String, mpsc::Sender<OperatorCheckpointCommand>>,
    sink_commands: &BTreeMap<String, mpsc::Sender<SinkCheckpointCommand>>,
    epoch: Epoch,
) -> crate::Result<()> {
    for command in operator_commands.values() {
        command
            .send(OperatorCheckpointCommand::Terminal(epoch))
            .await
            .map_err(|_| checkpoint_channel_closed("operator commands"))?;
    }
    for command in sink_commands.values() {
        command
            .send(SinkCheckpointCommand::Terminal(epoch))
            .await
            .map_err(|_| checkpoint_channel_closed("sink commands"))?;
    }
    Ok(())
}

// Extracted from runner.rs; manifest publication retains its durable ordering.
// #lizard forgives
#[allow(
    clippy::too_many_arguments,
    reason = "manifest publication owns the checkpoint, participant, cancellation, metrics, and epoch boundaries"
)]
async fn publish_epoch_manifest(
    checkpoint: &OpenedCheckpointRuntime,
    coordinator: &CheckpointCoordinatorHandle,
    sources: &BTreeMap<String, SourceProgress>,
    sink_commands: &BTreeMap<String, mpsc::Sender<SinkCheckpointCommand>>,
    cancellation: &CancellationToken,
    metrics: &MetricsRecorder,
    assembly: &mut EpochManifestAssembly,
    epoch: Epoch,
) -> crate::Result<()> {
    assembly.expect_epoch(epoch)?;
    checkpoint
        .status
        .advance(epoch, CheckpointPhase::SinksPrecommitted);
    assembly.advance_metrics(metrics, CheckpointPhase::SinksPrecommitted)?;
    let manifest = build_epoch_manifest(checkpoint, assembly, epoch)?;
    let (state_bytes, manifest_bytes) = checkpoint_manifest_sizes(&manifest)?;
    metrics.record_checkpoint_manifest(state_bytes, manifest_bytes)?;
    let publication = checkpoint
        .transaction
        .publish_cancellable(
            PreparedEpochManifest {
                manifest,
                staged_segments: BTreeMap::new(),
            },
            cancellation,
        )
        .await;
    let publication = match publication {
        Ok(publication) => publication,
        Err(_) if cancellation.is_cancelled() => return Ok(()),
        Err(error) => return Err(error),
    };
    match publication {
        ManifestPublication::Durable => {
            assembly.settlement_phase = Some(DurableSettlementPhase::Published);
            checkpoint
                .status
                .advance(epoch, CheckpointPhase::ManifestDurable);
            assembly.advance_metrics(metrics, CheckpointPhase::ManifestDurable)?;
            settle_durable_manifest(
                coordinator,
                sources,
                &assembly.sources,
                sink_commands,
                DurableSettlementRequest {
                    epoch,
                    terminal: assembly.terminal,
                    acknowledgement_timeout: checkpoint.config.checkpoint_timeout,
                    phase: assembly
                        .settlement_phase
                        .as_mut()
                        .expect("durable publication starts settlement"),
                },
            )
            .await
        }
        ManifestPublication::Installed {
            parent_synced,
            error,
        } => {
            if parent_synced {
                assembly.settlement_phase = Some(DurableSettlementPhase::Published);
                checkpoint
                    .status
                    .advance(epoch, CheckpointPhase::ManifestDurable);
                assembly.advance_metrics(metrics, CheckpointPhase::ManifestDurable)?;
                if !cancellation.is_cancelled() {
                    assembly.deferred_publication_error = Some(error);
                }
                settle_durable_manifest(
                    coordinator,
                    sources,
                    &assembly.sources,
                    sink_commands,
                    DurableSettlementRequest {
                        epoch,
                        terminal: assembly.terminal,
                        acknowledgement_timeout: checkpoint.config.checkpoint_timeout,
                        phase: assembly
                            .settlement_phase
                            .as_mut()
                            .expect("durable publication starts settlement"),
                    },
                )
                .await?;
            } else {
                assembly.manifest_installed_unknown = true;
                checkpoint.status.installed_unknown(epoch);
                notify_sink_preserve(sink_commands, epoch).await?;
                return if cancellation.is_cancelled() {
                    Ok(())
                } else {
                    Err(error)
                };
            }
            Ok(())
        }
    }
}

pub(super) async fn settle_durable_manifest(
    coordinator: &CheckpointCoordinatorHandle,
    sources: &BTreeMap<String, SourceProgress>,
    source_entries: &BTreeMap<String, SourceManifestEntry>,
    sink_commands: &BTreeMap<String, mpsc::Sender<SinkCheckpointCommand>>,
    request: DurableSettlementRequest<'_>,
) -> crate::Result<()> {
    let DurableSettlementRequest {
        epoch,
        terminal,
        acknowledgement_timeout,
        phase,
    } = request;
    if *phase != DurableSettlementPhase::Published {
        return Err(checkpoint_protocol_error(
            epoch,
            "durable manifest settlement did not start at publication",
        ));
    }
    notify_sink_manifest_durable(sink_commands, epoch, terminal).await?;
    phase.advance(
        DurableSettlementPhase::Published,
        DurableSettlementPhase::SinksCommanded,
        epoch,
    )?;
    coordinator.manifest_durable(epoch).await?;
    phase.advance(
        DurableSettlementPhase::SinksCommanded,
        DurableSettlementPhase::CoordinatorDurable,
        epoch,
    )?;
    tokio::time::timeout(
        acknowledgement_timeout,
        acknowledge_durable_source_cursors(sources, source_entries),
    )
    .await
    .map_err(|_| CalcFlowError::Internal {
        message: format!(
            "durable source acknowledgement timed out for checkpoint epoch {}",
            epoch.as_u64()
        ),
    })??;
    phase.advance(
        DurableSettlementPhase::CoordinatorDurable,
        DurableSettlementPhase::SourceAcked,
        epoch,
    )
}

async fn acknowledge_durable_source_cursors(
    sources: &BTreeMap<String, SourceProgress>,
    entries: &BTreeMap<String, SourceManifestEntry>,
) -> crate::Result<()> {
    for (source_id, entry) in entries {
        let Some(source) = sources.get(source_id) else {
            if entry.ended {
                continue;
            }
            return Err(CalcFlowError::Internal {
                message: format!("durable manifest names unknown source {source_id:?}"),
            });
        };
        source
            .acknowledge_durable_cursor(entry.cursor.as_ref())
            .await?;
    }
    Ok(())
}

async fn retain_completed_epoch(
    checkpoint: &OpenedCheckpointRuntime,
    cancellation: &CancellationToken,
    metrics: &MetricsRecorder,
    epoch: Epoch,
) -> crate::Result<()> {
    checkpoint.status.sinks_committed(epoch);
    #[cfg(test)]
    if checkpoint.inject_fault(CheckpointFaultPoint::Retention, cancellation)? {
        return Ok(());
    }
    let retained = checkpoint
        .transaction
        .retain_cancellable(&checkpoint.identity, None, cancellation)
        .await;
    if cancellation.is_cancelled() {
        return Ok(());
    }
    let report = match retained {
        Ok(report) => report,
        Err(error) => {
            checkpoint
                .status
                .fail(CheckpointFailureCategory::Maintenance);
            return Err(error);
        }
    };
    #[cfg(not(test))]
    let _ = cancellation;
    metrics.record_checkpoint_orphan_cleanup(report.removed_orphan_segments)?;
    checkpoint.status.complete(epoch);
    Ok(())
}

pub(super) async fn notify_sink_manifest_durable(
    sink_commands: &BTreeMap<String, mpsc::Sender<SinkCheckpointCommand>>,
    epoch: Epoch,
    terminal: bool,
) -> crate::Result<()> {
    let mut first_error = None;
    for sender in sink_commands.values() {
        let command = if terminal {
            SinkCheckpointCommand::TerminalManifestDurable(epoch)
        } else {
            SinkCheckpointCommand::ManifestDurable(epoch)
        };
        if sender.send(command).await.is_err() && first_error.is_none() {
            first_error = Some(checkpoint_channel_closed("sink commands"));
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

async fn notify_sink_preserve(
    sink_commands: &BTreeMap<String, mpsc::Sender<SinkCheckpointCommand>>,
    epoch: Epoch,
) -> crate::Result<()> {
    let mut first_error = None;
    for sender in sink_commands.values() {
        let sent = sender.send(SinkCheckpointCommand::Preserve(epoch)).await;
        if sent.is_err() && first_error.is_none() {
            first_error = Some(checkpoint_channel_closed("sink commands"));
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

pub(super) async fn notify_sink_abort(
    sink_commands: &BTreeMap<String, mpsc::Sender<SinkCheckpointCommand>>,
    epoch: Epoch,
) {
    futures::future::join_all(sink_commands.values().map(|sender| async move {
        let _ = tokio::time::timeout(
            CONNECTOR_CLOSE_TIMEOUT,
            sender.send(SinkCheckpointCommand::Abort(epoch)),
        )
        .await;
    }))
    .await;
}

async fn wait_for_terminal_source_cuts(
    sources: &BTreeMap<String, SourceProgress>,
    cancellation: &CancellationToken,
) -> crate::Result<BTreeMap<super::super::progress::BindingIdentity, DurableSourceCut>> {
    let cuts = try_join_all(sources.iter().map(|(source_id, progress)| async move {
        let cut = progress.wait_for_terminal_cut(cancellation).await?;
        Ok::<_, CalcFlowError>((
            super::super::progress::BindingIdentity::new(source_id.as_str())?,
            cut.durable(source_id)?,
        ))
    }))
    .await?;
    Ok(cuts.into_iter().collect())
}

#[allow(
    clippy::too_many_arguments,
    reason = "terminal admission compares every prepared participant set atomically"
)]
pub(super) async fn maybe_request_terminal_checkpoint(
    coordinator: &CheckpointCoordinatorHandle,
    expected_operators: &BTreeSet<String>,
    expected_sinks: &BTreeSet<String>,
    ready_operators: &BTreeSet<String>,
    ready_sinks: &BTreeSet<String>,
    sources_ready: bool,
    request_active: &mut bool,
    terminal_request_active: &mut bool,
) -> crate::Result<()> {
    if sources_ready
        && ready_operators == expected_operators
        && ready_sinks == expected_sinks
        && !*request_active
    {
        coordinator.request(CheckpointRequest::Terminal).await?;
        *request_active = true;
        *terminal_request_active = true;
    }
    Ok(())
}

async fn pause_checkpoint_sources(
    sources: &BTreeMap<String, SourceProgress>,
    epoch: Epoch,
    cancellation: &CancellationToken,
) -> crate::Result<BTreeMap<super::super::progress::BindingIdentity, DurableSourceCut>> {
    let cuts = try_join_all(sources.iter().map(|(source_id, progress)| async move {
        let cut = progress.barrier(epoch, cancellation).await?;
        Ok::<_, CalcFlowError>((
            super::super::progress::BindingIdentity::new(source_id.as_str())?,
            cut.durable(source_id)?,
        ))
    }))
    .await;
    match cuts {
        Ok(cuts) => Ok(cuts.into_iter().collect()),
        Err(error) => {
            abort_checkpoint_sources(sources, epoch);
            Err(error)
        }
    }
}

fn abort_checkpoint_sources(sources: &BTreeMap<String, SourceProgress>, epoch: Epoch) {
    for source in sources.values() {
        let _ = source.abort_checkpoint(epoch);
    }
}

fn admit_operator_ack(ack: &OperatorCheckpointAck) -> crate::Result<()> {
    if let Some(credit) = &ack.capture_credit {
        super::super::operator_task::join_checkpoint_credit::admit_ack_clone(
            credit,
            &ack.node_id,
            &ack.state,
        )?;
    }
    Ok(())
}

async fn accept_operator_ack(
    ack: OperatorCheckpointAck,
    coordinator: &CheckpointCoordinatorHandle,
    assembly: &mut EpochManifestAssembly,
    status: &CheckpointStatusHandle,
) -> crate::Result<()> {
    assembly.expect_epoch(ack.epoch)?;
    #[cfg(test)]
    super::super::soak::diagnostics::operator_ack_received(&ack.node_id, ack.epoch);
    admit_operator_ack(&ack)?;
    insert_identical(
        &mut assembly.operators,
        &ack.node_id,
        ack.state.clone(),
        ack.epoch,
        "operator",
    )?;
    if let Some(credit) = &ack.capture_credit {
        assembly
            .operator_checkpoint_credits
            .insert(ack.node_id.clone(), Arc::clone(credit));
    }
    if let Some(working) = ack.working {
        assembly
            .working_states
            .entry(ack.node_id.clone())
            .or_insert(working);
    }
    coordinator
        .ack(CheckpointAck::operator(
            &ack.node_id,
            ack.epoch,
            &checkpoint_digest(&ack.state)?,
        ))
        .await?;
    status.acknowledge_operators(ack.epoch, assembly.operators.len());
    Ok(())
}

async fn accept_sink_ack(
    ack: SinkCheckpointAck,
    coordinator: &CheckpointCoordinatorHandle,
    assembly: &mut EpochManifestAssembly,
    status: &CheckpointStatusHandle,
) -> crate::Result<()> {
    assembly.expect_epoch(ack.epoch)?;
    insert_identical(
        &mut assembly.sink_outputs,
        &ack.output_id,
        ack.sinks.clone(),
        ack.epoch,
        "sink output",
    )?;
    coordinator
        .ack(CheckpointAck::sink_precommit(
            &ack.output_id,
            ack.epoch,
            &checkpoint_digest(&ack.sinks)?,
        ))
        .await?;
    status.acknowledge_sink_precommits(ack.epoch, assembly.sink_outputs.len());
    Ok(())
}

async fn accept_sink_finalization(
    finalization: SinkFinalizeAck,
    coordinator: &CheckpointCoordinatorHandle,
    assembly: &mut EpochManifestAssembly,
    status: &CheckpointStatusHandle,
) -> crate::Result<()> {
    assembly.expect_epoch(finalization.epoch)?;
    assembly
        .finalized_sink_outputs
        .insert(finalization.output_id.clone());
    coordinator
        .ack(CheckpointAck::sink_commit(
            &finalization.output_id,
            finalization.epoch,
        ))
        .await?;
    status.acknowledge_sink_commits(finalization.epoch, assembly.finalized_sink_outputs.len());
    Ok(())
}

fn insert_identical<T: Clone + Eq>(
    entries: &mut BTreeMap<String, T>,
    id: &str,
    value: T,
    epoch: Epoch,
    kind: &str,
) -> crate::Result<()> {
    match entries.get(id) {
        Some(previous) if previous == &value => Ok(()),
        Some(_) => Err(checkpoint_protocol_error(
            epoch,
            &format!("conflicting duplicate {kind} ack for {id:?}"),
        )),
        None => {
            entries.insert(id.into(), value);
            Ok(())
        }
    }
}

fn build_epoch_manifest(
    checkpoint: &OpenedCheckpointRuntime,
    assembly: &EpochManifestAssembly,
    epoch: Epoch,
) -> crate::Result<CheckpointManifest> {
    let mut sinks = BTreeMap::new();
    for output_sinks in assembly.sink_outputs.values() {
        for (sink_id, entry) in output_sinks {
            if sinks.insert(sink_id.clone(), entry.clone()).is_some() {
                return Err(checkpoint_protocol_error(
                    epoch,
                    &format!("sink ID {sink_id:?} is bound to more than one output"),
                ));
            }
        }
    }
    if assembly.sources.keys().cloned().collect::<BTreeSet<_>>() != checkpoint.identity.source_ids
        || assembly.operators.keys().cloned().collect::<BTreeSet<_>>()
            != checkpoint.identity.operator_ids
        || sinks.keys().cloned().collect::<BTreeSet<_>>() != checkpoint.identity.sink_ids
    {
        return Err(checkpoint_protocol_error(
            epoch,
            "manifest participant IDs do not match the prepared job",
        ));
    }
    CheckpointManifest::new(CheckpointManifestFields {
        pipeline_name: checkpoint.identity.pipeline_name.clone(),
        pipeline_fingerprint: checkpoint.identity.pipeline_fingerprint.clone(),
        runtime_config_hash: checkpoint.identity.runtime_config_hash.clone(),
        epoch,
        created_at: Utc::now(),
        recovery_status: RecoveryStatus::Final,
        sources: assembly.sources.clone(),
        operators: assembly.operators.clone(),
        sinks,
        static_inputs: checkpoint.identity.static_inputs.clone(),
    })
}

fn checkpoint_manifest_sizes(manifest: &CheckpointManifest) -> crate::Result<(u64, u64)> {
    let state_bytes = manifest
        .operators()
        .values()
        .flat_map(|operator| operator.segments.iter())
        .chain(
            manifest
                .sinks()
                .values()
                .flat_map(|sink| sink.segments.iter()),
        )
        .try_fold(0_u64, |total, handle| {
            total
                .checked_add(handle.byte_len())
                .ok_or_else(|| CalcFlowError::InvalidArgument {
                    field: "runtime.metrics.checkpoint.state_bytes".into(),
                    message: "counter overflow".into(),
                })
        })?;
    let manifest_bytes = u64::try_from(manifest.canonical_bytes()?.len()).map_err(|_| {
        CalcFlowError::InvalidArgument {
            field: "runtime.metrics.checkpoint.manifest_bytes".into(),
            message: "counter overflow".into(),
        }
    })?;
    Ok((state_bytes, manifest_bytes))
}

fn checkpoint_digest(value: &impl Serialize) -> crate::Result<String> {
    let value = serde_json::to_value(value).map_err(|error| CalcFlowError::Format {
        message: error.to_string(),
    })?;
    let canonical = crate::canonical_json(&value)?;
    Ok(hex::encode(Sha256::digest(canonical.as_bytes())))
}

fn checkpoint_channel_closed(channel: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("checkpoint {channel} channel closed"),
    }
}
