//! Payload-free, deterministic metrics for the crate-private M2 runtime.
//!
//! Recorder methods accept only stable component IDs selected during pure
//! preflight plus numeric costs, sequences, and durations. Batch IDs, cursor
//! payloads, attributes, row values, secrets, and arbitrary labels have no
//! representation in this module.

use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;

use super::{
    ChannelMetrics, EnvelopeCost, StreamMessage,
    checkpoint::coordinator::CheckpointPhase,
    runner::{ContinuousJobState, FailureOrigin, RuntimeFailure, TerminalCause},
};
use crate::{CalcFlowError, EdgeBudget, Result};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct JobMetrics {
    pub(crate) terminal_state: Option<ContinuousJobState>,
    pub(crate) terminal_cause: Option<TerminalCause>,
    pub(crate) task_errors: u64,
    pub(crate) reaper_joins: u64,
    pub(crate) reaper_join_errors: u64,
    pub(crate) abandoned_runner_drops: u64,
    pub(crate) metrics_overflowed: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct EdgeRuntimeMetrics {
    pub(crate) message_slot_limit: usize,
    pub(crate) channel: ChannelMetrics,
    pub(crate) input_batches: u64,
    pub(crate) input_rows: u64,
    pub(crate) input_bytes: u64,
    pub(crate) output_batches: u64,
    pub(crate) output_rows: u64,
    pub(crate) output_bytes: u64,
    pub(crate) drop_invariant_violated: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SourceMetrics {
    pub(crate) poll_count: u64,
    pub(crate) data_batches: u64,
    pub(crate) data_rows: u64,
    pub(crate) data_bytes: u64,
    pub(crate) fully_fanned_out_batches: u64,
    pub(crate) fully_fanned_out_rows: u64,
    pub(crate) fully_fanned_out_bytes: u64,
    pub(crate) latest_sequence: Option<u64>,
    pub(crate) ended: bool,
    pub(crate) errors: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct OperatorMetrics {
    pub(crate) input_batches: u64,
    pub(crate) input_rows: u64,
    pub(crate) input_bytes: u64,
    pub(crate) fully_fanned_out_batches: u64,
    pub(crate) fully_fanned_out_rows: u64,
    pub(crate) fully_fanned_out_bytes: u64,
    pub(crate) processing_duration: Duration,
    pub(crate) errors: u64,
    pub(crate) late_rows: u64,
    pub(crate) affected_batches: u64,
    pub(crate) max_lateness_micros: Option<u64>,
    pub(crate) null_event_time_rows: u64,
    pub(crate) null_event_time_batches: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SinkMetrics {
    pub(crate) delivered_batches: u64,
    pub(crate) delivered_rows: u64,
    pub(crate) delivered_bytes: u64,
    pub(crate) write_duration: Duration,
    pub(crate) errors: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct CheckpointMetrics {
    pub(crate) requested: u64,
    pub(crate) completed: u64,
    pub(crate) failed: u64,
    pub(crate) terminal_requested: u64,
    pub(crate) terminal_completed: u64,
    pub(crate) terminal_failed: u64,
    pub(crate) phase_duration: BTreeMap<CheckpointPhase, Duration>,
    pub(crate) total_duration: Duration,
    pub(crate) alignment_duration: Duration,
    pub(crate) state_bytes: u64,
    pub(crate) manifest_bytes: u64,
    pub(crate) restore_duration: Duration,
    pub(crate) sink_commit_retries: u64,
    pub(crate) orphan_segments_removed: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct M2MetricsSnapshot {
    pub(crate) job: JobMetrics,
    pub(crate) edges: BTreeMap<String, EdgeRuntimeMetrics>,
    pub(crate) sources: BTreeMap<String, SourceMetrics>,
    pub(crate) nodes: BTreeMap<String, OperatorMetrics>,
    pub(crate) sinks: BTreeMap<String, SinkMetrics>,
    pub(crate) checkpoints: CheckpointMetrics,
}

pub(crate) trait MetricsClock: Send + Sync {
    fn now(&self) -> Duration;
}

struct MonotonicMetricsClock {
    origin: Instant,
}

impl Default for MonotonicMetricsClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl MetricsClock for MonotonicMetricsClock {
    fn now(&self) -> Duration {
        self.origin.elapsed()
    }
}

#[derive(Clone)]
pub(crate) struct MetricsTimer {
    started: Duration,
    clock: Arc<dyn MetricsClock>,
}

impl MetricsTimer {
    pub(super) fn elapsed(&self, component_id: &str, counter: &'static str) -> Result<Duration> {
        self.clock
            .now()
            .checked_sub(self.started)
            .ok_or_else(|| metrics_error(component_id, counter, "clock moved backwards"))
    }
}

struct MetricsInner {
    snapshot: Mutex<M2MetricsSnapshot>,
    clock: Arc<dyn MetricsClock>,
    strict_ids: bool,
}

/// A cloneable handle over one bounded stable-ID metrics registry.
#[derive(Clone)]
pub(crate) struct MetricsRecorder(Arc<MetricsInner>);

impl Default for MetricsRecorder {
    /// Test/legacy paths use a payload-free no-op recorder. Production M2
    /// wiring always constructs a strict registry from the compiled plan.
    fn default() -> Self {
        Self(Arc::new(MetricsInner {
            snapshot: Mutex::new(M2MetricsSnapshot::default()),
            clock: Arc::new(MonotonicMetricsClock::default()),
            strict_ids: false,
        }))
    }
}

impl MetricsRecorder {
    pub(crate) fn new(
        edges: impl IntoIterator<Item = (String, EdgeBudget)>,
        sources: impl IntoIterator<Item = String>,
        nodes: impl IntoIterator<Item = String>,
        sinks: impl IntoIterator<Item = String>,
    ) -> Self {
        Self::with_clock(
            edges,
            sources,
            nodes,
            sinks,
            Arc::new(MonotonicMetricsClock::default()),
        )
    }

    fn with_clock(
        edges: impl IntoIterator<Item = (String, EdgeBudget)>,
        sources: impl IntoIterator<Item = String>,
        nodes: impl IntoIterator<Item = String>,
        sinks: impl IntoIterator<Item = String>,
        clock: Arc<dyn MetricsClock>,
    ) -> Self {
        let snapshot = M2MetricsSnapshot {
            job: JobMetrics::default(),
            edges: edges
                .into_iter()
                .map(|(id, budget)| {
                    (
                        id,
                        EdgeRuntimeMetrics {
                            message_slot_limit: budget.max_rows,
                            ..EdgeRuntimeMetrics::default()
                        },
                    )
                })
                .collect(),
            sources: sources
                .into_iter()
                .map(|id| (id, SourceMetrics::default()))
                .collect(),
            nodes: nodes
                .into_iter()
                .map(|id| (id, OperatorMetrics::default()))
                .collect(),
            sinks: sinks
                .into_iter()
                .map(|id| (id, SinkMetrics::default()))
                .collect(),
            checkpoints: CheckpointMetrics::default(),
        };
        Self(Arc::new(MetricsInner {
            snapshot: Mutex::new(snapshot),
            clock,
            strict_ids: true,
        }))
    }

    pub(crate) fn snapshot(&self) -> M2MetricsSnapshot {
        self.0.snapshot.lock().clone()
    }

    pub(crate) fn timer(&self) -> MetricsTimer {
        MetricsTimer {
            started: self.0.clock.now(),
            clock: Arc::clone(&self.0.clock),
        }
    }

    pub(crate) fn record_source_poll(&self, source_id: &str) -> Result<()> {
        self.with_source(source_id, |metrics| {
            checked_inc(&mut metrics.poll_count, source_id, "poll_count")
        })
    }

    pub(crate) fn record_source_data(
        &self,
        source_id: &str,
        cost: EnvelopeCost,
        sequence: u64,
    ) -> Result<()> {
        self.with_source(source_id, |metrics| {
            let (batches, rows, bytes) = traffic_values(cost)?;
            let next_batches =
                checked_sum(metrics.data_batches, batches, source_id, "data_batches")?;
            let next_rows = checked_sum(metrics.data_rows, rows, source_id, "data_rows")?;
            let next_bytes = checked_sum(metrics.data_bytes, bytes, source_id, "data_bytes")?;
            metrics.data_batches = next_batches;
            metrics.data_rows = next_rows;
            metrics.data_bytes = next_bytes;
            metrics.latest_sequence = Some(sequence);
            Ok(())
        })
    }

    pub(crate) fn record_source_output(&self, source_id: &str, cost: EnvelopeCost) -> Result<()> {
        self.with_source(source_id, |metrics| {
            checked_traffic(
                &mut metrics.fully_fanned_out_batches,
                &mut metrics.fully_fanned_out_rows,
                &mut metrics.fully_fanned_out_bytes,
                cost,
                source_id,
                "fully_fanned_out",
            )
        })
    }

    pub(crate) fn record_source_end(&self, source_id: &str) -> Result<()> {
        self.with_source(source_id, |metrics| {
            metrics.ended = true;
            Ok(())
        })
    }

    pub(crate) fn record_operator_input(&self, node_id: &str, cost: EnvelopeCost) -> Result<()> {
        self.with_node(node_id, |metrics| {
            checked_traffic(
                &mut metrics.input_batches,
                &mut metrics.input_rows,
                &mut metrics.input_bytes,
                cost,
                node_id,
                "input",
            )
        })
    }

    pub(crate) fn record_operator_output(&self, node_id: &str, cost: EnvelopeCost) -> Result<()> {
        self.with_node(node_id, |metrics| {
            checked_traffic(
                &mut metrics.fully_fanned_out_batches,
                &mut metrics.fully_fanned_out_rows,
                &mut metrics.fully_fanned_out_bytes,
                cost,
                node_id,
                "fully_fanned_out",
            )
        })
    }

    pub(crate) fn record_operator_processing(
        &self,
        node_id: &str,
        timer: &MetricsTimer,
    ) -> Result<()> {
        let elapsed = timer.elapsed(node_id, "processing_duration")?;
        self.with_node(node_id, |metrics| {
            metrics.processing_duration = metrics
                .processing_duration
                .checked_add(elapsed)
                .ok_or_else(|| metrics_error(node_id, "processing_duration", "counter overflow"))?;
            Ok(())
        })
    }

    pub(crate) fn observe_operator_window_metrics(
        &self,
        node_id: &str,
        progress: &super::operator_task::OperatorProgressSnapshot,
    ) -> Result<()> {
        self.with_node(node_id, |metrics| {
            metrics.late_rows = progress.late_rows;
            metrics.affected_batches = progress.affected_batches;
            metrics.max_lateness_micros = progress.max_lateness_micros;
            metrics.null_event_time_rows = progress.null_event_time_rows;
            metrics.null_event_time_batches = progress.null_event_time_batches;
            Ok(())
        })
    }

    pub(crate) fn record_sink_delivery(
        &self,
        sink_id: &str,
        cost: EnvelopeCost,
        timer: &MetricsTimer,
    ) -> Result<()> {
        let elapsed = timer.elapsed(sink_id, "write_duration")?;
        self.with_sink(sink_id, |metrics| {
            let (batches, rows, bytes) = traffic_values(cost)?;
            let next_batches = checked_sum(
                metrics.delivered_batches,
                batches,
                sink_id,
                "delivered_batches",
            )?;
            let next_rows = checked_sum(metrics.delivered_rows, rows, sink_id, "delivered_rows")?;
            let next_bytes =
                checked_sum(metrics.delivered_bytes, bytes, sink_id, "delivered_bytes")?;
            let next_duration = metrics
                .write_duration
                .checked_add(elapsed)
                .ok_or_else(|| metrics_error(sink_id, "write_duration", "counter overflow"))?;
            metrics.delivered_batches = next_batches;
            metrics.delivered_rows = next_rows;
            metrics.delivered_bytes = next_bytes;
            metrics.write_duration = next_duration;
            Ok(())
        })
    }

    /// Records one successful enqueue while the channel queue lock is held.
    pub(super) fn record_edge_enqueue(
        &self,
        edge_id: &str,
        traffic: EdgeTraffic,
        blocked_elapsed: Option<Duration>,
    ) -> Result<()> {
        self.with_edge(edge_id, |metrics| {
            let (batches, rows, bytes) = next_edge_input(metrics, traffic, edge_id)?;
            let channel =
                next_edge_enqueue_channel(&metrics.channel, traffic, blocked_elapsed, edge_id)?;
            metrics.input_batches = batches;
            metrics.input_rows = rows;
            metrics.input_bytes = bytes;
            metrics.channel = channel;
            Ok(())
        })
    }

    pub(super) fn record_edge_blocked(&self, edge_id: &str) -> Result<()> {
        self.with_edge(edge_id, |metrics| {
            checked_inc(&mut metrics.channel.blocked_sends, edge_id, "blocked_sends")
        })
    }

    /// Records one successful dequeue before the infallible queue pop and
    /// reservation release, while the channel queue lock is held.
    pub(super) fn record_edge_dequeue(&self, edge_id: &str, traffic: EdgeTraffic) -> Result<()> {
        self.with_edge(edge_id, |metrics| {
            let (batches, rows, bytes) = next_edge_output(metrics, traffic, edge_id)?;
            let channel = next_edge_dequeue_channel(&metrics.channel, traffic, edge_id)?;
            metrics.output_batches = batches;
            metrics.output_rows = rows;
            metrics.output_bytes = bytes;
            metrics.channel = channel;
            Ok(())
        })
    }

    pub(super) fn record_edge_drop(&self, edge_id: &str, released: EnvelopeCost) {
        let mut snapshot = self.0.snapshot.lock();
        if let Some(metrics) = snapshot.edges.get_mut(edge_id) {
            let decremented = metrics
                .channel
                .queue_depth
                .checked_sub(released.messages())
                .zip(metrics.channel.charged_rows.checked_sub(released.rows()))
                .zip(metrics.channel.charged_bytes.checked_sub(released.bytes()));
            match decremented {
                Some(((queue_depth, charged_rows), charged_bytes)) => {
                    metrics.channel.queue_depth = queue_depth;
                    metrics.channel.charged_rows = charged_rows;
                    metrics.channel.charged_bytes = charged_bytes;
                }
                None => metrics.drop_invariant_violated = true,
            }
        }
    }

    pub(crate) fn record_terminal(&self, state: ContinuousJobState, cause: TerminalCause) {
        let mut snapshot = self.0.snapshot.lock();
        if snapshot.job.terminal_state.is_none() {
            snapshot.job.terminal_state = Some(state);
            snapshot.job.terminal_cause = Some(cause);
        }
    }

    pub(crate) fn record_checkpoint_requested(&self, terminal: bool) -> Result<()> {
        let mut snapshot = self.0.snapshot.lock();
        let requested = checked_sum(snapshot.checkpoints.requested, 1, "checkpoint", "requested")?;
        let terminal_requested = if terminal {
            checked_sum(
                snapshot.checkpoints.terminal_requested,
                1,
                "checkpoint",
                "terminal_requested",
            )?
        } else {
            snapshot.checkpoints.terminal_requested
        };
        snapshot.checkpoints.requested = requested;
        snapshot.checkpoints.terminal_requested = terminal_requested;
        Ok(())
    }

    pub(crate) fn record_checkpoint_promoted_terminal(&self) -> Result<()> {
        let mut snapshot = self.0.snapshot.lock();
        snapshot.checkpoints.terminal_requested = checked_sum(
            snapshot.checkpoints.terminal_requested,
            1,
            "checkpoint",
            "terminal_requested",
        )?;
        Ok(())
    }

    pub(crate) fn record_checkpoint_phase(
        &self,
        phase: CheckpointPhase,
        elapsed: Duration,
    ) -> Result<()> {
        let mut snapshot = self.0.snapshot.lock();
        let previous = snapshot
            .checkpoints
            .phase_duration
            .get(&phase)
            .copied()
            .unwrap_or_default();
        let duration = checked_duration(previous, elapsed, "phase_duration")?;
        let alignment_duration = if phase == CheckpointPhase::SourcesCut {
            checked_duration(
                snapshot.checkpoints.alignment_duration,
                elapsed,
                "alignment_duration",
            )?
        } else {
            snapshot.checkpoints.alignment_duration
        };
        snapshot.checkpoints.phase_duration.insert(phase, duration);
        snapshot.checkpoints.alignment_duration = alignment_duration;
        Ok(())
    }

    pub(crate) fn record_checkpoint_manifest(
        &self,
        state_bytes: u64,
        manifest_bytes: u64,
    ) -> Result<()> {
        let mut snapshot = self.0.snapshot.lock();
        let next_state = checked_sum(
            snapshot.checkpoints.state_bytes,
            state_bytes,
            "checkpoint",
            "state_bytes",
        )?;
        let next_manifest = checked_sum(
            snapshot.checkpoints.manifest_bytes,
            manifest_bytes,
            "checkpoint",
            "manifest_bytes",
        )?;
        snapshot.checkpoints.state_bytes = next_state;
        snapshot.checkpoints.manifest_bytes = next_manifest;
        Ok(())
    }

    pub(crate) fn record_checkpoint_completed(
        &self,
        terminal: bool,
        elapsed: Duration,
    ) -> Result<()> {
        let mut snapshot = self.0.snapshot.lock();
        let completed = checked_sum(snapshot.checkpoints.completed, 1, "checkpoint", "completed")?;
        let terminal_completed = if terminal {
            checked_sum(
                snapshot.checkpoints.terminal_completed,
                1,
                "checkpoint",
                "terminal_completed",
            )?
        } else {
            snapshot.checkpoints.terminal_completed
        };
        let duration = checked_duration(
            snapshot.checkpoints.total_duration,
            elapsed,
            "total_duration",
        )?;
        snapshot.checkpoints.completed = completed;
        snapshot.checkpoints.terminal_completed = terminal_completed;
        snapshot.checkpoints.total_duration = duration;
        Ok(())
    }

    pub(crate) fn record_checkpoint_failed(&self, terminal: bool) -> Result<()> {
        let mut snapshot = self.0.snapshot.lock();
        let failed = checked_sum(snapshot.checkpoints.failed, 1, "checkpoint", "failed")?;
        let terminal_failed = if terminal {
            checked_sum(
                snapshot.checkpoints.terminal_failed,
                1,
                "checkpoint",
                "terminal_failed",
            )?
        } else {
            snapshot.checkpoints.terminal_failed
        };
        snapshot.checkpoints.failed = failed;
        snapshot.checkpoints.terminal_failed = terminal_failed;
        Ok(())
    }

    pub(crate) fn record_checkpoint_restore(
        &self,
        elapsed: Duration,
        sink_commit_retries: usize,
    ) -> Result<()> {
        let retries = u64::try_from(sink_commit_retries)
            .map_err(|_| metrics_error("checkpoint", "sink_commit_retries", "counter overflow"))?;
        let mut snapshot = self.0.snapshot.lock();
        let restore_duration = checked_duration(
            snapshot.checkpoints.restore_duration,
            elapsed,
            "restore_duration",
        )?;
        let retries = checked_sum(
            snapshot.checkpoints.sink_commit_retries,
            retries,
            "checkpoint",
            "sink_commit_retries",
        )?;
        snapshot.checkpoints.restore_duration = restore_duration;
        snapshot.checkpoints.sink_commit_retries = retries;
        Ok(())
    }

    pub(crate) fn record_checkpoint_orphan_cleanup(&self, removed: usize) -> Result<()> {
        let removed = u64::try_from(removed).map_err(|_| {
            metrics_error("checkpoint", "orphan_segments_removed", "counter overflow")
        })?;
        let mut snapshot = self.0.snapshot.lock();
        snapshot.checkpoints.orphan_segments_removed = checked_sum(
            snapshot.checkpoints.orphan_segments_removed,
            removed,
            "checkpoint",
            "orphan_segments_removed",
        )?;
        Ok(())
    }

    pub(crate) fn record_reaper_join(&self) -> Result<()> {
        let mut snapshot = self.0.snapshot.lock();
        let result = checked_inc(&mut snapshot.job.reaper_joins, "job", "reaper_joins");
        if result.is_err() {
            snapshot.job.metrics_overflowed = true;
        }
        result
    }

    pub(crate) fn record_reaper_join_error(&self) -> Result<()> {
        let mut snapshot = self.0.snapshot.lock();
        let result = checked_inc(
            &mut snapshot.job.reaper_join_errors,
            "job",
            "reaper_join_errors",
        );
        if result.is_err() {
            snapshot.job.metrics_overflowed = true;
        }
        result
    }

    pub(crate) fn record_abandoned_runner_drop(&self) -> Result<()> {
        let mut snapshot = self.0.snapshot.lock();
        let result = checked_inc(
            &mut snapshot.job.abandoned_runner_drops,
            "job",
            "abandoned_runner_drops",
        );
        if result.is_err() {
            snapshot.job.metrics_overflowed = true;
        }
        result
    }

    /// Accounts terminal failures without ever feeding a metrics overflow
    /// back into the same error-accounting path.
    pub(crate) fn account_terminal_errors_once(
        &self,
        failures: &[Arc<RuntimeFailure>],
    ) -> Option<Arc<RuntimeFailure>> {
        let mut ordered = failures.iter().collect::<Vec<_>>();
        ordered.sort_by(|left, right| left.origin.cmp(&right.origin));
        let mut snapshot = self.0.snapshot.lock();
        let mut first_overflow: Option<(String, &'static str)> = None;
        for failure in ordered {
            attempt_terminal_increment(
                &mut snapshot.job.task_errors,
                "job",
                "task_errors",
                &mut first_overflow,
            );
            match &failure.origin {
                FailureOrigin::OperatorEntry { node_id } => {
                    attempt_node_error(&mut snapshot.nodes, node_id, &mut first_overflow);
                }
                FailureOrigin::SourceOpen { binding_id }
                | FailureOrigin::SourceClose { binding_id } => {
                    attempt_source_error(&mut snapshot.sources, binding_id, &mut first_overflow);
                }
                FailureOrigin::SinkOpen { output_id, sink_id }
                | FailureOrigin::SinkClose { output_id, sink_id }
                | FailureOrigin::SinkWrite { output_id, sink_id }
                | FailureOrigin::SinkCheckpoint { output_id, sink_id } => attempt_sink_error(
                    &mut snapshot.sinks,
                    &sink_metric_id(output_id, sink_id),
                    &mut first_overflow,
                ),
                FailureOrigin::SinkIngress { output_id, .. } => {
                    for (sink_id, metrics) in snapshot
                        .sinks
                        .iter_mut()
                        .filter(|(id, _)| id.starts_with(&sink_metric_prefix(output_id)))
                    {
                        attempt_terminal_increment(
                            &mut metrics.errors,
                            sink_id,
                            "errors",
                            &mut first_overflow,
                        );
                    }
                }
                FailureOrigin::Task { task_name, .. } => {
                    if let Some(source_id) = task_name
                        .strip_prefix("source:")
                        .and_then(|name| name.rsplit_once(':').map(|(id, _)| id))
                    {
                        attempt_source_error(&mut snapshot.sources, source_id, &mut first_overflow);
                    } else if let Some(node_id) = task_name.strip_prefix("operator:") {
                        attempt_node_error(&mut snapshot.nodes, node_id, &mut first_overflow);
                    }
                }
                FailureOrigin::Preflight
                | FailureOrigin::RunnerLifecycle
                | FailureOrigin::Metrics { .. } => {}
            }
        }
        let (component_id, counter) = first_overflow?;
        snapshot.job.metrics_overflowed = true;
        Some(Arc::new(RuntimeFailure {
            origin: FailureOrigin::Metrics {
                component_id: component_id.clone(),
                counter,
            },
            error: metrics_error(&component_id, counter, "counter overflow"),
        }))
    }

    #[cfg(test)]
    pub(crate) fn preset_job_task_errors_for_test(&self, value: u64) {
        self.0.snapshot.lock().job.task_errors = value;
    }

    fn with_edge(
        &self,
        edge_id: &str,
        record: impl FnOnce(&mut EdgeRuntimeMetrics) -> Result<()>,
    ) -> Result<()> {
        let mut snapshot = self.0.snapshot.lock();
        match snapshot.edges.get_mut(edge_id) {
            Some(metrics) => record(metrics),
            None if !self.0.strict_ids => Ok(()),
            None => Err(unregistered_id("edge", edge_id)),
        }
    }

    fn with_source(
        &self,
        source_id: &str,
        record: impl FnOnce(&mut SourceMetrics) -> Result<()>,
    ) -> Result<()> {
        let mut snapshot = self.0.snapshot.lock();
        match snapshot.sources.get_mut(source_id) {
            Some(metrics) => record(metrics),
            None if !self.0.strict_ids => Ok(()),
            None => Err(unregistered_id("source", source_id)),
        }
    }

    fn with_node(
        &self,
        node_id: &str,
        record: impl FnOnce(&mut OperatorMetrics) -> Result<()>,
    ) -> Result<()> {
        let mut snapshot = self.0.snapshot.lock();
        match snapshot.nodes.get_mut(node_id) {
            Some(metrics) => record(metrics),
            None if !self.0.strict_ids => Ok(()),
            None => Err(unregistered_id("node", node_id)),
        }
    }

    fn with_sink(
        &self,
        sink_id: &str,
        record: impl FnOnce(&mut SinkMetrics) -> Result<()>,
    ) -> Result<()> {
        let mut snapshot = self.0.snapshot.lock();
        match snapshot.sinks.get_mut(sink_id) {
            Some(metrics) => record(metrics),
            None if !self.0.strict_ids => Ok(()),
            None => Err(unregistered_id("sink", sink_id)),
        }
    }
}

fn next_edge_input(
    metrics: &EdgeRuntimeMetrics,
    traffic: EdgeTraffic,
    edge_id: &str,
) -> Result<(u64, u64, u64)> {
    Ok((
        checked_sum(
            metrics.input_batches,
            traffic.batches,
            edge_id,
            "input_batches",
        )?,
        checked_sum(metrics.input_rows, traffic.rows, edge_id, "input_rows")?,
        checked_sum(metrics.input_bytes, traffic.bytes, edge_id, "input_bytes")?,
    ))
}

fn next_edge_output(
    metrics: &EdgeRuntimeMetrics,
    traffic: EdgeTraffic,
    edge_id: &str,
) -> Result<(u64, u64, u64)> {
    Ok((
        checked_sum(
            metrics.output_batches,
            traffic.batches,
            edge_id,
            "output_batches",
        )?,
        checked_sum(metrics.output_rows, traffic.rows, edge_id, "output_rows")?,
        checked_sum(metrics.output_bytes, traffic.bytes, edge_id, "output_bytes")?,
    ))
}

fn next_edge_enqueue_channel(
    channel: &ChannelMetrics,
    traffic: EdgeTraffic,
    blocked_elapsed: Option<Duration>,
    edge_id: &str,
) -> Result<ChannelMetrics> {
    let mut next = channel.clone();
    next.queue_depth = checked_add_usize(
        channel.queue_depth,
        traffic.messages,
        edge_id,
        "queue_depth",
    )?;
    next.charged_rows = checked_add_usize(
        channel.charged_rows,
        traffic.cost_rows,
        edge_id,
        "charged_rows",
    )?;
    next.charged_bytes = checked_add_usize(
        channel.charged_bytes,
        traffic.cost_bytes,
        edge_id,
        "charged_bytes",
    )?;
    next.blocked_duration = next_blocked_duration(channel, blocked_elapsed, edge_id)?;
    next.high_water_depth = channel.high_water_depth.max(next.queue_depth);
    next.high_water_rows = channel.high_water_rows.max(next.charged_rows);
    next.high_water_bytes = channel.high_water_bytes.max(next.charged_bytes);
    Ok(next)
}

fn next_blocked_duration(
    channel: &ChannelMetrics,
    blocked_elapsed: Option<Duration>,
    edge_id: &str,
) -> Result<Duration> {
    blocked_elapsed.map_or(Ok(channel.blocked_duration), |elapsed| {
        channel
            .blocked_duration
            .checked_add(elapsed)
            .ok_or_else(|| metrics_error(edge_id, "blocked_duration", "counter overflow"))
    })
}

fn next_edge_dequeue_channel(
    channel: &ChannelMetrics,
    traffic: EdgeTraffic,
    edge_id: &str,
) -> Result<ChannelMetrics> {
    let mut next = channel.clone();
    next.queue_depth = checked_sub_usize(
        channel.queue_depth,
        traffic.messages,
        edge_id,
        "queue_depth",
    )?;
    next.charged_rows = checked_sub_usize(
        channel.charged_rows,
        traffic.cost_rows,
        edge_id,
        "charged_rows",
    )?;
    next.charged_bytes = checked_sub_usize(
        channel.charged_bytes,
        traffic.cost_bytes,
        edge_id,
        "charged_bytes",
    )?;
    Ok(next)
}

fn checked_add_usize(
    value: usize,
    add: usize,
    edge_id: &str,
    counter: &'static str,
) -> Result<usize> {
    value
        .checked_add(add)
        .ok_or_else(|| metrics_error(edge_id, counter, "counter overflow"))
}

fn checked_sub_usize(
    value: usize,
    sub: usize,
    edge_id: &str,
    counter: &'static str,
) -> Result<usize> {
    value
        .checked_sub(sub)
        .ok_or_else(|| metrics_error(edge_id, counter, "counter underflow"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct EdgeTraffic {
    batches: u64,
    rows: u64,
    bytes: u64,
    messages: usize,
    cost_rows: usize,
    cost_bytes: usize,
}

impl EdgeTraffic {
    pub(super) fn of_message(message: &StreamMessage, cost: EnvelopeCost) -> Result<Self> {
        let is_data = message.as_data().is_some();
        Ok(Self {
            batches: u64::from(is_data),
            rows: if is_data {
                u64::try_from(cost.rows())
                    .map_err(|_| metrics_error("edge", "rows", "counter overflow"))?
            } else {
                0
            },
            bytes: if is_data {
                u64::try_from(cost.bytes())
                    .map_err(|_| metrics_error("edge", "bytes", "counter overflow"))?
            } else {
                0
            },
            messages: cost.messages(),
            cost_rows: cost.rows(),
            cost_bytes: cost.bytes(),
        })
    }
}

pub(crate) fn sink_metric_id(output_id: &str, sink_id: &str) -> String {
    format!(
        "sink/{}/{}",
        hex::encode(output_id.as_bytes()),
        hex::encode(sink_id.as_bytes())
    )
}

fn sink_metric_prefix(output_id: &str) -> String {
    format!("sink/{}/", hex::encode(output_id.as_bytes()))
}

fn checked_traffic(
    batches: &mut u64,
    rows: &mut u64,
    bytes: &mut u64,
    cost: EnvelopeCost,
    component_id: &str,
    prefix: &str,
) -> Result<()> {
    let (add_batches, add_rows, add_bytes) = traffic_values(cost)?;
    let next_batches = checked_sum(
        *batches,
        add_batches,
        component_id,
        owned_counter(prefix, "batches"),
    )?;
    let next_rows = checked_sum(*rows, add_rows, component_id, owned_counter(prefix, "rows"))?;
    let next_bytes = checked_sum(
        *bytes,
        add_bytes,
        component_id,
        owned_counter(prefix, "bytes"),
    )?;
    *batches = next_batches;
    *rows = next_rows;
    *bytes = next_bytes;
    Ok(())
}

fn owned_counter(prefix: &str, suffix: &str) -> &'static str {
    match (prefix, suffix) {
        ("input", "batches") => "input_batches",
        ("input", "rows") => "input_rows",
        ("input", "bytes") => "input_bytes",
        ("fully_fanned_out", "batches") => "fully_fanned_out_batches",
        ("fully_fanned_out", "rows") => "fully_fanned_out_rows",
        ("fully_fanned_out", "bytes") => "fully_fanned_out_bytes",
        _ => "counter",
    }
}

fn traffic_values(cost: EnvelopeCost) -> Result<(u64, u64, u64)> {
    Ok((
        1,
        u64::try_from(cost.rows())
            .map_err(|_| metrics_error("traffic", "rows", "counter overflow"))?,
        u64::try_from(cost.bytes())
            .map_err(|_| metrics_error("traffic", "bytes", "counter overflow"))?,
    ))
}

fn checked_inc(value: &mut u64, component_id: &str, counter: &'static str) -> Result<()> {
    *value = value
        .checked_add(1)
        .ok_or_else(|| metrics_error(component_id, counter, "counter overflow"))?;
    Ok(())
}

fn checked_sum(value: u64, add: u64, component_id: &str, counter: &'static str) -> Result<u64> {
    value
        .checked_add(add)
        .ok_or_else(|| metrics_error(component_id, counter, "counter overflow"))
}

fn checked_duration(value: Duration, add: Duration, counter: &'static str) -> Result<Duration> {
    value
        .checked_add(add)
        .ok_or_else(|| metrics_error("checkpoint", counter, "counter overflow"))
}

fn attempt_source_error(
    sources: &mut BTreeMap<String, SourceMetrics>,
    source_id: &str,
    overflow: &mut Option<(String, &'static str)>,
) {
    if let Some(metrics) = sources.get_mut(source_id) {
        attempt_terminal_increment(&mut metrics.errors, source_id, "errors", overflow);
    }
}

fn attempt_node_error(
    nodes: &mut BTreeMap<String, OperatorMetrics>,
    node_id: &str,
    overflow: &mut Option<(String, &'static str)>,
) {
    if let Some(metrics) = nodes.get_mut(node_id) {
        attempt_terminal_increment(&mut metrics.errors, node_id, "errors", overflow);
    }
}

fn attempt_sink_error(
    sinks: &mut BTreeMap<String, SinkMetrics>,
    sink_id: &str,
    overflow: &mut Option<(String, &'static str)>,
) {
    if let Some(metrics) = sinks.get_mut(sink_id) {
        attempt_terminal_increment(&mut metrics.errors, sink_id, "errors", overflow);
    }
}

fn attempt_terminal_increment(
    value: &mut u64,
    component_id: &str,
    counter: &'static str,
    overflow: &mut Option<(String, &'static str)>,
) {
    match value.checked_add(1) {
        Some(next) => *value = next,
        None if overflow.is_none() => *overflow = Some((component_id.to_owned(), counter)),
        None => {}
    }
}

fn metrics_error(
    component_id: &str,
    counter: &'static str,
    message: &'static str,
) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: format!("runtime.metrics.{component_id}.{counter}"),
        message: message.into(),
    }
}

fn unregistered_id(kind: &str, id: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("unregistered streaming metrics {kind} ID {id:?}"),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
        time::Duration,
    };

    use super::{MetricsClock, MetricsRecorder};
    use crate::{
        CalcFlowError, EdgeBudget,
        runtime::streaming::{
            EnvelopeCost,
            runner::{FailureOrigin, RuntimeFailure},
        },
    };

    #[derive(Default)]
    struct TestClock(AtomicU64);

    impl MetricsClock for TestClock {
        fn now(&self) -> Duration {
            Duration::from_nanos(self.0.load(Ordering::SeqCst))
        }
    }

    #[test]
    fn stable_ids_and_injected_clock_produce_deterministic_snapshot() {
        let clock = Arc::new(TestClock::default());
        let recorder = MetricsRecorder::with_clock(
            [
                ("edge-b".into(), EdgeBudget::new(3, 64).unwrap()),
                ("edge-a".into(), EdgeBudget::new(2, 64).unwrap()),
            ],
            ["source".into()],
            ["node".into()],
            ["sink".into()],
            clock.clone(),
        );
        recorder
            .record_operator_input("node", EnvelopeCost::new(1, 2, 16))
            .unwrap();
        let timer = recorder.timer();
        clock.0.store(17, Ordering::SeqCst);
        recorder.record_operator_processing("node", &timer).unwrap();

        let snapshot = recorder.snapshot();
        assert_eq!(
            snapshot.edges.keys().collect::<Vec<_>>(),
            ["edge-a", "edge-b"]
        );
        assert_eq!(snapshot.nodes["node"].input_batches, 1);
        assert_eq!(snapshot.nodes["node"].input_rows, 2);
        assert_eq!(
            snapshot.nodes["node"].processing_duration,
            Duration::from_nanos(17)
        );
    }

    #[test]
    fn edge_message_slot_limit_is_copied_from_the_validated_budget() {
        let recorder = MetricsRecorder::new(
            [("edge".into(), EdgeBudget::new(3, 64).unwrap())],
            [],
            [],
            [],
        );

        assert_eq!(recorder.snapshot().edges["edge"].message_slot_limit, 3);
    }

    #[test]
    fn terminal_error_overflow_is_one_non_recursive_secondary() {
        let recorder = MetricsRecorder::new([], [], ["node".into()], []);
        recorder.preset_job_task_errors_for_test(u64::MAX);
        let failure = Arc::new(RuntimeFailure {
            origin: FailureOrigin::OperatorEntry {
                node_id: "node".into(),
            },
            error: CalcFlowError::Operator {
                node_id: "node".into(),
                message: "primary".into(),
            },
        });

        let overflow = recorder
            .account_terminal_errors_once(&[failure])
            .expect("full task counter yields one bounded secondary");

        assert!(matches!(overflow.origin, FailureOrigin::Metrics { .. }));
        let snapshot = recorder.snapshot();
        assert_eq!(snapshot.job.task_errors, u64::MAX);
        assert!(snapshot.job.metrics_overflowed);
        assert_eq!(snapshot.nodes["node"].errors, 1);
    }

    #[test]
    fn ordinary_counter_overflow_is_exact_and_never_wraps() {
        let recorder = MetricsRecorder::new([], ["source".into()], [], []);
        {
            let mut snapshot = recorder.0.snapshot.lock();
            snapshot.sources.get_mut("source").unwrap().poll_count = u64::MAX;
        }

        let error = recorder.record_source_poll("source").unwrap_err();

        assert!(matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, ref message }
                if field == "runtime.metrics.source.poll_count" && message == "counter overflow"
        ));
        assert_eq!(recorder.snapshot().sources["source"].poll_count, u64::MAX);
    }

    #[test]
    fn edge_drop_underflow_sets_bounded_invariant_flag_without_partial_decrement() {
        let recorder = MetricsRecorder::new(
            [("edge".into(), EdgeBudget::new(1, 8).unwrap())],
            [],
            [],
            [],
        );
        {
            let mut snapshot = recorder.0.snapshot.lock();
            let edge = snapshot.edges.get_mut("edge").unwrap();
            edge.channel.queue_depth = 1;
            edge.channel.charged_rows = 0;
            edge.channel.charged_bytes = 1;
        }

        recorder.record_edge_drop("edge", EnvelopeCost::new(1, 1, 1));

        let edge = &recorder.snapshot().edges["edge"];
        assert_eq!(edge.channel.queue_depth, 1);
        assert_eq!(edge.channel.charged_rows, 0);
        assert_eq!(edge.channel.charged_bytes, 1);
        assert!(edge.drop_invariant_violated);
    }
}
