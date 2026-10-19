use super::{Result, StreamingJob};

fn operator_started(operator: &crate::OperatorStatus) -> bool {
    operator.input_batches != 0
        || operator.fanned_out_batches != 0
        || operator.errors != 0
        || operator.ended
}

fn asof_admission_started(asof: &crate::StreamAsofJoinStatus) -> bool {
    asof.left.accepted_rows != 0
        || asof.right.accepted_rows != 0
        || asof.left.watermark_micros.is_some()
        || asof.right.watermark_micros.is_some()
}

fn asof_ingress_stopped(asof: &crate::StreamAsofJoinStatus) -> bool {
    asof.left.ended || asof.right.ended || asof.left.idle || asof.right.idle
}

fn join_started(join: &crate::StreamJoinStatus) -> bool {
    join.emitted_match_rows != 0
        || join.state_limit_failures != 0
        || join.match_limit_failures != 0
        || [&join.left, &join.right].into_iter().any(join_side_started)
}

fn join_side_started(side: &crate::StreamJoinSideStatus) -> bool {
    join_side_rows_changed(side) || side.watermark_micros.is_some() || side.idle || side.ended
}

fn join_side_rows_changed(side: &crate::StreamJoinSideStatus) -> bool {
    side.retained_rows != 0
        || side.retained_bytes != 0
        || side.evicted_rows != 0
        || side.late_rows != 0
        || side.null_event_time_rows != 0
        || side.null_key_rows != 0
}

fn require_join_cpu_completion(
    state: crate::JobState,
    cause: Option<crate::TerminalCause>,
    errors: u64,
) -> Result<()> {
    if state != crate::JobState::Completed
        || cause != Some(crate::TerminalCause::NaturalEnd)
        || errors != 0
    {
        return Err(crate::benchmark_internals::invalid(
            "join CPU capture requires successful natural completion",
        ));
    }
    Ok(())
}

impl StreamingJob {
    /// Starts join phase diagnostics before any operator processes input.
    /// The benchmark must keep its sources gated until this call returns.
    ///
    /// # Errors
    /// Rejects jobs without joins, started operators, and reused captures.
    #[doc(hidden)]
    pub fn benchmark_start_join_cpu_trace(&self) -> Result<()> {
        let status = self.status();
        let joins = self.stream_join_status();
        let asofs = self.stream_asof_join_status();
        if status.state != crate::JobState::Running
            || status.task_errors != 0
            || (joins.is_empty() && asofs.is_empty())
            || status.operators.values().any(operator_started)
            || asofs
                .values()
                .any(|value| value != &crate::StreamAsofJoinStatus::default())
            || joins.values().any(join_started)
        {
            return Err(crate::benchmark_internals::invalid(
                "join CPU capture requires an empty join job",
            ));
        }
        self.benchmark_context
            .join_cpu_trace()
            .start(self.id(), self.benchmark_context.fingerprint())
    }

    /// Marks the existing benchmark processing timer boundaries.
    ///
    /// # Errors
    /// Rejects inactive captures and missing, repeated, or out-of-order labels.
    #[doc(hidden)]
    pub fn benchmark_mark_join_cpu_trace(&self, label: &str) -> Result<()> {
        self.benchmark_context.join_cpu_trace().mark(label)
    }

    /// Seals phase diagnostics after all instrumented operations have settled.
    ///
    /// # Errors
    /// Rejects incomplete or unsuccessful jobs and invalid or unsettled captures.
    #[doc(hidden)]
    pub fn benchmark_take_join_cpu_trace(&self) -> Result<serde_json::Value> {
        let status = self.status();
        require_join_cpu_completion(status.state, status.terminal_cause, status.task_errors)?;
        self.benchmark_context.join_cpu_trace().take()
    }

    /// Enables the fixed ASOF benchmark recorder before source feeding starts.
    ///
    /// # Errors
    /// Rejects nonempty jobs, graphs without exactly one ASOF, and reused captures.
    #[doc(hidden)]
    pub fn benchmark_start_dispatch_trace(&self, allow_barriers: bool) -> Result<()> {
        let statuses = self.stream_asof_join_status();
        let status = self.status();
        if statuses.len() != 1
            || status.operators.is_empty()
            || status.operators.values().any(operator_started)
        {
            return Err(crate::benchmark_internals::invalid(
                "dispatch capture requires one empty ASOF job",
            ));
        }
        let (node, asof) = statuses.first_key_value().expect("one ASOF");
        if asof_admission_started(asof) || asof_ingress_stopped(asof) || asof.state_rows != 0 {
            return Err(crate::benchmark_internals::invalid(
                "ASOF processing already started",
            ));
        }
        let context = &self.benchmark_context;
        context
            .dispatch_trace()
            .start(self.id(), context.fingerprint(), node, allow_barriers)?;
        context
            .dispatch_trace()
            .register_nodes(&status.operators.into_keys().collect::<Vec<_>>())
    }

    /// Seals benchmark events and serializes them after the processing timer.
    ///
    /// # Errors
    /// Rejects missing, reused, overflowed, or invalid captures.
    #[doc(hidden)]
    pub fn benchmark_take_dispatch_trace(&self) -> Result<serde_json::Value> {
        self.benchmark_context.dispatch_trace().take()
    }

    /// Returns the last completely dispatched watermark for one benchmark node.
    #[doc(hidden)]
    #[must_use]
    pub fn benchmark_dispatch_frontier(&self, node_id: &str) -> Option<i64> {
        self.benchmark_context.dispatch_trace().frontier(node_id)
    }
}

#[cfg(test)]
mod join_cpu_tests {
    use crate::{JobState, TerminalCause};

    #[test]
    fn test_join_cpu_trace_completion_rejects_partial_failed_and_cancelled_jobs() {
        for state in [
            JobState::Running,
            JobState::Draining,
            JobState::Cancelled,
            JobState::Failed,
            JobState::RecoveryRequired,
        ] {
            assert!(super::require_join_cpu_completion(state, None, 0).is_err());
        }
        assert!(
            super::require_join_cpu_completion(
                JobState::Completed,
                Some(TerminalCause::GracefulShutdown),
                0
            )
            .is_err()
        );
        assert!(
            super::require_join_cpu_completion(
                JobState::Completed,
                Some(TerminalCause::NaturalEnd),
                1
            )
            .is_err()
        );
        assert!(
            super::require_join_cpu_completion(
                JobState::Completed,
                Some(TerminalCause::NaturalEnd),
                0
            )
            .is_ok()
        );
    }
}
