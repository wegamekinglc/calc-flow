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

impl StreamingJob {
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
