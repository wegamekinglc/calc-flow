//! Preserve canonical checkpoints while removing only a finalized left prefix.

use super::{PreparedOutput, emit_output};
use crate::operator::asof::{
    StreamAsofJoinOperator, checked,
    checkpoint::PreparedSegment,
    reason,
    state::{Inventory, LeftPrefix},
};
use crate::{Result, StreamCollector, StreamOperatorContext, StreamingFailureReason};
use datafusion::execution::memory_pool::MemoryReservation;

struct PreparedPrefix {
    segment: Option<PreparedSegment>,
    deferred_len: Option<u64>,
    _workspace: MemoryReservation,
}

impl PreparedPrefix {
    fn index_bytes(&self) -> u64 {
        self.segment
            .as_ref()
            .map(|segment| segment.capacity() as u64)
            .or(self.deferred_len)
            .map_or(0, |capacity| capacity + 64)
    }
}

impl StreamAsofJoinOperator {
    #[tracing::instrument(
        name = "asof.prefix_commit",
        level = "debug",
        skip_all,
        fields(operator = %self.name, rows = output.prefix.count)
    )]
    pub(super) async fn commit_prefix_output(
        &mut self,
        output: PreparedOutput,
        headroom: MemoryReservation,
        context: &StreamOperatorContext<'_>,
        collector: &mut dyn StreamCollector,
    ) -> Result<()> {
        let count = output.prefix.count;
        let next_sequence = checked(&self.name, self.next_output_sequence, count as u64)?;
        let mut status = self.output_status(count, output.matched)?;
        drop(headroom);
        let prepared = self
            .prepare_prefix_checkpoint(&output.prefix, context)
            .await?;
        let current = Inventory {
            identities: self.status.state_rows,
            right_payloads: self.status.retained_right_rows,
            identity_only: self.status.identity_only_rows,
            bytes: self.status.state_bytes,
        };
        let previous_index_bytes = self
            .prepared
            .as_ref()
            .map(|segment| segment.capacity() as u64)
            .or(self.deferred_index_len)
            .map_or(0, |capacity| capacity + 64);
        let inventory = self.state.inventory_after_left_prefix(
            &output.prefix,
            current,
            previous_index_bytes,
            prepared.index_bytes(),
            &self.name,
        )?;
        if inventory.identities > self.spec.limits().max_state_rows()
            || inventory.bytes > self.spec.limits().max_state_bytes()
        {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofStateLimitExceeded,
                "stream_asof_join state limits exceeded",
            ));
        }
        status.pending_left_rows = (self.state.left.len() - count) as u64;
        status.retained_right_rows = inventory.right_payloads;
        status.identity_only_rows = inventory.identity_only;
        status.state_rows = inventory.identities;
        status.state_bytes = inventory.bytes;
        emit_output(output.batch, context, collector).await?;
        // Nothing after sink acceptance can fail or yield before installation.
        self.state.commit_matched_left_prefix(&output.prefix);
        // Right payloads remain until finish_progress sweeps them once.
        self.swept = None;
        self.status = status;
        self.prepared = prepared.segment;
        self.deferred_index_len = prepared.deferred_len;
        self.next_output_sequence = next_sequence;
        debug_assert!(
            {
                let inventory = self
                    .state
                    .inventory(self.prepared.as_ref(), &self.name)
                    .expect("committed prefix inventory");
                inventory.bytes + self.deferred_index_len.map_or(0, |length| length + 64)
                    == self.status.state_bytes
            },
            "prefix inventory must match committed gauge"
        );
        drop(output.workspace);
        Ok(())
    }

    async fn prepare_prefix_checkpoint(
        &self,
        prefix: &LeftPrefix,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedPrefix> {
        let remaining = self.state.left.len() - prefix.count;
        if remaining == 0 && self.state.right.is_empty() {
            return Ok(PreparedPrefix {
                segment: None,
                deferred_len: None,
                _workspace: self.reserve_workspace(0)?,
            });
        }
        let removed = prefix.index_bytes;
        if let Some(length) = self.deferred_index_len {
            let remaining_len = length - removed;
            let workspace = self.reserve_workspace(remaining_len)?;
            context.check_cancelled()?;
            tokio::task::yield_now().await;
            context.check_cancelled()?;
            return Ok(PreparedPrefix {
                segment: None,
                deferred_len: Some(remaining_len),
                _workspace: workspace,
            });
        }
        let current = self
            .prepared
            .as_ref()
            .expect("committed ASOF state has a prepared segment");
        let removed = usize::try_from(removed).expect("encoded state fits address domain");
        // Same reservation the eager copy took, so workspace failures stay put;
        // capture later materializes exactly these canonical bytes.
        let workspace = self.reserve_workspace((current.len() - removed) as u64)?;
        context.check_cancelled()?;
        tokio::task::yield_now().await;
        context.check_cancelled()?;
        Ok(PreparedPrefix {
            segment: Some(current.drain_left(removed, remaining as u64)),
            deferred_len: None,
            _workspace: workspace,
        })
    }
}
