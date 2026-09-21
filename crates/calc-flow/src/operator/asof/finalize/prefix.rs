//! Preserve canonical checkpoints while removing only a finalized left prefix.

use super::{PreparedOutput, emit_output};
use crate::operator::asof::{
    StreamAsofJoinOperator, SweepStamp, checked,
    checkpoint::{PreparedCheckpoint, left_prefix_length},
    state::{InventoryDelta, LeftOrder},
};
use crate::{Result, StateSegment, StreamCollector, StreamOperatorContext};
use datafusion::execution::memory_pool::MemoryReservation;

impl StreamAsofJoinOperator {
    pub(super) async fn commit_prefix_output(
        &mut self,
        keys: &[LeftOrder],
        output: PreparedOutput,
        headroom: MemoryReservation,
        context: &StreamOperatorContext<'_>,
        collector: &mut dyn StreamCollector,
    ) -> Result<()> {
        let next_sequence = checked(&self.name, self.next_output_sequence, keys.len() as u64)?;
        let mut status = self.output_status(keys.len(), output.matched)?;
        let mut delta = InventoryDelta::default();
        for key in keys {
            delta.remove_left(&key.1, &key.2, &self.state.left[key]);
        }
        drop(headroom);
        let prepared = self.prepare_prefix_checkpoint(keys.len(), context).await?;
        let inventory = self.candidate_inventory(delta, prepared.segment.as_ref(), &status)?;
        status.pending_left_rows = (self.state.left.len() - keys.len()) as u64;
        status.retained_right_rows = inventory.right_payloads;
        status.identity_only_rows = inventory.identity_only;
        status.state_rows = inventory.identities;
        status.state_bytes = inventory.bytes;
        emit_output(output.batch, context, collector).await?;
        // Nothing after sink acceptance can fail or yield before the committed prefix is installed.
        for key in keys {
            self.state.left.remove(key);
        }
        self.swept = Some(SweepStamp::current(&status));
        self.status = status;
        self.prepared = prepared.segment;
        self.next_output_sequence = next_sequence;
        debug_assert!(
            self.state
                .inventory(self.prepared.as_ref(), &self.name)
                .is_ok_and(|walked| walked == inventory)
        );
        drop(output.workspace);
        Ok(())
    }

    async fn prepare_prefix_checkpoint(
        &self,
        count: usize,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedCheckpoint> {
        let remaining = self.state.left.len() - count;
        if remaining == 0 && self.state.right.is_empty() {
            return Ok(PreparedCheckpoint {
                segment: None,
                _workspace: self.reserve_workspace(0)?,
            });
        }
        let current = self
            .prepared
            .as_ref()
            .expect("committed ASOF state has a prepared segment");
        let removed = usize::try_from(left_prefix_length(&self.state, count, &self.name)?)
            .expect("encoded state fits address domain");
        let capacity = current.bytes().len() - removed;
        let workspace = self.reserve_workspace(capacity as u64)?;
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(&current.bytes()[..8]);
        bytes.extend_from_slice(&(remaining as u64).to_le_bytes());
        bytes.extend_from_slice(&current.bytes()[16..24]);
        let bytes = copy_checkpoint_suffix(bytes, &current.bytes()[24 + removed..], context).await?;
        Ok(PreparedCheckpoint {
            segment: Some(StateSegment::new(bytes)),
            _workspace: workspace,
        })
    }
}

async fn copy_checkpoint_suffix(
    mut bytes: Vec<u8>,
    suffix: &[u8],
    context: &StreamOperatorContext<'_>,
) -> Result<Vec<u8>> {
    for chunk in suffix.chunks(64 * 1024) {
        context.check_cancelled()?;
        bytes.extend_from_slice(chunk);
        tokio::task::yield_now().await;
    }
    context.check_cancelled()?;
    Ok(bytes)
}
