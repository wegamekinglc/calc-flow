//! Preserve canonical checkpoints while removing only a finalized left prefix.

use super::{PreparedOutput, emit_output};
use crate::operator::asof::{
    StreamAsofJoinOperator, checked,
    checkpoint::PreparedSegment,
    reason,
    state::{Inventory, LeftPrefix, PreparedLeftDrain, PreparedPayloadRemoval},
};
use crate::{Result, StreamCollector, StreamOperatorContext, StreamingFailureReason};
use datafusion::execution::memory_pool::MemoryReservation;
use std::future::Future;

struct PreparedPrefix {
    segment: Option<PreparedSegment>,
    deferred_len: Option<u64>,
    drain: PreparedLeftDrain,
    inventory: Inventory,
    pool: PreparedPayloadRemoval,
    _workspace: MemoryReservation,
    _drain_workspace: MemoryReservation,
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
        let inventory = prepared.inventory;
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
        self.state
            .commit_prepared_left_prefix(&output.prefix, prepared.drain, prepared.pool);
        // Right payloads remain until finish_progress sweeps them once.
        self.swept = None;
        self.status = status;
        self.prepared = prepared.segment;
        self.deferred_index_len = prepared.deferred_len;
        self.next_output_sequence = next_sequence;
        debug_assert!(
            {
                let inventory = self
                    .current_inventory(self.prepared.as_ref())
                    .expect("committed prefix inventory");
                inventory.bytes + self.deferred_index_len.map_or(0, |length| length + 256)
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
        let (drain, drain_workspace) = self.prepare_left_drain(prefix, context).await?;
        let (length, inventory, _) = self.state.project_capacity_prefix(
            self.capacity_snapshot(),
            prefix,
            &drain,
            &self.name,
        )?;
        let workspace = self.reserve_workspace(length)?;
        let pool = self
            .prepare_pool_compaction(&prefix.batches, context)
            .await?;
        context.check_cancelled()?;
        tokio::task::yield_now().await;
        context.check_cancelled()?;
        Ok(PreparedPrefix {
            segment: None,
            deferred_len: (length != 0).then_some(length),
            drain,
            inventory,
            pool,
            _workspace: workspace,
            _drain_workspace: drain_workspace,
        })
    }

    async fn prepare_left_drain(
        &self,
        prefix: &LeftPrefix,
        context: &StreamOperatorContext<'_>,
    ) -> Result<(PreparedLeftDrain, MemoryReservation)> {
        let bytes = self.state.left.drain_workspace_bytes(
            &prefix.batches,
            &self.state.batches,
            &self.name,
        )?;
        let workspace = self.reserve_workspace(bytes)?;
        let input = self
            .state
            .left
            .drain_input(&prefix.batches, &self.state.batches);
        if bytes == 0 {
            return Ok((input.prepare(), workspace));
        }
        context.check_cancelled()?;
        // The reservation and immutable metadata follow detached work, so
        // cancellation cannot release its workspace before its buffers.
        let worker = drain_worker(workspace, move || input.prepare());
        let prepared = await_drain(worker, context).await?;
        context.check_cancelled()?;
        Ok(prepared)
    }
}

async fn await_drain(
    worker: impl Future<Output = Result<(PreparedLeftDrain, MemoryReservation)>>,
    context: &StreamOperatorContext<'_>,
) -> Result<(PreparedLeftDrain, MemoryReservation)> {
    tokio::select! {
        result = worker => result,
        () = context.job().cancellation().cancelled() => {
            context.check_cancelled()?;
            unreachable!("cancelled ASOF prefix compaction")
        }
    }
}

async fn drain_worker<R: Send + 'static>(
    workspace: MemoryReservation,
    work: impl FnOnce() -> R + Send + 'static,
) -> Result<(R, MemoryReservation)> {
    tokio::task::spawn_blocking(move || (work(), workspace))
        .await
        .map_err(|error| crate::CalcFlowError::Internal {
            message: format!("ASOF prefix compaction task failed: {error}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
    use std::sync::Arc;

    #[tokio::test(flavor = "current_thread")]
    async fn dropped_compaction_worker_retains_input_reservation_until_exit() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(8_192));
        let workspace = MemoryConsumer::new("drain-test").register(&pool);
        workspace.try_grow(4_096).unwrap();
        let owner = Arc::new(vec![0_u8; 4_096]);
        let retained = owner.clone();
        let weak = Arc::downgrade(&owner);
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let gate = Arc::new(std::sync::Barrier::new(2));
        let worker_gate = gate.clone();
        let releaser = std::thread::spawn(move || {
            let _ = release_rx.recv_timeout(std::time::Duration::from_secs(1));
            gate.wait();
        });
        let mut future = Box::pin(drain_worker(workspace, move || {
            started_tx.send(()).unwrap();
            worker_gate.wait();
            drop(retained);
        }));
        assert!(futures::poll!(future.as_mut()).is_pending());
        started_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        drop(future);
        drop(owner);
        assert!(weak.upgrade().is_some());
        assert_eq!(pool.reserved(), 4_096);
        release_tx.send(()).unwrap();
        tokio::task::spawn_blocking(move || releaser.join().unwrap())
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while pool.reserved() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(weak.upgrade().is_none());
    }
}
