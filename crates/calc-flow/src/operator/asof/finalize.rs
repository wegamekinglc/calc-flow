use super::{
    StreamAsofJoinOperator, StreamAsofJoinStatus, checked, reason,
    state::{LeftOrder, State},
};
use crate::{
    Batch, BatchMetadata, CalcFlowError, EventTime, JsonMap, Result, StateSegment, StreamCollector,
    StreamOperatorContext, StreamingFailureReason,
};
use datafusion::execution::memory_pool::MemoryReservation;

struct PreparedOutput {
    batch: Batch,
    matched: u64,
    workspace: MemoryReservation,
}

impl StreamAsofJoinOperator {
    pub(super) async fn finalize(
        &mut self,
        frontier: Option<i64>,
        ended: bool,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        loop {
            context.check_cancelled()?;
            let state_workspace = self.state_workspace(&self.state)?;
            let headroom = self.checkpoint_workspace()?;
            let mut keys = self.finalizable_keys(frontier, ended, context);
            if keys.is_empty() {
                break;
            }
            let prepared_output = self.prepare_output(&mut keys, context).await?;
            self.commit_output(&keys, prepared_output, headroom, context, output)
                .await?;
            drop(state_workspace);
        }
        self.finish_progress(frontier, ended, context).await
    }

    fn checkpoint_workspace(&self) -> Result<MemoryReservation> {
        self.reserve_workspace(super::checkpoint::encoded_length(&self.state, &self.name)?)
    }

    fn finalizable_keys(
        &self,
        frontier: Option<i64>,
        ended: bool,
        context: &StreamOperatorContext<'_>,
    ) -> Vec<LeftOrder> {
        self.state
            .left
            .keys()
            .take_while(|(time, _, _)| ended || frontier.is_some_and(|bound| *time < bound))
            .take(context.output_budget().max_rows.min(128))
            .cloned()
            .collect()
    }

    async fn commit_output(
        &mut self,
        keys: &[LeftOrder],
        output: PreparedOutput,
        headroom: MemoryReservation,
        context: &StreamOperatorContext<'_>,
        collector: &mut dyn StreamCollector,
    ) -> Result<()> {
        let next_sequence = checked(&self.name, self.next_output_sequence, keys.len() as u64)?;
        let (next, mut status) = self.output_state(keys, output.matched)?;
        drop(headroom);
        let prepared = self.prepare_checkpoint(&next, context).await?;
        self.checked_inventory(&next, prepared.segment.as_ref(), &mut status)?;
        emit_output(output.batch, context, collector).await?;
        self.install(next, status, prepared);
        self.next_output_sequence = next_sequence;
        drop(output.workspace);
        Ok(())
    }

    fn output_state(
        &self,
        keys: &[LeftOrder],
        matched: u64,
    ) -> Result<(State, StreamAsofJoinStatus)> {
        let mut next = self.state.clone();
        for key in keys {
            next.left.remove(key);
        }
        let mut status = self.status.clone();
        status.emitted_left_rows =
            checked(&self.name, status.emitted_left_rows, keys.len() as u64)?;
        status.matched_rows = checked(&self.name, status.matched_rows, matched)?;
        status.unmatched_rows = checked(
            &self.name,
            status.unmatched_rows,
            keys.len() as u64 - matched,
        )?;
        let evicted = next.evict(&status, self.spec.tolerance_micros());
        status.evicted_right_rows = checked(&self.name, status.evicted_right_rows, evicted)?;
        Ok((next, status))
    }

    async fn finish_progress(
        &mut self,
        frontier: Option<i64>,
        ended: bool,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let workspace = self.state_workspace(&self.state)?;
        let mut next = self.state.clone();
        let mut status = self.status.clone();
        let evicted = next.evict(&status, self.spec.tolerance_micros());
        status.evicted_right_rows = checked(&self.name, status.evicted_right_rows, evicted)?;
        let prepared = self.prepare_checkpoint(&next, context).await?;
        self.checked_inventory(&next, prepared.segment.as_ref(), &mut status)?;
        status.output_watermark_micros = frontier
            .and_then(|time| time.checked_sub(1))
            .map(EventTime::from_micros)
            .or(status.output_watermark_micros);
        self.install(next, status, prepared);
        self.terminal = ended;
        drop(workspace);
        Ok(())
    }

    async fn prepare_output(
        &mut self,
        keys: &mut Vec<LeftOrder>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedOutput> {
        loop {
            context.check_cancelled()?;
            match self.output_attempt(keys, context).await {
                Ok(output) => return Ok(output),
                Err(error) if keys.len() > 1 && retryable(&error) => keys.truncate(keys.len() / 2),
                Err(error) => return Err(error),
            }
        }
    }

    async fn output_attempt(
        &mut self,
        keys: &[LeftOrder],
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedOutput> {
        let rows = keys
            .iter()
            .map(|key| {
                (
                    &self.state.left[key],
                    self.state
                        .candidate(&key.1, key.0, self.spec.tolerance_micros()),
                )
            })
            .collect::<Vec<_>>();
        let matched = rows.iter().filter(|(_, right)| right.is_some()).count() as u64;
        let bytes = output_workspace(&rows, &self.name)?;
        let workspace = self.reserve_workspace(bytes)?;
        let result = self
            .runtime
            .materialize(
                &rows,
                &self.spec,
                &self.schemas,
                &self.schema_digests,
                &self.name,
            )
            .await?;
        let batch = self.output_batch(&result, context)?;
        Ok(PreparedOutput {
            batch,
            matched,
            workspace,
        })
    }
    fn output_batch(&self, result: &Batch, context: &StreamOperatorContext<'_>) -> Result<Batch> {
        let batch = Batch::table(
            result.table_payload()?.batches().to_vec(),
            BatchMetadata::new(&self.name, self.next_output_sequence, JsonMap::new())?,
        )?;
        if batch.estimated_bytes()? > context.output_budget().max_bytes {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofOutputLimitExceeded,
                "ASOF output row exceeds the output edge budget",
            ));
        }
        Ok(batch)
    }
}

fn retryable(error: &CalcFlowError) -> bool {
    matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded
                | StreamingFailureReason::AsofOutputLimitExceeded,
            ..
        }
    )
}

fn output_workspace(rows: &[(&StateSegment, Option<&StateSegment>)], name: &str) -> Result<u64> {
    rows.iter().try_fold(0, |total, (left, right)| {
        let row = checked(
            name,
            left.bytes().len() as u64,
            right.map_or(0, |value| value.bytes().len() as u64),
        )?;
        let estimate = checked(name, row, 4096)?.checked_mul(8).ok_or_else(|| {
            reason(
                name,
                StreamingFailureReason::AsofCounterOverflow,
                "ASOF output workspace arithmetic overflowed",
            )
        })?;
        checked(name, total, estimate)
    })
}

async fn emit_output(
    batch: Batch,
    context: &StreamOperatorContext<'_>,
    output: &mut dyn StreamCollector,
) -> Result<()> {
    context.check_cancelled()?;
    tokio::select! {
        result = output.emit("output", batch) => result,
        () = context.job().cancellation().cancelled() => context.check_cancelled(),
    }
}
