use super::{
    StreamAsofJoinOperator, StreamAsofJoinStatus, SweepStamp, checked,
    columnar::{Progress, Selection},
    reason,
};
use crate::operator::asof::kernel::KernelContext;
use crate::{
    Batch, BatchMetadata, CalcFlowError, EventTime, JsonMap, Result, StreamCollector,
    StreamingFailureReason,
};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::memory_pool::MemoryReservation;

#[cfg(test)]
thread_local! {
    static MATERIALIZATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn take_materializations() -> usize {
    MATERIALIZATIONS.with(|count| count.replace(0))
}

struct PreparedOutput {
    selection: Selection,
    batch: Batch,
    matched: u64,
    _workspace: MemoryReservation,
    _scratch: MemoryReservation,
}

impl StreamAsofJoinOperator {
    pub(super) async fn finalize(
        &mut self,
        frontier: Option<i64>,
        ended: bool,
        context: &dyn KernelContext,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        loop {
            context.check_cancelled()?;
            let Some(prepared) = self.prepare_output(frontier, ended, context)? else {
                break;
            };
            let rows = prepared.selection.len();
            let matched = prepared.matched;
            let status = self.output_status(rows, matched)?;
            let next_output_sequence = checked(&self.name, self.next_output_sequence, rows as u64)?;
            context.check_cancelled()?;
            tokio::select! {
                biased;
                // An acknowledged delivery belongs to the committed prefix.
                result = output.emit("output", prepared.batch) => result?,
                () = context.cancellation().cancelled() => return context.check_cancelled(),
            }
            self.state.commit(&prepared.selection)?;
            self.status = status;
            self.next_output_sequence = next_output_sequence;
            self.swept = None;
            self.refresh_state_gauges();
        }
        self.finish_progress(frontier, ended)
    }

    fn prepare_output(
        &self,
        frontier: Option<i64>,
        ended: bool,
        context: &dyn KernelContext,
    ) -> Result<Option<PreparedOutput>> {
        let mut count =
            self.state
                .finalizable_len(frontier, ended, context.output_budget().max_rows);
        while count > 0 {
            context.check_cancelled()?;
            match self.prepare_selected_output(count, context) {
                Ok(output) => return Ok(Some(output)),
                Err(error) if count > 1 && retryable(&error) => count /= 2,
                Err(error) => return Err(error),
            }
        }
        Ok(None)
    }

    fn prepare_selected_output(
        &self,
        count: usize,
        context: &dyn KernelContext,
    ) -> Result<PreparedOutput> {
        let columns = self.outputs[0]
            .schema()
            .expect("exact ASOF output")
            .fields()
            .len();
        let scratch = self.reserve_workspace(
            u64::try_from(self.state.selection_scratch_charge(count, columns))
                .expect("ASOF selection charge fits u64"),
        )?;
        let selection = self.state.select_prefix(count);
        let workspace = self.reserve_workspace(
            u64::try_from(
                self.state
                    .output_charge(&selection, self.output_projection.as_deref())?,
            )
            .expect("ASOF output charge fits u64"),
        )?;
        let record = self.materialize_selection(&selection)?;
        let batch =
            self.prepare_output_batch(&selection, record, context.output_budget().max_bytes)?;
        Ok(PreparedOutput {
            matched: selection.matched() as u64,
            selection,
            batch,
            _workspace: workspace,
            _scratch: scratch,
        })
    }

    fn materialize_selection(&self, selection: &Selection) -> Result<RecordBatch> {
        #[cfg(test)]
        MATERIALIZATIONS.with(|count| count.set(count.get() + 1));
        self.state.materialize(
            selection,
            self.outputs[0].schema().expect("exact ASOF output"),
            self.output_projection.as_deref(),
        )
    }

    fn prepare_output_batch(
        &self,
        selection: &Selection,
        record: RecordBatch,
        max_bytes: usize,
    ) -> Result<Batch> {
        let batch = Batch::table(
            vec![record],
            BatchMetadata::new(&self.name, self.next_output_sequence, JsonMap::new())?,
        )?;
        if batch.estimated_bytes()? > max_bytes {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofOutputLimitExceeded,
                "ASOF output row exceeds the output edge budget",
            ));
        }
        debug_assert_eq!(batch.num_rows(), selection.len());
        Ok(batch)
    }

    fn output_status(&self, rows: usize, matched: u64) -> Result<StreamAsofJoinStatus> {
        let mut status = self.status.clone();
        status.emitted_left_rows = checked(&self.name, status.emitted_left_rows, rows as u64)?;
        status.matched_rows = checked(&self.name, status.matched_rows, matched)?;
        status.unmatched_rows = checked(&self.name, status.unmatched_rows, rows as u64 - matched)?;
        Ok(status)
    }

    fn finish_progress(&mut self, frontier: Option<i64>, ended: bool) -> Result<()> {
        let stamp = SweepStamp::current(&self.status);
        if self.swept != Some(stamp) {
            let evicted = self.state.evict(Progress {
                left_watermark: self.status.left.watermark_micros.map(EventTime::as_micros),
                right_watermark: self.status.right.watermark_micros.map(EventTime::as_micros),
                left_ended: self.status.left.ended,
                right_ended: self.status.right.ended,
            });
            self.status.evicted_right_rows =
                checked(&self.name, self.status.evicted_right_rows, evicted.payloads)?;
            self.swept = Some(stamp);
        }
        self.status.output_watermark_micros = frontier
            .and_then(|time| time.checked_sub(1))
            .map(EventTime::from_micros)
            .or(self.status.output_watermark_micros);
        self.terminal = ended;
        if ended && self.state.is_empty() {
            self.state = super::ColumnarState::new(
                &self.name,
                &self.spec,
                [self.schemas[0].clone(), self.schemas[1].clone()],
                &self.runtime_pool,
            )?;
        }
        self.refresh_state_gauges();
        Ok(())
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
