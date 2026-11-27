use super::{
    StreamAsofJoinOperator, StreamAsofJoinStatus, SweepStamp, checked,
    columnar::{AlignedRuns, ColumnarState, OutputPlan, Progress, Selection},
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
    selection: PreparedSelection,
    batch: Batch,
    _workspace: MemoryReservation,
    _scratch: MemoryReservation,
}

enum PreparedSelection {
    Rows(Selection),
    Aligned(AlignedRuns),
}

impl PreparedSelection {
    fn len(&self) -> usize {
        match self {
            Self::Rows(selection) => selection.len(),
            Self::Aligned(runs) => runs.rows,
        }
    }

    fn matched(&self) -> usize {
        match self {
            Self::Rows(selection) => selection.matched(),
            Self::Aligned(runs) => runs.rows,
        }
    }

    fn plan<'a>(
        &'a self,
        state: &'a ColumnarState,
        projection: Option<&'a [(usize, usize)]>,
    ) -> Result<OutputPlan<'a>> {
        match self {
            Self::Rows(rows) => state.plan_output(rows, projection),
            Self::Aligned(runs) => state.plan_aligned_output(*runs, projection),
        }
    }

    fn commit(self, state: &mut ColumnarState) -> Result<()> {
        match self {
            Self::Rows(rows) => state.commit(&rows),
            Self::Aligned(runs) => state.commit_alignment(runs),
        }
    }
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
            let matched = prepared.selection.matched() as u64;
            let status = self.output_status(rows, matched)?;
            let next_output_sequence = checked(&self.name, self.next_output_sequence, rows as u64)?;
            context.check_cancelled()?;
            tokio::select! {
                biased;
                // An acknowledged delivery belongs to the committed prefix.
                result = output.emit("output", prepared.batch) => result?,
                () = context.cancellation().cancelled() => return context.check_cancelled(),
            }
            prepared.selection.commit(&mut self.state)?;
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
        let selection = self.state.complete_alignment(count).map_or_else(
            || PreparedSelection::Rows(self.state.select_prefix(count)),
            PreparedSelection::Aligned,
        );
        let plan = selection.plan(&self.state, self.output_projection.as_deref())?;
        let workspace = self
            .reserve_workspace(u64::try_from(plan.charge).expect("ASOF output charge fits u64"))?;
        #[cfg(test)]
        MATERIALIZATIONS.with(|count| count.set(count.get() + 1));
        let record = plan.materialize(self.outputs[0].schema().expect("exact ASOF output"))?;
        let batch =
            self.prepare_output_batch(selection.len(), record, context.output_budget().max_bytes)?;
        Ok(PreparedOutput {
            selection,
            batch,
            _workspace: workspace,
            _scratch: scratch,
        })
    }

    fn prepare_output_batch(
        &self,
        rows: usize,
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
        debug_assert_eq!(batch.num_rows(), rows);
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
            self.state = ColumnarState::new(
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
