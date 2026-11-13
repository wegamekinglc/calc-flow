use super::{BoundedOrderOperator, SortHistory, owned};
use crate::operator::stream::{LateMetricDelta, PreparedLateMetrics, accumulate_late_metrics};
use crate::{Batch, LatePolicySpec, Result, StreamOperatorContext};
use datafusion::arrow::record_batch::RecordBatch;

impl BoundedOrderOperator {
    pub(super) const fn metric_storage(&self) -> usize {
        match self.spec.late_policy() {
            LatePolicySpec::Drop { .. } => 256,
            _ => 0,
        }
    }
    pub(super) fn closed_data(
        &mut self,
        raw: Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let state = self.stream.as_ref().unwrap();
        let pool: std::sync::Arc<dyn datafusion::execution::memory_pool::MemoryPool> =
            state.pool.clone();
        let _transaction = owned::reserve(&pool, 512, "late metric transaction")?;
        let delta = self.late_delta(&raw, context)?;
        let update = self.prepare_late(delta, context)?;
        drop(raw);
        context.check_cancelled()?;
        if let Some((next, staged)) = update {
            let state = self.stream.as_mut().unwrap();
            state.metrics = next;
            state.capture = None;
            staged.commit();
        }
        Ok(())
    }
    fn late_delta(
        &self,
        raw: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<LateMetricDelta> {
        let SortHistory::ClosedWindow {
            event_time_column,
            start_micros,
            end_micros,
        } = self.spec.history()
        else {
            return Err(owned::error("data after final EOF"));
        };
        let watermark = self
            .stream
            .as_ref()
            .unwrap()
            .watermark
            .filter(|w| w.as_micros() >= *end_micros)
            .ok_or_else(|| owned::error("data after final EOF"))?
            .as_micros();
        let mut delta = raw.table_payload()?.batches().iter().try_fold(
            LateMetricDelta::default(),
            |delta, record| {
                accumulate_late_metrics(
                    delta,
                    scan(
                        record,
                        event_time_column,
                        *start_micros..*end_micros,
                        watermark,
                        &|| context.check_cancelled(),
                    )?,
                )
            },
        )?;
        delta.affected_batches = u64::from(delta.late_rows > 0);
        Ok(delta)
    }
    fn prepare_late(
        &self,
        delta: LateMetricDelta,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<(LateMetricDelta, PreparedLateMetrics)>> {
        if delta.late_rows == 0 {
            return Ok(None);
        }
        if !matches!(self.spec.late_policy(), LatePolicySpec::Drop { .. }) {
            return Err(owned::error("late data after final relation"));
        }
        let next = accumulate_late_metrics(self.stream.as_ref().unwrap().metrics, delta)?;
        let staged = context.prepare_window_metrics(delta)?;
        Ok(Some((next, staged)))
    }
}
fn scan(
    record: &RecordBatch,
    column: &str,
    window: std::ops::Range<i64>,
    watermark: i64,
    check: &dyn Fn() -> Result<()>,
) -> Result<LateMetricDelta> {
    let mut delta = LateMetricDelta::default();
    for row in 0..record.num_rows() {
        owned::check_row(row, check)?;
        let Some(ts) = owned::event_time(record, row, column)? else {
            continue;
        };
        if window.contains(&ts) {
            let lateness =
                u64::try_from(i128::from(watermark) - i128::from(ts)).map_err(owned::error)?;
            delta = accumulate_late_metrics(
                delta,
                LateMetricDelta {
                    late_rows: 1,
                    max_lateness_micros: Some(lateness),
                    ..LateMetricDelta::default()
                },
            )?;
        }
    }
    Ok(delta)
}
