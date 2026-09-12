use super::{
    Arc, BTreeMap, Batch, BufferedRow, EventTime, LateMetricDelta, PreparedLateMetrics,
    RecordBatch, Result, RollingOperator, RowIdentity, StreamCollector, StreamOperatorContext,
    accumulate_late_metrics, closing_coordinate, operator_error, read_buffered_row,
    record_late_row,
};
use crate::operator::late_output::{
    LateOutputPlan, PreparedLateOutput,
    identity::{InputRow, input_rows},
};

struct PreparedInput<'a> {
    accepted: BTreeMap<RowIdentity, BufferedRow>,
    metrics: PreparedLateMetrics,
    late: LateOutputPlan<'a>,
}

type PendingInput<'a> = (
    BTreeMap<RowIdentity, BufferedRow>,
    LateMetricDelta,
    crate::operator::PreparedLateMetrics,
    PreparedLateOutput<'a>,
);

impl RollingOperator {
    pub(super) async fn process_late_data(
        &mut self,
        batch: &Batch,
        watermark: Option<EventTime>,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.validate_late_data(batch, context)?;
        if self.try_buffer_ordered(batch.table_payload()?, watermark, context.rolling_metrics())? {
            self.install_context_identity(context);
            return Ok(());
        }
        let (accepted, next_metrics, metrics, prepared) =
            self.prepare_late_callback(batch, watermark, context)?;
        self.state.late_output_failed = true;
        let next_sequence = prepared.emit(output).await?;
        self.state.buffer.extend(accepted);
        self.state.ordered.clear();
        self.state.metrics = next_metrics;
        self.state.next_late_output_sequence = next_sequence;
        metrics.commit();
        self.install_context_identity(context);
        self.state.late_output_failed = false;
        Ok(())
    }

    fn validate_late_data(&self, batch: &Batch, context: &StreamOperatorContext<'_>) -> Result<()> {
        crate::operator::late_output::ensure_can_continue(
            self.state.late_output_failed,
            context.operator_id(),
        )?;
        context.check_cancelled()?;
        self.input_ports[0].validate(batch, context.operator_id())
    }

    fn prepare_late_callback<'a>(
        &self,
        batch: &'a Batch,
        watermark: Option<EventTime>,
        context: &'a StreamOperatorContext<'_>,
    ) -> Result<PendingInput<'a>> {
        let input = self.prepare_late_input(batch, watermark, context)?;
        let delta = input.metrics.into_delta();
        let next_metrics = accumulate_late_metrics(self.state.metrics, delta)?;
        let metrics = context.prepare_window_metrics(delta)?;
        let output = input.late.prepare()?;
        Ok((input.accepted, next_metrics, metrics, output))
    }

    fn prepare_late_input<'a>(
        &self,
        batch: &'a Batch,
        watermark: Option<EventTime>,
        context: &'a StreamOperatorContext<'_>,
    ) -> Result<PreparedInput<'a>> {
        let node = context.operator_id();
        let late = LateOutputPlan::new(
            batch,
            node,
            watermark.map_or(0, EventTime::as_micros),
            context.output_budget(),
            self.state.next_late_output_sequence,
            Arc::clone(
                self.output_ports[1]
                    .schema()
                    .expect("late schema is compiled"),
            ),
        )?;
        let mut prepared = PreparedInput {
            accepted: self.prepare_ordered_buffer(context.rolling_metrics())?,
            metrics: PreparedLateMetrics::default(),
            late,
        };
        for row in input_rows(batch)? {
            self.classify_late_row(&mut prepared, &row, watermark, node)?;
        }
        Ok(prepared)
    }

    fn classify_late_row(
        &self,
        prepared: &mut PreparedInput<'_>,
        row: &InputRow<'_>,
        watermark: Option<EventTime>,
        node: &str,
    ) -> Result<()> {
        let event_time = self.read_late_identity(row.record, row.row_index, node)?;
        if self.is_late(event_time, watermark, row.envelope_index, node)? {
            self.stage_late_row(prepared, row, event_time, watermark, node)
        } else {
            let accepted = read_buffered_row(row.record, row.row_index, &self.compiled, node)?;
            self.stage_accepted_row(&mut prepared.accepted, accepted, node)
        }
    }

    fn stage_late_row(
        &self,
        prepared: &mut PreparedInput<'_>,
        row: &InputRow<'_>,
        event_time: i64,
        watermark: Option<EventTime>,
        node: &str,
    ) -> Result<()> {
        record_late_row(&mut prepared.metrics, watermark, event_time, node)?;
        let closing = closing_coordinate(event_time, self.spec.allowed_lateness_micros, node)?;
        prepared.late.push(
            row.record_index,
            row.row_index,
            row.diagnostic_index(node)?,
            event_time,
            closing,
        )
    }

    pub(super) fn read_late_identity(
        &self,
        record: &RecordBatch,
        row: usize,
        node: &str,
    ) -> Result<i64> {
        use crate::operator::late_output::identity::{event_time, validate_keys};
        let time = event_time(record, row, self.compiled.event_time_index, node, "rolling")?;
        let keys = self
            .compiled
            .partition_columns
            .iter()
            .map(|column| (column.index, false))
            .chain(
                self.compiled
                    .sequence_columns
                    .iter()
                    .map(|column| (column.index, true)),
            );
        validate_keys(record, row, keys, node, "rolling")?;
        Ok(time)
    }

    fn stage_accepted_row(
        &self,
        accepted: &mut BTreeMap<RowIdentity, BufferedRow>,
        row: BufferedRow,
        node: &str,
    ) -> Result<()> {
        if self.state.buffer.contains_key(&row.identity) || accepted.contains_key(&row.identity) {
            return Err(operator_error(
                node,
                &format!(
                    "duplicate row identity at event_time_micros={}",
                    row.identity.event_time
                ),
            ));
        }
        accepted.insert(row.identity.clone(), row);
        Ok(())
    }
}
