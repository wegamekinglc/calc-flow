use super::{
    AcceptedRows, Arc, BTreeSet, Batch, BufferedRow, CrossSectionOperator, EventTime, GroupKey,
    LateMetricDelta, PreparedLateMetrics, RecordBatch, Result, RowIdentity, StreamCollector,
    StreamOperatorContext, accumulate_late_metrics, operator_error, read_row, record_late_row,
};
use crate::operator::late_output::{
    LateOutputPlan, PreparedLateOutput,
    identity::{InputRow, input_rows},
};

struct PreparedInput<'a> {
    accepted: AcceptedRows,
    metrics: PreparedLateMetrics,
    late: LateOutputPlan<'a>,
    identities: BTreeSet<RowIdentity>,
}

type PendingInput<'a> = (
    AcceptedRows,
    LateMetricDelta,
    crate::operator::PreparedLateMetrics,
    PreparedLateOutput<'a>,
);

impl CrossSectionOperator {
    pub(super) async fn process_late_data(
        &mut self,
        batch: &Batch,
        watermark: Option<EventTime>,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.validate_late_data(batch, context)?;
        let (accepted, next_metrics, metrics, prepared) =
            self.prepare_late_callback(batch, watermark, context)?;
        self.state.late_output_failed = true;
        let next_sequence = prepared.emit(output).await?;
        self.install_accepted_rows(accepted);
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
            accepted: Vec::new(),
            metrics: PreparedLateMetrics::default(),
            late,
            identities: BTreeSet::new(),
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
        let group = GroupKey {
            base: self.compiled.group_base(event_time, node)?,
            partition: Vec::new(),
        };
        if self.is_late(&group, watermark, row.envelope_index, event_time, node)? {
            self.stage_late_row(prepared, row, &group, event_time, watermark, node)
        } else {
            self.stage_normal_row(prepared, row, node)
        }
    }

    fn stage_late_row(
        &self,
        prepared: &mut PreparedInput<'_>,
        row: &InputRow<'_>,
        group: &GroupKey,
        event_time: i64,
        watermark: Option<EventTime>,
        node: &str,
    ) -> Result<()> {
        record_late_row(&mut prepared.metrics, watermark, event_time, node)?;
        let closing = self.compiled.group_closing(group, node)?;
        prepared.late.push(
            row.record_index,
            row.row_index,
            row.diagnostic_index(node)?,
            event_time,
            closing,
        )
    }

    fn stage_normal_row(
        &self,
        prepared: &mut PreparedInput<'_>,
        row: &InputRow<'_>,
        node: &str,
    ) -> Result<()> {
        let accepted = read_row(row.record, row.row_index, &self.compiled, node)?;
        let group = self.compiled.group_key(&accepted, node)?;
        self.stage_accepted_row(
            &mut prepared.accepted,
            &mut prepared.identities,
            group,
            accepted,
            node,
        )
    }

    pub(super) fn read_late_identity(
        &self,
        record: &RecordBatch,
        row: usize,
        node: &str,
    ) -> Result<i64> {
        use crate::operator::late_output::identity::{event_time, validate_keys};
        let time = event_time(
            record,
            row,
            self.compiled.event_time_index,
            node,
            "cross-section",
        )?;
        let keys = self
            .compiled
            .entity_columns
            .iter()
            .map(|column| (column.index, false))
            .chain(
                self.compiled
                    .sequence_columns
                    .iter()
                    .map(|column| (column.index, true)),
            );
        validate_keys(record, row, keys, node, "cross-section")?;
        validate_keys(
            record,
            row,
            self.compiled
                .partition_columns
                .iter()
                .map(|column| (column.index, false)),
            node,
            "cross-section",
        )?;
        Ok(time)
    }

    fn stage_accepted_row(
        &self,
        accepted: &mut AcceptedRows,
        identities: &mut BTreeSet<RowIdentity>,
        group: GroupKey,
        row: BufferedRow,
        node: &str,
    ) -> Result<()> {
        if self.state.identity_groups.contains_key(&row.identity)
            || !identities.insert(row.identity.clone())
        {
            return Err(operator_error(
                node,
                &format!(
                    "duplicate row identity at event_time_micros={}",
                    row.identity.event_time
                ),
            ));
        }
        accepted.push((group, row.identity.clone(), row));
        Ok(())
    }
}
