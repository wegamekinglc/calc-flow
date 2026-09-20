use super::{
    AcceptedRows, BTreeSet, Batch, BufferedRow, CrossSectionOperator, EventTime, GroupKey,
    LateMetricDelta, RecordBatch, Result, RowIdentity, StreamCollector, StreamOperatorContext,
    operator_error, read_row,
};
use crate::operator::PreparedLateMetrics;
use crate::operator::late_output::{
    self, LateOutputPlan, LateRowTally, PreparedLateOutput,
    identity::{InputRow, event_time, input_rows, validate_keys},
    record_late_row,
};

struct PreparedInput<'a> {
    accepted: AcceptedRows,
    metrics: LateRowTally,
    late: LateOutputPlan<'a>,
    identities: BTreeSet<RowIdentity>,
}

type PendingInput<'a> = (
    AcceptedRows,
    LateMetricDelta,
    PreparedLateMetrics,
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
        late_output::validate_late_data(
            self.state.late_output_failed,
            &self.input_ports[0],
            batch,
            context,
        )?;
        let (accepted, next_metrics, metrics, prepared) =
            self.prepare_late_callback(batch, watermark, context)?;
        // Poison before emit: only the commit below clears the flag, so a
        // failed or cancelled emission forbids live retry; a durable cut
        // is the only recovery. This poison -> emit -> commit -> unpoison
        // sequence is deliberately duplicated in rolling/late.rs
        // (borrow/async constraints block a shared helper); change both.
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

    fn prepare_late_callback<'a>(
        &self,
        batch: &'a Batch,
        watermark: Option<EventTime>,
        context: &'a StreamOperatorContext<'_>,
    ) -> Result<PendingInput<'a>> {
        let input = self.prepare_late_input(batch, watermark, context)?;
        late_output::prepare_late_callback(
            input.accepted,
            input.metrics,
            input.late,
            self.state.metrics,
            context,
        )
    }

    fn prepare_late_input<'a>(
        &self,
        batch: &'a Batch,
        watermark: Option<EventTime>,
        context: &'a StreamOperatorContext<'_>,
    ) -> Result<PreparedInput<'a>> {
        let node = context.operator_id();
        let late = late_output::new_plan(
            batch,
            self.state.next_late_output_sequence,
            &self.output_ports[1],
            watermark,
            context,
        )?;
        let mut prepared = PreparedInput {
            accepted: Vec::new(),
            metrics: LateRowTally::default(),
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
