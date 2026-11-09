use super::selection::Selection;
use crate::operator::rolling::{
    BTreeMap, BufferedRow, CompiledRollingSpec, LateMetricDelta, Result, RollingMetricsRecorder,
    RollingStreamState, RollingWork, RowIdentity, Schema, TableBatch, operator_error, state_codec,
};

pub(super) struct Classification<'a> {
    pub state: &'a RollingStreamState,
    pub compiled: &'a CompiledRollingSpec,
    pub node: &'a str,
    pub observer: Option<&'a RollingMetricsRecorder>,
}
impl Classification<'_> {
    pub(super) fn run(
        &self,
        table: &TableBatch,
        selection: &Selection,
        schema: &Schema,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<(BTreeMap<RowIdentity, BufferedRow>, LateMetricDelta)> {
        let mut accepted = BTreeMap::new();
        for selected in &selection.rows {
            check()?;
            let row = state_codec::read_buffered_row_checked(
                &table.batches()[selected.record],
                selected.offset,
                self.compiled,
                schema,
                self.node,
                check,
            )?;
            self.accept(&mut accepted, row, selected.envelope)?;
        }
        check()?;
        Ok((accepted, selection.delta))
    }
    fn accept(
        &self,
        accepted: &mut BTreeMap<RowIdentity, BufferedRow>,
        row: BufferedRow,
        input_row: usize,
    ) -> Result<()> {
        super::super::validate_row(
            &self.compiled.indicator_groups,
            &row.values,
            self.node,
            super::super::Coordinate::InputRow(input_row),
        )?;
        if let Some(recorder) = self.observer {
            recorder.add(RollingWork::OrderProofRows, 1);
        }
        if self.state.buffer.contains_key(&row.identity) || accepted.contains_key(&row.identity) {
            return Err(operator_error(
                self.node,
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
