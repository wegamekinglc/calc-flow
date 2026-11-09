//! Owns all diagnostic Arrow work before the first late emission.

use std::sync::Arc;

use datafusion::arrow::{array::ArrayRef, record_batch::RecordBatch};

use super::super::super::{
    chunk_requirements,
    footprint::Footprint,
    footprint_state, materialize,
    math_requirements::bytes::{add, slots, total},
    scalar_materialization_plan, scalar_output_plan,
};
use super::super::{
    retained,
    selection::{Filter, Row, Selection},
};
use super::{Check, Work};
use crate::EventTime;
use crate::operator::rolling::{
    Batch, Result, RollingStreamState, ScalarValue, SchemaRef, closing_coordinate, operator_error,
};

#[derive(Default)]
pub(super) struct Prepared {
    pub batches: Vec<Batch>,
    pub footprint: Footprint,
    pub reserved: u64,
}
struct Late<'a> {
    work: &'a mut Work,
    schema: SchemaRef,
    base: Footprint,
    scratch: u64,
}

impl Work {
    pub(super) fn prepare_late(
        &mut self,
        state: &mut RollingStreamState,
        check: Check<'_>,
    ) -> Result<Prepared> {
        let Some(schema) = self.late_schema.clone() else {
            return Ok(Prepared::default());
        };
        let (_, base) = footprint_state::state_checked(state, &self.node, check)?;
        let prepared = Late {
            work: self,
            schema,
            base,
            scratch: 0,
        }
        .run(check)?;
        let count = u64::try_from(prepared.batches.len())
            .map_err(|_| operator_error(&self.node, "late output chunk count overflowed"))?;
        state.next_late_output_sequence = state
            .next_late_output_sequence
            .checked_add(count)
            .ok_or_else(|| operator_error(&self.node, "late output sequence overflowed"))?;
        Ok(prepared)
    }
}
impl Late<'_> {
    fn reserve(&mut self, extra: u64) -> Result<()> {
        self.work.credit.adjust(
            self.base.clone(),
            add(self.scratch, extra)?,
            self.work.budget,
            &self.work.node,
        )
    }
    fn run(mut self, check: Check<'_>) -> Result<Prepared> {
        let selection = self.select(check)?;
        if selection.rows.is_empty() {
            return Ok(Prepared::default());
        }
        self.prepare_selected(selection, check)
    }
    fn prepare_selected(&mut self, selection: Selection, check: Check<'_>) -> Result<Prepared> {
        crate::operator::late_output::validate_scratch(
            selection.rows.len(),
            self.work.edge,
            &self.work.node,
        )?;
        self.admit_rows(&selection, check)?;
        let rows = self.rows(&selection, check)?;
        drop(selection);
        let plan = self.plan_output(&rows, check)?;
        let (record, footprint) = self.output(&rows, check)?;
        self.finish_output(record, footprint, plan.output_bytes, check)
    }
    fn finish_output(
        &mut self,
        record: RecordBatch,
        footprint: Footprint,
        quota: u64,
        check: Check<'_>,
    ) -> Result<Prepared> {
        let actual = footprint.incremental_bytes(&self.base, &self.work.node)?;
        let residual = quota.checked_sub(actual).ok_or_else(|| {
            operator_error(
                &self.work.node,
                "state.late_output exceeded the admitted component",
            )
        })?;
        let (batches, chunk_bytes) = self.chunk(&record, &footprint, quota, check)?;
        drop(record);
        check()?;
        Ok(Prepared {
            batches,
            footprint,
            reserved: add(residual, chunk_bytes)?,
        })
    }
    fn select(&mut self, check: Check<'_>) -> Result<Selection> {
        let count = self.work.input.num_rows();
        self.scratch = slots::<Row>(count)?;
        self.reserve(0)?;
        Filter {
            compiled: &self.work.compiled,
            watermark: self.work.watermark,
            policy: self.work.policy,
            lateness: self.work.lateness,
            node: &self.work.node,
        }
        .select_late(&self.work.input, count, check)
    }
    fn admit_rows(&mut self, selection: &Selection, check: Check<'_>) -> Result<()> {
        let table = self.work.input.table_payload()?;
        let row_columns = || {
            selection
                .rows
                .iter()
                .map(|row| (table.batches()[row.record].columns(), row.offset))
        };
        self.base.merge(
            &retained::plan_selected(row_columns(), &self.work.node, check)?,
            &self.work.node,
        )?;
        let values = scalar_materialization_plan::plan_selected(row_columns(), check)
            .map_err(|error| super::super::materialization_error(error, &self.work.node))?;
        self.scratch = total([
            Ok(self.scratch),
            Ok(values),
            slots::<Vec<ScalarValue>>(selection.rows.len()),
            slots::<ScalarValue>(
                selection
                    .rows
                    .len()
                    .checked_mul(self.schema.fields().len())
                    .ok_or_else(|| {
                        operator_error(&self.work.node, "late scalar count overflowed")
                    })?,
            ),
            slots::<ArrayRef>(self.schema.fields().len()),
            self.diagnostic_string_bytes(selection.rows.len()),
        ])?;
        self.reserve(0)
    }
    fn diagnostic_string_bytes(&self, rows: usize) -> Result<u64> {
        let per_row = total([
            u64::try_from(self.work.node.len()).map_err(|_| overflow(&self.work.node)),
            u64::try_from(self.work.input.metadata().source().len())
                .map_err(|_| overflow(&self.work.node)),
            Ok(5 + 8),
        ])?;
        per_row
            .checked_mul(u64::try_from(rows).map_err(|_| overflow(&self.work.node))?)
            .ok_or_else(|| overflow(&self.work.node))
    }
    fn rows(&self, selection: &Selection, check: Check<'_>) -> Result<Vec<Vec<ScalarValue>>> {
        let mut rows = Vec::with_capacity(selection.rows.len());
        for row in &selection.rows {
            check()?;
            rows.push(self.row(row, check)?);
        }
        Ok(rows)
    }
    fn row(&self, selected: &Row, check: Check<'_>) -> Result<Vec<ScalarValue>> {
        let record = &self.work.input.table_payload()?.batches()[selected.record];
        let mut values = Vec::with_capacity(self.schema.fields().len());
        for (index, column) in record.columns().iter().enumerate() {
            values.push(materialize::read_scalar_as(
                column.as_ref(),
                selected.offset,
                self.work.schema.field(index).data_type(),
                &self.work.node,
                check,
            )?);
        }
        let time = crate::operator::late_output::identity::event_time(
            record,
            selected.offset,
            self.work.compiled.event_time_index,
            &self.work.node,
            "rolling",
        )?;
        self.diagnostics(&mut values, selected, time)?;
        Ok(values)
    }
    fn diagnostics(&self, values: &mut Vec<ScalarValue>, row: &Row, time: i64) -> Result<()> {
        let index = u64::try_from(row.envelope).map_err(|_| overflow(&self.work.node))?;
        let closing = closing_coordinate(time, self.work.lateness, &self.work.node)?;
        values.extend([
            ScalarValue::Utf8(Some(self.work.node.as_str().to_owned())),
            ScalarValue::Utf8(Some("input".into())),
            ScalarValue::Int64(Some(time)),
            ScalarValue::Int64(Some(closing)),
            ScalarValue::Int64(Some(self.work.watermark.map_or(0, EventTime::as_micros))),
            ScalarValue::Utf8(Some("late_row".into())),
            ScalarValue::Utf8(Some(self.work.input.metadata().source().into())),
            ScalarValue::UInt64(Some(self.work.input.metadata().sequence())),
            ScalarValue::UInt64(Some(index)),
        ]);
        Ok(())
    }
    fn plan_output(
        &mut self,
        rows: &[Vec<ScalarValue>],
        check: Check<'_>,
    ) -> Result<scalar_output_plan::OutputPlan> {
        let mut cell = |col: usize, row: usize| rows[row].get(col);
        let prefix =
            scalar_output_plan::planning_workspace(&self.schema, rows.len(), &mut cell, check)?;
        self.reserve(prefix)?;
        let plan = scalar_output_plan::plan(&self.schema, rows.len(), &mut cell, check)?;
        self.reserve(add(plan.output_bytes, plan.scratch_bytes)?)?;
        Ok(plan)
    }
    fn output(
        &self,
        rows: &[Vec<ScalarValue>],
        check: Check<'_>,
    ) -> Result<(RecordBatch, Footprint)> {
        let mut columns = Vec::with_capacity(self.schema.fields().len());
        let mut footprint = Footprint::default();
        for (column, field) in self.schema.fields().iter().enumerate() {
            let array = scalar_output_plan::encode_column(
                field.data_type(),
                rows.len(),
                &mut |row| rows[row].get(column),
                check,
            )?;
            footprint.array_checked(&array, &self.work.node, check)?;
            columns.push(array);
        }
        let record = RecordBatch::try_new(Arc::clone(&self.schema), columns)
            .map_err(|_| operator_error(&self.work.node, "late output schema is invalid"))?;
        Ok((record, footprint))
    }
    fn chunk(
        &mut self,
        record: &RecordBatch,
        footprint: &Footprint,
        output_bytes: u64,
        check: Check<'_>,
    ) -> Result<(Vec<Batch>, u64)> {
        use crate::operator::output_chunk::{OutputChunkErrors, chunk_output_record_checked};
        let (source, plan) = self.chunk_plan(record, output_bytes, check)?;
        let batches = chunk_output_record_checked(
            record,
            &source,
            self.work.source.next_late_output_sequence,
            self.work.edge,
            OutputChunkErrors::ROLLING,
            check,
        )?;
        let mut base = self.base.clone();
        base.merge(footprint, &self.work.node)?;
        verify_chunks(&batches, &base, plan.output_bytes, &self.work.node, check)?;
        Ok((batches, plan.output_bytes))
    }
    fn chunk_plan(
        &mut self,
        record: &RecordBatch,
        output_bytes: u64,
        check: Check<'_>,
    ) -> Result<(String, chunk_requirements::ChunkRequirements)> {
        let source_bytes = late_source_bytes(&self.work.node)?;
        self.reserve(add(output_bytes, source_bytes)?)?;
        let source = source_name(&self.work.node, source_bytes)?;
        let plan = chunk_requirements::plan(record, self.work.edge, &source, check)?;
        self.admit_chunks(plan, output_bytes, source_bytes)?;
        Ok((source, plan))
    }
    fn admit_chunks(
        &mut self,
        plan: chunk_requirements::ChunkRequirements,
        output: u64,
        source: u64,
    ) -> Result<()> {
        self.reserve(total([
            Ok(output),
            Ok(plan.output_bytes),
            Ok(plan.scratch_bytes),
            Ok(source),
        ])?)
    }
}
fn verify_chunks(
    batches: &Vec<Batch>,
    base: &Footprint,
    quota: u64,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    let mut footprint = Footprint::default();
    footprint.add_owned(slots::<Batch>(batches.capacity())?, node)?;
    for batch in batches {
        accumulate_batch(&mut footprint, batch, node, check)?;
    }
    if footprint.incremental_bytes(base, node)? > quota {
        return Err(operator_error(
            node,
            "state.late_chunks exceeded the admitted component",
        ));
    }
    Ok(())
}
fn accumulate_batch(
    footprint: &mut Footprint,
    batch: &Batch,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    for record in batch.table_payload()?.batches() {
        for column in record.columns() {
            footprint.array_checked(column, node, check)?;
        }
    }
    Ok(())
}
fn overflow(node: &str) -> crate::CalcFlowError {
    operator_error(node, "late output byte bound overflowed")
}

fn late_source_bytes(node: &str) -> Result<u64> {
    let count = node.len().checked_add(5).ok_or_else(|| overflow(node))?;
    u64::try_from(count).map_err(|_| overflow(node))
}

fn source_name(node: &str, count: u64) -> Result<String> {
    let count = usize::try_from(count).map_err(|_| overflow(node))?;
    let mut value = String::with_capacity(count);
    value.push_str(node);
    value.push_str(".late");
    Ok(value)
}
