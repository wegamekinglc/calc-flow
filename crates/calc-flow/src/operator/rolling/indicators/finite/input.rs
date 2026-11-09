use datafusion::arrow::record_batch::RecordBatch;

use super::super::{input::retained, scalar_materialization_plan};
use super::{Check, Work, bytes, owned_slots};
use crate::operator::rolling::{
    BufferedRow, KeyValue, Result, ScalarValue, operator_error, state_codec,
};

impl Work {
    pub(super) fn read(&mut self, check: Check<'_>) -> Result<Vec<BufferedRow>> {
        let count = self.input_count(check)?;
        self.admit_input(count, check)?;
        let rows = self.read_rows(count, check)?;
        check()?;
        Ok(rows)
    }

    fn admit_input(&mut self, count: usize, check: Check<'_>) -> Result<()> {
        self.quote = retained::plan(self.input.table_payload()?, &self.node, check)?;
        self.scratch = self.input_requirement(count, check)?;
        self.reserve(0)
    }

    fn input_count(&self, check: Check<'_>) -> Result<usize> {
        let mut count = 0_usize;
        for record in self.input.table_payload()?.batches() {
            check()?;
            count = count
                .checked_add(record.num_rows())
                .ok_or_else(bytes::invalid)?;
        }
        Ok(count)
    }

    fn input_requirement(&self, count: usize, check: Check<'_>) -> Result<u64> {
        let mut total = owned_slots::<BufferedRow>(count)?;
        for record in self.input.table_payload()?.batches() {
            check()?;
            total = bytes::add(total, self.record_requirement(record, check)?)?;
        }
        Ok(total)
    }

    fn record_requirement(&self, record: &RecordBatch, check: Check<'_>) -> Result<u64> {
        bytes::total([
            bytes::repeat(
                owned_slots::<ScalarValue>(record.num_columns()),
                record.num_rows(),
            ),
            scalar_materialization_plan::plan(record.columns(), check),
            self.key_requirement(record, check),
        ])
    }

    fn key_requirement(&self, record: &RecordBatch, check: Check<'_>) -> Result<u64> {
        let vectors = bytes::total([
            bytes::growing::<Option<KeyValue>>(self.compiled.partition_columns.len()),
            bytes::growing::<KeyValue>(self.compiled.sequence_columns.len()),
        ])?;
        let mut total = bytes::repeat(Ok(vectors), record.num_rows())?;
        for key in self
            .compiled
            .partition_columns
            .iter()
            .chain(&self.compiled.sequence_columns)
        {
            check()?;
            total = bytes::add(
                total,
                scalar_materialization_plan::plan(&record.columns()[key.index..=key.index], check)?,
            )?;
        }
        Ok(total)
    }

    fn read_rows(&self, count: usize, check: Check<'_>) -> Result<Vec<BufferedRow>> {
        let mut rows = Vec::with_capacity(count);
        for record in self.input.table_payload()?.batches() {
            for index in 0..record.num_rows() {
                check()?;
                let row = self.read_row(record, index, rows.len(), check)?;
                rows.push(row);
            }
        }
        if rows.len() != count {
            return Err(operator_error(&self.node, "finite input row count changed"));
        }
        Ok(rows)
    }

    fn read_row(
        &self,
        record: &RecordBatch,
        row: usize,
        envelope: usize,
        check: Check<'_>,
    ) -> Result<BufferedRow> {
        let value = state_codec::read_buffered_row_checked(
            record,
            row,
            &self.compiled,
            &self.input_schema,
            &self.node,
            check,
        )?;
        super::super::validate_row(
            &self.compiled.indicator_groups,
            &value.values,
            &self.node,
            super::super::Coordinate::InputRow(envelope),
        )?;
        Ok(value)
    }
}
