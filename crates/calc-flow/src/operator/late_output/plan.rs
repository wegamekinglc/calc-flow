use std::{collections::BTreeMap, sync::Arc};

use datafusion::arrow::{
    array::{ArrayRef, Int64Array, StringArray, UInt64Array},
    datatypes::SchemaRef,
    record_batch::RecordBatch,
};

use crate::{Batch, BatchMetadata, CalcFlowError, EdgeBudget, Result, StreamCollector};

struct LateRow {
    record: usize,
    row: usize,
    index: u64,
    event_time: i64,
    closing_time: i64,
    bytes: usize,
}

pub(crate) struct LateOutputPlan<'a> {
    input: &'a Batch,
    schema: SchemaRef,
    node: &'a str,
    watermark: i64,
    first_sequence: u64,
    budget: EdgeBudget,
    rows: Vec<LateRow>,
}

pub(crate) struct PreparedLateOutput<'a>(LateOutputPlan<'a>, u64);

impl<'a> LateOutputPlan<'a> {
    #[cfg(test)]
    pub(crate) fn scratch_usage(&self) -> (usize, usize) {
        (self.rows.len(), self.rows.capacity() * size_of::<LateRow>())
    }

    #[cfg(test)]
    pub(crate) const fn scratch_row_bytes() -> usize {
        size_of::<LateRow>()
    }

    pub(crate) fn new(
        input: &'a Batch,
        node: &'a str,
        watermark: i64,
        budget: EdgeBudget,
        first_sequence: u64,
        schema: SchemaRef,
    ) -> Result<Self> {
        input.table_payload()?;
        EdgeBudget::new(budget.max_rows, budget.max_bytes)?;
        Ok(Self {
            input,
            schema,
            node,
            watermark,
            first_sequence,
            budget,
            rows: Vec::new(),
        })
    }

    pub(crate) fn push(
        &mut self,
        record: usize,
        row: usize,
        index: u64,
        event_time: i64,
        closing_time: i64,
    ) -> Result<()> {
        self.check_scratch_capacity()?;
        self.reserve_scratch()?;
        self.rows.push(LateRow {
            record,
            row,
            index,
            event_time,
            closing_time,
            bytes: 0,
        });
        Ok(())
    }

    fn check_scratch_capacity(&self) -> Result<()> {
        let count = self
            .rows
            .len()
            .checked_add(1)
            .ok_or_else(|| error(self.node, "late scratch row count overflowed"))?;
        let bytes = count
            .checked_mul(size_of::<LateRow>())
            .ok_or_else(|| error(self.node, "late scratch byte count overflowed"))?;
        if count > self.budget.max_rows || bytes > self.budget.max_bytes {
            return Err(error(
                self.node,
                &format!(
                    "late scratch requires rows={count}, bytes={bytes}; limits rows={}, bytes={}",
                    self.budget.max_rows, self.budget.max_bytes
                ),
            ));
        }
        Ok(())
    }

    fn reserve_scratch(&mut self) -> Result<()> {
        if self.rows.is_empty() {
            let capacity = self
                .input
                .num_rows()
                .min(self.budget.max_rows)
                .min(self.budget.max_bytes / size_of::<LateRow>());
            self.rows.try_reserve_exact(capacity).map_err(|cause| {
                error(
                    self.node,
                    &format!("late scratch allocation failed: {cause}"),
                )
            })?;
        }
        Ok(())
    }

    pub(crate) fn prepare(mut self) -> Result<PreparedLateOutput<'a>> {
        for index in 0..self.rows.len() {
            self.rows[index].bytes = self.prepare_row(&self.rows[index])?;
        }
        if !self.rows.is_empty() {
            BatchMetadata::new(
                format!("{}.late", self.node),
                self.first_sequence,
                BTreeMap::new(),
            )?;
        }
        let next = self.next_sequence()?;
        Ok(PreparedLateOutput(self, next))
    }

    fn row_bytes(&self, row: &LateRow) -> Result<usize> {
        let input = self.input.table_payload()?.batches()[row.record].slice(row.row, 1);
        let raw_bytes = Batch::table(vec![input], BatchMetadata::default())?.estimated_bytes()?;
        let bytes = raw_bytes
            .checked_add(self.diagnostic_bytes()?)
            .ok_or_else(|| error(self.node, "late row byte count overflowed"))?;
        Ok(bytes)
    }

    fn prepare_row(&self, row: &LateRow) -> Result<usize> {
        let bytes = self.row_bytes(row)?;
        if bytes > self.budget.max_bytes {
            return Err(error(
                self.node,
                &format!(
                    "output_row_too_large: output late row_index={} requires {bytes} bytes, exceeding max_bytes={}",
                    row.index, self.budget.max_bytes
                ),
            ));
        }
        // Validate Arrow construction before the first emit, without retaining chunks.
        self.record(row)?;
        Ok(bytes)
    }

    fn next_sequence(&self) -> Result<u64> {
        let mut next = self.first_sequence;
        let mut start = 0;
        while start < self.rows.len() {
            start = self.chunk_end(start);
            next = next.checked_add(1).ok_or_else(|| {
                error(self.node, "late output sequence overflowed before emission")
            })?;
        }
        Ok(next)
    }

    fn diagnostic_bytes(&self) -> Result<usize> {
        // Five 64-bit values, four Utf8 offsets, and the constant string values.
        [self.node, self.input.metadata().source()]
            .into_iter()
            .try_fold(69_usize, |bytes, value| {
                i32::try_from(value.len())
                    .map_err(|_| error(self.node, "late diagnostic Utf8 offset overflowed"))?;
                bytes
                    .checked_add(value.len())
                    .ok_or_else(|| error(self.node, "late diagnostic byte count overflowed"))
            })
    }

    fn chunk_end(&self, start: usize) -> usize {
        let mut end = start;
        let mut bytes = 0;
        while end < self.rows.len() && end - start < self.budget.max_rows {
            let cost = self.rows[end].bytes;
            if cost > self.budget.max_bytes - bytes {
                break;
            }
            bytes += cost;
            end += 1;
        }
        end
    }

    fn record(&self, row: &LateRow) -> Result<RecordBatch> {
        let table = self.input.table_payload()?;
        let input = table.batches()[row.record].slice(row.row, 1);
        let string = |value: &str| Arc::new(StringArray::from(vec![value])) as ArrayRef;
        let int = |value| Arc::new(Int64Array::from(vec![value])) as ArrayRef;
        let uint = |value| Arc::new(UInt64Array::from(vec![value])) as ArrayRef;
        let diagnostics = [
            string(self.node),
            string("input"),
            int(row.event_time),
            int(row.closing_time),
            int(self.watermark),
            string("late_row"),
            string(self.input.metadata().source()),
            uint(self.input.metadata().sequence()),
            uint(row.index),
        ];
        RecordBatch::try_new(
            Arc::clone(&self.schema),
            input.columns().iter().cloned().chain(diagnostics).collect(),
        )
        .map_err(|error| CalcFlowError::Internal {
            message: error.to_string(),
        })
    }
}

impl PreparedLateOutput<'_> {
    pub(crate) async fn emit(self, output: &mut dyn StreamCollector) -> Result<u64> {
        let Self(plan, next) = self;
        let mut start = 0;
        let mut sequence = plan.first_sequence;
        while start < plan.rows.len() {
            let end = plan.chunk_end(start);
            let records = plan.rows[start..end]
                .iter()
                .map(|row| plan.record(row))
                .collect::<Result<Vec<_>>>()?;
            let metadata =
                BatchMetadata::new(format!("{}.late", plan.node), sequence, BTreeMap::new())?;
            output
                .emit("late", Batch::table(records, metadata)?)
                .await?;
            start = end;
            sequence += 1;
        }
        Ok(next)
    }
}

fn error(node: &str, message: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node.into(),
        message: message.into(),
    }
}
