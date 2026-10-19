//! Pack immutable worker records into envelopes sized by the downstream edge.

use std::collections::BTreeMap;

use datafusion::arrow::{array::ArrayRef, record_batch::RecordBatch};

use super::FundedMorsel;
use crate::operator::join::{columnar::Quantum, operator_error};
use crate::operator::output_chunk::validate_output_sequence_range;
use crate::{Batch, BatchMetadata, EdgeBudget, Result, StreamCollector, StreamOperatorContext};

#[derive(Default)]
struct Envelope {
    rows: usize,
    records: usize,
}

#[derive(Clone)]
struct Records<'a> {
    morsels: &'a [FundedMorsel],
    morsel: usize,
    record: usize,
}

impl<'a> Iterator for Records<'a> {
    type Item = &'a RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(morsel) = self.morsels.get(self.morsel) {
            if let Some(record) = morsel.records.get(self.record) {
                self.record += 1;
                if record.num_rows() > 0 {
                    return Some(record);
                }
            } else {
                self.morsel += 1;
                self.record = 0;
            }
        }
        None
    }
}

pub(super) fn control_bytes(records: usize, columns: usize) -> Option<usize> {
    let per_record = (2 * size_of::<Envelope>())
        .checked_add(2 * size_of::<RecordBatch>())?
        .checked_add(columns.checked_mul(size_of::<ArrayRef>())?)?
        .checked_add(256)?;
    records.checked_mul(per_record)
}

pub(super) async fn emit(
    morsels: &[FundedMorsel],
    name: &str,
    context: &StreamOperatorContext<'_>,
    output: &mut dyn StreamCollector,
    sequence: &mut u64,
) -> Result<()> {
    let records = Records {
        morsels,
        morsel: 0,
        record: 0,
    };
    let plan = plan(records.clone(), name, context).await?;
    validate_output_sequence_range(name, *sequence, plan.len())?;
    let mut records = records.peekable();
    let mut offset = 0;
    let mut quantum = Quantum::default();
    for envelope in plan {
        context.check_cancelled()?;
        let mut parts = Vec::with_capacity(envelope.records);
        let mut remaining = envelope.rows;
        while remaining > 0 {
            quantum.step(context, 1, 0).await?;
            let record = records.peek().expect("planned output record");
            let rows = remaining.min(record.num_rows() - offset);
            parts.push(slice(record, offset, rows));
            remaining -= rows;
            offset += rows;
            if offset == record.num_rows() {
                records.next();
                offset = 0;
            }
        }
        let metadata = BatchMetadata::new(name, *sequence, BTreeMap::new())?;
        let batch = Batch::table(parts, metadata)?;
        output.emit("output", batch).await?;
        *sequence += 1;
    }
    Ok(())
}

async fn plan(
    records: Records<'_>,
    name: &str,
    context: &StreamOperatorContext<'_>,
) -> Result<Vec<Envelope>> {
    let capacity = records
        .morsels
        .iter()
        .try_fold(0_usize, |count, morsel| {
            count.checked_add(morsel.records.len())
        })
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| operator_error(name, "Join envelope plan size overflowed"))?;
    let mut result = Vec::with_capacity(capacity);
    let budget = context.output_budget();
    let mut current = Envelope::default();
    let mut bytes = 0;
    let mut quantum = Quantum::default();
    for record in records {
        quantum.step(context, 1, 0).await?;
        let whole_bytes = measure(record, 0, record.num_rows())?;
        if record.num_rows() > budget.max_rows || whole_bytes > budget.max_bytes {
            return Err(operator_error(
                name,
                "validated Join chunk exceeded its row or byte budget",
            ));
        }
        let mut offset = 0;
        while offset < record.num_rows() {
            context.check_cancelled()?;
            let available = EdgeBudget {
                max_rows: budget.max_rows - current.rows,
                max_bytes: budget.max_bytes - bytes,
            };
            let (rows, charge) = prefix(record, offset, whole_bytes, available)?;
            if rows > 0 {
                current.rows += rows;
                current.records += 1;
                bytes += charge;
                offset += rows;
            }
            if offset < record.num_rows()
                || current.rows == budget.max_rows
                || bytes == budget.max_bytes
            {
                push(&mut result, std::mem::take(&mut current), capacity, name)?;
                bytes = 0;
            }
        }
    }
    if current.rows > 0 {
        push(&mut result, current, capacity, name)?;
    }
    Ok(result)
}

fn push(plan: &mut Vec<Envelope>, envelope: Envelope, capacity: usize, name: &str) -> Result<()> {
    if envelope.rows == 0 || plan.len() == capacity {
        return Err(operator_error(
            name,
            "Join envelope exceeded its funded plan",
        ));
    }
    plan.push(envelope);
    Ok(())
}

fn prefix(
    record: &RecordBatch,
    offset: usize,
    whole_bytes: usize,
    budget: EdgeBudget,
) -> Result<(usize, usize)> {
    let limit = (record.num_rows() - offset).min(budget.max_rows);
    if limit == 0 {
        return Ok((0, 0));
    }
    let bytes = if offset == 0 && limit == record.num_rows() {
        whole_bytes
    } else {
        measure(record, offset, limit)?
    };
    if bytes <= budget.max_bytes {
        return Ok((limit, bytes));
    }
    let mut low = 0;
    let mut high = limit - 1;
    let mut charge = 0;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let bytes = measure(record, offset, middle)?;
        if bytes <= budget.max_bytes {
            low = middle;
            charge = bytes;
        } else {
            high = middle - 1;
        }
    }
    Ok((low, charge))
}

fn slice(record: &RecordBatch, offset: usize, rows: usize) -> RecordBatch {
    if offset == 0 && rows == record.num_rows() {
        record.clone()
    } else {
        record.slice(offset, rows)
    }
}

fn measure(record: &RecordBatch, offset: usize, rows: usize) -> Result<usize> {
    Batch::table(vec![slice(record, offset, rows)], BatchMetadata::default())?.estimated_bytes()
}
