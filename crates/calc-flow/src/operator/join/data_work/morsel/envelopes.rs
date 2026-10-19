//! Pack immutable worker records into envelopes sized by the downstream edge.

use std::{collections::BTreeMap, iter::Peekable};

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
    let plan = plan(records.clone(), name, context, *sequence).await?;
    let mut records = records.peekable();
    let mut offset = 0;
    let mut quantum = Quantum::default();
    for envelope in plan {
        context.check_cancelled()?;
        let mut parts = Vec::with_capacity(envelope.records);
        let mut remaining = envelope.rows;
        while remaining > 0 {
            quantum.step(context, 1, 0).await?;
            append_part(&mut parts, &mut records, &mut offset, &mut remaining);
        }
        let batch = output_batch(name, *sequence, parts)?;
        output.emit("output", batch).await?;
        *sequence += 1;
    }
    Ok(())
}

fn append_part(
    parts: &mut Vec<RecordBatch>,
    records: &mut Peekable<Records<'_>>,
    offset: &mut usize,
    remaining: &mut usize,
) {
    let record = records.peek().expect("planned output record");
    let rows = (*remaining).min(record.num_rows() - *offset);
    parts.push(slice(record, *offset, rows));
    *remaining -= rows;
    *offset += rows;
    if *offset == record.num_rows() {
        records.next();
        *offset = 0;
    }
}

fn output_batch(name: &str, sequence: u64, parts: Vec<RecordBatch>) -> Result<Batch> {
    let metadata = BatchMetadata::new(name, sequence, BTreeMap::new())?;
    Batch::table(parts, metadata)
}

async fn plan(
    records: Records<'_>,
    name: &str,
    context: &StreamOperatorContext<'_>,
    sequence: u64,
) -> Result<Vec<Envelope>> {
    let capacity = plan_capacity(records.morsels, name)?;
    let mut builder = EnvelopePlan {
        result: Vec::with_capacity(capacity),
        capacity,
        current: Envelope::default(),
        bytes: 0,
    };
    let budget = context.output_budget();
    let mut quantum = Quantum::default();
    for record in records {
        quantum.step(context, 1, 0).await?;
        builder.append(record, budget, name, context)?;
    }
    let result = builder.finish(name)?;
    validate_output_sequence_range(name, sequence, result.len())?;
    Ok(result)
}

fn plan_capacity(morsels: &[FundedMorsel], name: &str) -> Result<usize> {
    morsels
        .iter()
        .try_fold(0_usize, |count, morsel| {
            count.checked_add(morsel.records.len())
        })
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| operator_error(name, "Join envelope plan size overflowed"))
}

struct EnvelopePlan {
    result: Vec<Envelope>,
    capacity: usize,
    current: Envelope,
    bytes: usize,
}

impl EnvelopePlan {
    fn append(
        &mut self,
        record: &RecordBatch,
        budget: EdgeBudget,
        name: &str,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let whole_bytes = validated_bytes(record, budget, name)?;
        let mut offset = 0;
        while offset < record.num_rows() {
            context.check_cancelled()?;
            let available = EdgeBudget {
                max_rows: budget.max_rows - self.current.rows,
                max_bytes: budget.max_bytes - self.bytes,
            };
            let (rows, charge) = prefix(record, offset, whole_bytes, available)?;
            if rows > 0 {
                self.current.rows += rows;
                self.current.records += 1;
                self.bytes += charge;
                offset += rows;
            }
            if self.needs_flush(record, offset, budget) {
                self.flush(name)?;
            }
        }
        Ok(())
    }

    fn needs_flush(&self, record: &RecordBatch, offset: usize, budget: EdgeBudget) -> bool {
        offset < record.num_rows()
            || self.current.rows == budget.max_rows
            || self.bytes == budget.max_bytes
    }

    fn flush(&mut self, name: &str) -> Result<()> {
        push(
            &mut self.result,
            std::mem::take(&mut self.current),
            self.capacity,
            name,
        )?;
        self.bytes = 0;
        Ok(())
    }

    fn finish(mut self, name: &str) -> Result<Vec<Envelope>> {
        if self.current.rows > 0 {
            push(&mut self.result, self.current, self.capacity, name)?;
        }
        Ok(self.result)
    }
}

fn validated_bytes(record: &RecordBatch, budget: EdgeBudget, name: &str) -> Result<usize> {
    let bytes = measure(record, 0, record.num_rows())?;
    if record.num_rows() > budget.max_rows || bytes > budget.max_bytes {
        return Err(operator_error(
            name,
            "validated Join chunk exceeded its row or byte budget",
        ));
    }
    Ok(bytes)
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
    fitting_prefix(record, offset, limit, budget.max_bytes)
}

fn fitting_prefix(
    record: &RecordBatch,
    offset: usize,
    limit: usize,
    max_bytes: usize,
) -> Result<(usize, usize)> {
    let mut low = 0;
    let mut high = limit - 1;
    let mut charge = 0;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let bytes = measure(record, offset, middle)?;
        if bytes <= max_bytes {
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
