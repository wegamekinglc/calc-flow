//! Preflight matched-pair ranges before allocating independently owned output chunks.

use std::{
    collections::{BTreeMap, btree_map::Entry},
    ops::Range,
};

use datafusion::arrow::{datatypes::SchemaRef, record_batch::RecordBatch};

use super::{AdmittedRow, MatchedPair, StoredRow, materialize_output_record};
use crate::operator::output_chunk::OutputChunkErrors;
use crate::operator::row_cost::RowCosts;
use crate::{Batch, BatchMetadata, CalcFlowError, EdgeBudget, Result};

pub(super) struct JoinOutput<'a> {
    pub(super) schema: &'a SchemaRef,
    pub(super) admitted: &'a [AdmittedRow],
    pub(super) opposite: &'a [StoredRow],
    pub(super) matched: &'a [MatchedPair],
    pub(super) incoming_is_left: bool,
    pub(super) operator_id: &'a str,
}

struct FlatRow {
    bytes: usize,
    nulls: Vec<bool>,
}

#[derive(Default)]
struct FlatChunk {
    rows: usize,
    bytes: usize,
    nulls: Vec<bool>,
}

impl FlatChunk {
    fn fits(&self, incoming: &FlatRow, opposite: &FlatRow, budget: EdgeBudget) -> bool {
        self.rows < budget.max_rows
            && self
                .bytes_with(incoming, opposite)
                .is_some_and(|bytes| bytes <= budget.max_bytes)
    }

    fn bytes_with(&self, incoming: &FlatRow, opposite: &FlatRow) -> Option<usize> {
        let null_columns = incoming.nulls.iter().chain(&opposite.nulls);
        let bitmap_count = null_columns
            .enumerate()
            .filter(|(index, null)| **null || self.nulls.get(*index).copied().unwrap_or(false))
            .count();
        incoming
            .bytes
            .checked_add(opposite.bytes)
            .and_then(|bytes| self.bytes.checked_add(bytes))
            .and_then(|bytes| {
                bitmap_count
                    .checked_mul((self.rows + 1).div_ceil(8))
                    .and_then(|bitmap| bytes.checked_add(bitmap))
            })
    }

    fn push(&mut self, incoming: &FlatRow, opposite: &FlatRow) {
        self.rows += 1;
        self.bytes += incoming.bytes + opposite.bytes;
        let nulls = incoming.nulls.iter().chain(&opposite.nulls);
        if self.nulls.is_empty() {
            self.nulls.extend(nulls.copied());
        } else {
            for (current, added) in self.nulls.iter_mut().zip(nulls) {
                *current |= added;
            }
        }
    }
}

impl JoinOutput<'_> {
    pub(super) fn materialize(&self, range: Range<usize>) -> Result<RecordBatch> {
        materialize_output_record(
            self.schema,
            self.admitted,
            self.opposite,
            &self.matched[range],
            self.incoming_is_left,
            self.operator_id,
        )
    }

    pub(super) fn ranges(&self, budget: EdgeBudget) -> Result<Vec<Range<usize>>> {
        let incoming = flat_rows(
            self.matched
                .iter()
                .map(|pair| (pair.pos, &self.admitted[pair.pos].record)),
        )?;
        let opposite = flat_rows(self.matched.iter().map(|pair| {
            (
                pair.opposite_index,
                &self.opposite[pair.opposite_index].record,
            )
        }))?;
        if let (Some(incoming), Some(opposite)) = (incoming, opposite) {
            return self.flat_ranges(&incoming, &opposite, budget);
        }
        self.nested_ranges(budget)
    }

    fn flat_ranges(
        &self,
        incoming: &BTreeMap<usize, FlatRow>,
        opposite: &BTreeMap<usize, FlatRow>,
        budget: EdgeBudget,
    ) -> Result<Vec<Range<usize>>> {
        let mut ranges = Vec::new();
        let mut start = 0;
        let mut chunk = FlatChunk::default();
        for (index, pair) in self.matched.iter().enumerate() {
            let (left, right) = (&incoming[&pair.pos], &opposite[&pair.opposite_index]);
            if !chunk.fits(left, right, budget) {
                if start == index {
                    return Err(oversized_row(
                        chunk.bytes_with(left, right).unwrap_or(usize::MAX),
                        budget,
                    ));
                }
                ranges.push(start..index);
                start = index;
                chunk = FlatChunk::default();
                if !chunk.fits(left, right, budget) {
                    return Err(oversized_row(
                        chunk.bytes_with(left, right).unwrap_or(usize::MAX),
                        budget,
                    ));
                }
            }
            chunk.push(left, right);
        }
        if start < self.matched.len() {
            ranges.push(start..self.matched.len());
        }
        Ok(ranges)
    }

    fn nested_ranges(&self, budget: EdgeBudget) -> Result<Vec<Range<usize>>> {
        let incoming = estimated_rows(
            self.matched
                .iter()
                .map(|pair| (pair.pos, &self.admitted[pair.pos].record)),
        )?;
        let opposite = estimated_rows(self.matched.iter().map(|pair| {
            (
                pair.opposite_index,
                &self.opposite[pair.opposite_index].record,
            )
        }))?;
        let mut ranges = Vec::new();
        let mut start = 0;
        let mut bytes = 0_usize;
        for (index, pair) in self.matched.iter().enumerate() {
            let row_bytes = incoming[&pair.pos].saturating_add(opposite[&pair.opposite_index]);
            if nested_range_is_full(start..index, bytes, row_bytes, budget) {
                self.validate_nested_range(start..index, budget, &mut ranges)?;
                start = index;
                bytes = 0;
            }
            bytes = bytes.saturating_add(row_bytes);
        }
        if start < self.matched.len() {
            self.validate_nested_range(start..self.matched.len(), budget, &mut ranges)?;
        }
        Ok(ranges)
    }

    fn validate_nested_range(
        &self,
        range: Range<usize>,
        budget: EdgeBudget,
        ranges: &mut Vec<Range<usize>>,
    ) -> Result<()> {
        // Drop each bounded preflight allocation before validating another range.
        let bytes = estimated(&self.materialize(range.clone())?)?;
        if bytes <= budget.max_bytes {
            ranges.push(range);
        } else if range.len() == 1 {
            return Err(oversized_row(bytes, budget));
        } else {
            let mid = range.start + range.len() / 2;
            self.validate_nested_range(range.start..mid, budget, ranges)?;
            self.validate_nested_range(mid..range.end, budget, ranges)?;
        }
        Ok(())
    }
}

fn nested_range_is_full(
    range: Range<usize>,
    bytes: usize,
    row_bytes: usize,
    budget: EdgeBudget,
) -> bool {
    range.start < range.end
        && (range.end - range.start >= budget.max_rows
            || bytes.saturating_add(row_bytes) > budget.max_bytes)
}

fn flat_rows<'a>(
    records: impl Iterator<Item = (usize, &'a RecordBatch)>,
) -> Result<Option<BTreeMap<usize, FlatRow>>> {
    let mut rows = BTreeMap::new();
    for (index, record) in records {
        let Entry::Vacant(entry) = rows.entry(index) else {
            continue;
        };

        #[cfg(test)]
        tests::FLAT_VISITS.with(|count| count.set(count.get() + 1));
        let Some(bytes) = RowCosts::try_total(record)? else {
            return Ok(None);
        };
        let nulls = record
            .columns()
            .iter()
            .map(|column| column.nulls().is_some_and(|nulls| nulls.is_null(0)))
            .collect::<Vec<_>>();
        let null_count = nulls.iter().filter(|null| **null).count();
        entry.insert(FlatRow {
            bytes: bytes - null_count,
            nulls,
        });
    }
    Ok(Some(rows))
}

fn estimated_rows<'a>(
    records: impl Iterator<Item = (usize, &'a RecordBatch)>,
) -> Result<BTreeMap<usize, usize>> {
    let mut rows = BTreeMap::new();
    for (index, record) in records {
        if let Entry::Vacant(entry) = rows.entry(index) {
            entry.insert(estimated(record)?);
        }
    }
    Ok(rows)
}

fn estimated(record: &RecordBatch) -> Result<usize> {
    #[cfg(test)]
    tests::NESTED_VISITS.with(|count| count.set(count.get() + 1));
    Batch::table(vec![record.clone()], BatchMetadata::default())?.estimated_bytes()
}

fn oversized_row(bytes: usize, budget: EdgeBudget) -> CalcFlowError {
    OutputChunkErrors::STREAM_JOIN.over_budget_row(bytes, budget.max_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EventTime;
    use datafusion::arrow::{
        array::{ArrayRef, Int64Array, ListArray},
        datatypes::{DataType, Field, Int64Type, Schema},
    };
    use std::{cell::Cell, sync::Arc};

    thread_local! {
        pub(super) static FLAT_VISITS: Cell<usize> = const { Cell::new(0) };
        pub(super) static NESTED_VISITS: Cell<usize> = const { Cell::new(0) };
    }

    fn sparse_preflight(nested: bool, retained: usize) {
        let column: ArrayRef = if nested {
            Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>([Some(
                vec![Some(7_i64)],
            )]))
        } else {
            Arc::new(Int64Array::from(vec![7]))
        };
        let field = Field::new("value", column.data_type().clone(), true);
        let record =
            RecordBatch::try_new(Arc::new(Schema::new(vec![field.clone()])), vec![column]).unwrap();
        let schema = Arc::new(Schema::new(vec![field.clone(), field.with_name("other")]));
        let admitted = vec![AdmittedRow {
            record: record.clone(),
            event_time: EventTime::from_micros(100),
            row_id: 0,
            retain: true,
        }];
        let opposite = vec![
            StoredRow {
                record,
                event_time: EventTime::from_micros(100),
                row_id: 0,
                charge: 0,
                encoded_key: Arc::new(vec![]),
            };
            retained
        ];
        let matched = [
            MatchedPair {
                pos: 0,
                opposite_index: retained - 1,
            },
            MatchedPair {
                pos: 0,
                opposite_index: 0,
            },
            MatchedPair {
                pos: 0,
                opposite_index: retained - 1,
            },
        ];
        FLAT_VISITS.with(|count| count.set(0));
        NESTED_VISITS.with(|count| count.set(0));
        let output = JoinOutput {
            schema: &schema,
            admitted: &admitted,
            opposite: &opposite,
            matched: &matched,
            incoming_is_left: true,
            operator_id: "sparse",
        };
        let ranges = output.ranges(EdgeBudget::new(2, 4096).unwrap()).unwrap();
        assert_eq!(ranges, vec![0..2, 2..3]);
        for range in ranges {
            let rows = range.len();
            let batch = output.materialize(range).unwrap();
            assert_eq!(batch.num_rows(), rows);
            assert_eq!(batch.column(0).data_type() == &DataType::Int64, !nested);
        }
        assert!(
            FLAT_VISITS.with(Cell::get) <= 3,
            "flat costs must visit only distinct matched rows"
        );
        assert!(
            NESTED_VISITS.with(Cell::get) <= 5,
            "nested costs must visit only matched rows and bounded ranges"
        );
    }

    #[test]
    fn test_sparse_flat_preflight_cost_is_independent_of_retained_rows() {
        for retained in [1_000, 100_000] {
            sparse_preflight(false, retained);
        }
    }

    #[test]
    fn test_sparse_nested_preflight_cost_is_independent_of_retained_rows() {
        for retained in [1_000, 100_000] {
            sparse_preflight(true, retained);
        }
    }
}
