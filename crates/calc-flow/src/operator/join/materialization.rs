//! Preflight matched-pair ranges before allocating independently owned output chunks.

use std::{
    collections::{BTreeMap, btree_map::Entry},
    ops::Range,
};

use datafusion::arrow::{datatypes::SchemaRef, record_batch::RecordBatch};

use super::{AdmittedRow, MatchedPair, StoredRow, materialize_output_record};
use crate::batch::checked_accumulate;
use crate::operator::output_chunk::OutputChunkErrors;
use crate::operator::row_cost::{fixed_width, variable_offsets};
use crate::{Batch, BatchMetadata, CalcFlowError, EdgeBudget, Result};

pub(super) struct JoinOutput<'a> {
    pub(super) schema: &'a SchemaRef,
    pub(super) admitted: &'a [AdmittedRow],
    pub(super) opposite: &'a [StoredRow],
    pub(super) matched: &'a [MatchedPair],
    pub(super) incoming_is_left: bool,
    pub(super) operator_id: &'a str,
    /// Conservative per-admitted-row logical byte charges aligned with
    /// `admitted`. `None` when the schema keeps the generic measured path.
    pub(super) admitted_charges: Option<&'a [u64]>,
}

struct FlatRow {
    bytes: usize,
    nulls: Vec<usize>,
}

#[derive(Default)]
struct FlatChunk {
    rows: usize,
    bytes: usize,
    nulls: Vec<bool>,
    null_count: usize,
}

impl FlatChunk {
    fn fits(&self, incoming: &FlatRow, opposite: &FlatRow, budget: EdgeBudget) -> bool {
        self.rows < budget.max_rows
            && self
                .bytes_with(incoming, opposite)
                .is_some_and(|bytes| bytes <= budget.max_bytes)
    }

    fn bytes_with(&self, incoming: &FlatRow, opposite: &FlatRow) -> Option<usize> {
        let bitmap_count = self.null_count
            + incoming
                .nulls
                .iter()
                .chain(&opposite.nulls)
                .filter(|index| !self.nulls.get(**index).copied().unwrap_or(false))
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
        for &index in incoming.nulls.iter().chain(&opposite.nulls) {
            if index >= self.nulls.len() {
                self.nulls.resize(index + 1, false);
            }
            if !self.nulls[index] {
                self.nulls[index] = true;
                self.null_count += 1;
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
        if let Some(charges) = self.admitted_charges
            && self.single_chunk_fits(charges, budget)
        {
            let mut single = Vec::with_capacity(1);
            single.push(0..self.matched.len());
            return Ok(single);
        }
        if self
            .schema
            .fields()
            .iter()
            .all(|field| fixed_width(field.data_type()).is_some())
        {
            self.flat_ranges(budget)
        } else {
            self.nested_ranges(budget)
        }
    }

    /// Conservative whole-output fit check using cached logical charges.
    ///
    /// Each per-row charge includes the state-row overhead and a null-bitmap
    /// allowance, so the summed bound never under-estimates the detailed flat
    /// model. Only the row cap and byte sum decide; no per-row schema work.
    fn single_chunk_fits(&self, charges: &[u64], budget: EdgeBudget) -> bool {
        if self.matched.is_empty() || self.matched.len() > budget.max_rows {
            return false;
        }
        let limit = budget.max_bytes as u128;
        let mut total = 0_u128;
        for pair in self.matched {
            let incoming = charges[pair.pos];
            let opposite = self.opposite[pair.opposite_index].charge;
            total += u128::from(incoming) + u128::from(opposite);
            if total > limit {
                return false;
            }
        }
        true
    }

    fn flat_ranges(&self, budget: EdgeBudget) -> Result<Vec<Range<usize>>> {
        // Matched positions are dense indices into both sides, so per-row
        // costs cache in plain vectors instead of hashed maps.
        let mut incoming: Vec<Option<FlatRow>> = (0..self.admitted.len()).map(|_| None).collect();
        let mut opposite: Vec<Option<FlatRow>> = (0..self.opposite.len()).map(|_| None).collect();
        let mut ranges = Vec::new();
        let mut start = 0;
        let mut chunk = FlatChunk::default();
        for (index, pair) in self.matched.iter().enumerate() {
            let record = &self.admitted[pair.pos].record;
            let left = flat_row(&mut incoming, pair.pos, record, 0)?;
            let right = flat_row(
                &mut opposite,
                pair.opposite_index,
                &self.opposite[pair.opposite_index].record,
                record.num_columns(),
            )?;
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

fn flat_row<'a>(
    rows: &'a mut [Option<FlatRow>],
    index: usize,
    record: &RecordBatch,
    column_offset: usize,
) -> Result<&'a FlatRow> {
    if rows[index].is_none() {
        #[cfg(test)]
        tests::FLAT_VISITS.with(|count| count.set(count.get() + 1));
        let mut bytes = 0;
        let mut nulls = Vec::new();
        for (column_index, column) in record.columns().iter().enumerate() {
            let width = fixed_width(column.data_type()).expect("flat schema was checked");
            bytes = checked_accumulate(bytes, width, "batch")?;
            if let Some(offsets) = variable_offsets(column.as_ref()) {
                bytes = checked_accumulate(bytes, offsets.total_width(), "batch")?;
            }
            if column.nulls().is_some_and(|nulls| nulls.is_null(0)) {
                nulls.push(column_offset + column_index);
            }
        }
        rows[index] = Some(FlatRow { bytes, nulls });
    }
    Ok(rows[index].as_ref().expect("just populated"))
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

/// Conservative logical byte charges for every row of one incoming record.
///
/// Returns `None` when any column lacks the flat width model, so the charge
/// fast path never runs for nested schemas. Each row carries the state-row
/// overhead plus a whole-schema null-bitmap allowance, keeping the sum an
/// upper bound of the detailed flat model for single-chunk decisions.
pub(super) fn flat_row_charges(record: &RecordBatch, overhead: u64) -> Result<Option<Vec<u64>>> {
    let bitmap_per_row = record.num_columns().div_ceil(8);
    let overhead = usize::try_from(overhead).map_err(|_| overflow_charge())?;
    let mut charges = vec![overhead + bitmap_per_row; record.num_rows()];
    for column in record.columns() {
        let Some(width) = fixed_width(column.data_type()) else {
            return Ok(None);
        };
        for charge in &mut charges {
            *charge = checked_accumulate(*charge, width, "batch")?;
        }
        if let Some(offsets) = variable_offsets(column.as_ref()) {
            offsets.add_charges(&mut charges)?;
        }
    }
    let charges = charges
        .into_iter()
        .map(|charge| u64::try_from(charge).map_err(|_| overflow_charge()))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(charges))
}

fn overflow_charge() -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "batch".into(),
        message: "row charge overflowed".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EventTime;
    use datafusion::arrow::{
        array::{ArrayRef, Int64Array, ListArray, StringArray},
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
            admitted_charges: None,
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

    #[test]
    fn test_flat_preflight_avoids_per_row_scratch_allocations() {
        let record = RecordBatch::try_from_iter(vec![
            (
                "key",
                Arc::new(StringArray::from(vec!["K0000000"])) as ArrayRef,
            ),
            ("value", Arc::new(Int64Array::from(vec![7])) as ArrayRef),
        ])
        .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("left_key", DataType::Utf8, false),
            Field::new("left_value", DataType::Int64, false),
            Field::new("right_key", DataType::Utf8, false),
            Field::new("right_value", DataType::Int64, false),
        ]));
        let admitted = (0..1_000)
            .map(|_| AdmittedRow {
                record: record.clone(),
                event_time: EventTime::from_micros(100),
                row_id: 0,
                retain: true,
            })
            .collect::<Vec<_>>();
        let opposite = vec![
            StoredRow {
                record,
                event_time: EventTime::from_micros(100),
                row_id: 0,
                charge: 0,
                encoded_key: Arc::new(vec![]),
            };
            1_000
        ];
        for fanout in [1, 10] {
            let matched = (0..1_000)
                .flat_map(|pos| {
                    (0..fanout).map(move |offset| MatchedPair {
                        pos,
                        opposite_index: (pos + offset) % 1_000,
                    })
                })
                .collect::<Vec<_>>();
            let output = JoinOutput {
                schema: &schema,
                admitted: &admitted,
                opposite: &opposite,
                matched: &matched,
                incoming_is_left: true,
                operator_id: "allocation",
                admitted_charges: None,
            };
            let measured = allocation_counter::measure(|| {
                let ranges = output
                    .ranges(EdgeBudget::new(1_000, 40_000).unwrap())
                    .unwrap();
                assert_eq!(ranges.len(), fanout);
                assert_eq!(ranges.iter().map(Range::len).sum::<usize>(), matched.len());
            });
            assert!(
                measured.count_total < 1_000,
                "fanout {fanout}: preflight allocated {} times; flat row scratch must be reused",
                measured.count_total
            );
        }
    }

    #[test]
    fn test_single_chunk_output_takes_the_charge_fast_path() {
        let column: ArrayRef = Arc::new(Int64Array::from(vec![7_i64; 4]));
        let field = Field::new("value", DataType::Int64, false);
        let record =
            RecordBatch::try_new(Arc::new(Schema::new(vec![field.clone()])), vec![column]).unwrap();
        let schema = Arc::new(Schema::new(vec![
            field.clone().with_name("left_value"),
            field.with_name("right_value"),
        ]));
        let admitted: Vec<AdmittedRow> = (0..4)
            .map(|index| AdmittedRow {
                record: record.slice(index, 1),
                event_time: EventTime::from_micros(100),
                row_id: index as u64,
                retain: true,
            })
            .collect();
        let opposite: Vec<StoredRow> = (0..4)
            .map(|index| StoredRow {
                record: record.slice(index, 1),
                event_time: EventTime::from_micros(100),
                row_id: index as u64,
                charge: 81,
                encoded_key: Arc::new(vec![]),
            })
            .collect();
        let matched: Vec<MatchedPair> = (0..4)
            .map(|index| MatchedPair {
                pos: index,
                opposite_index: index,
            })
            .collect();
        let charges = vec![81_u64; 4];
        let output = JoinOutput {
            schema: &schema,
            admitted: &admitted,
            opposite: &opposite,
            matched: &matched,
            incoming_is_left: true,
            operator_id: "fast",
            admitted_charges: Some(&charges),
        };
        FLAT_VISITS.with(|count| count.set(0));
        let ranges = output.ranges(EdgeBudget::new(4, 1 << 20).unwrap()).unwrap();
        assert_eq!(ranges, vec![0..4]);
        assert_eq!(
            FLAT_VISITS.with(Cell::get),
            0,
            "a provably single-chunk output must not visit matched rows"
        );
    }

    #[test]
    fn test_charge_fast_path_falls_back_when_the_bound_exceeds_the_budget() {
        let column: ArrayRef = Arc::new(Int64Array::from(vec![7_i64; 4]));
        let field = Field::new("value", DataType::Int64, false);
        let record =
            RecordBatch::try_new(Arc::new(Schema::new(vec![field.clone()])), vec![column]).unwrap();
        let schema = Arc::new(Schema::new(vec![
            field.clone().with_name("left_value"),
            field.with_name("right_value"),
        ]));
        let admitted: Vec<AdmittedRow> = (0..4)
            .map(|index| AdmittedRow {
                record: record.slice(index, 1),
                event_time: EventTime::from_micros(100),
                row_id: index as u64,
                retain: true,
            })
            .collect();
        let opposite: Vec<StoredRow> = (0..4)
            .map(|index| StoredRow {
                record: record.slice(index, 1),
                event_time: EventTime::from_micros(100),
                row_id: index as u64,
                charge: 81,
                encoded_key: Arc::new(vec![]),
            })
            .collect();
        let matched: Vec<MatchedPair> = (0..4)
            .map(|index| MatchedPair {
                pos: index,
                opposite_index: index,
            })
            .collect();
        let charges = vec![81_u64; 4];
        let output = JoinOutput {
            schema: &schema,
            admitted: &admitted,
            opposite: &opposite,
            matched: &matched,
            incoming_is_left: true,
            operator_id: "fallback",
            admitted_charges: Some(&charges),
        };
        FLAT_VISITS.with(|count| count.set(0));
        // 4 pairs x (81 + 81) = 648 charge bytes exceeds 200, but exact flat
        // widths fit, so planning falls back to the detailed flat scan.
        let ranges = output.ranges(EdgeBudget::new(2, 200).unwrap()).unwrap();
        assert_eq!(ranges, vec![0..2, 2..4]);
        assert!(FLAT_VISITS.with(Cell::get) > 0);
    }
}
