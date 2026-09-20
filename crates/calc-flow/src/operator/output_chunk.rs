//! Shared edge-budget chunking for stateful operator output records.
//!
//! Join, window, rolling, and cross-section outputs leave the operator as
//! messages that must individually fit the edge budget. This module is the
//! single implementation every built-in operator uses: the sequence range is
//! validated before any batch is built, a record that fits one message skips
//! the per-row scan, per-row charges come from the allocation-free
//! [`RowCosts`] measurement for flat columns, and dictionary or list columns
//! charge their unsliced child data once per chunk range instead of once per
//! row (Arrow `ArrayData::slice` never slices non-struct children).
//!
//! Range charges never under-report [`Batch::estimated_bytes`] for the
//! emitted chunk, so the per-chunk recheck in `build_output_batches` is an
//! internal invariant rather than a budget decision.

use std::collections::BTreeMap;
use std::mem::size_of;

use datafusion::arrow::{
    array::{Array, ArrayData, StructArray},
    buffer::NullBuffer,
    datatypes::{DataType, UnionMode},
    record_batch::RecordBatch,
};

use super::row_cost::{ColumnCharges, RowCosts};
use crate::{Batch, BatchMetadata, CalcFlowError, EdgeBudget, Result, batch::checked_accumulate};

/// Per-operator wording for chunking failure messages.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OutputChunkErrors {
    /// Operator family label inserted into message templates.
    family: &'static str,
    /// Whether the over-budget row error quotes measured and budget bytes.
    quote_bytes: bool,
}

impl OutputChunkErrors {
    /// Stream Join output chunking failures.
    pub(crate) const STREAM_JOIN: Self = Self::new("stream Join", false);
    /// Window aggregate output chunking failures.
    pub(crate) const WINDOW: Self = Self::new("window", true);
    /// Rolling output chunking failures.
    pub(crate) const ROLLING: Self = Self::new("rolling", true);
    /// Cross-section output chunking failures.
    pub(crate) const CROSS_SECTION: Self = Self::new("cross-section", true);

    const fn new(family: &'static str, quote_bytes: bool) -> Self {
        Self {
            family,
            quote_bytes,
        }
    }

    /// The over-budget single-row failure shared by every operator.
    fn over_budget_row(&self, bytes: usize, max_bytes: usize) -> CalcFlowError {
        let message = if self.quote_bytes {
            format!(
                "one {} output row requires {bytes} bytes, exceeding the effective edge byte budget {max_bytes}",
                self.family
            )
        } else {
            format!(
                "one {} output row exceeds the effective edge byte budget",
                self.family
            )
        };
        CalcFlowError::InvalidArgument {
            field: "message.bytes".into(),
            message,
        }
    }
}

/// Splits one operator output record into edge-budget-sized messages.
///
/// Every chunk carries its own sequence and its own exact budget recheck, so
/// the whole record is validated before the first message leaves the
/// operator; a row that alone exceeds the byte budget fails loudly.
pub(crate) fn chunk_output_record(
    record: &RecordBatch,
    operator_id: &str,
    first_sequence: u64,
    budget: EdgeBudget,
    errors: OutputChunkErrors,
) -> Result<Vec<Batch>> {
    if record.num_rows() == 0 {
        return Ok(Vec::new());
    }
    // The emission must fit the remaining sequence range before any batch is
    // built, so a restored u64::MAX sequence fails instead of emitting.
    validate_output_sequence_range(operator_id, first_sequence, 1)?;
    // A record that already fits one message skips the per-row charge scan.
    if let Some(batch) = whole_record_batch(record, operator_id, first_sequence, budget)? {
        return Ok(vec![batch]);
    }
    chunked_record_batches(record, operator_id, first_sequence, budget, errors)
}

/// The whole-record fast path, or `None` when the record needs the per-row
/// charge scan to split into edge-budget-sized chunks.
fn whole_record_batch(
    record: &RecordBatch,
    operator_id: &str,
    first_sequence: u64,
    budget: EdgeBudget,
) -> Result<Option<Batch>> {
    if record.num_rows() > budget.max_rows {
        return Ok(None);
    }
    let metadata = BatchMetadata::new(operator_id, first_sequence, BTreeMap::new())?;
    let batch = Batch::table(vec![record.clone()], metadata)?;
    if batch.estimated_bytes()? > budget.max_bytes {
        return Ok(None);
    }
    Ok(Some(batch))
}

/// The per-row charge path for a record that does not fit one message.
fn chunked_record_batches(
    record: &RecordBatch,
    operator_id: &str,
    first_sequence: u64,
    budget: EdgeBudget,
    errors: OutputChunkErrors,
) -> Result<Vec<Batch>> {
    let plan = ChargePlan::read(record)?;
    let ranges = plan.chunk_ranges(record.num_rows(), budget, operator_id, errors)?;
    validate_output_sequence_range(operator_id, first_sequence, ranges.len())?;
    build_output_batches(record, operator_id, first_sequence, budget, ranges, errors)
}

/// Charging scheme for one output record.
///
/// `rows` is the per-row own-buffer charge (allocation-free for flat
/// columns, fixed per-row key/offset/view bytes for nested columns).
/// `nulls` tracks every null-bearing column so a chunk range can top the
/// per-null-cell row charges up to the sliced validity bitmap the envelope
/// estimator bills. `chunk` is the fixed per-chunk charge for child subtrees
/// (dictionaries, list values) that Arrow slicing keeps at full length.
struct ChargePlan<'a> {
    rows: RowCosts,
    nulls: Vec<&'a NullBuffer>,
    chunk: usize,
}

impl<'a> ChargePlan<'a> {
    fn read(record: &'a RecordBatch) -> Result<Self> {
        if let Some(rows) = RowCosts::try_new(record)? {
            let nulls = record
                .columns()
                .iter()
                .filter_map(|column| column.nulls())
                .collect();
            return Ok(Self {
                rows,
                nulls,
                chunk: 0,
            });
        }
        let mut charges = ColumnCharges::default();
        let mut chunk = 0_usize;
        for column in record.columns() {
            accumulate_column(column.as_ref(), &mut charges, &mut chunk)?;
        }
        let rows = charges.materialize(record.num_rows())?;
        Ok(Self {
            rows,
            nulls: charges.validity,
            chunk,
        })
    }

    /// The charge of one row emitted alone, including per-chunk constants.
    fn single_row_charge(&self, row: usize) -> Result<usize> {
        checked_accumulate(self.rows.get(row), self.chunk, "batch")
    }

    fn chunk_ranges(
        &self,
        row_count: usize,
        budget: EdgeBudget,
        operator_id: &str,
        errors: OutputChunkErrors,
    ) -> Result<Vec<(usize, usize)>> {
        // A row that cannot fit a message on its own fails loudly before any
        // chunk range is formed.
        for row in 0..row_count {
            let charge = self.single_row_charge(row)?;
            if charge > budget.max_bytes {
                return Err(errors.over_budget_row(charge, budget.max_bytes));
            }
        }
        let mut ranges = Vec::<(usize, usize)>::new();
        let mut null_counts = vec![0_usize; self.nulls.len()];
        let mut start = 0;
        while start < row_count {
            let end = self.scan_chunk_end(row_count, budget, start, &mut null_counts)?;
            if end == start {
                return Err(operator_error(
                    operator_id,
                    &format!(
                        "validated {} output row did not fit the effective edge budget",
                        errors.family
                    ),
                ));
            }
            ranges.push((start, end));
            start = end;
        }
        Ok(ranges)
    }

    /// Extends one chunk from `start` until the next row would exceed the
    /// row or byte budget, charging unsliced child data once per chunk.
    fn scan_chunk_end(
        &self,
        row_count: usize,
        budget: EdgeBudget,
        start: usize,
        null_counts: &mut [usize],
    ) -> Result<usize> {
        null_counts.fill(0);
        let mut end = start;
        let mut row_bytes = self.chunk;
        while end < row_count && end - start < budget.max_rows {
            let adjustment = self.validity_adjustment(end - start + 1, end, null_counts)?;
            let Some(next_bytes) = row_bytes.checked_add(self.rows.get(end)) else {
                break;
            };
            let Some(candidate) = next_bytes.checked_add(adjustment) else {
                break;
            };
            if candidate > budget.max_bytes {
                break;
            }
            for (count, nulls) in null_counts.iter_mut().zip(self.nulls.iter()) {
                *count += usize::from(nulls.is_null(end));
            }
            row_bytes = next_bytes;
            end += 1;
        }
        Ok(end)
    }

    /// Tops the per-null-cell row charges up to the sliced validity bitmap
    /// the envelope estimator bills for the candidate range ending at `end`.
    fn validity_adjustment(
        &self,
        candidate_len: usize,
        end: usize,
        null_counts: &[usize],
    ) -> Result<usize> {
        let mut adjustment = 0_usize;
        for (count, nulls) in null_counts.iter().zip(self.nulls.iter()) {
            let in_range = count + usize::from(nulls.is_null(end));
            let bitmap = if in_range == 0 {
                0
            } else {
                candidate_len.div_ceil(8)
            };
            adjustment = checked_accumulate(adjustment, bitmap.saturating_sub(in_range), "batch")?;
        }
        Ok(adjustment)
    }
}

/// Charges one column into the per-row and per-chunk buckets.
///
/// Flat columns price per row without allocation; struct columns recurse
/// because Arrow slices their children; every other nested column charges
/// its own per-row buffers and its unsliced child data once per chunk.
fn accumulate_column<'a>(
    column: &'a dyn Array,
    charges: &mut ColumnCharges<'a>,
    chunk: &mut usize,
) -> Result<()> {
    if matches!(column.data_type(), DataType::Struct(_)) {
        return accumulate_struct_column(column, charges, chunk);
    }
    if charges.add_flat_column(column)? {
        return Ok(());
    }
    accumulate_nested_column(column, charges, chunk)
}

/// Struct columns recurse because Arrow slices their children; the struct's
/// own validity bitmap still joins the per-null-cell charges.
fn accumulate_struct_column<'a>(
    column: &'a dyn Array,
    charges: &mut ColumnCharges<'a>,
    chunk: &mut usize,
) -> Result<()> {
    charges.validity.extend(column.nulls());
    let struct_array = column
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("struct column matches its data type");
    for child in struct_array.columns() {
        accumulate_column(child, charges, chunk)?;
    }
    Ok(())
}

/// Nested columns charge their own per-row buffers and keep each unsliced
/// child subtree as one fixed charge per chunk range.
fn accumulate_nested_column<'a>(
    column: &'a dyn Array,
    charges: &mut ColumnCharges<'a>,
    chunk: &mut usize,
) -> Result<()> {
    charges.fixed = checked_accumulate(
        charges.fixed,
        nested_row_width(column.data_type())?,
        "batch",
    )?;
    charges.validity.extend(column.nulls());
    for child in column.to_data().child_data() {
        *chunk = checked_accumulate(*chunk, child_slice_memory(child)?, "batch")?;
    }
    Ok(())
}

/// The bytes Arrow keeps at full length for one unsliced child subtree.
fn child_slice_memory(child: &ArrayData) -> Result<usize> {
    child
        .get_slice_memory_size()
        .map_err(|error| CalcFlowError::InvalidArgument {
            field: "batch".into(),
            message: format!("Arrow slice memory could not be measured: {error}"),
        })
}

/// Per-row charge of a nested column's own buffers: dictionary keys, list
/// offsets, view pointers, or union discriminants. Child subtrees are
/// charged once per chunk range, not per row.
fn nested_row_width(data_type: &DataType) -> Result<usize> {
    Ok(match data_type {
        DataType::Dictionary(key, _) => key
            .primitive_width()
            .expect("Arrow dictionary keys are validated integers"),
        DataType::List(_) | DataType::Map(_, _) => size_of::<i32>(),
        DataType::LargeList(_) => size_of::<i64>(),
        DataType::ListView(_) => 2 * size_of::<i32>(),
        DataType::LargeListView(_) => 2 * size_of::<i64>(),
        DataType::FixedSizeList(..) | DataType::RunEndEncoded(..) => 0,
        DataType::Union(_, UnionMode::Sparse) => size_of::<i8>(),
        DataType::Union(_, UnionMode::Dense) => size_of::<i8>() + size_of::<i32>(),
        DataType::Utf8View | DataType::BinaryView => size_of::<u128>(),
        DataType::FixedSizeBinary(width) => {
            usize::try_from(*width).expect("Arrow fixed-size binary width is validated")
        }
        other => {
            return Err(CalcFlowError::Internal {
                message: format!("output charging does not support {other}"),
            });
        }
    })
}

fn validate_output_sequence_range(
    operator_id: &str,
    first_sequence: u64,
    range_count: usize,
) -> Result<()> {
    let chunk_count = u64::try_from(range_count).map_err(|_| {
        operator_error(
            operator_id,
            "output chunk count does not fit the sequence range",
        )
    })?;
    if first_sequence.checked_add(chunk_count).is_none() {
        return Err(operator_error(
            operator_id,
            "output sequence overflowed before emission",
        ));
    }
    Ok(())
}

fn build_output_batches(
    record: &RecordBatch,
    operator_id: &str,
    first_sequence: u64,
    budget: EdgeBudget,
    ranges: Vec<(usize, usize)>,
    errors: OutputChunkErrors,
) -> Result<Vec<Batch>> {
    ranges
        .into_iter()
        .enumerate()
        .map(|(ordinal, (start, end))| {
            let ordinal = u64::try_from(ordinal).map_err(|_| {
                operator_error(operator_id, "output chunk ordinal does not fit u64")
            })?;
            let sequence = first_sequence
                .checked_add(ordinal)
                .expect("complete sequence range validated above");
            let metadata = BatchMetadata::new(operator_id, sequence, BTreeMap::new())?;
            let batch = Batch::table(vec![record.slice(start, end - start)], metadata)?;
            if batch.estimated_bytes()? > budget.max_bytes {
                return Err(operator_error(
                    operator_id,
                    &format!(
                        "conservative row charges underreported a {} output chunk",
                        errors.family
                    ),
                ));
            }
            Ok(batch)
        })
        .collect()
}

fn operator_error(operator_id: &str, message: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: operator_id.into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::{
        array::{
            ArrayRef, DictionaryArray, Int32Array, Int64Array, ListArray, StringArray, StructArray,
        },
        datatypes::{Field, Int32Type},
    };

    use super::*;

    fn estimated(record: &RecordBatch) -> usize {
        Batch::table(vec![record.clone()], BatchMetadata::default())
            .unwrap()
            .estimated_bytes()
            .unwrap()
    }

    fn dictionary_record(rows: usize) -> RecordBatch {
        let values = Arc::new(StringArray::from(
            (0..256)
                .map(|index| format!("{index:0256}"))
                .collect::<Vec<_>>(),
        ));
        let keys = Int32Array::from_iter_values(
            (0..rows).map(|row| i32::try_from(row).expect("test row index fits i32") % 256),
        );
        let dictionary = DictionaryArray::<Int32Type>::new(keys, values);
        RecordBatch::try_from_iter(vec![("payload", Arc::new(dictionary) as ArrayRef)]).unwrap()
    }

    #[test]
    fn dictionary_columns_charge_their_dictionary_once_per_chunk() {
        // Regression for the per-row dictionary over-charge: charging the
        // unified dictionary once per row forced one row per message (10_000
        // chunks here); charging it once per chunk keeps chunking at the
        // byte budget.
        let record = dictionary_record(10_000);
        let budget = EdgeBudget::new(10_000, 80_000).unwrap();
        let chunks =
            chunk_output_record(&record, "join", 0, budget, OutputChunkErrors::STREAM_JOIN)
                .unwrap();

        assert_eq!(
            chunks.iter().map(Batch::num_rows).sum::<usize>(),
            record.num_rows()
        );
        assert!(
            chunks.len() <= 6,
            "10_000 dictionary rows must chunk at the byte budget, not one row per message: got {} chunks",
            chunks.len()
        );
        for (sequence, chunk) in chunks.iter().enumerate() {
            assert!(chunk.estimated_bytes().unwrap() <= budget.max_bytes);
            assert_eq!(
                chunk.metadata().sequence(),
                u64::try_from(sequence).unwrap()
            );
        }
    }

    #[test]
    fn list_columns_charge_their_values_once_per_chunk() {
        let values = Int64Array::from_iter_values(0..4_096_i64);
        let offsets = (0..=2_048_i32).map(|index| index * 2).collect::<Vec<_>>();
        let lists = ListArray::new(
            Arc::new(Field::new("item", DataType::Int64, false)),
            datafusion::arrow::buffer::OffsetBuffer::new(offsets.into()),
            Arc::new(values),
            None,
        );
        let record =
            RecordBatch::try_from_iter(vec![("payload", Arc::new(lists) as ArrayRef)]).unwrap();
        let whole = estimated(&record);
        // The value buffer dominates; each emitted chunk carries it once.
        // With the per-row over-charge every row cost the whole value buffer
        // and this record degenerated to one row per message (2_048 chunks).
        let budget = EdgeBudget::new(2_048, whole - 4).unwrap();
        let chunks =
            chunk_output_record(&record, "window", 0, budget, OutputChunkErrors::WINDOW).unwrap();

        assert_eq!(
            chunks.iter().map(Batch::num_rows).sum::<usize>(),
            record.num_rows()
        );
        assert!(
            chunks.len() <= 4,
            "list values must be charged once per chunk, not per row: {} chunks",
            chunks.len()
        );
        for chunk in &chunks {
            assert!(chunk.estimated_bytes().unwrap() <= budget.max_bytes);
        }
    }

    #[test]
    fn chunk_charges_never_under_report_the_envelope_estimator() {
        let record = RecordBatch::try_from_iter(vec![
            (
                "flat",
                Arc::new(Int64Array::from(vec![Some(1), None, Some(3), Some(4)])) as ArrayRef,
            ),
            (
                "text",
                Arc::new(StringArray::from(vec![
                    Some("wide"),
                    None,
                    Some("中文"),
                    Some(""),
                ])) as ArrayRef,
            ),
            (
                "payload",
                Arc::new(DictionaryArray::<Int32Type>::new(
                    Int32Array::from(vec![Some(0), None, Some(1), Some(0)]),
                    Arc::new(StringArray::from(vec!["paid", "other"])),
                )) as ArrayRef,
            ),
            (
                "nested",
                Arc::new(StructArray::from(vec![(
                    Arc::new(Field::new("inner", DataType::Int64, true)),
                    Arc::new(Int64Array::from(vec![Some(1), None, Some(3), Some(4)])) as ArrayRef,
                )])) as ArrayRef,
            ),
        ])
        .unwrap();
        let plan = ChargePlan::read(&record).unwrap();
        // Every contiguous range must stay at or above the estimator charge
        // for the same slice.
        for start in 0..record.num_rows() {
            for end in start + 1..=record.num_rows() {
                let len = end - start;
                let mut charged = (start..end)
                    .try_fold(plan.chunk, |total, row| {
                        checked_accumulate(total, plan.rows.get(row), "batch")
                    })
                    .unwrap();
                for nulls in &plan.nulls {
                    let in_range = (start..end).filter(|&row| nulls.is_null(row)).count();
                    if in_range > 0 {
                        charged += len.div_ceil(8).saturating_sub(in_range);
                    }
                }
                let slice = record.slice(start, len);
                assert!(
                    charged >= estimated(&slice),
                    "range {start}..{end}: charged {charged} below estimator {}",
                    estimated(&slice)
                );
            }
        }
    }

    #[test]
    fn empty_record_emits_nothing_without_touching_the_sequence() {
        let record = dictionary_record(10).slice(0, 0);
        let chunks = chunk_output_record(
            &record,
            "rolling",
            u64::MAX,
            EdgeBudget::default(),
            OutputChunkErrors::ROLLING,
        )
        .unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn non_empty_record_validates_the_sequence_before_building() {
        let record = dictionary_record(4);
        let error = chunk_output_record(
            &record,
            "join",
            u64::MAX,
            EdgeBudget::default(),
            OutputChunkErrors::STREAM_JOIN,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("output sequence overflowed before emission"),
            "{error}"
        );
    }

    #[test]
    fn over_budget_row_messages_keep_per_operator_wording() {
        let join = OutputChunkErrors::STREAM_JOIN.over_budget_row(9, 4);
        assert_eq!(
            join.to_string(),
            "invalid message.bytes: one stream Join output row exceeds the effective edge byte budget"
        );
        for (errors, family) in [
            (OutputChunkErrors::WINDOW, "window"),
            (OutputChunkErrors::ROLLING, "rolling"),
            (OutputChunkErrors::CROSS_SECTION, "cross-section"),
        ] {
            let error = errors.over_budget_row(9, 4);
            assert!(
                matches!(error, CalcFlowError::InvalidArgument { ref field, .. } if field == "message.bytes"),
                "{error:?}"
            );
            assert_eq!(
                error.to_string(),
                format!(
                    "invalid message.bytes: one {family} output row requires 9 bytes, exceeding the effective edge byte budget 4"
                )
            );
        }
    }

    #[test]
    fn whole_record_fast_path_charges_dictionary_columns_exactly_once() {
        let record = dictionary_record(64);
        let whole = estimated(&record);
        let chunks = chunk_output_record(
            &record,
            "rolling",
            3,
            EdgeBudget::new(64, whole).unwrap(),
            OutputChunkErrors::ROLLING,
        )
        .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].metadata().sequence(), 3);
    }
}
