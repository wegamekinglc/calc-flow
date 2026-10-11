use super::super::super::{MatchedPair, materialization::JoinOutput, native_lookup::scratch_error};
use super::{ColumnResult, Columns, MemoryReservation, Result, RowPayload};
use crate::runtime::streaming::gather_work::cleanup_control_bytes;
use datafusion::arrow::datatypes::SchemaRef;

pub(super) fn input_bytes(rows: usize, schema: &SchemaRef, name: &str) -> Result<usize> {
    let per_side = 2 * size_of::<RowPayload>() + 4 * size_of::<usize>() + 256;
    let columns = schema.fields().len();
    rows.checked_mul(2 * per_side)
        .and_then(|bytes| bytes.checked_add(columns.checked_mul(rows.checked_mul(16)?)?))
        .and_then(|bytes| {
            bytes.checked_add(size_of::<Columns>() + 2 * size_of::<MemoryReservation>() + 4096)
        })
        .and_then(|bytes| bytes.checked_add(cleanup_control_bytes::<Vec<ColumnResult>>()))
        .and_then(|bytes| bytes.checked_add(name.len().checked_mul(4)?))
        .ok_or_else(|| scratch_error(name))
}

pub(super) fn output_bytes(output: &JoinOutput<'_>, matched: &[MatchedPair]) -> Result<usize> {
    let charges = output.admitted_charges.expect("flat output eligibility");
    let bytes = matched
        .iter()
        .try_fold(0_u64, |total, pair| {
            total
                .checked_add(charges[pair.pos])?
                .checked_add(output.opposite[pair.opposite_index].charge)
        })
        .and_then(|bytes| usize::try_from(bytes).ok());
    bytes
        .and_then(|bytes| bytes.checked_mul(2))
        .and_then(|bytes| bytes.checked_add(output.schema.fields().len().checked_mul(4096)?))
        .ok_or_else(|| scratch_error(output.operator_id))
}
