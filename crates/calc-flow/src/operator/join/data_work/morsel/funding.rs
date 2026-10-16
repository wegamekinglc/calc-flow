use super::{GatherStop, GatherWork, JoinOutput, Result};
use crate::operator::join::{MatchedPair, PayloadRef, columnar::RowPayload};
use crate::operator::row_cost::{Offsets, fixed_width, terminal_offset_width, variable_offsets};
use datafusion::arrow::datatypes::DataType;

const MAX_SCRATCH: usize = 512 * 1024 * 1024;
const MAX_UNIT_OUTPUT: usize = 16 * 1024 * 1024;

pub(in crate::operator::join::data_work) fn pair_credit_bytes(
    count: usize,
    pinned: usize,
) -> Option<usize> {
    let bytes = count
        .checked_mul(size_of::<MatchedPair>() + 256)?
        .checked_sub(pinned)?;
    // The retained snapshot uses part of the existing pair allowance. Preserve
    // enough separate funding for both selection arrays in the serial fallback.
    (bytes >= count.checked_mul(3 * size_of::<usize>())?).then_some(bytes)
}

pub(super) fn scratch_bytes(matches: usize, rows: usize, columns: usize) -> Option<usize> {
    // Selection/ranges, both take/interleave routes, source maps, flat-row cost
    // maps and null inventories. All units share this prepaid upper bound.
    let per_match = 512usize.checked_add(columns.checked_mul(16)?)?;
    matches
        .checked_mul(per_match)?
        .checked_add(rows.checked_mul(size_of::<u64>())?)?
        .checked_add(columns.checked_mul(4096)?)?
        .checked_add(size_of::<GatherWork>() + 4096)
        .filter(|&bytes| bytes <= MAX_SCRATCH)
}

pub(super) fn owned_payloads(output: &JoinOutput<'_>, stop: &GatherStop) -> Result<bool> {
    for (index, pair) in output.matched.iter().enumerate() {
        if index % 4096 == 0 {
            stop.check_abandoned()?;
        }
        let (left, right) = output.records(&pair);
        let funded = |payload: PayloadRef<'_>| {
            matches!(
                payload.owner(),
                Some(RowPayload::Shared { .. } | RowPayload::RestoredV2 { .. })
            )
        };
        if !funded(left) || !funded(right) {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) fn output_bytes(output: &JoinOutput<'_>, records: usize) -> Option<usize> {
    let (fixed, variable) = schema_bytes(output)?;
    let bytes = fixed
        .checked_mul(output.matched.len())?
        .checked_add(variable_bytes(output, &variable)?)?;
    funded_bytes(bytes, records, output.schema.fields().len())
}

fn funded_bytes(bytes: usize, records: usize, columns: usize) -> Option<usize> {
    // Buffer growth headroom plus per-array controls, bitmap padding and the
    // terminal byte-array offset in each chunk. This is physical funding, not V1.
    bytes
        .checked_mul(2)?
        .checked_add(records.checked_mul(columns.checked_mul(4096)?)?)
        .filter(|&bytes| bytes <= MAX_UNIT_OUTPUT)
}

fn schema_bytes(output: &JoinOutput<'_>) -> Option<(usize, Vec<usize>)> {
    let mut fixed = 0usize;
    let mut variable = Vec::new();
    for (ordinal, field) in output.schema.fields().iter().enumerate() {
        let data_type = field.data_type();
        // Null fixed-width values still occupy their full values buffer. One
        // validity byte per row also covers bitmap rounding conservatively.
        fixed = fixed
            .checked_add(physical_width(data_type)?)?
            .checked_add(1)?;
        if data_type.is_string() || matches!(data_type, DataType::Binary | DataType::LargeBinary) {
            variable.push(output.logical_column(ordinal));
        }
    }
    Some((fixed, variable))
}

fn physical_width(data_type: &DataType) -> Option<usize> {
    match data_type {
        DataType::FixedSizeBinary(width) => usize::try_from(*width).ok(),
        _ => fixed_width(data_type).map(|width| width - terminal_offset_width(data_type)),
    }
}

fn variable_bytes(output: &JoinOutput<'_>, columns: &[usize]) -> Option<usize> {
    if columns.is_empty() {
        return Some(0);
    }
    output.matched.iter().try_fold(0usize, |total, pair| {
        total.checked_add(selected_bytes(output, &pair, columns)?)
    })
}

fn selected_bytes(output: &JoinOutput<'_>, pair: &MatchedPair, columns: &[usize]) -> Option<usize> {
    let (left, right) = output.records(pair);
    columns.iter().try_fold(0usize, |bytes, &index| {
        let (payload, column) = if index < left.num_columns() {
            (left, index)
        } else {
            (right, index - left.num_columns())
        };
        bytes.checked_add(value_bytes(payload, column)?)
    })
}

fn value_bytes(payload: PayloadRef<'_>, column: usize) -> Option<usize> {
    let row = payload.offset();
    match variable_offsets(payload.columns()[column].as_ref())? {
        Offsets::Narrow(offsets) => usize::try_from(offsets[row + 1] - offsets[row]).ok(),
        Offsets::Wide(offsets) => usize::try_from(offsets[row + 1] - offsets[row]).ok(),
    }
}
