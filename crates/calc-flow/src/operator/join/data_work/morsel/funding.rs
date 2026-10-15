use super::{GatherStop, GatherWork, JoinOutput, Result};
use crate::operator::join::{
    PayloadRef, STREAM_JOIN_STATE_ROW_OVERHEAD_BYTES_V1, columnar::RowPayload,
    native_lookup::scratch_error,
};

const MAX_SCRATCH: usize = 512 * 1024 * 1024;
const MAX_UNIT_OUTPUT: usize = 16 * 1024 * 1024;

pub(in crate::operator::join::data_work) fn pair_credit_bytes(
    count: usize,
    pinned: usize,
) -> Option<usize> {
    let bytes = count
        .checked_mul(size_of::<crate::operator::join::MatchedPair>() + 256)?
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

pub(super) fn output_bytes(output: &JoinOutput<'_>, records: usize) -> Result<Option<usize>> {
    let bytes = output.matched.iter().try_fold(0u64, |total, pair| {
        total.checked_add(payload_bytes(output, &pair)?)
    });
    let bytes = bytes
        .and_then(|bytes| usize::try_from(bytes).ok())
        .and_then(|bytes| bytes.checked_mul(2))
        .and_then(|bytes| {
            bytes.checked_add(records.checked_mul(output.schema.fields().len().checked_mul(4096)?)?)
        })
        .ok_or_else(|| scratch_error(output.operator_id))?;
    Ok((bytes <= MAX_UNIT_OUTPUT).then_some(bytes))
}

fn payload_bytes(
    output: &JoinOutput<'_>,
    pair: &crate::operator::join::MatchedPair,
) -> Option<u64> {
    // Neither retained row metadata nor canonical key storage is copied to
    // output. Keep payload/null allowances, including unprojected columns.
    let overhead = (2 * STREAM_JOIN_STATE_ROW_OVERHEAD_BYTES_V1 + 16)
        .checked_add(u64::try_from(output.opposite.key(pair.opposite_index).len()).ok()?)?;
    output.admitted_charges.expect("flat morsel charges")[pair.pos]
        .checked_add(output.opposite.charge(pair.opposite_index))?
        .checked_sub(overhead)
}
