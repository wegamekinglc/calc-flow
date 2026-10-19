use super::{
    FundedMorsel, FundedProbeInputs, GatherStop, GatherWork, JoinOutput, ProbeInputs, Result,
};
use crate::StreamOperatorContext;
use crate::operator::join::{
    MatchedPair, PayloadRef,
    columnar::{Quantum, RowPayload},
    output_gather,
    retained::Rows,
};
use crate::operator::row_cost::{Offsets, fixed_width, terminal_offset_width, variable_offsets};
use datafusion::{
    arrow::{array::ArrayRef, datatypes::DataType, record_batch::RecordBatch},
    execution::memory_pool::MemoryReservation,
};
use std::sync::Arc;

#[cfg(test)]
#[path = "../../tests/morsel_tests/rowed_funding.rs"]
mod rowed_funding_tests;

#[cfg(test)]
#[path = "../../tests/morsel_tests/cpu_budget.rs"]
mod cpu_budget_tests;

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

pub(super) fn output_control_bytes(units: usize) -> Option<usize> {
    units
        .checked_mul(size_of::<Option<FundedMorsel>>() + size_of::<FundedMorsel>())?
        .checked_add(128)
}

pub(super) fn scratch_bytes(
    matches: usize,
    rows: usize,
    columns: usize,
    units: usize,
) -> Option<usize> {
    // Selection/ranges, both take/interleave routes, source maps, flat-row cost
    // maps and null inventories. All units share this prepaid upper bound.
    match_scratch_bytes(matches, columns)?
        .checked_add(rows.checked_mul(size_of::<u64>())?)?
        .checked_add(worker_scratch_bytes(columns, units)?)?
        .checked_add(output_control_bytes(units)?)?
        .checked_add(size_of::<GatherWork>() + 4096)
}

fn match_scratch_bytes(matches: usize, columns: usize) -> Option<usize> {
    let per_match = 512usize.checked_add(columns.checked_mul(16)?)?;
    matches.checked_mul(per_match)
}

fn worker_scratch_bytes(columns: usize, units: usize) -> Option<usize> {
    columns.checked_mul(units)?.checked_mul(4096)
}

pub(super) async fn fund_inputs(
    inputs: &Arc<ProbeInputs>,
    scratch: &MemoryReservation,
    base: usize,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<FundedProbeInputs>> {
    let Some(bytes) = input_bytes(inputs, context).await? else {
        return Ok(None);
    };
    let Some(total) = base.checked_add(bytes).filter(|&total| {
        total
            .checked_add(inputs.control.as_ref().map_or(0, |credit| credit.size()))
            .is_some()
    }) else {
        return Ok(None);
    };
    if scratch.try_grow(total).is_err() {
        return Ok(None);
    }
    Ok(Some(FundedProbeInputs {
        inputs: Arc::clone(inputs),
        _credit: scratch.split(bytes),
    }))
}

async fn input_bytes(
    inputs: &ProbeInputs,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<usize>> {
    use crate::operator::join::AdmittedRows;

    let mut bytes = 0usize;
    let mut quantum = Quantum::default();
    for group in (&inputs.admitted as &dyn AdmittedRows).groups() {
        quantum.step(context, 1, 0).await?;
        match group.row(0).record.owner() {
            Some(RowPayload::Rowed { parent, .. }) => {
                let Some(controls) = parent
                    .num_columns()
                    .checked_mul(size_of::<ArrayRef>())
                    .and_then(|columns| columns.checked_add(size_of::<RecordBatch>()))
                    .and_then(|controls| controls.checked_add(2 * size_of::<usize>()))
                else {
                    return Ok(None);
                };
                let Some(charged) = bytes.checked_add(controls) else {
                    return Ok(None);
                };
                bytes = charged;
                for column in parent.columns() {
                    quantum.step(context, 1, 0).await?;
                    if !output_gather::flat_owned_type(column.data_type()) {
                        return Ok(None);
                    }
                    // Include backing capacities and conservative array/Buffer controls.
                    let Some(charged) = bytes
                        .checked_add(column.get_array_memory_size())
                        .and_then(|total| total.checked_add(4096))
                    else {
                        return Ok(None);
                    };
                    bytes = charged;
                }
            }
            Some(RowPayload::Shared { .. } | RowPayload::RestoredV2 { .. }) => {}
            _ => return Ok(None),
        }
    }
    context.check_cancelled()?;
    Ok(Some(bytes))
}

pub(super) fn owned_payloads(
    inputs: &FundedProbeInputs,
    pairs: &crate::operator::join::matches::Selection,
    stop: &GatherStop,
) -> Result<bool> {
    for (index, pair) in pairs.iter().enumerate() {
        if index % 4096 == 0 {
            stop.check_abandoned()?;
        }
        let opposite = inputs
            .opposite
            .as_ref()
            .expect("owned morsel state")
            .payload(pair.opposite_index);
        let funded = |payload: PayloadRef<'_>| {
            matches!(
                payload.owner(),
                Some(RowPayload::Shared { .. } | RowPayload::RestoredV2 { .. })
            )
        };
        if !funded(opposite) {
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
    // terminal byte-array offset in each chunk, plus envelope control storage.
    bytes
        .checked_mul(2)?
        .checked_add(records.checked_mul(columns.checked_mul(4096)?)?)?
        .checked_add(super::envelopes::control_bytes(records, columns)?)
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
