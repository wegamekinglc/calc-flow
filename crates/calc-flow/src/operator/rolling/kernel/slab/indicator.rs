use datafusion::arrow::array::Array;

use super::super::{
    OutputStorage, Result, Statistic, TypedFrame, TypedGroupInput, TypedGroupPlan, TypedOutputKind,
    TypedOutputPlan, internal_error, valid_float64,
};
use crate::{
    accumulate::NumericAccumulator,
    array::indicators::{AtrKernel, MacdKernel, MacdParameters, RsiKernel},
    state::{EntityId, StateTable, ValueKind},
};

#[derive(Clone, Copy)]
pub(super) struct EntityRow<'a> {
    pub(super) entity: EntityId,
    pub(super) first: bool,
    pub(super) node_id: &'a str,
}

pub(super) struct RowReadouts<'a> {
    pub(super) accumulators: &'a [NumericAccumulator],
    pub(super) values: Vec<[Option<f64>; 3]>,
    pub(super) valid_counts: Option<Vec<u64>>,
}

pub(super) fn supports_plan(groups: &[TypedGroupPlan], outputs: &[TypedOutputPlan]) -> bool {
    let numeric_input = groups.iter().find_map(|group| match group {
        TypedGroupPlan::Numeric { input_index, .. } => Some(*input_index),
        _ => None,
    });
    let has_indicator = groups.iter().any(|group| {
        matches!(
            group,
            TypedGroupPlan::Rsi { .. } | TypedGroupPlan::Atr { .. } | TypedGroupPlan::Macd { .. }
        )
    });
    !groups.is_empty()
        && groups.iter().all(|group| match group {
            TypedGroupPlan::Numeric {
                input_index,
                frame: TypedFrame::Rows(_),
                ..
            } => Some(*input_index) == numeric_input,
            TypedGroupPlan::Rsi { .. }
            | TypedGroupPlan::Atr { .. }
            | TypedGroupPlan::Macd { .. } => true,
            TypedGroupPlan::Ewma { .. } => !has_indicator,
            _ => false,
        })
        && !outputs.is_empty()
        && outputs.iter().all(|output| {
            groups
                .get(output.group)
                .is_some_and(|group| output_supported(group, output))
        })
}

fn output_supported(group: &TypedGroupPlan, output: &TypedOutputPlan) -> bool {
    match group {
        TypedGroupPlan::Numeric { .. } => match output.kind {
            TypedOutputKind::Statistic(Statistic::Count) => output.storage == OutputStorage::Count,
            TypedOutputKind::Statistic(
                Statistic::Sum | Statistic::Mean | Statistic::Variance | Statistic::Stddev,
            ) => output.storage == OutputStorage::Float64,
            _ => false,
        },
        TypedGroupPlan::Rsi { .. } => {
            matches!(output.kind, TypedOutputKind::Rsi) && output.storage == OutputStorage::Float64
        }
        TypedGroupPlan::Atr { .. } => {
            matches!(output.kind, TypedOutputKind::Atr) && output.storage == OutputStorage::Float64
        }
        TypedGroupPlan::Macd { .. } => {
            matches!(output.kind, TypedOutputKind::Macd(component) if component < 3)
                && output.storage == OutputStorage::Float64
        }
        TypedGroupPlan::Ewma { .. } => {
            matches!(output.kind, TypedOutputKind::Ewma) && output.storage == OutputStorage::Float64
        }
        _ => false,
    }
}

pub(super) fn numeric_only(groups: &[TypedGroupPlan]) -> bool {
    groups
        .iter()
        .all(|group| matches!(group, TypedGroupPlan::Numeric { .. }))
}

pub(super) fn numeric_ordinal(groups: &[TypedGroupPlan], group: usize) -> usize {
    groups[..group]
        .iter()
        .filter(|candidate| matches!(candidate, TypedGroupPlan::Numeric { .. }))
        .count()
}

pub(super) fn add_columns(table: &mut StateTable, groups: &[TypedGroupPlan]) {
    for (index, group) in groups.iter().enumerate() {
        for field in field_names(group) {
            table.add_keyed_column(&format!("group_{index}_{field}"), ValueKind::UInt64);
        }
    }
}

fn field_names(group: &TypedGroupPlan) -> &'static [&'static str] {
    const RSI: &[&str] = &[
        "positions",
        "has_previous",
        "seed_count",
        "previous_price",
        "mean_gain",
        "mean_loss",
    ];
    const ATR: &[&str] = &[
        "positions",
        "has_previous",
        "seed_count",
        "previous_close",
        "mean_tr",
    ];
    const MACD: &[&str] = &[
        "positions",
        "fast_count",
        "fast_value",
        "slow_count",
        "slow_value",
        "signal_count",
        "signal_value",
    ];
    const EWMA: &[&str] = &["valid_count", "value"];
    match group {
        TypedGroupPlan::Rsi { .. } => RSI,
        TypedGroupPlan::Atr { .. } => ATR,
        TypedGroupPlan::Macd { .. } => MACD,
        TypedGroupPlan::Ewma { .. } => EWMA,
        _ => &[],
    }
}

pub(super) fn update<'a>(
    table: &'a mut StateTable,
    position: EntityRow<'_>,
    groups: &[TypedGroupPlan],
    inputs: &[TypedGroupInput],
    row: usize,
    nan_as_value: bool,
) -> Result<RowReadouts<'a>> {
    if groups.len() != inputs.len() {
        return Err(internal_error("rolling slab group input width mismatch"));
    }
    let EntityRow {
        entity,
        first,
        node_id,
    } = position;
    let mut values = vec![[None; 3]; groups.len()];
    let mut valid_counts = groups
        .iter()
        .any(|group| matches!(group, TypedGroupPlan::Ewma { .. }))
        .then(|| vec![0; groups.len()]);
    let mut numeric_updated = false;
    for (index, (group, input)) in groups.iter().zip(inputs).enumerate() {
        match (group, input) {
            (TypedGroupPlan::Numeric { .. }, TypedGroupInput::Single(input)) => {
                if !numeric_updated {
                    let value = valid_float64(input, row, nan_as_value);
                    if first {
                        table.push_numeric_rows(0, entity, value)?;
                    } else {
                        table.push_numeric_rows_recurrent(0, entity, value)?;
                    }
                    numeric_updated = true;
                }
            }
            (TypedGroupPlan::Rsi { .. }, TypedGroupInput::Rsi(input, field, period)) => {
                let base = column_base(groups, index);
                let mut kernel = read_rsi(table, entity, base)?;
                values[index][0] = kernel.push(
                    input.is_valid(row).then_some(input.value(row)),
                    *period,
                    field,
                )?;
                write_rsi(table, entity, base, kernel)?;
            }
            (TypedGroupPlan::Atr { .. }, TypedGroupInput::Atr(inputs, fields, period)) => {
                let base = column_base(groups, index);
                let mut kernel = read_atr(table, entity, base)?;
                let [high, low, close] = inputs;
                values[index][0] = kernel.push(
                    [
                        high.is_valid(row).then_some(high.value(row)),
                        low.is_valid(row).then_some(low.value(row)),
                        close.is_valid(row).then_some(close.value(row)),
                    ],
                    *period,
                    fields.each_ref().map(String::as_str),
                )?;
                write_atr(table, entity, base, kernel)?;
            }
            (TypedGroupPlan::Macd { options, .. }, TypedGroupInput::Single(input)) => {
                let base = column_base(groups, index);
                let mut kernel = read_macd(table, entity, base)?;
                values[index] = kernel.push(
                    input.is_valid(row).then_some(input.value(row)),
                    &MacdParameters::new(*options),
                )?;
                write_macd(table, entity, base, kernel)?;
            }
            (
                TypedGroupPlan::Ewma {
                    alpha, cumulative, ..
                },
                TypedGroupInput::Single(input),
            ) => {
                let base = column_base(groups, index);
                let (count, value) = crate::array::rolling::next_ewma(
                    table.u64(base, entity)?,
                    f64::from_bits(table.u64(base + 1, entity)?),
                    valid_float64(input, row, false),
                    *alpha,
                    *cumulative,
                )
                .ok_or_else(|| {
                    super::super::operator_error(node_id, "rolling EWMA sample count overflowed")
                })?;
                valid_counts.as_mut().expect("EWMA counts are allocated")[index] = count;
                values[index][0] = Some(value);
                write_fields(table, entity, base, &[count, value.to_bits()])?;
            }
            _ => return Err(internal_error("rolling slab group input mismatch")),
        }
    }
    let accumulators = if numeric_updated {
        table.numeric_accumulators(0, entity)?
    } else {
        &[]
    };
    Ok(RowReadouts {
        accumulators,
        values,
        valid_counts,
    })
}

fn column_base(groups: &[TypedGroupPlan], index: usize) -> usize {
    1 + groups[..index]
        .iter()
        .map(|group| field_names(group).len())
        .sum::<usize>()
}

fn read_rsi(table: &StateTable, entity: EntityId, base: usize) -> Result<RsiKernel> {
    Ok(RsiKernel {
        positions: table.u64(base, entity)?,
        has_previous: table.u64(base + 1, entity)? != 0,
        seed_count: table.u64(base + 2, entity)?,
        previous_price: f64::from_bits(table.u64(base + 3, entity)?),
        mean_gain: f64::from_bits(table.u64(base + 4, entity)?),
        mean_loss: f64::from_bits(table.u64(base + 5, entity)?),
    })
}

fn write_rsi(
    table: &mut StateTable,
    entity: EntityId,
    base: usize,
    kernel: RsiKernel,
) -> Result<()> {
    let values = [
        kernel.positions,
        u64::from(kernel.has_previous),
        kernel.seed_count,
        kernel.previous_price.to_bits(),
        kernel.mean_gain.to_bits(),
        kernel.mean_loss.to_bits(),
    ];
    write_fields(table, entity, base, &values)
}

fn read_atr(table: &StateTable, entity: EntityId, base: usize) -> Result<AtrKernel> {
    Ok(AtrKernel {
        positions: table.u64(base, entity)?,
        has_previous: table.u64(base + 1, entity)? != 0,
        seed_count: table.u64(base + 2, entity)?,
        previous_close: f64::from_bits(table.u64(base + 3, entity)?),
        mean_tr: f64::from_bits(table.u64(base + 4, entity)?),
    })
}

fn write_atr(
    table: &mut StateTable,
    entity: EntityId,
    base: usize,
    kernel: AtrKernel,
) -> Result<()> {
    let values = [
        kernel.positions,
        u64::from(kernel.has_previous),
        kernel.seed_count,
        kernel.previous_close.to_bits(),
        kernel.mean_tr.to_bits(),
    ];
    write_fields(table, entity, base, &values)
}

fn read_macd(table: &StateTable, entity: EntityId, base: usize) -> Result<MacdKernel> {
    Ok(MacdKernel {
        positions: table.u64(base, entity)?,
        fast_count: table.u64(base + 1, entity)?,
        fast_value: f64::from_bits(table.u64(base + 2, entity)?),
        slow_count: table.u64(base + 3, entity)?,
        slow_value: f64::from_bits(table.u64(base + 4, entity)?),
        signal_count: table.u64(base + 5, entity)?,
        signal_value: f64::from_bits(table.u64(base + 6, entity)?),
    })
}

fn write_macd(
    table: &mut StateTable,
    entity: EntityId,
    base: usize,
    kernel: MacdKernel,
) -> Result<()> {
    let values = [
        kernel.positions,
        kernel.fast_count,
        kernel.fast_value.to_bits(),
        kernel.slow_count,
        kernel.slow_value.to_bits(),
        kernel.signal_count,
        kernel.signal_value.to_bits(),
    ];
    write_fields(table, entity, base, &values)
}

fn write_fields(
    table: &mut StateTable,
    entity: EntityId,
    base: usize,
    values: &[u64],
) -> Result<()> {
    for (offset, &value) in values.iter().enumerate() {
        table.set_u64(base + offset, entity, value)?;
    }
    Ok(())
}
