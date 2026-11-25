use datafusion::arrow::array::{Array, Float64Array, Int64Array, UInt64Array};

use super::super::{
    LagValue, OutputStorage, Result, Statistic, TypedEwmaState, TypedFrame, TypedGroupInput,
    TypedGroupPlan, TypedOutputKind, TypedOutputPlan, TypedWindowState,
    cumulative::TypedCumulativeState, internal_error, valid_float64,
};
use crate::{
    CalcFlowError,
    accumulate::NumericAccumulator,
    array::indicators::{AtrKernel, MacdKernel, MacdParameters, RsiKernel},
    operator::rolling::{
        cumulative::CumulativeAccumulator,
        indicators::{Coordinate, numeric_error},
    },
    state::{EntityId, StateTable, ValueKind},
};

#[derive(Clone, Copy)]
pub(super) struct EntityRow<'a> {
    pub(super) entity: EntityId,
    pub(super) first: bool,
    pub(super) ordered_position: u64,
    pub(super) node_id: &'a str,
}

pub(super) struct RowReadouts<'a> {
    pub(super) accumulators: &'a [NumericAccumulator],
    pub(super) values: Vec<[Option<f64>; 3]>,
    pub(super) valid_counts: Option<Vec<u64>>,
    pub(super) cumulative: Option<Vec<Option<TypedWindowState>>>,
    pub(super) lag_values: Option<Vec<(LagValue, LagValue)>>,
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
    let count_input = groups.iter().find_map(|group| match group {
        TypedGroupPlan::Count { input_index, .. } => Some(*input_index),
        _ => None,
    });
    let only_counts = groups
        .iter()
        .all(|group| matches!(group, TypedGroupPlan::Count { .. }));
    let lag_input = groups.iter().find_map(|group| match group {
        TypedGroupPlan::Lag { input_index, .. } => Some(*input_index),
        _ => None,
    });
    let only_lags = groups
        .iter()
        .all(|group| matches!(group, TypedGroupPlan::Lag { .. }));
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
            TypedGroupPlan::Ewma { .. } | TypedGroupPlan::Cumulative { .. } => !has_indicator,
            TypedGroupPlan::Count { input_index, .. } => {
                only_counts && Some(*input_index) == count_input
            }
            TypedGroupPlan::Lag {
                input_index,
                storage,
                ..
            } => {
                only_lags
                    && Some(*input_index) == lag_input
                    && (storage.is_signed() || storage.is_unsigned() || storage.is_float())
            }
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
        TypedGroupPlan::Count { .. } => {
            matches!(output.kind, TypedOutputKind::Statistic(Statistic::Count))
                && output.storage == OutputStorage::Count
        }
        TypedGroupPlan::Lag { storage, .. } => {
            matches!(output.kind, TypedOutputKind::Lag | TypedOutputKind::Delta)
                && (storage.is_signed() || storage.is_unsigned() || storage.is_float())
                && output.storage == *storage
        }
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
        TypedGroupPlan::Cumulative { storage, .. } => {
            if !matches!(
                storage,
                OutputStorage::Int8
                    | OutputStorage::Int16
                    | OutputStorage::Int32
                    | OutputStorage::Int64
                    | OutputStorage::UInt8
                    | OutputStorage::UInt16
                    | OutputStorage::UInt32
                    | OutputStorage::UInt64
                    | OutputStorage::Float32
                    | OutputStorage::Float64
            ) {
                return false;
            }
            let expected = match output.kind {
                TypedOutputKind::Statistic(Statistic::Sum) => match storage {
                    OutputStorage::Int8
                    | OutputStorage::Int16
                    | OutputStorage::Int32
                    | OutputStorage::Int64 => OutputStorage::Int64,
                    OutputStorage::UInt8
                    | OutputStorage::UInt16
                    | OutputStorage::UInt32
                    | OutputStorage::UInt64 => OutputStorage::UInt64,
                    OutputStorage::Float32 | OutputStorage::Float64 => OutputStorage::Float64,
                    _ => return false,
                },
                TypedOutputKind::Statistic(Statistic::Min | Statistic::Max) => *storage,
                TypedOutputKind::Statistic(Statistic::Variance) => OutputStorage::Float64,
                _ => return false,
            };
            output.storage == expected
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
        .filter(|candidate| {
            matches!(
                candidate,
                TypedGroupPlan::Numeric { .. } | TypedGroupPlan::Count { .. }
            )
        })
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
    const CUMULATIVE: &[&str] = &[
        "valid_count",
        "value_kind",
        "value_low",
        "value_high",
        "origin",
        "mean",
        "m2",
        "has_infinity",
    ];
    match group {
        TypedGroupPlan::Rsi { .. } => RSI,
        TypedGroupPlan::Atr { .. } => ATR,
        TypedGroupPlan::Macd { .. } => MACD,
        TypedGroupPlan::Ewma { .. } => EWMA,
        TypedGroupPlan::Cumulative { .. } => CUMULATIVE,
        _ => &[],
    }
}

type Scratch = (
    Option<Vec<u64>>,
    Option<Vec<Option<TypedWindowState>>>,
    Option<Vec<(LagValue, LagValue)>>,
);

fn scratch_groups(groups: &[TypedGroupPlan]) -> Scratch {
    let valid_counts = groups
        .iter()
        .any(|group| matches!(group, TypedGroupPlan::Ewma { .. }))
        .then(|| vec![0; groups.len()]);
    let cumulative = groups
        .iter()
        .any(|group| matches!(group, TypedGroupPlan::Cumulative { .. }))
        .then(|| vec![None; groups.len()]);
    let lag_values = groups
        .iter()
        .any(|group| {
            matches!(group, TypedGroupPlan::Lag { storage, .. } if storage.is_signed() || storage.is_unsigned())
        })
        .then(|| vec![(LagValue::Float(None), LagValue::Float(None)); groups.len()]);
    (valid_counts, cumulative, lag_values)
}

fn ordered_error(error: CalcFlowError, position: EntityRow<'_>) -> CalcFlowError {
    numeric_error(
        error,
        position.node_id,
        Coordinate::OrderedPosition(position.ordered_position),
    )
}

fn push_rsi(
    table: &mut StateTable,
    position: EntityRow<'_>,
    base: usize,
    input: &Float64Array,
    row: usize,
    period: u64,
    field: &str,
) -> Result<Option<f64>> {
    let mut kernel = read_rsi(table, position.entity, base)?;
    let value = kernel
        .push(
            input.is_valid(row).then_some(input.value(row)),
            period,
            field,
        )
        .map_err(|error| ordered_error(error, position))?;
    write_rsi(table, position.entity, base, kernel)?;
    Ok(value)
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
    let EntityRow { entity, first, .. } = position;
    let mut values = vec![[None; 3]; groups.len()];
    let (mut valid_counts, mut cumulative, mut lag_values) = scratch_groups(groups);
    let (mut numeric_updated, mut lag_updated) = (false, false);
    for (index, (group, input)) in groups.iter().zip(inputs).enumerate() {
        match (group, input) {
            (
                TypedGroupPlan::Numeric { .. } | TypedGroupPlan::Count { .. },
                input @ (TypedGroupInput::Single(_) | TypedGroupInput::Values(_)),
            ) => {
                numeric_updated |= update_numeric(
                    table,
                    entity,
                    first,
                    input,
                    row,
                    nan_as_value,
                    !numeric_updated,
                )?;
            }
            (TypedGroupPlan::Lag { periods, .. }, TypedGroupInput::Single(input)) => {
                let (reference, current, updated) =
                    update_lag(table, entity, *periods, input, row, !lag_updated)?;
                values[index][0] = reference;
                values[index][1] = current;
                lag_updated |= updated;
            }
            (
                TypedGroupPlan::Lag { periods, .. },
                input @ (TypedGroupInput::Signed(_) | TypedGroupInput::Unsigned(_)),
            ) => {
                let (reference, current, updated) =
                    update_integer_lag(table, entity, *periods, input, row, !lag_updated)?;
                lag_values
                    .as_mut()
                    .expect("integer lag values are allocated")[index] = (reference, current);
                lag_updated |= updated;
            }
            (TypedGroupPlan::Rsi { .. }, TypedGroupInput::Rsi(input, field, period)) => {
                let base = column_base(groups, index);
                values[index][0] = push_rsi(table, position, base, input, row, *period, field)?;
            }
            (TypedGroupPlan::Atr { .. }, TypedGroupInput::Atr(inputs, fields, period)) => {
                let mut kernel = read_atr(table, entity, column_base(groups, index))?;
                let [high, low, close] = inputs;
                values[index][0] = kernel
                    .push(
                        [
                            high.is_valid(row).then_some(high.value(row)),
                            low.is_valid(row).then_some(low.value(row)),
                            close.is_valid(row).then_some(close.value(row)),
                        ],
                        *period,
                        fields.each_ref().map(String::as_str),
                    )
                    .map_err(|error| ordered_error(error, position))?;
                write_atr(table, entity, column_base(groups, index), kernel)?;
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
            (TypedGroupPlan::Ewma { .. }, TypedGroupInput::Single(input)) => {
                let (count, value) =
                    update_ewma(table, position, groups, index, group, input, row)?;
                valid_counts.as_mut().expect("EWMA counts are allocated")[index] = count;
                values[index][0] = Some(value);
            }
            (TypedGroupPlan::Cumulative { .. }, input) => {
                let summary = update_cumulative(table, position, groups, index, input, row, group)?;
                cumulative.as_mut().expect("cumulative state is allocated")[index] = Some(summary);
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
        cumulative,
        lag_values,
    })
}

fn update_numeric(
    table: &mut StateTable,
    entity: EntityId,
    first: bool,
    input: &TypedGroupInput,
    row: usize,
    nan_as_value: bool,
    push: bool,
) -> Result<bool> {
    let value = match input {
        TypedGroupInput::Single(input) => valid_float64(input, row, nan_as_value),
        TypedGroupInput::Values(input) => input.is_valid(row).then_some(1.0),
        _ => return Err(internal_error("rolling numeric input mismatch")),
    };
    if push {
        push_numeric(table, entity, value, first)?;
    }
    Ok(push)
}

fn push_numeric(
    table: &mut StateTable,
    entity: EntityId,
    value: Option<f64>,
    first: bool,
) -> Result<()> {
    if first {
        table.push_numeric_rows(0, entity, value)
    } else {
        table.push_numeric_rows_recurrent(0, entity, value)
    }
}

fn update_lag(
    table: &mut StateTable,
    entity: EntityId,
    periods: usize,
    input: &Float64Array,
    row: usize,
    push: bool,
) -> Result<(Option<f64>, Option<f64>, bool)> {
    let current = input.is_valid(row).then_some(input.value(row));
    let reference = if periods == 0 {
        current
    } else {
        table.f64_lag(0, entity, periods)?
    };
    let updated = periods > 0 && push;
    if updated {
        table.push_f64(0, entity, current)?;
    }
    Ok((reference, current, updated))
}

fn update_integer_lag(
    table: &mut StateTable,
    entity: EntityId,
    periods: usize,
    input: &TypedGroupInput,
    row: usize,
    push: bool,
) -> Result<(LagValue, LagValue, bool)> {
    match input {
        TypedGroupInput::Signed(input) => {
            update_signed_lag(table, entity, periods, input, row, push)
        }
        TypedGroupInput::Unsigned(input) => {
            update_unsigned_lag(table, entity, periods, input, row, push)
        }
        _ => Err(internal_error("rolling integer lag input mismatch")),
    }
}

fn update_signed_lag(
    table: &mut StateTable,
    entity: EntityId,
    periods: usize,
    input: &Int64Array,
    row: usize,
    push: bool,
) -> Result<(LagValue, LagValue, bool)> {
    let current = input.is_valid(row).then_some(input.value(row));
    let reference = if periods == 0 {
        current
    } else {
        table.i64_lag(0, entity, periods)?
    };
    if periods > 0 && push {
        table.push_i64(0, entity, current)?;
    }
    Ok((
        LagValue::Signed(reference),
        LagValue::Signed(current),
        periods > 0 && push,
    ))
}

fn update_unsigned_lag(
    table: &mut StateTable,
    entity: EntityId,
    periods: usize,
    input: &UInt64Array,
    row: usize,
    push: bool,
) -> Result<(LagValue, LagValue, bool)> {
    let current = input.is_valid(row).then_some(input.value(row));
    let reference = if periods == 0 {
        current
    } else {
        table.u64_lag(0, entity, periods)?
    };
    if periods > 0 && push {
        table.push_u64(0, entity, current)?;
    }
    Ok((
        LagValue::Unsigned(reference),
        LagValue::Unsigned(current),
        periods > 0 && push,
    ))
}

fn update_ewma(
    table: &mut StateTable,
    position: EntityRow<'_>,
    groups: &[TypedGroupPlan],
    index: usize,
    group: &TypedGroupPlan,
    input: &Float64Array,
    row: usize,
) -> Result<(u64, f64)> {
    let TypedGroupPlan::Ewma {
        alpha, cumulative, ..
    } = group
    else {
        return Err(internal_error("rolling EWMA plan mismatch"));
    };
    let base = column_base(groups, index);
    let mut state = TypedEwmaState::new(*alpha, *cumulative).load(table, position.entity, base)?;
    state.update(valid_float64(input, row, false), position.node_id)?;
    state.save(table, position.entity, base)?;
    Ok((state.valid_count, state.value))
}

impl TypedEwmaState {
    fn load(mut self, table: &StateTable, entity: EntityId, base: usize) -> Result<Self> {
        self.valid_count = table.u64(base, entity)?;
        self.value = f64::from_bits(table.u64(base + 1, entity)?);
        Ok(self)
    }

    fn save(self, table: &mut StateTable, entity: EntityId, base: usize) -> Result<()> {
        write_fields(
            table,
            entity,
            base,
            &[self.valid_count, self.value.to_bits()],
        )
    }
}

pub(super) fn fill_ewma(
    table: &mut StateTable,
    entities: &[EntityId],
    locals: &[usize],
    input: &Float64Array,
    seed: TypedEwmaState,
    node_id: &str,
    mut emit: impl FnMut(&TypedEwmaState) -> Result<()>,
) -> Result<()> {
    // Column zero counts transitions; a sole EWMA group owns columns one and two.
    let mut states = Vec::with_capacity(entities.len());
    for &entity in entities {
        states.push(seed.load(table, entity, 1)?);
    }
    for (row, &local) in locals.iter().enumerate() {
        let state = &mut states[local];
        state.update(valid_float64(input, row, false), node_id)?;
        emit(state)?;
    }
    for (&entity, state) in entities.iter().zip(states) {
        state.save(table, entity, 1)?;
    }
    Ok(())
}

fn update_cumulative(
    table: &mut StateTable,
    position: EntityRow<'_>,
    groups: &[TypedGroupPlan],
    index: usize,
    input: &TypedGroupInput,
    row: usize,
    group: &TypedGroupPlan,
) -> Result<TypedWindowState> {
    let TypedGroupPlan::Cumulative {
        statistic, storage, ..
    } = group
    else {
        return Err(internal_error("rolling cumulative plan mismatch"));
    };
    let (statistic, storage) = (*statistic, *storage);
    let base = column_base(groups, index);
    let mut summary = read_cumulative(table, position.entity, base, statistic, storage)?;
    summary.update(input, row, position.node_id)?;
    write_fields(table, position.entity, base, &summary.accumulator.columns())?;
    Ok(TypedWindowState::Cumulative(summary))
}

fn read_cumulative(
    table: &StateTable,
    entity: EntityId,
    base: usize,
    statistic: Statistic,
    storage: OutputStorage,
) -> Result<TypedCumulativeState> {
    let mut columns = [0; 8];
    for (offset, value) in columns.iter_mut().enumerate() {
        *value = table.u64(base + offset, entity)?;
    }
    let mut summary = TypedCumulativeState::new(statistic);
    summary.accumulator =
        CumulativeAccumulator::from_columns(columns, statistic, &storage.data_type())?;
    Ok(summary)
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
