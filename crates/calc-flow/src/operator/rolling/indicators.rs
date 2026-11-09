//! Table adapters for the shared native indicator transitions.

mod capture;
mod cells;
pub(super) use capture::Capture;
mod codec;
mod encoded_output;
mod ipc_encode_plan;
mod math_requirements;
pub(super) use cells::{ComputedCells, compute as compute_cells};
mod chunk_requirements;
pub(super) mod controlled;
mod finalization;
mod finite;
mod footprint;
mod footprint_state;
mod input;
pub(super) mod ipc_preflight;
pub(super) mod materialize;
mod native;
mod numeric_diagnostic;
pub(super) use numeric_diagnostic::Coordinate;
use numeric_diagnostic::numeric_error;
mod prefix;
mod resources;
mod scalar_materialization_plan;
mod scalar_output_plan;
mod state_record;
mod state_schema_requirements;
pub(super) use resources::Resources;
mod storage;
pub(super) use storage::StateOwner;

pub(super) fn initial_state(
    compiled: &super::CompiledRollingSpec,
    node: &str,
) -> Result<(StateOwner, Option<Resources>)> {
    if compiled.indicator_groups.is_empty() {
        return Ok((
            StateOwner::new(super::RollingStreamState::default(), false),
            None,
        ));
    }
    let (resources, state) = Resources::for_indicator(node)?;
    Ok((StateOwner::Shared(state), Some(resources)))
}

mod decoded_record;
mod ipc_decode_plan;
mod recovery;
mod restore;
mod state_decode_requirements;
pub(crate) use recovery::{IndicatorRollingRecord, RollingIoReservation};

use datafusion::{
    arrow::datatypes::{DataType, Schema},
    common::ScalarValue,
};

use super::{CompiledEvaluation, RollingOutputSpec, RollingSpec, compile_error, internal_error};
use crate::{
    CalcFlowError, Result,
    array::indicators::{AtrKernel, RsiKernel},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum Group {
    Rsi {
        input: usize,
        field: String,
        period: u64,
    },
    Atr {
        inputs: [usize; 3],
        fields: [String; 3],
        period: u64,
    },
}

#[derive(Clone, Copy)]
pub(super) enum State {
    Rsi(RsiKernel),
    Atr(AtrKernel),
}

impl std::fmt::Debug for State {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Rsi(_) => "RsiState",
            Self::Atr(_) => "AtrState",
        })
    }
}

pub(super) fn validate_spec(spec: &RollingSpec) -> Result<()> {
    for (index, output) in spec.outputs.iter().enumerate() {
        let period = match output {
            RollingOutputSpec::Rsi { period, .. } | RollingOutputSpec::Atr { period, .. } => {
                *period
            }
            _ => continue,
        };
        if spec.state_layout_version != super::ROLLING_INDICATOR_STATE_LAYOUT_VERSION {
            return Err(invalid(
                "rolling.state_layout_version",
                "RSI/ATR require rolling state layout 5",
            ));
        }
        if period == 0 {
            return Err(invalid(
                &format!("rolling.outputs[{index}].period"),
                "must be greater than zero",
            ));
        }
    }
    Ok(())
}

fn invalid(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: field.into(),
        message: message.into(),
    }
}

fn float_column(schema: &Schema, name: &str) -> Result<usize> {
    let index = super::compile::exact_field_index(schema, name)?;
    if !matches!(
        schema.field(index).data_type(),
        DataType::Float32 | DataType::Float64
    ) {
        return Err(compile_error(format!(
            "rolling indicator input {name:?} requires Float32 or Float64"
        )));
    }
    Ok(index)
}

pub(super) fn compile(
    schema: &Schema,
    output: &RollingOutputSpec,
    groups: &mut Vec<Group>,
) -> Result<CompiledEvaluation> {
    let group = declaration(schema, output)?;
    let index = if let Some(index) = groups.iter().position(|existing| *existing == group) {
        index
    } else {
        groups.push(group);
        groups.len() - 1
    };
    Ok(CompiledEvaluation::Indicator(index))
}

fn declaration(schema: &Schema, output: &RollingOutputSpec) -> Result<Group> {
    match output {
        RollingOutputSpec::Rsi { input, period, .. } => Ok(Group::Rsi {
            input: float_column(schema, input)?,
            field: input.clone(),
            period: *period,
        }),
        RollingOutputSpec::Atr {
            high,
            low,
            close,
            period,
            ..
        } => Ok(Group::Atr {
            inputs: [
                float_column(schema, high)?,
                float_column(schema, low)?,
                float_column(schema, close)?,
            ],
            fields: [high.clone(), low.clone(), close.clone()],
            period: *period,
        }),
        _ => Err(internal_error("expected an indicator output")),
    }
}

impl Group {
    pub(super) fn fresh(&self) -> State {
        match self {
            Self::Rsi { .. } => State::Rsi(RsiKernel::default()),
            Self::Atr { .. } => State::Atr(AtrKernel::default()),
        }
    }

    fn push(&self, state: &mut State, row: &[ScalarValue]) -> Result<Option<f64>> {
        match (self, state) {
            (
                Self::Rsi {
                    input,
                    field,
                    period,
                },
                State::Rsi(state),
            ) => state.push(sample(&row[*input]), *period, field),
            (
                Self::Atr {
                    inputs,
                    fields,
                    period,
                },
                State::Atr(state),
            ) => state.push(
                inputs.map(|index| sample(&row[index])),
                *period,
                fields.each_ref().map(String::as_str),
            ),
            _ => Err(internal_error("indicator group has the wrong state")),
        }
    }
}

fn sample(value: &ScalarValue) -> Option<f64> {
    match value {
        ScalarValue::Float32(value) => value.map(f64::from),
        ScalarValue::Float64(value) => *value,
        _ => None,
    }
}

pub(super) fn advance(
    groups: &[Group],
    states: &mut [State],
    row: &[ScalarValue],
    node: &str,
    position: u64,
) -> Result<Vec<Option<f64>>> {
    groups
        .iter()
        .zip(states)
        .map(|(group, state)| {
            group
                .push(state, row)
                .map_err(|error| numeric_error(error, node, Coordinate::OrderedPosition(position)))
        })
        .collect()
}

pub(super) fn validate_row(
    groups: &[Group],
    row: &[ScalarValue],
    node: &str,
    coordinate: Coordinate,
) -> Result<()> {
    for group in groups {
        group
            .validate_row(row)
            .map_err(|error| numeric_error(error, node, coordinate))?;
    }
    Ok(())
}

impl Group {
    fn validate_row(&self, row: &[ScalarValue]) -> Result<()> {
        use crate::array::indicators::{validate_atr_samples, validate_rsi_sample};
        match self {
            Self::Rsi { input, field, .. } => {
                validate_rsi_sample(sample(&row[*input]), field)?;
            }
            Self::Atr { inputs, fields, .. } => {
                validate_atr_samples(
                    inputs.map(|index| sample(&row[index])),
                    fields.each_ref().map(String::as_str),
                )?;
            }
        }
        Ok(())
    }
}

impl State {
    pub(super) const fn logical_bytes(&self) -> u64 {
        match self {
            Self::Rsi(_) => 48,
            Self::Atr(_) => 40,
        }
    }
}

pub(super) fn workers_for_layout(
    layout: u32,
    resources: Option<&Resources>,
) -> Option<std::sync::Arc<crate::array::RollingWorkers>> {
    if layout != super::ROLLING_INDICATOR_STATE_LAYOUT_VERSION {
        return None;
    }
    Some(resources.map_or_else(
        || std::sync::Arc::new(crate::array::RollingWorkers::default()),
        Resources::workers,
    ))
}

pub(super) fn pending_for_layout(
    layout: u32,
    node: &str,
) -> Option<std::sync::Arc<crate::array::RollingPending>> {
    (layout == super::ROLLING_INDICATOR_STATE_LAYOUT_VERSION)
        .then(|| std::sync::Arc::new(crate::array::RollingPending::rolling(node)))
}

mod work_identity;
