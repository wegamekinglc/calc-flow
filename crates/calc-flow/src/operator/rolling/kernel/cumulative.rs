//! Primitive Arrow transitions sharing the durable cumulative summary.

use datafusion::common::ScalarValue;

use super::{
    Array, CompiledAggregate, CompiledRollingOutput, CompiledWindowGroup, DerivedBuilder,
    OutputStorage, Schema, Statistic, TypedGroupInput, TypedGroupPlan, TypedOutputKind,
    TypedOutputPlan, extrema_storage, internal_error, typed_output_mismatch, valid_float64,
};
use crate::{
    Result,
    operator::rolling::cumulative::{CumulativeAccumulator, NumericValue},
};

#[derive(Clone, Debug)]
pub(super) struct TypedCumulativeState {
    pub(super) accumulator: CumulativeAccumulator,
    statistic: Statistic,
}

impl TypedCumulativeState {
    pub(super) fn new(statistic: Statistic) -> Self {
        Self {
            accumulator: CumulativeAccumulator::default(),
            statistic,
        }
    }

    pub(super) fn update(&mut self, input: &TypedGroupInput, row: usize, node: &str) -> Result<()> {
        let sample = match input {
            TypedGroupInput::Single(values) => {
                valid_float64(values, row, false).map(|value| NumericValue::Float(value.to_bits()))
            }
            TypedGroupInput::Signed(values) => values
                .is_valid(row)
                .then(|| NumericValue::Signed(values.value(row))),
            TypedGroupInput::Unsigned(values) => values
                .is_valid(row)
                .then(|| NumericValue::Unsigned(values.value(row))),
            TypedGroupInput::Pair(..) => {
                return Err(internal_error("cumulative input is not primitive"));
            }
        };
        if let Some(sample) = sample {
            self.accumulator.add_numeric(sample, self.statistic, node)?;
        }
        Ok(())
    }

    pub(super) fn append(
        &self,
        builder: &mut DerivedBuilder,
        output: &TypedOutputPlan,
        node: &str,
    ) -> Result<()> {
        if self.statistic == Statistic::Variance {
            let DerivedBuilder::Float(builder, _) = builder else {
                return Err(typed_output_mismatch());
            };
            builder.append_option(self.accumulator.variance(output.ddof));
            return Ok(());
        }
        match (builder, self.accumulator.scalar(node)?) {
            (DerivedBuilder::Signed(builder, _), ScalarValue::Int64(Some(value))) => {
                builder.append_value(value);
            }
            (DerivedBuilder::Unsigned(builder, _), ScalarValue::UInt64(Some(value))) => {
                builder.append_value(value);
            }
            (DerivedBuilder::Float(builder, _), ScalarValue::Float64(Some(value))) => {
                builder.append_value(value);
            }
            _ => return Err(typed_output_mismatch()),
        }
        Ok(())
    }
}

pub(super) fn compile_group(
    input_schema: &Schema,
    input_index: usize,
    statistic: Statistic,
) -> std::result::Result<TypedGroupPlan, String> {
    let storage = if statistic == Statistic::Variance {
        super::require_numeric_type(input_schema, input_index)?;
        OutputStorage::Float64
    } else {
        extrema_storage(input_schema.field(input_index).data_type())?
    };
    Ok(TypedGroupPlan::Cumulative {
        input_index,
        statistic,
        storage,
    })
}

pub(super) fn compile_output(
    output: &CompiledRollingOutput,
    aggregate: &CompiledAggregate,
) -> std::result::Result<TypedOutputPlan, String> {
    super::require_generated_transition(
        &format!("cumulative_{}", aggregate.statistic.name()),
        super::GeneratedTransition::Cumulative,
    )?;
    Ok(TypedOutputPlan {
        group: aggregate.group,
        kind: TypedOutputKind::Statistic(aggregate.statistic),
        storage: extrema_storage(&output.output_type)?,
        min_periods: aggregate.min_periods,
        ddof: aggregate.ddof,
    })
}

/// Layout 4 identities predate typed cumulative execution. Freeze that identity
/// while selecting a faster physical implementation of the same transition.
pub(super) fn legacy_fallback_reason(
    input_schema: &Schema,
    groups: &[CompiledWindowGroup],
) -> Option<String> {
    if !groups
        .iter()
        .any(|group| matches!(group, CompiledWindowGroup::Cumulative { .. }))
    {
        return None;
    }
    for group in groups {
        if matches!(group, CompiledWindowGroup::Cumulative { .. }) {
            return Some("cumulative summary uses the general row kernel".into());
        }
        if let Err(reason) = super::compile_typed_group(input_schema, group) {
            return Some(reason);
        }
    }
    None
}
