use std::fmt;

use crate::{
    Result,
    array::{
        MacdOptions,
        indicators::{MacdKernel, MacdParameters},
    },
    operator::rolling::{RollingOutputSpec, Schema, compile_error},
};

#[derive(Clone)]
pub(in super::super) struct MacdGroup {
    pub(super) input: usize,
    pub(super) options: MacdOptions,
    parameters: MacdParameters,
}
impl PartialEq for MacdGroup {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input && self.options == other.options
    }
}
impl Eq for MacdGroup {}
impl fmt::Debug for MacdGroup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MacdGroup")
            .field("input", &self.input)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}
impl MacdGroup {
    pub(super) fn compile(schema: &Schema, output: &RollingOutputSpec) -> Result<Self> {
        let RollingOutputSpec::MacdComponent { input, .. } = output else {
            return Err(super::internal_error("expected MACD component declaration"));
        };
        let index = super::super::compile::exact_field_index(schema, input)?;
        if !matches!(
            schema.field(index).data_type(),
            super::DataType::UInt8
                | super::DataType::UInt16
                | super::DataType::UInt32
                | super::DataType::UInt64
                | super::DataType::Int8
                | super::DataType::Int16
                | super::DataType::Int32
                | super::DataType::Int64
                | super::DataType::Float32
                | super::DataType::Float64
        ) {
            return Err(compile_error(format!(
                "rolling MACD input {input:?} requires integer or floating values"
            )));
        }
        let options = options(output)?;
        Ok(Self {
            input: index,
            options,
            parameters: MacdParameters::new(options),
        })
    }
    pub(super) fn push(
        &self,
        state: &mut MacdKernel,
        row: &[super::ScalarValue],
    ) -> Result<[Option<f64>; 3]> {
        let sample = &row[self.input];
        let value = (!sample.is_null()).then(|| super::super::float_sample(sample));
        state.push(value, &self.parameters)
    }
}

pub(super) fn options(output: &RollingOutputSpec) -> Result<MacdOptions> {
    let RollingOutputSpec::MacdComponent {
        fast_span,
        slow_span,
        signal_span,
        min_periods,
        signal_min_periods,
        ..
    } = output
    else {
        return Err(super::internal_error("expected MACD component declaration"));
    };
    MacdOptions::new(*fast_span, *slow_span, *signal_span)?
        .with_min_periods(*min_periods)?
        .with_signal_min_periods(*signal_min_periods)
}
