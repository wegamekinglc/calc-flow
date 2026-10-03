use super::{AggregateFunctionExpr, ArrayRef, GroupsAccumulator, Result, ScalarValue, df_error};
use datafusion::{
    arrow::{
        array::{ArrowNativeTypeOp, Float64Array},
        datatypes::DataType,
    },
    common::{DataFusionError, Result as DataFusionResult},
    logical_expr::EmitTo,
};

pub(super) fn selected(expression: &AggregateFunctionExpr) -> bool {
    expression.fun().name() == "sum" && expression.field().data_type() == &DataType::Float64
}

pub(super) enum PartialAccumulator {
    Native(Box<dyn GroupsAccumulator>),
    OrderedSum(Vec<Option<f64>>),
}

impl PartialAccumulator {
    pub fn new(expression: &AggregateFunctionExpr, name: &str) -> Result<Self> {
        if selected(expression) {
            Ok(Self::OrderedSum(Vec::new()))
        } else {
            expression
                .create_groups_accumulator()
                .map(Self::Native)
                .map_err(|error| df_error(name, error))
        }
    }

    pub fn seed(
        &mut self,
        rank: usize,
        saved: &ScalarValue,
        count: usize,
        name: &str,
    ) -> Result<()> {
        match self {
            Self::OrderedSum(sums) => {
                let ScalarValue::Float64(saved) = saved else {
                    return Err(df_error(name, "ordered SUM state must be Float64"));
                };
                sums.resize(count, None);
                sums[rank] = *saved;
                Ok(())
            }
            Self::Native(accumulator) => {
                if saved.is_null() {
                    return Ok(());
                }
                let reset = super::grouped_float::reset(saved, name)?
                    .to_array()
                    .map_err(|error| df_error(name, error))?;
                let saved = saved.to_array().map_err(|error| df_error(name, error))?;
                let filter = datafusion::arrow::array::BooleanArray::from(vec![true]);
                accumulator
                    .merge_batch(&[reset], &[rank], Some(&filter), count)
                    .map_err(|error| df_error(name, error))?;
                accumulator
                    .merge_batch(&[saved], &[rank], Some(&filter), count)
                    .map_err(|error| df_error(name, error))
            }
        }
    }

    pub fn update_batch(
        &mut self,
        values: &[ArrayRef],
        indices: &[usize],
        count: usize,
    ) -> DataFusionResult<()> {
        match self {
            Self::Native(accumulator) => accumulator.update_batch(values, indices, None, count),
            Self::OrderedSum(sums) => {
                let values = values[0]
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        DataFusionError::Internal("ordered SUM input must be Float64".into())
                    })?;
                sums.resize(count, None);
                for (&rank, value) in indices.iter().zip(values.iter()) {
                    if let Some(value) = value {
                        sums[rank] = Some(sums[rank].unwrap_or(0.0).add_wrapping(value));
                    }
                }
                Ok(())
            }
        }
    }

    pub fn state(&mut self, emit_to: EmitTo) -> DataFusionResult<Vec<ArrayRef>> {
        match self {
            Self::Native(accumulator) => accumulator.state(emit_to),
            Self::OrderedSum(sums) => Ok(vec![std::sync::Arc::new(Float64Array::from(
                emit_to.take_needed(sums),
            ))]),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Self::Native(accumulator) => accumulator.size(),
            Self::OrderedSum(sums) => {
                size_of::<Self>() + sums.capacity() * size_of::<Option<f64>>()
            }
        }
    }
}
