use super::{AggregateFunctionExpr, ArrayRef, GroupsAccumulator, Result, ScalarValue, df_error};
use datafusion::{
    arrow::{
        array::{ArrowNativeTypeOp, Float64Array, UInt64Array},
        datatypes::DataType,
    },
    common::{DataFusionError, Result as DataFusionResult},
    logical_expr::EmitTo,
};

pub(super) fn selected(expression: &AggregateFunctionExpr) -> bool {
    matches!(expression.fun().name(), "sum" | "avg")
        && expression.field().data_type() == &DataType::Float64
}

pub(super) enum PartialAccumulator {
    Native(Box<dyn GroupsAccumulator>),
    Ordered(OrderedState),
}

pub(super) struct OrderedState {
    sums: Vec<Option<f64>>,
    counts: Option<Vec<u64>>,
}

impl PartialAccumulator {
    pub fn new(expression: &AggregateFunctionExpr, name: &str) -> Result<Self> {
        if selected(expression) {
            Ok(Self::Ordered(OrderedState {
                sums: Vec::new(),
                counts: (expression.fun().name() == "avg").then(Vec::new),
            }))
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
        saved: &[ScalarValue],
        count: usize,
        name: &str,
    ) -> Result<()> {
        match self {
            Self::Ordered(state) => {
                let ScalarValue::Float64(sum) = saved.last().expect("ordered state") else {
                    return Err(df_error(name, "ordered numeric sum must be Float64"));
                };
                state.sums.resize(count, None);
                state.sums[rank] = *sum;
                if let Some(counts) = &mut state.counts {
                    let ScalarValue::UInt64(saved) = saved[0] else {
                        return Err(df_error(name, "ordered AVG count must be UInt64"));
                    };
                    counts.resize(count, 0);
                    counts[rank] = saved.unwrap_or(0);
                }
                Ok(())
            }
            Self::Native(accumulator) => {
                let saved = &saved[0];
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
            Self::Ordered(state) => {
                let values = values[0]
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        DataFusionError::Internal("ordered numeric input must be Float64".into())
                    })?;
                state.sums.resize(count, None);
                if let Some(counts) = &mut state.counts {
                    counts.resize(count, 0);
                }
                for (&rank, value) in indices.iter().zip(values.iter()) {
                    if let Some(value) = value {
                        state.sums[rank] =
                            Some(state.sums[rank].unwrap_or(0.0).add_wrapping(value));
                        if let Some(counts) = &mut state.counts {
                            counts[rank] = counts[rank].checked_add(1).ok_or_else(|| {
                                DataFusionError::Internal("ordered AVG count overflowed".into())
                            })?;
                        }
                    }
                }
                Ok(())
            }
        }
    }

    pub fn state(&mut self, emit_to: EmitTo) -> DataFusionResult<Vec<ArrayRef>> {
        match self {
            Self::Native(accumulator) => accumulator.state(emit_to),
            Self::Ordered(state) => {
                let sums = emit_to.take_needed(&mut state.sums);
                let mut arrays = Vec::new();
                if let Some(counts) = &mut state.counts {
                    let counts = emit_to.take_needed(counts);
                    arrays.push(std::sync::Arc::new(UInt64Array::from(
                        counts
                            .into_iter()
                            .zip(&sums)
                            .map(|(count, sum)| sum.map(|_| count))
                            .collect::<Vec<_>>(),
                    )) as ArrayRef);
                }
                arrays.push(std::sync::Arc::new(Float64Array::from(sums)));
                Ok(arrays)
            }
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Self::Native(accumulator) => accumulator.size(),
            Self::Ordered(state) => {
                size_of::<Self>()
                    + state.sums.capacity() * size_of::<Option<f64>>()
                    + state
                        .counts
                        .as_ref()
                        .map_or(0, |counts| counts.capacity() * size_of::<u64>())
            }
        }
    }
}

pub(super) fn result(state: &[ScalarValue], name: &str) -> Result<ScalarValue> {
    match state {
        [value] => Ok(value.clone()),
        [ScalarValue::UInt64(count), ScalarValue::Float64(sum)] => Ok(ScalarValue::Float64(
            sum.zip(*count).map(|(sum, count)| average(sum, count)),
        )),
        _ => Err(df_error(name, "sequential aggregate state is invalid")),
    }
}

#[allow(clippy::cast_precision_loss)]
fn average(sum: f64, count: u64) -> f64 {
    sum / count as f64
}
