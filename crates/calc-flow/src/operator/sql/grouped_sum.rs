use super::{AggregateFunctionExpr, ArrayRef, GroupsAccumulator, Result, ScalarValue, df_error};
use datafusion::{
    arrow::{array::BooleanArray, datatypes::DataType},
    common::Result as DataFusionResult,
    logical_expr::EmitTo,
};

pub(super) fn selected(expression: &AggregateFunctionExpr) -> bool {
    matches!(expression.fun().name(), "sum" | "avg")
        && expression.field().data_type() == &DataType::Float64
}

pub(super) struct PartialAccumulator {
    native: Box<dyn GroupsAccumulator>,
    numeric: bool,
}

impl PartialAccumulator {
    pub fn new(expression: &AggregateFunctionExpr, name: &str) -> Result<Self> {
        Ok(Self {
            native: expression
                .create_groups_accumulator()
                .map_err(|error| df_error(name, error))?,
            numeric: selected(expression),
        })
    }

    pub fn seed_batch<'a>(
        &mut self,
        saved: impl Clone + Iterator<Item = (usize, &'a [ScalarValue])>,
        count: usize,
        name: &str,
    ) -> Result<()> {
        let saved = saved.filter(|(_, state)| state.last().is_some_and(|value| !value.is_null()));
        let Some((_, first)) = saved.clone().next() else {
            return Ok(());
        };
        let ranks = saved.clone().map(|(rank, _)| rank).collect::<Vec<_>>();
        let values = (0..first.len())
            .map(|field| {
                ScalarValue::iter_to_array(saved.clone().map(|(_, state)| state[field].clone()))
            })
            .collect::<DataFusionResult<Vec<_>>>()
            .map_err(|error| df_error(name, error))?;
        let filter = BooleanArray::from(vec![true; ranks.len()]);
        if self.numeric {
            self.native
                .merge_batch(&values, &ranks, Some(&filter), count)
                .map_err(|error| df_error(name, error))
        } else {
            let reset = super::grouped_float::reset(first.last().expect("sequential state"), name)?
                .to_array_of_size(ranks.len())
                .map_err(|error| df_error(name, error))?;
            self.native
                .merge_batch(&[reset], &ranks, Some(&filter), count)
                .map_err(|error| df_error(name, error))?;
            self.native
                .merge_batch(&values, &ranks, Some(&filter), count)
                .map_err(|error| df_error(name, error))
        }
    }

    pub fn update_batch(
        &mut self,
        values: &[ArrayRef],
        indices: &[usize],
        filter: Option<&BooleanArray>,
        count: usize,
    ) -> DataFusionResult<()> {
        self.native.update_batch(values, indices, filter, count)
    }

    pub fn state(&mut self, emit_to: EmitTo) -> DataFusionResult<Vec<ArrayRef>> {
        self.native.state(emit_to)
    }

    pub fn size(&self) -> usize {
        self.native.size()
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

#[cfg(test)]
#[path = "grouped_seed_tests.rs"]
mod tests;
