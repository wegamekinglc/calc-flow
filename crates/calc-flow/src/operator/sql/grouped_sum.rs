use super::{AggregateFunctionExpr, ArrayRef, GroupsAccumulator, Result, ScalarValue, df_error};
use datafusion::{
    arrow::datatypes::DataType, common::Result as DataFusionResult, logical_expr::EmitTo,
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

    pub fn seed(
        &mut self,
        rank: usize,
        saved: &[ScalarValue],
        count: usize,
        name: &str,
    ) -> Result<()> {
        let value = saved.last().expect("sequential state");
        if value.is_null() {
            return Ok(());
        }
        let filter = datafusion::arrow::array::BooleanArray::from(vec![true]);
        if self.numeric {
            let values = saved
                .iter()
                .map(ScalarValue::to_array)
                .collect::<DataFusionResult<Vec<_>>>()
                .map_err(|error| df_error(name, error))?;
            self.native
                .merge_batch(&values, &[rank], Some(&filter), count)
                .map_err(|error| df_error(name, error))
        } else {
            let reset = super::grouped_float::reset(value, name)?
                .to_array()
                .map_err(|error| df_error(name, error))?;
            let saved = value.to_array().map_err(|error| df_error(name, error))?;
            self.native
                .merge_batch(&[reset], &[rank], Some(&filter), count)
                .map_err(|error| df_error(name, error))?;
            self.native
                .merge_batch(&[saved], &[rank], Some(&filter), count)
                .map_err(|error| df_error(name, error))
        }
    }

    pub fn update_batch(
        &mut self,
        values: &[ArrayRef],
        indices: &[usize],
        count: usize,
    ) -> DataFusionResult<()> {
        self.native.update_batch(values, indices, None, count)
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
