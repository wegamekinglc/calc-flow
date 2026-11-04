use super::{AggregateFunctionExpr, ArrayRef, GroupsAccumulator, Result, ScalarValue, df_error};
use datafusion::{
    arrow::{
        array::{BooleanArray, new_empty_array},
        datatypes::DataType,
    },
    common::Result as DataFusionResult,
    logical_expr::EmitTo,
};

pub(super) fn selected(expression: &AggregateFunctionExpr) -> bool {
    matches!(expression.fun().name(), "sum" | "avg")
        && expression.field().data_type() == &DataType::Float64
}

pub(super) fn integer_sum(expression: &AggregateFunctionExpr) -> bool {
    expression
        .fun()
        .inner()
        .is::<datafusion::functions_aggregate::sum::Sum>()
        && !expression.is_distinct()
        && expression.order_bys().is_empty()
        && matches!(
            expression.field().data_type(),
            DataType::Int64 | DataType::UInt64
        )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MergeKind {
    Reset,
    Numeric,
    IntegerSum,
}

pub(super) struct PartialAccumulator {
    native: Box<dyn GroupsAccumulator>,
    merge_kind: MergeKind,
    ordered_last: bool,
    ignore_nulls: bool,
}

impl PartialAccumulator {
    pub fn new(expression: &AggregateFunctionExpr, name: &str) -> Result<Self> {
        Ok(Self {
            native: expression
                .create_groups_accumulator()
                .map_err(|error| df_error(name, error))?,
            merge_kind: if integer_sum(expression) {
                MergeKind::IntegerSum
            } else if selected(expression) {
                MergeKind::Numeric
            } else {
                MergeKind::Reset
            },
            ordered_last: super::ordered_last::selected(expression),
            ignore_nulls: expression.ignore_nulls(),
        })
    }

    pub fn seed_batch<'a>(
        &mut self,
        saved: impl Clone + Iterator<Item = (usize, &'a [ScalarValue])>,
        count: usize,
        name: &str,
    ) -> Result<()> {
        let ordered_last = self.ordered_last;
        let saved = saved.filter(|(_, state)| seed_present(state, ordered_last));
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
        if self.ordered_last {
            return self
                .native
                .merge_batch(&values, &ranks, count)
                .map_err(|error| df_error(name, error));
        }
        // Sparse merges must track unseen groups even when all saved values are valid.
        let empty = new_empty_array(&first.last().expect("sequential state").data_type());
        self.native
            .update_batch(
                &[empty],
                &[],
                Some(&BooleanArray::from(Vec::<bool>::new())),
                count,
            )
            .map_err(|error| df_error(name, error))?;
        if self.merge_kind == MergeKind::Reset {
            let reset = super::grouped_float::reset(first.last().expect("sequential state"), name)?
                .to_array_of_size(ranks.len())
                .map_err(|error| df_error(name, error))?;
            self.native
                .merge_batch(&[reset], &ranks, count)
                .map_err(|error| df_error(name, error))?;
        }
        self.native
            .merge_batch(&values, &ranks, count)
            .map_err(|error| df_error(name, error))
    }

    pub fn ordered_last(&self) -> bool {
        self.ordered_last
    }
    pub fn integer_sum(&self) -> bool {
        self.merge_kind == MergeKind::IntegerSum
    }
    pub fn ignore_nulls(&self) -> bool {
        self.ignore_nulls
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

    pub fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        indices: &[usize],
        count: usize,
    ) -> DataFusionResult<()> {
        if self.integer_sum()
            && let Some(first) = values.first()
        {
            let empty = new_empty_array(first.data_type());
            self.native.update_batch(
                &[empty],
                &[],
                Some(&BooleanArray::from(Vec::<bool>::new())),
                count,
            )?;
        }
        self.native.merge_batch(values, indices, count)
    }

    pub fn size(&self) -> usize {
        self.native.size()
    }
}

fn seed_present(state: &[ScalarValue], ordered_last: bool) -> bool {
    if ordered_last {
        !state.is_empty()
    } else {
        state.last().is_some_and(|value| !value.is_null())
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
