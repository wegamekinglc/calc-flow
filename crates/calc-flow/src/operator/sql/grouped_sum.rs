use super::{AggregateFunctionExpr, ArrayRef, GroupsAccumulator, Result, ScalarValue, df_error};
use std::mem::size_of;

use datafusion::{
    arrow::{
        array::{Array, BooleanArray, new_empty_array},
        datatypes::DataType,
    },
    common::{DataFusionError, Result as DataFusionResult},
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

pub(super) fn boolean_extrema(expression: &AggregateFunctionExpr) -> bool {
    matches!(expression.fun().name(), "min" | "max")
        && expression.field().data_type() == &DataType::Boolean
        && !expression.is_distinct()
        && expression.order_bys().is_empty()
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
        let native: Box<dyn GroupsAccumulator> = if boolean_extrema(expression) {
            Box::new(BooleanExtrema::new(expression.fun().name() == "max"))
        } else {
            expression
                .create_groups_accumulator()
                .map_err(|error| df_error(name, error))?
        };
        Ok(Self {
            native,
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

    pub fn seed_batch<S: AsRef<[ScalarValue]>>(
        &mut self,
        saved: impl Clone + Iterator<Item = (usize, S)>,
        count: usize,
        name: &str,
    ) -> Result<()> {
        let ordered_last = self.ordered_last;
        let saved = saved.filter(|(_, state)| seed_present(state.as_ref(), ordered_last));
        let Some((_, first_state)) = saved.clone().next() else {
            return Ok(());
        };
        let first = first_state.as_ref();
        let ranks = saved.clone().map(|(rank, _)| rank).collect::<Vec<_>>();
        let values = (0..first.len())
            .map(|field| {
                ScalarValue::iter_to_array(
                    saved
                        .clone()
                        .map(|(_, state)| state.as_ref()[field].clone()),
                )
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

struct BooleanExtrema {
    values: Vec<Option<bool>>,
    maximum: bool,
}

impl BooleanExtrema {
    fn new(maximum: bool) -> Self {
        Self {
            values: Vec::new(),
            maximum,
        }
    }

    fn take(&mut self, emit_to: EmitTo) -> Vec<Option<bool>> {
        match emit_to {
            EmitTo::All => std::mem::take(&mut self.values),
            EmitTo::First(groups) => self.values.drain(..groups).collect(),
        }
    }
}

impl GroupsAccumulator for BooleanExtrema {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> DataFusionResult<()> {
        let [values] = values else {
            return Err(DataFusionError::Plan(
                "Boolean extrema expects one input array".into(),
            ));
        };
        let values = values
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| DataFusionError::Plan("Boolean extrema input differs".into()))?;
        self.values.resize(total_num_groups, None);
        for (row, group) in group_indices.iter().enumerate() {
            if opt_filter.is_some_and(|filter| filter.is_null(row) || !filter.value(row))
                || values.is_null(row)
            {
                continue;
            }
            let incoming = values.value(row);
            let current = &mut self.values[*group];
            *current = Some(match *current {
                None => incoming,
                Some(current) if self.maximum => current.max(incoming),
                Some(current) => current.min(incoming),
            });
        }
        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> DataFusionResult<ArrayRef> {
        Ok(std::sync::Arc::new(BooleanArray::from(self.take(emit_to))))
    }

    fn state(&mut self, emit_to: EmitTo) -> DataFusionResult<Vec<ArrayRef>> {
        self.evaluate(emit_to).map(|array| vec![array])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        total_num_groups: usize,
    ) -> DataFusionResult<()> {
        self.update_batch(values, group_indices, None, total_num_groups)
    }

    fn size(&self) -> usize {
        size_of::<Self>() + self.values.capacity() * size_of::<Option<bool>>()
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
