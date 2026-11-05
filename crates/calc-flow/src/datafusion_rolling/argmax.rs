use std::{
    cmp::Ordering,
    collections::VecDeque,
    mem::{size_of, size_of_val},
    sync::Arc,
};

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::{
    arrow::{
        array::{Array, ArrayRef, Float64Array},
        datatypes::DataType,
    },
    common::{DataFusionError, ScalarValue},
    functions_aggregate::{array_agg::ArrayAgg, count::Count as CountAggregate, min_max::Max},
    logical_expr::{
        Accumulator, AggregateUDF, AggregateUDFImpl, Expr, LogicalPlan, Operator, Projection,
        Signature, Volatility, Window, WindowFrameUnits, WindowFunctionDefinition,
        expr::{WindowFunction, WindowFunctionParams},
        function::AccumulatorArgs,
    },
};
use datafusion_functions_nested::position::ArrayPosition;

#[derive(Debug, PartialEq, Eq, Hash)]
struct SqlArgmax {
    signature: Signature,
}

impl SqlArgmax {
    fn new() -> Self {
        Self {
            signature: Signature::uniform(1, vec![DataType::Float64], Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for SqlArgmax {
    fn name(&self) -> &'static str {
        "calc_flow_sql_argmax"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> datafusion::common::Result<DataType> {
        if arg_types != [DataType::Float64] {
            return Err(DataFusionError::Plan(
                "calc_flow_sql_argmax requires one Float64 input".to_owned(),
            ));
        }
        Ok(DataType::Float64)
    }

    fn accumulator(
        &self,
        _args: AccumulatorArgs,
    ) -> datafusion::common::Result<Box<dyn Accumulator>> {
        Ok(Box::new(SqlArgmaxAccumulator::default()))
    }

    fn create_sliding_accumulator(
        &self,
        _args: AccumulatorArgs,
    ) -> datafusion::common::Result<Box<dyn Accumulator>> {
        Ok(Box::new(SqlArgmaxAccumulator::default()))
    }
}

#[derive(Debug, Default)]
struct SqlArgmaxAccumulator {
    next_row: u64,
    retracted_rows: u64,
    maxima: VecDeque<(u64, f64)>,
}

impl SqlArgmaxAccumulator {
    #[expect(clippy::cast_precision_loss)]
    fn current(&self) -> ScalarValue {
        if let Some((row, _)) = self.maxima.front() {
            return ScalarValue::Float64(Some((self.next_row - 1 - row) as f64));
        }
        if self.next_row == self.retracted_rows {
            return ScalarValue::Float64(None);
        }
        ScalarValue::Float64(Some((self.next_row - 1 - self.retracted_rows) as f64))
    }
}

fn sql_max_cmp(left: f64, right: f64) -> Ordering {
    left.total_cmp(&right)
}

impl Accumulator for SqlArgmaxAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> datafusion::common::Result<()> {
        let [values] = values else {
            return Err(DataFusionError::Execution(
                "calc_flow_sql_argmax requires one input array".to_owned(),
            ));
        };
        let values = values
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                DataFusionError::Execution("calc_flow_sql_argmax input is not Float64".to_owned())
            })?;
        for row in 0..values.len() {
            if !values.is_null(row) {
                let value = values.value(row);
                while self
                    .maxima
                    .back()
                    .is_some_and(|(_, current)| sql_max_cmp(value, *current) == Ordering::Greater)
                {
                    self.maxima.pop_back();
                }
                self.maxima.push_back((self.next_row, value));
            }
            self.next_row += 1;
        }
        Ok(())
    }

    fn retract_batch(&mut self, values: &[ArrayRef]) -> datafusion::common::Result<()> {
        let [values] = values else {
            return Err(DataFusionError::Execution(
                "calc_flow_sql_argmax requires one input array".to_owned(),
            ));
        };
        for row in 0..values.len() {
            if !values.is_null(row)
                && self
                    .maxima
                    .front()
                    .is_some_and(|(current, _)| *current == self.retracted_rows)
            {
                self.maxima.pop_front();
            }
            self.retracted_rows += 1;
        }
        Ok(())
    }

    fn merge_batch(&mut self, _states: &[ArrayRef]) -> datafusion::common::Result<()> {
        Err(DataFusionError::Execution(
            "calc_flow_sql_argmax only supports ordered window evaluation".to_owned(),
        ))
    }

    fn state(&mut self) -> datafusion::common::Result<Vec<ScalarValue>> {
        Ok(vec![self.current()])
    }

    fn evaluate(&mut self) -> datafusion::common::Result<ScalarValue> {
        Ok(self.current())
    }

    fn supports_retract_batch(&self) -> bool {
        true
    }

    fn size(&self) -> usize {
        size_of_val(self) + self.maxima.capacity() * size_of::<(u64, f64)>()
    }
}

struct ArgmaxShape {
    input: datafusion::common::Column,
    parameters: WindowFunctionParams,
    sources: [Expr; 3],
}

pub(super) fn rewrite(plan: &LogicalPlan) -> datafusion::common::Result<(LogicalPlan, usize)> {
    let mut rewritten = 0;
    let transformed = plan.clone().transform_up(|node| {
        let Some(next) = rewrite_projection(&node)? else {
            return Ok(Transformed::no(node));
        };
        rewritten += 1;
        Ok(Transformed::yes(next))
    })?;
    Ok((transformed.data, rewritten))
}

pub(super) fn is_rewritten(expression: &Expr) -> bool {
    let expression = match expression {
        Expr::Alias(alias) => alias.expr.as_ref(),
        expression => expression,
    };
    let Expr::WindowFunction(window) = expression else {
        return false;
    };
    let WindowFunctionDefinition::AggregateUDF(function) = &window.fun else {
        return false;
    };
    function.inner().downcast_ref::<SqlArgmax>().is_some()
}

fn rewrite_projection(node: &LogicalPlan) -> datafusion::common::Result<Option<LogicalPlan>> {
    let LogicalPlan::Projection(projection) = node else {
        return Ok(None);
    };
    let LogicalPlan::Window(window) = projection.input.as_ref() else {
        return Ok(None);
    };
    let Some((index, shape)) = single_argmax(projection, window) else {
        return Ok(None);
    };
    if !float_input(window, &shape.input) {
        return Ok(None);
    }
    let next_window = argmax_window(window, shape.parameters)?;
    let Some(expressions) = projection_expressions(projection, window, &next_window, index) else {
        return Ok(None);
    };
    Ok(Some(LogicalPlan::Projection(
        Projection::try_new_with_schema(
            expressions,
            Arc::new(next_window),
            Arc::clone(&projection.schema),
        )?,
    )))
}

fn single_argmax(projection: &Projection, window: &Window) -> Option<(usize, ArgmaxShape)> {
    let mut matches = projection
        .expr
        .iter()
        .enumerate()
        .filter_map(|(index, expression)| {
            argmax_shape(expression, window).map(|shape| (index, shape))
        });
    let (index, shape) = matches.next()?;
    if matches.next().is_some() || !exact_window_sources(window, &shape.sources) {
        return None;
    }
    Some((index, shape))
}

fn exact_window_sources(window: &Window, sources: &[Expr; 3]) -> bool {
    window.window_expr.len() == sources.len()
        && sources
            .iter()
            .all(|source| window.window_expr.contains(source))
}

fn float_input(window: &Window, input: &datafusion::common::Column) -> bool {
    window
        .input
        .schema()
        .index_of_column(input)
        .ok()
        .map(|input| window.input.schema().field(input).data_type())
        == Some(&DataType::Float64)
}

fn argmax_window(
    window: &Window,
    parameters: WindowFunctionParams,
) -> datafusion::common::Result<LogicalPlan> {
    let replacement = Expr::WindowFunction(Box::new(WindowFunction {
        fun: WindowFunctionDefinition::AggregateUDF(Arc::new(AggregateUDF::from(SqlArgmax::new()))),
        params: parameters,
    }));
    Ok(LogicalPlan::Window(Window::try_new(
        vec![replacement],
        Arc::clone(&window.input),
    )?))
}

fn projection_expressions(
    projection: &Projection,
    window: &Window,
    next_window: &LogicalPlan,
    index: usize,
) -> Option<Vec<Expr>> {
    let output_column = Expr::Column(datafusion::common::Column::from(
        next_window
            .schema()
            .qualified_field(window.input.schema().fields().len()),
    ));
    let mut expressions = projection.expr.clone();
    let (qualifier, _) = projection.schema.qualified_field(index);
    expressions[index] =
        output_column.alias_qualified(qualifier.cloned(), projection.schema.field(index).name());
    if expressions.iter().enumerate().any(|(other, expression)| {
        other != index
            && !matches!(expression, Expr::Column(column) if window.input.schema().has_column(column))
    }) {
        return None;
    }
    Some(expressions)
}

fn argmax_shape(expression: &Expr, window: &Window) -> Option<ArgmaxShape> {
    let (count_reference, position_expression) = outer_age(expression)?;
    let count = resolve_window(count_reference, window)?;
    let (array, maximum) = position_windows(position_expression, window)?;
    if !valid_aggregates(&count, &array, &maximum) {
        return None;
    }
    bounded_ordered_frame(&array.params)?;
    let input = shared_input(&array, &maximum)?;
    let mut parameters = array.params.clone();
    parameters.args = vec![Expr::Column(input.clone())];
    Some(ArgmaxShape {
        input,
        parameters,
        sources: [
            Expr::WindowFunction(Box::new(count)),
            Expr::WindowFunction(Box::new(array)),
            Expr::WindowFunction(Box::new(maximum)),
        ],
    })
}

fn outer_age(expression: &Expr) -> Option<(&Expr, &Expr)> {
    let expression = match expression {
        Expr::Alias(alias) => alias.expr.as_ref(),
        expression => expression,
    };
    let Expr::Cast(cast) = expression else {
        return None;
    };
    let Expr::BinaryExpr(binary) = cast.expr.as_ref() else {
        return None;
    };
    if cast.field.data_type() != &DataType::Float64 || binary.op != Operator::Minus {
        return None;
    }
    Some((&binary.left, &binary.right))
}

fn position_windows(
    expression: &Expr,
    window: &Window,
) -> Option<(WindowFunction, WindowFunction)> {
    let Expr::Cast(cast) = expression else {
        return None;
    };
    let Expr::ScalarFunction(position) = cast.expr.as_ref() else {
        return None;
    };
    if cast.field.data_type() != &DataType::Int64 {
        return None;
    }
    position.func.inner().downcast_ref::<ArrayPosition>()?;
    let [array, maximum] = position.args.as_slice() else {
        return None;
    };
    Some((
        resolve_window(array, window)?,
        resolve_window(maximum, window)?,
    ))
}

fn valid_aggregates(
    count: &WindowFunction,
    array: &WindowFunction,
    maximum: &WindowFunction,
) -> bool {
    is_count_all(count)
        && is_aggregate(array)
        && is_max(maximum)
        && same_window(count, array)
        && same_window(array, maximum)
}

fn shared_input(
    array: &WindowFunction,
    maximum: &WindowFunction,
) -> Option<datafusion::common::Column> {
    let [Expr::Column(input)] = array.params.args.as_slice() else {
        return None;
    };
    if maximum.params.args != [Expr::Column(input.clone())] {
        return None;
    }
    Some(input.clone())
}

fn bounded_ordered_frame(parameters: &WindowFunctionParams) -> Option<()> {
    if parameters.window_frame.units != WindowFrameUnits::Rows || parameters.order_by.is_empty() {
        return None;
    }
    if bounded_preceding_rows(&parameters.window_frame.start_bound).is_none()
        || !matches!(
            parameters.window_frame.end_bound,
            datafusion::logical_expr::WindowFrameBound::CurrentRow
        )
    {
        return None;
    }
    Some(())
}

fn resolve_window(expression: &Expr, window: &Window) -> Option<WindowFunction> {
    let expression = match expression {
        Expr::Alias(alias) => alias.expr.as_ref(),
        expression => expression,
    };
    match expression {
        Expr::WindowFunction(function) => Some(function.as_ref().clone()),
        Expr::Column(column) => {
            let input_fields = window.input.schema().fields().len();
            let index = window
                .schema
                .index_of_column(column)
                .ok()?
                .checked_sub(input_fields)?;
            let Expr::WindowFunction(function) = window.window_expr.get(index)? else {
                return None;
            };
            Some(function.as_ref().clone())
        }
        _ => None,
    }
}

fn is_count_all(window: &WindowFunction) -> bool {
    let WindowFunctionDefinition::AggregateUDF(function) = &window.fun else {
        return false;
    };
    function.inner().downcast_ref::<CountAggregate>().is_some()
        && matches!(
            window.params.args.as_slice(),
            [Expr::Literal(ScalarValue::Int64(Some(1)), _)]
        )
        && supported_options(&window.params)
}

fn is_aggregate(window: &WindowFunction) -> bool {
    let WindowFunctionDefinition::AggregateUDF(function) = &window.fun else {
        return false;
    };
    function.inner().downcast_ref::<ArrayAgg>().is_some() && supported_options(&window.params)
}

fn is_max(window: &WindowFunction) -> bool {
    let WindowFunctionDefinition::AggregateUDF(function) = &window.fun else {
        return false;
    };
    function.inner().downcast_ref::<Max>().is_some() && supported_options(&window.params)
}

fn supported_options(parameters: &WindowFunctionParams) -> bool {
    parameters.filter.is_none() && !parameters.distinct && parameters.null_treatment.is_none()
}

fn same_window(left: &WindowFunction, right: &WindowFunction) -> bool {
    let mut left = left.params.clone();
    let mut right = right.params.clone();
    left.args.clear();
    right.args.clear();
    left == right
}

fn bounded_preceding_rows(bound: &datafusion::logical_expr::WindowFrameBound) -> Option<u64> {
    match bound {
        datafusion::logical_expr::WindowFrameBound::Preceding(ScalarValue::UInt64(Some(rows))) => {
            rows.checked_add(1)
        }
        datafusion::logical_expr::WindowFrameBound::CurrentRow => Some(1),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliding_argmax_preserves_oldest_ties_and_null_row_positions() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(3.0),
            Some(5.0),
            Some(5.0),
            None,
            Some(2.0),
            Some(1.0),
            Some(8.0),
        ]));
        let mut accumulator = SqlArgmaxAccumulator::default();
        let mut actual = Vec::new();
        for row in 0..values.len() {
            accumulator.update_batch(&[values.slice(row, 1)]).unwrap();
            if row >= 3 {
                accumulator
                    .retract_batch(&[values.slice(row - 3, 1)])
                    .unwrap();
            }
            actual.push(accumulator.evaluate().unwrap());
        }
        assert_eq!(
            actual,
            [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0].map(|value| ScalarValue::Float64(Some(value)))
        );
    }

    #[test]
    fn sliding_argmax_uses_oldest_null_when_no_nonnull_value_remains() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![None, None]));
        let mut accumulator = SqlArgmaxAccumulator::default();
        assert_eq!(accumulator.evaluate().unwrap(), ScalarValue::Float64(None));
        accumulator
            .update_batch(std::slice::from_ref(&values))
            .unwrap();
        assert_eq!(
            accumulator.evaluate().unwrap(),
            ScalarValue::Float64(Some(1.0))
        );
        accumulator.retract_batch(&[values.slice(0, 1)]).unwrap();
        assert_eq!(
            accumulator.evaluate().unwrap(),
            ScalarValue::Float64(Some(0.0))
        );
        assert_eq!(accumulator.retracted_rows, 1);
    }

    #[test]
    fn sliding_argmax_distinguishes_signed_zero_maxima() {
        let mut accumulator = SqlArgmaxAccumulator::default();
        for (row, value) in [-0.0, 0.0, -1.0].into_iter().enumerate() {
            if row == 2 {
                accumulator
                    .retract_batch(&[Arc::new(Float64Array::from(vec![-0.0]))])
                    .unwrap();
            }
            accumulator
                .update_batch(&[Arc::new(Float64Array::from(vec![value]))])
                .unwrap();
            assert_eq!(
                accumulator.evaluate().unwrap(),
                ScalarValue::Float64(Some([0.0, 0.0, 1.0][row]))
            );
        }
    }

    #[test]
    fn array_position_uses_the_expected_concrete_udf() {
        assert!(
            datafusion_functions_nested::position::array_position_udf()
                .inner()
                .downcast_ref::<ArrayPosition>()
                .is_some()
        );
    }

    #[tokio::test]
    async fn composite_sql_window_is_rewritten() {
        use datafusion::{
            arrow::datatypes::{Field, Schema, TimeUnit},
            datasource::MemTable,
            prelude::SessionContext,
        };

        let context = SessionContext::new();
        context.register_udf(
            datafusion_functions_nested::position::array_position_udf()
                .as_ref()
                .clone(),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("sequence", DataType::Int64, false),
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("price", DataType::Float64, true),
        ]));
        context
            .register_table(
                "input",
                Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![]]).unwrap()),
            )
            .unwrap();
        let query = "SELECT sequence, CAST(COUNT(*) OVER trailing - CAST(array_position(array_agg(price) OVER trailing, MAX(price) OVER trailing) AS BIGINT) AS DOUBLE) AS value FROM input WINDOW trailing AS (PARTITION BY symbol ORDER BY event_time, sequence ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)";
        let plan = context.sql(query).await.unwrap();
        let (rewritten, count) = rewrite(plan.logical_plan()).unwrap();
        assert_eq!(count, 1, "{rewritten:#?}");
        assert_eq!(rewritten.schema().field(1).data_type(), &DataType::Float64);
    }
}
