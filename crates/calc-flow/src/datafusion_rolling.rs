//! Fail-closed `DataFusion` physical planning for Calc Flow rolling kernels.

use std::{fmt, sync::Arc};

use async_trait::async_trait;
use datafusion::{
    arrow::{datatypes::SchemaRef, record_batch::RecordBatch},
    catalog::Session,
    common::{
        DataFusionError, Result as DataFusionResult, ScalarValue,
        tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    },
    execution::{
        TaskContext,
        context::QueryPlanner,
        memory_pool::{MemoryConsumer, MemoryReservation},
    },
    functions_aggregate::{average::Avg, count::Count as CountAggregate, sum::Sum},
    logical_expr::{
        Expr, LogicalPlan, WindowFrameBound, WindowFrameUnits, WindowFunctionDefinition,
        expr::{WindowFunction, WindowFunctionParams},
    },
    physical_expr::{
        Distribution, EquivalenceProperties, OrderingRequirements, PhysicalExpr,
        expressions::{Column, Literal},
        window::{SlidingAggregateWindowExpr, WindowExpr},
    },
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties,
        InputDistributionRequirements, InputOrderMode, PlanProperties, SendableRecordBatchStream,
        Statistics,
        metrics::{
            BaselineMetrics, Count, ExecutionPlanMetricsSet, Gauge, MetricBuilder, MetricsSet,
        },
        statistics::StatisticsArgs,
        stream::RecordBatchStreamAdapter,
        windows::BoundedWindowAggExec,
    },
    physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner},
};
use futures::{StreamExt, stream};
use parking_lot::Mutex;

use crate::operator::{
    DataFusionRollingKernel, DataFusionRollingMetrics, DataFusionRollingState,
    DataFusionRollingWindow,
};

mod argmax;
#[cfg(test)]
mod shared_rows_tests;
mod sql_aggregate;
mod sql_sum_range;
use sql_aggregate::{SqlRollingKernel, SqlRollingState};
use sql_sum_range::{SqlSumRangeKernel, SqlSumRangeState};

#[derive(Clone, Debug, Default)]
pub(crate) struct RollingRewriteAuditSnapshot {
    pub candidate_windows: usize,
    pub rewritten_windows: usize,
    pub fallback_reasons: Vec<String>,
    contains_count: bool,
}

#[derive(Debug, Default)]
pub(crate) struct RollingRewriteAudit {
    inner: Mutex<RollingRewriteAuditSnapshot>,
}

impl RollingRewriteAudit {
    pub(crate) fn snapshot(&self) -> RollingRewriteAuditSnapshot {
        self.inner.lock().clone()
    }

    fn replace(&self, snapshot: RollingRewriteAuditSnapshot) {
        *self.inner.lock() = snapshot;
    }
}

#[derive(Debug)]
pub(crate) struct CalcFlowQueryPlanner {
    audit: Arc<RollingRewriteAudit>,
}

impl CalcFlowQueryPlanner {
    pub(crate) fn new(audit: Arc<RollingRewriteAudit>) -> Self {
        Self { audit }
    }
}

#[async_trait]
impl QueryPlanner for CalcFlowQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &dyn Session,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let (logical_plan, argmax_windows) = argmax::rewrite(logical_plan)?;
        let mut eligibility = inspect_logical_windows(&logical_plan);
        eligibility.candidate_windows += argmax_windows;
        eligibility.rewritten_windows = argmax_windows;
        let planner = DefaultPhysicalPlanner::default();
        let plan = planner
            .create_physical_plan(&logical_plan, session_state)
            .await?;
        if eligibility.candidate_windows == 0 || !eligibility.fallback_reasons.is_empty() {
            self.audit.replace(eligibility);
            return Ok(plan);
        }

        let original = Arc::clone(&plan);
        let mut rewritten_windows = argmax_windows;
        let transformed = plan.transform_up(|node| {
            let Some(window) = node.downcast_ref::<BoundedWindowAggExec>() else {
                return Ok(Transformed::no(node));
            };
            let Some(exec) =
                CalcFlowRollingExec::try_from_window(window, eligibility.contains_count)?
            else {
                return Ok(Transformed::no(node));
            };
            rewritten_windows += exec.window_count;
            Ok(Transformed::yes(Arc::new(exec) as Arc<dyn ExecutionPlan>))
        })?;
        let mut audit = eligibility;
        audit.rewritten_windows = rewritten_windows;
        if rewritten_windows != audit.candidate_windows {
            audit.fallback_reasons.push(format!(
                "physical_window_shape_not_supported:{}_of_{}",
                rewritten_windows, audit.candidate_windows
            ));
            if audit.contains_count {
                audit.rewritten_windows = argmax_windows;
                self.audit.replace(audit);
                return Ok(original);
            }
        }
        self.audit.replace(audit);
        Ok(transformed.data)
    }
}

fn inspect_logical_windows(plan: &LogicalPlan) -> RollingRewriteAuditSnapshot {
    let mut audit = RollingRewriteAuditSnapshot::default();
    inspect_logical_node(plan, &mut audit);
    audit
}

fn inspect_logical_node(plan: &LogicalPlan, audit: &mut RollingRewriteAuditSnapshot) {
    if let LogicalPlan::Window(window) = plan {
        for expression in &window.window_expr {
            if argmax::is_rewritten(expression) {
                continue;
            }
            audit.candidate_windows += 1;
            audit.contains_count |= logical_window_function(expression)
                .and_then(inspect_logical_aggregate)
                .is_ok_and(|aggregate| aggregate == AggregateKind::Count);
            if let Err(reason) = inspect_logical_window(expression) {
                audit.fallback_reasons.push(reason.to_owned());
            }
        }
    }
    for input in plan.inputs() {
        inspect_logical_node(input, audit);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AggregateKind {
    Average,
    Count,
    Sum,
}

fn inspect_logical_window(expression: &Expr) -> Result<(), &'static str> {
    let window = logical_window_function(expression)?;
    let aggregate = inspect_logical_aggregate(window)?;
    inspect_logical_parameters(&window.params, aggregate)
}

fn logical_window_function(expression: &Expr) -> Result<&WindowFunction, &'static str> {
    let expression = match expression {
        Expr::Alias(alias) => alias.expr.as_ref(),
        expression => expression,
    };
    let Expr::WindowFunction(window) = expression else {
        return Err("logical_window_is_not_a_function");
    };
    Ok(window)
}

fn inspect_logical_aggregate(window: &WindowFunction) -> Result<AggregateKind, &'static str> {
    let WindowFunctionDefinition::AggregateUDF(function) = &window.fun else {
        return Err("window_function_is_not_an_aggregate");
    };
    if function.inner().downcast_ref::<Avg>().is_some() {
        return Ok(AggregateKind::Average);
    }
    if function.inner().downcast_ref::<CountAggregate>().is_some() {
        return Ok(AggregateKind::Count);
    }
    if function.inner().downcast_ref::<Sum>().is_some() {
        return Ok(AggregateKind::Sum);
    }
    Err("window_aggregate_is_not_avg_count_or_sum")
}

fn inspect_logical_parameters(
    parameters: &WindowFunctionParams,
    aggregate: AggregateKind,
) -> Result<(), &'static str> {
    if !logical_argument_supported(parameters, aggregate) {
        let reason = match aggregate {
            AggregateKind::Average => "avg_argument_is_not_one_column",
            AggregateKind::Count => "count_argument_is_not_one_column",
            AggregateKind::Sum => "sum_argument_is_not_one_column",
        };
        return Err(reason);
    }
    if !logical_partition_supported(parameters) {
        return Err("partition_keys_are_not_simple_columns");
    }
    if !logical_order_supported(parameters) {
        return Err("ordering_is_not_simple_ascending_columns");
    }
    if !logical_options_supported(parameters) {
        let reason = match aggregate {
            AggregateKind::Average => "avg_filter_distinct_or_null_treatment_is_not_supported",
            AggregateKind::Count => "count_filter_distinct_or_null_treatment_is_not_supported",
            AggregateKind::Sum => "sum_filter_distinct_or_null_treatment_is_not_supported",
        };
        return Err(reason);
    }
    if !logical_frame_supported(parameters, aggregate) {
        return Err(if aggregate == AggregateKind::Sum {
            "window_frame_is_not_bounded_range_to_current_row"
        } else {
            "window_frame_is_not_bounded_rows_to_current_row"
        });
    }
    Ok(())
}

fn logical_argument_supported(parameters: &WindowFunctionParams, aggregate: AggregateKind) -> bool {
    match parameters.args.as_slice() {
        [Expr::Column(_)] => true,
        [Expr::Literal(value, _)] if aggregate == AggregateKind::Count => {
            count_literal_supported(value)
        }
        _ => false,
    }
}

fn count_literal_supported(value: &ScalarValue) -> bool {
    matches!(
        value,
        ScalarValue::Int64(Some(1)) | ScalarValue::UInt8(Some(1))
    )
}

fn logical_partition_supported(parameters: &WindowFunctionParams) -> bool {
    !parameters.partition_by.is_empty()
        && parameters
            .partition_by
            .iter()
            .all(|expression| matches!(expression, Expr::Column(_)))
}

fn logical_order_supported(parameters: &WindowFunctionParams) -> bool {
    !parameters.order_by.is_empty()
        && parameters
            .order_by
            .iter()
            .all(|sort| sort.asc && matches!(sort.expr, Expr::Column(_)))
}

fn logical_options_supported(parameters: &WindowFunctionParams) -> bool {
    parameters.filter.is_none() && !parameters.distinct && parameters.null_treatment.is_none()
}

fn logical_frame_supported(parameters: &WindowFunctionParams, aggregate: AggregateKind) -> bool {
    if !matches!(
        parameters.window_frame.end_bound,
        WindowFrameBound::CurrentRow
    ) {
        return false;
    }
    if aggregate == AggregateKind::Sum {
        return parameters.window_frame.units == WindowFrameUnits::Range
            && bounded_preceding_range(&parameters.window_frame.start_bound).is_some();
    }
    parameters.window_frame.units == WindowFrameUnits::Rows
        && bounded_preceding_rows(&parameters.window_frame.start_bound).is_some()
}

fn bounded_preceding_rows(bound: &WindowFrameBound) -> Option<u64> {
    match bound {
        WindowFrameBound::Preceding(ScalarValue::UInt64(Some(rows))) => rows.checked_add(1),
        WindowFrameBound::CurrentRow => Some(1),
        _ => None,
    }
}

fn bounded_preceding_range(bound: &WindowFrameBound) -> Option<u64> {
    match bound {
        WindowFrameBound::Preceding(ScalarValue::IntervalMonthDayNano(Some(value))) => {
            month_day_nano_micros(value.months, value.days, value.nanoseconds)
        }
        WindowFrameBound::Preceding(ScalarValue::IntervalDayTime(Some(value))) => {
            day_time_micros(value.days, value.milliseconds)
        }
        WindowFrameBound::Preceding(ScalarValue::DurationMicrosecond(Some(micros))) => {
            positive_duration_micros(*micros)
        }
        _ => None,
    }
}

fn month_day_nano_micros(months: i32, days: i32, nanoseconds: i64) -> Option<u64> {
    if months != 0 || days != 0 {
        return None;
    }
    u64::try_from(nanoseconds)
        .ok()
        .filter(|nanos| *nanos > 0 && nanos % 1_000 == 0)
        .map(|nanos| nanos / 1_000)
}

fn day_time_micros(days: i32, milliseconds: i32) -> Option<u64> {
    if days != 0 {
        return None;
    }
    u64::try_from(milliseconds)
        .ok()
        .filter(|millis| *millis > 0)
        .and_then(|millis| millis.checked_mul(1_000))
}

fn positive_duration_micros(micros: i64) -> Option<u64> {
    u64::try_from(micros).ok().filter(|micros| *micros > 0)
}

#[derive(Clone, Debug)]
enum RollingExecutionKernel {
    Shared(DataFusionRollingKernel),
    Sql(SqlRollingKernel),
    SqlSumRange(SqlSumRangeKernel),
}

enum RollingExecutionState {
    Shared(DataFusionRollingState),
    Sql(SqlRollingState),
    SqlSumRange(SqlSumRangeState),
}

struct RollingExecutionBatch {
    input: RecordBatch,
    columns: Vec<datafusion::arrow::array::ArrayRef>,
    state: RollingExecutionState,
    metrics: DataFusionRollingMetrics,
}

impl RollingExecutionKernel {
    fn compile(
        schema: &datafusion::arrow::datatypes::Schema,
        partitions: &[usize],
        order: &[usize],
        windows: &[DataFusionRollingWindow],
        sql_compatibility: bool,
    ) -> Option<Self> {
        if let Some(kernel) = SqlSumRangeKernel::compile(schema, partitions, order, windows) {
            return Some(Self::SqlSumRange(kernel));
        }
        if sql_compatibility {
            SqlRollingKernel::compile(schema, partitions, order, windows).map(Self::Sql)
        } else {
            DataFusionRollingKernel::compile(schema, partitions, order, windows).map(Self::Shared)
        }
    }

    fn initial_state(&self) -> RollingExecutionState {
        match self {
            Self::Shared(_) => RollingExecutionState::Shared(DataFusionRollingState::default()),
            Self::Sql(_) => RollingExecutionState::Sql(SqlRollingState::default()),
            Self::SqlSumRange(_) => RollingExecutionState::SqlSumRange(SqlSumRangeState::default()),
        }
    }

    fn fingerprint(&self) -> &str {
        match self {
            Self::Shared(kernel) => kernel.fingerprint(),
            Self::Sql(kernel) => kernel.fingerprint(),
            Self::SqlSumRange(kernel) => kernel.fingerprint(),
        }
    }

    fn estimated_state_bytes_per_entity(&self) -> usize {
        match self {
            Self::Shared(kernel) => kernel.estimated_state_bytes_per_entity(),
            Self::Sql(kernel) => kernel.estimated_state_bytes_per_entity(),
            Self::SqlSumRange(_) => SqlSumRangeKernel::estimated_state_bytes_per_entity(),
        }
    }

    fn update_and_fill_owned(
        &self,
        state: RollingExecutionState,
        input: &RecordBatch,
    ) -> crate::Result<RollingExecutionBatch> {
        match (self, state) {
            (Self::Shared(kernel), RollingExecutionState::Shared(state)) => {
                let result = kernel.update_and_fill_owned(state, input)?;
                Ok(RollingExecutionBatch {
                    input: input.clone(),
                    columns: result.columns,
                    state: RollingExecutionState::Shared(result.state),
                    metrics: result.metrics,
                })
            }
            (Self::Sql(kernel), RollingExecutionState::Sql(state)) => {
                kernel.update_and_fill(&state, input)
            }
            (Self::SqlSumRange(kernel), RollingExecutionState::SqlSumRange(state)) => {
                kernel.update_and_fill(state, input)
            }
            _ => Err(crate::CalcFlowError::Internal {
                message: "SQL rolling state does not match the physical kernel".to_owned(),
            }),
        }
    }

    fn finish_owned(
        &self,
        state: RollingExecutionState,
        input: &RecordBatch,
    ) -> crate::Result<Option<RollingExecutionBatch>> {
        match (self, state) {
            (Self::SqlSumRange(kernel), RollingExecutionState::SqlSumRange(state)) => {
                kernel.finish(state, input)
            }
            (Self::Shared(_), RollingExecutionState::Shared(state)) => {
                let _ = state;
                Ok(None)
            }
            (Self::Sql(_), RollingExecutionState::Sql(state)) => {
                let _ = state;
                Ok(None)
            }
            _ => Err(crate::CalcFlowError::Internal {
                message: "SQL rolling state does not match the physical kernel".to_owned(),
            }),
        }
    }
}

#[derive(Debug)]
struct CalcFlowRollingExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    kernel: RollingExecutionKernel,
    window_count: usize,
    required_distribution: Distribution,
    required_ordering: Option<OrderingRequirements>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl CalcFlowRollingExec {
    fn try_from_window(
        window: &BoundedWindowAggExec,
        sql_compatibility: bool,
    ) -> DataFusionResult<Option<Self>> {
        if !matches!(window.input_order_mode, InputOrderMode::Sorted) {
            return Ok(None);
        }
        let Some((partition_indices, order_indices, windows)) = physical_windows(window)? else {
            return Ok(None);
        };
        let input = Arc::clone(window.input());
        let Some(kernel) = RollingExecutionKernel::compile(
            input.schema().as_ref(),
            &partition_indices,
            &order_indices,
            &windows,
            sql_compatibility,
        ) else {
            return Ok(None);
        };
        let schema = window.schema();
        let equivalence = match input.output_ordering() {
            Some(ordering) => {
                EquivalenceProperties::new_with_orderings(Arc::clone(&schema), [ordering.clone()])
            }
            None => EquivalenceProperties::new(Arc::clone(&schema)),
        };
        let properties = PlanProperties::new(
            equivalence,
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        let required_distribution = window
            .input_distribution_requirements()
            .into_per_child()
            .into_iter()
            .next()
            .unwrap_or(Distribution::UnspecifiedDistribution);
        let required_ordering = window
            .required_input_ordering()
            .into_iter()
            .next()
            .flatten();
        Ok(Some(Self {
            input,
            schema,
            kernel,
            window_count: windows.len(),
            required_distribution,
            required_ordering,
            properties: Arc::new(properties),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }
}

type PhysicalRollingWindowPlan = (Vec<usize>, Vec<usize>, Vec<DataFusionRollingWindow>);

fn physical_windows(
    window: &BoundedWindowAggExec,
) -> DataFusionResult<Option<PhysicalRollingWindowPlan>> {
    let Some(first) = window.window_expr().first() else {
        return Ok(None);
    };
    let Some(partition_indices) = physical_columns(first.partition_by()) else {
        return Ok(None);
    };
    let Some(order_indices) = physical_order_columns(first.order_by()) else {
        return Ok(None);
    };
    let mut windows = Vec::with_capacity(window.window_expr().len());
    for expression in window.window_expr() {
        let Some(window) = physical_window(
            expression.as_ref(),
            window,
            &partition_indices,
            &order_indices,
        )?
        else {
            return Ok(None);
        };
        windows.push(window);
    }
    Ok(Some((partition_indices, order_indices, windows)))
}

fn physical_window(
    expression: &dyn WindowExpr,
    parent: &BoundedWindowAggExec,
    partition_indices: &[usize],
    order_indices: &[usize],
) -> DataFusionResult<Option<DataFusionRollingWindow>> {
    if !physical_window_order_matches(expression, partition_indices, order_indices) {
        return Ok(None);
    }
    let Some(sliding) = expression
        .as_any()
        .downcast_ref::<SlidingAggregateWindowExpr>()
    else {
        return Ok(None);
    };
    let Some(aggregate) = physical_aggregate_supported(sliding) else {
        return Ok(None);
    };
    let Some(input_index) = physical_window_input(expression)
        .or_else(|| count_literal_input(expression, parent, aggregate))
    else {
        return Ok(None);
    };
    let (rows, duration_micros) = if aggregate == AggregateKind::Sum {
        (None, physical_window_range(expression))
    } else {
        (physical_window_rows(expression), None)
    };
    if rows.is_none() && duration_micros.is_none() {
        return Ok(None);
    }
    let field = expression.field()?;
    Ok(Some(DataFusionRollingWindow {
        input_index,
        output_name: field.name().to_owned(),
        rows: rows.unwrap_or(0),
        is_count: aggregate == AggregateKind::Count,
        is_sum: aggregate == AggregateKind::Sum,
        duration_micros,
    }))
}

fn physical_window_order_matches(
    expression: &dyn WindowExpr,
    partition_indices: &[usize],
    order_indices: &[usize],
) -> bool {
    physical_columns(expression.partition_by()).as_deref() == Some(partition_indices)
        && physical_order_columns(expression.order_by()).as_deref() == Some(order_indices)
}

fn physical_aggregate_supported(expression: &SlidingAggregateWindowExpr) -> Option<AggregateKind> {
    let aggregate = expression.get_aggregate_expr();
    if aggregate.is_distinct() || aggregate.ignore_nulls() || !aggregate.order_bys().is_empty() {
        return None;
    }
    if aggregate.fun().inner().downcast_ref::<Avg>().is_some() {
        return Some(AggregateKind::Average);
    }
    if aggregate
        .fun()
        .inner()
        .downcast_ref::<CountAggregate>()
        .is_some()
    {
        return Some(AggregateKind::Count);
    }
    aggregate
        .fun()
        .inner()
        .downcast_ref::<Sum>()
        .map(|_| AggregateKind::Sum)
}

fn count_literal_input(
    expression: &dyn WindowExpr,
    parent: &BoundedWindowAggExec,
    aggregate: AggregateKind,
) -> Option<usize> {
    if aggregate != AggregateKind::Count {
        return None;
    }
    let arguments = expression.expressions();
    let [argument] = arguments.as_slice() else {
        return None;
    };
    if !count_literal_supported(argument.downcast_ref::<Literal>()?.value()) {
        return None;
    }
    parent.window_expr().iter().find_map(|candidate| {
        let sliding = candidate
            .as_any()
            .downcast_ref::<SlidingAggregateWindowExpr>()?;
        if physical_aggregate_supported(sliding) != Some(AggregateKind::Average)
            || candidate.get_window_frame() != expression.get_window_frame()
        {
            return None;
        }
        let index = physical_window_input(candidate.as_ref())?;
        (!parent.input().schema().field(index).is_nullable()).then_some(index)
    })
}

fn physical_window_input(expression: &dyn WindowExpr) -> Option<usize> {
    let arguments = expression.expressions();
    let [argument] = arguments.as_slice() else {
        return None;
    };
    argument.downcast_ref::<Column>().map(Column::index)
}

fn physical_window_rows(expression: &dyn WindowExpr) -> Option<u64> {
    let frame = expression.get_window_frame();
    if frame.units != WindowFrameUnits::Rows
        || !matches!(frame.end_bound, WindowFrameBound::CurrentRow)
    {
        return None;
    }
    bounded_preceding_rows(&frame.start_bound)
}

fn physical_window_range(expression: &dyn WindowExpr) -> Option<u64> {
    let frame = expression.get_window_frame();
    if frame.units != WindowFrameUnits::Range
        || !matches!(frame.end_bound, WindowFrameBound::CurrentRow)
    {
        return None;
    }
    bounded_preceding_range(&frame.start_bound)
}

fn physical_columns(expressions: &[Arc<dyn PhysicalExpr>]) -> Option<Vec<usize>> {
    expressions
        .iter()
        .map(|expression| expression.downcast_ref::<Column>().map(Column::index))
        .collect()
}

fn physical_order_columns(
    expressions: &[datafusion::physical_expr::PhysicalSortExpr],
) -> Option<Vec<usize>> {
    if expressions.is_empty()
        || expressions
            .iter()
            .any(|expression| expression.options.descending)
    {
        return None;
    }
    expressions
        .iter()
        .map(|expression| expression.expr.downcast_ref::<Column>().map(Column::index))
        .collect()
}

impl DisplayAs for CalcFlowRollingExec {
    fn fmt_as(&self, format: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match format {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "CalcFlowRollingExec: windows={}, route=sorted_partitions, kernel={}, state_bytes_per_entity={}",
                self.window_count,
                self.kernel.fingerprint(),
                self.kernel.estimated_state_bytes_per_entity()
            ),
            DisplayFormatType::TreeRender => write!(f, "windows={}", self.window_count),
        }
    }
}

impl ExecutionPlan for CalcFlowRollingExec {
    fn name(&self) -> &str {
        Self::static_name()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn input_distribution_requirements(&self) -> InputDistributionRequirements {
        InputDistributionRequirements::new(vec![self.required_distribution.clone()])
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        // The compiled kernel reads Arrow columns directly.
        Ok(TreeNodeRecursion::Continue)
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        vec![self.required_ordering.clone()]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "CalcFlowRollingExec requires exactly one child".to_owned(),
            ));
        }
        let input = children.swap_remove(0);
        if input.schema() != self.input.schema() {
            return Err(DataFusionError::Internal(
                "CalcFlowRollingExec child schema changed".to_owned(),
            ));
        }
        let mut next = Self {
            input,
            schema: Arc::clone(&self.schema),
            kernel: self.kernel.clone(),
            window_count: self.window_count,
            required_distribution: self.required_distribution.clone(),
            required_ordering: self.required_ordering.clone(),
            properties: Arc::clone(&self.properties),
            metrics: ExecutionPlanMetricsSet::new(),
        };
        next.properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&next.schema)),
            next.input.output_partitioning().clone(),
            next.input.pipeline_behavior(),
            next.input.boundedness(),
        ));
        Ok(Arc::new(next))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let input = self.input.execute(partition, Arc::clone(&context))?;
        let input_schema = input.schema();
        let reservation = MemoryConsumer::new(format!("CalcFlowRollingExec[{partition}]"))
            .register(context.memory_pool());
        let state = RollingStream {
            input,
            input_schema,
            schema: Arc::clone(&self.schema),
            kernel: self.kernel.clone(),
            state: self.kernel.initial_state(),
            reservation,
            metrics: RollingPartitionMetrics::new(&self.metrics, partition),
        };
        let schema = Arc::clone(&self.schema);
        let output = stream::try_unfold(state, |mut stream| async move {
            let metrics = stream.metrics.clone();
            let timer = metrics.baseline.elapsed_compute().timer();
            let state = std::mem::replace(&mut stream.state, stream.kernel.initial_state());
            let transition = if let Some(batch) = stream.input.next().await.transpose()? {
                stream
                    .kernel
                    .update_and_fill_owned(state, &batch)
                    .map_err(|error| calc_flow_execution_error(&error))?
            } else {
                let empty = RecordBatch::new_empty(stream.input_schema.clone());
                let Some(transition) = stream
                    .kernel
                    .finish_owned(state, &empty)
                    .map_err(|error| calc_flow_execution_error(&error))?
                else {
                    stream.metrics.baseline.done();
                    return Ok(None);
                };
                transition
            };
            let output = stream.apply(transition)?;
            stream.metrics.baseline.record_output(output.num_rows());
            timer.done();
            Ok(Some((output, stream)))
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics_from_inputs(
        &self,
        _input_stats: &[Arc<Statistics>],
        _args: &StatisticsArgs,
    ) -> DataFusionResult<Arc<Statistics>> {
        Ok(Arc::new(Statistics::new_unknown(&self.schema)))
    }
}

struct RollingStream {
    input: SendableRecordBatchStream,
    input_schema: SchemaRef,
    schema: SchemaRef,
    kernel: RollingExecutionKernel,
    state: RollingExecutionState,
    reservation: MemoryReservation,
    metrics: RollingPartitionMetrics,
}

impl RollingStream {
    fn apply(&mut self, transition: RollingExecutionBatch) -> DataFusionResult<RecordBatch> {
        self.reservation
            .try_resize(transition.metrics.state_bytes)?;
        self.metrics.record(transition.metrics);
        self.state = transition.state;
        let mut columns = transition.input.columns().to_vec();
        columns.extend(transition.columns);
        Ok(RecordBatch::try_new(Arc::clone(&self.schema), columns)?)
    }
}

#[derive(Clone)]
struct RollingPartitionMetrics {
    baseline: BaselineMetrics,
    input_rows: Count,
    input_validation_ns: Count,
    order_proof_ns: Count,
    entity_encode_ns: Count,
    kernel_ns: Count,
    output_build_ns: Count,
    entities: Gauge,
    state_bytes: Gauge,
}

impl RollingPartitionMetrics {
    fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        let counter = |name| MetricBuilder::new(metrics).counter(name, partition);
        let gauge = |name| MetricBuilder::new(metrics).gauge(name, partition);
        Self {
            baseline: BaselineMetrics::new(metrics, partition),
            input_rows: counter("input_rows"),
            input_validation_ns: counter("input_validation_ns"),
            order_proof_ns: counter("order_proof_ns"),
            entity_encode_ns: counter("entity_encode_ns"),
            kernel_ns: counter("kernel_ns"),
            output_build_ns: counter("output_build_ns"),
            entities: gauge("entities"),
            state_bytes: gauge("state_bytes"),
        }
    }

    fn record(&self, metrics: DataFusionRollingMetrics) {
        self.input_rows.add(metrics.input_rows);
        self.input_validation_ns
            .add(saturating_usize(metrics.input_validation_ns));
        self.order_proof_ns
            .add(saturating_usize(metrics.order_proof_ns));
        self.entity_encode_ns
            .add(saturating_usize(metrics.entity_encode_ns));
        self.kernel_ns.add(saturating_usize(metrics.kernel_ns));
        self.output_build_ns
            .add(saturating_usize(metrics.output_build_ns));
        self.entities.set(metrics.entities);
        self.state_bytes.set(metrics.state_bytes);
    }
}

fn saturating_usize(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

fn calc_flow_execution_error(error: &crate::CalcFlowError) -> DataFusionError {
    DataFusionError::Execution(error.to_string())
}
