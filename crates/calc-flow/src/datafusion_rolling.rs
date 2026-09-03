//! Fail-closed `DataFusion` physical planning for Calc Flow rolling kernels.

use std::{fmt, sync::Arc};

use async_trait::async_trait;
use datafusion::{
    arrow::{datatypes::SchemaRef, record_batch::RecordBatch},
    common::{
        DataFusionError, Result as DataFusionResult, ScalarValue,
        tree_node::{Transformed, TreeNode},
    },
    execution::{
        TaskContext,
        context::{QueryPlanner, SessionState},
        memory_pool::{MemoryConsumer, MemoryReservation},
    },
    logical_expr::{
        Expr, LogicalPlan, WindowFrameBound, WindowFrameUnits, WindowFunctionDefinition,
    },
    physical_expr::{
        Distribution, EquivalenceProperties, OrderingRequirements, expressions::Column,
        window::SlidingAggregateWindowExpr,
    },
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, InputOrderMode,
        PlanProperties, SendableRecordBatchStream, Statistics,
        metrics::{
            BaselineMetrics, Count, ExecutionPlanMetricsSet, Gauge, MetricBuilder, MetricsSet,
        },
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

#[derive(Clone, Debug, Default)]
pub(crate) struct RollingRewriteAuditSnapshot {
    pub candidate_windows: usize,
    pub rewritten_windows: usize,
    pub fallback_reasons: Vec<String>,
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
        session_state: &SessionState,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let eligibility = inspect_logical_windows(logical_plan);
        let planner = DefaultPhysicalPlanner::default();
        let plan = planner
            .create_physical_plan(logical_plan, session_state)
            .await?;
        if eligibility.candidate_windows == 0 || !eligibility.fallback_reasons.is_empty() {
            self.audit.replace(eligibility);
            return Ok(plan);
        }

        let mut rewritten_windows = 0;
        let transformed = plan.transform_up(|node| {
            let Some(window) = node.downcast_ref::<BoundedWindowAggExec>() else {
                return Ok(Transformed::no(node));
            };
            let Some(exec) = CalcFlowRollingExec::try_from_window(window)? else {
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
            audit.candidate_windows += 1;
            if let Err(reason) = inspect_logical_window(expression) {
                audit.fallback_reasons.push(reason.to_owned());
            }
        }
    }
    for input in plan.inputs() {
        inspect_logical_node(input, audit);
    }
}

fn inspect_logical_window(expression: &Expr) -> Result<(), &'static str> {
    let expression = match expression {
        Expr::Alias(alias) => alias.expr.as_ref(),
        expression => expression,
    };
    let Expr::WindowFunction(window) = expression else {
        return Err("logical_window_is_not_a_function");
    };
    let WindowFunctionDefinition::AggregateUDF(function) = &window.fun else {
        return Err("window_function_is_not_an_aggregate");
    };
    if !function.name().eq_ignore_ascii_case("avg") {
        return Err("window_aggregate_is_not_avg");
    }
    let parameters = &window.params;
    if parameters.args.len() != 1 || !matches!(parameters.args[0], Expr::Column(_)) {
        return Err("avg_argument_is_not_one_column");
    }
    if parameters.partition_by.is_empty()
        || parameters
            .partition_by
            .iter()
            .any(|expression| !matches!(expression, Expr::Column(_)))
    {
        return Err("partition_keys_are_not_simple_columns");
    }
    if parameters.order_by.is_empty()
        || parameters
            .order_by
            .iter()
            .any(|sort| !sort.asc || !matches!(sort.expr, Expr::Column(_)))
    {
        return Err("ordering_is_not_simple_ascending_columns");
    }
    if parameters.filter.is_some() || parameters.distinct || parameters.null_treatment.is_some() {
        return Err("avg_filter_distinct_or_null_treatment_is_not_supported");
    }
    if parameters.window_frame.units != WindowFrameUnits::Rows
        || !matches!(
            parameters.window_frame.end_bound,
            WindowFrameBound::CurrentRow
        )
        || bounded_preceding_rows(&parameters.window_frame.start_bound).is_none()
    {
        return Err("window_frame_is_not_bounded_rows_to_current_row");
    }
    Ok(())
}

fn bounded_preceding_rows(bound: &WindowFrameBound) -> Option<u64> {
    match bound {
        WindowFrameBound::Preceding(ScalarValue::UInt64(Some(rows))) => rows.checked_add(1),
        WindowFrameBound::CurrentRow => Some(1),
        _ => None,
    }
}

#[derive(Debug)]
struct CalcFlowRollingExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    kernel: DataFusionRollingKernel,
    window_count: usize,
    required_distribution: Distribution,
    required_ordering: Option<OrderingRequirements>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl CalcFlowRollingExec {
    fn try_from_window(window: &BoundedWindowAggExec) -> DataFusionResult<Option<Self>> {
        if !matches!(window.input_order_mode, InputOrderMode::Sorted) {
            return Ok(None);
        }
        let Some((partition_indices, order_indices, windows)) = physical_windows(window)? else {
            return Ok(None);
        };
        let input = Arc::clone(window.input());
        let Some(kernel) = DataFusionRollingKernel::compile(
            input.schema().as_ref(),
            &partition_indices,
            &order_indices,
            &windows,
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
            .required_input_distribution()
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
        if physical_columns(expression.partition_by()).as_ref() != Some(&partition_indices)
            || physical_order_columns(expression.order_by()).as_ref() != Some(&order_indices)
        {
            return Ok(None);
        }
        let Some(sliding) = expression
            .as_any()
            .downcast_ref::<SlidingAggregateWindowExpr>()
        else {
            return Ok(None);
        };
        let aggregate = sliding.get_aggregate_expr();
        if !aggregate.fun().name().eq_ignore_ascii_case("avg")
            || aggregate.is_distinct()
            || aggregate.ignore_nulls()
            || !aggregate.order_bys().is_empty()
        {
            return Ok(None);
        }
        let arguments = expression.expressions();
        let [argument] = arguments.as_slice() else {
            return Ok(None);
        };
        let Some(argument) = argument.downcast_ref::<Column>() else {
            return Ok(None);
        };
        let Some(rows) = bounded_preceding_rows(&expression.get_window_frame().start_bound) else {
            return Ok(None);
        };
        if expression.get_window_frame().units != WindowFrameUnits::Rows
            || !matches!(
                expression.get_window_frame().end_bound,
                WindowFrameBound::CurrentRow
            )
        {
            return Ok(None);
        }
        let field = expression.field()?;
        windows.push(DataFusionRollingWindow {
            input_index: argument.index(),
            output_name: field.name().to_owned(),
            rows,
        });
    }
    Ok(Some((partition_indices, order_indices, windows)))
}

fn physical_columns(
    expressions: &[Arc<dyn datafusion::physical_expr::PhysicalExpr>],
) -> Option<Vec<usize>> {
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
                "CalcFlowRollingExec: windows={}, kernel={}, state_bytes_per_entity={}",
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

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![self.required_distribution.clone()]
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
        let reservation = MemoryConsumer::new(format!("CalcFlowRollingExec[{partition}]"))
            .register(context.memory_pool());
        let state = RollingStream {
            input,
            schema: Arc::clone(&self.schema),
            kernel: self.kernel.clone(),
            state: DataFusionRollingState::default(),
            reservation,
            metrics: RollingPartitionMetrics::new(&self.metrics, partition),
        };
        let schema = Arc::clone(&self.schema);
        let output = stream::try_unfold(state, |mut stream| async move {
            let Some(batch) = stream.input.next().await.transpose()? else {
                stream.metrics.baseline.done();
                return Ok(None);
            };
            let timer = stream.metrics.baseline.elapsed_compute().timer();
            let transition = stream
                .kernel
                .update_and_fill(&stream.state, &batch)
                .map_err(|error| calc_flow_execution_error(&error))?;
            stream
                .reservation
                .try_resize(transition.metrics.state_bytes)?;
            stream.metrics.record(transition.metrics);
            stream.state = transition.state;
            let mut columns = batch.columns().to_vec();
            columns.extend(transition.columns);
            let output = RecordBatch::try_new(Arc::clone(&stream.schema), columns)?;
            stream.metrics.baseline.record_output(output.num_rows());
            timer.done();
            Ok(Some((output, stream)))
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        Ok(Arc::new(Statistics::new_unknown(&self.schema)))
    }
}

struct RollingStream {
    input: SendableRecordBatchStream,
    schema: SchemaRef,
    kernel: DataFusionRollingKernel,
    state: DataFusionRollingState,
    reservation: MemoryReservation,
    metrics: RollingPartitionMetrics,
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
