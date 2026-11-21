//! Immutable row-local expressions with fresh per-call filter state.

use std::sync::Arc;

use datafusion::{
    arrow::{compute::filter_record_batch, datatypes::SchemaRef, record_batch::RecordBatch},
    common::{DataFusionError, cast::as_boolean_array},
    physical_expr::{PhysicalExpr, projection::Projector},
    physical_plan::{
        ExecutionPlan, coalesce::LimitedBatchCoalescer, filter::FilterExec,
        projection::ProjectionExec,
    },
};

use crate::{
    Batch, Result,
    datafusion::{
        MAX_SQL_RESULT_BYTES, MAX_SQL_RESULT_ROWS, datafusion_error, push_bounded_sql_batch,
    },
    operator::{StreamOperatorContext, sync_operator::SyncContext},
};

use super::memory_scan;

pub(crate) struct CompiledExpression {
    steps: Vec<Step>,
    schema: SchemaRef,
}

enum Step {
    Columns(Vec<usize>),
    Project(Projector),
    Filter(Filter),
}

struct Filter {
    predicate: Arc<dyn PhysicalExpr>,
    projection: Option<Vec<usize>>,
    schema: SchemaRef,
    batch_size: usize,
}

struct ActiveStep {
    coalescer: Option<LimitedBatchCoalescer>,
}

/// Reusable filter allocations; each evaluation starts with fresh logical
/// coalescer state and releases partial state on both success and failure.
pub(crate) struct CompiledExpressionState {
    steps: Vec<ActiveStep>,
}

#[derive(Default)]
struct Output {
    batches: Vec<RecordBatch>,
    rows: usize,
    bytes: usize,
}

impl CompiledExpression {
    pub(crate) fn try_new(plan: &Arc<dyn ExecutionPlan>) -> Option<Self> {
        let mut steps = Vec::new();
        append_steps(plan.as_ref(), &mut steps)?;
        Some(Self {
            steps,
            schema: plan.schema(),
        })
    }

    pub(crate) fn state(&self) -> CompiledExpressionState {
        CompiledExpressionState {
            steps: self.steps.iter().map(ActiveStep::new).collect(),
        }
    }

    pub(crate) async fn evaluate(
        &self,
        input: &Batch,
        node_id: &str,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Batch> {
        let mut state = self.state();
        let mut output = Output::default();
        for (index, record) in input.table_payload()?.batches().iter().enumerate() {
            if index > 0 {
                tokio::task::yield_now().await;
            }
            push(
                &self.steps,
                &mut state.steps,
                record.clone(),
                &mut output,
                node_id,
                context,
            )?;
        }
        finish(self, &mut state, input, output, node_id, context)
    }

    pub(crate) fn evaluate_sync(
        &self,
        state: &mut CompiledExpressionState,
        input: &Batch,
        node_id: &str,
        context: &SyncContext<'_>,
    ) -> Result<Batch> {
        let result = evaluate_sync(self, state, input, node_id, context);
        state.reset(&self.steps);
        result
    }
}

fn evaluate_sync(
    program: &CompiledExpression,
    state: &mut CompiledExpressionState,
    input: &Batch,
    node_id: &str,
    context: &SyncContext<'_>,
) -> Result<Batch> {
    let mut output = Output::default();
    for record in input.table_payload()?.batches() {
        context.check_cancelled()?;
        push(
            &program.steps,
            &mut state.steps,
            record.clone(),
            &mut output,
            node_id,
            context,
        )?;
    }
    finish(program, state, input, output, node_id, context)
}

fn finish<C: Cancellation>(
    program: &CompiledExpression,
    state: &mut CompiledExpressionState,
    input: &Batch,
    mut output: Output,
    node_id: &str,
    context: &C,
) -> Result<Batch> {
    flush(
        &program.steps,
        &mut state.steps,
        &mut output,
        node_id,
        context,
    )?;
    context.check()?;
    if output.batches.is_empty() {
        output
            .batches
            .push(RecordBatch::new_empty(Arc::clone(&program.schema)));
    }
    Batch::table(output.batches, input.metadata().clone())
}

trait Cancellation {
    fn check(&self) -> Result<()>;
}

impl Cancellation for StreamOperatorContext<'_> {
    fn check(&self) -> Result<()> {
        self.check_cancelled()
    }
}

impl Cancellation for SyncContext<'_> {
    fn check(&self) -> Result<()> {
        self.check_cancelled()
    }
}

fn append_steps(plan: &dyn ExecutionPlan, steps: &mut Vec<Step>) -> Option<()> {
    if plan.fetch().is_some() {
        return None;
    }
    if let Some(scan) = memory_scan(plan) {
        if !scan.sort_information().is_empty() {
            return None;
        }
        if let Some(columns) = scan.projection() {
            steps.push(Step::Columns(columns.clone()));
        }
        return Some(());
    }
    let children = plan.children();
    let [input] = children.as_slice() else {
        return None;
    };
    append_steps(input.as_ref(), steps)?;
    steps.push(compile_step(plan, input.as_ref())?);
    Some(())
}

fn compile_step(plan: &dyn ExecutionPlan, input: &dyn ExecutionPlan) -> Option<Step> {
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        let projector = projection
            .projection_expr()
            .make_projector(&input.schema())
            .ok()?;
        return (projector.output_schema() == &plan.schema()).then_some(Step::Project(projector));
    }
    let filter = plan.downcast_ref::<FilterExec>()?;
    Some(Step::Filter(Filter {
        predicate: Arc::clone(filter.predicate()),
        projection: filter.projection().as_ref().map(|indices| indices.to_vec()),
        schema: plan.schema(),
        batch_size: filter.batch_size(),
    }))
}

impl ActiveStep {
    fn new(step: &Step) -> Self {
        let coalescer = match step {
            Step::Filter(filter) => Some(new_coalescer(filter)),
            _ => None,
        };
        Self { coalescer }
    }
}

impl CompiledExpressionState {
    fn reset(&mut self, steps: &[Step]) {
        for (step, active) in steps.iter().zip(&mut self.steps) {
            if let Step::Filter(filter) = step {
                active.coalescer = Some(new_coalescer(filter));
            }
        }
    }
}

fn new_coalescer(filter: &Filter) -> LimitedBatchCoalescer {
    LimitedBatchCoalescer::new(Arc::clone(&filter.schema), filter.batch_size, None)
}

fn push<C: Cancellation>(
    steps: &[Step],
    active: &mut [ActiveStep],
    record: RecordBatch,
    output: &mut Output,
    node_id: &str,
    context: &C,
) -> Result<()> {
    context.check()?;
    let Some((head, tail)) = steps.split_first() else {
        return push_bounded_sql_batch(
            &mut output.batches,
            &mut output.rows,
            &mut output.bytes,
            record,
            MAX_SQL_RESULT_ROWS,
            MAX_SQL_RESULT_BYTES,
        );
    };
    let (active_head, active_tail) = active.split_first_mut().expect("matching active step");
    match head {
        Step::Columns(indices) => {
            let record = record
                .project(indices)
                .map_err(|error| datafusion_error(Some(node_id), DataFusionError::from(error)))?;
            push(tail, active_tail, record, output, node_id, context)
        }
        Step::Project(projector) => {
            let record = projector
                .project_batch(&record)
                .map_err(|error| datafusion_error(Some(node_id), error))?;
            push(tail, active_tail, record, output, node_id, context)
        }
        Step::Filter(filter) => push_filter(
            filter,
            active_head,
            tail,
            active_tail,
            record,
            output,
            (node_id, context),
        ),
    }
}

fn push_filter<C: Cancellation>(
    filter: &Filter,
    active: &mut ActiveStep,
    tail: &[Step],
    active_tail: &mut [ActiveStep],
    record: RecordBatch,
    output: &mut Output,
    node_context: (&str, &C),
) -> Result<()> {
    let (node_id, context) = node_context;
    let record = filter_record(filter, record, node_id)?;
    let coalescer = active.coalescer.as_mut().expect("filter call state");
    coalescer
        .push_batch(record)
        .map_err(|error| datafusion_error(Some(node_id), error))?;
    drain(coalescer, tail, active_tail, output, node_id, context)
}

fn filter_record(filter: &Filter, record: RecordBatch, node_id: &str) -> Result<RecordBatch> {
    let mask = filter
        .predicate
        .evaluate(&record)
        .and_then(|value| value.into_array(record.num_rows()))
        .map_err(|error| datafusion_error(Some(node_id), error))?;
    let projected = match &filter.projection {
        Some(indices) => record
            .project(indices)
            .map_err(|error| datafusion_error(Some(node_id), DataFusionError::from(error)))?,
        None => record,
    };
    let mask = as_boolean_array(&mask).map_err(|_| {
        datafusion_error(
            Some(node_id),
            DataFusionError::Internal(
                "Cannot create filter_array from non-boolean predicates".into(),
            ),
        )
    })?;
    filter_record_batch(&projected, mask)
        .map_err(|error| datafusion_error(Some(node_id), DataFusionError::from(error)))
}

fn flush<C: Cancellation>(
    steps: &[Step],
    active: &mut [ActiveStep],
    output: &mut Output,
    node_id: &str,
    context: &C,
) -> Result<()> {
    for index in 0..steps.len() {
        let (_, tail_steps) = steps.split_at(index + 1);
        let (head_active, tail_active) = active.split_at_mut(index + 1);
        if let Some(coalescer) = head_active[index].coalescer.as_mut() {
            coalescer
                .finish()
                .map_err(|error| datafusion_error(Some(node_id), error))?;
            drain(coalescer, tail_steps, tail_active, output, node_id, context)?;
        }
    }
    Ok(())
}

fn drain<C: Cancellation>(
    coalescer: &mut LimitedBatchCoalescer,
    tail: &[Step],
    active_tail: &mut [ActiveStep],
    output: &mut Output,
    node_id: &str,
    context: &C,
) -> Result<()> {
    while let Some(record) = coalescer.next_completed_batch() {
        push(tail, active_tail, record, output, node_id, context)?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) async fn check_arrow_error_envelope(context: &StreamOperatorContext<'_>) {
    use datafusion::{
        arrow::{
            datatypes::{DataType, Field, Schema},
            record_batch::RecordBatchOptions,
        },
        datasource::memory::{DataSourceExec, MemorySourceConfig},
        physical_plan::memory::MemoryStream,
    };
    use futures::StreamExt;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        true,
    )]));
    let source = MemorySourceConfig::try_new(&[vec![]], schema.clone(), Some(vec![0])).unwrap();
    let plan: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(source);
    let compiled = CompiledExpression::try_new(&plan).unwrap();
    let record = RecordBatch::try_new_with_options(
        Arc::new(Schema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )
    .unwrap();
    let input = Batch::table(vec![record.clone()], crate::BatchMetadata::default()).unwrap();
    // Deliberately bypass cache schema matching to exercise an invariant error.
    let mut reference = MemoryStream::try_new(vec![record], schema, Some(vec![0])).unwrap();
    let expected = reference.next().await.unwrap().unwrap_err();
    let actual = compiled
        .evaluate(&input, "compiled", context)
        .await
        .unwrap_err();
    assert_eq!(
        actual.to_string(),
        datafusion_error(Some("compiled"), expected).to_string()
    );
}
