//! Fresh finite indicator calculation owned by a bounded computation worker.

use std::sync::Arc;

use tokio::sync::OwnedSemaphorePermit;

use super::{footprint::Footprint, math_requirements::bytes, resources::WorkCredit};
use crate::operator::rolling::{
    BufferedRow, CompiledRollingSpec, RollingOperator, RollingStreamState, ScalarValue, SchemaRef,
    StateBudget, operator_error,
};
use crate::{
    Batch, BatchOperatorContext, CalcFlowError, CancellationToken, Result, array::RollingScope,
};

mod input;
mod order;
mod output;
mod preparation;

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

fn owned_slots<T>(count: usize) -> Result<u64> {
    std::alloc::Layout::array::<T>(count).map_err(|_| bytes::invalid())?;
    bytes::slots::<T>(count)
}

struct Work {
    input: Batch,
    compiled: Arc<CompiledRollingSpec>,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    budget: StateBudget,
    node: Arc<crate::array::RollingWorkerIdentity>,
    run_id: String,
    generation: CancellationToken,
    quote: Footprint,
    scratch: u64,
    credit: WorkCredit,
    guard: OwnedSemaphorePermit,
}

struct Evaluated {
    rows: Vec<BufferedRow>,
    columns: Vec<Vec<Option<ScalarValue>>>,
    _state: RollingStreamState,
}

struct Output {
    record: datafusion::arrow::record_batch::RecordBatch,
    footprint: Footprint,
    residual: u64,
}

struct Prepared {
    _identity: Arc<crate::array::RollingWorkerIdentity>,
    batch: Batch,
    generation: CancellationToken,
    _credit: WorkCredit,
    _guard: OwnedSemaphorePermit,
}

impl RollingOperator {
    pub(in super::super) async fn process_indicator_finite(
        &self,
        input: Batch,
        context: &BatchOperatorContext<'_>,
    ) -> Result<Batch> {
        let node = context.run.node_id().unwrap_or(&self.name);
        let work = self.finite_work(input, context, node)?;
        let prepared = self
            .native_checkpoint_workers()
            .run_rolling_credited(
                RollingScope::Batch(context.run.clone()),
                Arc::clone(&work.node),
                move |check| work.run(check),
            )
            .await
            .map_err(|error| operation_error(error, node))?;
        context.run.check_cancelled()?;
        check_generation(&prepared.generation, context.run.run_id())?;
        Ok(prepared.batch)
    }

    fn finite_work(
        &self,
        input: Batch,
        context: &BatchOperatorContext<'_>,
        node: &str,
    ) -> Result<Work> {
        let resources = self
            .indicator_resources
            .as_ref()
            .expect("indicator resources");
        let guard = self
            .try_native_checkpoint_guard(node)?
            .expect("indicator gate");
        let identity = super::work_identity::reserve(
            resources,
            node,
            context.run.run_id().len(),
            self.state_budget,
        )?;
        Ok(Work {
            input,
            compiled: Arc::clone(&self.compiled),
            input_schema: Arc::clone(self.input_ports[0].schema().expect("compiled input")),
            output_schema: Arc::clone(self.output_ports[0].schema().expect("compiled output")),
            budget: self.state_budget,
            node: identity,
            run_id: context.run.run_id().into(),
            generation: resources.work_control(),
            quote: Footprint::default(),
            scratch: 0,
            credit: resources.begin_work(),
            guard,
        })
    }
}

impl Work {
    fn run(mut self, check: Check<'_>) -> Result<Prepared> {
        let generation = self.generation.clone();
        let run_id = std::mem::take(&mut self.run_id);
        self.run_checked(&mut || {
            check()?;
            check_generation(&generation, &run_id)
        })
    }

    fn run_checked(mut self, check: Check<'_>) -> Result<Prepared> {
        let evaluated = self.evaluate(check)?;
        let output = self.materialize(&evaluated, check)?;
        drop(evaluated);
        self.finish(output, check)
    }

    fn reserve(&mut self, extra: u64) -> Result<()> {
        self.credit.adjust(
            self.quote.clone(),
            bytes::add(self.scratch, extra)?,
            self.budget,
            &self.node,
        )
    }
}

fn check_generation(token: &CancellationToken, run_id: &str) -> Result<()> {
    if token.is_cancelled() {
        return Err(CalcFlowError::Cancelled {
            run_id: run_id.into(),
        });
    }
    Ok(())
}

fn operation_error(error: CalcFlowError, node: &str) -> CalcFlowError {
    match error {
        CalcFlowError::Format { message } => operator_error(node, &message),
        CalcFlowError::InvalidArgument { field, message } => {
            operator_error(node, &format!("{field}: {message}"))
        }
        other => other,
    }
}

#[cfg(test)]
mod tests;
