//! One complete history per invocation; compiled operators retain no candidates.
use super::{
    StreamAsofJoinOperator,
    kernel::{FiniteContext, KernelContext},
};
use crate::{
    Batch, BatchMetadata, BatchOperator, BatchOperatorContext, CalcFlowError, Result,
    StreamCollector,
};
use async_trait::async_trait;
use datafusion::arrow::record_batch::RecordBatch;
use std::collections::BTreeMap;

#[derive(Default)]
struct Output(Vec<RecordBatch>);
#[async_trait]
impl StreamCollector for Output {
    async fn emit(&mut self, _port: &str, batch: Batch) -> Result<()> {
        self.0.extend_from_slice(batch.table_payload()?.batches());
        Ok(())
    }
}

#[async_trait]
impl BatchOperator for StreamAsofJoinOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        context.run.check_cancelled()?;
        let mut scratch = Self::new(
            &self.name,
            self.schemas[0].clone(),
            self.schemas[1].clone(),
            self.spec.clone(),
        )?;
        scratch.finite_run(inputs, context.run).await
    }
}
impl StreamAsofJoinOperator {
    async fn finite_run(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        run: &crate::RunContext,
    ) -> Result<BTreeMap<String, Batch>> {
        let context = FiniteContext::new(run, self.spec.limits().max_state_bytes());
        self.finite_admission(inputs, &context).await?;
        // Both complete histories are admitted before any prefix can be finalized.
        self.status.left.ended = true;
        self.status.right.ended = true;
        let mut output = Output::default();
        self.finalize(None, true, &context, &mut output).await?;
        run.check_cancelled()?;
        self.finite_batch(output)
    }
    async fn finite_admission(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &dyn KernelContext,
    ) -> Result<()> {
        let left = required(inputs, "left")?;
        let right = required(inputs, "right")?;
        self.inputs[0].validate(left, "left")?;
        self.inputs[1].validate(right, "right")?;
        self.admit_batch("left", left, context).await?;
        self.admit_batch("right", right, context).await
    }
    fn finite_batch(&self, mut output: Output) -> Result<BTreeMap<String, Batch>> {
        if output.0.is_empty() {
            output
                .0
                .push(RecordBatch::new_empty(self.schemas[2].clone()));
        }
        let metadata = BatchMetadata::new(&self.name, 0, BTreeMap::new())?;
        Ok(BTreeMap::from([(
            "output".into(),
            Batch::table(output.0, metadata)?,
        )]))
    }
}

fn required<'a>(inputs: &'a BTreeMap<String, Batch>, side: &str) -> Result<&'a Batch> {
    inputs
        .get(side)
        .ok_or_else(|| CalcFlowError::InvalidArgument {
            field: format!("inputs.{side}"),
            message: "required ASOF input is missing".into(),
        })
}
