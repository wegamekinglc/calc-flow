use super::{StreamJoinOperator, production};
use crate::{
    Batch, BatchMetadata, BatchOperator, BatchOperatorContext, EdgeCollector, OperatorMetadata,
    Result, RunContext, StreamJobContext, StreamOperator, StreamOperatorContext,
};
use async_trait::async_trait;
use datafusion::arrow::record_batch::RecordBatch;
use std::collections::BTreeMap;

#[async_trait]
impl BatchOperator for StreamJoinOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        self.process_finite(inputs, context.run).await
    }
}

impl StreamJoinOperator {
    pub(crate) async fn process_finite(
        &self,
        inputs: &BTreeMap<String, Batch>,
        run: &RunContext,
    ) -> Result<BTreeMap<String, Batch>> {
        run.check_cancelled()?;
        let mut scratch = Self::new(
            &self.name,
            self.input_schema(0).clone(),
            self.input_schema(1).clone(),
            self.spec.clone(),
        )?;
        scratch.run_core = Some(Box::new(production::RunCore::new(&mut scratch)?));
        let job = StreamJobContext::new(
            1,
            run.run_id(),
            run.settings().clone(),
            run.deadline().copied(),
            run.cancellation().clone(),
        )
        .with_checkpointing(false);
        let name = scratch.name.clone();
        let context = StreamOperatorContext::new(&job, &name, None);
        let mut output = EdgeCollector::new(scratch.output_ports().to_vec());
        for side in ["left", "right"] {
            let input = required(inputs, side, &scratch.name)?;
            scratch
                .process_data(side, input.clone(), &context, &mut output)
                .await?;
        }
        if self.spec.join_type == super::StreamJoinType::LeftAnti {
            scratch.on_end(&context, &mut output).await?;
        }
        run.check_cancelled()?;
        let schema = scratch.output_ports[0]
            .schema()
            .expect("compiled Join output schema");
        let mut records = Vec::new();
        for message in output.drain("output") {
            let batch = message.as_data().expect("finite Join emits data only");
            records.extend(batch.table_payload()?.batches().iter().cloned());
        }
        if records.is_empty() {
            records.push(RecordBatch::new_empty(schema.clone()));
        }
        let metadata = BatchMetadata::new(&scratch.name, 0, BTreeMap::new())?;
        Ok(BTreeMap::from([(
            "output".into(),
            Batch::table(records, metadata)?,
        )]))
    }
}

fn required<'a>(inputs: &'a BTreeMap<String, Batch>, side: &str, name: &str) -> Result<&'a Batch> {
    inputs
        .get(side)
        .ok_or_else(|| super::operator_error(name, &format!("missing required {side} input")))
}
