//! Sequential delivery keeps column funding alive until the collector accepts.

use super::super::{operator_error, output_gather};
use super::JoinOutput;
use crate::runtime::streaming::gather_work::AttemptCleanup;
use crate::{
    Batch, BatchMetadata, DataFusionRuntime, Result, StreamCollector, StreamOperatorContext,
};
use datafusion::arrow::record_batch::RecordBatch;
use std::{collections::BTreeMap, ops::Range};

impl JoinOutput<'_> {
    pub(in crate::operator::join) async fn emit_ranges(
        &self,
        ranges: Vec<Range<usize>>,
        runtime: &DataFusionRuntime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
        sequence: &mut u64,
        cleanup: &mut Option<AttemptCleanup>,
    ) -> Result<()> {
        for range in ranges {
            context.check_cancelled()?;
            let (record, funding) = self.gather(range, runtime, context, cleanup).await?;
            let message = self.message(record, *sequence, context.output_budget().max_bytes)?;
            output.emit("output", message).await?;
            drop(funding);
            *sequence += 1;
        }
        Ok(())
    }

    async fn gather(
        &self,
        range: Range<usize>,
        runtime: &DataFusionRuntime,
        context: &StreamOperatorContext<'_>,
        cleanup: &mut Option<AttemptCleanup>,
    ) -> Result<(RecordBatch, Vec<output_gather::ColumnResult>)> {
        match output_gather::materialize(self, range.clone(), runtime, context, cleanup).await? {
            Some(gathered) => Ok(gathered),
            None => Ok((self.materialize(range)?, Vec::new())),
        }
    }

    fn message(&self, record: RecordBatch, sequence: u64, limit: usize) -> Result<Batch> {
        let metadata = BatchMetadata::new(self.operator_id, sequence, BTreeMap::new())?;
        let message = Batch::table(vec![record], metadata)?;
        if message.estimated_bytes()? > limit {
            return Err(operator_error(
                self.operator_id,
                "validated Join chunk exceeded its byte budget",
            ));
        }
        Ok(message)
    }
}
