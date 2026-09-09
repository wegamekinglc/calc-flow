use super::{
    StreamAsofJoinOperator,
    admission::{ValidatedInput, times},
    identity_compare,
};
use crate::{Batch, Result, StreamOperatorContext, StreamingFailureReason};
use datafusion::arrow::record_batch::RecordBatch;

impl StreamAsofJoinOperator {
    pub(super) async fn validate_duplicates_without_workspace(
        &mut self,
        batch: &Batch,
        input: ValidatedInput,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let side = if input.index == 0 {
            self.spec.left()
        } else {
            self.spec.right()
        };
        let batches = batch.table_payload()?.batches();
        let mut duplicates = 0;
        for (batch_index, record) in batches.iter().enumerate() {
            for row in 0..record.num_rows() {
                context.check_cancelled()?;
                let time = times(record, side).value(row);
                if input.watermark.is_some_and(|wm| time < wm) {
                    continue;
                }
                let mut exists = self.existing_identity(record, row, input.index, context)?;
                for (previous_index, previous) in batches[..=batch_index].iter().enumerate() {
                    if exists {
                        break;
                    }
                    let end = if previous_index == batch_index {
                        row
                    } else {
                        previous.num_rows()
                    };
                    for prior in 0..end {
                        context.check_cancelled()?;
                        if times(previous, side).value(prior) == time
                            && identity_compare::equal(
                                (record, row),
                                (previous, prior),
                                side.keys(),
                            )
                            && identity_compare::equal(
                                (record, row),
                                (previous, prior),
                                side.sequence_by(),
                            )
                        {
                            exists = true;
                            break;
                        }
                    }
                }
                duplicates += u64::from(exists);
                tokio::task::yield_now().await;
            }
        }
        let status = if input.index == 0 {
            &mut self.status.left
        } else {
            &mut self.status.right
        };
        status.duplicate_rows = super::checked(&self.name, status.duplicate_rows, duplicates)?;
        if duplicates == 0 {
            return Ok(());
        }
        Err(super::reason(
            &self.name,
            StreamingFailureReason::AsofDuplicateIdentity,
            "input contains a duplicate key/event-time/sequence identity",
        ))
    }

    fn existing_identity(
        &self,
        batch: &RecordBatch,
        row: usize,
        index: usize,
        context: &StreamOperatorContext<'_>,
    ) -> Result<bool> {
        let side = if index == 0 {
            self.spec.left()
        } else {
            self.spec.right()
        };
        let time = times(batch, side).value(row);
        if index == 0 {
            for (existing, key, sequence) in self.state.left.keys() {
                context.check_cancelled()?;
                if *existing == time
                    && identity_compare::encoded_equal(batch, row, side.keys(), key)
                    && identity_compare::encoded_equal(batch, row, side.sequence_by(), sequence)
                {
                    return Ok(true);
                }
            }
        } else {
            for (key, bucket) in &self.state.right {
                context.check_cancelled()?;
                if identity_compare::encoded_equal(batch, row, side.keys(), key) {
                    for (existing, sequence) in bucket.keys() {
                        context.check_cancelled()?;
                        if *existing == time
                            && identity_compare::encoded_equal(
                                batch,
                                row,
                                side.sequence_by(),
                                sequence,
                            )
                        {
                            return Ok(true);
                        }
                    }
                }
            }
        }
        Ok(false)
    }
}
