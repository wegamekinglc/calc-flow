use super::{
    AsofJoinSide, StreamAsofJoinOperator,
    admission::{ValidatedInput, times},
    identity_compare,
};
use crate::{Batch, Result, StreamOperatorContext};
use datafusion::arrow::record_batch::RecordBatch;

type InputRow<'a> = (&'a RecordBatch, usize);

impl StreamAsofJoinOperator {
    pub(super) async fn validate_duplicates_without_workspace(
        &mut self,
        batch: &Batch,
        input: ValidatedInput,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let side = input.side(&self.spec);
        let batches = batch.table_payload()?.batches();
        let mut duplicates = 0;
        for (batch_index, record) in batches.iter().enumerate() {
            for row in 0..record.num_rows() {
                context.check_cancelled()?;
                if input.is_late(times(record, side).value(row)) {
                    continue;
                }
                let exists =
                    self.duplicate_input_identity(batches, (batch_index, row), input, context)?;
                duplicates += u64::from(exists);
                tokio::task::yield_now().await;
            }
        }
        self.record_duplicates(input.index, duplicates)
    }

    fn duplicate_input_identity(
        &self,
        batches: &[RecordBatch],
        current: (usize, usize),
        input: ValidatedInput,
        context: &StreamOperatorContext<'_>,
    ) -> Result<bool> {
        Ok(
            self.existing_identity((&batches[current.0], current.1), input, context)?
                || prior_identity(
                    batches,
                    current.0,
                    current.1,
                    input.side(&self.spec),
                    context,
                )?,
        )
    }

    fn existing_identity(
        &self,
        row: InputRow<'_>,
        input: ValidatedInput,
        context: &StreamOperatorContext<'_>,
    ) -> Result<bool> {
        let side = input.side(&self.spec);
        if input.index == 0 {
            self.existing_left_identity(row, side, context)
        } else {
            self.existing_right_identity(row, side, context)
        }
    }

    fn existing_left_identity(
        &self,
        row: InputRow<'_>,
        side: &AsofJoinSide,
        context: &StreamOperatorContext<'_>,
    ) -> Result<bool> {
        for (time, key, sequence) in self.state.left.keys() {
            context.check_cancelled()?;
            if times(row.0, side).value(row.1) == *time
                && identity_compare::encoded_equal(row.0, row.1, side.keys(), key)
                && identity_compare::encoded_equal(row.0, row.1, side.sequence_by(), sequence)
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn existing_right_identity(
        &self,
        row: InputRow<'_>,
        side: &AsofJoinSide,
        context: &StreamOperatorContext<'_>,
    ) -> Result<bool> {
        for (key, bucket) in &self.state.right {
            context.check_cancelled()?;
            if !identity_compare::encoded_equal(row.0, row.1, side.keys(), key) {
                continue;
            }
            for (time, sequence) in bucket.keys() {
                context.check_cancelled()?;
                if times(row.0, side).value(row.1) == *time
                    && identity_compare::encoded_equal(row.0, row.1, side.sequence_by(), sequence)
                {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

fn prior_identity(
    batches: &[RecordBatch],
    batch_index: usize,
    row: usize,
    side: &AsofJoinSide,
    context: &StreamOperatorContext<'_>,
) -> Result<bool> {
    let current = (&batches[batch_index], row);
    for (index, previous) in batches[..=batch_index].iter().enumerate() {
        let end = if index == batch_index {
            row
        } else {
            previous.num_rows()
        };
        for prior in 0..end {
            context.check_cancelled()?;
            if same_identity(current, (previous, prior), side) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn same_identity(left: InputRow<'_>, right: InputRow<'_>, side: &AsofJoinSide) -> bool {
    times(left.0, side).value(left.1) == times(right.0, side).value(right.1)
        && identity_compare::equal(left, right, side.keys())
        && identity_compare::equal(left, right, side.sequence_by())
}
