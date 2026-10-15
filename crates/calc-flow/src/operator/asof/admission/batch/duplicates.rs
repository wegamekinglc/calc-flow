//! Duplicate proofs merge normalized positions with retained identity history.

#[cfg(test)]
use super::WORK;
use super::{
    Columns, InputKeys, Ordering, Result, StreamAsofJoinOperator, StreamOperatorContext,
    check_input_cancellation, state,
};
use datafusion::execution::memory_pool::MemoryReservation;

impl Columns {
    pub(super) fn duplicates(
        &mut self,
        operator: &StreamAsofJoinOperator,
        side: usize,
        keys: &InputKeys,
        ranks: &[u32],
        context: &StreamOperatorContext<'_>,
    ) -> Result<u64> {
        let mut sorted = self.order.clone();
        let compare = |a: &(usize, u32), b: &(usize, u32)| {
            self.records[a.0].compare(a.1 as usize, &self.records[b.0], b.1 as usize, ranks)
        };
        if !sorted.windows(2).all(|p| compare(&p[0], &p[1]).is_le()) {
            sorted.sort_unstable_by(compare);
            #[cfg(test)]
            WORK.with_borrow_mut(|work| work.sorted_rows += sorted.len());
        }
        if side == 1 {
            self.order = sorted;
            self.group_keys(keys, context)?;
            return self.right_duplicates(operator, keys, context);
        }
        self.order = sorted;
        self.left_duplicates(operator, keys, context)
    }

    fn left_duplicates(
        &self,
        operator: &StreamAsofJoinOperator,
        keys: &InputKeys,
        context: &StreamOperatorContext<'_>,
    ) -> Result<u64> {
        let skip_resident = self.order.first().is_none_or(|&first| {
            operator
                .state
                .left
                .last_key_value()
                .is_none_or(|(last, _)| self.compare_left(first, &last, keys).is_gt())
        });
        let Some(_cursor_credit) = left_cursor_credit(operator, skip_resident)? else {
            return self.scalar_left_duplicates(operator, 0, keys, context);
        };
        self.merge_left_duplicates(operator, keys, skip_resident, context)
    }

    fn merge_left_duplicates(
        &self,
        operator: &StreamAsofJoinOperator,
        keys: &InputKeys,
        skip_resident: bool,
        context: &StreamOperatorContext<'_>,
    ) -> Result<u64> {
        let mut resident = (!skip_resident)
            .then(|| operator.state.left.iter())
            .into_iter()
            .flatten()
            .peekable();
        let mut previous = None;
        let mut duplicates = 0;
        let mut ordinal = 0;
        for &position in &self.order {
            check_input_cancellation(ordinal, context)?;
            ordinal += 1;
            while resident
                .peek()
                .is_some_and(|(identity, _)| self.compare_left(position, identity, keys).is_gt())
            {
                check_input_cancellation(ordinal, context)?;
                ordinal += 1;
                resident.next();
            }
            let repeated = previous.is_some_and(|last| self.same_identity(last, position));
            let retained = resident
                .peek()
                .is_some_and(|(identity, _)| self.compare_left(position, identity, keys).is_eq());
            duplicates += u64::from(repeated || retained);
            previous = Some(position);
        }
        context.check_cancelled()?;
        Ok(duplicates)
    }

    fn compare_left(
        &self,
        (batch, row): (usize, u32),
        identity: &(&i64, &state::Encoding, state::SequenceRef<'_>),
        keys: &InputKeys,
    ) -> Ordering {
        let record = &self.records[batch];
        record
            .times
            .value(row as usize)
            .cmp(identity.0)
            .then_with(|| {
                keys.values[record.keys[row as usize] as usize]
                    .encoding
                    .cmp(identity.1)
            })
            .then_with(|| {
                record
                    .sequences
                    .with_row(row as usize, |sequence| sequence.cmp(identity.2.as_slice()))
            })
    }

    fn scalar_left_duplicates(
        &self,
        operator: &StreamAsofJoinOperator,
        side: usize,
        keys: &InputKeys,
        context: &StreamOperatorContext<'_>,
    ) -> Result<u64> {
        let skip_resident = self.order.first().is_none_or(|&(batch, row)| {
            let record = &self.records[batch];
            let identity = (
                record.times.value(row as usize),
                keys.values[record.keys[row as usize] as usize]
                    .encoding
                    .clone(),
                record.sequences.row(row as usize),
            );
            super::super::identities_are_after_state(
                &operator.state,
                side,
                std::iter::once(&identity),
            )
        });
        let mut duplicates = 0;
        let mut previous = None;
        for (ordinal, &(batch, row)) in self.order.iter().enumerate() {
            check_input_cancellation(ordinal, context)?;
            let record = &self.records[batch];
            let identity = (
                record.times.value(row as usize),
                keys.values[record.keys[row as usize] as usize]
                    .encoding
                    .clone(),
                record.sequences.row(row as usize),
            );
            if previous.is_some_and(|last| self.same_identity(last, (batch, row)))
                || (!skip_resident && operator.state.contains_identity(side, &identity))
            {
                duplicates += 1;
            }
            previous = Some((batch, row));
        }
        context.check_cancelled()?;
        Ok(duplicates)
    }

    fn group_keys(&mut self, keys: &InputKeys, context: &StreamOperatorContext<'_>) -> Result<()> {
        self.key_rows = keys
            .values
            .iter()
            .map(|key| Vec::with_capacity(key.rows))
            .collect();
        for (ordinal, &(batch, row)) in self.order.iter().enumerate() {
            check_input_cancellation(ordinal, context)?;
            self.key_rows[self.records[batch].keys[row as usize] as usize].push((batch, row));
        }
        Ok(())
    }

    fn right_duplicates(
        &self,
        operator: &StreamAsofJoinOperator,
        keys: &InputKeys,
        context: &StreamOperatorContext<'_>,
    ) -> Result<u64> {
        self.key_rows
            .iter()
            .enumerate()
            .try_fold(0, |total, (id, _)| {
                let bucket = operator
                    .state
                    .right
                    .get(&keys.values[id].encoding)
                    .filter(|bucket| !self.follows_tail(id, bucket));
                self.right_key_duplicates(id, bucket, context)
                    .map(|count| total + count)
            })
    }

    fn right_key_duplicates(
        &self,
        key: usize,
        bucket: Option<&state::RightBucket>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<u64> {
        context.check_cancelled()?;
        let mut resident = bucket
            .into_iter()
            .flat_map(state::RightBucket::iter)
            .peekable();
        let mut duplicates = 0;
        let mut ordinal = 0;
        let mut previous = None;
        for &position in &self.key_rows[key] {
            check_input_cancellation(ordinal, context)?;
            ordinal += 1;
            if previous.is_some_and(|last| self.compare_orders(last, position).is_eq()) {
                duplicates += 1;
            } else {
                while resident
                    .peek()
                    .is_some_and(|(order, _)| self.compare_order(position, order).is_gt())
                {
                    check_input_cancellation(ordinal, context)?;
                    ordinal += 1;
                    resident.next();
                }
                duplicates += u64::from(
                    resident
                        .peek()
                        .is_some_and(|(order, _)| self.compare_order(position, order).is_eq()),
                );
            }
            previous = Some(position);
        }
        context.check_cancelled()?;
        Ok(duplicates)
    }

    fn compare_orders(&self, a: (usize, u32), b: (usize, u32)) -> Ordering {
        let right = &self.records[b.0];
        right.sequences.with_row(b.1 as usize, |sequence| {
            self.compare_bytes(a, right.times.value(b.1 as usize), sequence)
        })
    }

    fn same_identity(&self, a: (usize, u32), b: (usize, u32)) -> bool {
        self.records[a.0].keys[a.1 as usize] == self.records[b.0].keys[b.1 as usize]
            && self.compare_orders(a, b).is_eq()
    }

    fn compare_order(
        &self,
        position: (usize, u32),
        order: &(&i64, state::SequenceRef<'_>),
    ) -> Ordering {
        self.compare_bytes(position, *order.0, order.1.as_ref())
    }

    fn compare_bytes(&self, (batch, row): (usize, u32), time: i64, sequence: &[u8]) -> Ordering {
        let record = &self.records[batch];
        record.times.value(row as usize).cmp(&time).then_with(|| {
            record
                .sequences
                .with_row(row as usize, |bytes| bytes.cmp(sequence))
        })
    }

    fn follows_tail(&self, key: usize, bucket: &state::RightBucket) -> bool {
        self.key_rows[key].first().is_none_or(|&first| {
            bucket
                .last_key_value()
                .is_none_or(|(order, _)| self.compare_order(first, &order).is_gt())
        })
    }

    pub(in super::super) fn appends_after_state(
        &self,
        operator: &StreamAsofJoinOperator,
        keys: &[(state::Encoding, usize)],
        context: &StreamOperatorContext<'_>,
    ) -> Result<bool> {
        for (id, (key, _)) in keys.iter().enumerate() {
            check_input_cancellation(id, context)?;
            if operator
                .state
                .right
                .get(key)
                .is_some_and(|bucket| !self.follows_tail(id, bucket))
            {
                return Ok(false);
            }
        }
        context.check_cancelled()?;
        Ok(true)
    }

    pub(in super::super) fn rows_for_key<'a>(
        &'a self,
        key: usize,
        payloads: &'a [state::RowRef],
    ) -> impl DoubleEndedIterator<Item = (i64, state::Encoding, state::RowRef)> + ExactSizeIterator + 'a
    {
        self.key_rows[key].iter().map(move |&(batch, row)| {
            let record = &self.records[batch];
            (
                record.times.value(row as usize),
                record.sequences.row(row as usize),
                payloads[batch].with_row(row),
            )
        })
    }
}

fn left_cursor_credit(
    operator: &StreamAsofJoinOperator,
    skip_resident: bool,
) -> Result<Option<MemoryReservation>> {
    let bytes = if skip_resident {
        0
    } else {
        operator.state.left.iter_workspace_bytes()
    };
    match operator.reserve_workspace(bytes) {
        Ok(credit) => Ok(Some(credit)),
        Err(crate::CalcFlowError::OperatorReason {
            reason_code: crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        }) => Ok(None),
        Err(error) => Err(error),
    }
}
