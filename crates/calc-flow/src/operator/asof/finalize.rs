use super::{
    StreamAsofJoinOperator, StreamAsofJoinStatus, SweepStamp, checked, reason,
    state::{self, BatchKey, LeftOrder, RowPayload},
    workspace::ColumnWorkspace,
};
use crate::{
    Batch, BatchMetadata, CalcFlowError, EventTime, JsonMap, Result, StreamCollector,
    StreamOperatorContext, StreamingFailureReason,
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::collections::BTreeMap;

mod prefix;

struct PreparedOutput {
    batch: Batch,
    matched: u64,
    workspace: MemoryReservation,
}

impl StreamAsofJoinOperator {
    #[tracing::instrument(
        name = "asof.finalize",
        level = "debug",
        skip_all,
        fields(operator = %self.name, frontier, ended)
    )]
    pub(super) async fn finalize(
        &mut self,
        frontier: Option<i64>,
        ended: bool,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        loop {
            context.check_cancelled()?;
            if !self.has_finalizable(frontier, ended) {
                break;
            }
            // Defer right-side eviction until all ready left rows are emitted.
            // Each accepted prefix can then share the committed right segment.
            let headroom = self.checkpoint_workspace()?;
            let (mut keys, key_workspace) = self.finalizable_keys(frontier, ended, context)?;
            let prepared_output = self
                .prepare_output(&mut keys, &key_workspace, context)
                .await?;
            self.commit_prefix_output(&keys, prepared_output, headroom, context, output)
                .await?;
            drop(key_workspace);
        }
        self.finish_progress(frontier, ended, context).await
    }

    fn checkpoint_workspace(&self) -> Result<MemoryReservation> {
        self.reserve_workspace(
            self.prepared
                .as_ref()
                .map(|segment| segment.len() as u64)
                .or(self.deferred_index_len)
                .unwrap_or(0),
        )
    }

    fn has_finalizable(&self, frontier: Option<i64>, ended: bool) -> bool {
        self.state
            .left
            .first_key_value()
            .is_some_and(|((time, _, _), _)| ended || frontier.is_some_and(|bound| *time < bound))
    }

    fn finalizable_keys(
        &self,
        frontier: Option<i64>,
        ended: bool,
        context: &StreamOperatorContext<'_>,
    ) -> Result<(Vec<LeftOrder>, MemoryReservation)> {
        const MAX_KEYS: usize = 64_000;
        let limit = context.output_budget().max_rows.min(MAX_KEYS);
        let count = self.count_finalizable_keys(limit, frontier, ended, context)?;
        let (count, reservation) = self.reserve_finalizable_keys(count)?;
        let mut keys = Vec::with_capacity(count);
        for (index, key) in self.state.left.keys().take(count).enumerate() {
            if index % 1_024 == 0 {
                context.check_cancelled()?;
            }
            keys.push(key.clone());
        }
        context.check_cancelled()?;
        Ok((keys, reservation))
    }

    fn count_finalizable_keys(
        &self,
        limit: usize,
        frontier: Option<i64>,
        ended: bool,
        context: &StreamOperatorContext<'_>,
    ) -> Result<usize> {
        let mut count = 0;
        for (index, (time, _, _)) in self.state.left.keys().take(limit).enumerate() {
            if index % 1_024 == 0 {
                context.check_cancelled()?;
            }
            if !ended && frontier.is_none_or(|bound| *time >= bound) {
                break;
            }
            count += 1;
        }
        context.check_cancelled()?;
        Ok(count)
    }

    fn reserve_finalizable_keys(&self, mut count: usize) -> Result<(usize, MemoryReservation)> {
        loop {
            match self.reserve_workspace(key_workspace_bytes(count, count)) {
                Ok(reservation) => return Ok((count, reservation)),
                Err(_) if count > 1 => count /= 2,
                Err(error) => return Err(error),
            }
        }
    }

    fn output_status(&self, rows: usize, matched: u64) -> Result<StreamAsofJoinStatus> {
        let mut status = self.status.clone();
        status.emitted_left_rows = checked(&self.name, status.emitted_left_rows, rows as u64)?;
        status.matched_rows = checked(&self.name, status.matched_rows, matched)?;
        status.unmatched_rows = checked(&self.name, status.unmatched_rows, rows as u64 - matched)?;
        Ok(status)
    }

    #[tracing::instrument(
        name = "asof.sweep",
        level = "debug",
        skip_all,
        fields(operator = %self.name, frontier, ended)
    )]
    async fn finish_progress(
        &mut self,
        frontier: Option<i64>,
        ended: bool,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let stamp = SweepStamp::current(&self.status);
        if self.swept == Some(stamp)
            || !state::eviction_pending(&self.state, &self.status, self.spec.tolerance_micros())
        {
            // Nothing was admitted, removed or evicted since the committed
            // state's last sweep, so re-encoding would reproduce the installed
            // segment byte for byte; only progress metadata moves.
            self.status.output_watermark_micros = frontier
                .and_then(|time| time.checked_sub(1))
                .map(EventTime::from_micros)
                .or(self.status.output_watermark_micros);
            self.terminal = ended;
            self.swept = Some(stamp);
            return Ok(());
        }
        let workspace = self.reserve_workspace(self.state.batches.len() as u64 * 96)?;
        let preview =
            self.state
                .preview_eviction(&self.status, self.spec.tolerance_micros(), &self.name)?;
        let mut status = self.status.clone();
        status.evicted_right_rows = checked(
            &self.name,
            status.evicted_right_rows,
            preview.evicted_payloads,
        )?;
        status.retained_right_rows -= preview.evicted_payloads;
        status.identity_only_rows = checked(
            &self.name,
            status.identity_only_rows,
            preview.added_identity_only,
        )? - preview.removed_identity_only;
        status.state_rows -= preview.removed_identities;
        let previous_index_len = self
            .deferred_index_len
            .or_else(|| self.prepared.as_ref().map(|segment| segment.len() as u64))
            .expect("nonempty ASOF sweep has an index");
        let previous_index_bytes = self
            .prepared
            .as_ref()
            .map(|segment| segment.capacity() as u64)
            .or(self.deferred_index_len)
            .expect("nonempty ASOF sweep has an index")
            + 64;
        let next_index_len = previous_index_len - preview.removed_index_bytes;
        let next_index_bytes = if status.state_rows == 0 {
            0
        } else {
            next_index_len + 64
        };
        status.state_bytes =
            status.state_bytes - previous_index_bytes - preview.released_state_bytes
                + next_index_bytes;
        status.output_watermark_micros = frontier
            .and_then(|time| time.checked_sub(1))
            .map(EventTime::from_micros)
            .or(status.output_watermark_micros);
        context.check_cancelled()?;
        let evicted = self.state.evict(&status, self.spec.tolerance_micros());
        debug_assert_eq!(evicted, preview.evicted_payloads);
        self.status = status;
        self.prepared = None;
        self.deferred_index_len = (self.status.state_rows > 0).then_some(next_index_len);
        self.swept = Some(SweepStamp::current(&self.status));
        self.terminal = ended;
        debug_assert_eq!(
            self.state
                .inventory(None, &self.name)
                .expect("committed ASOF sweep inventory")
                .bytes
                + next_index_bytes,
            self.status.state_bytes,
            "swept inventory must match committed gauge"
        );
        drop(workspace);
        Ok(())
    }

    async fn prepare_output(
        &mut self,
        keys: &mut Vec<LeftOrder>,
        key_workspace: &MemoryReservation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedOutput> {
        loop {
            context.check_cancelled()?;
            match self.output_attempt(keys, context).await {
                Ok(output) => return Ok(output),
                Err(error) if keys.len() > 1 && retryable(&error) => {
                    keys.truncate(keys.len() / 2);
                    shrink_key_workspace(keys, key_workspace);
                }
                Err(error) => return Err(error),
            }
        }
    }

    #[tracing::instrument(
        name = "asof.output",
        level = "debug",
        skip_all,
        fields(operator = %self.name, rows = keys.len())
    )]
    async fn output_attempt(
        &mut self,
        keys: &[LeftOrder],
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedOutput> {
        let mut rows = Vec::with_capacity(keys.len());
        for (index, (key, left)) in self.state.left.iter().take(keys.len()).enumerate() {
            if index % 1_024 == 0 {
                context.check_cancelled()?;
            }
            rows.push((
                left,
                self.state
                    .candidate(&key.1, key.0, self.spec.tolerance_micros()),
            ));
        }
        context.check_cancelled()?;
        let matched = rows.iter().filter(|(_, right)| right.is_some()).count() as u64;
        let mut workspace = self.reserve_workspace(16 * 1024)?;
        let bytes = output_workspace(&rows, &self.schemas[1], &mut workspace, &self.name)?;
        let remaining = bytes.saturating_sub(workspace.size() as u64);
        grow_output_workspace(&mut workspace, remaining, &self.name)?;
        let (result, workspace) = self
            .runtime
            .materialize(&rows, &self.schemas[2], workspace)
            .await?;
        let batch = self.output_batch(&result, context)?;
        Ok(PreparedOutput {
            batch,
            matched,
            workspace,
        })
    }
    fn output_batch(&self, result: &Batch, context: &StreamOperatorContext<'_>) -> Result<Batch> {
        let batch = Batch::table(
            result.table_payload()?.batches().to_vec(),
            BatchMetadata::new(&self.name, self.next_output_sequence, JsonMap::new())?,
        )?;
        if batch.estimated_bytes()? > context.output_budget().max_bytes {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofOutputLimitExceeded,
                "ASOF output row exceeds the output edge budget",
            ));
        }
        Ok(batch)
    }
}

fn retryable(error: &CalcFlowError) -> bool {
    matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded
                | StreamingFailureReason::AsofOutputLimitExceeded,
            ..
        }
    )
}

fn key_workspace_bytes(key_capacity: usize, candidate_rows: usize) -> u64 {
    (key_capacity * size_of::<LeftOrder>()
        + candidate_rows * size_of::<(&RowPayload, Option<&RowPayload>)>()
        + key_capacity * 80 // bounded batch-reference counts during commit
        + 128) as u64
}

fn shrink_key_workspace(keys: &mut Vec<LeftOrder>, reservation: &MemoryReservation) {
    keys.shrink_to_fit();
    let needed = usize::try_from(key_workspace_bytes(keys.capacity(), keys.len()))
        .expect("bounded ASOF key scratch");
    reservation.shrink(reservation.size() - needed);
}

/// Reserve for direct Arrow copies, temporary value-buffer growth, position
/// spans, and per-source column metadata. The row charge is computed from
/// actual Arrow slice widths, including repeated right candidates; the fourfold
/// multiplier covers a growing output buffer and a simultaneous old buffer.
fn output_workspace(
    rows: &[(&RowPayload, Option<&RowPayload>)],
    right_schema: &datafusion::arrow::datatypes::Schema,
    workspace: &mut MemoryReservation,
    name: &str,
) -> Result<u64> {
    let mut columns = BTreeMap::<BatchKey, Vec<ColumnWorkspace>>::new();
    let raw = raw_output_bytes(rows, &mut columns, workspace, name)?;
    let buffers = output_buffer_bytes(rows, right_schema, raw, name)?;
    checked(
        name,
        buffers,
        output_bookkeeping_bytes(rows.len(), &columns, name)?,
    )
}

fn raw_output_bytes(
    rows: &[(&RowPayload, Option<&RowPayload>)],
    columns: &mut BTreeMap<BatchKey, Vec<ColumnWorkspace>>,
    workspace: &mut MemoryReservation,
    name: &str,
) -> Result<u64> {
    let mut raw = 0;
    for (left, right) in rows {
        raw = checked(name, raw, row_slice_bytes(left, columns, workspace, name)?)?;
        if let Some(right) = right {
            raw = checked(name, raw, row_slice_bytes(right, columns, workspace, name)?)?;
        }
    }
    Ok(raw)
}

fn output_buffer_bytes(
    rows: &[(&RowPayload, Option<&RowPayload>)],
    right_schema: &datafusion::arrow::datatypes::Schema,
    raw: u64,
    name: &str,
) -> Result<u64> {
    let unmatched = rows.iter().filter(|(_, right)| right.is_none()).count() as u64;
    let null_bytes = null_output_row_bytes(right_schema, name)?
        .checked_mul(unmatched)
        .ok_or_else(|| workspace_overflow(name))?;
    checked(name, raw, null_bytes)?
        .checked_mul(4)
        .ok_or_else(|| workspace_overflow(name))
}

fn output_bookkeeping_bytes(
    rows: usize,
    columns: &BTreeMap<BatchKey, Vec<ColumnWorkspace>>,
    name: &str,
) -> Result<u64> {
    let source_columns = columns
        .values()
        .try_fold(0, |count, fields| checked(name, count, fields.len() as u64))?;
    let row_scratch = (rows as u64)
        .checked_mul(256)
        .ok_or_else(|| workspace_overflow(name))?;
    let source_scratch = source_columns
        .checked_mul(512)
        .and_then(|value| value.checked_add(columns.len() as u64 * 128))
        .ok_or_else(|| workspace_overflow(name))?;
    checked(name, row_scratch, checked(name, source_scratch, 16 * 1024)?)
}

fn null_output_row_bytes(schema: &datafusion::arrow::datatypes::Schema, name: &str) -> Result<u64> {
    use datafusion::arrow::datatypes::DataType;
    schema.fields().iter().try_fold(0, |total, field| {
        let data_type = field.data_type();
        let value_bytes = match data_type {
            DataType::Null => 0,
            DataType::Boolean => 1,
            DataType::Utf8 | DataType::Binary => 4,
            DataType::LargeUtf8 | DataType::LargeBinary => 8,
            DataType::FixedSizeBinary(width) => {
                u64::try_from(*width).expect("validated ASOF fixed binary width")
            }
            _ => data_type
                .primitive_width()
                .expect("validated flat ASOF type") as u64,
        };
        checked(name, total, checked(name, value_bytes, 1)?)
    })
}

fn workspace_overflow(name: &str) -> CalcFlowError {
    reason(
        name,
        StreamingFailureReason::AsofCounterOverflow,
        "ASOF output workspace arithmetic overflowed",
    )
}

fn row_slice_bytes(
    row: &RowPayload,
    cache: &mut BTreeMap<BatchKey, Vec<ColumnWorkspace>>,
    workspace: &mut MemoryReservation,
    name: &str,
) -> Result<u64> {
    let columns = match cache.entry(row.batch.key) {
        std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
        std::collections::btree_map::Entry::Vacant(entry) => {
            grow_output_workspace(
                workspace,
                128 + row.batch.record.num_columns() as u64 * 512,
                name,
            )?;
            let columns = row
                .batch
                .record
                .columns()
                .iter()
                .cloned()
                .map(ColumnWorkspace::new)
                .collect::<Result<Vec<_>>>()?;
            entry.insert(columns)
        }
    };
    columns.iter().try_fold(0, |total, column| {
        checked(name, total, column.bytes(row.row, name)?)
    })
}

fn grow_output_workspace(
    reservation: &mut MemoryReservation,
    bytes: u64,
    name: &str,
) -> Result<()> {
    let bytes = usize::try_from(bytes).map_err(|_| {
        reason(
            name,
            StreamingFailureReason::AsofWorkspaceLimitExceeded,
            "ASOF output workspace exceeds address domain",
        )
    })?;
    reservation.try_grow(bytes).map_err(|_| {
        reason(
            name,
            StreamingFailureReason::AsofWorkspaceLimitExceeded,
            "ASOF aggregate workspace exceeds max_state_bytes",
        )
    })
}

async fn emit_output(
    batch: Batch,
    context: &StreamOperatorContext<'_>,
    output: &mut dyn StreamCollector,
) -> Result<()> {
    context.check_cancelled()?;
    tokio::select! {
        result = output.emit("output", batch) => result,
        () = context.job().cancellation().cancelled() => context.check_cancelled(),
    }
}

#[cfg(test)]
mod workspace_tests {
    use super::*;
    use crate::StateSegment;
    use datafusion::arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
    use std::sync::Arc;

    #[test]
    fn output_retry_releases_key_and_candidate_scratch() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let mut keys = (0..4_096)
            .map(|time| {
                (
                    time,
                    state::Encoding::from_slice(&[1]),
                    state::Encoding::from_slice(&[1]),
                )
            })
            .collect::<Vec<_>>();
        let reservation = MemoryConsumer::new("asof-test-keys").register(&pool);
        reservation
            .try_grow(usize::try_from(key_workspace_bytes(keys.capacity(), keys.len())).unwrap())
            .unwrap();
        let initial = reservation.size();
        keys.truncate(1);
        shrink_key_workspace(&mut keys, &reservation);
        assert!(reservation.size() < initial / 100);
        assert_eq!(
            reservation.size(),
            usize::try_from(key_workspace_bytes(keys.capacity(), keys.len())).unwrap()
        );
        assert_eq!(pool.reserved(), reservation.size());
    }

    #[test]
    fn source_column_cache_grows_reservation_before_each_wide_batch() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(36 * 1024));
        let schema = Arc::new(Schema::new(
            (0..16)
                .map(|index| Field::new(format!("value_{index}"), DataType::Int64, false))
                .collect::<Vec<_>>(),
        ));
        let record = Arc::new(
            RecordBatch::try_new(
                schema,
                (0..16)
                    .map(|_| Arc::new(Int64Array::from(vec![1])) as _)
                    .collect(),
            )
            .unwrap(),
        );
        let payloads = (0..4)
            .map(|id| RowPayload {
                batch: Arc::new(state::PayloadBatch {
                    key: (0, id),
                    record: record.clone(),
                    encoded: std::sync::OnceLock::from(StateSegment::new(Vec::new())),
                    encoded_charge_bytes: 0,
                    body_bytes: 0,
                }),
                row: 0,
            })
            .collect::<Vec<_>>();
        let rows = payloads.iter().map(|row| (row, None)).collect::<Vec<_>>();
        let mut reservation = MemoryConsumer::new("asof-test-output").register(&pool);
        reservation.try_grow(16 * 1024).unwrap();
        assert!(matches!(
            output_workspace(&rows, &record.schema(), &mut reservation, "asof"),
            Err(CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
                ..
            })
        ));
        assert!(pool.reserved() <= 36 * 1024);
        assert!(pool.reserved() > 16 * 1024);
    }
}
