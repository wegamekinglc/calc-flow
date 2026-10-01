use super::{
    StreamAsofJoinOperator, StreamAsofJoinStatus, SweepStamp, checked, reason,
    state::{self, BatchKey, LeftPrefix, PayloadView},
    workspace::{ColumnWorkspace, OutputColumns},
};
use crate::{
    Batch, BatchMetadata, CalcFlowError, EventTime, JsonMap, Result, StreamCollector,
    StreamOperatorContext, StreamingFailureReason,
};
use ahash::RandomState;
use datafusion::execution::memory_pool::MemoryReservation;
use std::collections::{BTreeMap, HashMap};

mod prefix;

type EvictionProjection = (state::EvictionPreview, u64, state::Inventory, u64);

struct PreparedOutput {
    batch: Batch,
    matched: u64,
    workspace: MemoryReservation,
    prefix: LeftPrefix,
}

struct MatchedPrefix<'a> {
    rows: Vec<(PayloadView<'a>, Option<PayloadView<'a>>)>,
    prefix: LeftPrefix,
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
            let (mut count, prefix_workspace) = self.finalizable_rows(frontier, ended, context)?;
            let prepared_output = self
                .prepare_output(&mut count, &prefix_workspace, context)
                .await?;
            self.commit_prefix_output(prepared_output, headroom, context, output)
                .await?;
            drop(prefix_workspace);
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

    fn finalizable_rows(
        &self,
        frontier: Option<i64>,
        ended: bool,
        context: &StreamOperatorContext<'_>,
    ) -> Result<(usize, MemoryReservation)> {
        const MAX_ROWS: usize = 64_000;
        context.check_cancelled()?;
        let limit = context.output_budget().max_rows.min(MAX_ROWS);
        let mut count = self.state.left.ready_prefix_len(limit, frontier, ended);
        loop {
            match self.reserve_workspace(
                prefix_workspace_bytes(count) + self.state.left.iter_workspace_bytes(),
            ) {
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
        self.finish_capacity_progress(frontier, ended, context)
            .await
    }

    fn capacity_eviction_projection(&self) -> Result<EvictionProjection> {
        let preview =
            self.state
                .preview_eviction(&self.status, self.spec.tolerance_micros(), &self.name)?;
        let (length, inventory, bytes) = self.state.project_capacity_eviction(
            self.capacity_snapshot(),
            &preview,
            &self.status,
            self.spec.tolerance_micros(),
            &self.name,
        )?;
        self.check_inventory_limits(&inventory)?;
        Ok((preview, length, inventory, bytes))
    }

    async fn finish_capacity_progress(
        &mut self,
        frontier: Option<i64>,
        ended: bool,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let staging = self.reserve_capacity_eviction_staging()?;
        let (preview, length, inventory, bytes) = self.capacity_eviction_projection()?;
        let columns = self.reserve_workspace(bytes)?;
        let status = self.capacity_progress_status(&preview, &inventory, frontier)?;
        let pool = self
            .prepare_pool_compaction(&preview.batches, context)
            .await?;
        let copies = self.checked_right_eviction_copies(context).await?;
        copies.install(&mut self.state.right);
        let evicted =
            self.state
                .evict_prepared(&status, self.spec.tolerance_micros(), pool, &preview);
        debug_assert_eq!(evicted, preview.evicted_payloads);
        self.status = status;
        self.prepared = None;
        self.deferred_index_len = (length != 0).then_some(length);
        self.swept = Some(SweepStamp::current(&self.status));
        self.terminal = ended;
        debug_assert_eq!(
            self.current_inventory(None)
                .expect("committed eviction inventory")
                .bytes
                + if length == 0 { 0 } else { length + 256 },
            self.status.state_bytes
        );
        drop((columns, staging));
        Ok(())
    }

    fn reserve_capacity_eviction_staging(&self) -> Result<MemoryReservation> {
        self.reserve_workspace(self.state.eviction_workspace_bytes(&self.name)?)
    }

    async fn checked_right_eviction_copies(
        &self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<super::copy::PreparedRightCopies> {
        let copies = self.prepare_right_eviction_copies(context).await?;
        context.check_cancelled()?;
        Ok(copies)
    }

    fn capacity_progress_status(
        &self,
        preview: &state::EvictionPreview,
        inventory: &state::Inventory,
        frontier: Option<i64>,
    ) -> Result<StreamAsofJoinStatus> {
        let mut status = self.status.clone();
        status.evicted_right_rows = checked(
            &self.name,
            status.evicted_right_rows,
            preview.evicted_payloads,
        )?;
        status.retained_right_rows = inventory.right_payloads;
        status.identity_only_rows = inventory.identity_only;
        status.state_rows = inventory.identities;
        status.state_bytes = inventory.bytes;
        status.output_watermark_micros = frontier
            .and_then(|time| time.checked_sub(1))
            .map(EventTime::from_micros)
            .or(status.output_watermark_micros);
        Ok(status)
    }

    async fn prepare_output(
        &mut self,
        count: &mut usize,
        prefix_workspace: &MemoryReservation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedOutput> {
        loop {
            context.check_cancelled()?;
            match self.output_attempt(*count, context).await {
                Ok(output) => return Ok(output),
                Err(error) if *count > 1 && retryable(&error) => {
                    *count /= 2;
                    shrink_prefix_workspace(
                        *count,
                        self.state.left.iter_workspace_bytes(),
                        prefix_workspace,
                    );
                }
                Err(error) => return Err(error),
            }
        }
    }

    #[tracing::instrument(
        name = "asof.output",
        level = "debug",
        skip_all,
        fields(operator = %self.name, rows = count)
    )]
    async fn output_attempt(
        &mut self,
        count: usize,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedOutput> {
        let cursor_workspace = self.cursor_workspace(count)?;
        let MatchedPrefix { rows, prefix } = match_output_prefix(
            &self.state,
            count,
            self.spec.tolerance_micros(),
            context,
            &self.name,
            cursor_workspace,
        )?;
        context.check_cancelled()?;
        let matched = rows.iter().filter(|(_, right)| right.is_some()).count() as u64;
        let mut workspace = self.reserve_workspace(16 * 1024)?;
        let bytes = output_workspace(
            &rows,
            &self.schemas[1],
            self.output_columns.as_ref(),
            &mut workspace,
            &self.name,
        )?;
        let remaining = bytes.saturating_sub(workspace.size() as u64);
        grow_output_workspace(&mut workspace, remaining, &self.name)?;
        let (result, workspace) = self
            .runtime
            .materialize(
                &rows,
                self.outputs[0].schema().expect("exact ASOF output"),
                workspace,
                || context.check_cancelled(),
            )
            .await?;
        let batch = self.output_batch(&result, context)?;
        Ok(PreparedOutput {
            batch,
            matched,
            workspace,
            prefix,
        })
    }

    fn cursor_workspace(&self, count: usize) -> Result<Option<MemoryReservation>> {
        let right_keys = self.state.right.len();
        if count < 1_024 || right_keys == 0 || right_keys > 4_096 || count < right_keys * 4 {
            return Ok(None);
        }
        // One hash slot, key, cursor, and allocator slack per right bucket.
        let charge = right_keys as u64 * 128 + 256;
        match self.reserve_workspace(charge) {
            Ok(reservation) => Ok(Some(reservation)),
            Err(error) if retryable(&error) => Ok(None),
            Err(error) => Err(error),
        }
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

fn match_output_prefix<'a>(
    state: &'a state::State,
    count: usize,
    tolerance: u64,
    context: &StreamOperatorContext<'_>,
    name: &str,
    cursor_workspace: Option<MemoryReservation>,
) -> Result<MatchedPrefix<'a>> {
    let matched = if cursor_workspace.is_some() {
        monotonic_candidate_rows(state, count, tolerance, context, name)?
    } else {
        binary_search_candidate_rows(state, count, tolerance, context, name)?
    };
    drop(cursor_workspace);
    Ok(matched)
}

fn binary_search_candidate_rows<'a>(
    state: &'a state::State,
    count: usize,
    tolerance: u64,
    context: &StreamOperatorContext<'_>,
    name: &str,
) -> Result<MatchedPrefix<'a>> {
    let mut rows = Vec::with_capacity(count);
    let mut prefix = LeftPrefix::default();
    for (index, (key, left)) in state.left.output_iter().take(count).enumerate() {
        if index % 1_024 == 0 {
            context.check_cancelled()?;
        }
        let left = state.batches.view(left);
        rows.push((
            left,
            state
                .candidate(key.1, *key.0, tolerance)
                .map(|row| state.batches.view(*row)),
        ));
        prefix.visit_owners(key.1, key.2, left.batch.key, name)?;
    }
    Ok(MatchedPrefix { rows, prefix })
}

fn monotonic_candidate_rows<'a>(
    state: &'a state::State,
    count: usize,
    tolerance: u64,
    context: &StreamOperatorContext<'_>,
    name: &str,
) -> Result<MatchedPrefix<'a>> {
    let first_time = state
        .left
        .first_key_value()
        .expect("nonempty ASOF prefix")
        .0
        .0;
    let first_time = *first_time;
    let mut cursors = HashMap::with_capacity_and_hasher(state.right.len(), RandomState::new());
    for (key, bucket) in &state.right {
        cursors.insert(key.clone(), (bucket, bucket.cursor_at(first_time)));
    }
    let mut rows = Vec::with_capacity(count);
    let mut prefix = LeftPrefix::default();
    for (index, (key, left)) in state.left.output_iter().take(count).enumerate() {
        if index % 1_024 == 0 {
            context.check_cancelled()?;
        }
        let right = cursors
            .get_mut(key.1)
            .and_then(|(bucket, next)| bucket.candidate_monotonic(*key.0, tolerance, next));
        let left = state.batches.view(left);
        rows.push((left, right.map(|row| state.batches.view(*row))));
        prefix.visit_owners(key.1, key.2, left.batch.key, name)?;
    }
    Ok(MatchedPrefix { rows, prefix })
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

fn prefix_workspace_bytes(count: usize) -> u64 {
    // Candidate references, batch-reference count tree (including a minimum
    // leaf), and allocator slack. No owned identity vector is constructed.
    (count * (size_of::<(PayloadView<'_>, Option<PayloadView<'_>>)>() + 640) + 2_048) as u64
}

fn shrink_prefix_workspace(count: usize, heap_bytes: u64, reservation: &MemoryReservation) {
    let needed = usize::try_from(prefix_workspace_bytes(count) + heap_bytes)
        .expect("bounded ASOF prefix scratch");
    reservation.shrink(reservation.size() - needed);
}

/// Reserve for direct Arrow copies, temporary value-buffer growth, position
/// spans, and per-source column metadata. The row charge is computed from
/// actual Arrow slice widths, including repeated right candidates; the fourfold
/// multiplier covers a growing output buffer and a simultaneous old buffer.
fn output_workspace(
    rows: &[(PayloadView<'_>, Option<PayloadView<'_>>)],
    right_schema: &datafusion::arrow::datatypes::Schema,
    selected: Option<&[Vec<usize>; 2]>,
    workspace: &mut MemoryReservation,
    name: &str,
) -> Result<u64> {
    let mut columns = BTreeMap::<BatchKey, OutputColumns>::new();
    let raw = raw_output_bytes(rows, selected, &mut columns, workspace, name)?;
    let buffers = output_buffer_bytes(
        rows,
        right_schema,
        selected.map(|columns| columns[1].as_slice()),
        raw,
        name,
    )?;
    checked(
        name,
        buffers,
        output_bookkeeping_bytes(rows.len(), &columns, name)?,
    )
}

fn raw_output_bytes(
    rows: &[(PayloadView<'_>, Option<PayloadView<'_>>)],
    selected: Option<&[Vec<usize>; 2]>,
    columns: &mut BTreeMap<BatchKey, OutputColumns>,
    workspace: &mut MemoryReservation,
    name: &str,
) -> Result<u64> {
    let left = raw_side_bytes(
        rows.iter().map(|(left, _)| *left),
        selected.map(|columns| columns[0].as_slice()),
        columns,
        workspace,
        name,
    )?;
    let right = raw_side_bytes(
        rows.iter().filter_map(|(_, right)| *right),
        selected.map(|columns| columns[1].as_slice()),
        columns,
        workspace,
        name,
    )?;
    checked(name, left, right)
}

fn raw_side_bytes<'a>(
    rows: impl Iterator<Item = PayloadView<'a>>,
    selected: Option<&[usize]>,
    columns: &mut BTreeMap<BatchKey, OutputColumns>,
    workspace: &mut MemoryReservation,
    name: &str,
) -> Result<u64> {
    let mut rows = rows.peekable();
    let mut raw = 0;
    while let Some(row) = rows.next() {
        let source = source_columns(&row, selected, columns, workspace, name)?;
        let bytes = if source.is_fixed_width() {
            let mut count = 1;
            while rows
                .peek()
                .is_some_and(|next| next.batch.key == row.batch.key)
            {
                rows.next();
                count += 1;
            }
            source.fixed_bytes(count, name)?
        } else {
            let mut end = row.row + 1;
            while rows
                .peek()
                .is_some_and(|next| next.batch.key == row.batch.key && next.row == end)
            {
                rows.next();
                end += 1;
            }
            source.range_bytes(row.row..end, name)?
        };
        raw = checked(name, raw, bytes)?;
    }
    Ok(raw)
}

fn output_buffer_bytes(
    rows: &[(PayloadView<'_>, Option<PayloadView<'_>>)],
    right_schema: &datafusion::arrow::datatypes::Schema,
    selected: Option<&[usize]>,
    raw: u64,
    name: &str,
) -> Result<u64> {
    let unmatched = rows.iter().filter(|(_, right)| right.is_none()).count() as u64;
    let null_bytes = null_output_row_bytes(right_schema, selected, name)?
        .checked_mul(unmatched)
        .ok_or_else(|| workspace_overflow(name))?;
    checked(name, raw, null_bytes)?
        .checked_mul(4)
        .ok_or_else(|| workspace_overflow(name))
}

fn output_bookkeeping_bytes(
    rows: usize,
    columns: &BTreeMap<BatchKey, OutputColumns>,
    name: &str,
) -> Result<u64> {
    let source_columns = columns.values().try_fold(0, |count, fields| {
        checked(name, count, fields.fields as u64)
    })?;
    let row_scratch = (rows as u64)
        .checked_mul(256)
        .ok_or_else(|| workspace_overflow(name))?;
    let source_scratch = source_columns
        .checked_mul(512)
        .and_then(|value| value.checked_add(columns.len() as u64 * 128))
        .ok_or_else(|| workspace_overflow(name))?;
    checked(name, row_scratch, checked(name, source_scratch, 16 * 1024)?)
}

fn null_output_row_bytes(
    schema: &datafusion::arrow::datatypes::Schema,
    selected: Option<&[usize]>,
    name: &str,
) -> Result<u64> {
    use datafusion::arrow::datatypes::DataType;
    schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(index, _)| selected.is_none_or(|selected| selected.contains(index)))
        .try_fold(0, |total, (_, field)| {
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

#[cfg(test)]
thread_local! {
    static SOURCE_PROBES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn take_source_probes() -> usize {
    SOURCE_PROBES.with(|probes| probes.replace(0))
}

fn source_columns<'a>(
    row: &PayloadView<'_>,
    selected: Option<&[usize]>,
    cache: &'a mut BTreeMap<BatchKey, OutputColumns>,
    workspace: &mut MemoryReservation,
    name: &str,
) -> Result<&'a OutputColumns> {
    #[cfg(test)]
    SOURCE_PROBES.with(|probes| probes.set(probes.get() + 1));
    Ok(match cache.entry(row.batch.key) {
        std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
        std::collections::btree_map::Entry::Vacant(entry) => {
            grow_output_workspace(
                workspace,
                128 + selected.map_or(row.batch.record.num_columns(), <[usize]>::len) as u64 * 512,
                name,
            )?;
            let columns = row
                .batch
                .record
                .columns()
                .iter()
                .enumerate()
                .filter(|(index, _)| selected.is_none_or(|selected| selected.contains(index)))
                .map(|(_, column)| column)
                .cloned()
                .map(ColumnWorkspace::new)
                .collect::<Result<Vec<_>>>()?;
            entry.insert(OutputColumns::new(columns, name)?)
        }
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

    fn payload(record: RecordBatch, key: BatchKey) -> state::RowPayload {
        state::RowPayload {
            batch: Arc::new(state::PayloadBatch {
                key,
                record: Arc::new(record),
                encoded: std::sync::OnceLock::from(StateSegment::new(Vec::new())),
                encoded_charge_bytes: 0,
                body_bytes: 0,
            }),
            row: 0,
        }
    }

    #[test]
    fn fixed_output_ranges_skip_per_column_length_probes() {
        use datafusion::arrow::array::{
            ArrayRef, BooleanArray, FixedSizeBinaryArray, Float64Array, NullArray,
        };
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![Some(1), None])),
            Arc::new(Float64Array::from(vec![Some(1.5), None])),
            Arc::new(BooleanArray::from(vec![Some(true), None])),
            Arc::new(NullArray::new(2)),
            Arc::new(
                FixedSizeBinaryArray::try_from_iter(
                    [b"abcd".as_slice(), b"efgh".as_slice()].into_iter(),
                )
                .unwrap(),
            ),
        ];
        let schema = Arc::new(Schema::new(
            arrays
                .iter()
                .enumerate()
                .map(|(i, a)| Field::new(format!("field_{i}"), a.data_type().clone(), true))
                .collect::<Vec<_>>(),
        ));
        let record = RecordBatch::try_new(schema, arrays).unwrap();
        let per_row = record
            .columns()
            .iter()
            .map(|column| {
                column
                    .to_data()
                    .slice(0, 1)
                    .get_slice_memory_size()
                    .unwrap() as u64
            })
            .sum::<u64>();
        let source = payload(record, (1, 0));
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let mut reservation = MemoryConsumer::new("output-test").register(&pool);
        super::super::workspace::take_range_calls();
        take_source_probes();
        let raw = raw_side_bytes(
            std::iter::repeat_n(source.view(), 100_000),
            None,
            &mut BTreeMap::new(),
            &mut reservation,
            "asof",
        )
        .unwrap();
        assert_eq!(raw, per_row * 100_000);
        assert_eq!(super::super::workspace::take_range_calls(), 0);
        assert_eq!(take_source_probes(), 1);
    }

    fn legacy_workspace_bytes(
        rows: &[(PayloadView<'_>, Option<PayloadView<'_>>)],
        schema: &Schema,
        selected: Option<&[Vec<usize>; 2]>,
    ) -> u64 {
        let mut sources = BTreeMap::new();
        let mut slice_bytes = 0;
        for (left, right) in rows {
            for (side, row) in [(0, Some(*left)), (1, *right)] {
                let Some(row) = row else { continue };
                let mut fields = 0;
                for (index, column) in row.batch.record.columns().iter().enumerate() {
                    if selected.is_some_and(|selected| !selected[side].contains(&index)) {
                        continue;
                    }
                    fields += 1;
                    slice_bytes += column
                        .to_data()
                        .slice(row.row, 1)
                        .get_slice_memory_size()
                        .unwrap() as u64;
                }
                sources.insert(row.batch.key, fields);
            }
        }
        let unmatched = rows.iter().filter(|(_, right)| right.is_none()).count() as u64;
        let null =
            null_output_row_bytes(schema, selected.map(|s| s[1].as_slice()), "asof").unwrap();
        (slice_bytes + unmatched * null) * 4
            + rows.len() as u64 * 256
            + sources.values().sum::<u64>() * 512
            + sources.len() as u64 * 128
            + 16 * 1024
    }

    #[test]
    fn output_source_charge_matches_arrow_slices_with_projection_and_repeats() {
        fn view(source: &state::RowPayload, row: usize) -> PayloadView<'_> {
            PayloadView {
                batch: source.batch.as_ref(),
                row,
            }
        }
        use datafusion::arrow::array::{
            ArrayRef, BinaryArray, BooleanArray, FixedSizeBinaryArray, Float64Array,
            LargeBinaryArray, LargeStringArray, NullArray, StringArray,
        };
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(vec![
                Some(1.5),
                None,
                Some(2.5),
                Some(3.5),
            ])),
            Arc::new(BooleanArray::from(vec![
                Some(true),
                None,
                Some(false),
                Some(true),
            ])),
            Arc::new(NullArray::new(4)),
            Arc::new(
                FixedSizeBinaryArray::try_from_iter(
                    [b"abcd".as_slice(), b"efgh", b"ijkl", b"mnop"].into_iter(),
                )
                .unwrap(),
            ),
            Arc::new(StringArray::from(vec![
                Some("outside"),
                None,
                Some(""),
                Some("text"),
            ])),
            Arc::new(LargeStringArray::from(vec![
                Some("outside"),
                Some("large"),
                None,
                Some(""),
            ])),
            Arc::new(BinaryArray::from(vec![
                Some(b"outside".as_slice()),
                None,
                Some(b"binary"),
                Some(b""),
            ])),
            Arc::new(LargeBinaryArray::from(vec![
                Some(b"outside".as_slice()),
                Some(b"large"),
                None,
                Some(b""),
            ])),
        ];
        let schema = Arc::new(Schema::new(
            arrays
                .iter()
                .enumerate()
                .map(|(i, a)| Field::new(format!("field_{i}"), a.data_type().clone(), true))
                .collect::<Vec<_>>(),
        ));
        let record = RecordBatch::try_new(schema.clone(), arrays).unwrap();
        let left_a = payload(record.slice(1, 3), (0, 0));
        let left_b = payload(record.clone(), (0, 1));
        let right_a = payload(record.slice(1, 3), (1, 0));
        let right_b = payload(record, (1, 1));
        let rows = vec![
            (view(&left_a, 0), Some(view(&right_a, 0))),
            (view(&left_a, 1), Some(view(&right_b, 2))),
            (view(&left_b, 3), Some(view(&right_a, 0))),
            (view(&left_a, 1), Some(view(&right_a, 1))),
            (view(&left_a, 2), Some(view(&right_a, 2))),
            (view(&left_b, 1), None),
            (view(&left_a, 0), Some(view(&right_b, 1))),
        ];
        let projections = [
            None,
            Some([vec![0, 1, 2, 3], vec![4, 5, 6, 7]]),
            Some([vec![0, 1, 2, 3], vec![0, 1, 2, 3]]),
            Some([vec![], vec![]]),
            Some([vec![4], vec![]]),
            Some([vec![], vec![0]]),
        ];
        for selected in &projections {
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
            let mut reservation = MemoryConsumer::new("output-test").register(&pool);
            let actual =
                output_workspace(&rows, &schema, selected.as_ref(), &mut reservation, "asof")
                    .unwrap();
            assert_eq!(
                actual,
                legacy_workspace_bytes(&rows, &schema, selected.as_ref())
            );
        }
    }

    #[test]
    fn fixed_output_runs_charge_reversed_rows_and_source_switches_exactly() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));
        let first = payload(
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]))],
            )
            .unwrap()
            .slice(1, 2),
            (1, 0),
        );
        let second = payload(
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![4, 5, 6]))]).unwrap(),
            (1, 1),
        );
        let rows = [
            (&first, 1),
            (&first, 0),
            (&first, 1),
            (&first, 1),
            (&second, 2),
            (&second, 0),
            (&first, 0),
            (&first, 1),
        ]
        .map(|(source, row)| PayloadView {
            batch: source.batch.as_ref(),
            row,
        });
        let expected = rows
            .iter()
            .map(|row| {
                row.batch
                    .record
                    .column(0)
                    .to_data()
                    .slice(row.row, 1)
                    .get_slice_memory_size()
                    .unwrap() as u64
            })
            .sum::<u64>();
        for selected in [None, Some([].as_slice())] {
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
            let mut reservation = MemoryConsumer::new("output-test").register(&pool);
            take_source_probes();
            let raw = raw_side_bytes(
                rows.iter().copied(),
                selected,
                &mut BTreeMap::new(),
                &mut reservation,
                "asof",
            )
            .unwrap();
            assert_eq!(raw, if selected.is_none() { expected } else { 0 });
            assert_eq!(take_source_probes(), 3);
            assert_eq!(
                reservation.size(),
                if selected.is_none() { 1280 } else { 256 }
            );
        }
    }

    #[test]
    fn output_retry_releases_key_and_candidate_scratch() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let mut count = 1_024;
        let reservation = MemoryConsumer::new("asof-test-keys").register(&pool);
        reservation
            .try_grow(usize::try_from(prefix_workspace_bytes(count)).unwrap())
            .unwrap();
        let initial = reservation.size();
        count = 1;
        shrink_prefix_workspace(count, 0, &reservation);
        assert!(reservation.size() < initial / 100);
        assert_eq!(
            reservation.size(),
            usize::try_from(prefix_workspace_bytes(count)).unwrap()
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
            .map(|id| state::RowPayload {
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
        let rows = payloads
            .iter()
            .map(|row| (row.view(), None))
            .collect::<Vec<_>>();
        let mut reservation = MemoryConsumer::new("asof-test-output").register(&pool);
        reservation.try_grow(16 * 1024).unwrap();
        assert!(matches!(
            output_workspace(&rows, &record.schema(), None, &mut reservation, "asof"),
            Err(CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
                ..
            })
        ));
        assert!(pool.reserved() <= 36 * 1024);
        assert!(pool.reserved() > 16 * 1024);
        drop(reservation);
        let mut reservation = MemoryConsumer::new("asof-test-projected-output").register(&pool);
        reservation.try_grow(16 * 1024).unwrap();
        let selected = [vec![], vec![0]];
        let required = output_workspace(
            &rows,
            &record.schema(),
            Some(&selected),
            &mut reservation,
            "asof",
        )
        .unwrap();
        let remaining = required.saturating_sub(reservation.size() as u64);
        grow_output_workspace(&mut reservation, remaining, "asof").unwrap();
        assert!(required <= 36 * 1024);
        assert!(pool.reserved() <= 36 * 1024);
    }
}
