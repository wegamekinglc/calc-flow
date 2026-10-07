use super::{
    StreamAsofJoinOperator, StreamAsofJoinStatus, SweepStamp, checked,
    output_plan::{OutputPlan, OutputPlanBuilder},
    reason,
    state::{self, LeftPrefix},
};
#[cfg(test)]
use super::{
    state::{BatchKey, PayloadView},
    workspace::{ColumnWorkspace, OutputColumns},
};
use crate::{
    Batch, BatchMetadata, CalcFlowError, EventTime, JsonMap, Result, StreamCollector,
    StreamOperatorContext, StreamingFailureReason,
};
use ahash::RandomState;
use datafusion::execution::memory_pool::MemoryReservation;
#[cfg(test)]
use std::collections::BTreeMap;
use std::{collections::HashMap, sync::Arc};

#[cfg(test)]
mod output_ranges;
#[cfg(test)]
mod owner_runs;
mod prefix;
mod probe;

type EvictionProjection = (state::EvictionPreview, u64, state::Inventory, u64);

struct CapacityEviction {
    preview: state::EvictionPreview,
    length: u64,
    inventory: state::Inventory,
    journal: super::checkpoint::index_v3::log::journal::Journal,
    credit: Arc<MemoryReservation>,
    retention_bytes: u64,
    columns: MemoryReservation,
}

struct PreparedOutput {
    batch: Batch,
    matched: u64,
    workspace: MemoryReservation,
    prefix: LeftPrefix,
}

struct MatchedPrefix {
    plan: OutputPlan,
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
            self.retirement.wait(context).await?;
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
            if ended && self.state.left.is_empty() && self.state.right.is_empty() {
                self.status.state_bytes -= self.checkpoint_log.bytes();
                self.checkpoint_log = super::checkpoint::LogState::default();
            }
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
        let (length, inventory, bytes) =
            self.state
                .project_capacity_eviction(self.capacity_snapshot(), &preview, &self.name)?;
        Ok((preview, length, inventory, bytes))
    }

    async fn finish_capacity_progress(
        &mut self,
        frontier: Option<i64>,
        ended: bool,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let staging = self.reserve_capacity_eviction_staging()?;
        let CapacityEviction {
            preview,
            length,
            inventory,
            journal,
            credit,
            retention_bytes,
            columns,
        } = self.prepare_capacity_eviction()?;
        let dictionary = self.state.right.prepare_compaction(
            self.state.right.len() - preview.projected_right.0,
            &columns,
            &self.runtime.pool,
            &self.name,
        )?;
        let status = self.capacity_progress_status(&preview, &inventory, frontier)?;
        let pool = self
            .prepare_pool_compaction(&preview.batches, context)
            .await?;
        let copies = self
            .checked_right_eviction_copies(&preview.selected, context)
            .await?;
        copies.install(&mut self.state.right);
        let evicted = self.state.evict_prepared(
            &status,
            self.spec.tolerance_micros(),
            pool,
            dictionary,
            &preview,
        );
        debug_assert_eq!(evicted, preview.evicted_payloads);
        self.status = status;
        self.checkpoint_log.install_journal(journal);
        self.checkpoint_log.credit = Some(credit);
        self.checkpoint_log.retention_bytes = retention_bytes;
        self.checkpoint_log.pending = None;
        self.checkpoint_log.dirty_cut = self.checkpoint_log.keeps_delta();
        self.prepared = None;
        self.deferred_index_len = (length != 0).then_some(length);
        self.swept = Some(SweepStamp::current(&self.status));
        self.terminal = ended;
        self.finish_capacity_log(ended);
        debug_assert_eq!(
            self.current_inventory(None)
                .expect("committed eviction inventory")
                .bytes,
            self.status.state_bytes
        );
        drop((preview, columns, staging));
        self.retirement.wait(context).await
    }

    fn prepare_capacity_eviction(&self) -> Result<CapacityEviction> {
        let (preview, length, inventory, bytes) = self.capacity_eviction_projection()?;
        let journal = self.prepare_log_eviction(&preview)?;
        let (credit, retention_bytes) =
            self.prepare_log_retention(&preview.owners, &preview.batches)?;
        let inventory = self.log_projection(inventory, length, &journal, retention_bytes)?;
        let columns = self.reserve_workspace(bytes)?;
        Ok(CapacityEviction {
            preview,
            length,
            inventory,
            journal,
            credit,
            retention_bytes,
            columns,
        })
    }

    fn finish_capacity_log(&mut self, ended: bool) {
        if ended && self.state.left.is_empty() && self.state.right.is_empty() {
            self.status.state_bytes -= self.checkpoint_log.bytes();
            self.checkpoint_log = super::checkpoint::LogState::default();
        }
    }

    fn reserve_capacity_eviction_staging(&self) -> Result<MemoryReservation> {
        self.reserve_workspace(self.state.eviction_workspace_bytes(
            &self.status,
            self.spec.tolerance_micros(),
            &self.name,
        )?)
    }

    async fn checked_right_eviction_copies(
        &self,
        selected: &[u32],
        context: &StreamOperatorContext<'_>,
    ) -> Result<super::copy::PreparedRightCopies> {
        let copies = self
            .prepare_right_eviction_copies(selected, context)
            .await?;
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
        let mut workspace = self.reserve_workspace(0)?;
        let MatchedPrefix { plan, prefix } =
            match_output_prefix(self, count, context, cursor_workspace, &mut workspace).await?;
        context.check_cancelled()?;
        let matched = plan.matched;
        let (result, workspace) = self
            .runtime
            .materialize_plan(
                plan,
                self.outputs[0].schema().expect("exact ASOF output"),
                workspace,
                &self.name,
                context,
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

async fn match_output_prefix(
    operator: &StreamAsofJoinOperator,
    count: usize,
    context: &StreamOperatorContext<'_>,
    cursor_workspace: Option<MemoryReservation>,
    workspace: &mut MemoryReservation,
) -> Result<MatchedPrefix> {
    let mut plan = OutputPlanBuilder::new(
        count,
        operator.output_columns.as_ref(),
        workspace,
        &operator.name,
    )?;
    let prefix = if let Some(matches) = probe::parallel_matches(operator, count, context).await? {
        parallel_candidate_rows(
            operator,
            count,
            context,
            &matches.rows,
            &mut plan,
            workspace,
        )
        .await?
    } else if cursor_workspace.is_some() {
        monotonic_candidate_rows(operator, count, context, &mut plan, workspace).await?
    } else {
        binary_search_candidate_rows(operator, count, context, &mut plan, workspace).await?
    };
    drop(cursor_workspace);
    finish_output_prefix(operator, plan, prefix, workspace)
}

fn finish_output_prefix(
    operator: &StreamAsofJoinOperator,
    plan: OutputPlanBuilder<'_>,
    prefix: LeftPrefix,
    workspace: &mut MemoryReservation,
) -> Result<MatchedPrefix> {
    plan.finish(operator.physical_schema(1), workspace, &operator.name)
        .map(|plan| MatchedPrefix { plan, prefix })
}

async fn parallel_candidate_rows(
    operator: &StreamAsofJoinOperator,
    count: usize,
    context: &StreamOperatorContext<'_>,
    matches: &[Option<state::RowRef>],
    plan: &mut OutputPlanBuilder<'_>,
    workspace: &mut MemoryReservation,
) -> Result<LeftPrefix> {
    let state = &operator.state;
    let mut prefix = LeftPrefix::default();
    let count = count.min(matches.len());
    let mut index = 0;
    for run in state.left.output_runs() {
        let amount = run.len().min(count - index);
        let mut source = 0;
        for (offset, (_, left)) in run.rows(amount).enumerate() {
            check_match_progress(index, context).await?;
            source = left_run_source(source, offset, left, operator, plan, workspace)?;
            plan.push_reference(
                source,
                left.row as usize,
                matches[index].map(|row| state.batches.view(row)),
                workspace,
                &operator.name,
            )?;
            index += 1;
        }
        run.visit_prefix(amount, &mut prefix, &state.batches, &operator.name)?;
        if index == count {
            break;
        }
    }
    Ok(prefix)
}

async fn binary_search_candidate_rows(
    operator: &StreamAsofJoinOperator,
    count: usize,
    context: &StreamOperatorContext<'_>,
    plan: &mut OutputPlanBuilder<'_>,
    workspace: &mut MemoryReservation,
) -> Result<LeftPrefix> {
    let state = &operator.state;
    let tolerance = operator.spec.tolerance_micros();
    let mut prefix = LeftPrefix::default();
    let mut index = 0;
    for run in state.left.output_runs() {
        let amount = run.len().min(count - index);
        let mut source = 0;
        for (offset, (key, left)) in run.rows(amount).enumerate() {
            check_match_progress(index, context).await?;
            source = left_run_source(source, offset, left, operator, plan, workspace)?;
            let right = state
                .candidate(key.1, *key.0, tolerance)
                .map(|row| state.batches.view(*row));
            plan.push_reference(source, left.row as usize, right, workspace, &operator.name)?;
            index += 1;
        }
        run.visit_prefix(amount, &mut prefix, &state.batches, &operator.name)?;
        if index == count {
            break;
        }
    }
    Ok(prefix)
}

async fn monotonic_candidate_rows(
    operator: &StreamAsofJoinOperator,
    count: usize,
    context: &StreamOperatorContext<'_>,
    plan: &mut OutputPlanBuilder<'_>,
    workspace: &mut MemoryReservation,
) -> Result<LeftPrefix> {
    let state = &operator.state;
    let first_time = *state
        .left
        .first_key_value()
        .expect("nonempty ASOF prefix")
        .0
        .0;
    let mut cursors = HashMap::with_capacity_and_hasher(state.right.len(), RandomState::new());
    cursors.extend(
        (&state.right)
            .into_iter()
            .map(|(key, bucket)| (key.clone(), (bucket, bucket.cursor_at(first_time)))),
    );
    let mut prefix = LeftPrefix::default();
    let mut index = 0;
    for run in state.left.output_runs() {
        let amount = run.len().min(count - index);
        let mut source = 0;
        for (offset, (key, left)) in run.rows(amount).enumerate() {
            check_match_progress(index, context).await?;
            source = left_run_source(source, offset, left, operator, plan, workspace)?;
            let right = cursors.get_mut(key.1).and_then(|(bucket, next)| {
                bucket.candidate_monotonic(*key.0, operator.spec.tolerance_micros(), next)
            });
            plan.push_reference(
                source,
                left.row as usize,
                right.map(|row| state.batches.view(*row)),
                workspace,
                &operator.name,
            )?;
            index += 1;
        }
        run.visit_prefix(amount, &mut prefix, &state.batches, &operator.name)?;
        if index == count {
            break;
        }
    }
    Ok(prefix)
}

fn left_run_source(
    source: usize,
    offset: usize,
    row: state::RowRef,
    operator: &StreamAsofJoinOperator,
    plan: &mut OutputPlanBuilder<'_>,
    workspace: &mut MemoryReservation,
) -> Result<usize> {
    if offset == 0 {
        plan.left_source(operator.state.batches.view(row), workspace, &operator.name)
    } else {
        Ok(source)
    }
}

async fn check_match_progress(index: usize, context: &StreamOperatorContext<'_>) -> Result<()> {
    if index % 1_024 == 0 {
        if index > 0 {
            tokio::task::yield_now().await;
        }
        context.check_cancelled()?;
    }
    Ok(())
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
    (count * 640 + 2_048) as u64
}

fn shrink_prefix_workspace(count: usize, heap_bytes: u64, reservation: &MemoryReservation) {
    let needed = usize::try_from(prefix_workspace_bytes(count) + heap_bytes)
        .expect("bounded ASOF prefix scratch");
    reservation.shrink(reservation.size() - needed);
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
        let bytes = source_run_bytes(row, source, &mut rows, name)?;
        raw = checked(name, raw, bytes)?;
    }
    Ok(raw)
}

#[cfg(test)]
fn source_run_bytes<'a>(
    row: PayloadView<'a>,
    source: &OutputColumns,
    rows: &mut std::iter::Peekable<impl Iterator<Item = PayloadView<'a>>>,
    name: &str,
) -> Result<u64> {
    if source.is_fixed_width() {
        let count = fixed_run_count(row.batch.key, rows);
        source.fixed_bytes(count, name)
    } else {
        let mut end = row.row + 1;
        while rows
            .peek()
            .is_some_and(|next| next.batch.key == row.batch.key && next.row == end)
        {
            rows.next();
            end += 1;
        }
        source.range_bytes(row.row..end, name)
    }
}

#[cfg(test)]
fn fixed_run_count<'a>(
    key: BatchKey,
    rows: &mut std::iter::Peekable<impl Iterator<Item = PayloadView<'a>>>,
) -> usize {
    let mut count = 1;
    while rows.peek().is_some_and(|next| next.batch.key == key) {
        rows.next();
        count += 1;
    }
    count
}

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
fn null_output_row_bytes(
    schema: &datafusion::arrow::datatypes::Schema,
    selected: Option<&[usize]>,
    name: &str,
) -> Result<u64> {
    super::output_plan::null_row_bytes(schema, selected, name)
}

#[cfg(test)]
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

#[cfg(test)]
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
            #[cfg(test)]
            super::workspace::record_output_source_registration();
            grow_output_workspace(
                workspace,
                128 + selected.map_or(row.batch.record.num_columns(), <[usize]>::len) as u64 * 512,
                name,
            )?;
            let columns = super::output_plan::selected_columns(&row.batch.record, selected)
                .map(ColumnWorkspace::new)
                .collect::<Result<Vec<_>>>()?;
            entry.insert(OutputColumns::new(columns, name)?)
        }
    })
}

#[cfg(test)]
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
    fn output_plan_charges_repeated_selected_columns() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let source = payload(
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(vec![7]))])
                .unwrap(),
            (0, 0),
        );
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let mut reservation = MemoryConsumer::new("output-test").register(&pool);
        let selected = [vec![0, 0, 0], Vec::new()];
        let mut builder =
            OutputPlanBuilder::new(1, Some(&selected), &mut reservation, "asof").unwrap();
        builder
            .push(source.view(), None, &mut reservation, "asof")
            .unwrap();
        let plan = builder.finish(&schema, &mut reservation, "asof").unwrap();
        assert_eq!(plan.raw_bytes, 24);
        assert_eq!(
            null_output_row_bytes(&schema, Some(&[0, 0, 0]), "asof").unwrap(),
            27
        );
    }

    #[test]
    fn output_plan_accounts_shared_source_separately_for_each_side() {
        use datafusion::arrow::array::StringArray;
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int64, false),
            Field::new("text", DataType::Utf8, false),
        ]));
        let source = payload(
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int64Array::from(vec![7])),
                    Arc::new(StringArray::from(vec!["longer than an integer"])),
                ],
            )
            .unwrap(),
            (0, 0),
        );
        let expected = source
            .batch
            .record
            .column(0)
            .to_data()
            .get_slice_memory_size()
            .unwrap()
            + source
                .batch
                .record
                .column(1)
                .to_data()
                .get_slice_memory_size()
                .unwrap();
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let mut reservation = MemoryConsumer::new("output-test").register(&pool);
        let selected = [vec![0], vec![1]];
        let mut builder =
            OutputPlanBuilder::new(1, Some(&selected), &mut reservation, "asof").unwrap();
        builder
            .push(source.view(), Some(source.view()), &mut reservation, "asof")
            .unwrap();
        let plan = builder
            .finish(&source.batch.record.schema(), &mut reservation, "asof")
            .unwrap();
        assert_eq!(plan.raw_bytes, expected as u64);
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

    fn mixed_output_record() -> RecordBatch {
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
        RecordBatch::try_new(schema, arrays).unwrap()
    }

    #[test]
    fn output_source_charge_matches_arrow_slices_with_projection_and_repeats() {
        fn view(source: &state::RowPayload, row: usize) -> PayloadView<'_> {
            PayloadView {
                batch: source.batch.as_ref(),
                row,
            }
        }
        let record = mixed_output_record();
        let schema = record.schema();
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
            let mut builder =
                OutputPlanBuilder::new(rows.len(), selected.as_ref(), &mut reservation, "asof")
                    .unwrap();
            for &(left, right) in &rows {
                builder.push(left, right, &mut reservation, "asof").unwrap();
            }
            let plan = builder.finish(&schema, &mut reservation, "asof").unwrap();
            let raw = rows
                .iter()
                .flat_map(|(left, right)| [(0, Some(*left)), (1, *right)])
                .filter_map(|(side, row)| row.map(|row| (side, row)))
                .map(|(side, row)| {
                    super::super::output_plan::selected_columns(
                        &row.batch.record,
                        selected.as_ref().map(|columns| columns[side].as_slice()),
                    )
                    .map(|column| {
                        column
                            .to_data()
                            .slice(row.row, 1)
                            .get_slice_memory_size()
                            .unwrap() as u64
                    })
                    .sum::<u64>()
                })
                .sum::<u64>();
            assert_eq!(plan.raw_bytes, raw);
            assert_eq!(plan.matched, 6);
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
