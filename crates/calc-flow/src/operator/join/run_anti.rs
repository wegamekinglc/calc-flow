use super::super::{
    JoinMetrics, JoinSide, PayloadRef, checked_metric, logical_cell_charge, native_lookup,
    output_gather,
};
use super::{
    EvictionPlan, Incoming, InputMatches, Probe, ProbeCursors, RunOutputContext, RunState,
    next_output_sequence, opposite, run_error,
};
use crate::operator::keyed::MORSEL_ROWS;
use crate::operator::output_chunk::{OutputChunkErrors, chunk_output_record_checked};
use crate::{Batch, Result, StreamCollector, StreamOperatorContext};
use datafusion::arrow::{
    array::{BooleanArray, BooleanBufferBuilder},
    record_batch::{RecordBatch, RecordBatchOptions},
};

#[cfg(test)]
thread_local! {
    pub(in crate::operator::join) static CLOSED_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Clone, Copy)]
pub(in crate::operator::join) struct AntiInput<'a> {
    pub(in crate::operator::join) batch: &'a Batch,
    pub(in crate::operator::join) null_ids: &'a [u64],
    #[cfg(test)]
    pub(in crate::operator::join) retained_columns: Option<&'a [usize]>,
}

impl RunState {
    pub(super) fn probe_anti(
        &mut self,
        side: JoinSide,
        incoming: &mut Incoming,
        matches: &mut InputMatches,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        if side == JoinSide::Left {
            let retained = self.anti_left_rows(incoming, matches)?;
            incoming.retain = retained;
        } else {
            self.mark_anti_windows(incoming, matches, check)?;
        }
        self.check_match_limit(matches.count)
    }

    fn anti_window<'s>(
        &'s self,
        side: JoinSide,
        incoming: &Incoming,
        row: usize,
        cursors: &mut ProbeCursors,
    ) -> Option<(&'s super::Bucket, std::ops::Range<usize>)> {
        let group = incoming.groups[row];
        let bucket = self.sides[opposite(side).index()].buckets.get(&group)?;
        let range =
            native_lookup::time_range(self.bounds, side == JoinSide::Left, incoming.times[row]);
        let found = cursors.window(group, incoming.len(), bucket, range);
        (found.matches() != 0).then_some((bucket, found.range()))
    }

    fn anti_left_rows(
        &self,
        incoming: &Incoming,
        matches: &mut InputMatches,
    ) -> Result<BooleanArray> {
        let mut retained = BooleanBufferBuilder::new(incoming.len());
        let mut cursors = ProbeCursors::default();
        for row in 0..incoming.len() {
            let absent = self
                .anti_window(JoinSide::Left, incoming, row, &mut cursors)
                .is_none();
            let pending = incoming.retained(row);
            if !absent {
                retained.append(false);
                continue;
            }
            retained.append(pending);
            if !pending {
                matches.absent_ids.push(incoming.row_ids[row]);
                matches.count = matches
                    .count
                    .checked_add(1)
                    .ok_or_else(|| run_error("unmatched count overflowed"))?;
                if matches.count as u64 > self.limits.max_matches_per_input_batch() {
                    break;
                }
            }
        }
        Ok(BooleanArray::new(retained.finish(), None))
    }

    fn mark_anti_windows(
        &self,
        incoming: &Incoming,
        matches: &mut InputMatches,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let mut cursors = ProbeCursors::default();
        let mut discarded = Probe::default();
        for row in 0..incoming.len() {
            if let Some((bucket, window)) =
                self.anti_window(JoinSide::Right, incoming, row, &mut cursors)
            {
                matches.witnesses.select::<false>(
                    &mut discarded,
                    row,
                    (incoming.groups[row], bucket),
                    window,
                    usize::MAX,
                    check,
                )?;
            }
        }
        Ok(())
    }
}

impl AntiInput<'_> {
    pub(super) fn materialize(
        self,
        state: &RunState,
        closed_ids: &[u64],
        output: &RunOutputContext<'_>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<Batch>> {
        let mut ids = merged_ids(closed_ids, self.null_ids).peekable();
        if ids.peek().is_none() {
            return Ok(Vec::new());
        }
        let mut base = state.next_row_id(JoinSide::Left);
        let mut sequence = state.next_output;
        let mut batches = Vec::new();
        for record in self.batch.table_payload()?.batches() {
            let end = next_output_sequence(base, record.num_rows())?;
            let rows = std::iter::from_fn(|| ids.next_if(|&id| id < end)).map(|id| {
                Some(PayloadRef::Record(
                    record,
                    usize::try_from(id - base).expect("row in record"),
                ))
            });
            let column = |ordinal| {
                let physical = output
                    .projection
                    .map_or(ordinal, |indices| indices[ordinal]);
                #[cfg(test)]
                let physical = self
                    .retained_columns
                    .map_or(physical, |columns| columns[physical]);
                physical
            };
            let mut chunks = LeftChunks::new(rows, output, column);
            sequence = chunks.append_to(&mut batches, &state.name, sequence, check)?;
            base = end;
        }
        Ok(batches)
    }
}

fn merged_ids<'a>(closed: &'a [u64], nulls: &'a [u64]) -> impl Iterator<Item = u64> + 'a {
    let mut closed = closed.iter().copied().peekable();
    let mut nulls = nulls.iter().copied().peekable();
    std::iter::from_fn(move || match (closed.peek(), nulls.peek()) {
        (Some(a), Some(b)) if a < b => closed.next(),
        (_, Some(_)) => nulls.next(),
        _ => closed.next(),
    })
}

struct LeftChunks<'a, 's, I, C>
where
    I: Iterator<Item = Option<PayloadRef<'a>>>,
{
    rows: std::iter::Peekable<I>,
    output: &'s RunOutputContext<'s>,
    column: C,
    scanned: usize,
}

impl<'a, 's, I, C> LeftChunks<'a, 's, I, C>
where
    I: Iterator<Item = Option<PayloadRef<'a>>>,
    C: Fn(usize) -> usize,
{
    fn new(rows: I, output: &'s RunOutputContext<'s>, column: C) -> Self {
        Self {
            rows: rows.peekable(),
            output,
            column,
            scanned: 0,
        }
    }

    fn append_to(
        &mut self,
        batches: &mut Vec<Batch>,
        name: &str,
        mut sequence: u64,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<u64> {
        while let Some(part) = self.next_batch(name, sequence, check)? {
            sequence = next_output_sequence(sequence, part.len())?;
            batches.extend(part);
        }
        Ok(sequence)
    }

    fn row_charge(&self, payload: PayloadRef<'_>) -> Result<usize> {
        self.output
            .schema
            .fields
            .iter()
            .enumerate()
            .try_fold(0usize, |sum, (ordinal, _)| {
                let column = payload.column((self.column)(ordinal));
                let charge =
                    usize::try_from(logical_cell_charge(column.as_ref(), payload.offset())?)
                        .map_err(|_| run_error("left row charge exceeds address space"))?;
                sum.checked_add(charge)
                    .ok_or_else(|| run_error("left row charge overflowed"))
            })
    }

    fn peek_payload(
        &mut self,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Option<PayloadRef<'a>>> {
        loop {
            if self.scanned.is_multiple_of(MORSEL_ROWS) {
                check()?;
            }
            match self.rows.peek() {
                None => return Ok(None),
                Some(Some(payload)) => return Ok(Some(*payload)),
                Some(None) => {
                    self.rows.next();
                    self.scanned += 1;
                }
            }
        }
    }

    fn select_rows(
        &mut self,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<PayloadRef<'a>>> {
        let mut selected = Vec::new();
        let mut bytes = 0usize;
        while selected.len() < self.output.budget.max_rows.min(MORSEL_ROWS) {
            let Some(payload) = self.peek_payload(check)? else {
                break;
            };
            let charge = self.row_charge(payload)?;
            if !selected.is_empty() && bytes.saturating_add(charge) > self.output.budget.max_bytes {
                break;
            }
            bytes = bytes.saturating_add(charge);
            selected.push(payload);
            self.rows.next();
            self.scanned += 1;
        }
        Ok(selected)
    }

    fn next_batch(
        &mut self,
        name: &str,
        sequence: u64,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Option<Vec<Batch>>> {
        let selected = self.select_rows(check)?;
        if selected.is_empty() {
            return Ok(None);
        }
        let count = selected.len();
        let gather = output_gather::SideGather::new(selected.into_iter());
        let columns = self
            .output
            .schema
            .fields
            .iter()
            .enumerate()
            .map(|(ordinal, field)| {
                output_gather::column(gather.as_ref(), (self.column)(ordinal), field.data_type())
            })
            .collect::<Result<Vec<_>>>()?;
        let record = RecordBatch::try_new_with_options(
            self.output.schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(count)),
        )
        .map_err(|err| run_error(format!("left output: {err}")))?;
        chunk_output_record_checked(
            &record,
            name,
            sequence,
            self.output.budget,
            OutputChunkErrors::STREAM_JOIN,
            check,
        )
        .map(Some)
    }
}

impl RunState {
    pub(in crate::operator::join) async fn release_absent(
        &mut self,
        cuts: [usize; 2],
        output: &RunOutputContext<'_>,
        mut metrics: JoinMetrics,
        context: &StreamOperatorContext<'_>,
        collector: &mut dyn StreamCollector,
    ) -> Result<JoinMetrics> {
        if cuts == [0, 0] {
            return Ok(metrics);
        }
        charge_evictions(&mut metrics, cuts, &self.name)?;
        let left_plan = self.plan_closed(JoinSide::Left, cuts[0], context).await?;
        let right_plan = self.plan_closed(JoinSide::Right, cuts[1], context).await?;
        let (emitted, sequence) = self
            .emit_closed(
                cuts[0],
                output,
                metrics.emitted_match_rows,
                context,
                collector,
            )
            .await?;
        metrics.emitted_match_rows = emitted;
        self.commit_closed([left_plan, right_plan]);
        self.next_output = sequence;
        Ok(metrics)
    }

    async fn emit_closed(
        &self,
        cut: usize,
        output: &RunOutputContext<'_>,
        previous: u64,
        context: &StreamOperatorContext<'_>,
        collector: &mut dyn StreamCollector,
    ) -> Result<(u64, u64)> {
        let (rows, sequence) = self.visit_closed(cut, output, context, None).await?;
        let emitted = checked_metric(previous, rows, &self.name, "emitted_match_rows")?;
        if rows != 0 {
            self.visit_closed(cut, output, context, Some(collector))
                .await?;
        }
        Ok((emitted, sequence))
    }

    fn commit_closed(&mut self, plans: [EvictionPlan; 2]) {
        for (side, plan) in [JoinSide::Left, JoinSide::Right].into_iter().zip(plans) {
            if plan.cut != 0 {
                self.apply_eviction(side, plan);
            }
        }
    }

    async fn plan_closed(
        &self,
        side: JoinSide,
        cut: usize,
        context: &StreamOperatorContext<'_>,
    ) -> Result<EvictionPlan> {
        let retained = &self.sides[side.index()];
        let mut plan = EvictionPlan::empty(cut);
        let expired =
            &retained.expiration[retained.expiration_head..retained.expiration_head + cut];
        for (i, block) in expired.chunks(MORSEL_ROWS).enumerate() {
            yield_between_blocks(i, context).await?;
            plan.count(block);
        }
        plan.finish(retained, &mut || context.check_cancelled())
    }

    async fn visit_closed(
        &self,
        cut: usize,
        output: &RunOutputContext<'_>,
        context: &StreamOperatorContext<'_>,
        emit: Option<&mut dyn StreamCollector>,
    ) -> Result<(u64, u64)> {
        let side = &self.sides[JoinSide::Left.index()];
        let entries = &side.expiration[side.expiration_head..side.expiration_head + cut];
        let mut progress = ClosedOutput {
            count: 0,
            sequence: self.next_output,
            emit,
        };
        for (i, block) in entries.chunks(MORSEL_ROWS).enumerate() {
            yield_between_blocks(i, context).await?;
            let rows = block
                .iter()
                .copied()
                .map(|entry| closed_payload(side, entry));
            let mut chunks = LeftChunks::new(rows, output, |ordinal| {
                output.projection.map_or(ordinal, |cols| cols[ordinal])
            });
            while let Some(batches) =
                chunks.next_batch(&self.name, progress.sequence, &mut || {
                    context.check_cancelled()
                })?
            {
                progress.accept(batches, context).await?;
            }
            context.check_cancelled()?;
        }
        Ok((progress.count, progress.sequence))
    }
}

fn charge_evictions(metrics: &mut JoinMetrics, cuts: [usize; 2], name: &str) -> Result<()> {
    for (side, cut) in [(&mut metrics.left, cuts[0]), (&mut metrics.right, cuts[1])] {
        side.evicted_rows = checked_metric(side.evicted_rows, cut as u64, name, "evicted_rows")?;
    }
    Ok(())
}

fn closed_payload(side: &super::Side, entry: super::LocatorEntry) -> Option<PayloadRef<'_>> {
    #[cfg(test)]
    CLOSED_READS.set(CLOSED_READS.get() + 1);
    let bucket = side.buckets[&entry.group].active();
    let pos = bucket
        .binary_search_by_key(&entry.identity(), |row| row.identity())
        .expect("validated active expiration belongs to its bucket");
    let locator = bucket[pos].locator;
    (!locator.witnessed).then(|| {
        PayloadRef::Record(
            &side.runs[&locator.run].run.payload,
            usize::from(locator.row),
        )
    })
}

struct ClosedOutput<'a> {
    count: u64,
    sequence: u64,
    emit: Option<&'a mut dyn StreamCollector>,
}

impl ClosedOutput<'_> {
    async fn accept(
        &mut self,
        batches: Vec<Batch>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        self.sequence = next_output_sequence(self.sequence, batches.len())?;
        for batch in batches {
            self.count = next_output_sequence(self.count, batch.num_rows())?;
            if let Some(collector) = self.emit.as_deref_mut() {
                context.check_cancelled()?;
                collector.emit("output", batch).await?;
            }
        }
        Ok(())
    }
}

async fn yield_between_blocks(index: usize, context: &StreamOperatorContext<'_>) -> Result<()> {
    if index != 0 {
        tokio::task::yield_now().await;
    }
    context.check_cancelled()
}
