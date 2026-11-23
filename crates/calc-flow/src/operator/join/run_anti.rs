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
        let left = side == JoinSide::Left;
        let opposite = &self.sides[opposite(side).index()];
        let mut cursors = ProbeCursors::default();
        let mut retained = left.then(|| BooleanBufferBuilder::new(incoming.len()));
        let mut discarded = Probe::default();
        for row in 0..incoming.len() {
            let group = incoming.groups[row];
            let range = native_lookup::time_range(self.bounds, left, incoming.times[row]);
            let window = opposite.buckets.get(&group).and_then(|bucket| {
                let bounds = cursors.window(group, incoming.len(), bucket, range);
                (bounds.matches() != 0).then_some((bucket, bounds.range()))
            });
            if let Some(retained) = &mut retained {
                let pending = incoming.retained(row);
                retained.append(window.is_none() && pending);
                if window.is_none() && !pending {
                    matches.absent_ids.push(incoming.row_ids[row]);
                    matches.count = matches
                        .count
                        .checked_add(1)
                        .ok_or_else(|| run_error("unmatched count overflowed"))?;
                    if matches.count as u64 > self.limits.max_matches_per_input_batch() {
                        break;
                    }
                }
            } else if let Some((bucket, found)) = window {
                matches.witnesses.select::<false>(
                    &mut discarded,
                    row,
                    (group, bucket),
                    found,
                    usize::MAX,
                    check,
                )?;
            }
        }
        self.check_match_limit(matches.count)?;
        if let Some(mut retained) = retained {
            incoming.retain = BooleanArray::new(retained.finish(), None);
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
        if self.null_ids.is_empty() && closed_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut closed = closed_ids.iter().copied().peekable();
        let mut nulls = self.null_ids.iter().copied().peekable();
        let mut ids = std::iter::from_fn(|| match (closed.peek(), nulls.peek()) {
            (Some(a), Some(b)) if a < b => closed.next(),
            (_, Some(_)) => nulls.next(),
            _ => closed.next(),
        })
        .peekable();
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
                if let Some(columns) = self.retained_columns {
                    return columns[physical];
                }
                physical
            };
            let mut chunks = LeftChunks::new(rows, output, column);
            while let Some(part) = chunks.next_batch(&state.name, sequence, check)? {
                sequence = next_output_sequence(sequence, part.len())?;
                batches.extend(part);
            }
            base = end;
        }
        Ok(batches)
    }
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

    fn next_batch(
        &mut self,
        name: &str,
        sequence: u64,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Option<Vec<Batch>>> {
        let mut selected = Vec::new();
        let mut bytes = 0usize;
        while selected.len() < self.output.budget.max_rows.min(MORSEL_ROWS) {
            if self.scanned.is_multiple_of(MORSEL_ROWS) {
                check()?;
            }
            let Some(item) = self.rows.peek() else { break };
            let Some(payload) = *item else {
                self.rows.next();
                self.scanned += 1;
                continue;
            };
            let charge = self.output.schema.fields.iter().enumerate().try_fold(
                0usize,
                |sum, (ordinal, _)| {
                    let column = payload.column((self.column)(ordinal));
                    let charge =
                        usize::try_from(logical_cell_charge(column.as_ref(), payload.offset())?)
                            .map_err(|_| run_error("left row charge exceeds address space"))?;
                    sum.checked_add(charge)
                        .ok_or_else(|| run_error("left row charge overflowed"))
                },
            )?;
            if !selected.is_empty() && bytes.saturating_add(charge) > self.output.budget.max_bytes {
                break;
            }
            bytes = bytes.saturating_add(charge);
            selected.push(payload);
            self.rows.next();
            self.scanned += 1;
        }
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
        for (side, cut) in [(&mut metrics.left, cuts[0]), (&mut metrics.right, cuts[1])] {
            side.evicted_rows =
                checked_metric(side.evicted_rows, cut as u64, &self.name, "evicted_rows")?;
        }
        let left_plan = self.plan_closed(JoinSide::Left, cuts[0], context).await?;
        let right_plan = self.plan_closed(JoinSide::Right, cuts[1], context).await?;
        let (rows, sequence) = self.visit_closed(cuts[0], output, context, None).await?;
        metrics.emitted_match_rows = checked_metric(
            metrics.emitted_match_rows,
            rows,
            &self.name,
            "emitted_match_rows",
        )?;
        if rows != 0 {
            self.visit_closed(cuts[0], output, context, Some(collector))
                .await?;
        }
        for (side, plan) in [(JoinSide::Left, left_plan), (JoinSide::Right, right_plan)] {
            if plan.cut != 0 {
                self.apply_eviction(side, plan);
            }
        }
        self.next_output = sequence;
        Ok(metrics)
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
        mut emit: Option<&mut dyn StreamCollector>,
    ) -> Result<(u64, u64)> {
        let side = &self.sides[JoinSide::Left.index()];
        let entries = &side.expiration[side.expiration_head..side.expiration_head + cut];
        let mut count = 0u64;
        let mut sequence = self.next_output;
        for (i, block) in entries.chunks(MORSEL_ROWS).enumerate() {
            yield_between_blocks(i, context).await?;
            let rows = block.iter().copied().map(|entry| {
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
            });
            let mut chunks = LeftChunks::new(rows, output, |ordinal| {
                output.projection.map_or(ordinal, |cols| cols[ordinal])
            });
            while let Some(batches) =
                chunks.next_batch(&self.name, sequence, &mut || context.check_cancelled())?
            {
                sequence = next_output_sequence(sequence, batches.len())?;
                for batch in batches {
                    count = next_output_sequence(count, batch.num_rows())?;
                    if let Some(collector) = emit.as_deref_mut() {
                        context.check_cancelled()?;
                        collector.emit("output", batch).await?;
                    }
                }
            }
            context.check_cancelled()?;
        }
        Ok((count, sequence))
    }
}

async fn yield_between_blocks(index: usize, context: &StreamOperatorContext<'_>) -> Result<()> {
    if index != 0 {
        tokio::task::yield_now().await;
    }
    context.check_cancelled()
}
