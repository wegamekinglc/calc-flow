//! Prepare contiguous columnar prefixes without rewriting fragment-local references.

use super::{StreamAsofJoinOperator, checked, retryable, state};
use crate::operator::asof::{
    output_plan::{FragmentCredits, OutputPlan, OutputPlanBuilder, Span, backing, grow_workspace},
    state::PayloadView,
};
use crate::{
    Result, StreamOperatorContext,
    runtime::streaming::gather_work::{GatherOperatorId, GatherStop, ParallelCpuWork},
};
use ahash::RandomState;
use datafusion::{
    arrow::{
        datatypes::{Field, Schema, SchemaRef},
        record_batch::RecordBatch,
    },
    execution::memory_pool::MemoryReservation,
};
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, OnceLock},
};

type Payloads = HashMap<u32, state::PayloadBatch, RandomState>;
type Buckets = HashMap<state::Encoding, Arc<state::RightBucket>, RandomState>;
type Route<'a> = (
    &'a Arc<state::RightBucket>,
    state::RightCursor<'a>,
    Option<(state::RowRef, usize)>,
);

pub(super) struct Prepared {
    pub(super) plans: Vec<OutputPlan>,
    pub(super) prefix: state::LeftPrefix,
    pub(super) credits: Arc<FragmentCredits>,
}

struct Fragment {
    plan: OutputPlan,
    prefix: state::LeftPrefix,
    credit: MemoryReservation,
    prefix_credit: MemoryReservation,
}

struct Work {
    units: Vec<Vec<state::OwnedOutputRun>>,
    payloads: Payloads,
    buckets: Buckets,
    selected: Option<[Vec<usize>; 2]>,
    right_schema: SchemaRef,
    name: String,
    tolerance: u64,
    credit: Arc<MemoryReservation>,
    _prefix_credit: Arc<MemoryReservation>,
    #[cfg(test)]
    hook: Option<Arc<dyn Fn(usize) + Send + Sync>>,
}

impl ParallelCpuWork for Work {
    type Output = Result<Fragment>;

    fn unit_count(&self) -> usize {
        self.units.len()
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<Self::Output> {
        // Optional workspace refusals must not become retained job failures.
        Ok(self.fragment(ordinal, stop))
    }
}

impl Work {
    fn view(&self, row: state::RowRef) -> PayloadView<'_> {
        PayloadView {
            batch: &self.payloads[&row.batch_handle()],
            row: row.row as usize,
        }
    }

    fn fragment(&self, ordinal: usize, stop: &GatherStop) -> Result<Fragment> {
        let runs = &self.units[ordinal];
        let count = runs.iter().map(|run| run.borrowed().len()).sum();
        #[cfg(feature = "benchmark-internals")]
        let _phase = stop
            .join_cpu_trace()
            .worker_stage(&self.name, "match_worker", count);
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(ordinal);
        }
        stop.check()?;
        let mut credit = self.credit.new_empty();
        let mut plan =
            OutputPlanBuilder::new(count, self.selected.as_ref(), &mut credit, &self.name)?
                .with_funded_sources();
        let prefix_credit = self.prefix_credit(runs, count)?;
        let mut prefix = state::LeftPrefix::default();
        self.append_runs(runs, &mut plan, &mut credit, &mut prefix, stop)?;
        let mut plan = plan.finish(&self.right_schema, &mut credit, &self.name)?;
        if let Some(span) = contiguous_right(&plan) {
            plan.right.spans.push(span);
        }
        Ok(Fragment {
            plan,
            prefix,
            credit,
            prefix_credit,
        })
    }

    fn append_runs(
        &self,
        runs: &[state::OwnedOutputRun],
        plan: &mut OutputPlanBuilder<'_>,
        credit: &mut MemoryReservation,
        prefix: &mut state::LeftPrefix,
        stop: &GatherStop,
    ) -> Result<()> {
        for run in runs {
            self.append_run(run, plan, credit, prefix, stop)?;
        }
        stop.check()
    }

    fn prefix_credit(
        &self,
        runs: &[state::OwnedOutputRun],
        count: usize,
    ) -> Result<MemoryReservation> {
        let credit = self.credit.new_empty();
        let original = super::prefix_workspace_bytes(count, 0, &self.name)?;
        let native = runs.iter().try_fold(2_048_u64, |bytes, run| {
            bytes.checked_add(run.native_prefix_workspace_bytes()?)
        });
        let bytes = native.map_or(original, |bytes| bytes.min(original));
        credit
            .try_grow(usize::try_from(bytes).map_err(|_| workspace_error(&self.name))?)
            .map_err(|_| workspace_error(&self.name))?;
        Ok(credit)
    }

    fn append_run(
        &self,
        owned: &state::OwnedOutputRun,
        plan: &mut OutputPlanBuilder<'_>,
        credit: &mut MemoryReservation,
        prefix: &mut state::LeftPrefix,
        stop: &GatherStop,
    ) -> Result<()> {
        let run = owned.borrowed();
        let count = run.len();
        let (first, left) = run.rows(1).next().expect("nonempty ASOF fragment");
        let source = plan.left_source(self.view(left), credit, &self.name)?;
        if let Some((keys, rows)) = run.dictionary_rows(count) {
            self.append_dictionary_rows(keys, rows, (source, *first.0), plan, credit, stop)?;
        } else {
            self.append_searched_rows(run, source, plan, credit, stop)?;
        }
        owned.visit_prefix(prefix, self.view(left).batch.key, &self.name)
    }

    fn append_dictionary_rows(
        &self,
        keys: &[Option<state::Encoding>],
        rows: impl Iterator<Item = (i64, usize, state::RowRef)>,
        left: (usize, i64),
        plan: &mut OutputPlanBuilder<'_>,
        credit: &mut MemoryReservation,
        stop: &GatherStop,
    ) -> Result<()> {
        let scratch = credit.new_empty();
        scratch
            .try_grow(keys.len() * 256 + 256)
            .map_err(|_| workspace_error(&self.name))?;
        let mut routes = keys
            .iter()
            .map(|key| {
                key.as_ref()
                    .and_then(|key| self.buckets.get(key))
                    .map(|bucket| (bucket, bucket.cursor_at(left.1), None))
            })
            .collect::<Vec<_>>();
        for (index, (time, key, row)) in rows.enumerate() {
            if index.is_multiple_of(1_024) {
                stop.check()?;
            }
            let matched = self.resolve_route(&mut routes[key], time, plan, credit)?;
            plan.left_position(left.0, row.row as usize, &self.name)?;
            plan.push_right_source(matched, &self.name)?;
        }
        Ok(())
    }

    fn append_searched_rows(
        &self,
        run: state::OutputRun<'_>,
        source: usize,
        plan: &mut OutputPlanBuilder<'_>,
        credit: &mut MemoryReservation,
        stop: &GatherStop,
    ) -> Result<()> {
        for (index, ((time, key, _), left)) in run.rows(run.len()).enumerate() {
            if index.is_multiple_of(1_024) {
                stop.check()?;
            }
            let right = self
                .buckets
                .get(key)
                .and_then(|bucket| bucket.candidate(*time, self.tolerance))
                .copied();
            plan.push_reference(
                source,
                left.row as usize,
                right.map(|row| self.view(row)),
                credit,
                &self.name,
            )?;
        }
        Ok(())
    }

    fn resolve_route(
        &self,
        route: &mut Option<Route<'_>>,
        time: i64,
        plan: &mut OutputPlanBuilder<'_>,
        credit: &mut MemoryReservation,
    ) -> Result<Option<(usize, usize)>> {
        match route {
            Some((bucket, cursor, latest)) => {
                let matched = bucket
                    .candidate_monotonic(time, self.tolerance, cursor)
                    .copied();
                self.resolve(matched, latest, plan, credit)
            }
            None => Ok(None),
        }
    }

    fn resolve(
        &self,
        row: Option<state::RowRef>,
        latest: &mut Option<(state::RowRef, usize)>,
        plan: &mut OutputPlanBuilder<'_>,
        credit: &mut MemoryReservation,
    ) -> Result<Option<(usize, usize)>> {
        let Some(row) = row else {
            return Ok(None);
        };
        if latest.is_none_or(|(previous, _)| previous != row) {
            *latest = Some((row, plan.right_source(self.view(row), credit, &self.name)?));
        }
        Ok(Some((
            latest.expect("resolved ASOF fragment source").1,
            row.row as usize,
        )))
    }
}

fn contiguous_right(plan: &OutputPlan) -> Option<Span> {
    if plan.right.has_nulls {
        return None;
    }
    let &(source, start) = plan.right.positions.first()?;
    let source_index = source.checked_sub(1)?;
    plan.right
        .positions
        .iter()
        .enumerate()
        .all(|(offset, &position)| position == (source, start + offset))
        .then_some(Span {
            source: source_index,
            start,
            end: start + plan.len,
        })
}

fn workspace_error(name: &str) -> crate::CalcFlowError {
    crate::operator::asof::reason(
        name,
        crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
        "ASOF segmented output workspace exceeds max_state_bytes",
    )
}

pub(super) async fn prepare(
    operator: &StreamAsofJoinOperator,
    count: usize,
    prefix_credit: Arc<MemoryReservation>,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<Prepared>> {
    let workers = context
        .job()
        .gather_owner()
        .parallelism()
        .min(count / 16_384);
    if workers < 2 {
        #[cfg(feature = "benchmark-internals")]
        note(
            context,
            &operator.name,
            "insufficient-work",
            count,
            workers,
            0,
        );
        return Ok(None);
    }
    let work = match capture(operator, count, workers, prefix_credit, context).await {
        Ok(Some(work)) => work,
        Ok(None) => {
            #[cfg(feature = "benchmark-internals")]
            note(context, &operator.name, "legacy-layout", count, workers, 0);
            return Ok(None);
        }
        Err(error) if retryable(&error) => {
            #[cfg(feature = "benchmark-internals")]
            note(
                context,
                &operator.name,
                "snapshot-budget",
                count,
                workers,
                0,
            );
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let input_credit = work.credit.clone();
    #[cfg(feature = "benchmark-internals")]
    note(
        context,
        &operator.name,
        "contiguous-fragments",
        count,
        workers,
        input_credit.size(),
    );
    let credit = match operator.cpu_work_credit(size_of::<Work>()) {
        Ok(credit) => credit,
        Err(error) if retryable(&error) => {
            #[cfg(feature = "benchmark-internals")]
            note(
                context,
                &operator.name,
                "control-budget",
                count,
                workers,
                input_credit.size(),
            );
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let scope = context
        .gather_client(GatherOperatorId::new(operator.name.clone().into()))
        .scope()?;
    let Some(output) =
        crate::operator::asof::cpu::finish_parallel_work(work, credit, &scope, context).await?
    else {
        #[cfg(feature = "benchmark-internals")]
        note(
            context,
            &operator.name,
            "admission-budget",
            count,
            workers,
            input_credit.size(),
        );
        return Ok(None);
    };
    assemble(operator, output, input_credit, count, context)
}

fn assemble(
    operator: &StreamAsofJoinOperator,
    output: crate::runtime::streaming::gather_work::WorkOutput<Vec<Result<Fragment>>>,
    input_credit: Arc<MemoryReservation>,
    count: usize,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<Prepared>> {
    #[cfg(not(feature = "benchmark-internals"))]
    let _ = (count, context);
    #[cfg(feature = "benchmark-internals")]
    let workers = output.value.len();
    let mut fragments = Vec::with_capacity(output.value.len());
    let mut refused = false;
    for result in output.value {
        match result {
            Ok(fragment) => fragments.push(fragment),
            Err(error) if retryable(&error) => refused = true,
            Err(error) => return Err(error),
        }
    }
    if refused {
        #[cfg(feature = "benchmark-internals")]
        note(
            context,
            &operator.name,
            "fragment-budget",
            count,
            workers,
            input_credit.size(),
        );
        return Ok(None);
    }
    #[cfg(feature = "benchmark-internals")]
    let _phase = context
        .job()
        .join_cpu_trace()
        .stage(&operator.name, "assembly", count);
    let mut plans = Vec::with_capacity(fragments.len());
    let mut credits = Vec::with_capacity(fragments.len() + 1);
    credits.push(output.credit);
    let mut prefix = state::LeftPrefix::default();
    for fragment in fragments {
        merge_prefix(&mut prefix, fragment.prefix, &operator.name)?;
        drop(fragment.prefix_credit);
        plans.push(fragment.plan);
        credits.push(fragment.credit);
    }
    Ok(Some(Prepared {
        plans,
        prefix,
        credits: Arc::new(FragmentCredits {
            _fragments: credits,
            _input: input_credit,
        }),
    }))
}

#[cfg(feature = "benchmark-internals")]
fn note(
    context: &StreamOperatorContext<'_>,
    name: &str,
    reason: &'static str,
    rows: usize,
    units: usize,
    bytes: usize,
) {
    context
        .job()
        .join_cpu_trace()
        .note(name, "probe_selection", reason, rows, units, bytes);
}

fn merge_prefix(
    target: &mut state::LeftPrefix,
    source: state::LeftPrefix,
    name: &str,
) -> Result<()> {
    for (batch, count) in source.batches {
        target.visit_batch(batch, count, name)?;
    }
    for (key, count) in source.keys {
        *target.keys.entry(key).or_default() += count;
    }
    for (owner, count) in source.owners {
        *target.owners.entry(owner).or_default() += count;
    }
    for (batch, owners) in source.sequence_owners {
        let target = target.sequence_owners.entry(batch).or_default();
        for (owner, count) in owners {
            *target.entry(owner).or_default() += count;
        }
    }
    Ok(())
}

async fn capture(
    operator: &StreamAsofJoinOperator,
    count: usize,
    workers: usize,
    prefix_credit: Arc<MemoryReservation>,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<Work>> {
    if !operator.state.left.legacy.is_empty() {
        return Ok(None);
    }
    let mut credit = capture_credit(operator, workers)?;
    let units = capture_units(operator, count, workers, &mut credit, context).await?;
    let payloads = capture_payloads(operator, &mut credit, context).await?;
    let schema = operator.physical_schema(1);
    let schema_bytes = schema.fields().iter().try_fold(1_024, |bytes, field| {
        checked(
            &operator.name,
            bytes,
            field.data_type().size() as u64 + 1_024,
        )
    })?;
    grow_workspace(&mut credit, schema_bytes, &operator.name)?;
    let right_schema = Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .enumerate()
            .map(|(index, field)| Field::new(index.to_string(), field.data_type().clone(), true))
            .collect::<Vec<_>>(),
    ));
    Ok(Some(Work {
        units,
        payloads,
        buckets: operator.state.right.owned_buckets().collect(),
        selected: operator.output_columns.clone(),
        right_schema,
        name: operator.name.clone(),
        tolerance: operator.spec.tolerance_micros(),
        credit: Arc::new(credit),
        _prefix_credit: prefix_credit,
        #[cfg(test)]
        hook: operator.match_hook.clone(),
    }))
}

fn capture_credit(operator: &StreamAsofJoinOperator, workers: usize) -> Result<MemoryReservation> {
    let retained = checked(
        &operator.name,
        operator.state.left.capacity_bytes(&operator.name)?,
        operator.state.right.metadata_bytes(),
    )?;
    let retained = checked(
        &operator.name,
        retained,
        operator.state.encoding_owner_allocation().0,
    )?;
    let control = checked(
        &operator.name,
        workers as u64 * 1_024 + 8_192,
        operator.state.right.len() as u64 * 512,
    )?;
    let control = checked(&operator.name, control, operator.name.len() as u64)?;
    operator.reserve_workspace(checked(&operator.name, retained, control)?)
}

async fn capture_units(
    operator: &StreamAsofJoinOperator,
    count: usize,
    workers: usize,
    credit: &mut MemoryReservation,
    context: &StreamOperatorContext<'_>,
) -> Result<Vec<Vec<state::OwnedOutputRun>>> {
    let mut units: Vec<Vec<state::OwnedOutputRun>> = (0..workers).map(|_| Vec::new()).collect();
    let width = count.div_ceil(workers);
    let mut position = 0;
    for (index, run) in operator.state.left.output_runs().enumerate() {
        super::check_match_progress(index, context).await?;
        let amount = run.len().min(count - position);
        let mut start = 0;
        while start < amount {
            let unit = (position / width).min(workers - 1);
            let take = (width - position % width).min(amount - start);
            grow_workspace(credit, 256, &operator.name)?;
            units[unit].push(
                run.owned_slice(start..start + take)
                    .expect("columnar ASOF run"),
            );
            position += take;
            start += take;
        }
        if position == count {
            break;
        }
    }
    Ok(units)
}

async fn capture_payloads(
    operator: &StreamAsofJoinOperator,
    credit: &mut MemoryReservation,
    context: &StreamOperatorContext<'_>,
) -> Result<Payloads> {
    let mut payloads = Payloads::default();
    for (index, (handle, payload, _)) in operator
        .state
        .batches
        .compaction_entries(&BTreeMap::new())
        .enumerate()
    {
        super::check_match_progress(index, context).await?;
        let backing = backing::source_bytes(&payload.record, credit, &operator.name)?;
        let fields = payload
            .record
            .columns()
            .iter()
            .try_fold(1_024, |bytes, column| {
                checked(
                    &operator.name,
                    bytes,
                    column.data_type().size() as u64 + 1_024,
                )
            })?;
        grow_workspace(
            credit,
            checked(&operator.name, backing, fields)?,
            &operator.name,
        )?;
        let schema = Arc::new(Schema::new(
            payload
                .record
                .columns()
                .iter()
                .enumerate()
                .map(|(index, column)| {
                    Field::new(index.to_string(), column.data_type().clone(), true)
                })
                .collect::<Vec<_>>(),
        ));
        let record = RecordBatch::try_new(schema, payload.record.columns().to_vec())
            .map_err(|error| crate::operator::asof::arrow_error(&error))?;
        payloads.insert(
            handle,
            state::PayloadBatch {
                key: payload.key,
                record: Arc::new(record),
                encoded: OnceLock::new(),
                encoded_charge_bytes: 0,
                body_bytes: 0,
            },
        );
    }
    Ok(payloads)
}

#[cfg(test)]
mod tests;
