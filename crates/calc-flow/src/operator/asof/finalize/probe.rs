use super::{StreamAsofJoinOperator, check_match_progress, checked, retryable, state};
use crate::{
    Result, StreamOperatorContext,
    runtime::streaming::gather_work::{GatherOperatorId, GatherStop, ParallelCpuWork},
};
use ahash::RandomState;
use datafusion::execution::memory_pool::MemoryReservation;
use std::{collections::HashMap, sync::Arc};

#[cfg(test)]
mod tests;

type Matches = Vec<(usize, Option<state::RowRef>)>;
type Route = (usize, usize);
type Routes<'a> = HashMap<&'a state::Encoding, Route, RandomState>;

#[derive(Default)]
struct Shard {
    buckets: Vec<Arc<state::RightBucket>>,
    rows: Vec<(usize, i64, usize)>,
}

struct ProbeWork {
    shards: Vec<Shard>,
    tolerance: u64,
    #[cfg(test)]
    hook: Option<Arc<dyn Fn(usize) + Send + Sync>>,
    _input_credit: MemoryReservation,
}

impl ParallelCpuWork for ProbeWork {
    type Output = Matches;

    fn unit_count(&self) -> usize {
        self.shards.len()
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<Matches> {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(ordinal);
        }
        stop.check()?;
        let shard = &self.shards[ordinal];
        let first_time = shard.rows.first().map_or(0, |row| row.1);
        let mut cursors = shard
            .buckets
            .iter()
            .map(|bucket| bucket.cursor_at(first_time))
            .collect::<Vec<_>>();
        let mut matches = Vec::with_capacity(shard.rows.len());
        for (index, &(position, time, bucket)) in shard.rows.iter().enumerate() {
            if index.is_multiple_of(1_024) {
                stop.check()?;
            }
            let row = shard.buckets[bucket]
                .candidate_monotonic(time, self.tolerance, &mut cursors[bucket])
                .copied();
            matches.push((position, row));
        }
        stop.check()?;
        Ok(matches)
    }
}

pub(super) struct ProbedRows {
    pub(super) rows: Vec<Option<state::RowRef>>,
    _credit: MemoryReservation,
}

pub(super) async fn parallel_matches(
    operator: &StreamAsofJoinOperator,
    count: usize,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<ProbedRows>> {
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(operator.state.right.len())
        .min(count / 16_384);
    if workers < 2 {
        return Ok(None);
    }
    let (work, credit) = match capture(operator, count, workers, context).await {
        Ok(input) => input,
        Err(error) if retryable(&error) => return Ok(None),
        Err(error) => return Err(error),
    };
    if work.shards.len() < 2 {
        return Ok(None);
    }
    let scope = context
        .gather_client(GatherOperatorId::new(Arc::from(operator.name.as_str())))
        .scope()?;
    let Some(output) =
        super::super::cpu::finish_parallel_work(work, credit, &scope, context).await?
    else {
        return Ok(None);
    };
    let rows = restore_order(output.value, count);
    Ok(Some(ProbedRows {
        rows,
        _credit: output.credit,
    }))
}

fn restore_order(matches: Vec<Matches>, count: usize) -> Vec<Option<state::RowRef>> {
    let mut rows = vec![None; count];
    for shard in matches {
        for (position, row) in shard {
            rows[position] = row;
        }
    }
    rows
}

async fn capture(
    operator: &StreamAsofJoinOperator,
    count: usize,
    workers: usize,
    context: &StreamOperatorContext<'_>,
) -> Result<(ProbeWork, MemoryReservation)> {
    let keys = count.min(operator.state.right.len());
    let input_credit = input_credit(operator, count, keys)?;
    let credit = operator.reserve_workspace(count as u64 * 48 + 1_024)?;
    let mut work = ProbeWork {
        shards: (0..workers).map(|_| Shard::default()).collect(),
        tolerance: operator.spec.tolerance_micros(),
        #[cfg(test)]
        hook: operator.match_hook.clone(),
        _input_credit: input_credit,
    };
    let mut routes = HashMap::with_capacity_and_hasher(keys, RandomState::new());
    let mut position = 0;
    for run in operator.state.left.output_runs() {
        if position == count {
            break;
        }
        let amount = run.len().min(count - position);
        capture_run(
            operator,
            run,
            amount,
            position,
            &mut work,
            &mut routes,
            context,
        )
        .await?;
        position += amount;
    }
    work.shards.retain(|shard| !shard.rows.is_empty());
    Ok((work, credit))
}

async fn capture_run<'a>(
    operator: &StreamAsofJoinOperator,
    run: state::OutputRun<'a>,
    count: usize,
    position: usize,
    work: &mut ProbeWork,
    routes: &mut Routes<'a>,
    context: &StreamOperatorContext<'_>,
) -> Result<()> {
    check_match_progress(position, context).await?;
    if let Some((keys, mut rows)) = run.dictionary_rows(count) {
        let mut inline = [None; 64];
        let mut heap = Vec::new();
        let cached = super::merge::route_slots(keys.len(), &mut inline, &mut heap);
        for (route, key) in cached.iter_mut().zip(keys) {
            *route = key
                .as_ref()
                .and_then(|key| capture_route(operator, key, work, routes));
        }
        let (time, key, _) = rows.next().expect("nonempty ASOF capture run");
        append_route(work, cached[key], position, time);
        for (offset, (time, key, _)) in rows.enumerate() {
            let index = position + offset + 1;
            check_match_progress(index, context).await?;
            append_route(work, cached[key], index, time);
        }
    } else {
        let mut rows = run.rows(count);
        let (key, _) = rows.next().expect("nonempty ASOF capture run");
        let route = capture_route(operator, key.1, work, routes);
        append_route(work, route, position, *key.0);
        for (offset, (key, _)) in rows.enumerate() {
            let index = position + offset + 1;
            check_match_progress(index, context).await?;
            let route = capture_route(operator, key.1, work, routes);
            append_route(work, route, index, *key.0);
        }
    }
    Ok(())
}

fn capture_route<'a>(
    operator: &StreamAsofJoinOperator,
    key: &'a state::Encoding,
    work: &mut ProbeWork,
    routes: &mut Routes<'a>,
) -> Option<Route> {
    if let Some(route) = routes.get(key) {
        return Some(*route);
    }
    let bucket = operator.state.right.owned_bucket(key)?;
    let shard = state::key_partition(key, work.shards.len());
    let target = &mut work.shards[shard];
    let index = target.buckets.len();
    target.buckets.push(bucket);
    routes.insert(key, (shard, index));
    Some((shard, index))
}

fn append_route(work: &mut ProbeWork, route: Option<Route>, position: usize, time: i64) {
    if let Some((shard, bucket)) = route {
        work.shards[shard].rows.push((position, time, bucket));
    }
}

fn input_credit(
    operator: &StreamAsofJoinOperator,
    count: usize,
    keys: usize,
) -> Result<MemoryReservation> {
    let temporary = checked(&operator.name, count as u64 * 64, keys as u64 * 256 + 8_192)?;
    let temporary = checked(
        &operator.name,
        temporary,
        operator.state.left.iter_workspace_bytes(),
    )?;
    let retained = checked(
        &operator.name,
        operator.state.right.metadata_bytes(),
        operator.state.encoding_owner_allocation().0,
    )?;
    operator.reserve_workspace(checked(&operator.name, temporary, retained)?)
}
