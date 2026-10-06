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
        .min(state::KEY_SHARDS)
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
    for (position, (key, _)) in operator.state.left.output_iter().take(count).enumerate() {
        check_match_progress(position, context).await?;
        let route = if let Some(route) = routes.get(key.1) {
            Some(*route)
        } else if let Some(bucket) = operator.state.right.owned_bucket(key.1) {
            let shard = state::key_shard(key.1) % workers;
            let target = &mut work.shards[shard];
            let index = target.buckets.len();
            target.buckets.push(bucket);
            routes.insert(key.1, (shard, index));
            Some((shard, index))
        } else {
            None
        };
        if let Some((shard, bucket)) = route {
            work.shards[shard].rows.push((position, *key.0, bucket));
        }
    }
    work.shards.retain(|shard| !shard.rows.is_empty());
    Ok((work, credit))
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
