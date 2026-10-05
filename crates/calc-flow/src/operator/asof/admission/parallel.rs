use super::{Admission, StreamAsofJoinOperator, state};
use crate::{
    Result, StreamOperatorContext,
    runtime::streaming::gather_work::{
        AdmissionFailure, GatherOperatorId, GatherStop, ParallelCpuWork,
    },
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::{collections::BTreeSet, sync::Arc};

type Rows = Vec<(state::RightOrder, state::RowRef)>;
type Buckets = Vec<(state::Encoding, Arc<state::RightBucket>)>;

struct InputBucket {
    key: state::Encoding,
    previous: Option<Arc<state::RightBucket>>,
    rows: Rows,
}

pub(super) struct AdmissionWork {
    shards: Vec<Vec<InputBucket>>,
    kind: state::SequenceKind,
    #[cfg(test)]
    hook: Option<Arc<dyn Fn(usize) + Send + Sync>>,
    _input_credit: MemoryReservation,
}

impl ParallelCpuWork for AdmissionWork {
    type Output = Buckets;

    fn unit_count(&self) -> usize {
        self.shards.len()
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<Buckets> {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(ordinal);
        }
        stop.check()?;
        self.shards[ordinal]
            .iter()
            .map(|input| {
                stop.check()?;
                let empty = state::RightBucket::with_sequence_kind(self.kind);
                let previous = input.previous.as_deref().unwrap_or(&empty);
                let bucket = previous.with_admitted(&input.rows, &|| stop.check())?;
                Ok((input.key.clone(), Arc::new(bucket)))
            })
            .collect()
    }
}

#[cfg(test)]
impl AdmissionWork {
    pub(super) fn key_units(&self) -> std::collections::BTreeMap<state::Encoding, usize> {
        self.shards
            .iter()
            .enumerate()
            .flat_map(|(unit, buckets)| {
                buckets.iter().map(move |bucket| (bucket.key.clone(), unit))
            })
            .collect()
    }
}

pub(in super::super) struct PreparedAdmission {
    buckets: Vec<Buckets>,
    references: Vec<state::RowRef>,
    _credit: MemoryReservation,
}

impl PreparedAdmission {
    pub(in super::super) fn install(
        self,
        state: &mut state::RightState,
        kind: state::SequenceKind,
        references: &[state::RowRef],
    ) {
        debug_assert_eq!(references, self.references);
        for (key, bucket) in self.buckets.into_iter().flatten() {
            state.replace_admitted(key, kind, bucket);
        }
    }
}

pub(in super::super) async fn prepare(
    operator: &StreamAsofJoinOperator,
    admission: &Admission,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<PreparedAdmission>> {
    let units = (admission.rows.len() / 4_096)
        .min(admission.right_capacities.len())
        .min(state::KEY_SHARDS);
    if units < 2 {
        return Ok(None);
    }
    let units = units.min(std::thread::available_parallelism().map_or(1, usize::from));
    if units < 2 {
        return Ok(None);
    }
    if appends_after_state(operator, admission, context)? {
        return Ok(None);
    }
    let (work, references, credit) = match capture(operator, admission, units, context).await {
        Ok(input) => input,
        Err(crate::CalcFlowError::OperatorReason {
            reason_code: crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        }) => return Ok(None),
        Err(error) => return Err(error),
    };
    let scope = context
        .gather_client(GatherOperatorId::new(Arc::from(operator.name.as_str())))
        .scope()?;
    let ticket = match scope
        .submit_parallel_work(Arc::new(work), credit, GatherStop::from_job(context.job()))
        .await
    {
        Ok(ticket) => ticket,
        Err(AdmissionFailure::Budget { .. }) => return Ok(None),
        Err(AdmissionFailure::Runtime(error)) => return Err(error),
    };
    let output = ticket.finish().await?;
    context.check_cancelled()?;
    Ok(Some(PreparedAdmission {
        buckets: output.value,
        references,
        _credit: output.credit,
    }))
}

fn appends_after_state(
    operator: &StreamAsofJoinOperator,
    admission: &Admission,
    context: &StreamOperatorContext<'_>,
) -> Result<bool> {
    let mut earliest = i64::MAX;
    for (ordinal, (identity, _)) in admission.rows.iter().enumerate() {
        if ordinal.is_multiple_of(1_024) {
            context.check_cancelled()?;
        }
        earliest = earliest.min(identity.0);
    }
    let latest = admission
        .right_capacities
        .iter()
        .filter_map(|(key, _)| {
            operator
                .state
                .right
                .get(key)?
                .last_key_value()
                .map(|((time, _), _)| *time)
        })
        .max();
    let Some(latest) = latest else {
        return Ok(false);
    };
    if latest < earliest {
        return Ok(true);
    }
    // Admission rows are ordered within each key.
    let mut previous = None;
    for (ordinal, (identity, _)) in admission.rows.iter().enumerate() {
        if ordinal.is_multiple_of(1_024) {
            context.check_cancelled()?;
        }
        if previous == Some(&identity.1) {
            continue;
        }
        previous = Some(&identity.1);
        if operator
            .state
            .right
            .get(&identity.1)
            .and_then(state::RightBucket::last_key_value)
            .is_some_and(|((time, sequence), _)| {
                (time, sequence.as_ref()) >= (&identity.0, &identity.2)
            })
        {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(super) async fn capture(
    operator: &StreamAsofJoinOperator,
    admission: &Admission,
    units: usize,
    context: &StreamOperatorContext<'_>,
) -> Result<(AdmissionWork, Vec<state::RowRef>, MemoryReservation)> {
    let name = &operator.name;
    let scratch = admission.rows.len() as u64 * 128
        + admission.right_capacities.len() as u64 * 512
        + admission.batches.len() as u64 * 128
        + 8_192;
    let input_credit = operator.reserve_workspace(scratch)?;
    let mut buffers = operator.state.encoding_owner_allocation().0;
    let mut owners = BTreeSet::new();
    for (ordinal, (identity, _)) in admission.rows.iter().enumerate() {
        cooperate(ordinal, context).await?;
        for encoding in [&identity.1, &identity.2] {
            if let Some((address, bytes)) = encoding.allocation() {
                if owners.insert(address) {
                    buffers = super::super::checked(name, buffers, bytes)?;
                }
            }
        }
    }
    let mut retained = buffers;
    let mut output = buffers
        + admission.right_capacities.len() as u64 * 256
        + admission.batches.len() as u64 * 16
        + 1_024;
    let kind = operator.state.sequence_kinds[1];
    for (key, count) in &admission.right_capacities {
        let empty = state::RightBucket::with_sequence_kind(kind);
        let bucket = operator.state.right.get(key).unwrap_or(&empty);
        retained = super::super::checked(name, retained, bucket.metadata_bytes() + 256)?;
        output = super::super::checked(name, output, bucket.projected_admission_bytes(*count))?;
    }
    input_credit
        .try_grow(usize::try_from(retained).map_err(|_| {
            super::super::reason(
                name,
                crate::StreamingFailureReason::AsofCounterOverflow,
                "ASOF parallel admission extent overflowed",
            )
        })?)
        .map_err(|_| {
            super::super::reason(
                name,
                crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
                "ASOF parallel admission input credit exceeded",
            )
        })?;
    let credit = operator.reserve_workspace(output)?;
    let references = operator
        .state
        .batches
        .admission_references(&admission.batches);
    let mut work = AdmissionWork {
        shards: (0..units).map(|_| Vec::new()).collect(),
        kind,
        #[cfg(test)]
        hook: operator.admission_hook.clone(),
        _input_credit: input_credit,
    };
    let mut routes = Vec::with_capacity(admission.right_capacities.len());
    for (key, count) in &admission.right_capacities {
        let shard = state::key_shard(key) % units;
        let target = &mut work.shards[shard];
        routes.push((shard, target.len()));
        target.push(InputBucket {
            key: key.clone(),
            previous: operator.state.right.owned_bucket(key),
            rows: Vec::with_capacity(*count),
        });
    }
    for (ordinal, (identity, payload)) in admission.rows.iter().enumerate() {
        cooperate(ordinal, context).await?;
        let (shard, bucket) = routes[payload.key_index as usize];
        let row = references[payload.batch_index].with_row(payload.row);
        work.shards[shard][bucket]
            .rows
            .push(((identity.0, identity.2.clone()), row));
    }
    Ok((work, references, credit))
}

async fn cooperate(ordinal: usize, context: &StreamOperatorContext<'_>) -> Result<()> {
    if ordinal.is_multiple_of(1_024) {
        context.check_cancelled()?;
    }
    if ordinal > 0 && ordinal.is_multiple_of(8_192) {
        tokio::task::yield_now().await;
    }
    Ok(())
}
