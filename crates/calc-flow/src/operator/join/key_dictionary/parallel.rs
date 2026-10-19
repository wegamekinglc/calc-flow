use std::{ops::Range, sync::Arc};

use datafusion::execution::memory_pool::MemoryReservation;

use super::super::{
    PayloadRef,
    admitted::AdmittedGroup,
    columnar::{COPY_BYTES, COPY_VISITS, RowPayload},
    counter_overflow,
};
use super::{
    BirthBudget, KeyFlow,
    local::{self, LocalColumn, LocalIntern},
};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, ParallelCpuWork, cleanup_control_bytes,
};
use crate::{Result, StreamOperatorContext};

const MIN_ROWS: usize = 64_000;
const MIN_ROWS_PER_UNIT: usize = 16_000;

#[cfg(test)]
#[path = "tests/cpu_budget.rs"]
mod cpu_budget_tests;

fn work_units(rows: usize, cpu_budget: usize) -> usize {
    cpu_budget.min(rows / MIN_ROWS_PER_UNIT)
}

struct KeyWork {
    payload: RowPayload,
    indices: Vec<usize>,
    offsets: Vec<(usize, bool)>,
    units: usize,
    name: Arc<str>,
    credit: Arc<MemoryReservation>,
    #[cfg(test)]
    hook: Option<super::tests::WorkerHook>,
}

struct Fragment {
    local: LocalIntern,
    start: usize,
    _credit: Arc<MemoryReservation>,
}

impl KeyWork {
    fn range(&self, ordinal: usize) -> Range<usize> {
        let width = self.offsets.len() / self.units;
        let remainder = self.offsets.len() % self.units;
        let start = ordinal * width + ordinal.min(remainder);
        let length = width + usize::from(ordinal < remainder);
        start..start + length
    }
}

impl ParallelCpuWork for KeyWork {
    type Output = Option<Fragment>;

    fn unit_count(&self) -> usize {
        self.units
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<Self::Output> {
        stop.check()?;
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(ordinal);
        }
        let columns = local::bind_columns(PayloadRef::from(&self.payload), &self.indices)
            .expect("prebound local key types");
        let range = self.range(ordinal);
        let mut local = LocalIntern::new(range.len(), self.credit.new_empty());
        let mut visits = 0;
        let mut bytes = 0;
        for (position, &(offset, retain)) in self.offsets[range.clone()].iter().enumerate() {
            let Some(width) = local::bounded_row_bytes(&columns, offset) else {
                return Ok(None);
            };
            let (distinct, _) = local.intern_row(&columns, offset, &self.name)?;
            local.locals.push(distinct);
            if retain {
                local.note_retained(
                    distinct,
                    u32::try_from(position).expect("group rows fit u32 ordinals"),
                    &self.name,
                )?;
            }
            visits += 1;
            bytes += width;
            if visits >= COPY_VISITS || bytes >= COPY_BYTES {
                stop.check()?;
                visits = 0;
                bytes = 0;
            }
        }
        stop.check()?;
        Ok(Some(Fragment {
            local,
            start: range.start,
            _credit: Arc::clone(&self.credit),
        }))
    }
}

pub(super) async fn intern_group(
    intern: &mut LocalIntern,
    columns: &[LocalColumn<'_>],
    group: AdmittedGroup<'_>,
    indices: &[usize],
    budget: &BirthBudget<'_>,
    flow: &mut KeyFlow<'_, '_>,
    context: Option<&StreamOperatorContext<'_>>,
) -> Result<bool> {
    let Some(context) = context.filter(|_| group.len() >= MIN_ROWS) else {
        return Ok(false);
    };
    let units = work_units(group.len(), context.job().gather_owner().parallelism());
    if units < 2 || columns.len() > COPY_VISITS {
        return Ok(false);
    }
    let Some(work) = capture(group, indices, units, budget, flow).await? else {
        return Ok(false);
    };
    let Some(fragments) = dispatch(work, context).await? else {
        return Ok(false);
    };
    if fragments.iter().any(Option::is_none) {
        return Ok(false);
    }
    for fragment in fragments.into_iter().flatten() {
        merge_fragment(intern, fragment, columns, budget.name, flow).await?;
    }
    Ok(true)
}

fn work_bytes(
    rows: usize,
    columns: usize,
    payload_columns: usize,
    units: usize,
    name: &str,
) -> Result<usize> {
    super::local_table_bytes(rows)
        .and_then(|bytes| bytes.checked_add(rows.checked_mul(size_of::<(usize, bool)>())?))
        .and_then(|bytes| bytes.checked_add(unit_control_bytes(columns, units)?))
        .and_then(|bytes| {
            bytes.checked_add(
                payload_columns.checked_mul(size_of::<datafusion::arrow::array::ArrayRef>())?,
            )
        })
        .and_then(|bytes| bytes.checked_add(name.len()))
        .and_then(|bytes| {
            bytes.checked_add(cleanup_control_bytes::<Vec<Option<Fragment>>>() + 4096)
        })
        .ok_or_else(|| counter_overflow(name, "parallel_key_capacity"))
}

fn unit_control_bytes(columns: usize, units: usize) -> Option<usize> {
    let headers = units
        .saturating_sub(1)
        .checked_mul(super::local_table_bytes(0)?)?;
    let indices = columns.checked_mul(size_of::<usize>())?;
    let bound_columns = columns
        .checked_mul(size_of::<LocalColumn<'_>>())?
        .checked_mul(units)?;
    headers.checked_add(indices)?.checked_add(bound_columns)
}

async fn capture(
    group: AdmittedGroup<'_>,
    indices: &[usize],
    units: usize,
    budget: &BirthBudget<'_>,
    flow: &mut KeyFlow<'_, '_>,
) -> Result<Option<Arc<KeyWork>>> {
    let record = group.row(0).record;
    let bytes = work_bytes(
        group.len(),
        indices.len(),
        record.num_columns(),
        units,
        budget.name,
    )?;
    let Some(credit) = super::grant(budget.master, bytes, budget.name)? else {
        return Ok(None);
    };
    let payload = match record {
        PayloadRef::Record(record, _) => RowPayload::from(record.clone()),
        PayloadRef::Payload(payload, _) => payload.clone(),
    };
    let mut offsets = Vec::with_capacity(group.len());
    for (position, row) in group.key_offsets().enumerate() {
        offsets.push(row);
        if (position + 1).is_multiple_of(COPY_VISITS) {
            flow.step(COPY_VISITS, 0).await?;
        }
    }
    flow.step(group.len() % COPY_VISITS, 0).await?;
    Ok(Some(Arc::new(KeyWork {
        payload,
        indices: indices.to_vec(),
        offsets,
        units,
        name: Arc::from(budget.name),
        credit: Arc::new(credit),
        #[cfg(test)]
        hook: super::tests::worker_hook(),
    })))
}

async fn dispatch(
    work: Arc<KeyWork>,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<Vec<Option<Fragment>>>> {
    let scope = context
        .gather_client(GatherOperatorId::new(Arc::clone(&work.name)))
        .scope()?;
    let credit = work.credit.new_empty();
    let stop = GatherStop::from_job(context.job());
    let retirement = context.job().gather_owner().retain_retirement()?;
    let mut cleanup = None;
    let ticket = scope
        .submit_observed_parallel_work(work, credit, stop.clone(), retirement, &mut cleanup)
        .await;
    let result = match ticket {
        Ok(ticket) => {
            let result = tokio::select! {
                biased;
                () = super::super::output_gather::cancelled(context.job()) => {
                    context.check_cancelled()?;
                    unreachable!("key stop requires cancellation or deadline");
                }
                result = ticket.finish() => result,
            };
            context.check_cancelled()?;
            let mut fragments = None;
            result?.install(|values| {
                fragments = Some(values);
                Ok(())
            })?;
            Ok(fragments)
        }
        Err(AdmissionFailure::Budget { .. }) => Ok(None),
        Err(AdmissionFailure::Runtime(error)) => Err(error),
    };
    context.check_cancelled()?;
    if let Some(cleanup) = cleanup {
        cleanup.wait(&stop).await?;
    }
    context.check_cancelled()?;
    result
}

async fn merge_fragment(
    intern: &mut LocalIntern,
    mut fragment: Fragment,
    columns: &[LocalColumn<'_>],
    name: &str,
    flow: &mut KeyFlow<'_, '_>,
) -> Result<()> {
    let base = u32::try_from(fragment.start).expect("group rows fit u32 ordinals");
    for distinct in 0..fragment.local.distinct {
        let ordinal = base + fragment.local.appearances[distinct];
        let offset = fragment.local.rows[distinct];
        let global = intern.intern_hashed(
            columns,
            offset,
            fragment.local.hashes[distinct],
            ordinal,
            name,
        )?;
        fragment.local.posts.push(global);
        let global = global as usize;
        intern.retains[global] = intern.retains[global]
            .checked_add(fragment.local.retains[distinct])
            .ok_or_else(|| counter_overflow(name, "key_references"))?;
        let first = fragment.local.first_retained[distinct];
        if first != u32::MAX {
            intern.first_retained[global] = intern.first_retained[global].min(base + first);
        }
        flow.step(
            1,
            local::bounded_row_bytes(columns, offset).expect("bounded worker keys"),
        )
        .await?;
    }
    for ids in fragment.local.locals.chunks(COPY_VISITS) {
        intern
            .locals
            .extend(ids.iter().map(|&id| fragment.local.posts[id as usize]));
        flow.step(ids.len(), 0).await?;
    }
    Ok(())
}
