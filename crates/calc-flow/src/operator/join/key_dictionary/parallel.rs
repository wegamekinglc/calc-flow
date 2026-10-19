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

mod remap;
pub(super) use remap::Group;

#[cfg(test)]
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct OwnerWork {
    pub(super) captured_offsets: usize,
    pub(super) remapped_ids: usize,
}

#[cfg(test)]
thread_local! {
    static OWNER_WORK: std::cell::Cell<OwnerWork> = const { std::cell::Cell::new(OwnerWork { captured_offsets: 0, remapped_ids: 0 }) };
}

#[cfg(test)]
pub(super) fn reset_owner_work() {
    OWNER_WORK.set(OwnerWork::default());
}

#[cfg(test)]
pub(super) fn owner_work() -> OwnerWork {
    OWNER_WORK.get()
}

#[cfg(test)]
pub(super) fn note_owner_remap(rows: usize) {
    let work = OWNER_WORK.get();
    OWNER_WORK.set(OwnerWork {
        remapped_ids: work.remapped_ids + rows,
        ..work
    });
}

#[cfg(test)]
#[path = "tests/cpu_budget.rs"]
mod cpu_budget_tests;

fn work_units(rows: usize, cpu_budget: usize) -> usize {
    cpu_budget.min(rows / MIN_ROWS_PER_UNIT)
}

struct KeyWork {
    payload: RowPayload,
    indices: Vec<usize>,
    rows: KeyRows,
    units: usize,
    name: Arc<str>,
    credit: Arc<MemoryReservation>,
    #[cfg(test)]
    hook: Option<super::tests::WorkerHook>,
}

enum KeyRows {
    Dense {
        range: Range<usize>,
        retained: Vec<u64>,
    },
    Sparse(Vec<(usize, bool)>),
}

impl KeyRows {
    fn len(&self) -> usize {
        match self {
            Self::Dense { range, .. } => range.len(),
            Self::Sparse(rows) => rows.len(),
        }
    }

    fn get(&self, ordinal: usize) -> (usize, bool) {
        match self {
            Self::Dense { range, retained } => (
                range.start + ordinal,
                retained[ordinal / 64] & (1_u64 << (ordinal % 64)) != 0,
            ),
            Self::Sparse(rows) => rows[ordinal],
        }
    }
}

struct Fragment {
    local: LocalIntern,
    start: usize,
    _credit: Arc<MemoryReservation>,
}

impl KeyWork {
    fn range(&self, ordinal: usize) -> Range<usize> {
        let width = self.rows.len() / self.units;
        let remainder = self.rows.len() % self.units;
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
        let mut flow = local::WorkerFlow::new(stop)?;
        let mut visits = 0;
        let mut bytes = 0;
        for (position, ordinal) in range.clone().enumerate() {
            let (offset, retain) = self.rows.get(ordinal);
            let Some((distinct, width)) =
                local.intern_worker_row(&columns, offset, &self.name, &mut flow)?
            else {
                return Ok(None);
            };
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
) -> Result<Option<Group>> {
    let Some(context) = context.filter(|_| group.len() >= MIN_ROWS) else {
        return Ok(None);
    };
    let units = work_units(group.len(), context.job().gather_owner().parallelism());
    if units < 2 || columns.len() > COPY_VISITS {
        return Ok(None);
    }
    let Some(work) = capture(group, indices, units, budget, flow).await? else {
        return Ok(None);
    };
    let payload = work.payload.clone();
    let credit = Arc::clone(&work.credit);
    let name = Arc::clone(&work.name);
    let Some(fragments) = dispatch(work, context, &credit, name).await? else {
        return Ok(None);
    };
    if fragments.iter().any(Option::is_none) {
        return Ok(None);
    }
    let mut fragments = fragments.into_iter().flatten().collect::<Vec<_>>();
    let mut flow = local::LocalFlow::new(flow);
    for fragment in &mut fragments {
        merge_fragment(intern, fragment, columns, budget.name, &mut flow).await?;
    }
    Ok(Some(Group::new(fragments, payload, credit)))
}

fn work_bytes(
    rows: usize,
    selection_bytes: usize,
    columns: usize,
    payload_columns: usize,
    units: usize,
    name: &str,
) -> Result<usize> {
    super::local_table_bytes(rows)
        .and_then(|bytes| bytes.checked_add(selection_bytes))
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
    let fragments = units.checked_mul(
        2 * size_of::<Fragment>() + size_of::<parking_lot::Mutex<Option<Fragment>>>(),
    )?;
    headers
        .checked_add(indices)?
        .checked_add(bound_columns)?
        .checked_add(fragments)
}

async fn capture(
    group: AdmittedGroup<'_>,
    indices: &[usize],
    units: usize,
    budget: &BirthBudget<'_>,
    flow: &mut KeyFlow<'_, '_>,
) -> Result<Option<Arc<KeyWork>>> {
    let record = group.row(0).record;
    let dense = group.dense_key_range();
    let selection_bytes = if dense.is_some() {
        group.len().div_ceil(64).checked_mul(size_of::<u64>())
    } else {
        group.len().checked_mul(size_of::<(usize, bool)>())
    }
    .ok_or_else(|| counter_overflow(budget.name, "parallel_key_capacity"))?;
    let bytes = work_bytes(
        group.len(),
        selection_bytes,
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
    let rows = if let Some(range) = dense {
        let mut retained = Vec::with_capacity(group.len().div_ceil(64));
        for start in (0..group.len()).step_by(64) {
            retained.push(group.retained_bits(start, (group.len() - start).min(64)));
            flow.step(1, size_of::<u64>()).await?;
        }
        KeyRows::Dense { range, retained }
    } else {
        let mut offsets = Vec::with_capacity(group.len());
        for (position, row) in group.key_offsets().enumerate() {
            #[cfg(test)]
            {
                let work = OWNER_WORK.get();
                OWNER_WORK.set(OwnerWork {
                    captured_offsets: work.captured_offsets + 1,
                    ..work
                });
            }
            offsets.push(row);
            if (position + 1).is_multiple_of(COPY_VISITS) {
                flow.step(COPY_VISITS, 0).await?;
            }
        }
        flow.step(group.len() % COPY_VISITS, 0).await?;
        KeyRows::Sparse(offsets)
    };
    Ok(Some(Arc::new(KeyWork {
        payload,
        indices: indices.to_vec(),
        rows,
        units,
        name: Arc::from(budget.name),
        credit: Arc::new(credit),
        #[cfg(test)]
        hook: super::tests::worker_hook(),
    })))
}

async fn dispatch<W: ParallelCpuWork>(
    work: Arc<W>,
    context: &StreamOperatorContext<'_>,
    credit: &Arc<MemoryReservation>,
    name: Arc<str>,
) -> Result<Option<Vec<W::Output>>> {
    let scope = context.gather_client(GatherOperatorId::new(name)).scope()?;
    let credit = credit.new_empty();
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
    fragment: &mut Fragment,
    columns: &[LocalColumn<'_>],
    name: &str,
    flow: &mut local::LocalFlow<'_, '_, '_>,
) -> Result<()> {
    let base = u32::try_from(fragment.start).expect("group rows fit u32 ordinals");
    for distinct in 0..fragment.local.distinct {
        let ordinal = base + fragment.local.appearances[distinct];
        let offset = fragment.local.rows[distinct];
        let global = intern
            .merge_hashed(
                columns,
                offset,
                fragment.local.hashes[distinct],
                ordinal,
                name,
                flow,
            )
            .await?;
        fragment.local.posts.push(global);
        let global = global as usize;
        intern.retains[global] = intern.retains[global]
            .checked_add(fragment.local.retains[distinct])
            .ok_or_else(|| counter_overflow(name, "key_references"))?;
        let first = fragment.local.first_retained[distinct];
        if first != u32::MAX {
            intern.first_retained[global] = intern.first_retained[global].min(base + first);
        }
        flow.step(1, 0).await?;
    }
    Ok(())
}
