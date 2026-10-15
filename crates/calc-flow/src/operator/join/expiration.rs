//! One expiration head per sorted incoming run, using stable chunk coordinates.

use super::{StoredRow, checkpoint_v2, retained::Rows};
use crate::{CalcFlowError, EventTime, Result, StreamOperatorContext};
use datafusion::execution::memory_pool::MemoryReservation;
use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap},
    sync::Arc,
};

#[derive(Clone, Copy)]
struct Entry {
    index: usize,
    ordinal: u128,
}

struct Run {
    entries: Vec<Entry>,
    start: usize,
    _credit: Option<MemoryReservation>,
}

type Head = Reverse<(EventTime, u64, usize)>;

#[derive(Default)]
pub(super) struct ExpirationIndex {
    runs: BTreeMap<usize, Run>,
    heads: BinaryHeap<Head>,
    next_run: usize,
    next_ordinal: u128,
    maximum: Option<EventTime>,
    origin: Option<MemoryReservation>,
    heap_credit: Option<MemoryReservation>,
    restored: bool,
}

pub(super) struct AppendPlan {
    entries: Vec<Entry>,
    credit: Option<MemoryReservation>,
    next_ordinal: u128,
    first: Option<(EventTime, u64)>,
    maximum: Option<EventTime>,
    heap: Option<(BinaryHeap<Head>, Option<MemoryReservation>)>,
}

impl ExpirationIndex {
    pub(super) fn restored(rows: &dyn Rows) -> Result<Self> {
        Self::restore(rows, None, &|| Ok(()))
    }

    pub(super) fn restored_funded(
        rows: &dyn Rows,
        credit: &MemoryReservation,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        Self::restore(rows, Some(credit), check)
    }

    fn restore(
        rows: &dyn Rows,
        credit: Option<&MemoryReservation>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        check()?;
        let mut index = Self {
            origin: credit.map(MemoryReservation::new_empty),
            restored: credit.is_some(),
            ..Self::default()
        };
        index.heap_credit = credit
            .map(|credit| checkpoint_v2::expiration_credit(credit, 0, size_of::<Self>()))
            .transpose()?;
        let mut plan = index.plan(rows.len(), false)?;
        for (ordinal, position) in rows.references().enumerate() {
            check()?;
            plan.entries.push(Entry {
                index: position,
                ordinal: ordinal as u128,
            });
            index.maximum = index.maximum.max(Some(rows.identity(position).0));
        }
        index.install(plan, rows);
        check()?;
        Ok(index)
    }

    fn plan(&mut self, count: usize, legacy: bool) -> Result<AppendPlan> {
        let next_ordinal = self
            .next_ordinal
            .checked_add(count as u128)
            .ok_or_else(overflow)?;
        let credit = if count == 0 {
            None
        } else {
            self.reserve(count, size_of::<Run>() + 256, legacy)?
        };
        let mut entries = Vec::new();
        entries.try_reserve_exact(count).map_err(|_| overflow())?;
        let heap = self.prepare_heads(count, legacy)?;
        Ok(AppendPlan {
            entries,
            credit,
            next_ordinal,
            first: None,
            maximum: None,
            heap,
        })
    }

    pub(super) fn fund_origin(&mut self, master: &MemoryReservation) {
        if self.origin.is_none() {
            self.origin = Some(master.new_empty());
        }
    }

    fn prepare_heads(
        &self,
        count: usize,
        legacy: bool,
    ) -> Result<Option<(BinaryHeap<Head>, Option<MemoryReservation>)>> {
        let required = self
            .heads
            .len()
            .checked_add(usize::from(count != 0))
            .ok_or_else(overflow)?;
        if required <= self.heads.capacity() {
            return Ok(None);
        }
        let capacity = required
            .max(4)
            .checked_next_power_of_two()
            .ok_or_else(overflow)?;
        let credit = self.reserve(capacity, size_of::<Self>(), legacy)?;
        let mut heads = BinaryHeap::with_capacity(capacity);
        heads.extend(self.heads.iter().copied());
        Ok(Some((heads, credit)))
    }

    fn reserve(
        &self,
        capacity: usize,
        controls: usize,
        legacy: bool,
    ) -> Result<Option<MemoryReservation>> {
        let Some(origin) = &self.origin else {
            return Ok(None);
        };
        if legacy && !self.restored {
            return checkpoint_v2::optional_expiration_credit(origin, capacity, controls);
        }
        checkpoint_v2::expiration_credit(origin, capacity, controls).map(Some)
    }

    fn install_heads(&mut self, heap: Option<(BinaryHeap<Head>, Option<MemoryReservation>)>) {
        if let Some((heads, credit)) = heap {
            let previous = std::mem::replace(&mut self.heads, heads);
            let old_credit = std::mem::replace(&mut self.heap_credit, credit);
            drop((previous, old_credit));
        }
    }

    #[cfg(test)]
    pub(super) fn funded_bytes(&self) -> usize {
        self.heap_credit.as_ref().map_or(0, MemoryReservation::size)
            + self
                .runs
                .values()
                .filter_map(|run| {
                    let Run {
                        _credit: credit, ..
                    } = run;
                    credit.as_ref()
                })
                .map(MemoryReservation::size)
                .sum::<usize>()
    }

    pub(super) async fn prepare_append(
        &mut self,
        incoming: &[StoredRow],
        indices: &[usize],
        context: &StreamOperatorContext<'_>,
        legacy: bool,
    ) -> Result<AppendPlan> {
        context.check_cancelled()?;
        let mut plan = self.plan(incoming.len(), legacy)?;
        let mut quantum = super::columnar::Quantum::default();
        for (offset, &index) in indices.iter().enumerate() {
            quantum.step(context, 1, size_of::<Entry>()).await?;
            plan.entries.push(Entry {
                index,
                ordinal: self.next_ordinal + offset as u128,
            });
        }
        plan.entries.sort_unstable_by_key(|entry| {
            let row = &incoming[(entry.ordinal - self.next_ordinal) as usize];
            (row.event_time, row.row_id)
        });
        plan.first = plan.entries.first().map(|entry| {
            let row = &incoming[(entry.ordinal - self.next_ordinal) as usize];
            (row.event_time, row.row_id)
        });
        plan.maximum = incoming.iter().map(|row| row.event_time).max();
        context.check_cancelled()?;
        Ok(plan)
    }

    pub(super) fn append(&mut self, mut plan: AppendPlan) {
        self.install_heads(plan.heap.take());
        self.maximum = self.maximum.max(plan.maximum);
        self.next_ordinal = plan.next_ordinal;
        if let Some((time, row_id)) = plan.first {
            self.install_sorted(plan, time, row_id);
        }
    }

    fn install(&mut self, mut plan: AppendPlan, rows: &dyn Rows) {
        self.install_heads(plan.heap.take());
        plan.entries
            .sort_unstable_by_key(|entry| rows.identity(entry.index));
        self.next_ordinal = plan.next_ordinal;
        if let Some(entry) = plan.entries.first() {
            let (time, row_id) = rows.identity(entry.index);
            self.install_sorted(plan, time, row_id);
        }
    }

    fn install_sorted(&mut self, plan: AppendPlan, time: EventTime, row_id: u64) {
        let id = self.next_run;
        self.next_run += 1;
        self.runs.insert(
            id,
            Run {
                entries: plan.entries,
                start: 0,
                _credit: plan.credit,
            },
        );
        self.heads.push(Reverse((time, row_id, id)));
    }

    pub(super) fn first_time(&self, _rows: &dyn Rows) -> Option<EventTime> {
        self.heads.peek().map(|head| head.0.0)
    }

    pub(super) fn last_time(&self) -> Option<EventTime> {
        self.maximum
    }

    pub(super) fn clear(&mut self) {
        self.heads.clear();
        self.runs.clear();
        self.maximum = None;
    }

    pub(super) fn pop(&mut self, rows: &dyn Rows) -> Option<(usize, u128)> {
        let Reverse((_, _, id)) = self.heads.pop()?;
        #[cfg(test)]
        super::note_join_work(|work| work.expiration_head_steps += 1);
        let run = self.runs.get_mut(&id).expect("live expiration run");
        let entry = run.entries[run.start];
        run.start += 1;
        if let Some(next) = run.entries.get(run.start) {
            let (time, row_id) = rows.identity(next.index);
            self.heads.push(Reverse((time, row_id, id)));
        } else {
            self.runs.remove(&id);
        }
        if self.heads.is_empty() {
            self.maximum = None;
        }
        Some((entry.index, entry.ordinal))
    }

    pub(super) fn identities(
        &self,
        rows: &dyn Rows,
    ) -> Vec<(u64, EventTime, Arc<super::columnar::FramedKey>)> {
        let mut ordered = self
            .runs
            .values()
            .flat_map(|run| run.entries[run.start..].iter())
            .collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|entry| entry.ordinal);
        ordered
            .into_iter()
            .map(|entry| {
                let (time, row_id) = rows.identity(entry.index);
                (row_id, time, Arc::clone(rows.key(entry.index)))
            })
            .collect()
    }
}

fn overflow() -> CalcFlowError {
    CalcFlowError::Internal {
        message: "Join expiration capacity or ordinal overflow".into(),
    }
}
