use std::sync::Arc;

use datafusion::execution::memory_pool::MemoryReservation;

use super::{StoredRow, checkpoint_v2, columnar};
use crate::{CalcFlowError, EventTime, Result, StreamOperatorContext};

// Arrays die before the reservation, including partial construction and replacement.
#[derive(Default)]
struct Storage {
    heap: Vec<usize>,
    positions: Vec<usize>,
    ordinals: Vec<u128>,
    credit: Option<MemoryReservation>,
}

#[derive(Default)]
pub(super) struct ExpirationIndex {
    storage: Storage,
    next_ordinal: u128,
}

pub(super) struct AppendPlan {
    replacement: Option<Storage>,
    next_ordinal: u128,
}

struct Rows<'a> {
    old: &'a [StoredRow],
    incoming: &'a [StoredRow],
}

impl Rows<'_> {
    fn key(&self, index: usize) -> (EventTime, u64) {
        let row = if index < self.old.len() {
            &self.old[index]
        } else {
            &self.incoming[index - self.old.len()]
        };
        (row.event_time, row.row_id)
    }
}

impl Storage {
    fn new(capacity: usize, origin: Option<&MemoryReservation>) -> Result<Self> {
        let credit = origin
            .map(|credit| checkpoint_v2::expiration_credit(credit, capacity, size_of::<Self>()))
            .transpose()?;
        let mut storage = Self {
            credit,
            ..Self::default()
        };
        allocate(&mut storage.heap, capacity)?;
        allocate(&mut storage.positions, capacity)?;
        allocate(&mut storage.ordinals, capacity)?;
        Ok(storage)
    }

    fn swap(&mut self, left: usize, right: usize) {
        self.heap.swap(left, right);
        self.positions[self.heap[left]] = left;
        self.positions[self.heap[right]] = right;
    }

    fn sift_up(&mut self, mut index: usize, rows: &Rows<'_>) {
        while index > 0 {
            let parent = (index - 1) / 2;
            if rows.key(self.heap[parent]) <= rows.key(self.heap[index]) {
                break;
            }
            self.swap(parent, index);
            index = parent;
        }
    }

    fn sift_down(&mut self, mut index: usize, rows: &Rows<'_>) {
        while index < self.heap.len() / 2 {
            let left = index * 2 + 1;
            let right = left + 1;
            let child = if right < self.heap.len()
                && rows.key(self.heap[right]) < rows.key(self.heap[left])
            {
                right
            } else {
                left
            };
            if rows.key(self.heap[index]) <= rows.key(self.heap[child]) {
                break;
            }
            self.swap(index, child);
            index = child;
        }
    }
}

impl ExpirationIndex {
    pub(super) fn restored(rows: &[StoredRow]) -> Result<Self> {
        Self::restore(rows, None, &|| Ok(()))
    }

    pub(super) fn restored_funded(
        rows: &[StoredRow],
        credit: &MemoryReservation,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        Self::restore(rows, Some(credit), check)
    }

    fn restore(
        rows: &[StoredRow],
        credit: Option<&MemoryReservation>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        check()?;
        let mut storage = Storage::new(rows.len(), credit)?;
        for index in 0..rows.len() {
            check()?;
            storage.heap.push(index);
            storage.positions.push(index);
            storage.ordinals.push(index as u128);
        }
        let view = Rows {
            old: rows,
            incoming: &[],
        };
        for index in (0..rows.len() / 2).rev() {
            check()?;
            storage.sift_down(index, &view);
        }
        check()?;
        Ok(Self {
            storage,
            next_ordinal: rows.len() as u128,
        })
    }

    pub(super) async fn prepare_append(
        &self,
        count: usize,
        context: &StreamOperatorContext<'_>,
    ) -> Result<AppendPlan> {
        context.check_cancelled()?;
        let required = self
            .storage
            .heap
            .len()
            .checked_add(count)
            .ok_or_else(overflow)?;
        let next_ordinal = self
            .next_ordinal
            .checked_add(count as u128)
            .ok_or_else(overflow)?;
        let replacement = if required <= self.storage.heap.capacity() {
            None
        } else {
            let capacity = required
                .max(4)
                .checked_next_power_of_two()
                .ok_or_else(overflow)?;
            let mut replacement = Storage::new(capacity, self.storage.credit.as_ref())?;
            let mut quantum = columnar::Quantum::default();
            for start in (0..self.storage.heap.len()).step_by(64) {
                let end = (start + 64).min(self.storage.heap.len());
                quantum
                    .step(context, end - start, (end - start) * 32)
                    .await?;
                replacement
                    .heap
                    .extend_from_slice(&self.storage.heap[start..end]);
                replacement
                    .positions
                    .extend_from_slice(&self.storage.positions[start..end]);
                replacement
                    .ordinals
                    .extend_from_slice(&self.storage.ordinals[start..end]);
            }
            Some(replacement)
        };
        context.check_cancelled()?;
        Ok(AppendPlan {
            replacement,
            next_ordinal,
        })
    }

    pub(super) fn append(&mut self, old: &[StoredRow], incoming: &[StoredRow], plan: AppendPlan) {
        if let Some(replacement) = plan.replacement {
            self.storage = replacement;
        }
        debug_assert_eq!(old.len(), self.storage.heap.len());
        let view = Rows { old, incoming };
        for offset in 0..incoming.len() {
            let dense = old.len() + offset;
            self.storage.heap.push(dense);
            self.storage.positions.push(dense);
            self.storage
                .ordinals
                .push(self.next_ordinal + offset as u128);
            self.storage.sift_up(dense, &view);
        }
        self.next_ordinal = plan.next_ordinal;
    }

    pub(super) fn first_time(&self, rows: &[StoredRow]) -> Option<EventTime> {
        self.storage
            .heap
            .first()
            .map(|index| rows[*index].event_time)
    }

    // The caller immediately performs the corresponding dense row swap_remove.
    pub(super) fn pop(&mut self, rows: &[StoredRow]) -> Option<(usize, u128)> {
        let dense = *self.storage.heap.first()?;
        let ordinal = self.storage.ordinals[dense];
        self.storage.heap.swap_remove(0);
        if let Some(&first) = self.storage.heap.first() {
            self.storage.positions[first] = 0;
            self.storage.sift_down(
                0,
                &Rows {
                    old: rows,
                    incoming: &[],
                },
            );
        }
        self.storage.positions.swap_remove(dense);
        self.storage.ordinals.swap_remove(dense);
        if let Some(&position) = self.storage.positions.get(dense) {
            self.storage.heap[position] = dense;
        }
        Some((dense, ordinal))
    }

    pub(super) fn identities(
        &self,
        rows: &[StoredRow],
    ) -> Vec<(u64, EventTime, Arc<columnar::FramedKey>)> {
        let mut ordered = self
            .storage
            .ordinals
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|(_, ordinal)| *ordinal);
        ordered
            .into_iter()
            .map(|(index, _)| {
                let row = &rows[index];
                (row.row_id, row.event_time, Arc::clone(&row.encoded_key))
            })
            .collect()
    }
}

fn allocate<T>(values: &mut Vec<T>, capacity: usize) -> Result<()> {
    values
        .try_reserve_exact(capacity)
        .map_err(|_| CalcFlowError::Internal {
            message: "Join expiration array allocation failed".into(),
        })?;
    if values.capacity() > capacity {
        return Err(CalcFlowError::Internal {
            message: "Join expiration allocation exceeds admitted capacity".into(),
        });
    }
    Ok(())
}

fn overflow() -> CalcFlowError {
    CalcFlowError::Internal {
        message: "Join expiration capacity or ordinal overflow".into(),
    }
}
