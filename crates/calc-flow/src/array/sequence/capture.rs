use std::{any::Any, collections::BTreeMap, sync::Arc};

use parking_lot::Mutex;

use super::codec::{self, SequenceSpec, SequenceState};
use crate::{Result, StateSegment, array::ArrayExecutionOptions};

type Key = (usize, usize);

#[derive(Default)]
struct Pool {
    total: usize,
    entries: BTreeMap<Key, Entry>,
}

struct Allocations {
    bytes: Arc<Vec<u8>>,
    checksum: Arc<String>,
}

impl Allocations {
    fn new(segment: &StateSegment) -> Self {
        let (bytes, checksum) = segment.shared_allocations();
        Self { bytes, checksum }
    }

    fn key(&self) -> Key {
        (
            Arc::as_ptr(&self.bytes) as usize,
            Arc::as_ptr(&self.checksum) as usize,
        )
    }
}

struct Entry {
    // Pin both allocations while their identity remains registered.
    _allocations: Allocations,
    groups: usize,
    bytes: usize,
}

#[derive(Clone, Default)]
pub(in crate::array) struct Captures {
    pool: Arc<Mutex<Pool>>,
}

struct Credit {
    pool: Arc<Mutex<Pool>>,
    bytes: usize,
}

struct Owner {
    previous: Option<Arc<dyn Any + Send + Sync>>,
    pool: Arc<Mutex<Pool>>,
    key: Key,
}

struct Plan {
    credit: Credit,
    other_live: usize,
    encoding_overhead: usize,
}

impl Captures {
    pub(in crate::array) fn bytes(&self) -> usize {
        self.pool.lock().total
    }

    pub(in crate::array) fn other_bytes(&self, segment: &StateSegment) -> Result<usize> {
        let key = Allocations::new(segment).key();
        let pool = self.pool.lock();
        let shared = pool.entries.get(&key).map_or(0, |entry| entry.bytes);
        pool.total
            .checked_sub(shared)
            .ok_or_else(|| codec::mismatch("peak_bytes", "capture credit invariant failed"))
    }

    pub(in crate::array) fn capture(
        &self,
        spec: &SequenceSpec,
        state: &SequenceState,
        execution: &ArrayExecutionOptions,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<StateSegment> {
        let plan = self.plan(state, execution)?;
        let bytes = codec::encode(spec, state, execution, plan.encoding_overhead, check)?;
        let segment = StateSegment::new_controlled(bytes, check)?;
        Self::finish(segment, plan, execution, check)
    }

    fn plan(&self, state: &SequenceState, execution: &ArrayExecutionOptions) -> Result<Plan> {
        let scratch = StateSegment::controlled_scratch_bytes();
        let other_live = sum(&[state.state_bytes(), scratch])?;
        let charge = sum(&[state.encoded_bytes()?, 64])?;
        let credit = self.reserve(charge, other_live, execution)?;
        let others = self
            .bytes()
            .checked_sub(credit.bytes)
            .ok_or_else(|| codec::mismatch("peak_bytes", "capture credit invariant failed"))?;
        let encoding_overhead = sum(&[others, 64, scratch - size_of::<u64>()])?;
        Ok(Plan {
            credit,
            other_live,
            encoding_overhead,
        })
    }

    fn finish(
        segment: StateSegment,
        mut plan: Plan,
        execution: &ArrayExecutionOptions,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<StateSegment> {
        plan.credit
            .adjust(segment_charge(&segment)?, plan.other_live, execution)?;
        check()?;
        let owner = plan.credit.install(Allocations::new(&segment))?;
        Ok(attach(segment, owner))
    }

    pub(in crate::array) fn retain(
        &self,
        segment: StateSegment,
        execution: &ArrayExecutionOptions,
        other_live: usize,
    ) -> Result<StateSegment> {
        codec::admit(execution, &[self.bytes(), other_live])?;
        let allocations = Allocations::new(&segment);
        if self.owns(&segment, allocations.key()) {
            return Ok(segment);
        }
        let owner = self.register(
            allocations,
            segment_charge(&segment)?,
            other_live,
            execution,
        )?;
        Ok(attach(segment, owner))
    }

    fn owns(&self, segment: &StateSegment, key: Key) -> bool {
        segment
            .accounting_owner::<Owner>()
            .is_some_and(|owner| Arc::ptr_eq(&owner.pool, &self.pool) && owner.key == key)
    }

    fn register(
        &self,
        allocations: Allocations,
        bytes: usize,
        other_live: usize,
        execution: &ArrayExecutionOptions,
    ) -> Result<Owner> {
        let key = allocations.key();
        let mut pool = self.pool.lock();
        codec::admit(execution, &[pool.total, other_live])?;
        pool.add(allocations, bytes, other_live, execution)?;
        Ok(Owner {
            previous: None,
            pool: self.pool.clone(),
            key,
        })
    }

    fn reserve(
        &self,
        bytes: usize,
        other_live: usize,
        execution: &ArrayExecutionOptions,
    ) -> Result<Credit> {
        let mut pool = self.pool.lock();
        let next = sum(&[pool.total, bytes])?;
        codec::admit(execution, &[next, other_live])?;
        pool.total = next;
        Ok(Credit {
            pool: self.pool.clone(),
            bytes,
        })
    }
}

impl Pool {
    fn add(
        &mut self,
        allocations: Allocations,
        bytes: usize,
        other_live: usize,
        execution: &ArrayExecutionOptions,
    ) -> Result<()> {
        let key = allocations.key();
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.groups = entry
                .groups
                .checked_add(1)
                .ok_or_else(|| codec::mismatch("peak_bytes", "capture owner count overflowed"))?;
            return Ok(());
        }
        let total = sum(&[self.total, bytes])?;
        codec::admit(execution, &[total, other_live])?;
        self.entries.insert(
            key,
            Entry {
                _allocations: allocations,
                groups: 1,
                bytes,
            },
        );
        self.total = total;
        Ok(())
    }

    fn release(&mut self, key: Key) {
        let entry = self
            .entries
            .get_mut(&key)
            .expect("capture allocation remains registered");
        entry.groups = entry
            .groups
            .checked_sub(1)
            .expect("capture owner remains live until drop");
        if entry.groups == 0 {
            let entry = self
                .entries
                .remove(&key)
                .expect("last capture group remains registered");
            self.total = self
                .total
                .checked_sub(entry.bytes)
                .expect("capture allocation remains reserved");
            drop(entry);
        }
    }
}

impl Credit {
    fn adjust(
        &mut self,
        bytes: usize,
        other_live: usize,
        execution: &ArrayExecutionOptions,
    ) -> Result<()> {
        let mut pool = self.pool.lock();
        let rest = pool
            .total
            .checked_sub(self.bytes)
            .ok_or_else(|| codec::mismatch("peak_bytes", "capture credit invariant failed"))?;
        let next = sum(&[rest, bytes])?;
        codec::admit(execution, &[next, other_live])?;
        pool.total = next;
        self.bytes = bytes;
        Ok(())
    }

    fn install(mut self, allocations: Allocations) -> Result<Owner> {
        let key = allocations.key();
        let mut pool = self.pool.lock();
        if pool.entries.contains_key(&key) {
            return Err(codec::mismatch(
                "peak_bytes",
                "fresh capture allocation already registered",
            ));
        }
        pool.entries.insert(
            key,
            Entry {
                _allocations: allocations,
                groups: 1,
                bytes: self.bytes,
            },
        );
        self.bytes = 0;
        Ok(Owner {
            previous: None,
            pool: self.pool.clone(),
            key,
        })
    }
}

impl Drop for Credit {
    fn drop(&mut self) {
        if self.bytes > 0 {
            let mut pool = self.pool.lock();
            pool.total = pool
                .total
                .checked_sub(self.bytes)
                .expect("capture credit remains reserved until owner drop");
        }
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        self.pool.lock().release(self.key);
        drop(self.previous.take());
    }
}

fn attach(segment: StateSegment, mut owner: Owner) -> StateSegment {
    segment.map_owner(move |previous| {
        owner.previous = previous;
        Arc::new(owner)
    })
}

fn segment_charge(segment: &StateSegment) -> Result<usize> {
    segment
        .retained_buffer_bytes()
        .ok_or_else(|| codec::mismatch("peak_bytes", "segment capacity overflowed usize"))
}

fn sum(charges: &[usize]) -> Result<usize> {
    charges
        .iter()
        .try_fold(0_usize, |total, bytes| total.checked_add(*bytes))
        .ok_or_else(|| codec::mismatch("peak_bytes", "live allocation count overflowed usize"))
}

pub(in crate::array) struct Preload {
    credit: Credit,
    old: usize,
    scratch: usize,
}
impl Captures {
    pub(in crate::array) fn preload(
        &self,
        length: usize,
        old: usize,
        execution: &ArrayExecutionOptions,
    ) -> Result<Preload> {
        let scratch = length
            .min(4096)
            .max(StateSegment::controlled_scratch_bytes());
        let credit = self.reserve(sum(&[length, 64])?, sum(&[old, scratch])?, execution)?;
        Ok(Preload {
            credit,
            old,
            scratch,
        })
    }
}
impl Preload {
    pub(in crate::array) fn allocate(
        &mut self,
        length: usize,
        execution: &ArrayExecutionOptions,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        check()?;
        let scratch = self.read_buffer(length.min(4096), execution)?;
        check()?;
        let bytes = self.encoded_buffer(length, execution)?;
        check()?;
        Ok((bytes, scratch))
    }
    fn read_buffer(&mut self, length: usize, execution: &ArrayExecutionOptions) -> Result<Vec<u8>> {
        let mut scratch = allocate_bytes(length)?;
        self.scratch = scratch
            .capacity()
            .max(StateSegment::controlled_scratch_bytes());
        self.credit.adjust(
            self.credit.bytes,
            sum(&[self.old, self.scratch])?,
            execution,
        )?;
        scratch.resize(length, 0);
        Ok(scratch)
    }
    fn encoded_buffer(
        &mut self,
        length: usize,
        execution: &ArrayExecutionOptions,
    ) -> Result<Vec<u8>> {
        let bytes = allocate_bytes(length)?;
        self.credit.adjust(
            sum(&[bytes.capacity(), 64])?,
            sum(&[self.old, self.scratch])?,
            execution,
        )?;
        Ok(bytes)
    }
    pub(in crate::array) fn finish(
        mut self,
        segment: StateSegment,
        execution: &ArrayExecutionOptions,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<StateSegment> {
        self.credit.adjust(
            segment_charge(&segment)?,
            sum(&[self.old, self.scratch])?,
            execution,
        )?;
        check()?;
        let owner = self.credit.install(Allocations::new(&segment))?;
        Ok(attach(segment, owner))
    }
}
fn allocate_bytes(length: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|error| codec::mismatch("encoded_bytes", &error.to_string()))?;
    Ok(bytes)
}
