use super::super::native_lookup::{NativeIndex, WindowCursor};
use super::{
    inputs::ProbeInputs,
    pairs::PairFragment,
    partition::{CountFragment, CountRows},
    worker::{CountWork, FillWork},
};
use crate::Result;
use crate::runtime::streaming::gather_work::cleanup_control_bytes;
use datafusion::execution::memory_pool::MemoryReservation;
use std::{alloc::Layout, sync::atomic::AtomicUsize};

fn sum(parts: &[usize]) -> Result<usize> {
    parts.iter().try_fold(0usize, |total, part| {
        total
            .checked_add(*part)
            .ok_or_else(|| super::super::native_lookup::scratch_error("join"))
    })
}

fn arc<T>() -> Result<usize> {
    Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .map(|(layout, _)| layout.pad_to_align().size())
        .map_err(|_| super::super::native_lookup::scratch_error("join"))
}

fn release_channel() -> Result<usize> {
    // Pinned Tokio oneshot layout, also accounted by the V2 writer.
    arc::<(
        AtomicUsize,
        Option<()>,
        [std::mem::MaybeUninit<std::task::Waker>; 2],
    )>()
}

fn snapshot_bytes(keys: usize, name: &str) -> Result<usize> {
    let slots = keys
        .checked_mul(size_of::<Option<u32>>())
        .ok_or_else(|| super::super::native_lookup::scratch_error(name))?;
    sum(&[
        arc::<ProbeInputs>()?,
        arc::<MemoryReservation>()?,
        release_channel()?,
        slots,
    ])
}

fn work_bytes(units: usize) -> Result<usize> {
    let results = units
        .checked_mul(size_of::<CountFragment>() + size_of::<usize>() + size_of::<PairFragment>())
        .ok_or_else(|| super::super::native_lookup::scratch_error("join"))?;
    sum(&[
        arc::<CountRows>()?,
        arc::<CountWork>()?,
        arc::<FillWork>()?,
        cleanup_control_bytes::<Vec<CountFragment>>(),
        cleanup_control_bytes::<Vec<PairFragment>>(),
        results,
    ])
}

pub(super) fn window_guards_bytes() -> Result<usize> {
    // Weak guards keep their Arc headers alive even after snapshot retirement.
    sum(&[arc::<MemoryReservation>()?, arc::<NativeIndex>()?])
}

fn fragment_controls(units: usize) -> Result<usize> {
    if units < 2 {
        return Ok(0);
    }
    sum(&[
        arc::<MemoryReservation>()?,
        super::super::metadata_validation::inventory::registration_controls()
            .ok_or_else(|| super::super::native_lookup::scratch_error("join"))?,
    ])
}

// Unit-local window cursors: each gather unit of the count and fill phases
// allocates one cursor per distinct probe key, so peak use is one set per unit.
fn cursor_bytes(keys: usize, units: usize, name: &str) -> Result<usize> {
    keys.checked_mul(units)
        .and_then(|cursors| cursors.checked_mul(size_of::<Option<WindowCursor>>()))
        .ok_or_else(|| super::super::native_lookup::scratch_error(name))
}

pub(super) fn input_bytes(keys: usize, units: usize, name: &str) -> Result<usize> {
    let caller = super::super::metadata_validation::inventory::caller_controls(name)
        .ok_or_else(|| super::super::native_lookup::scratch_error(name))?;
    let registration = super::super::metadata_validation::inventory::registration_controls()
        .ok_or_else(|| super::super::native_lookup::scratch_error(name))?;
    sum(&[
        snapshot_bytes(keys, name)?,
        work_bytes(units)?,
        fragment_controls(units)?,
        cursor_bytes(keys, units, name)?,
        caller,
        caller,
        registration,
        registration,
        // Deferred count snapshot, fallback lease and fused work descriptors.
        4096,
    ])
}
