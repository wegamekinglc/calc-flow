use super::{
    inputs::ProbeInputs,
    pairs::PairFragment,
    partition,
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

fn work_bytes() -> Result<usize> {
    sum(&[
        arc::<CountWork>()?,
        arc::<FillWork>()?,
        cleanup_control_bytes::<Vec<usize>>(),
        cleanup_control_bytes::<Vec<PairFragment>>(),
    ])
}

fn fragment_controls(rows: usize) -> Result<usize> {
    if partition::units(rows) < 2 {
        return Ok(0);
    }
    sum(&[
        arc::<MemoryReservation>()?,
        super::super::metadata_validation::inventory::registration_controls()
            .ok_or_else(|| super::super::native_lookup::scratch_error("join"))?,
    ])
}

pub(super) fn input_bytes(keys: usize, rows: usize, name: &str) -> Result<usize> {
    let caller = super::super::metadata_validation::inventory::caller_controls(name)
        .ok_or_else(|| super::super::native_lookup::scratch_error(name))?;
    let registration = super::super::metadata_validation::inventory::registration_controls()
        .ok_or_else(|| super::super::native_lookup::scratch_error(name))?;
    sum(&[
        snapshot_bytes(keys, name)?,
        work_bytes()?,
        fragment_controls(rows)?,
        caller,
        caller,
        registration,
        registration,
    ])
}
