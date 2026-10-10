use super::{
    bounds::{self, Certificate, Shape},
    funding::{OutputBuffer, OutputFunding},
    worker::{Fragment, FragmentWork},
};
use crate::runtime::streaming::gather_work::cleanup_control_bytes;
use datafusion::arrow::{
    array::{ArrayRef, BooleanArray, Int64Array, StringArray},
    buffer::Buffer,
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::{alloc::Layout, sync::atomic::AtomicUsize};

fn sum(parts: &[usize]) -> Option<usize> {
    parts
        .iter()
        .try_fold(0usize, |total, part| bounds::add(total, *part))
}

fn arc<T>() -> Option<usize> {
    Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .ok()
        .map(|(layout, _)| layout.pad_to_align().size())
}

fn release_channel() -> Option<usize> {
    arc::<(
        AtomicUsize,
        Option<()>,
        [std::mem::MaybeUninit<std::task::Waker>; 2],
    )>()
}

// Locked Arrow 58.3 without its optional pool feature.
fn buffer_owner() -> Option<usize> {
    bounds::add(2 * size_of::<usize>(), 5 * size_of::<usize>())
}

fn array_owner(shape: Shape) -> Option<usize> {
    match shape {
        Shape::Bits => arc::<BooleanArray>(),
        Shape::Primitive(_) => arc::<Int64Array>(),
        Shape::Bytes(_) => arc::<StringArray>(),
    }
}

fn wrapped_owner() -> Option<usize> {
    let owned = Layout::new::<AtomicUsize>()
        .extend(Layout::new::<OutputBuffer>())
        .ok()?
        .0
        .pad_to_align()
        .size();
    sum(&[owned, arc::<tokio_util::bytes::Bytes>()?, buffer_owner()?])
}

pub(super) fn final_column(shape: Shape, nullable: bool) -> Option<usize> {
    sum(&[
        rebound_controls(shape, nullable)?,
        bounds::mul(2 * shape.buffers(), size_of::<Buffer>())?,
    ])
}

fn rebound_controls(shape: Shape, nullable: bool) -> Option<usize> {
    let buffers = shape.buffers() + usize::from(nullable);
    sum(&[
        array_owner(shape)?,
        array_owner(shape)?,
        bounds::mul(buffers, bounds::add(buffer_owner()?, wrapped_owner()?)?)?,
    ])
}

pub(super) fn base(rows: usize, columns: usize, name: &str) -> Option<Certificate> {
    Some(Certificate {
        workspace: sum(&[snapshot(rows)?, fragments(rows, columns)?, calls(name)?])?,
        output: output_controls(columns, name)?,
    })
}

fn snapshot(rows: usize) -> Option<usize> {
    sum(&[
        arc::<FragmentWork>()?,
        arc::<MemoryReservation>()?,
        release_channel()?,
        bounds::mul(rows, 2 * size_of::<u64>())?,
        2 * buffer_owner()?,
    ])
}

fn fragments(rows: usize, columns: usize) -> Option<usize> {
    let count = super::units(rows, columns);
    sum(&[
        bounds::mul(count.next_power_of_two().max(4), size_of::<Fragment>())?,
        bounds::mul(columns, size_of::<ArrayRef>())?,
    ])
}

fn calls(name: &str) -> Option<usize> {
    let caller = super::super::metadata_validation::inventory::caller_controls(name)?;
    let registration = super::super::metadata_validation::inventory::registration_controls()?;
    sum(&[
        cleanup_control_bytes::<Vec<Fragment>>(),
        caller,
        registration,
    ])
}

fn output_controls(columns: usize, name: &str) -> Option<usize> {
    sum(&[
        arc::<OutputFunding>()?,
        super::super::metadata_validation::inventory::registration_controls()?,
        bounds::mul(columns, size_of::<ArrayRef>())?,
        batch_controls()?,
        metadata_source(name)?,
    ])
}

fn metadata_source(name: &str) -> Option<usize> {
    let bytes = Layout::array::<u8>(name.len()).ok()?;
    let shared = Layout::new::<[AtomicUsize; 2]>()
        .extend(bytes)
        .ok()?
        .0
        .pad_to_align()
        .size();
    bounds::add(name.len(), shared)
}

fn batch_controls() -> Option<usize> {
    sum(&[
        size_of::<RecordBatch>(),
        arc::<RecordBatch>()?,
        arc::<std::sync::OnceLock<Result<usize, String>>>()?,
        arc::<crate::JsonMap>()?,
    ])
}
