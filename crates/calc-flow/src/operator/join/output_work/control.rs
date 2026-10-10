use super::{
    bounds::{self, Certificate, Shape},
    funding::{OutputBuffer, OutputFunding},
    inputs::Selection,
    worker::{Fragment, FragmentWork, MergeWork},
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

pub(super) fn fragment_column(
    shape: Shape,
    _data_type: &datafusion::arrow::datatypes::DataType,
    nullable: bool,
) -> Option<usize> {
    let buffers = shape.buffers() + usize::from(nullable);
    sum(&[array_owner(shape)?, bounds::mul(buffers, buffer_owner()?)?])
}

pub(super) fn final_column(shape: Shape, nullable: bool, fragments: usize) -> Option<usize> {
    sum(&[
        rebound_controls(shape, nullable)?,
        builder_controls(shape)?,
        bounds::mul(fragments, size_of::<&dyn datafusion::arrow::array::Array>())?,
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

fn builder_controls(shape: Shape) -> Option<usize> {
    let reset = if let Shape::Bytes(width) = shape {
        4 * width
    } else {
        0
    };
    sum(&[
        4 * size_of::<Buffer>(),
        bounds::mul(2 * shape.buffers(), size_of::<Buffer>())?,
        reset,
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
        release_channel()?,
        bounds::mul(rows, size_of::<Selection>())?,
    ])
}

fn fragments(rows: usize, columns: usize) -> Option<usize> {
    let count = super::units(rows);
    sum(&[
        bounds::mul(rows, 2 * size_of::<u64>())?,
        bounds::mul(count, 2 * buffer_owner()?)?,
        bounds::mul(count.next_power_of_two().max(4), size_of::<Fragment>())?,
        bounds::mul(bounds::mul(count, columns)?, size_of::<ArrayRef>())?,
    ])
}

fn calls(name: &str) -> Option<usize> {
    let caller = super::super::metadata_validation::inventory::caller_controls(name)?;
    let registration = super::super::metadata_validation::inventory::registration_controls()?;
    sum(&[
        size_of::<MergeWork>(),
        cleanup_control_bytes::<Vec<Fragment>>(),
        cleanup_control_bytes::<RecordBatch>(),
        caller,
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
