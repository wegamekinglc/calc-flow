use std::sync::Arc;

use datafusion::{
    arrow::{array::Array, datatypes::DataType},
    execution::memory_pool::MemoryReservation,
};

use super::super::{bits, mutable_peak, offsets, values};
use super::{accounting, add, selected_bytes, sum};
use crate::Result;
use crate::operator::join::checkpoint_v2::payload::Funding;

#[derive(Clone, Copy)]
enum Layout {
    Primitive(usize),
    Bytes32,
    Bytes64,
}

impl Layout {
    fn for_type(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Utf8 | DataType::Binary => Some(Self::Bytes32),
            DataType::LargeUtf8 | DataType::LargeBinary => Some(Self::Bytes64),
            _ => data_type.primitive_width().map(Self::Primitive),
        }
    }

    fn selected_bytes(self, array: &dyn Array) -> Result<usize> {
        match self {
            Self::Primitive(_) => Ok(0),
            Self::Bytes32 => selected_bytes::<i32>(&array.to_data()),
            Self::Bytes64 => selected_bytes::<i64>(&array.to_data()),
        }
    }

    fn backing(self, rows: usize, bytes: usize) -> Result<usize> {
        let validity = mutable_peak(bits(rows)?, bits(rows)?)?;
        let values = match self {
            Self::Primitive(width) => values(rows, rows, width)?,
            Self::Bytes32 => byte_backing(rows, bytes, size_of::<i32>())?,
            Self::Bytes64 => byte_backing(rows, bytes, size_of::<i64>())?,
        };
        add(validity, values)
    }
}

pub(super) fn admit(
    arrays: &[&dyn Array],
    workspace: &MemoryReservation,
    resident: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<bool> {
    let Some((layout, first)) = compatible_layout(arrays, check)? else {
        return Ok(false);
    };
    // Arrow's typed concat uses one preallocated builder, without per-source extension closures.
    let controls = builder_controls(first.data_type())?;
    accounting::reserve(workspace, controls)?;
    let (rows, bytes) = dimensions(arrays, layout, check)?;
    admit_builder(layout, rows, bytes, controls, resident)?;
    check()?;
    Ok(true)
}

fn compatible_layout<'a>(
    arrays: &[&'a dyn Array],
    check: &dyn Fn() -> Result<()>,
) -> Result<Option<(Layout, &'a dyn Array)>> {
    let Some(first) = arrays.first() else {
        return Ok(None);
    };
    let Some(layout) = Layout::for_type(first.data_type()) else {
        return Ok(None);
    };
    for array in arrays {
        check()?;
        if array.data_type() != first.data_type() {
            return Ok(None);
        }
    }
    Ok(Some((layout, *first)))
}

fn builder_controls(data_type: &DataType) -> Result<usize> {
    sum(&[
        accounting::array_controls(1, 3)?,
        accounting::data_type_bytes(data_type)?,
    ])
}

fn admit_builder(
    layout: Layout,
    rows: usize,
    bytes: usize,
    controls: usize,
    resident: &Arc<Funding>,
) -> Result<()> {
    resident.grow(add(layout.backing(rows, bytes)?, controls)?)
}

fn dimensions(
    arrays: &[&dyn Array],
    layout: Layout,
    check: &dyn Fn() -> Result<()>,
) -> Result<(usize, usize)> {
    arrays.iter().try_fold((0, 0), |(rows, bytes), array| {
        check()?;
        Ok((
            add(rows, array.len())?,
            add(bytes, layout.selected_bytes(*array)?)?,
        ))
    })
}

fn byte_backing(rows: usize, bytes: usize, width: usize) -> Result<usize> {
    add(offsets(rows, rows, width)?, mutable_peak(bytes, rows)?)
}
