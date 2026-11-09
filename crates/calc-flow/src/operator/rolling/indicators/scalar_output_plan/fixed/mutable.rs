use super::super::{Check, Result, invalid, math};
use super::{Cell, Geometry};
use datafusion::arrow::datatypes::DataType;
use math::{add, buffer_headers, bytes, mul, size, sum};
#[path = "mutable/fresh.rs"]
mod fresh;
#[path = "mutable/sources.rs"]
mod sources;
pub(in crate::operator::rolling::indicators::scalar_output_plan) use fresh::plan as scalar_union;
use sources::{Child, Kind, Root, Sources};
#[path = "mutable/buffers.rs"]
pub(super) mod buffers;
#[path = "mutable/bytes.rs"]
mod bytes;
#[path = "mutable/dictionary.rs"]
mod dictionary;
#[path = "mutable/normal.rs"]
mod normal;
#[path = "mutable/run.rs"]
mod run;
#[path = "mutable/union.rs"]
mod union;
#[path = "mutable/views.rs"]
mod views;

pub(super) fn supports(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Union(..)
            | DataType::RunEndEncoded(..)
            | DataType::Dictionary(..)
            | DataType::Struct(_)
            | DataType::List(_)
            | DataType::LargeList(_)
            | DataType::ListView(_)
            | DataType::LargeListView(_)
            | DataType::Map(..)
            | DataType::FixedSizeList(..)
    )
}
pub(super) fn initial(slots: u64) -> Geometry {
    Geometry {
        slots,
        values: 0,
        child_capacity: slots,
        buffers: 2,
        descriptors: 0,
        workspace: 0,
        builders: 1,
    }
}
pub(super) fn complete(
    dtype: &DataType,
    rows: usize,
    repair: bool,
    cell: &mut Cell<'_, '_>,
    base: Geometry,
    check: Check<'_>,
) -> Result<Geometry> {
    let DataType::FixedSizeList(field, width) = dtype else {
        return Err(invalid("fixed list datatype"));
    };
    if !supports(field.data_type()) {
        return Ok(base);
    }
    let width = usize::try_from(*width).map_err(|_| invalid("fixed list width"))?;
    let mut sources = Root {
        count: rows,
        width,
        repair,
        cell,
    };
    let cost = node(field.data_type(), base.slots, &mut sources, check)?;
    Ok(Geometry {
        values: cost.output,
        workspace: cost.scratch,
        buffers: base.buffers,
        builders: add(cost.builders, 1)?,
        descriptors: cost.descriptors,
        ..base
    })
}
#[derive(Clone, Copy, Default)]
struct Cost {
    output: u64,
    scratch: u64,
    descriptors: u64,
    builders: u64,
}
impl Cost {
    fn add(self, other: Self) -> Result<Self> {
        Ok(Self {
            output: add(self.output, other.output)?,
            scratch: add(self.scratch, other.scratch)?,
            descriptors: add(self.descriptors, other.descriptors)?,
            builders: add(self.builders, other.builders)?,
        })
    }
}
fn node(
    dtype: &DataType,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    check()?;
    let buffers = buffers::plan(dtype, capacity, sources, check)?;
    let local = local(dtype, buffers, check)?;
    local.add(descendants(dtype, capacity, sources, check)?)
}
fn local(dtype: &DataType, buffers: buffers::Buffers, check: Check<'_>) -> Result<Cost> {
    let dtype_clone = super::super::clones::datatype(dtype, check)?;
    Ok(Cost {
        output: buffers.output,
        scratch: sum([
            Ok(buffers.scratch),
            buffer_headers(buffers.count),
            Ok(dtype_clone),
        ])?,
        descriptors: 0,
        builders: 1,
    })
}
fn descendants(
    dtype: &DataType,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    if let DataType::Union(fields, mode) = dtype {
        return union::children(fields, *mode, capacity, sources, check);
    }
    if let DataType::RunEndEncoded(ends, value) = dtype {
        return run::plan(
            ends.data_type(),
            value.data_type(),
            capacity,
            sources,
            check,
        );
    }
    if let DataType::Dictionary(key, value) = dtype {
        return dictionary::plan(key, value, sources, check);
    }
    children(dtype, capacity, sources, check)
}

fn children(
    dtype: &DataType,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    match dtype {
        DataType::List(field) => child(field.data_type(), capacity, Kind::List32, sources, check),
        DataType::LargeList(field) => {
            child(field.data_type(), capacity, Kind::List64, sources, check)
        }
        DataType::ListView(field) => {
            child(field.data_type(), capacity, Kind::View32, sources, check)
        }
        DataType::LargeListView(field) => {
            child(field.data_type(), capacity, Kind::View64, sources, check)
        }
        DataType::Map(field, _) => child(field.data_type(), capacity, Kind::Map, sources, check),
        _ => other_children(dtype, capacity, sources, check),
    }
}
fn other_children(
    dtype: &DataType,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    match dtype {
        DataType::FixedSizeList(field, width) => {
            let width = usize::try_from(*width).map_err(|_| invalid("fixed list width"))?;
            child(
                field.data_type(),
                mul(capacity, size(width)?)?,
                Kind::Fixed(width),
                sources,
                check,
            )
        }
        DataType::Struct(fields) => fields_cost(fields, capacity, sources, check),
        _ => Ok(Cost::default()),
    }
}
fn fields_cost(
    fields: &datafusion::arrow::datatypes::Fields,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    let mut cost = Cost::default();
    for (index, field) in fields.iter().enumerate() {
        cost = cost.add(child(
            field.data_type(),
            capacity,
            Kind::Struct(index),
            sources,
            check,
        )?)?;
    }
    Ok(cost)
}
fn child(
    dtype: &DataType,
    capacity: u64,
    kind: Kind,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    let mut child = Child {
        parent: sources,
        kind,
    };
    let mut cost = node(dtype, capacity, &mut child, check)?;
    cost.descriptors = add(
        cost.descriptors,
        bytes::<datafusion::arrow::array::MutableArrayData<'_>>(1)?,
    )?;
    Ok(cost)
}

pub(super) fn normal(
    dtype: &DataType,
    count: usize,
    rows: u64,
    at: &mut super::super::concatenated::At<'_>,
    check: Check<'_>,
) -> Result<Geometry> {
    let cost = normal::plan(dtype, count, rows, at, check)?;
    Ok(Geometry {
        slots: rows,
        values: cost.output,
        child_capacity: rows,
        buffers: 0,
        descriptors: cost.descriptors,
        workspace: cost.scratch,
        builders: cost.builders,
    })
}
