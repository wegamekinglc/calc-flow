use super::super::{columnar::Quantum, materialization::JoinOutput};
use super::control;
use crate::{Result, StreamOperatorContext};
use datafusion::arrow::{
    array::Array,
    datatypes::{DataType, TimeUnit},
};
use std::ops::Range;

#[derive(Clone, Copy)]
pub(super) enum Shape {
    Bits,
    Primitive(usize),
    Bytes(usize),
}

impl Shape {
    pub(super) fn of(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Boolean => Some(Self::Bits),
            DataType::Utf8 | DataType::Binary => Some(Self::Bytes(4)),
            DataType::LargeUtf8 | DataType::LargeBinary => Some(Self::Bytes(8)),
            DataType::Time32(TimeUnit::Microsecond | TimeUnit::Nanosecond)
            | DataType::Time64(TimeUnit::Second | TimeUnit::Millisecond) => None,
            _ => data_type.primitive_width().map(Self::Primitive),
        }
    }

    pub(super) fn buffers(self) -> usize {
        if matches!(self, Self::Bytes(_)) { 2 } else { 1 }
    }

    fn data(self, rows: usize, selected_bytes: usize) -> Option<usize> {
        match self {
            Self::Bits => bitmap(rows),
            Self::Primitive(width) => mul(rows, width),
            Self::Bytes(offset) => add(mul(rows.checked_add(1)?, offset)?, selected_bytes),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Certificate {
    pub(super) workspace: usize,
    pub(super) output: usize,
}

pub(super) fn add(left: usize, right: usize) -> Option<usize> {
    left.checked_add(right)
        .filter(|bytes| isize::try_from(*bytes).is_ok())
}

pub(super) fn mul(left: usize, right: usize) -> Option<usize> {
    left.checked_mul(right)
        .filter(|bytes| isize::try_from(*bytes).is_ok())
}

fn bitmap(rows: usize) -> Option<usize> {
    mul(rows.div_ceil(64), 8)
}

pub(super) async fn certify(
    materializer: &JoinOutput<'_>,
    range: Range<usize>,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<Certificate>> {
    if super::units(range.len(), materializer.schema.fields().len()) < 2
        || materializer.schema.fields().is_empty()
    {
        return Ok(None);
    }
    if !same_parents(materializer, range.clone(), context).await? {
        return Ok(None);
    }
    let Some(mut required) = control::base(
        range.len(),
        materializer.schema.fields().len(),
        materializer.operator_id,
    ) else {
        return Ok(None);
    };
    for column in 0..materializer.schema.fields().len() {
        let Some(cost) = column_cost(materializer, range.clone(), column, context).await? else {
            return Ok(None);
        };
        let Some(workspace) = add(required.workspace, cost.workspace) else {
            return Ok(None);
        };
        let Some(output) = add(required.output, cost.output) else {
            return Ok(None);
        };
        required = Certificate { workspace, output };
    }
    Ok(Some(required))
}

async fn same_parents(
    materializer: &JoinOutput<'_>,
    range: Range<usize>,
    context: &StreamOperatorContext<'_>,
) -> Result<bool> {
    let first = materializer.payloads(&materializer.matched[range.start]);
    let (Some(left), Some(right)) = (first.0.shared_columns(), first.1.shared_columns()) else {
        return Ok(false);
    };
    let mut quantum = Quantum::default();
    for pair in &materializer.matched[range] {
        quantum.step(context, 1, 16).await?;
        let next = materializer.payloads(pair);
        if !next
            .0
            .shared_columns()
            .is_some_and(|columns| std::ptr::eq(columns, left))
            || !next
                .1
                .shared_columns()
                .is_some_and(|columns| std::ptr::eq(columns, right))
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn column<'a>(
    materializer: &'a JoinOutput<'_>,
    pair: &super::super::MatchedPair,
    index: usize,
) -> (&'a dyn Array, usize) {
    let (left, right) = materializer.payloads(pair);
    if index < left.num_columns() {
        (left.column(index).as_ref(), left.offset())
    } else {
        (
            right.column(index - left.num_columns()).as_ref(),
            right.offset(),
        )
    }
}

async fn selected_bytes(
    materializer: &JoinOutput<'_>,
    range: Range<usize>,
    index: usize,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<usize>> {
    let mut quantum = Quantum::default();
    let mut bytes = 0usize;
    for pair in &materializer.matched[range] {
        quantum.step(context, 1, 16).await?;
        let (array, row) = column(materializer, pair, index);
        let Some(width) = selected_width(array, row) else {
            return Ok(None);
        };
        let Some(total) = add(bytes, width) else {
            return Ok(None);
        };
        bytes = total;
    }
    Ok(Some(bytes))
}

fn selected_width(array: &dyn Array, row: usize) -> Option<usize> {
    use crate::operator::row_cost::{Offsets, variable_offsets};
    match variable_offsets(array)? {
        Offsets::Narrow(offsets) => {
            usize::try_from(offsets[row + 1].checked_sub(offsets[row])?).ok()
        }
        Offsets::Wide(offsets) => usize::try_from(offsets[row + 1].checked_sub(offsets[row])?).ok(),
    }
}

async fn column_cost(
    materializer: &JoinOutput<'_>,
    range: Range<usize>,
    index: usize,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<Certificate>> {
    let data_type = materializer.schema.field(index).data_type();
    let Some(shape) = Shape::of(data_type) else {
        return Ok(None);
    };
    let bytes = if matches!(shape, Shape::Bytes(_)) {
        let Some(bytes) = selected_bytes(materializer, range.clone(), index, context).await? else {
            return Ok(None);
        };
        bytes
    } else {
        0
    };
    if matches!(shape, Shape::Bytes(4)) && bytes > i32::MAX as usize {
        return Ok(None);
    }
    let nullable = column(materializer, &materializer.matched[range.start], index)
        .0
        .nulls()
        .is_some();
    Ok(cost(shape, range.len(), bytes, nullable))
}

fn cost(shape: Shape, rows: usize, bytes: usize, nullable: bool) -> Option<Certificate> {
    let data = add(
        shape.data(rows, bytes)?,
        if nullable { bitmap(rows)? } else { 0 },
    )?;
    Some(Certificate {
        workspace: 0,
        output: add(data, control::final_column(shape, nullable)?)?,
    })
}
