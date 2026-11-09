//! Additional owned chunks and temporary construction heap; caller-owned
//! canonical record arrays and backing remain paid by the surrounding work.

use crate::{EdgeBudget, Result};
use datafusion::arrow::record_batch::RecordBatch;

use super::math_requirements::bytes;
use bytes::{growing, repeat, slots, total};

#[path = "chunk_requirements/flat.rs"]
mod flat;
#[path = "chunk_requirements/geometry.rs"]
mod geometry;
#[path = "chunk_requirements/shape.rs"]
mod shape;

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ChunkRequirements {
    pub output_bytes: u64,
    pub scratch_bytes: u64,
}

pub(super) fn batch_envelope(columns: usize, node: &str) -> Result<ChunkRequirements> {
    Ok(ChunkRequirements {
        output_bytes: geometry::batch(columns, node.len())?,
        scratch_bytes: geometry::batch_temporary(node.len())?,
    })
}

pub(super) fn plan(
    record: &RecordBatch,
    budget: EdgeBudget,
    node: &str,
    check: Check<'_>,
) -> Result<ChunkRequirements> {
    check()?;
    if record.num_rows() == 0 {
        return Ok(ChunkRequirements::default());
    }
    let shape = shape::record(record, check)?;
    let fast = whole(record, budget, check)?;
    let cost = requirements(record, node, &shape, fast)?;
    check()?;
    Ok(cost)
}
fn whole(record: &RecordBatch, budget: EdgeBudget, check: Check<'_>) -> Result<bool> {
    if record.num_rows() > budget.max_rows {
        return Ok(false);
    }
    Ok(flat::record(record, check)?
        .is_some_and(|bytes| u64::try_from(budget.max_bytes).is_ok_and(|limit| bytes <= limit)))
}
fn requirements(
    record: &RecordBatch,
    node: &str,
    shape: &shape::Shape,
    fast: bool,
) -> Result<ChunkRequirements> {
    let chunks = if fast { 1 } else { record.num_rows() };
    let batch = geometry::batch(record.num_columns(), node.len());
    let descriptors = if fast { 0 } else { shape.descriptors };
    let output_bytes = total([
        repeat(total([batch, Ok(descriptors)]), chunks),
        output_vector(chunks, fast),
    ])?;
    let scratch_bytes = scratch(record.num_rows(), node, shape, fast)?;
    Ok(ChunkRequirements {
        output_bytes,
        scratch_bytes,
    })
}
fn output_vector(chunks: usize, fast: bool) -> Result<u64> {
    if fast {
        slots::<crate::Batch>(1)
    } else {
        growing::<crate::Batch>(chunks)
    }
}
fn scratch(rows: usize, node: &str, shape: &shape::Shape, fast: bool) -> Result<u64> {
    let base = total([Ok(shape.metadata), geometry::batch_temporary(node.len())]);
    if fast {
        return base;
    }
    total([
        base,
        Ok(shape.slice_peak),
        slots::<usize>(rows),
        growing::<crate::operator::row_cost::Offsets<'_>>(shape.nodes),
        growing::<&datafusion::arrow::buffer::NullBuffer>(shape.nodes),
        slots::<usize>(shape.nodes),
        growing::<(usize, usize)>(rows),
    ])
}

#[cfg(test)]
#[path = "chunk_requirements/tests.rs"]
mod tests;
