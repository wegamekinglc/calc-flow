//! New owners created by the controlled layout-five decoder, excluding its pinned IPC record.

use datafusion::arrow::{datatypes::Schema, record_batch::RecordBatch};

use super::super::CompiledRollingSpec;
use crate::{CalcFlowError, Result};

mod alloc;
mod cells;
mod census;
mod containers;
mod json;
mod source;
mod windows;

use alloc::add;
use source::Source;
type Check<'a> = &'a mut dyn FnMut() -> Result<()>;
#[derive(Clone, Copy, Default)]
pub(super) struct Plan {
    pub(super) candidate_bytes: u64,
    pub(super) scratch_bytes: u64,
}
impl Plan {
    fn merge(self, other: Self) -> Result<Self> {
        Ok(Self {
            candidate_bytes: add(self.candidate_bytes, other.candidate_bytes)?,
            scratch_bytes: add(self.scratch_bytes, other.scratch_bytes)?,
        })
    }
}
pub(super) fn plan(
    record: &RecordBatch,
    input: &Schema,
    compiled: &CompiledRollingSpec,
    check: Check<'_>,
) -> Result<Plan> {
    check()?;
    let counts = census::scan(record, check)?;
    let source = Source::new(record, input, compiled, counts)?;
    let plan = requirements(&source, check)?;
    check()?;
    Ok(plan)
}
pub(super) fn retained_rows(record: &RecordBatch, check: Check<'_>) -> Result<u64> {
    census::scan(record, check)?.retained_rows()
}
pub(super) fn vector_construction<T>(count: usize) -> Result<u64> {
    let vector = alloc::growing::<T>(count)?;
    add(vector.retained, vector.overlap)
}
fn requirements(source: &Source<'_>, check: Check<'_>) -> Result<Plan> {
    let containers = containers::plan(source, check)?;
    let cells = cells::plan(source, check)?;
    let windows = windows::plan(source, check)?;
    containers.merge(cells)?.merge(windows)
}
fn invalid() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling restore workspace geometry is invalid".into(),
    }
}
#[cfg(test)]
mod tests;
