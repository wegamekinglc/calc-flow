use std::mem::size_of;

use super::{
    super::{BufferedRow, Result, ScalarValue, TableBatch, operator_error},
    resources,
};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;
mod classify;
pub(super) mod retained;
mod selection;
mod worker;

fn row_descriptors(rows: usize, columns: usize, node: &str) -> Result<u64> {
    let cells = columns
        .checked_mul(size_of::<ScalarValue>())
        .and_then(|cells| cells.checked_add(256 + size_of::<BufferedRow>()))
        .and_then(|row| row.checked_mul(rows))
        .ok_or_else(|| overflow(node))?;
    resources::size(cells, node)
}

fn materialization_error(error: crate::CalcFlowError, node: &str) -> crate::CalcFlowError {
    match error {
        crate::CalcFlowError::Format { .. } | crate::CalcFlowError::InvalidArgument { .. } => {
            overflow(node)
        }
        other => other,
    }
}
fn overflow(node: &str) -> crate::CalcFlowError {
    operator_error(
        node,
        "rolling scalar materialization byte bound is invalid or overflowed",
    )
}
