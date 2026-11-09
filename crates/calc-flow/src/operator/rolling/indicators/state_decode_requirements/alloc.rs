use crate::{CalcFlowError, Result};
use std::{alloc::Layout, mem::size_of};

#[derive(Clone, Copy, Default)]
pub(super) struct Growing {
    pub(super) retained: u64,
    pub(super) overlap: u64,
}
pub(super) fn growing<T>(count: usize) -> Result<Growing> {
    if count == 0 || size_of::<T>() == 0 {
        return Ok(Growing::default());
    }
    let capacity = count
        .max(minimum::<T>())
        .checked_next_power_of_two()
        .ok_or_else(invalid)?;
    let overlap = if capacity > minimum::<T>() {
        exact::<T>(capacity / 2)?
    } else {
        0
    };
    Ok(Growing {
        retained: exact::<T>(capacity)?,
        overlap,
    })
}
fn minimum<T>() -> usize {
    match size_of::<T>() {
        1 => 8,
        2..=1024 => 4,
        _ => 1,
    }
}
pub(super) fn exact<T>(count: usize) -> Result<u64> {
    let layout = Layout::array::<T>(count).map_err(|_| invalid())?;
    u64::try_from(layout.size()).map_err(|_| invalid())
}
pub(super) fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right).ok_or_else(invalid)
}
fn invalid() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling restore allocation size is invalid".into(),
    }
}

#[cfg(test)]
#[path = "alloc_tests.rs"]
mod tests;
