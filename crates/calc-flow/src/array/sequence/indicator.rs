use super::{Budget, Output};
use crate::{
    CalcFlowError, Result,
    array::{ArrayDType, ArrayExecutionOptions, ArrayView, invalid},
};

mod atr;
mod rsi;

pub use atr::{AtrState, atr};
pub use rsi::{RsiState, rsi};

pub(super) fn validate(input: ArrayView<'_>, name: &str) -> Result<()> {
    if input.shape().len() != 1 {
        return Err(invalid(
            &format!("{name}.shape"),
            "expected a rank-one floating vector",
        ));
    }
    if input.dtype() == ArrayDType::Boolean {
        return Err(invalid(
            &format!("{name}.dtype"),
            "expected float32 or float64, received bool",
        ));
    }
    Ok(())
}

pub(super) fn positions(previous: u64, len: usize) -> Result<()> {
    let rows = u64::try_from(len)
        .ok()
        .and_then(|rows| previous.checked_add(rows));
    if rows.is_none() {
        return Err(invalid(
            "input.rows",
            "processed position count would exceed u64::MAX",
        ));
    }
    Ok(())
}

fn allocate(
    len: usize,
    charges: [usize; 3],
    execution: &ArrayExecutionOptions,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Output> {
    Output::allocate(
        &Budget {
            peak_field: "execution.max_allocation_bytes",
            old: charges[0],
            candidate: charges[1],
            captures: charges[2],
            scratch: 0,
            len,
            execution,
        },
        check,
    )
}

fn indexed(error: CalcFlowError, index: usize) -> CalcFlowError {
    let CalcFlowError::InvalidArgument { field, message } = error else {
        return error;
    };
    if field == "input.rows" {
        return invalid(&field, message);
    }
    let message = if message == "must be greater than or equal to inputs.low" {
        format!("must be greater than or equal to inputs.low[{index}]")
    } else {
        message
    };
    invalid(&format!("{field}[{index}]"), message)
}

fn charges(len: usize, eager: bool, state: usize, captures: usize) -> [usize; 3] {
    let old = if eager { 0 } else { state };
    let candidate = if eager || len != 0 { state } else { 0 };
    [old, candidate, captures]
}

#[cfg(test)]
mod tests;
