#[cfg(test)]
#[path = "numeric_diagnostic/tests.rs"]
mod tests;
use std::fmt;

use crate::CalcFlowError;

#[derive(Clone, Copy)]
pub(in super::super) enum Coordinate {
    InputRow(usize),
    PendingRow(usize),
    AcceptedRow(usize),
    OrderedPosition(u64),
}

impl fmt::Display for Coordinate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputRow(row) => {
                write!(formatter, "input row {row}; ordered position not assigned")
            }
            Self::PendingRow(row) => write!(
                formatter,
                "pending row {row}; ordered position not assigned"
            ),
            Self::AcceptedRow(row) => write!(
                formatter,
                "accepted row {row}; ordered position not assigned"
            ),
            Self::OrderedPosition(position) => write!(formatter, "ordered position {position}"),
        }
    }
}

pub(super) fn numeric_error(
    error: CalcFlowError,
    node: &str,
    coordinate: Coordinate,
) -> CalcFlowError {
    match error {
        CalcFlowError::InvalidArgument { field, message } => CalcFlowError::InvalidArgument {
            field,
            message: format!("rolling node {node:?}, {coordinate}: {message}"),
        },
        error => error,
    }
}
