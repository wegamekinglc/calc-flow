//! Defer historical payload diagnostics until existing restore checks finish.

use std::cell::RefCell;

use super::{StoredRow, validate_restored_row_payload};
use crate::CalcFlowError;

pub(super) type Issue = RefCell<Option<CalcFlowError>>;

#[derive(Clone, Copy)]
pub(super) struct Validation<'a> {
    pub(super) event_index: usize,
    pub(super) issue: &'a Issue,
}

impl Validation<'_> {
    pub(super) fn record(self, row: &StoredRow, key_indices: &[usize], name: &str, side: &str) {
        if self.issue.borrow().is_some() {
            return;
        }
        if let Err(error) =
            validate_restored_row_payload(row, self.event_index, key_indices, name, side)
        {
            self.issue.replace(Some(error));
        }
    }
}
