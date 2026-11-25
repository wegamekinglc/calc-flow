//! Shared-key columnar state and selection-driven Arrow ASOF kernels.

mod checkpoint;
mod compact;
mod materialize;
mod run;
mod state;

use run::RowLocator;
pub(super) use run::{logical_filtered_record_charge, logical_record_selection_charge};
#[cfg(test)]
pub(super) use state::EvictionSummary;
pub(super) use state::{ColumnarState, Progress};
#[cfg(test)]
pub(super) use state::{take_finality_probes, take_selections};

use datafusion::arrow::{array::BooleanArray, buffer::BooleanBuffer};

#[cfg(test)]
thread_local! {
    pub(super) static VERIFIED_RUN_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) struct Selection {
    pub(super) left: Vec<RowLocator>,
    pub(super) right: Vec<RowLocator>,
    pub(super) right_validity: BooleanBuffer,
}

impl Selection {
    pub(super) fn new(left: Vec<RowLocator>, right: Vec<RowLocator>, validity: Vec<bool>) -> Self {
        debug_assert_eq!(left.len(), right.len());
        debug_assert_eq!(left.len(), validity.len());
        let right_validity = BooleanArray::from(validity).values().clone();
        Self {
            left,
            right,
            right_validity,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.left.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.left.is_empty()
    }

    pub(super) fn matched(&self) -> usize {
        self.right_validity.count_set_bits()
    }
}

#[cfg(test)]
mod tests;
