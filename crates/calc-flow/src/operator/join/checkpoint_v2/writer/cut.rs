use crate::{Epoch, Result};

use super::buffer::error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Cut {
    pub(super) generation: u64,
    pub(super) captured_epoch: Option<Epoch>,
    pub(super) revision: u64,
    pub(super) dirty_revision: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct Tracker {
    pub(super) generation: u64,
    pub(super) revision: u64,
    pub(super) dirty_revision: u64,
}

impl Tracker {
    pub(super) fn snapshot(self, captured_epoch: Option<Epoch>) -> Cut {
        Cut {
            generation: self.generation,
            captured_epoch,
            revision: self.revision,
            dirty_revision: self.dirty_revision,
        }
    }

    pub(super) fn advanced(self, dirty: bool) -> Result<Self> {
        Ok(Self {
            revision: increment(self.revision)?,
            dirty_revision: if dirty {
                increment(self.dirty_revision)?
            } else {
                self.dirty_revision
            },
            ..self
        })
    }

    pub(super) fn next_owner(self) -> Result<Self> {
        Ok(Self {
            generation: increment(self.generation)?,
            revision: 0,
            dirty_revision: 0,
        })
    }
}

fn increment(value: u64) -> Result<u64> {
    value
        .checked_add(1)
        .ok_or_else(|| error("V2 checkpoint cut counter overflow"))
}
