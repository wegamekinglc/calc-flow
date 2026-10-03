//! Immutable native index bytes used by capacity verification.

use crate::StateSegment;

#[derive(Clone)]
pub(in super::super) struct PreparedSegment(StateSegment);

impl PreparedSegment {
    #[cfg(test)]
    pub(in super::super) const fn new(segment: StateSegment) -> Self {
        Self(segment)
    }
    pub(in super::super) fn len(&self) -> usize {
        self.0.bytes().len()
    }
    pub(in super::super) fn capacity(&self) -> usize {
        self.0.bytes_arc().capacity()
    }
    #[cfg(test)]
    pub(in super::super) fn canonical(&self) -> StateSegment {
        self.0.clone()
    }
}
