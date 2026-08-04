//! Strongly typed time and checkpoint identity for the v3 stream runtime.
//!
//! `EventTime` (D1) and `Epoch` (D9) replace the v2 opaque UUID occurrence:
//! every watermark, window bound, and checkpoint barrier now carries a
//! business value with a total order.

mod event_time;

pub use event_time::EventTime;

use serde::{Deserialize, Serialize};

use crate::{CalcFlowError, Result};

/// A checkpoint identifier (spec D9).
///
/// Epochs start at `1` for a fresh job lineage and increase by exactly one
/// per injected checkpoint barrier; `0` is reserved as the "no checkpoint"
/// sentinel and is unconstructable through [`Epoch::new`]. Deserialization
/// routes through the same guard, so a persisted `0` is rejected rather than
/// resurrecting the sentinel.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "u64")]
pub struct Epoch(u64);

impl Epoch {
    /// The first checkpoint of a fresh job lineage (D9.2).
    pub const INITIAL: Epoch = Epoch(1);

    /// Wraps a non-zero epoch value; returns `None` for the reserved `0`.
    pub fn new(value: u64) -> Option<Self> {
        (value != 0).then_some(Self(value))
    }

    /// Returns the exact epoch value.
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Returns the next epoch in the lineage (D9.2).
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Internal`] when the counter is exhausted.
    pub fn next(self) -> Result<Self> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or_else(|| CalcFlowError::Internal {
                message: "epoch counter exhausted".into(),
            })
    }
}

impl TryFrom<u64> for Epoch {
    type Error = CalcFlowError;

    fn try_from(value: u64) -> Result<Self> {
        Self::new(value).ok_or_else(|| CalcFlowError::InvalidArgument {
            field: "epoch".into(),
            message: "0 is the reserved no-checkpoint sentinel".into(),
        })
    }
}
