//! Reusable logical row and byte limits for retained stream state.

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

use crate::{CalcFlowError, Result};

/// Maximum retained rows and logical bytes for one stream state component.
///
/// The byte count is an operator-defined deterministic logical charge; it is
/// not a process-wide resident-memory measurement.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StateBudget {
    #[schemars(range(min = 1, max = 9_007_199_254_740_991_u64))]
    max_rows: u64,
    #[schemars(range(min = 1, max = 9_007_199_254_740_991_u64))]
    max_bytes: u64,
}

impl StateBudget {
    /// Creates positive, JSON-safe state limits.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when a limit is zero or
    /// exceeds the largest exactly representable JSON integer.
    pub fn new(max_rows: u64, max_bytes: u64) -> Result<Self> {
        const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;
        for (field, value) in [("max_rows", max_rows), ("max_bytes", max_bytes)] {
            if value == 0 || value > MAX_SAFE_JSON_INTEGER {
                return Err(CalcFlowError::InvalidArgument {
                    field: format!("state_budget.{field}"),
                    message: format!("must be between 1 and {MAX_SAFE_JSON_INTEGER}"),
                });
            }
        }
        Ok(Self {
            max_rows,
            max_bytes,
        })
    }

    /// Maximum retained rows.
    pub const fn max_rows(self) -> u64 {
        self.max_rows
    }

    /// Maximum logical retained bytes.
    pub const fn max_bytes(self) -> u64 {
        self.max_bytes
    }

    /// Returns whether both charges fit this budget.
    pub const fn allows(self, rows: u64, bytes: u64) -> bool {
        rows <= self.max_rows && bytes <= self.max_bytes
    }
}

impl Default for StateBudget {
    fn default() -> Self {
        Self {
            max_rows: 1_000_000,
            max_bytes: 256 << 20,
        }
    }
}

impl<'de> Deserialize<'de> for StateBudget {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            max_rows: u64,
            max_bytes: u64,
        }

        let fields = Fields::deserialize(deserializer)?;
        Self::new(fields.max_rows, fields.max_bytes).map_err(D::Error::custom)
    }
}
