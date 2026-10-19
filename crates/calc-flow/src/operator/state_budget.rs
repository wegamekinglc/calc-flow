//! Reusable logical row and byte limits for retained stream state.

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

use crate::{CalcFlowError, Result};

const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;

/// Maximum retained rows and logical bytes for one stream state component.
///
/// The byte count is an operator-defined deterministic logical charge; it is
/// not a process-wide resident-memory measurement.
/// The default uses the largest JSON-safe counters. Use [`StateBudget::new`]
/// to impose operational row and byte limits.
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
            max_rows: MAX_SAFE_JSON_INTEGER,
            max_bytes: MAX_SAFE_JSON_INTEGER,
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

#[cfg(test)]
mod tests {
    use super::StateBudget;

    #[test]
    fn cpu_aware_default_state_budget_allows_large_incoming_batches() {
        assert!(
            StateBudget::default().allows(2_000_000, 600 << 20),
            "default retained-state limits must admit large batches without an implicit byte ceiling"
        );
    }

    #[test]
    fn cpu_aware_default_state_budget_preserves_json_roundtrip() {
        let budget = StateBudget::default();
        let encoded = serde_json::to_value(budget).unwrap();
        assert_eq!(encoded["max_rows"].as_u64(), Some(budget.max_rows()));
        assert_eq!(encoded["max_bytes"].as_u64(), Some(budget.max_bytes()));
        assert_eq!(
            serde_json::from_value::<StateBudget>(encoded).unwrap(),
            budget
        );

        let maximum = StateBudget::new(9_007_199_254_740_991, 9_007_199_254_740_991).unwrap();
        let encoded = serde_json::to_string(&maximum).unwrap();
        assert_eq!(
            serde_json::from_str::<StateBudget>(&encoded).unwrap(),
            maximum
        );
    }

    #[test]
    fn cpu_aware_explicit_state_budget_keeps_finite_row_and_byte_limits() {
        let budget = StateBudget::new(1_000_000, 256 << 20).unwrap();
        assert!(budget.allows(1_000_000, 256 << 20));
        assert!(!budget.allows(1_000_001, 256 << 20));
        assert!(!budget.allows(1_000_000, (256 << 20) + 1));
        assert!(!budget.allows(2_000_000, 600 << 20));
        let encoded = serde_json::to_string(&budget).unwrap();
        let restored = serde_json::from_str::<StateBudget>(&encoded).unwrap();
        assert_eq!(restored, budget);
        assert!(!restored.allows(2_000_000, 600 << 20));
    }
}
