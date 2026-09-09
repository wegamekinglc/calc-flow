use crate::{CalcFlowError, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use std::{collections::BTreeSet, time::Duration};

const MAX_SAFE: u64 = 9_007_199_254_740_991;

/// Policy for rows strictly below their own ingress watermark.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AsofLatePolicy {
    /// Fail the complete admission attempt.
    #[default]
    Error,
    /// Discard late rows and count them.
    Drop,
}

/// Total logical state limits, shared by both ASOF inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AsofStateLimits {
    #[schemars(range(min = 1, max = 9_007_199_254_740_991_u64))]
    max_state_rows: u64,
    #[schemars(range(min = 1, max = 9_007_199_254_740_991_u64))]
    max_state_bytes: u64,
}

impl AsofStateLimits {
    /// Creates positive limits in the exact JSON integer domain.
    ///
    /// # Errors
    /// Returns an invalid argument when either limit is zero or too large.
    pub fn new(max_state_rows: u64, max_state_bytes: u64) -> Result<Self> {
        positive_safe(max_state_rows, "limits.max_state_rows")?;
        positive_safe(max_state_bytes, "limits.max_state_bytes")?;
        Ok(Self {
            max_state_rows,
            max_state_bytes,
        })
    }
    /// Maximum retained identities across both sides.
    pub const fn max_state_rows(self) -> u64 {
        self.max_state_rows
    }
    /// Maximum charged persistent bytes and independent workspace bytes.
    pub const fn max_state_bytes(self) -> u64 {
        self.max_state_bytes
    }
}

impl<'de> Deserialize<'de> for AsofStateLimits {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            max_state_rows: u64,
            max_state_bytes: u64,
        }
        let fields = Fields::deserialize(deserializer)?;
        Self::new(fields.max_state_rows, fields.max_state_bytes).map_err(D::Error::custom)
    }
}

/// Exact identity and output naming declaration for one ASOF input.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AsofJoinSide {
    #[schemars(length(min = 1))]
    keys: Vec<String>,
    event_time: String,
    #[schemars(length(min = 1))]
    sequence_by: Vec<String>,
    prefix: String,
}

impl AsofJoinSide {
    /// Creates a side declaration without borrowing caller-owned containers.
    ///
    /// # Errors
    /// Rejects empty or repeated column names and non-identifier prefixes.
    pub fn new(
        keys: Vec<String>,
        event_time: String,
        sequence_by: Vec<String>,
        prefix: String,
    ) -> Result<Self> {
        column_names(&keys, "keys")?;
        column_names(&sequence_by, "sequence_by")?;
        if event_time.is_empty() {
            return Err(invalid("event_time", "must not be empty"));
        }
        if !super::super::is_portable_identifier(&prefix) {
            return Err(invalid("prefix", "must be a portable ASCII identifier"));
        }
        Ok(Self {
            keys,
            event_time,
            sequence_by,
            prefix,
        })
    }
    /// Positionally matched key columns.
    pub fn keys(&self) -> &[String] {
        &self.keys
    }
    /// UTC microsecond event time column.
    pub fn event_time(&self) -> &str {
        &self.event_time
    }
    /// Typed lexicographic identity columns.
    pub fn sequence_by(&self) -> &[String] {
        &self.sequence_by
    }
    /// Prefix used before two underscores and the source field name.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

impl<'de> Deserialize<'de> for AsofJoinSide {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            keys: Vec<String>,
            event_time: String,
            sequence_by: Vec<String>,
            prefix: String,
        }
        let f = Fields::deserialize(deserializer)?;
        Self::new(f.keys, f.event_time, f.sequence_by, f.prefix).map_err(D::Error::custom)
    }
}

/// Version-one backward, left-preserving, final ASOF configuration.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StreamAsofJoinSpec {
    left: AsofJoinSide,
    right: AsofJoinSide,
    #[schemars(range(min = 0, max = 9_007_199_254_740_991_u64))]
    tolerance_micros: u64,
    limits: AsofStateLimits,
    late_policy: AsofLatePolicy,
}

impl StreamAsofJoinSpec {
    /// Creates an inclusive backward ASOF declaration with late-row errors.
    ///
    /// # Errors
    /// Rejects sub-microsecond or oversized tolerance, unequal key counts,
    /// and equal output prefixes.
    pub fn new(
        left: AsofJoinSide,
        right: AsofJoinSide,
        tolerance: Duration,
        limits: AsofStateLimits,
    ) -> Result<Self> {
        if tolerance.subsec_nanos() % 1_000 != 0 || tolerance.as_micros() > u128::from(MAX_SAFE) {
            return Err(invalid(
                "tolerance_micros",
                "must be an exact non-negative JSON-safe integer microsecond duration",
            ));
        }
        if left.keys.len() != right.keys.len() {
            return Err(invalid(
                "right.keys",
                "must have the same length as left.keys",
            ));
        }
        if left.prefix == right.prefix {
            return Err(invalid("right.prefix", "must differ from left.prefix"));
        }
        Ok(Self {
            left,
            right,
            tolerance_micros: u64::try_from(tolerance.as_micros())
                .map_err(|_| invalid("tolerance_micros", "duration exceeds the integer domain"))?,
            limits,
            late_policy: AsofLatePolicy::Error,
        })
    }
    /// Selects how rows below their own watermark are handled.
    #[must_use]
    pub const fn with_late_policy(mut self, policy: AsofLatePolicy) -> Self {
        self.late_policy = policy;
        self
    }
    /// Left identity declaration.
    pub const fn left(&self) -> &AsofJoinSide {
        &self.left
    }
    /// Right identity declaration.
    pub const fn right(&self) -> &AsofJoinSide {
        &self.right
    }
    /// Inclusive historical distance in microseconds.
    pub const fn tolerance_micros(&self) -> u64 {
        self.tolerance_micros
    }
    /// Shared persistent and transient limits.
    pub const fn limits(&self) -> AsofStateLimits {
        self.limits
    }
    /// Late-row handling policy.
    pub const fn late_policy(&self) -> AsofLatePolicy {
        self.late_policy
    }
}

impl<'de> Deserialize<'de> for StreamAsofJoinSpec {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            left: AsofJoinSide,
            right: AsofJoinSide,
            tolerance_micros: u64,
            limits: AsofStateLimits,
            late_policy: AsofLatePolicy,
        }
        let f = Fields::deserialize(deserializer)?;
        Self::new(
            f.left,
            f.right,
            Duration::from_micros(f.tolerance_micros),
            f.limits,
        )
        .map(|spec| spec.with_late_policy(f.late_policy))
        .map_err(D::Error::custom)
    }
}

fn positive_safe(value: u64, field: &str) -> Result<()> {
    if value == 0 || value > MAX_SAFE {
        Err(invalid(field, "must be a positive JSON-safe integer"))
    } else {
        Ok(())
    }
}
fn column_names(names: &[String], field: &str) -> Result<()> {
    let unique: BTreeSet<_> = names.iter().collect();
    if names.is_empty() || unique.len() != names.len() || names.iter().any(String::is_empty) {
        return Err(invalid(
            field,
            "must contain distinct non-empty column names",
        ));
    }
    Ok(())
}
pub(super) fn invalid(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: format!("stream_asof_join.{field}"),
        message: message.into(),
    }
}
