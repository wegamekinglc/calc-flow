use crate::{EventTime, Result, StateBudget, dataframe::invalid};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Complete input domain for final-only DISTINCT.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DistinctHistory {
    /// The fixed finite cut promised by a bounded source.
    BoundedSource,
    /// One fixed UTC half-open interval, finalized by watermark or normal EOF.
    ClosedWindow {
        /// An unchanged or purely renamed timestamp input column.
        event_time_column: String,
        /// Inclusive UTC microsecond coordinate.
        start_micros: i64,
        /// Exclusive UTC microsecond coordinate.
        end_micros: i64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum IngressPolicy {
    CopyVisibleV2,
}

/// DISTINCT domain and mandatory owned peak limits.
///
/// Every input is copied. Unknown external backing temporarily borrowed during
/// synchronous admission is outside `max_bytes`; retained data is always owned.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DistinctSpec {
    ingress_policy: IngressPolicy,
    history: DistinctHistory,
    state_budget: StateBudget,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    ingress_policy: IngressPolicy,
    history: DistinctHistory,
    state_budget: StateBudget,
}
impl<'de> Deserialize<'de> for DistinctSpec {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        Wire::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}
impl TryFrom<Wire> for DistinctSpec {
    type Error = crate::CalcFlowError;
    fn try_from(wire: Wire) -> Result<Self> {
        if let DistinctHistory::ClosedWindow {
            event_time_column,
            start_micros,
            end_micros,
        } = &wire.history
        {
            validate_window(event_time_column, *start_micros, *end_micros)?;
        }
        Ok(Self {
            ingress_policy: wire.ingress_policy,
            history: wire.history,
            state_budget: wire.state_budget,
        })
    }
}
fn validate_window(name: &str, start: i64, end: i64) -> Result<()> {
    if name.is_empty() || start >= end {
        return Err(invalid(
            "distinct.spec.history",
            "invalid_literal",
            "closed_window requires a nonempty time column and start < end for [start,end)",
        ));
    }
    Ok(())
}
impl DistinctSpec {
    /// Selects a fixed UTC interval, awaiting its actual closure.
    ///
    /// Uses `copy_visible_v2`: node-owned peak is bounded; the temporary raw
    /// ingress loan and other graph owners are outside this quota.
    ///
    /// # Errors
    /// Rejects empty timestamp names and non-increasing boundaries.
    pub fn closed_window(
        event_time_column: &str,
        start: EventTime,
        end: EventTime,
        state_budget: StateBudget,
    ) -> Result<Self> {
        validate_window(event_time_column, start.as_micros(), end.as_micros())?;
        Ok(Self {
            ingress_policy: IngressPolicy::CopyVisibleV2,
            history: DistinctHistory::ClosedWindow {
                event_time_column: event_time_column.into(),
                start_micros: start.as_micros(),
                end_micros: end.as_micros(),
            },
            state_budget,
        })
    }
    /// Selects a bounded source's complete history, finalized only at normal EOF.
    /// Uses `copy_visible_v2`: the budget covers node-owned peak, excluding the
    /// temporary raw ingress loan and allocations held by other graph owners.
    pub const fn bounded_source(state_budget: StateBudget) -> Self {
        Self {
            ingress_policy: IngressPolicy::CopyVisibleV2,
            history: DistinctHistory::BoundedSource,
            state_budget,
        }
    }
    /// Returns the historical domain.
    pub const fn history(&self) -> &DistinctHistory {
        &self.history
    }
    /// Returns inclusive node-owned limits.
    pub const fn state_budget(&self) -> StateBudget {
        self.state_budget
    }
}
