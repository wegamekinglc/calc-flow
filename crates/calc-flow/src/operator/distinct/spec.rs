use crate::{EventTime, LateErrorScope, LatePolicySpec, Result, StateBudget, dataframe::invalid};
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
#[schemars(extend(
    "if" = {"properties":{"history":{"properties":{"kind":{"const":"bounded_source"}}}}},
    "then" = {"properties":{"late_policy":{"properties":{"kind":{"const":"error"}}}}}
))]
pub struct DistinctSpec {
    ingress_policy: IngressPolicy,
    history: DistinctHistory,
    state_budget: StateBudget,
    #[serde(default = "default_policy", skip_serializing_if = "is_error")]
    #[schemars(with = "LocalLatePolicySchema")]
    late_policy: LatePolicySpec,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    ingress_policy: IngressPolicy,
    history: DistinctHistory,
    state_budget: StateBudget,
    #[serde(default = "default_policy")]
    late_policy: LatePolicySpec,
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
        Self {
            ingress_policy: wire.ingress_policy,
            history: wire.history,
            state_budget: wire.state_budget,
            late_policy: default_policy(),
        }
        .with_late_policy(wire.late_policy)
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
            late_policy: default_policy(),
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
            late_policy: default_policy(),
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
    /// Selects this window's persistent policy; global compile options do not override it.
    ///
    /// # Errors
    /// Rejects side output, unsupported metric versions and bounded-source Drop.
    pub fn with_late_policy(mut self, policy: LatePolicySpec) -> Result<Self> {
        let supported = match policy {
            LatePolicySpec::Error {
                scope: LateErrorScope::Envelope,
            } => true,
            LatePolicySpec::Drop { metrics_version: 1 } => {
                matches!(self.history, DistinctHistory::ClosedWindow { .. })
            }
            _ => false,
        };
        if !supported {
            return Err(invalid(
                "distinct.spec.late_policy",
                "invalid_literal",
                "requires Error(envelope), or closed-window Drop(metrics_version=1)",
            ));
        }
        self.late_policy = policy;
        Ok(self)
    }
    /// Returns the local persistent late policy.
    pub const fn late_policy(&self) -> LatePolicySpec {
        self.late_policy
    }
    pub(crate) const fn state_layout(&self) -> u32 {
        match self.late_policy {
            LatePolicySpec::Drop { .. } => 3,
            _ => 2,
        }
    }
}

const fn default_policy() -> LatePolicySpec {
    LatePolicySpec::Error {
        scope: LateErrorScope::Envelope,
    }
}
fn is_error(policy: &LatePolicySpec) -> bool {
    *policy == default_policy()
}
struct LocalLatePolicySchema;
impl JsonSchema for LocalLatePolicySchema {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "DistinctLatePolicy".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({"oneOf":[
            {"type":"object","additionalProperties":false,"required":["kind","scope"],"properties":{"kind":{"const":"error"},"scope":{"const":"envelope"}}},
            {"type":"object","additionalProperties":false,"required":["kind","metrics_version"],"properties":{"kind":{"const":"drop"},"metrics_version":{"type":"integer","const":1}}}
        ]})
    }
}
