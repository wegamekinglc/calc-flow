use crate::{
    DistinctSpec, LatePolicySpec, NullPlacement, Result, SortDirection, StateBudget,
    dataframe::invalid,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A validated fixed UTC interval, flattened into a sorting history declaration.
///
/// ```
/// use calc_flow::{EventTime, SortSpec, SortWindow, StateBudget};
/// let window = SortWindow::new("ts", EventTime::from_micros(-10), EventTime::from_micros(10))?;
/// let spec = SortSpec::closed_window(window, StateBudget::new(1000, 8 << 20)?)?;
/// # Ok::<(), calc_flow::CalcFlowError>(())
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SortWindow {
    event_time_column: String,
    start: crate::EventTime,
    end: crate::EventTime,
}
impl SortWindow {
    /// Declares the half-open interval `[start, end)` without closing it.
    /// # Errors
    /// Rejects an empty column or an end at or before start.
    pub fn new(column: &str, start: crate::EventTime, end: crate::EventTime) -> Result<Self> {
        if column.is_empty() {
            return Err(invalid(
                "sort_window.event_time_column",
                "invalid_literal",
                "requires a nonempty timestamp column",
            ));
        }
        if start >= end {
            return Err(invalid(
                "sort_window.end",
                "invalid_literal",
                "end must be greater than start",
            ));
        }
        Ok(Self {
            event_time_column: column.into(),
            start,
            end,
        })
    }
    /// Original or purely renamed input timestamp column.
    pub fn event_time_column(&self) -> &str {
        &self.event_time_column
    }
    /// Inclusive microsecond coordinate.
    pub const fn start(&self) -> crate::EventTime {
        self.start
    }
    /// Exclusive microsecond coordinate.
    pub const fn end(&self) -> crate::EventTime {
        self.end
    }

    pub(crate) fn from_history(
        column: &str,
        start: crate::EventTime,
        end: crate::EventTime,
    ) -> Result<Self> {
        Self::new(column, start, end).map_err(|error| match error {
            crate::CalcFlowError::InvalidArgument { message, .. } => {
                crate::CalcFlowError::InvalidArgument {
                    field: "sort.spec.history".into(),
                    message,
                }
            }
            other => other,
        })
    }
}

/// The complete finite domain accepted by a final ordering node.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SortHistory {
    /// A fixed finite source cut, finalized at normal EOF.
    BoundedSource,
    /// One fixed UTC half-open interval.
    ClosedWindow {
        /// Original or purely renamed input timestamp.
        event_time_column: String,
        /// Inclusive microsecond coordinate.
        start_micros: i64,
        /// Exclusive microsecond coordinate.
        end_micros: i64,
    },
}

/// Explicit ordering of equal primary keys.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SortTies {
    /// Only primary-key order is promised; boundary membership is unspecified.
    Unspecified,
    /// Reproducible input fields, checked unique over the whole accepted history.
    Sequence {
        /// Original nonnullable fields checked unique over H.
        #[schemars(with = "SequenceColumnsSchema")]
        columns: Vec<String>,
    },
}

/// One direct column comparator. NULL rows are retained.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(try_from = "KeyWire", deny_unknown_fields)]
pub struct SortKey {
    #[schemars(length(min = 1))]
    column: String,
    direction: SortDirection,
    #[schemars(with = "SortNullsSchema")]
    nulls: NullPlacement,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct KeyWire {
    #[schemars(length(min = 1))]
    column: String,
    direction: SortDirection,
    #[schemars(with = "SortNullsSchema")]
    nulls: NullPlacement,
}
impl TryFrom<KeyWire> for SortKey {
    type Error = crate::CalcFlowError;
    fn try_from(w: KeyWire) -> Result<Self> {
        Self::new(&w.column, w.direction, w.nulls)
    }
}
impl SortKey {
    /// Creates a comparator.
    /// # Errors
    /// Rejects an empty column or NULL exclusion.
    pub fn new(column: &str, direction: SortDirection, nulls: NullPlacement) -> Result<Self> {
        if column.is_empty() || nulls == NullPlacement::Exclude {
            return Err(invalid(
                "sort.by",
                "invalid_literal",
                "nonempty column and NULL first/last required",
            ));
        }
        Ok(Self {
            column: column.into(),
            direction,
            nulls,
        })
    }
    /// Input column name.
    pub fn column(&self) -> &str {
        &self.column
    }
    /// Comparator direction.
    pub const fn direction(&self) -> SortDirection {
        self.direction
    }
    /// Position of NULL values.
    pub const fn nulls(&self) -> NullPlacement {
        self.nulls
    }
}

/// Explicit complete-history ordering with a mandatory node-owned peak budget.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(try_from = "Wire", deny_unknown_fields)]
#[schemars(extend(
    "if"={"properties":{"history":{"properties":{"kind":{"const":"bounded_source"}}}}},
    "then"={"properties":{"late_policy":{"properties":{"kind":{"const":"error"}}}}}
))]
pub struct SortSpec {
    ingress_policy: IngressPolicy,
    history: SortHistory,
    #[schemars(with = "SortBudgetSchema")]
    state_budget: StateBudget,
    #[schemars(range(min = 0, max = 4_294_967_295_u64))]
    offset: u64,
    ties: SortTies,
    #[schemars(with = "SortLatePolicySchema")]
    late_policy: LatePolicySpec,
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(rename = "SortIngressPolicy")]
enum IngressPolicy {
    CopyVisibleV2,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Wire {
    ingress_policy: IngressPolicy,
    history: SortHistory,
    #[schemars(with = "SortBudgetSchema")]
    state_budget: StateBudget,
    #[schemars(range(min = 0, max = 4_294_967_295_u64))]
    offset: u64,
    ties: SortTies,
    #[schemars(with = "SortLatePolicySchema")]
    late_policy: LatePolicySpec,
}
impl TryFrom<Wire> for SortSpec {
    type Error = crate::CalcFlowError;
    fn try_from(w: Wire) -> Result<Self> {
        let _ = w.ingress_policy;
        let base = match w.history {
            SortHistory::BoundedSource => Self::bounded_source(w.state_budget)?,
            SortHistory::ClosedWindow {
                event_time_column,
                start_micros,
                end_micros,
            } => Self::closed_window(
                SortWindow::from_history(
                    &event_time_column,
                    crate::EventTime::from_micros(start_micros),
                    crate::EventTime::from_micros(end_micros),
                )?,
                w.state_budget,
            )?,
        }
        .with_offset(w.offset)?
        .with_late_policy(w.late_policy)?;
        match w.ties {
            SortTies::Unspecified => Ok(base),
            SortTies::Sequence { columns } => base.with_sequence_by(columns),
        }
    }
}
impl SortSpec {
    /// Selects the fixed cut of a bounded source, finalized at actual EOF.
    /// # Errors
    /// Rejects row budgets larger than `UInt32` indices can address.
    pub fn bounded_source(state_budget: StateBudget) -> Result<Self> {
        if state_budget.max_rows() > u64::from(u32::MAX) {
            return Err(invalid(
                "sort.spec.state_budget.max_rows",
                "invalid_literal",
                "requires 1..=4294967295 rows",
            ));
        }
        Ok(Self {
            ingress_policy: IngressPolicy::CopyVisibleV2,
            history: SortHistory::BoundedSource,
            state_budget,
            offset: 0,
            ties: SortTies::Unspecified,
            late_policy: LatePolicySpec::Error {
                scope: crate::LateErrorScope::Envelope,
            },
        })
    }
    /// Selects one UTC half-open window, finalized at watermark or normal EOF.
    /// # Errors
    /// Rejects row budgets larger than `UInt32` indices can address.
    pub fn closed_window(window: SortWindow, budget: StateBudget) -> Result<Self> {
        Ok(Self {
            history: SortHistory::ClosedWindow {
                event_time_column: window.event_time_column,
                start_micros: window.start.as_micros(),
                end_micros: window.end.as_micros(),
            },
            ..Self::bounded_source(budget)?
        })
    }
    /// Skips this many ordered rows.
    /// # Errors
    /// Rejects offsets outside the inclusive `UInt32` domain.
    pub fn with_offset(self, offset: u64) -> Result<Self> {
        checked_limit(offset, "sort.spec.offset")?;
        Ok(Self { offset, ..self })
    }
    /// Selects explicit reproducible input fields for stable ties.
    /// # Errors
    /// Requires nonempty, unique, nonempty column names.
    pub fn with_sequence_by(self, columns: Vec<String>) -> Result<Self> {
        let unique = columns.iter().collect::<std::collections::BTreeSet<_>>();
        if columns.is_empty()
            || unique.len() != columns.len()
            || columns.iter().any(String::is_empty)
        {
            return Err(invalid(
                "sort.spec.ties",
                "invalid_literal",
                "nonempty unique sequence columns required",
            ));
        }
        Ok(Self {
            ties: SortTies::Sequence { columns },
            ..self
        })
    }
    /// Sets a local Error or window Drop policy.
    /// # Errors
    /// Rejects Drop outside a window and unsupported policy versions.
    pub fn with_late_policy(self, policy: LatePolicySpec) -> Result<Self> {
        let _ = self.domain_spec()?.with_late_policy(policy)?;
        Ok(Self {
            late_policy: policy,
            ..self
        })
    }
    /// Accepted history domain.
    pub const fn history(&self) -> &SortHistory {
        &self.history
    }
    /// Mandatory node-owned peak limits.
    pub const fn state_budget(&self) -> StateBudget {
        self.state_budget
    }
    /// Number of rows to skip.
    pub const fn offset(&self) -> u64 {
        self.offset
    }
    /// Tie comparator declaration.
    pub const fn ties(&self) -> &SortTies {
        &self.ties
    }
    /// Local persistent lateness policy.
    pub const fn late_policy(&self) -> LatePolicySpec {
        self.late_policy
    }
    pub(crate) const fn state_layout(&self) -> u32 {
        if matches!(self.late_policy, LatePolicySpec::Drop { .. }) {
            2
        } else {
            1
        }
    }
    pub(crate) fn domain_spec(&self) -> Result<DistinctSpec> {
        match &self.history {
            SortHistory::BoundedSource => Ok(DistinctSpec::bounded_source(self.state_budget)),
            SortHistory::ClosedWindow {
                event_time_column,
                start_micros,
                end_micros,
            } => DistinctSpec::closed_window(
                event_time_column,
                crate::EventTime::from_micros(*start_micros),
                crate::EventTime::from_micros(*end_micros),
                self.state_budget,
            ),
        }
    }
}
pub(crate) fn checked_limit(value: u64, field: &str) -> Result<()> {
    if value > u64::from(u32::MAX) {
        return Err(invalid(
            field,
            "invalid_literal",
            "requires 0..=4294967295 (UInt32 indices)",
        ));
    }
    Ok(())
}
pub(crate) fn validate_fetch(fetch: Option<u64>, spec: &SortSpec) -> Result<()> {
    if let Some(k) = fetch {
        checked_limit(k, "top_k.k")?;
        checked_limit(
            spec.offset()
                .checked_add(k)
                .ok_or_else(|| invalid("top_k.k", "invalid_literal", "offset+k overflow"))?,
            "top_k.k (offset+k)",
        )?;
    }
    Ok(())
}

struct SortNullsSchema;
impl JsonSchema for SortNullsSchema {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SortNulls".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({"type":"string","enum":["first","last"]})
    }
}
pub(in crate::operator) struct SequenceColumnsSchema;
impl JsonSchema for SequenceColumnsSchema {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SortSequenceColumns".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({"type":"array","minItems":1,"uniqueItems":true,"items":{"type":"string","minLength":1}})
    }
}
pub(in crate::operator) struct SortBudgetSchema;
impl JsonSchema for SortBudgetSchema {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SortBudget".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({"type":"object","additionalProperties":false,"required":["max_rows","max_bytes"],"properties":{"max_rows":{"type":"integer","minimum":1,"maximum":4_294_967_295_u64},"max_bytes":{"type":"integer","minimum":1,"maximum":9_007_199_254_740_991_u64}}})
    }
}
pub(in crate::operator) struct SortLatePolicySchema;
impl JsonSchema for SortLatePolicySchema {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "SortLatePolicy".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({"oneOf":[{"type":"object","additionalProperties":false,"required":["kind","scope"],"properties":{"kind":{"const":"error"},"scope":{"const":"envelope"}}},{"type":"object","additionalProperties":false,"required":["kind","metrics_version"],"properties":{"kind":{"const":"drop"},"metrics_version":{"type":"integer","const":1}}}]})
    }
}
