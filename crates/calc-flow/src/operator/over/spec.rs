use crate::{
    EventTime, LatePolicySpec, NullPlacement, Result, SortDirection, SortKey, SortSpec, SortWindow,
    StateBudget, dataframe::invalid,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// One partition and total order shared by all analytic expressions.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(try_from = "OrderWire", deny_unknown_fields)]
pub struct OverOrder {
    #[serde(rename = "partition_by")]
    #[schemars(with = "PartitionColumnsSchema")]
    partition: Vec<String>,
    #[schemars(length(min = 1))]
    #[serde(rename = "order_by")]
    primary: Vec<SortKey>,
    #[schemars(with = "crate::operator::order::SequenceColumnsSchema")]
    #[serde(rename = "sequence_by")]
    sequence: Vec<String>,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct OrderWire {
    #[serde(rename = "partition_by")]
    #[schemars(with = "PartitionColumnsSchema")]
    partition: Vec<String>,
    #[schemars(length(min = 1))]
    #[serde(rename = "order_by")]
    primary: Vec<SortKey>,
    #[schemars(with = "crate::operator::order::SequenceColumnsSchema")]
    #[serde(rename = "sequence_by")]
    sequence: Vec<String>,
}
impl TryFrom<OrderWire> for OverOrder {
    type Error = crate::CalcFlowError;
    fn try_from(w: OrderWire) -> Result<Self> {
        Self::new(w.partition, w.primary, w.sequence)
    }
}
impl OverOrder {
    /// Declares disjoint partition, primary order and reproducible sequence fields.
    /// # Errors
    /// Requires nonempty order/sequence lists and unique nonempty names.
    pub fn new(
        partition_by: Vec<String>,
        order_by: Vec<SortKey>,
        sequence_by: Vec<String>,
    ) -> Result<Self> {
        if order_by.is_empty() || sequence_by.is_empty() {
            return Err(invalid(
                "over.spec.order.sequence_by",
                "ordering_required",
                "nonempty primary order and reproducible sequence required",
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for (role, names) in [
            (
                "partition_by",
                partition_by.iter().map(String::as_str).collect::<Vec<_>>(),
            ),
            ("order_by", order_by.iter().map(SortKey::column).collect()),
            (
                "sequence_by",
                sequence_by.iter().map(String::as_str).collect(),
            ),
        ] {
            for (index, name) in names.iter().enumerate() {
                let path = format!("over.spec.order.{role}[{index}]");
                crate::dataframe::check_name(name, &path)?;
                if !seen.insert(*name) {
                    return Err(invalid(
                        &path,
                        "duplicate_name",
                        "partition/order/sequence names must be disjoint and unique",
                    ));
                }
            }
        }
        Ok(Self {
            partition: partition_by,
            primary: order_by,
            sequence: sequence_by,
        })
    }
    /// Partition fields; empty means one partition for H.
    pub fn partition_by(&self) -> &[String] {
        &self.partition
    }
    /// Primary comparators within a partition.
    pub fn order_by(&self) -> &[SortKey] {
        &self.primary
    }
    /// Ascending tie fields, checked globally unique over H.
    pub fn sequence_by(&self) -> &[String] {
        &self.sequence
    }
    pub(crate) fn keys(&self) -> Result<Vec<SortKey>> {
        let mut keys = self
            .partition
            .iter()
            .map(|name| SortKey::new(name, SortDirection::Ascending, NullPlacement::Last))
            .collect::<Result<Vec<_>>>()?;
        keys.extend_from_slice(&self.primary);
        Ok(keys)
    }
}

/// A fixed UTC interval; declaring it does not close the history.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(try_from = "WindowWire", deny_unknown_fields)]
pub struct OverWindow {
    #[schemars(length(min = 1))]
    event_time_column: String,
    start_micros: i64,
    end_micros: i64,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct WindowWire {
    #[schemars(length(min = 1))]
    event_time_column: String,
    start_micros: i64,
    end_micros: i64,
}
impl TryFrom<WindowWire> for OverWindow {
    type Error = crate::CalcFlowError;
    fn try_from(w: WindowWire) -> Result<Self> {
        Self::new(
            &w.event_time_column,
            EventTime::from_micros(w.start_micros),
            EventTime::from_micros(w.end_micros),
        )
    }
}
impl OverWindow {
    /// Selects the half-open interval [start,end).
    /// # Errors
    /// Requires a nonempty column and start < end.
    pub fn new(event_time_column: &str, start: EventTime, end: EventTime) -> Result<Self> {
        crate::dataframe::check_name(event_time_column, "over.spec.history.event_time_column")?;
        if start >= end {
            return Err(invalid(
                "over.spec.history.end_micros",
                "invalid_literal",
                "end must be greater than start",
            ));
        }
        Ok(Self {
            event_time_column: event_time_column.into(),
            start_micros: start.as_micros(),
            end_micros: end.as_micros(),
        })
    }
    /// Original or purely renamed input timestamp column.
    pub fn event_time_column(&self) -> &str {
        &self.event_time_column
    }
    /// Inclusive microsecond coordinate.
    pub const fn start(&self) -> EventTime {
        EventTime::from_micros(self.start_micros)
    }
    /// Exclusive microsecond coordinate.
    pub const fn end(&self) -> EventTime {
        EventTime::from_micros(self.end_micros)
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
#[schemars(rename = "OverHistory")]
enum History {
    BoundedSource,
    ClosedWindow {
        #[serde(flatten)]
        window: OverWindow,
    },
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(rename = "OverIngressPolicy")]
enum Ingress {
    CopyVisibleV2,
}

/// A complete H, mandatory owned peak budget and local late policy.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(try_from = "SpecWire", deny_unknown_fields)]
#[schemars(extend(
    "if"={"properties":{"history":{"properties":{"kind":{"const":"bounded_source"}}}}},
    "then"={"properties":{"late_policy":{"properties":{"kind":{"const":"error"}}}}}
))]
pub struct OverSpec {
    ingress_policy: Ingress,
    history: History,
    order: OverOrder,
    #[schemars(with = "crate::operator::order::SortBudgetSchema")]
    state_budget: StateBudget,
    #[schemars(with = "crate::operator::order::SortLatePolicySchema")]
    late_policy: LatePolicySpec,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SpecWire {
    ingress_policy: Ingress,
    history: History,
    order: OverOrder,
    #[schemars(with = "crate::operator::order::SortBudgetSchema")]
    state_budget: StateBudget,
    #[schemars(with = "crate::operator::order::SortLatePolicySchema")]
    late_policy: LatePolicySpec,
}
impl TryFrom<SpecWire> for OverSpec {
    type Error = crate::CalcFlowError;
    fn try_from(w: SpecWire) -> Result<Self> {
        let _ = w.ingress_policy;
        let spec = match w.history {
            History::BoundedSource => Self::bounded_source(w.order, w.state_budget)?,
            History::ClosedWindow { window } => {
                Self::closed_window(window, w.order, w.state_budget)?
            }
        };
        spec.with_late_policy(w.late_policy)
    }
}
impl OverSpec {
    /// Finalize at actual EOF of a fixed bounded source cut.
    /// # Errors
    /// Rejects row budgets outside the `UInt32` index domain.
    pub fn bounded_source(order: OverOrder, state_budget: StateBudget) -> Result<Self> {
        SortSpec::bounded_source(state_budget).map_err(super::schema::remap)?;
        Ok(Self {
            ingress_policy: Ingress::CopyVisibleV2,
            history: History::BoundedSource,
            order,
            state_budget,
            late_policy: LatePolicySpec::Error {
                scope: crate::LateErrorScope::Envelope,
            },
        })
    }
    /// Select one fixed interval, finalized by proved watermark or normal EOF.
    /// # Errors
    /// Rejects row budgets outside the `UInt32` index domain.
    pub fn closed_window(
        window: OverWindow,
        order: OverOrder,
        state_budget: StateBudget,
    ) -> Result<Self> {
        Ok(Self {
            history: History::ClosedWindow { window },
            ..Self::bounded_source(order, state_budget)?
        })
    }
    /// Sets the local Error or window Drop policy.
    /// # Errors
    /// Rejects Drop for bounded-source history and unsupported policy versions.
    pub fn with_late_policy(self, policy: LatePolicySpec) -> Result<Self> {
        self.sort_spec()?
            .with_late_policy(policy)
            .map_err(super::schema::remap)?;
        Ok(Self {
            late_policy: policy,
            ..self
        })
    }
    /// Fixed window, or None for the complete bounded source cut.
    pub fn window(&self) -> Option<&OverWindow> {
        match &self.history {
            History::BoundedSource => None,
            History::ClosedWindow { window } => Some(window),
        }
    }
    /// Shared partition and comparator declaration.
    pub const fn order(&self) -> &OverOrder {
        &self.order
    }
    /// Mandatory node-owned limits; all of H remains retained.
    pub const fn state_budget(&self) -> StateBudget {
        self.state_budget
    }
    /// Local persisted late policy.
    pub const fn late_policy(&self) -> LatePolicySpec {
        self.late_policy
    }
    pub(crate) fn sort_spec(&self) -> Result<SortSpec> {
        let spec = match self.window() {
            None => SortSpec::bounded_source(self.state_budget)?,
            Some(w) => SortSpec::closed_window(
                SortWindow::new(w.event_time_column(), w.start(), w.end())?,
                self.state_budget,
            )?,
        };
        spec.with_sequence_by(self.order.sequence.clone())?
            .with_late_policy(self.late_policy)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum FrameKind {
    Cumulative,
    Preceding { rows: u32 },
}
/// An inclusive ROWS frame ending at the current row.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct RowsFrame(FrameKind);
impl RowsFrame {
    /// All preceding partition rows and the current row.
    pub const fn cumulative() -> Self {
        Self(FrameKind::Cumulative)
    }
    /// At most rows preceding rows, plus the current row (zero is current only).
    /// # Errors
    /// Rejects values above 4294967295.
    pub fn preceding(rows: u64) -> Result<Self> {
        Ok(Self(FrameKind::Preceding {
            rows: u32::try_from(rows).map_err(|_| {
                invalid(
                    "over.expressions.frame",
                    "invalid_literal",
                    "requires 0..=4294967295",
                )
            })?,
        }))
    }
    pub(crate) fn start(&self, row: usize, partition_start: usize) -> usize {
        match self.0 {
            FrameKind::Cumulative => partition_start,
            FrameKind::Preceding { rows } => row.saturating_sub(rows as usize).max(partition_start),
        }
    }
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum AnalyticKind {
    RowNumber {
        #[schemars(length(min = 1))]
        name: String,
    },
    Lag {
        #[schemars(length(min = 1))]
        column: String,
        #[schemars(length(min = 1))]
        name: String,
    },
    Sum {
        #[schemars(length(min = 1))]
        column: String,
        frame: RowsFrame,
        #[schemars(length(min = 1))]
        name: String,
    },
}
/// An input-only analytic expression; aliases are appended in declaration order.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(try_from = "AnalyticKind", into = "AnalyticKind")]
pub struct AnalyticExpr(pub(crate) AnalyticKind);
impl From<AnalyticExpr> for AnalyticKind {
    fn from(e: AnalyticExpr) -> Self {
        e.0
    }
}
impl TryFrom<AnalyticKind> for AnalyticExpr {
    type Error = crate::CalcFlowError;
    fn try_from(e: AnalyticKind) -> Result<Self> {
        match e {
            AnalyticKind::RowNumber { name } => Self::row_number(&name),
            AnalyticKind::Lag { column, name } => Self::lag(&column, &name),
            AnalyticKind::Sum {
                column,
                frame,
                name,
            } => Self::sum(&column, frame, &name),
        }
    }
}
impl AnalyticExpr {
    /// Numbers partition rows from one, producing nonnullable `UInt64`.
    /// # Errors
    /// Rejects an empty name.
    pub fn row_number(name: &str) -> Result<Self> {
        crate::dataframe::check_name(name, "over.expressions.name")?;
        Ok(Self(AnalyticKind::RowNumber { name: name.into() }))
    }
    /// Reads the preceding row with NULL default and RESPECT NULLS.
    /// # Errors
    /// Rejects an empty column or name.
    pub fn lag(column: &str, name: &str) -> Result<Self> {
        Self::validate_operand(column, name)?;
        Ok(Self(AnalyticKind::Lag {
            column: column.into(),
            name: name.into(),
        }))
    }
    /// Sums integers over the explicit frame with checked `Int64`/`UInt64` outputs.
    /// # Errors
    /// Rejects an empty column or name; dtype is checked without reading rows.
    pub fn sum(column: &str, frame: RowsFrame, name: &str) -> Result<Self> {
        Self::validate_operand(column, name)?;
        Ok(Self(AnalyticKind::Sum {
            column: column.into(),
            frame,
            name: name.into(),
        }))
    }
    fn validate_operand(column: &str, name: &str) -> Result<()> {
        crate::dataframe::check_name(column, "over.expressions.column")?;
        crate::dataframe::check_name(name, "over.expressions.name")
    }
    pub(crate) fn name(&self) -> &str {
        match &self.0 {
            AnalyticKind::RowNumber { name }
            | AnalyticKind::Lag { name, .. }
            | AnalyticKind::Sum { name, .. } => name,
        }
    }
}

struct PartitionColumnsSchema;
impl JsonSchema for PartitionColumnsSchema {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "OverPartitionColumns".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({"type":"array","uniqueItems":true,"items":{"type":"string","minLength":1}})
    }
}
