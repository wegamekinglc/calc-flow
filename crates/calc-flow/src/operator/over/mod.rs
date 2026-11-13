//! Explicit bounded analytic configuration and native partition/ROWS evaluation.
mod spec;
pub use spec::{AnalyticExpr, OverOrder, OverSpec, OverWindow, RowsFrame};
mod budget;
mod evaluate;
pub(crate) mod schema;

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Declaration {
    pub spec: OverSpec,
    pub expressions: Vec<AnalyticExpr>,
}
pub(crate) use budget::declaration_bound;
pub(in crate::operator) use budget::workspace_bound;
pub(crate) use evaluate::evaluate;
