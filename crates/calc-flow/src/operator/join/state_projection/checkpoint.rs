//! Inner layout 3 binds logical declarations to their retained physical payload.

use datafusion::arrow::datatypes::SchemaRef;
use serde::{Deserialize, Serialize};

use super::super::StreamJoinOperator;

mod schema_digest;
pub(in crate::operator::join) use schema_digest::digest;

pub(in crate::operator::join) const CHARGE_MODE: &str = "v1_admission";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::operator::join) struct Descriptor {
    pub(in crate::operator::join) columns: [Vec<usize>; 2],
    pub(in crate::operator::join) logical_schema_digests: [String; 2],
    pub(in crate::operator::join) schema_digests: [String; 2],
}

impl Descriptor {
    pub(super) fn new(
        operator: &StreamJoinOperator,
        columns: &[Vec<usize>; 2],
        schemas: &[SchemaRef; 2],
    ) -> Self {
        Self {
            columns: columns.clone(),
            logical_schema_digests: [
                digest(operator.input_schema(0)),
                digest(operator.input_schema(1)),
            ],
            schema_digests: [digest(&schemas[0]), digest(&schemas[1])],
        }
    }
}
