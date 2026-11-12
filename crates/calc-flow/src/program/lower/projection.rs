use super::{Lowered, join::mapped_identity, retained_ordering};
use crate::{
    NamedExpr, QueryOrderingFacts, TableExpr,
    dataframe::{RelationKind, Scalar},
};
use datafusion::arrow::datatypes::SchemaRef;

impl Lowered {
    pub(super) fn projection_ordering(
        &self,
        table: &TableExpr,
        schema: &SchemaRef,
    ) -> QueryOrderingFacts {
        let RelationKind::Project {
            input,
            columns,
            append,
        } = &table.0.kind
        else {
            unreachable!("dispatched projection")
        };
        let ordering = &self.orderings[&input.0.id];
        if !self.join_paths.contains(&table.0.id) {
            return retained_ordering(ordering, schema);
        }
        QueryOrderingFacts {
            temporal_identity: ordering.temporal_identity.as_ref().and_then(|identity| {
                mapped_identity(identity, |name| {
                    projected_column(input, columns, *append, name)
                })
            }),
            ..ordering.clone()
        }
    }
}

fn projected_column(
    input: &TableExpr,
    columns: &[NamedExpr],
    append: bool,
    name: &str,
) -> Option<String> {
    if append {
        return Some(name.into());
    }
    columns
        .iter()
        .filter(|column| {
            matches!(column.value.0.as_ref(), Scalar::Column { lineage, name: source }
                if lineage == input.lineage() && source == name)
        })
        .min_by_key(|column| column.name != name)
        .map(|column| column.name.clone())
}
