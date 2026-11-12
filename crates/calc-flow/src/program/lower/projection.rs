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
    let expected = column_origin(input, name)?;
    columns
        .iter()
        .filter(|column| {
            direct_column(input, column)
                .and_then(|source| column_origin(input, source))
                .is_some_and(|origin| origin == expected)
        })
        .min_by_key(|column| column.name != name)
        .map(|column| column.name.clone())
}

fn direct_column<'a>(input: &TableExpr, column: &'a NamedExpr) -> Option<&'a str> {
    match column.value.0.as_ref() {
        Scalar::Column { lineage, name } if lineage == input.lineage() => Some(name),
        _ => None,
    }
}

fn column_origin<'a>(table: &'a TableExpr, name: &'a str) -> Option<(&'a str, &'a str)> {
    match &table.0.kind {
        RelationKind::Project {
            input,
            columns,
            append,
        } => match columns.iter().find(|column| column.name == name) {
            Some(column) => column_origin(input, direct_column(input, column)?),
            None if *append => column_origin(input, name),
            None => None,
        },
        RelationKind::Filter { input, .. } => column_origin(input, name),
        _ => Some((&table.0.id, name)),
    }
}
