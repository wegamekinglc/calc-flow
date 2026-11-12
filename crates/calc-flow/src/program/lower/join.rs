use super::{Lowered, Prepared, children, node_id, port_spec};
use crate::{
    JoinOutputOrdering, OperatorMetadata, QueryTemporalIdentity, StreamJoinOperator,
    StreamJoinSpec, dataframe::invalid,
};
use crate::{
    NodeSpec, OperatorSpec, QueryOrderingFacts, Result, TableExpr, dataframe::RelationKind,
};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::datatypes::{DataType, TimeUnit};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn contains_join(table: &TableExpr) -> bool {
    matches!(
        table.0.kind,
        RelationKind::InnerJoin { .. } | RelationKind::AsofJoin { .. }
    ) || children(table, "").iter().any(|(_, c, _)| contains_join(c))
}
pub(super) fn sql_paths(order: &[(TableExpr, String)]) -> BTreeSet<String> {
    let mut paths = BTreeSet::new();
    for (table, _) in order {
        if contains_join(table) {
            paths.insert(table.0.id.clone());
        }
        if let RelationKind::InnerJoin { inputs, .. } | RelationKind::AsofJoin { inputs, .. } =
            &table.0.kind
        {
            for input in inputs {
                ancestors(input, &mut paths);
            }
        }
    }
    paths
}
fn ancestors(table: &TableExpr, paths: &mut BTreeSet<String>) {
    paths.insert(table.0.id.clone());
    for (_, child, _) in children(table, "") {
        ancestors(&child, paths);
    }
}
impl Lowered {
    pub(super) fn prepare_join(
        &self,
        table: &TableExpr,
        inputs: &[TableExpr; 2],
        spec: &StreamJoinSpec,
        ordering: Option<&JoinOutputOrdering>,
        path: &str,
    ) -> Result<Prepared> {
        let schemas = [
            &self.schemas[&inputs[0].0.id],
            &self.schemas[&inputs[1].0.id],
        ];
        for (index, (side, time)) in [
            ("left", spec.left_event_time()),
            ("right", spec.right_event_time()),
        ]
        .into_iter()
        .enumerate()
        {
            let keys = if index == 0 {
                spec.left_keys()
            } else {
                spec.right_keys()
            };
            self.validate_join_side(&inputs[index], schemas[index], side, time, keys, path)?;
        }
        let operator = StreamJoinOperator::new(
            &node_id(table),
            schemas[0].clone(),
            schemas[1].clone(),
            spec.clone(),
        )?;
        let output = operator.output_ports()[0]
            .schema()
            .expect("exact join schema")
            .clone();
        let identity = ordering
            .map(|order| self.join_ordering(inputs, spec, order, &output, path))
            .transpose()?;
        Ok(Prepared {
            node: NodeSpec {
                id: node_id(table),
                operator: OperatorSpec::StreamJoin { spec: spec.clone() },
                input_ports: vec![
                    port_spec("left", schemas[0]),
                    port_spec("right", schemas[1]),
                ],
                output_ports: vec![port_spec("output", &output)],
                position: None,
            },
            schema: output,
            kind: "stream_join",
            query: None,
            ordering: QueryOrderingFacts {
                temporal_identity: identity,
                ..Default::default()
            },
        })
    }
    pub(super) fn validate_join_side(
        &self,
        input: &TableExpr,
        schema: &SchemaRef,
        side: &str,
        time: &str,
        keys: &[String],
        path: &str,
    ) -> Result<()> {
        validate_input_time(input, schema, time, side, path)?;
        let side_path = format!("{path}.{side}");
        let identity = self.orderings[&input.0.id]
            .temporal_identity
            .as_ref()
            .ok_or_else(|| {
                invalid(
                    &side_path,
                    "ordering_required",
                    "join input requires entity, event time and complete sequence identity",
                )
            })?;
        validate_identity(schema, identity, &side_path)?;
        validate_public_keys(schema, keys, &format!("{path}.{side}_keys"))?;
        validate_nested_time(input, identity, time, &format!("{path}.{side}_event_time"))
    }

    fn join_ordering(
        &self,
        inputs: &[TableExpr; 2],
        spec: &StreamJoinSpec,
        order: &JoinOutputOrdering,
        output: &SchemaRef,
        path: &str,
    ) -> Result<QueryTemporalIdentity> {
        let entities: Vec<_> = spec
            .left_keys()
            .iter()
            .map(|k| format!("{}__{k}", spec.left_prefix()))
            .collect();
        let sequences: Vec<_> = inputs
            .iter()
            .zip([spec.left_prefix(), spec.right_prefix()])
            .flat_map(|(input, prefix)| {
                self.orderings[&input.0.id]
                    .temporal_identity
                    .as_ref()
                    .into_iter()
                    .flat_map(move |id| {
                        id.sequence_by.iter().map(move |k| format!("{prefix}__{k}"))
                    })
            })
            .collect();
        let expected_times = [
            format!("{}__{}", spec.left_prefix(), spec.left_event_time()),
            format!("{}__{}", spec.right_prefix(), spec.right_event_time()),
        ];
        for (field, valid) in [
            ("entity_by", order.entity_by == entities),
            ("event_time", expected_times.contains(&order.event_time)),
            (
                "sequence_by",
                !sequences.is_empty() && order.sequence_by == sequences,
            ),
        ] {
            if !valid {
                return Err(invalid(
                    &format!("{path}.output_ordering.{field}"),
                    "ordering_required",
                    "expected prefixed left keys, one join timestamp and complete left then right sequence",
                ));
            }
        }
        validate_time(
            output,
            &order.event_time,
            &format!("{path}.output_ordering.event_time"),
        )?;
        for name in order.entity_by.iter().chain(&order.sequence_by) {
            let field = output
                .field_with_name(name)
                .map_err(|e| invalid(path, "ordering_required", e))?;
            if field.is_nullable() {
                return Err(invalid(
                    &format!("{path}.output_ordering"),
                    "ordering_required",
                    format!("identity field {name:?} must be non-null"),
                ));
            }
        }
        Ok(QueryTemporalIdentity {
            entity_by: order.entity_by.clone(),
            event_time: order.event_time.clone(),
            sequence_by: order.sequence_by.clone(),
        })
    }
    pub(super) fn join_sql_ordering(
        &self,
        table: &TableExpr,
        query: &str,
        tables: &BTreeMap<String, TableExpr>,
    ) -> Result<QueryOrderingFacts> {
        if !self.join_paths.contains(&table.0.id) || tables.len() != 1 {
            return Ok(QueryOrderingFacts::default());
        }
        let (alias, input) = tables.first_key_value().expect("one input");
        self.join_passthrough_ordering(query, alias, input)
    }

    fn join_passthrough_ordering(
        &self,
        query: &str,
        alias: &str,
        input: &TableExpr,
    ) -> Result<QueryOrderingFacts> {
        let Some(columns) = crate::sql::sql_passthrough_columns(query, alias)? else {
            return Ok(QueryOrderingFacts::default());
        };
        let identity = self.orderings[&input.0.id]
            .temporal_identity
            .as_ref()
            .and_then(|identity| mapped_identity(identity, &columns));
        Ok(QueryOrderingFacts {
            temporal_identity: identity,
            ..QueryOrderingFacts::default()
        })
    }
}
fn validate_time(schema: &SchemaRef, name: &str, path: &str) -> Result<()> {
    let field = schema
        .field_with_name(name)
        .map_err(|e| invalid(path, "unresolved_type", e))?;
    if field.is_nullable()
        || field.data_type() != &DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
    {
        return Err(invalid(
            path,
            "ordering_required",
            format!("{name:?} requires non-null timestamp[us, UTC], found {field:?}"),
        ));
    }
    Ok(())
}
fn input_path(table: &TableExpr, path: &str) -> Result<()> {
    match &table.0.kind {
        RelationKind::Input { .. }
        | RelationKind::InnerJoin { .. }
        | RelationKind::AsofJoin { .. } => Ok(()),
        RelationKind::Filter { input, .. } | RelationKind::Project { input, .. } => {
            input_path(input, path)
        }
        RelationKind::Sql { query, tables } if tables.len() == 1 => {
            let (alias, input) = tables.first_key_value().expect("one input");
            if crate::sql::sql_passthrough_columns(query, alias)?.is_some() {
                input_path(input, path)
            } else {
                Err(invalid(
                    path,
                    "capability_mismatch",
                    "join requires deterministic row-local SQL",
                ))
            }
        }
        _ => Err(invalid(
            path,
            "capability_mismatch",
            "join paths require sources, bounded inner joins and row-local transforms",
        )),
    }
}
fn time_source(table: &TableExpr, name: &str) -> Result<bool> {
    match &table.0.kind {
        RelationKind::Input { schema, .. } => Ok(schema.field_with_name(name).is_ok()),
        RelationKind::AsofJoin { spec, .. } => {
            Ok(name == format!("{}__{}", spec.left().prefix(), spec.left().event_time()))
        }
        RelationKind::InnerJoin { ordering, .. } => {
            Ok(ordering.as_ref().is_some_and(|o| o.event_time == name))
        }
        RelationKind::Filter { input, .. } => time_source(input, name),
        RelationKind::Project {
            input,
            columns,
            append,
        } => projected_time_source(input, columns, *append, name),
        RelationKind::Sql { query, tables } if tables.len() == 1 => {
            sql_time_source(query, tables, name)
        }
        _ => Ok(false),
    }
}

fn validate_identity(
    schema: &SchemaRef,
    identity: &QueryTemporalIdentity,
    path: &str,
) -> Result<()> {
    if identity.entity_by.is_empty() || identity.sequence_by.is_empty() {
        return Err(invalid(
            path,
            "ordering_required",
            "join inputs require nonempty entity and sequence keys",
        ));
    }
    validate_time(schema, &identity.event_time, &format!("{path}.event_time"))?;
    for name in &identity.sequence_by {
        let field = schema
            .field_with_name(name)
            .map_err(|e| invalid(path, "ordering_required", e))?;
        if field.is_nullable() || matches!(field.data_type(), DataType::Float32 | DataType::Float64)
        {
            return Err(invalid(
                &format!("{path}.sequence_by"),
                "ordering_required",
                "sequence fields require non-null portable total order; floating keys are forbidden",
            ));
        }
    }
    Ok(())
}

fn validate_input_time(
    input: &TableExpr,
    schema: &SchemaRef,
    time: &str,
    side: &str,
    path: &str,
) -> Result<()> {
    input_path(input, &format!("{path}.{side}"))?;
    let time_path = format!("{path}.{side}_event_time");
    validate_time(schema, time, &time_path)?;
    if !time_source(input, time)? {
        return Err(invalid(
            &time_path,
            "ordering_required",
            "join time must be an unchanged input column or pure rename",
        ));
    }
    Ok(())
}
fn validate_nested_time(
    input: &TableExpr,
    identity: &QueryTemporalIdentity,
    time: &str,
    path: &str,
) -> Result<()> {
    if contains_join(input) && identity.event_time != time {
        return Err(invalid(
            path,
            "ordering_required",
            "nested join event time must match the declared post-join time",
        ));
    }
    Ok(())
}
fn validate_public_keys(schema: &SchemaRef, keys: &[String], path: &str) -> Result<()> {
    for (index, key) in keys.iter().enumerate() {
        let field_path = format!("{path}[{index}]");
        let field = schema
            .field_with_name(key)
            .map_err(|e| invalid(&field_path, "unresolved_type", e))?;
        if matches!(field.data_type(), DataType::Timestamp(unit, _) if *unit != TimeUnit::Microsecond)
        {
            return Err(invalid(
                &field_path,
                "unsupported_type",
                "public timestamp equality keys require microseconds",
            ));
        }
    }
    Ok(())
}
fn mapped_column(columns: &BTreeMap<String, Option<String>>, name: &str) -> Option<String> {
    columns
        .iter()
        .find_map(|(out, source)| (source.as_deref() == Some(name)).then(|| out.clone()))
        .or_else(|| columns.contains_key("*").then(|| name.to_owned()))
}
fn mapped_identity(
    identity: &QueryTemporalIdentity,
    columns: &BTreeMap<String, Option<String>>,
) -> Option<QueryTemporalIdentity> {
    Some(QueryTemporalIdentity {
        entity_by: identity
            .entity_by
            .iter()
            .map(|n| mapped_column(columns, n))
            .collect::<Option<_>>()?,
        event_time: mapped_column(columns, &identity.event_time)?,
        sequence_by: identity
            .sequence_by
            .iter()
            .map(|n| mapped_column(columns, n))
            .collect::<Option<_>>()?,
    })
}
fn projected_time_source(
    input: &TableExpr,
    columns: &[crate::NamedExpr],
    append: bool,
    name: &str,
) -> Result<bool> {
    match columns.iter().find(|c| c.name == name) {
        Some(c) => match c.value.0.as_ref() {
            crate::dataframe::Scalar::Column { name, .. } => time_source(input, name),
            _ => Ok(false),
        },
        None => Ok(append && time_source(input, name)?),
    }
}
fn sql_time_source(query: &str, tables: &BTreeMap<String, TableExpr>, name: &str) -> Result<bool> {
    let (alias, input) = tables.first_key_value().expect("one input");
    let Some(columns) = crate::sql::sql_passthrough_columns(query, alias)? else {
        return Ok(false);
    };
    match columns.get(name) {
        Some(Some(source)) => time_source(input, source),
        None if columns.contains_key("*") => time_source(input, name),
        _ => Ok(false),
    }
}
