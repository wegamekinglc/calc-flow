use super::{
    Lowered, NodeSpec, OperatorSpec, Prepared, QueryOrderingFacts, RelationKind, Result, SchemaRef,
    TableExpr, children, node_id, port_spec,
};
use crate::{
    OperatorMetadata, WindowAggregateOperator, WindowSpec,
    dataframe::{Scalar, invalid},
};
use datafusion::arrow::datatypes::{DataType, TimeUnit};

impl Lowered {
    pub(super) fn prepare_window(
        &self,
        table: &TableExpr,
        input: &TableExpr,
        spec: &WindowSpec,
        path: &str,
    ) -> Result<Prepared> {
        let role = path.to_owned();
        input_path(input, &format!("{role}.value"))?;
        if !time_source(input, &spec.event_time_column)? {
            return Err(invalid(
                &format!("{role}.event_time"),
                "capability_mismatch",
                "event window time must be an unchanged input column or pure rename",
            ));
        }
        let schema = &self.schemas[&input.0.id];
        validate_fields(schema, spec, &role)?;
        let id = node_id(table);
        let operator = WindowAggregateOperator::new(&id, schema.clone(), spec.clone())?;
        let output = operator.output_ports()[0]
            .schema()
            .expect("exact window schema")
            .clone();
        Ok(Prepared {
            node: NodeSpec {
                id,
                operator: OperatorSpec::Window { spec: spec.clone() },
                input_ports: vec![port_spec("input", schema)],
                output_ports: vec![port_spec("output", &output)],
                position: None,
            },
            schema: output,
            kind: "window",
            query: None,
            ordering: QueryOrderingFacts::default(),
        })
    }
}
pub(super) fn contains_window(table: &TableExpr) -> bool {
    matches!(table.0.kind, RelationKind::Window { .. })
        || children(table, "")
            .iter()
            .any(|(_, child, _)| contains_window(child))
}
fn input_path(table: &TableExpr, path: &str) -> Result<()> {
    match &table.0.kind {
        RelationKind::Input { .. } => Ok(()),
        RelationKind::Project { input, .. } | RelationKind::Filter { input, .. } => {
            input_path(input, path)
        }
        RelationKind::Sql { query, tables } if tables.len() == 1 => {
            let (alias, input) = tables.first_key_value().expect("one input");
            if crate::sql::sql_passthrough_columns(query, alias)?.is_none() {
                return Err(invalid(
                    path,
                    "capability_mismatch",
                    "window SQL must be deterministic single-input row-local SQL",
                ));
            }
            input_path(input, path)
        }
        _ => Err(invalid(
            path,
            "capability_mismatch",
            "window paths require one window and row-local transformations",
        )),
    }
}
fn time_source(table: &TableExpr, name: &str) -> Result<bool> {
    match &table.0.kind {
        RelationKind::Input { .. } => Ok(true),
        RelationKind::Filter { input, .. } => time_source(input, name),
        RelationKind::Project {
            input,
            columns,
            append,
        } => {
            if let Some(column) = columns.iter().find(|c| c.name == name) {
                if let Scalar::Column { name, .. } = column.value.0.as_ref() {
                    time_source(input, name)
                } else {
                    Ok(false)
                }
            } else if *append {
                time_source(input, name)
            } else {
                Ok(false)
            }
        }
        RelationKind::Sql { query, tables } if tables.len() == 1 => {
            let (alias, input) = tables.first_key_value().expect("one input");
            let Some(columns) = crate::sql::sql_passthrough_columns(query, alias)? else {
                return Ok(false);
            };
            let source = columns
                .get(name)
                .map(String::as_str)
                .or_else(|| columns.contains_key("*").then_some(name));
            match source {
                Some(name) => time_source(input, name),
                None => Ok(false),
            }
        }
        _ => Ok(false),
    }
}
fn field<'a>(schema: &'a SchemaRef, name: &str, path: &str) -> Result<&'a DataType> {
    schema
        .field_with_name(name)
        .map(datafusion::arrow::datatypes::Field::data_type)
        .map_err(|_| {
            invalid(
                path,
                "unresolved_type",
                format!("unknown window input field {name:?}"),
            )
        })
}
fn ordered(dtype: &DataType) -> bool {
    match dtype {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Date32
        | DataType::Date64 => true,
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            tz.as_deref().is_none_or(|tz| tz == "UTC")
        }
        _ => false,
    }
}
fn validate_fields(schema: &SchemaRef, spec: &WindowSpec, path: &str) -> Result<()> {
    let role = format!("{path}.event_time");
    let dtype = field(schema, &spec.event_time_column, &role)?;
    let time = matches!(dtype, DataType::Timestamp(TimeUnit::Millisecond, None))
        || matches!(dtype, DataType::Timestamp(TimeUnit::Microsecond, tz) if tz.as_deref().is_none_or(|tz| tz == "UTC"));
    if !time {
        return Err(invalid(
            &role,
            "unsupported_type",
            format!(
                "window event time requires timestamp[ms] naive or timestamp[us] naive/UTC, found {dtype}"
            ),
        ));
    }
    for (index, name) in spec.group_by.iter().enumerate() {
        let role = format!("{path}.group_by[{index}]");
        let dtype = field(schema, name, &role)?;
        if !ordered(dtype) {
            return Err(invalid(
                &role,
                "unsupported_type",
                format!("unsupported window group type {dtype}"),
            ));
        }
    }
    for (index, aggregate) in spec.aggregates.iter().enumerate() {
        let role = format!("{path}.aggregates[{index}].column");
        let dtype = field(schema, &aggregate.column, &role)?;
        let supported = match aggregate.function {
            crate::AggregateFunction::Count => true,
            crate::AggregateFunction::Min | crate::AggregateFunction::Max => ordered(dtype),
            _ => dtype.is_integer() || dtype.is_floating(),
        };
        if !supported {
            return Err(invalid(
                &role,
                "unsupported_type",
                format!("unsupported window aggregate input {dtype}"),
            ));
        }
    }
    Ok(())
}
