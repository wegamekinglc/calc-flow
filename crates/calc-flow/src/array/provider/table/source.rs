use super::{
    super::ir::{self, DType},
    document,
};
use crate::{BatchKind, Port, Result, array::invalid};
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use std::{collections::BTreeSet, sync::Arc};

pub(super) struct Selection {
    pub name: String,
    pub columns: Vec<usize>,
    pub schema: SchemaRef,
    pub dtype: DType,
    pub static_input: Option<ir::Input>,
}

pub(super) fn select(declarations: &[document::Input], ports: &[Port]) -> Result<Selection> {
    let (name, columns) = table_declaration(declarations)?;
    let static_input = static_input(declarations)?;
    exact_inputs(declarations, ports)?;
    let schema = table_schema(ports, name)?;
    let (selected, dtype) = column_layout(columns, schema)?;
    Ok(Selection {
        name: name.to_owned(),
        columns: selected,
        schema: Arc::clone(schema),
        dtype,
        static_input,
    })
}
fn table_declaration(declarations: &[document::Input]) -> Result<(&str, &[String])> {
    let tables = declarations
        .iter()
        .filter_map(|input| match input {
            document::Input::Table { name, columns } => Some((name.as_str(), columns.as_slice())),
            document::Input::Array { .. } => None,
        })
        .collect::<Vec<_>>();
    let [(name, columns)] = tables.as_slice() else {
        return Err(invalid(
            "options.inputs",
            "requires exactly one table input",
        ));
    };
    Ok((*name, *columns))
}
fn table_schema<'a>(ports: &'a [Port], name: &str) -> Result<&'a SchemaRef> {
    named_port(ports, name, BatchKind::Table)?
        .schema()
        .ok_or_else(|| invalid("inputs.schema", "exact table schema is required"))
}
fn column_layout(columns: &[String], schema: &SchemaRef) -> Result<(Vec<usize>, DType)> {
    validate_columns(columns)?;
    let selected = indices(columns, schema)?;
    let dtype = validate_dtype(&selected, schema)?;
    Ok((selected, dtype))
}
fn static_input(declarations: &[document::Input]) -> Result<Option<ir::Input>> {
    let inputs = declarations
        .iter()
        .filter_map(|input| match input {
            document::Input::Array {
                name,
                binding: document::Binding::Static,
                dtype,
                shape,
            } => Some(ir::Input {
                name: name.clone(),
                dtype: *dtype,
                shape: shape.clone(),
            }),
            document::Input::Table { .. } => None,
        })
        .collect::<Vec<_>>();
    match inputs.as_slice() {
        [] => Ok(None),
        [input] => Ok(Some(input.clone())),
        _ => Err(invalid(
            "options.inputs",
            "at most one static matrix is supported",
        )),
    }
}
fn exact_inputs(declarations: &[document::Input], ports: &[Port]) -> Result<()> {
    let expected = declarations
        .iter()
        .map(document::Input::name)
        .collect::<BTreeSet<_>>();
    let actual = ports.iter().map(Port::name).collect::<BTreeSet<_>>();
    if expected != actual || ports.len() != declarations.len() || actual.len() != ports.len() {
        return Err(invalid(
            "inputs",
            "required input ports must exactly match declarations",
        ));
    }
    for input in declarations {
        let kind = match input {
            document::Input::Table { .. } => BatchKind::Table,
            document::Input::Array { .. } => BatchKind::Array,
        };
        named_port(ports, input.name(), kind)?;
    }
    Ok(())
}
fn named_port<'a>(ports: &'a [Port], name: &str, kind: BatchKind) -> Result<&'a Port> {
    let port = ports
        .iter()
        .find(|port| port.name() == name)
        .ok_or_else(|| invalid("inputs", "required port is missing"))?;
    if port.kind() != kind || !port.required() {
        return Err(invalid(
            "inputs",
            "required port kind differs from declaration",
        ));
    }
    Ok(port)
}
fn validate_columns(columns: &[String]) -> Result<()> {
    if columns.is_empty() || columns.iter().collect::<BTreeSet<_>>().len() != columns.len() {
        return Err(invalid(
            "options.inputs.columns",
            "selected columns must be nonempty and unique",
        ));
    }
    Ok(())
}
fn indices(columns: &[String], schema: &SchemaRef) -> Result<Vec<usize>> {
    columns
        .iter()
        .map(|name| {
            schema
                .index_of(name)
                .map_err(|_| invalid("options.inputs.columns", "selected column is missing"))
        })
        .collect()
}
fn validate_dtype(columns: &[usize], schema: &SchemaRef) -> Result<DType> {
    let first = schema.field(columns[0]).data_type();
    if columns
        .iter()
        .any(|index| schema.field(*index).data_type() != first)
    {
        return Err(invalid(
            "inputs.dtype",
            "selected columns must have one homogeneous native dtype",
        ));
    }
    match first {
        DataType::Boolean => Ok(DType::Bool),
        DataType::Float32 => Ok(DType::Float32),
        DataType::Float64 => Ok(DType::Float64),
        _ => Err(invalid(
            "inputs.dtype",
            "selected columns require bool, float32 or float64",
        )),
    }
}
pub(super) fn port<'a>(ports: &'a [Port], name: &str, kind: BatchKind) -> Result<&'a Port> {
    let [port] = ports else {
        return Err(invalid("ports", "requires exactly one port"));
    };
    if port.name() != name || port.kind() != kind || !port.required() {
        return Err(invalid("ports", "required port must match declaration"));
    }
    Ok(port)
}
