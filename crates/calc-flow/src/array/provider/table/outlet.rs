use super::super::{infer, ir};
use super::{OutputMode, document, source};
use crate::{BatchKind, Port, Result, array::invalid};
use datafusion::arrow::datatypes::{DataType, Field, SchemaRef};
use std::{collections::BTreeSet, sync::Arc};

pub(super) fn reference(outputs: &[document::Output]) -> Result<ir::Output> {
    match only(outputs)? {
        document::Output::Table { name, node, .. } => Ok(ir::Output {
            name: name.clone(),
            node: node.clone(),
            assertion: None,
        }),
        document::Output::Array {
            name,
            node,
            dtype,
            shape,
        } => Ok(ir::Output {
            name: name.clone(),
            node: node.clone(),
            assertion: Some(ir::Assertion {
                dtype: *dtype,
                shape: shape.clone(),
            }),
        }),
    }
}
fn only(outputs: &[document::Output]) -> Result<&document::Output> {
    let [output] = outputs else {
        return Err(invalid("options.outputs", "requires exactly one output"));
    };
    Ok(output)
}
pub(super) fn validate(
    outputs: &[document::Output],
    ports: &[Port],
    source: &source::Selection,
    descriptor: infer::Descriptor,
) -> Result<OutputMode> {
    match only(outputs)? {
        document::Output::Array { name, .. } => {
            let _ = source::port(ports, name, BatchKind::Array)?;
            Ok(OutputMode::Array)
        }
        document::Output::Table { .. } => table(only(outputs)?, ports, source, descriptor),
    }
}
fn table(
    output: &document::Output,
    ports: &[Port],
    source: &source::Selection,
    descriptor: infer::Descriptor,
) -> Result<OutputMode> {
    let document::Output::Table {
        name,
        attach_to,
        names,
        ..
    } = output
    else {
        unreachable!("table outlet selected");
    };
    if attach_to != &source.name {
        return Err(invalid(
            "options.outputs.attach_to",
            "must reference the table input",
        ));
    }
    validate_names(names, &source.schema, width(descriptor.shape)?)?;
    let target = source::port(ports, name, BatchKind::Table)?;
    attached_schema(target, &source.schema, names, descriptor.dtype)?;
    Ok(OutputMode::Table(names.clone()))
}
fn attached_schema(
    target: &Port,
    source: &SchemaRef,
    names: &[String],
    dtype: ir::DType,
) -> Result<()> {
    let mut expected = source.fields().to_vec();
    expected.extend(
        names
            .iter()
            .map(|name| Arc::new(Field::new(name, arrow_dtype(dtype), true))),
    );
    if target.schema().is_none_or(|schema| {
        schema.fields() != &expected.into() || schema.metadata() != source.metadata()
    }) {
        return Err(invalid(
            "outputs.schema",
            "attached schema differs from declaration",
        ));
    }
    Ok(())
}
fn width(shape: infer::Shape) -> Result<usize> {
    match shape {
        infer::Shape::Vector(_) => Ok(1),
        infer::Shape::Matrix(_, Some(width)) if width > 0 => Ok(width),
        _ => Err(invalid(
            "options.outputs.shape",
            "attached output requires a known positive width",
        )),
    }
}
fn validate_names(names: &[String], schema: &SchemaRef, width: usize) -> Result<()> {
    if names.len() != width {
        return Err(invalid(
            "options.outputs.names",
            "appended names must match the output width",
        ));
    }
    unique_names(names, schema)
}
fn unique_names(names: &[String], schema: &SchemaRef) -> Result<()> {
    let unique = names.iter().collect::<BTreeSet<_>>().len() == names.len();
    let conflicting = names
        .iter()
        .any(|name| name.is_empty() || schema.index_of(name).is_ok());
    if !unique || conflicting {
        return Err(invalid(
            "options.outputs.names",
            "appended names must be unique and collision-free",
        ));
    }
    Ok(())
}

fn arrow_dtype(dtype: ir::DType) -> DataType {
    match dtype {
        ir::DType::Bool => DataType::Boolean,
        ir::DType::Float32 => DataType::Float32,
        ir::DType::Float64 => DataType::Float64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BatchKind, CalcFlowError};
    use datafusion::arrow::datatypes::Schema;
    use std::collections::HashMap;

    #[test]
    fn test_attached_schema_rejects_changed_arrow_metadata_at_admission() {
        let metadata = HashMap::from([("owner".into(), "research".into())]);
        let source = Arc::new(Schema::new_with_metadata(
            vec![Field::new("x", DataType::Float64, false)],
            metadata.clone(),
        ));
        let fields = vec![
            Field::new("x", DataType::Float64, false),
            Field::new("score", DataType::Float64, true),
        ];
        for actual_metadata in [HashMap::new(), metadata] {
            let matches = actual_metadata == *source.metadata();
            let target = Port::with_schema_ref(
                "output",
                BatchKind::Table,
                true,
                Some(Arc::new(Schema::new_with_metadata(
                    fields.clone(),
                    actual_metadata,
                ))),
            )
            .unwrap();
            let result = attached_schema(&target, &source, &["score".into()], ir::DType::Float64);
            if matches {
                assert!(result.is_ok());
            } else {
                assert!(
                    matches!(result,Err(CalcFlowError::InvalidArgument{field,..}) if field=="outputs.schema")
                );
            }
        }
    }
}
