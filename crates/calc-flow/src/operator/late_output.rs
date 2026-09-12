//! Shared diagnostic output schema for rolling and cross-section operators.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use crate::{BatchKind, CalcFlowError, LatePolicySpec, Port, Result};

pub(crate) fn output_ports(
    policy: LatePolicySpec,
    input: &Schema,
    output: SchemaRef,
) -> Result<Vec<Port>> {
    let mut ports = vec![Port::with_schema_ref(
        "output",
        BatchKind::Table,
        true,
        Some(output),
    )?];
    if matches!(policy, LatePolicySpec::SideOutput { .. }) {
        ports.push(Port::with_schema_ref(
            "late",
            BatchKind::Table,
            true,
            Some(schema(input)),
        )?);
    }
    Ok(ports)
}

fn schema(input: &Schema) -> SchemaRef {
    let diagnostics = [
        ("_cf_late_node", DataType::Utf8),
        ("_cf_late_input_port", DataType::Utf8),
        ("_cf_late_event_time_micros", DataType::Int64),
        ("_cf_late_closing_time_micros", DataType::Int64),
        ("_cf_late_watermark_micros", DataType::Int64),
        ("_cf_late_reason", DataType::Utf8),
        ("_cf_late_source", DataType::Utf8),
        ("_cf_late_sequence", DataType::UInt64),
        ("_cf_late_row_index", DataType::UInt64),
    ];
    let fields = input
        .fields()
        .iter()
        .cloned()
        .chain(
            diagnostics
                .into_iter()
                .map(|(name, data_type)| Arc::new(Field::new(name, data_type, false))),
        )
        .collect::<Vec<_>>();
    Arc::new(Schema::new_with_metadata(fields, input.metadata().clone()))
}

pub(crate) fn validate_policy(policy: LatePolicySpec, kind: &str) -> Result<()> {
    if let LatePolicySpec::SideOutput {
        metrics_version,
        schema_version,
    } = policy
    {
        for (field, version) in [
            ("metrics_version", metrics_version),
            ("schema_version", schema_version),
        ] {
            if version != 1 {
                return Err(CalcFlowError::InvalidArgument {
                    field: format!("{kind}.late_policy.{field}"),
                    message: format!("side_output {field} must equal 1; found {version}"),
                });
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_input(policy: LatePolicySpec, input: &Schema, kind: &str) -> Result<()> {
    if matches!(policy, LatePolicySpec::SideOutput { .. }) {
        if let Some((index, field)) = input
            .fields()
            .iter()
            .enumerate()
            .find(|(_, field)| field.name().starts_with("_cf_late_"))
        {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("{kind}.input_schema[{index}].name"),
                message: format!(
                    "input field {:?} is reserved by late schema version 1",
                    field.name()
                ),
            });
        }
    }
    Ok(())
}

pub(crate) fn ensure_execution_enabled(policy: LatePolicySpec, node_id: &str) -> Result<()> {
    if matches!(policy, LatePolicySpec::SideOutput { .. }) {
        return Err(disabled_error(node_id));
    }
    Ok(())
}

pub(crate) fn disabled_error(node_id: &str) -> CalcFlowError {
    CalcFlowError::Compile {
        message: format!(
            "node {node_id:?}: unsupported_capability: late side output execution is not enabled; dual-output runtime and recovery validation is pending"
        ),
    }
}
