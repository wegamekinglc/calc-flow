//! Shared diagnostic output schema for rolling and cross-section operators.

#[cfg(test)]
mod tests;

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use serde::{Deserialize, Serialize};

use crate::{BatchKind, CalcFlowError, LatePolicySpec, Port, Result};

pub(super) mod identity;
mod plan;
pub(super) use plan::{LateOutputPlan, PreparedLateOutput};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LateOutputSnapshot {
    version: u32,
    schema_version: u32,
    next_sequence: u64,
}

impl LateOutputSnapshot {
    pub(super) fn new(policy: LatePolicySpec, next_sequence: u64) -> Option<Self> {
        match policy {
            LatePolicySpec::SideOutput { schema_version, .. } => Some(Self {
                version: 1,
                schema_version,
                next_sequence,
            }),
            _ => None,
        }
    }

    pub(super) fn restore_sequence(policy: LatePolicySpec, snapshot: Option<&Self>) -> Result<u64> {
        match (policy, snapshot) {
            (LatePolicySpec::SideOutput { schema_version, .. }, Some(snapshot))
                if snapshot.version == 1 && snapshot.schema_version == schema_version =>
            {
                Ok(snapshot.next_sequence)
            }
            (LatePolicySpec::SideOutput { .. }, _) => Err(CalcFlowError::CheckpointMismatch {
                message: "checkpoint is missing a compatible late_output version 1 object".into(),
            }),
            (_, Some(_)) => Err(CalcFlowError::CheckpointMismatch {
                message: "checkpoint unexpectedly contains late_output for a disabled policy"
                    .into(),
            }),
            (_, None) => Ok(0),
        }
    }
}

pub(super) fn deserialize_snapshot<'de, D>(
    deserializer: D,
) -> std::result::Result<Option<LateOutputSnapshot>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    LateOutputSnapshot::deserialize(deserializer).map(Some)
}

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

pub(super) fn ensure_can_continue(failed: bool, node_id: &str) -> Result<()> {
    if failed {
        return Err(CalcFlowError::Operator {
            node_id: node_id.into(),
            message: "late output emission failed or was cancelled; live callback retry is forbidden; recover from a durable cut".into(),
        });
    }
    Ok(())
}
