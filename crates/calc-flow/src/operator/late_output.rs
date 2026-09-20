//! Shared diagnostic output schema for rolling and cross-section operators.

#[cfg(test)]
mod tests;

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use serde::{Deserialize, Serialize};

use super::{LateMetricDelta, PreparedLateMetrics, StreamOperatorContext, accumulate_late_metrics};
use crate::{Batch, BatchKind, CalcFlowError, EventTime, LatePolicySpec, Port, Result};

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
    match policy {
        LatePolicySpec::SideOutput {
            metrics_version,
            schema_version,
        } => {
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
        LatePolicySpec::Drop { metrics_version } if metrics_version != 1 => {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("{kind}.late_policy.metrics_version"),
                message: "unsupported late-metrics version".into(),
            });
        }
        _ => {}
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

/// Per-envelope tally of late rows staged by rolling and cross-section
/// classification; converts into one `LateMetricDelta` at prepare time.
#[derive(Default)]
pub(super) struct LateRowTally {
    pub(super) late_rows: u64,
    pub(super) max_lateness_micros: Option<u64>,
}

impl LateRowTally {
    pub(super) fn into_delta(self) -> LateMetricDelta {
        LateMetricDelta {
            late_rows: self.late_rows,
            affected_batches: u64::from(self.late_rows > 0),
            max_lateness_micros: self.max_lateness_micros,
            ..LateMetricDelta::default()
        }
    }
}

pub(super) fn record_late_row(
    tally: &mut LateRowTally,
    watermark: Option<EventTime>,
    event_time: i64,
    node_id: &str,
) -> Result<()> {
    let Some(watermark) = watermark else {
        return Ok(());
    };
    tally.late_rows = tally
        .late_rows
        .checked_add(1)
        .ok_or_else(|| error(node_id, "late row counter overflowed"))?;
    let lateness = u64::try_from(i128::from(watermark.as_micros()) - i128::from(event_time))
        .map_err(|_| error(node_id, "late row distance overflowed"))?;
    tally.max_lateness_micros = Some(
        tally
            .max_lateness_micros
            .map_or(lateness, |maximum| maximum.max(lateness)),
    );
    Ok(())
}

pub(super) fn validate_late_data(
    failed: bool,
    input: &Port,
    batch: &Batch,
    context: &StreamOperatorContext<'_>,
) -> Result<()> {
    ensure_can_continue(failed, context.operator_id())?;
    context.check_cancelled()?;
    input.validate(batch, context.operator_id())
}

pub(super) fn new_plan<'a>(
    batch: &'a Batch,
    next_sequence: u64,
    late_port: &Port,
    watermark: Option<EventTime>,
    context: &'a StreamOperatorContext<'_>,
) -> Result<LateOutputPlan<'a>> {
    LateOutputPlan::new(
        batch,
        context.operator_id(),
        watermark.map_or(0, EventTime::as_micros),
        context.output_budget(),
        next_sequence,
        Arc::clone(late_port.schema().expect("late schema is compiled")),
    )
}

pub(super) fn prepare_late_callback<'a, T>(
    accepted: T,
    tally: LateRowTally,
    late: LateOutputPlan<'a>,
    current: LateMetricDelta,
    context: &StreamOperatorContext<'_>,
) -> Result<(
    T,
    LateMetricDelta,
    PreparedLateMetrics,
    PreparedLateOutput<'a>,
)> {
    let delta = tally.into_delta();
    let next_metrics = accumulate_late_metrics(current, delta)?;
    let metrics = context.prepare_window_metrics(delta)?;
    let output = late.prepare()?;
    Ok((accepted, next_metrics, metrics, output))
}

pub(super) fn reject_batch_mode(policy: LatePolicySpec, name: &str) -> Result<()> {
    if matches!(policy, LatePolicySpec::SideOutput { .. }) {
        return Err(CalcFlowError::Compile {
            message: format!(
                "node {name:?}: unsupported_mode: late side output requires stream mode"
            ),
        });
    }
    Ok(())
}

fn error(node: &str, message: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node.into(),
        message: message.into(),
    }
}
