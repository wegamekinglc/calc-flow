use std::sync::Arc;

use datafusion::execution::memory_pool::MemoryReservation;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::super::{StateSegment, incremental, sql_state_error};
use crate::{DataFusionConfig, DataFusionRuntime, JsonMap, Result};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CompactIdentity {
    pub query_sha256: String,
    pub input_alias: String,
    pub runtime_config: DataFusionConfig,
    pub logical_schema_sha256: String,
    pub physical_schema_sha256: String,
    pub retained_ordinals: Vec<usize>,
    pub state_schema_sha256: String,
    pub output_schema_sha256: String,
    pub native_descriptor: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::operator::sql) struct QuotaLedger {
    pub rows: u64,
    pub bytes: u64,
    pub seen_input: bool,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SegmentDigests {
    pub logical_schema: String,
    pub group_state: String,
    pub batch_metadata: String,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CompactControl {
    pub state_layout: u32,
    pub state_accounting: u32,
    pub native_semantics: u32,
    pub datafusion_version: String,
    pub state_policy: incremental::grouped_float::Policy,
    pub coalescer: incremental::global_record::coalescer::Inventory,
    pub identity: CompactIdentity,
    pub ledger: QuotaLedger,
    pub groups: u64,
    pub group_log: super::log::LogDescriptor,
    pub segments: SegmentDigests,
}

pub(super) struct PaidControl {
    pub segment: StateSegment,
    _reservation: Arc<MemoryReservation>,
}

pub(super) struct DecodedControl {
    pub value: CompactControl,
    _reservation: MemoryReservation,
}

impl CompactControl {
    pub(super) fn validate_identity(&self, trusted: &CompactIdentity) -> Result<()> {
        if self.state_layout != 3
            || self.state_accounting != 3
            || self.native_semantics != 1
            || self.datafusion_version != "54.0.0"
            || self.identity != *trusted
            || !self.ledger.seen_input
        {
            return Err(sql_state_error("SQL compact control identity is invalid"));
        }
        Ok(())
    }

    pub(super) fn validate_segments(
        &self,
        logical_schema: &StateSegment,
        group_state: &StateSegment,
        batch_metadata: &StateSegment,
    ) -> Result<()> {
        if self.segments.logical_schema != logical_schema.sha256()
            || self.segments.group_state != group_state.sha256()
            || self.segments.batch_metadata != batch_metadata.sha256()
            || self.identity.logical_schema_sha256 != logical_schema.sha256()
        {
            return Err(sql_state_error("SQL compact segment census is invalid"));
        }
        Ok(())
    }

    pub(super) fn inline_metadata(&self, control: &StateSegment) -> JsonMap {
        JsonMap::from([
            ("state_layout".into(), json!(self.state_layout)),
            ("state_accounting".into(), json!(self.state_accounting)),
            ("query_sha256".into(), json!(self.identity.query_sha256)),
            ("rows".into(), json!(self.ledger.rows)),
            ("bytes".into(), json!(self.ledger.bytes)),
            ("control_sha256".into(), json!(control.sha256())),
        ])
    }

    pub(super) fn validate_inline(&self, inline: &JsonMap, control: &StateSegment) -> Result<()> {
        if *inline != self.inline_metadata(control) {
            return Err(sql_state_error(
                "SQL compact inline control mirrors are invalid",
            ));
        }
        Ok(())
    }
}

pub(super) fn encode(
    runtime: &DataFusionRuntime,
    control: &CompactControl,
    name: &str,
    check_cancelled: &dyn Fn() -> Result<()>,
) -> Result<PaidControl> {
    check_cancelled()?;
    let reservation = runtime.incremental_reservation(name);
    let bound = encode_bound(control, name)?;
    incremental::ensure_reservation(&reservation, bound, name)?;
    let bytes = serde_json::to_vec(control).map_err(|error| sql_state_error(&error.to_string()))?;
    if bytes
        .capacity()
        .checked_add(256)
        .is_none_or(|size| size > bound)
    {
        return Err(sql_state_error(
            "SQL compact control exceeded its prepaid bound",
        ));
    }
    check_cancelled()?;
    let reservation = Arc::new(reservation);
    Ok(PaidControl {
        segment: StateSegment::new(bytes).with_owner(reservation.clone()),
        _reservation: reservation,
    })
}

pub(super) fn decode(
    runtime: &DataFusionRuntime,
    segment: &StateSegment,
    name: &str,
    check_cancelled: &dyn Fn() -> Result<()>,
) -> Result<DecodedControl> {
    check_cancelled()?;
    let reservation = runtime.incremental_reservation(name);
    incremental::ensure_reservation(
        &reservation,
        incremental::checked_bytes(8192, [(segment.bytes().len(), 64)], name)?,
        name,
    )?;
    let value = crate::json::parse_json_value(segment.bytes(), "SQL compact control")?;
    let value =
        serde_json::from_value(value).map_err(|error| sql_state_error(&error.to_string()))?;
    check_cancelled()?;
    Ok(DecodedControl {
        value,
        _reservation: reservation,
    })
}

fn encode_bound(control: &CompactControl, name: &str) -> Result<usize> {
    let identity = &control.identity;
    let strings = [
        control.state_policy.label(),
        control.datafusion_version.as_str(),
        identity.query_sha256.as_str(),
        identity.input_alias.as_str(),
        identity.logical_schema_sha256.as_str(),
        identity.physical_schema_sha256.as_str(),
        identity.state_schema_sha256.as_str(),
        identity.output_schema_sha256.as_str(),
        control.segments.logical_schema.as_str(),
        control.segments.group_state.as_str(),
        control.segments.batch_metadata.as_str(),
    ];
    let base = strings.iter().try_fold(8192, |bound, value| {
        incremental::checked_bytes(bound, [(value.len(), 12)], name)
    })?;
    let base = incremental::checked_bytes(base, [(identity.retained_ordinals.len(), 64)], name)?;
    let base = incremental::checked_bytes(base, [(control.group_log.frames.len(), 4096)], name)?;
    incremental::checked_bytes(
        base,
        [(value_bound(&identity.native_descriptor, 2, name)?, 2)],
        name,
    )
}

fn value_bound(value: &Value, depth: usize, name: &str) -> Result<usize> {
    if depth > crate::json::MAX_JSON_DEPTH {
        return Err(sql_state_error(
            "SQL compact native descriptor exceeds JSON depth",
        ));
    }
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => Ok(64),
        Value::String(value) => incremental::checked_bytes(64, [(value.len(), 6)], name),
        Value::Array(values) => values.iter().try_fold(64, |bound, value| {
            incremental::checked_bytes(bound, [(value_bound(value, depth + 1, name)?, 1)], name)
        }),
        Value::Object(values) => values.iter().try_fold(64, |bound, (key, value)| {
            incremental::checked_bytes(
                bound,
                [
                    (1, 128),
                    (key.len(), 6),
                    (value_bound(value, depth + 1, name)?, 1),
                ],
                name,
            )
        }),
    }
}

#[cfg(test)]
#[path = "control_tests.rs"]
mod tests;
