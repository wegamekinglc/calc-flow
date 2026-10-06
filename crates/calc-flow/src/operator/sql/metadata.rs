use std::sync::Arc;

use datafusion::execution::memory_pool::MemoryReservation;
use serde_json::Value;

use super::{StateSegment, incremental, sql_state_error};
use crate::{BatchMetadata, DataFusionRuntime, JsonMap, Result};

pub(super) struct SqlMetadata {
    pub segment: StateSegment,
    _reservation: Arc<MemoryReservation>,
}

pub(super) fn reserve(
    runtime: &DataFusionRuntime,
    metadata: &BatchMetadata,
    name: &str,
) -> Result<MemoryReservation> {
    let attributes = metadata
        .attributes()
        .iter()
        .try_fold(0usize, |bytes, (key, value)| {
            add(bytes, entry_bytes(key, value, 2)?)
        })?;
    let source = metadata
        .source()
        .len()
        .checked_mul(6)
        .ok_or_else(overflow)?;
    let bound = incremental::checked_bytes(4096, [(add(source, attributes)?, 2)], name)?;
    let reservation = runtime.incremental_reservation(name);
    incremental::ensure_reservation(&reservation, bound, name)?;
    Ok(reservation)
}

pub(super) fn encode(
    metadata: &BatchMetadata,
    reservation: MemoryReservation,
    mut check_cancelled: impl FnMut() -> Result<()>,
) -> Result<Arc<SqlMetadata>> {
    check_cancelled()?;
    let bytes =
        serde_json::to_vec(metadata).map_err(|error| sql_state_error(&error.to_string()))?;
    if bytes
        .capacity()
        .checked_add(256)
        .is_none_or(|bytes| bytes > reservation.size())
    {
        return Err(sql_state_error("SQL metadata exceeded its prepaid bound"));
    }
    check_cancelled()?;
    let reservation = Arc::new(reservation);
    Ok(Arc::new(SqlMetadata {
        segment: StateSegment::new(bytes).with_owner(reservation.clone()),
        _reservation: reservation,
    }))
}

pub(super) fn decode(
    runtime: &DataFusionRuntime,
    segment: &StateSegment,
    name: &str,
) -> Result<(BatchMetadata, Arc<SqlMetadata>)> {
    let reservation = runtime.incremental_reservation(name);
    let bound = incremental::checked_bytes(4096, [(segment.bytes().len(), 32)], name)?;
    incremental::ensure_reservation(&reservation, bound, name)?;
    let value = crate::json::parse_json_value(segment.bytes(), "SQL batch metadata")?;
    let Value::Object(mut object) = value else {
        return Err(sql_state_error("SQL batch metadata must be an object"));
    };
    if object.len() != 3 {
        return Err(sql_state_error("SQL batch metadata has unknown fields"));
    }
    let Some(Value::String(source)) = object.remove("source") else {
        return Err(sql_state_error("SQL batch metadata has no source"));
    };
    let sequence = object
        .remove("sequence")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| sql_state_error("SQL batch metadata has no sequence"))?;
    let Some(Value::Object(attributes)) = object.remove("attributes") else {
        return Err(sql_state_error("SQL batch metadata has no attributes"));
    };
    let attributes: JsonMap = attributes.into_iter().collect();
    let metadata = BatchMetadata::new(source, sequence, attributes)?;
    let reservation = Arc::new(reservation);
    Ok((
        metadata,
        Arc::new(SqlMetadata {
            segment: segment.clone().with_owner(reservation.clone()),
            _reservation: reservation,
        }),
    ))
}

fn value_bytes(value: &Value, depth: usize) -> Result<usize> {
    if depth > crate::json::MAX_JSON_DEPTH {
        return Err(sql_state_error(
            "SQL batch metadata exceeds the maximum JSON depth",
        ));
    }
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => Ok(64),
        Value::String(value) => add(64, value.len().checked_mul(6).ok_or_else(overflow)?),
        Value::Array(values) => values.iter().try_fold(64usize, |bytes, value| {
            add(bytes, value_bytes(value, depth + 1)?)
        }),
        Value::Object(values) => values.iter().try_fold(64usize, |bytes, (key, value)| {
            add(bytes, add(128, entry_bytes(key, value, depth + 1)?)?)
        }),
    }
}

fn entry_bytes(key: &str, value: &Value, depth: usize) -> Result<usize> {
    add(
        key.len().checked_mul(6).ok_or_else(overflow)?,
        value_bytes(value, depth)?,
    )
}

fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right).ok_or_else(overflow)
}

fn overflow() -> crate::CalcFlowError {
    sql_state_error("SQL metadata charge overflowed")
}
