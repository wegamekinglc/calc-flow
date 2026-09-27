//! Deterministic logical charge for retained rolling stream state.

use std::mem::size_of;

use datafusion::arrow::datatypes::DataType;

use super::{
    BTreeMap, BufferedRow, CompiledRollingSpec, EntityRollingState, HistoryUpdates, KeyValue,
    RecordBatch, Result, RollingHistories, RowIdentity, ScalarValue, WindowState, operator_error,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct StateCharge {
    pub(super) rows: u64,
    pub(super) bytes: u64,
}

impl StateCharge {
    pub(super) fn checked_add(self, other: Self, node: &str) -> Result<Self> {
        Ok(Self {
            rows: self
                .rows
                .checked_add(other.rows)
                .ok_or_else(|| operator_error(node, "rolling state row charge overflowed"))?,
            bytes: self
                .bytes
                .checked_add(other.bytes)
                .ok_or_else(|| operator_error(node, "rolling state byte charge overflowed"))?,
        })
    }

    pub(super) fn checked_sub(self, other: Self) -> Result<Self> {
        Ok(Self {
            rows: self
                .rows
                .checked_sub(other.rows)
                .ok_or_else(|| super::internal_error("rolling state row charge underflowed"))?,
            bytes: self
                .bytes
                .checked_sub(other.bytes)
                .ok_or_else(|| super::internal_error("rolling state byte charge underflowed"))?,
        })
    }
}

fn charged_usize(value: usize, node: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| operator_error(node, "rolling state byte charge overflowed"))
}

pub(super) fn key_bytes<'a>(values: impl Iterator<Item = &'a KeyValue>, node: &str) -> Result<u64> {
    values
        .filter_map(|value| match value {
            KeyValue::String(text) => Some(text.len()),
            _ => None,
        })
        .try_fold(0_u64, |total, size| {
            total
                .checked_add(charged_usize(size, node)?)
                .ok_or_else(|| operator_error(node, "rolling state byte charge overflowed"))
        })
}

fn scalar_bytes(values: &[ScalarValue], node: &str) -> Result<u64> {
    values.iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(charged_usize(value.size(), node)?)
            .ok_or_else(|| operator_error(node, "rolling state byte charge overflowed"))
    })
}

pub(super) fn scalar_history_row_charge(values: &[ScalarValue], node: &str) -> Result<StateCharge> {
    Ok(StateCharge {
        rows: 1,
        bytes: scalar_bytes(values, node)?,
    })
}

pub(super) fn history_entity_base_charge(
    key: &[Option<KeyValue>],
    windows: usize,
    node: &str,
) -> Result<StateCharge> {
    let keys = key_bytes(key.iter().filter_map(Option::as_ref), node)?;
    let windows = charged_usize(size_of::<WindowState>(), node)?
        .checked_mul(charged_usize(windows, node)?)
        .ok_or_else(|| operator_error(node, "rolling state byte charge overflowed"))?;
    Ok(StateCharge {
        rows: 1,
        bytes: 256_u64
            .checked_add(keys)
            .and_then(|total| total.checked_add(windows))
            .ok_or_else(|| operator_error(node, "rolling state byte charge overflowed"))?,
    })
}

pub(super) fn buffered_row_charge(row: &BufferedRow, node: &str) -> Result<StateCharge> {
    let keys = key_bytes(
        row.identity
            .entity
            .iter()
            .filter_map(Option::as_ref)
            .chain(row.identity.sequence.iter()),
        node,
    )?;
    let values = scalar_bytes(&row.values, node)?;
    let bytes = 256_u64
        .checked_add(keys)
        .and_then(|total| total.checked_add(values))
        .ok_or_else(|| operator_error(node, "rolling state byte charge overflowed"))?;
    Ok(StateCharge { rows: 1, bytes })
}

pub(super) fn buffered_charge<'a>(
    mut rows: impl Iterator<Item = &'a BufferedRow>,
    node: &str,
) -> Result<StateCharge> {
    rows.try_fold(StateCharge::default(), |charge, row| {
        charge.checked_add(buffered_row_charge(row, node)?, node)
    })
}

pub(super) fn ordered_record_charge(
    record: &RecordBatch,
    node: &str,
) -> Result<Option<StateCharge>> {
    let Some(payload) = super::super::row_cost::RowCosts::try_total(record)? else {
        return Ok(None);
    };
    let rows = charged_usize(record.num_rows(), node)?;
    let timezone_bytes = record
        .schema()
        .fields()
        .iter()
        .try_fold(0_u64, |total, field| {
            let DataType::Timestamp(_, Some(timezone)) = field.data_type() else {
                return Ok::<_, super::CalcFlowError>(total);
            };
            total
                .checked_add(charged_usize(timezone.len(), node)?)
                .ok_or_else(|| operator_error(node, "rolling state byte charge overflowed"))
        })?;
    let per_row = 256_u64
        .checked_add(
            charged_usize(size_of::<ScalarValue>(), node)?
                .checked_mul(charged_usize(record.num_columns(), node)?)
                .ok_or_else(|| operator_error(node, "rolling state byte charge overflowed"))?,
        )
        .and_then(|total| total.checked_add(timezone_bytes))
        .ok_or_else(|| operator_error(node, "rolling state byte charge overflowed"))?;
    let payload = charged_usize(payload, node)?;
    let bytes = per_row
        .checked_mul(rows)
        .and_then(|total| total.checked_add(payload))
        .ok_or_else(|| operator_error(node, "rolling state byte charge overflowed"))?;
    Ok(Some(StateCharge { rows, bytes }))
}

pub(super) fn ordered_buffer_record_charge(
    record: &RecordBatch,
    compiled: &CompiledRollingSpec,
    node: &str,
) -> Result<Option<StateCharge>> {
    let Some(mut charge) = ordered_record_charge(record, node)? else {
        return Ok(None);
    };
    for key in compiled
        .partition_columns
        .iter()
        .chain(&compiled.sequence_columns)
    {
        let column = record.column(key.index);
        if let Some(offsets) = super::super::row_cost::variable_offsets(column.as_ref()) {
            charge.bytes = charge
                .bytes
                .checked_add(charged_usize(offsets.total_width(), node)?)
                .ok_or_else(|| operator_error(node, "rolling state byte charge overflowed"))?;
        }
    }
    Ok(Some(charge))
}

pub(super) fn ordered_buffer_charge<'a>(
    records: impl Iterator<Item = &'a RecordBatch>,
    compiled: &CompiledRollingSpec,
    node: &str,
) -> Result<Option<StateCharge>> {
    let mut charge = StateCharge::default();
    for record in records {
        let Some(next) = ordered_buffer_record_charge(record, compiled, node)? else {
            return Ok(None);
        };
        charge = charge.checked_add(next, node)?;
    }
    Ok(Some(charge))
}

pub(super) fn history_entity_charge(
    key: &[Option<KeyValue>],
    state: &EntityRollingState,
    node: &str,
) -> Result<StateCharge> {
    let mut charge = history_entity_base_charge(key, state.windows.len(), node)?;
    for row in &state.rows {
        charge = charge.checked_add(scalar_history_row_charge(row, node)?, node)?;
    }
    for record in &state.columnar.records {
        let Some(next) = ordered_record_charge(record, node)? else {
            return Err(super::internal_error(
                "rolling columnar history has unsupported state charge",
            ));
        };
        charge = charge.checked_add(next, node)?;
    }
    for window in &state.windows {
        if let WindowState::Extrema(extrema) = window {
            for (key, value) in &extrema.queue {
                let dynamic = key_bytes(key.sequence.iter(), node)?
                    .checked_add(charged_usize(value.size(), node)?)
                    .ok_or_else(|| operator_error(node, "rolling state byte charge overflowed"))?;
                charge.bytes = charge
                    .bytes
                    .checked_add(dynamic)
                    .ok_or_else(|| operator_error(node, "rolling state byte charge overflowed"))?;
            }
        }
    }
    Ok(charge)
}

pub(super) fn histories_charge(histories: &RollingHistories, node: &str) -> Result<StateCharge> {
    histories
        .by_entity
        .iter()
        .try_fold(StateCharge::default(), |charge, (key, state)| {
            charge.checked_add(history_entity_charge(key, state, node)?, node)
        })
}

pub(super) fn changed_histories_charge(
    histories: &RollingHistories,
    touched: &HistoryUpdates,
    node: &str,
) -> Result<(StateCharge, StateCharge)> {
    let mut old = StateCharge::default();
    let mut new = StateCharge::default();
    for (key, state) in touched {
        if let Some(previous) = histories.by_entity.get(key) {
            old = old.checked_add(history_entity_charge(key, previous, node)?, node)?;
        }
        new = new.checked_add(history_entity_charge(key, state, node)?, node)?;
    }
    Ok((old, new))
}

pub(super) fn full_state_charge(
    buffer: &BTreeMap<RowIdentity, BufferedRow>,
    ordered: &super::ordered_stream::OrderedStreamBuffer,
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    node: &str,
) -> Result<StateCharge> {
    let buffered = buffered_charge(buffer.values(), node)?;
    let Some(ordered) = ordered_buffer_charge(ordered.records(), compiled, node)? else {
        return Err(super::internal_error(
            "rolling ordered state has unsupported state charge",
        ));
    };
    buffered
        .checked_add(ordered, node)?
        .checked_add(histories_charge(histories, node)?, node)
}
