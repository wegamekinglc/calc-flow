use datafusion::execution::memory_pool::MemoryReservation;
use serde_json::Value;

#[cfg(test)]
use super::recovery_test_hooks;

use super::{
    OperatorStateSnapshot, Result, RetainedSqlInput, SqlOperator, incremental, sql_state_error,
};

pub(crate) struct PreparedSqlRestore {
    retained: Option<RetainedSqlInput>,
    compact: Option<super::compact::RestoredCompact>,
    retained_capture: Option<std::sync::Arc<super::current_retained::RetainedCapture>>,
}

impl SqlOperator {
    pub(crate) fn reserve_recovery_envelope(
        &mut self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<MemoryReservation> {
        let bound = recovery_envelope_bytes(snapshot, &self.name)?;
        self.stream_state.runtime()?;
        let reservation = self
            .retention_runtime()?
            .incremental_reservation(&self.name);
        incremental::ensure_reservation(&reservation, bound, &self.name)?;
        Ok(reservation)
    }

    pub(crate) fn reserve_checkpoint_envelope(
        &mut self,
        node_id_bytes: usize,
    ) -> Result<MemoryReservation> {
        let reservation = self.reserve_recovery_envelope(&OperatorStateSnapshot::default())?;
        let bound = incremental::checked_bytes(
            reservation.size(),
            [(node_id_bytes, 4), (size_of::<String>(), 4)],
            &self.name,
        )?;
        incremental::ensure_reservation(&reservation, bound, &self.name)?;
        Ok(reservation)
    }

    pub(crate) fn prepare_restore(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        check_cancelled: &dyn Fn() -> Result<()>,
    ) -> Result<PreparedSqlRestore> {
        check_cancelled()?;
        if snapshot.inline_metadata.is_empty() && snapshot.segments.is_empty() {
            return Ok(PreparedSqlRestore {
                retained: None,
                compact: None,
                retained_capture: None,
            });
        }
        let layout = current_layout(snapshot)?;
        self.stream_state.runtime()?;
        let prepared = self.prepare_layout_restore(snapshot, layout, check_cancelled)?;
        check_cancelled()?;
        Ok(prepared)
    }

    fn prepare_layout_restore(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        layout: u64,
        check_cancelled: &dyn Fn() -> Result<()>,
    ) -> Result<PreparedSqlRestore> {
        if layout == 3 {
            let compact = super::compact::prepare_restore(self, snapshot, check_cancelled)?;
            #[cfg(test)]
            recovery_test_hooks::prepared_compact(self, &compact.state, &compact.native)?;
            Ok(PreparedSqlRestore {
                retained: None,
                compact: Some(compact),
                retained_capture: None,
            })
        } else {
            let (retained, capture) =
                super::current_retained::prepare_restore(self, snapshot, check_cancelled)?;
            #[cfg(test)]
            recovery_test_hooks::prepared(self, &retained)?;
            Ok(PreparedSqlRestore {
                retained: Some(retained),
                compact: None,
                retained_capture: Some(capture),
            })
        }
    }

    pub(crate) fn install_restore(&mut self, prepared: PreparedSqlRestore) {
        #[cfg(test)]
        if let Some(retained) = &prepared.retained {
            recovery_test_hooks::installing(self, retained);
        }
        #[cfg(test)]
        if let Some(compact) = &prepared.compact {
            recovery_test_hooks::installing_metadata(self, &compact.state.metadata);
        }
        self.incremental_checked = prepared.compact.is_some()
            || prepared
                .retained
                .as_ref()
                .is_some_and(|state| state.rows != 0);
        self.retained_capture = prepared.retained_capture;
        if let Some(compact) = prepared.compact {
            self.incremental = Some(compact.native);
            self.compact = Some(Box::new(compact.state));
            self.retained = None;
        } else {
            self.incremental = None;
            self.compact = None;
            self.retained = prepared.retained;
        }
    }
}

#[cfg(test)]
#[path = "restore_tests.rs"]
mod tests;

fn recovery_value_bytes(value: &Value) -> Option<usize> {
    let payload = match value {
        Value::String(value) => Some(value.len()),
        Value::Array(values) => values.iter().try_fold(0usize, |bytes, value| {
            bytes.checked_add(recovery_value_bytes(value)?)
        }),
        Value::Object(values) => values.iter().try_fold(0usize, |bytes, (key, value)| {
            bytes
                .checked_add(key.len())?
                .checked_add(recovery_value_bytes(value)?)
        }),
        _ => Some(0),
    }?;
    payload.checked_add(256)
}

fn recovery_envelope_bytes(snapshot: &OperatorStateSnapshot, name: &str) -> Result<usize> {
    let metadata = snapshot
        .inline_metadata
        .iter()
        .try_fold(0usize, |bytes, (key, value)| {
            bytes
                .checked_add(key.len())
                .and_then(|bytes| bytes.checked_add(recovery_value_bytes(value)?))
        });
    let handles = snapshot.segments.keys().try_fold(0usize, |bytes, key| {
        bytes.checked_add(key.len().checked_add(256)?)
    });
    metadata
        .zip(handles)
        .and_then(|(metadata, handles)| metadata.checked_add(handles))
        .and_then(|bytes| bytes.checked_add(name.len().checked_mul(4)?))
        .and_then(|bytes| bytes.checked_mul(4))
        .and_then(|bytes| bytes.checked_add(4096))
        .ok_or_else(|| sql_state_error("SQL recovery envelope size overflowed"))
}

fn current_layout(snapshot: &OperatorStateSnapshot) -> Result<u64> {
    match snapshot
        .inline_metadata
        .get("state_layout")
        .and_then(Value::as_u64)
    {
        Some(layout @ (3 | 4)) => Ok(layout),
        _ => Err(sql_state_error(
            "SQL checkpoint layout is unsupported (expected 3 or 4)",
        )),
    }
}
