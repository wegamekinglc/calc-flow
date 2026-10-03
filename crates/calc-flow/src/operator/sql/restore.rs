use serde_json::Value;

use super::{
    Batch, OperatorStateSnapshot, Result, RetainedSqlInput, SqlOperator, StateSegment,
    decode_sql_state, incremental, ipc, metadata, record_copy_reservation, retention,
    sql_state_error,
};

pub(crate) struct PreparedSqlRestore {
    retained: Option<RetainedSqlInput>,
}

struct LegacyRestoreInput {
    projection: Option<std::sync::Arc<retention::SqlProjection>>,
    segment: StateSegment,
    rows: u64,
    bytes: u64,
}

impl SqlOperator {
    pub(crate) fn prepare_restore(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        check_cancelled: &dyn Fn() -> Result<()>,
    ) -> Result<PreparedSqlRestore> {
        check_cancelled()?;
        if snapshot.inline_metadata.is_empty() && snapshot.segments.is_empty() {
            return Ok(PreparedSqlRestore { retained: None });
        }
        self.stream_state.runtime()?;
        let LegacyRestoreInput {
            projection,
            segment,
            rows,
            bytes,
        } = self.legacy_restore_input(snapshot)?;
        let projected = projection.is_some();
        let runtime = self.retention_runtime()?;
        let backing = runtime.incremental_reservation(&self.name);
        let decode_bytes =
            incremental::checked_bytes(4096, [(segment.bytes().len(), 4)], &self.name)?;
        incremental::ensure_reservation(&backing, decode_bytes, &self.name)?;
        check_cancelled()?;
        let mut batch = decode_sql_state(segment.bytes())?;
        check_cancelled()?;
        let mut metadata_segment = None;
        if let Some(projection) = &projection {
            if batch.table_payload()?.schema() != projection.columns.physical_schema() {
                return Err(sql_state_error(
                    "SQL checkpoint physical schema does not match its trusted dependencies",
                ));
            }
            let (metadata, encoded) =
                metadata::decode(runtime, &snapshot.segments["batch-metadata"], &self.name)?;
            batch = Batch::table(batch.table_payload()?.batches().to_vec(), metadata)?;
            metadata_segment = Some(encoded);
        } else if self.input_ports[0]
            .schema()
            .is_some_and(|schema| schema != batch.table_payload().expect("decoded table").schema())
        {
            return Err(sql_state_error(
                "SQL legacy logical schema does not match the declared input",
            ));
        }
        self.validate_checkpoint_charge(&batch, rows, bytes)?;
        let table = batch.table_payload()?;
        let copies = record_copy_reservation(
            runtime,
            &self.name,
            table.batches().len(),
            table.schema().fields().len(),
            1,
        )?;
        let backing = std::sync::Arc::new(backing);
        let mut retained = RetainedSqlInput {
            records: Vec::new(),
            metadata: batch.metadata().clone(),
            projection,
            projection_checked: projected,
            backing_reservations: vec![backing.clone()],
            reservation: None,
            segment: Some(ipc::SqlInputSegment::restored(segment, backing)),
            metadata_segment,
            rows,
            bytes,
        };
        retained.reserve_append(
            table.batches().len(),
            runtime,
            &self.name,
            table.schema().fields().len(),
        )?;
        retained.records.extend(table.batches().iter().cloned());
        drop(copies);
        check_cancelled()?;
        Ok(PreparedSqlRestore {
            retained: Some(retained),
        })
    }

    fn legacy_restore_input(&self, snapshot: &OperatorStateSnapshot) -> Result<LegacyRestoreInput> {
        let projected = snapshot.inline_metadata.contains_key("state_layout");
        let projection = if projected {
            Some(self.read_projection(snapshot)?)
        } else {
            if !self.checkpoint_matches(snapshot) {
                return Err(sql_state_error(
                    "SQL aggregate checkpoint does not match this operator",
                ));
            }
            None
        };
        let input_key = if projected {
            "input-projected"
        } else {
            "input"
        };
        let segment = snapshot
            .segments
            .get(input_key)
            .ok_or_else(|| sql_state_error("SQL checkpoint has no retained input segment"))?
            .clone();
        let rows = snapshot
            .inline_metadata
            .get("rows")
            .and_then(Value::as_u64)
            .ok_or_else(|| sql_state_error("SQL checkpoint has no row count"))?;
        let bytes = snapshot
            .inline_metadata
            .get("bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| sql_state_error("SQL checkpoint has no byte count"))?;
        Ok(LegacyRestoreInput {
            projection,
            segment,
            rows,
            bytes,
        })
    }

    pub(crate) fn install_restore(&mut self, prepared: PreparedSqlRestore) {
        self.incremental = None;
        self.incremental_checked = false;
        self.retained = prepared.retained;
    }
}

#[cfg(test)]
#[path = "restore_tests.rs"]
mod tests;
