use datafusion::execution::memory_pool::MemoryReservation;
use serde_json::Value;

use super::{
    Batch, DataFusionRuntime, OperatorStateSnapshot, Result, RetainedSqlInput, SqlOperator,
    StateSegment, decode_sql_state, incremental, ipc, metadata, record_copy_reservation, retention,
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
        self.prepare_nonempty_restore(snapshot, check_cancelled)
    }

    fn prepare_nonempty_restore(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        check_cancelled: &dyn Fn() -> Result<()>,
    ) -> Result<PreparedSqlRestore> {
        self.stream_state.runtime()?;
        let input = self.legacy_restore_input(snapshot)?;
        let runtime = self.retention_runtime()?;
        let backing = reserve_restore_decode(runtime, &input.segment, &self.name)?;
        let (batch, metadata_segment, copies) =
            self.decode_restore_batch(snapshot, &input, runtime, check_cancelled)?;
        let retained =
            self.prepare_restored_input(&batch, input, metadata_segment, backing, copies, runtime)?;
        check_cancelled()?;
        Ok(PreparedSqlRestore {
            retained: Some(retained),
        })
    }

    fn decode_restore_batch(
        &self,
        snapshot: &OperatorStateSnapshot,
        input: &LegacyRestoreInput,
        runtime: &DataFusionRuntime,
        check_cancelled: &dyn Fn() -> Result<()>,
    ) -> Result<(
        Batch,
        Option<std::sync::Arc<metadata::SqlMetadata>>,
        MemoryReservation,
    )> {
        check_cancelled()?;
        let batch = decode_sql_state(input.segment.bytes())?;
        check_cancelled()?;
        let (batch, metadata_segment) =
            self.validate_restore_schema(batch, snapshot, input, runtime)?;
        self.validate_checkpoint_charge(&batch, input.rows, input.bytes)?;
        let table = batch.table_payload()?;
        let copies = record_copy_reservation(
            runtime,
            &self.name,
            table.batches().len(),
            table.schema().fields().len(),
            1,
        )?;
        Ok((batch, metadata_segment, copies))
    }

    fn validate_restore_schema(
        &self,
        batch: Batch,
        snapshot: &OperatorStateSnapshot,
        input: &LegacyRestoreInput,
        runtime: &DataFusionRuntime,
    ) -> Result<(Batch, Option<std::sync::Arc<metadata::SqlMetadata>>)> {
        if let Some(projection) = &input.projection {
            self.decode_projected_restore_batch(&batch, snapshot, projection, runtime)
        } else if self.input_ports[0]
            .schema()
            .is_some_and(|schema| schema != batch.table_payload().expect("decoded table").schema())
        {
            Err(sql_state_error(
                "SQL legacy logical schema does not match the declared input",
            ))
        } else {
            Ok((batch, None))
        }
    }

    fn decode_projected_restore_batch(
        &self,
        batch: &Batch,
        snapshot: &OperatorStateSnapshot,
        projection: &retention::SqlProjection,
        runtime: &DataFusionRuntime,
    ) -> Result<(Batch, Option<std::sync::Arc<metadata::SqlMetadata>>)> {
        if batch.table_payload()?.schema() != projection.columns.physical_schema() {
            return Err(sql_state_error(
                "SQL checkpoint physical schema does not match its trusted dependencies",
            ));
        }
        let (metadata, encoded) =
            metadata::decode(runtime, &snapshot.segments["batch-metadata"], &self.name)?;
        let batch = Batch::table(batch.table_payload()?.batches().to_vec(), metadata)?;
        Ok((batch, Some(encoded)))
    }

    fn prepare_restored_input(
        &self,
        batch: &Batch,
        input: LegacyRestoreInput,
        metadata_segment: Option<std::sync::Arc<metadata::SqlMetadata>>,
        backing: MemoryReservation,
        copies: MemoryReservation,
        runtime: &DataFusionRuntime,
    ) -> Result<RetainedSqlInput> {
        let LegacyRestoreInput {
            projection,
            segment,
            rows,
            bytes,
        } = input;
        let projected = projection.is_some();
        let table = batch.table_payload()?;
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
        Ok(retained)
    }

    fn legacy_restore_input(&self, snapshot: &OperatorStateSnapshot) -> Result<LegacyRestoreInput> {
        let projected = snapshot.inline_metadata.contains_key("state_layout");
        let projection = self.legacy_restore_projection(snapshot, projected)?;
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
        let (rows, bytes) = legacy_restore_counts(snapshot)?;
        Ok(LegacyRestoreInput {
            projection,
            segment,
            rows,
            bytes,
        })
    }

    fn legacy_restore_projection(
        &self,
        snapshot: &OperatorStateSnapshot,
        projected: bool,
    ) -> Result<Option<std::sync::Arc<retention::SqlProjection>>> {
        if projected {
            self.read_projection(snapshot).map(Some)
        } else if !self.checkpoint_matches(snapshot) {
            Err(sql_state_error(
                "SQL aggregate checkpoint does not match this operator",
            ))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn install_restore(&mut self, prepared: PreparedSqlRestore) {
        self.incremental = None;
        self.incremental_checked = false;
        self.retained = prepared.retained;
    }
}

fn reserve_restore_decode(
    runtime: &DataFusionRuntime,
    segment: &StateSegment,
    name: &str,
) -> Result<MemoryReservation> {
    let backing = runtime.incremental_reservation(name);
    let decode_bytes = incremental::checked_bytes(4096, [(segment.bytes().len(), 4)], name)?;
    incremental::ensure_reservation(&backing, decode_bytes, name)?;
    Ok(backing)
}

fn legacy_restore_counts(snapshot: &OperatorStateSnapshot) -> Result<(u64, u64)> {
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
    Ok((rows, bytes))
}

#[cfg(test)]
#[path = "restore_tests.rs"]
mod tests;
