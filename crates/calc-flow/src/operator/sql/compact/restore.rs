use datafusion::{arrow::datatypes::SchemaRef, execution::memory_pool::MemoryReservation};

use super::super::{
    SqlOperator, decode_sql_state,
    incremental::{self, IncrementalSql},
    metadata, retention, sql_state_error,
};
use super::{
    capture::CompactCapture,
    control, identity,
    storage::{self, CompactSqlState},
};
use crate::{OperatorStateSnapshot, Result};

pub(in crate::operator::sql) struct RestoredCompact {
    pub(in crate::operator::sql) state: CompactSqlState,
    pub(in crate::operator::sql) native: Box<IncrementalSql>,
}

pub(in crate::operator::sql) fn prepare(
    operator: &SqlOperator,
    snapshot: &OperatorStateSnapshot,
    check: &dyn Fn() -> Result<()>,
) -> Result<RestoredCompact> {
    check()?;
    if operator.aliases.len() != 1 || !operator.stream_aggregate || !operator.udfs.is_empty() {
        return Err(sql_state_error(
            "SQL compact checkpoint requires one native aggregate input",
        ));
    }
    validate_inventory(snapshot)?;
    let runtime = operator.retention_runtime()?;
    let _decode = decode_credit(operator, snapshot)?;
    let decoded = control::decode(
        runtime,
        &snapshot.segments["control"],
        &operator.name,
        check,
    )?;
    decoded
        .value
        .validate_inline(&snapshot.inline_metadata, &snapshot.segments["control"])?;
    decoded.value.validate_segments(
        &snapshot.segments["logical-schema"],
        &snapshot.segments["group-state"],
        &snapshot.segments["batch-metadata"],
    )?;
    decoded
        .value
        .group_log
        .validate(snapshot, decoded.value.groups, decoded.value.ledger)?;
    storage::validate_budget(operator, decoded.value.ledger)?;
    let logical = trusted_logical_schema(operator, snapshot)?;
    check()?;
    let projection = retention::SqlProjection::resolve(
        runtime,
        &operator.validated,
        &operator.aliases[0],
        logical.clone(),
        &operator.name,
    )?;
    let columns = storage::columns(operator, logical.clone(), projection)?;
    let native = trusted_plan(operator, logical, columns.physical.clone())?;
    check()?;
    let descriptor = native.native_descriptor(&operator.name)?;
    let identity = identity::build(
        operator,
        &columns.logical,
        &columns.physical,
        columns.ordinals.clone(),
        &descriptor,
    )?;
    decoded.value.validate_identity(&identity.value)?;
    let native = restore_groups(native, snapshot, &decoded.value, check, &operator.name)?;
    check()?;
    let (latest, metadata_owner) = metadata::decode(
        runtime,
        &snapshot.segments["batch-metadata"],
        &operator.name,
    )?;
    let capture = CompactCapture::restored(
        operator,
        snapshot,
        decoded.value.ledger,
        decoded.value.group_log.clone(),
        latest.clone(),
        metadata_owner,
    )?;
    let state = CompactSqlState::restored(
        operator,
        columns,
        decoded.value.ledger,
        latest,
        Some(capture),
    )?;
    check()?;
    Ok(RestoredCompact {
        state,
        native: Box::new(native),
    })
}

fn restore_groups(
    mut native: IncrementalSql,
    snapshot: &OperatorStateSnapshot,
    control: &control::CompactControl,
    check: &dyn Fn() -> Result<()>,
    name: &str,
) -> Result<IncrementalSql> {
    let state = decode_sql_state(snapshot.segments["group-state"].bytes())?;
    validate_state_census(&state, control.group_log.base_groups)?;
    native.restore_grouped_proof(
        &control.state_policy,
        state.num_rows(),
        control.ledger.rows,
        name,
    )?;
    let mut native = native.import_native_state(
        state.table_payload()?.batches(),
        control.group_log.base_ledger.rows,
        control.group_log.base_ledger.seen_input,
        check,
        name,
    )?;
    for frame in &control.group_log.frames {
        check()?;
        let state = decode_sql_state(snapshot.segments[&frame.id].bytes())?;
        validate_state_census(&state, frame.groups)?;
        native.apply_delta_state(
            state.table_payload()?.batches(),
            frame.ledger.rows,
            check,
            name,
        )?;
    }
    if native.group_count() as u64 != control.groups {
        return Err(sql_state_error(
            "SQL compact reconstructed group census is invalid",
        ));
    }
    native.validate_checkpoint_history(
        control.ledger.rows,
        control.ledger.seen_input,
        check,
        name,
    )?;
    check()?;
    Ok(native)
}

fn validate_inventory(snapshot: &OperatorStateSnapshot) -> Result<()> {
    let required = ["batch-metadata", "control", "group-state", "logical-schema"];
    if snapshot.segments.len() > 4 + super::log::MAX_FRAMES
        || required
            .iter()
            .any(|id| !snapshot.segments.contains_key(*id))
        || snapshot
            .segments
            .keys()
            .any(|id| !required.contains(&id.as_str()) && !id.starts_with("group-delta-"))
    {
        return Err(sql_state_error(
            "SQL compact checkpoint segment inventory is invalid",
        ));
    }
    Ok(())
}

fn trusted_logical_schema(
    operator: &SqlOperator,
    snapshot: &OperatorStateSnapshot,
) -> Result<SchemaRef> {
    let logical = retention::decode_schema(&snapshot.segments["logical-schema"])?;
    if operator.input_ports[0]
        .schema()
        .is_some_and(|declared| declared != &logical)
    {
        return Err(sql_state_error(
            "SQL compact logical schema does not match the declared input",
        ));
    }
    Ok(logical)
}

fn trusted_plan(
    operator: &SqlOperator,
    logical: SchemaRef,
    physical: SchemaRef,
) -> Result<IncrementalSql> {
    let native = IncrementalSql::plan_sync(
        operator.retention_runtime()?,
        &operator.validated,
        &operator.aliases[0],
        logical,
        physical,
        &operator.name,
    )?
    .ok_or_else(|| sql_state_error("SQL compact checkpoint query is not native eligible"))?;
    let descriptor = native.native_descriptor(&operator.name)?;
    if operator.output_ports[0]
        .schema()
        .is_some_and(|expected| expected != &descriptor.output_schema)
    {
        return Err(sql_state_error(
            "SQL compact output schema does not match the declared output",
        ));
    }
    Ok(native)
}

fn validate_state_census(batch: &crate::Batch, groups: u64) -> Result<()> {
    if batch.metadata() != &crate::BatchMetadata::default()
        || u64::try_from(batch.num_rows()).ok() != Some(groups)
    {
        return Err(sql_state_error(
            "SQL compact native group census is invalid",
        ));
    }
    Ok(())
}

fn decode_credit(
    operator: &SqlOperator,
    snapshot: &OperatorStateSnapshot,
) -> Result<MemoryReservation> {
    let bytes = snapshot
        .segments
        .values()
        .try_fold(32768, |bound, segment| {
            incremental::checked_bytes(bound, [(segment.bytes().len(), 16)], &operator.name)
        })?;
    let reservation = operator
        .retention_runtime()?
        .incremental_reservation(&operator.name);
    incremental::ensure_reservation(&reservation, bytes, &operator.name)?;
    Ok(reservation)
}
