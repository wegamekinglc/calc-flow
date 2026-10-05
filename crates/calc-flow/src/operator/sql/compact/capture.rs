use std::{collections::BTreeMap, sync::Arc};

use datafusion::execution::memory_pool::MemoryReservation;

use super::super::{
    SqlCheckpointInput, SqlOperator, encode_sql_state_async,
    incremental::{self, IncrementalSql},
    ipc, metadata, record_copy_reservation, sql_state_error,
};
use super::{
    control::{self, CompactControl, QuotaLedger, SegmentDigests},
    identity,
    log::{GroupLog, LogDescriptor, Mode},
    storage::CompactSqlState,
};
use crate::{
    Batch, BatchMetadata, DataFusionRuntime, OperatorStateSnapshot, Result, StreamOperatorContext,
};

pub(in crate::operator::sql) struct CompactCapture {
    pub snapshot: OperatorStateSnapshot,
    group_state: GroupLog,
    ledger: QuotaLedger,
    metadata: BatchMetadata,
    _metadata: Arc<metadata::SqlMetadata>,
    _reservation: Arc<MemoryReservation>,
}

impl CompactCapture {
    #[cfg(test)]
    pub(in crate::operator::sql) fn checkpoint_fee_for_test(&self) -> &Arc<MemoryReservation> {
        let Self {
            _reservation: reservation,
            ..
        } = self;
        reservation
    }

    pub(super) fn restored(
        operator: &SqlOperator,
        snapshot: &OperatorStateSnapshot,
        ledger: QuotaLedger,
        descriptor: LogDescriptor,
        metadata: BatchMetadata,
        metadata_owner: Arc<metadata::SqlMetadata>,
    ) -> Result<Arc<Self>> {
        let reservation = capture_credit(operator, descriptor.frames.len())?;
        let bytes = snapshot
            .segments
            .values()
            .try_fold(16384, |bytes, segment| {
                incremental::checked_bytes(bytes, [(segment.bytes().len(), 2)], &operator.name)
            })?;
        incremental::ensure_reservation(&reservation, bytes, &operator.name)?;
        let reservation = Arc::new(reservation);
        let mut snapshot = snapshot.clone();
        let mut group_state = GroupLog::restored(&snapshot, descriptor);
        group_state.base = group_state.base.with_owner(reservation.clone());
        for frame in &mut group_state.frames {
            frame.segment = frame.segment.clone().with_owner(reservation.clone());
        }
        snapshot
            .segments
            .insert("batch-metadata".into(), metadata_owner.segment.clone());
        for segment in snapshot.segments.values_mut() {
            *segment = segment.clone().with_owner(reservation.clone());
        }
        Ok(Arc::new(Self {
            snapshot,
            group_state,
            ledger,
            metadata,
            _metadata: metadata_owner,
            _reservation: reservation,
        }))
    }
}

struct CaptureParts {
    identity: identity::PaidIdentity,
    group_state: GroupLog,
    metadata: Arc<metadata::SqlMetadata>,
    groups: usize,
    policy: incremental::grouped_float::Policy,
    coalescer: Option<incremental::global_record::coalescer::Capture>,
    reservation: Arc<MemoryReservation>,
}

fn mode(state: &CompactSqlState, native: &IncrementalSql) -> Mode {
    state.capture.as_ref().map_or(Mode::Full, |capture| {
        capture
            .group_state
            .mode(native.checkpoint_changes(), native.group_count())
    })
}

fn frame_count(state: &CompactSqlState, native: &IncrementalSql) -> usize {
    state.capture.as_ref().map_or(0, |capture| {
        capture.group_state.frame_count(mode(state, native))
    })
}

fn cached(state: &CompactSqlState) -> Option<Arc<CompactCapture>> {
    state
        .capture
        .as_ref()
        .filter(|capture| capture.ledger == state.ledger && capture.metadata == state.metadata)
        .cloned()
}

pub(in crate::operator::sql) fn prepare(
    operator: &SqlOperator,
    state: &CompactSqlState,
    native: &IncrementalSql,
    check: &dyn Fn() -> Result<()>,
) -> Result<Arc<CompactCapture>> {
    check()?;
    if let Some(capture) = cached(state) {
        return Ok(capture);
    }
    let runtime = operator.retention_runtime()?;
    let reservation = Arc::new(capture_credit(operator, frame_count(state, native))?);
    let (parts, _descriptor) =
        sync_capture_parts(operator, state, native, runtime, reservation, check)?;
    finish(operator, state, parts, check)
}

fn sync_capture_parts(
    operator: &SqlOperator,
    state: &CompactSqlState,
    native: &IncrementalSql,
    runtime: &DataFusionRuntime,
    reservation: Arc<MemoryReservation>,
    check: &dyn Fn() -> Result<()>,
) -> Result<(
    CaptureParts,
    incremental::compact_state::NativeStateDescriptor,
)> {
    let (descriptor, group_state) = state_segment(operator, state, native, check)?;
    let identity = identity::build(
        operator,
        &state.columns.logical,
        &state.columns.physical,
        state.columns.ordinals.clone(),
        &descriptor,
    )?;
    check()?;
    let metadata = metadata::encode(
        &state.metadata,
        metadata::reserve(runtime, &state.metadata, &operator.name)?,
        check,
    )?;
    let parts = CaptureParts {
        identity,
        group_state,
        metadata,
        groups: native.group_count(),
        policy: native.checkpoint_policy(),
        coalescer: native.capture_coalescer(&operator.name, check)?,
        reservation,
    };
    Ok((parts, descriptor))
}

pub(in crate::operator::sql) async fn prepare_async(
    operator: &SqlOperator,
    state: &CompactSqlState,
    native: &IncrementalSql,
    context: &StreamOperatorContext<'_>,
) -> Result<Arc<CompactCapture>> {
    context.check_cancelled()?;
    if let Some(capture) = cached(state) {
        return Ok(capture);
    }
    let reservation = Arc::new(capture_credit(operator, frame_count(state, native))?);
    let input = prepare_async_input(operator, state, native, context).await?;
    finish_async_capture(operator, state, native, input, reservation, context).await
}

struct AsyncCaptureInput {
    input: Option<SqlCheckpointInput>,
    metadata: Option<(BatchMetadata, MemoryReservation)>,
    identity: identity::PaidIdentity,
    descriptor: Option<incremental::compact_state::NativeStateDescriptor>,
    mode: Mode,
    groups: usize,
}

async fn prepare_async_input(
    operator: &SqlOperator,
    state: &CompactSqlState,
    native: &IncrementalSql,
    context: &StreamOperatorContext<'_>,
) -> Result<AsyncCaptureInput> {
    let mode = mode(state, native);
    let (export, descriptor) = async_state_export(native, mode, &operator.name, context).await?;
    let description = export.as_ref().map_or_else(
        || descriptor.as_ref().expect("reused native descriptor"),
        |export| &export.descriptor,
    );
    let groups = native.group_count();
    let identity = identity::build(
        operator,
        &state.columns.logical,
        &state.columns.physical,
        state.columns.ordinals.clone(),
        description,
    )?;
    let input = match export {
        Some(export) => Some(native_checkpoint_input(operator, export, context).await?),
        None => None,
    };
    let runtime = operator.retention_runtime()?;
    let metadata_fee = metadata::reserve(runtime, &state.metadata, &operator.name)?;
    let metadata = Some((state.metadata.clone(), metadata_fee));
    Ok(AsyncCaptureInput {
        input,
        metadata,
        identity,
        descriptor,
        mode,
        groups,
    })
}

async fn finish_async_capture(
    operator: &SqlOperator,
    state: &CompactSqlState,
    native: &IncrementalSql,
    input: AsyncCaptureInput,
    reservation: Arc<MemoryReservation>,
    context: &StreamOperatorContext<'_>,
) -> Result<Arc<CompactCapture>> {
    context.check_cancelled()?;
    let attempt = tokio_util::sync::CancellationToken::new();
    let _cancel_on_drop = attempt.clone().drop_guard();
    let (segment, metadata, export) =
        encode_sql_state_async(input.input, input.metadata, context.job().clone(), attempt).await?;
    context.check_cancelled()?;
    let group_state = GroupLog::finish(
        state.capture.as_ref().map(|capture| &capture.group_state),
        input.mode,
        segment.as_ref().map(|segment| segment.segment.clone()),
        input.groups,
        native.checkpoint_changes(),
        state.ledger,
    )?;
    let metadata = metadata.expect("encoded compact metadata");
    let capture = finish(
        operator,
        state,
        CaptureParts {
            identity: input.identity,
            group_state,
            metadata,
            groups: input.groups,
            policy: native.checkpoint_policy(),
            coalescer: native
                .capture_coalescer_async(&operator.name, context)
                .await?,
            reservation,
        },
        &|| context.check_cancelled(),
    )?;
    drop(export);
    drop(input.descriptor);
    Ok(capture)
}

async fn async_state_export(
    native: &IncrementalSql,
    mode: Mode,
    name: &str,
    context: &StreamOperatorContext<'_>,
) -> Result<(
    Option<incremental::compact_state::PaidNativeStateRecords>,
    Option<incremental::compact_state::NativeStateDescriptor>,
)> {
    let export = match mode {
        Mode::Carry => None,
        Mode::Full => Some(
            native
                .export_native_state_async(name, || context.check_cancelled())
                .await?,
        ),
        Mode::Delta => Some(
            native
                .export_dirty_state_async(name, || context.check_cancelled())
                .await?,
        ),
    };
    let descriptor = if export.is_none() {
        Some(native.native_descriptor(name)?)
    } else {
        None
    };
    Ok((export, descriptor))
}

async fn native_checkpoint_input(
    operator: &SqlOperator,
    export: incremental::compact_state::PaidNativeStateRecords,
    context: &StreamOperatorContext<'_>,
) -> Result<SqlCheckpointInput> {
    let runtime = operator.retention_runtime()?;
    let reservation = record_copy_reservation(
        runtime,
        &operator.name,
        export.records().len(),
        export.descriptor.wire_schema.fields().len(),
        2,
    )?;
    #[cfg(test)]
    let native_name = {
        reservation
            .try_grow(incremental::checked_bytes(
                128,
                [(operator.name.capacity(), 1)],
                &operator.name,
            )?)
            .map_err(|error| crate::CalcFlowError::DataFusion {
                node_id: Some(operator.name.clone()),
                message: error.to_string(),
            })?;
        Some(operator.name.clone())
    };
    let mut records = Vec::with_capacity(export.records().len());
    for chunk in export.records().chunks(128) {
        context.check_cancelled()?;
        records.extend_from_slice(chunk);
        tokio::task::yield_now().await;
    }
    context.check_cancelled()?;
    Ok(SqlCheckpointInput {
        records,
        metadata: BatchMetadata::default(),
        reservation,
        input_reservation: runtime.incremental_reservation(&operator.name),
        native: Some(export),
        #[cfg(test)]
        native_name,
    })
}

fn finish(
    operator: &SqlOperator,
    state: &CompactSqlState,
    parts: CaptureParts,
    check: &dyn Fn() -> Result<()>,
) -> Result<Arc<CompactCapture>> {
    let runtime = operator.retention_runtime()?;
    let logical = state.columns.logical_segment.clone();
    let control = CompactControl {
        state_layout: 3,
        state_accounting: 3,
        native_semantics: 1,
        datafusion_version: "54.0.0".into(),
        state_policy: parts.policy,
        coalescer: parts.coalescer.as_ref().map_or(
            incremental::global_record::coalescer::Inventory::None,
            incremental::global_record::coalescer::Capture::inventory,
        ),
        identity: parts.identity.value,
        ledger: state.ledger,
        group_log: parts.group_state.descriptor.clone(),
        groups: u64::try_from(parts.groups)
            .map_err(|_| sql_state_error("SQL compact group count exceeds u64"))?,
        segments: SegmentDigests {
            logical_schema: logical.sha256().into(),
            group_state: parts.group_state.base.sha256().into(),
            batch_metadata: parts.metadata.segment.sha256().into(),
        },
    };
    let encoded = control::encode(runtime, &control, &operator.name, check)?;
    let mut snapshot = OperatorStateSnapshot {
        inline_metadata: control.inline_metadata(&encoded.segment),
        segments: BTreeMap::from([
            ("control".into(), encoded.segment.clone()),
            ("logical-schema".into(), logical),
            ("batch-metadata".into(), parts.metadata.segment.clone()),
        ]),
    };
    parts.group_state.segments(&mut snapshot.segments);
    if let Some(coalescer) = parts.coalescer {
        snapshot
            .segments
            .insert("global-complete".into(), coalescer.complete);
        snapshot
            .segments
            .insert("global-tail".into(), coalescer.tail);
    }
    for segment in snapshot.segments.values_mut() {
        *segment = segment.clone().with_owner(parts.reservation.clone());
    }
    check()?;
    Ok(Arc::new(CompactCapture {
        snapshot,
        group_state: parts.group_state,
        ledger: state.ledger,
        metadata: state.metadata.clone(),
        _metadata: parts.metadata,
        _reservation: parts.reservation,
    }))
}

fn state_segment(
    operator: &SqlOperator,
    state: &CompactSqlState,
    native: &IncrementalSql,
    check: &dyn Fn() -> Result<()>,
) -> Result<(incremental::compact_state::NativeStateDescriptor, GroupLog)> {
    let mode = mode(state, native);
    if matches!(mode, Mode::Carry) {
        let capture = state.capture.as_ref().expect("carried group log");
        return Ok((
            native.native_descriptor(&operator.name)?,
            capture.group_state.clone(),
        ));
    }
    let export = sync_state_export(native, mode, &operator.name, check)?;
    let encoded = encode_native_segment(operator, &export, check)?;
    let segment = encoded.segment.clone();
    let log = GroupLog::finish(
        state.capture.as_ref().map(|capture| &capture.group_state),
        mode,
        Some(segment),
        native.group_count(),
        native.checkpoint_changes(),
        state.ledger,
    )?;
    Ok((export.into_descriptor(), log))
}

fn sync_state_export(
    native: &IncrementalSql,
    mode: Mode,
    name: &str,
    check: &dyn Fn() -> Result<()>,
) -> Result<incremental::compact_state::PaidNativeStateRecords> {
    match mode {
        Mode::Delta => native.export_dirty_state(name, check),
        Mode::Full => native.export_native_state(name, check),
        Mode::Carry => unreachable!(),
    }
}

fn encode_native_segment(
    operator: &SqlOperator,
    export: &incremental::compact_state::PaidNativeStateRecords,
    check: &dyn Fn() -> Result<()>,
) -> Result<Arc<ipc::SqlInputSegment>> {
    let runtime = operator.retention_runtime()?;
    let shell = record_copy_reservation(
        runtime,
        &operator.name,
        export.records().len(),
        export.descriptor.wire_schema.fields().len(),
        2,
    )?;
    let wire = Batch::table(export.records().to_vec(), BatchMetadata::default())?;
    #[cfg(test)]
    super::direct_async_tests::before_ipc(
        &operator.name,
        export.records(),
        export.reserved_bytes(),
    );
    let encoded = ipc::encode(
        &wire,
        operator
            .retention_runtime()?
            .incremental_reservation(&operator.name),
        check,
    )?;
    drop(wire);
    drop(shell);
    Ok(encoded)
}

fn capture_credit(operator: &SqlOperator, frames: usize) -> Result<MemoryReservation> {
    let reservation = operator
        .retention_runtime()?
        .incremental_reservation(&operator.name);
    let bytes = incremental::checked_bytes(16384, [(frames, 2048)], &operator.name)?;
    incremental::ensure_reservation(&reservation, bytes, &operator.name)?;
    Ok(reservation)
}

#[cfg(test)]
mod current_safety_tests;
