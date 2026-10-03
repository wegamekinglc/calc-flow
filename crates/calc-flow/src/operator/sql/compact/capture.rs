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
    storage::CompactSqlState,
};
use crate::{Batch, BatchMetadata, OperatorStateSnapshot, Result, StreamOperatorContext};

pub(in crate::operator::sql) struct CompactCapture {
    pub snapshot: OperatorStateSnapshot,
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
        metadata: BatchMetadata,
        metadata_owner: Arc<metadata::SqlMetadata>,
    ) -> Result<Arc<Self>> {
        let reservation = capture_credit(operator)?;
        let bytes = snapshot
            .segments
            .values()
            .try_fold(16384, |bytes, segment| {
                incremental::checked_bytes(bytes, [(segment.bytes().len(), 2)], &operator.name)
            })?;
        incremental::ensure_reservation(&reservation, bytes, &operator.name)?;
        let reservation = Arc::new(reservation);
        let mut snapshot = snapshot.clone();
        snapshot
            .segments
            .insert("batch-metadata".into(), metadata_owner.segment.clone());
        for segment in snapshot.segments.values_mut() {
            *segment = segment.clone().with_owner(reservation.clone());
        }
        Ok(Arc::new(Self {
            snapshot,
            ledger,
            metadata,
            _metadata: metadata_owner,
            _reservation: reservation,
        }))
    }
}

struct CaptureParts {
    identity: identity::PaidIdentity,
    group_state: super::super::StateSegment,
    metadata: Arc<metadata::SqlMetadata>,
    groups: usize,
    reservation: Arc<MemoryReservation>,
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
    let reservation = Arc::new(capture_credit(operator)?);
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
    finish(
        operator,
        state,
        CaptureParts {
            identity,
            group_state,
            metadata,
            groups: descriptor.group_count,
            reservation,
        },
        check,
    )
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
    let reservation = Arc::new(capture_credit(operator)?);
    let export = if state.capture.is_none() {
        Some(
            native
                .export_native_state_async(&operator.name, || context.check_cancelled())
                .await?,
        )
    } else {
        None
    };
    let descriptor = if export.is_none() {
        Some(native.native_descriptor(&operator.name)?)
    } else {
        None
    };
    let description = export.as_ref().map_or_else(
        || descriptor.as_ref().expect("reused native descriptor"),
        |export| &export.descriptor,
    );
    let groups = description.group_count;
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
    context.check_cancelled()?;
    let attempt = tokio_util::sync::CancellationToken::new();
    let _cancel_on_drop = attempt.clone().drop_guard();
    let (segment, metadata, export) =
        encode_sql_state_async(input, metadata, context.job().clone(), attempt).await?;
    context.check_cancelled()?;
    let group_state = segment.map_or_else(
        || {
            state
                .capture
                .as_ref()
                .expect("reused native segment")
                .snapshot
                .segments["group-state"]
                .clone()
        },
        |segment| segment.segment.clone(),
    );
    let metadata = metadata.expect("encoded compact metadata");
    let capture = finish(
        operator,
        state,
        CaptureParts {
            identity,
            group_state,
            metadata,
            groups,
            reservation,
        },
        &|| context.check_cancelled(),
    )?;
    drop(export);
    drop(descriptor);
    Ok(capture)
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
        state_policy: "exact_numeric_v1".into(),
        identity: parts.identity.value,
        ledger: state.ledger,
        groups: u64::try_from(parts.groups)
            .map_err(|_| sql_state_error("SQL compact group count exceeds u64"))?,
        segments: SegmentDigests {
            logical_schema: logical.sha256().into(),
            group_state: parts.group_state.sha256().into(),
            batch_metadata: parts.metadata.segment.sha256().into(),
        },
    };
    let encoded = control::encode(runtime, &control, &operator.name, check)?;
    let mut snapshot = OperatorStateSnapshot {
        inline_metadata: control.inline_metadata(&encoded.segment),
        segments: BTreeMap::from([
            ("control".into(), encoded.segment.clone()),
            ("group-state".into(), parts.group_state),
            ("logical-schema".into(), logical),
            ("batch-metadata".into(), parts.metadata.segment.clone()),
        ]),
    };
    for segment in snapshot.segments.values_mut() {
        *segment = segment.clone().with_owner(parts.reservation.clone());
    }
    check()?;
    Ok(Arc::new(CompactCapture {
        snapshot,
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
) -> Result<(
    incremental::compact_state::NativeStateDescriptor,
    super::super::StateSegment,
)> {
    if let Some(capture) = &state.capture {
        return Ok((
            native.native_descriptor(&operator.name)?,
            capture.snapshot.segments["group-state"].clone(),
        ));
    }
    let export = native.export_native_state(&operator.name, check)?;
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
    let segment = encoded.segment.clone();
    drop(wire);
    drop(shell);
    Ok((export.into_descriptor(), segment))
}

fn capture_credit(operator: &SqlOperator) -> Result<MemoryReservation> {
    let reservation = operator
        .retention_runtime()?
        .incremental_reservation(&operator.name);
    incremental::ensure_reservation(&reservation, 16384, &operator.name)?;
    Ok(reservation)
}

#[cfg(test)]
mod current_safety_tests;
