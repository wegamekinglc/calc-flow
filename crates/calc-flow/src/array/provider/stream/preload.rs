use std::{collections::BTreeMap, sync::Arc};

use tokio::sync::{OwnedMutexGuard, OwnedSemaphorePermit};

use super::{
    NativeStreamOperator,
    inventory::{self, NativeCheckpointRecord, NativeSequenceFailure, NativeSequenceInventory},
    options::Options,
};
use crate::{
    OperatorManifestEntry, OperatorStateSnapshot, Result, RollingOperator, StateHandle,
    StateSegment, StreamJobContext,
    array::{
        provider::worker::Scope,
        sequence::{capture::Preload, codec},
    },
    pipeline::CompiledStreamOperator,
    state::{
        LocalStateLineageBackend, ManifestTransaction, NativeSegmentReader, PreloadedSnapshot,
    },
};

struct Leases {
    transaction: Arc<ManifestTransaction>,
    _operation: OwnedMutexGuard<()>,
    _publication: OwnedMutexGuard<()>,
    _pending: Option<OwnedSemaphorePermit>,
    _lineage: Arc<LocalStateLineageBackend>,
}
struct Read {
    reader: NativeSegmentReader,
    prepaid: Preload,
    configured: Arc<Options>,
    leases: Leases,
}
struct Complete {
    segment: StateSegment,
    leases: Leases,
}
impl Read {
    fn run(mut self, check: &mut dyn FnMut() -> Result<()>) -> Result<Complete> {
        let (bytes, scratch) =
            self.prepaid
                .allocate(self.reader.byte_len(), &self.configured.execution, check)?;
        let segment = self.reader.read(bytes, scratch, check)?;
        let segment = self
            .prepaid
            .finish(segment, &self.configured.execution, check)?;
        check()?;
        Ok(Complete {
            segment,
            leases: self.leases,
        })
    }
}
impl NativeStreamOperator {
    async fn read(
        &self,
        transaction: &Arc<ManifestTransaction>,
        node: &str,
        entry: &OperatorManifestEntry,
        job: &StreamJobContext,
    ) -> Result<Complete> {
        let pending = self.pending.enter(job).await?;
        let operation = transaction
            .native_load_guard(|| job.check_cancelled())
            .await?;
        let (lineage, reader) = transaction.native_reader(node, entry)?;
        let publication = lineage
            .native_publication_guard(|| job.check_cancelled())
            .await?;
        let prepaid = self.captures.preload(
            reader.byte_len(),
            self.state.state_bytes(),
            &self.configured.execution,
        )?;
        let read = Read {
            reader,
            prepaid,
            configured: Arc::clone(&self.configured),
            leases: Leases {
                transaction: Arc::clone(transaction),
                _operation: operation,
                _publication: publication,
                _pending: Some(pending),
                _lineage: lineage,
            },
        };
        self.workers
            .run_scoped(Scope::stream(job), "array_sequence", move |check| {
                read.run(check)
            })
            .await
    }
}
impl NativeSequenceInventory {
    pub(crate) async fn load_snapshot(
        &self,
        operator: &CompiledStreamOperator,
        node: &str,
        entry: &OperatorManifestEntry,
        transaction: &Arc<ManifestTransaction>,
        job: &StreamJobContext,
    ) -> std::result::Result<PreloadedSnapshot, NativeSequenceFailure> {
        let record = self
            .record(node)
            .expect("native recovery node was admitted");
        let load = async {
            record.admit_entry(entry)?;
            match record.as_ref() {
                NativeCheckpointRecord::Sequence(record) => {
                    sequence_snapshot(record, operator, node, entry, transaction, job)
                        .await
                        .map(PreloadedSnapshot::uncredited)
                }
                NativeCheckpointRecord::IndicatorRolling(record) => {
                    rolling_snapshot(record, operator, node, entry, transaction, job).await
                }
            }
        };
        load.await.map_err(|error| record.failure(error))
    }
}
async fn sequence_snapshot(
    record: &inventory::NativeSequenceRecord,
    operator: &CompiledStreamOperator,
    node: &str,
    entry: &OperatorManifestEntry,
    transaction: &Arc<ManifestTransaction>,
    job: &StreamJobContext,
) -> Result<OperatorStateSnapshot> {
    let operator = inventory::compiled(operator)
        .ok_or_else(|| codec::mismatch("identity", "requires actual native sequence operator"))?;
    let complete = operator.read(transaction, node, entry, job).await?;
    job.check_cancelled()?;
    verified(&complete, record.handle(entry)?);
    let Complete { segment, leases } = complete;
    drop(leases);
    Ok(OperatorStateSnapshot {
        inline_metadata: entry.inline_metadata.clone(),
        segments: BTreeMap::from([("state".into(), segment)]),
    })
}
async fn rolling_snapshot(
    record: &crate::operator::rolling::IndicatorRollingRecord,
    operator: &CompiledStreamOperator,
    node: &str,
    entry: &OperatorManifestEntry,
    transaction: &Arc<ManifestTransaction>,
    job: &StreamJobContext,
) -> Result<PreloadedSnapshot> {
    let operator = match operator {
        CompiledStreamOperator::Rolling(operator)
            if matches!(
                operator.state_layout_version(),
                crate::ROLLING_INDICATOR_STATE_LAYOUT_VERSION
                    | crate::ROLLING_MACD_STATE_LAYOUT_VERSION
            ) =>
        {
            operator
        }
        _ => {
            return Err(record.mismatch(
                "identity",
                "requires actual compiled rolling layout five operator",
            ));
        }
    };
    let segment = match record.handle(entry)? {
        Some(handle) => {
            let (segment, leases) = read_rolling(operator, node, entry, transaction, job).await?;
            leases.transaction.native_mark_verified(handle);
            drop(leases);
            Some((handle.segment_id(), segment))
        }
        None => None,
    };
    job.check_cancelled()?;
    record.preloaded_snapshot(&entry.inline_metadata, segment)
}
async fn read_rolling(
    operator: &RollingOperator,
    node: &str,
    entry: &OperatorManifestEntry,
    transaction: &Arc<ManifestTransaction>,
    job: &StreamJobContext,
) -> Result<(StateSegment, Leases)> {
    let pending = operator.native_checkpoint_guard(node, job).await?;
    let (reader, leases) = rolling_read_leases(transaction, node, entry, job, pending).await?;
    let complete = operator
        .read_native_segment(node, reader, leases, job)
        .await?;
    job.check_cancelled()?;
    Ok(complete)
}
async fn rolling_read_leases(
    transaction: &Arc<ManifestTransaction>,
    node: &str,
    entry: &OperatorManifestEntry,
    job: &StreamJobContext,
    pending: OwnedSemaphorePermit,
) -> Result<(NativeSegmentReader, Leases)> {
    let operation = transaction
        .native_load_guard(|| job.check_cancelled())
        .await?;
    let (lineage, reader) = transaction.native_reader(node, entry)?;
    let publication = lineage
        .native_publication_guard(|| job.check_cancelled())
        .await?;
    Ok((
        reader,
        Leases {
            transaction: Arc::clone(transaction),
            _operation: operation,
            _publication: publication,
            _pending: Some(pending),
            _lineage: lineage,
        },
    ))
}
fn verified(complete: &Complete, handle: &StateHandle) {
    complete.leases.transaction.native_mark_verified(handle);
}

#[cfg(test)]
#[path = "preload_tests.rs"]
mod tests;
