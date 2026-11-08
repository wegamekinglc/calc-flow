use std::{collections::BTreeMap, sync::Arc};

use tokio::sync::{OwnedMutexGuard, OwnedSemaphorePermit};

use super::{
    NativeStreamOperator,
    inventory::{self, NativeSequenceFailure, NativeSequenceInventory},
    options::Options,
};
use crate::{
    OperatorManifestEntry, OperatorStateSnapshot, Result, StateHandle, StateSegment,
    StreamJobContext,
    array::{
        provider::worker::Scope,
        sequence::{capture::Preload, codec},
    },
    pipeline::CompiledStreamOperator,
    state::{LocalStateLineageBackend, ManifestTransaction, NativeSegmentReader},
};

struct Leases {
    transaction: Arc<ManifestTransaction>,
    _operation: OwnedMutexGuard<()>,
    _publication: OwnedMutexGuard<()>,
    _pending: OwnedSemaphorePermit,
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
                _pending: pending,
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
    ) -> std::result::Result<OperatorStateSnapshot, NativeSequenceFailure> {
        let record = self
            .record(node)
            .expect("native recovery node was admitted");
        let load = async {
            record.admit_entry(entry)?;
            let operator = inventory::compiled(operator).ok_or_else(|| {
                codec::mismatch("identity", "requires actual native sequence operator")
            })?;
            let complete = operator.read(transaction, node, entry, job).await?;
            job.check_cancelled()?;
            verified(&complete, record.handle(entry)?);
            let Complete { segment, leases } = complete;
            drop(leases);
            Ok(OperatorStateSnapshot {
                inline_metadata: entry.inline_metadata.clone(),
                segments: BTreeMap::from([("state".into(), segment)]),
            })
        };
        load.await.map_err(|error| record.failure(error))
    }
}
fn verified(complete: &Complete, handle: &StateHandle) {
    complete.leases.transaction.native_mark_verified(handle);
}

#[cfg(test)]
#[path = "preload_tests.rs"]
mod tests;
