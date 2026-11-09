use std::sync::Arc;

use tokio::sync::{MutexGuard, OwnedMutexGuard};

use super::{ManifestTransaction, SessionSegments, owner_settled};
use crate::{
    CalcFlowError, CancellationToken, CheckpointManifest, Result, StateHandle,
    array::{NativeSequenceFailure, NativeSequenceInventory},
    state::LocalStateLineageBackend,
};

mod stage;

#[derive(Debug)]
pub(crate) enum NativeRecoveryError {
    Legacy(CalcFlowError),
    Native(NativeSequenceFailure),
}
impl From<CalcFlowError> for NativeRecoveryError {
    fn from(error: CalcFlowError) -> Self {
        Self::Legacy(error)
    }
}
impl NativeRecoveryError {
    pub(crate) fn into_error(self) -> CalcFlowError {
        match self {
            Self::Legacy(error) => error,
            Self::Native(failure) => failure.into_parts().1,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Validation {
    Recovery,
    Maintenance,
}
pub(super) struct CandidateRead<'a, 'g> {
    pub(super) name: &'a str,
    pub(super) session: Option<&'a parking_lot::Mutex<SessionSegments>>,
    pub(super) cancellation: &'a CancellationToken,
    pub(super) operation: &'a Operation<'g>,
    pub(super) validation: Validation,
}

pub(super) struct NativePolicy {
    pub(super) inventory: Arc<NativeSequenceInventory>,
    pub(super) local: Arc<LocalStateLineageBackend>,
}
pub(super) enum Operation<'a> {
    Legacy { _guard: MutexGuard<'a, ()> },
    Native { guard: Arc<OwnedMutexGuard<()>> },
}
impl Operation<'_> {
    pub(super) fn shared(&self) -> Arc<OwnedMutexGuard<()>> {
        match self {
            Self::Native { guard } => Arc::clone(guard),
            Self::Legacy { .. } => unreachable!("native I/O requires owned operation guard"),
        }
    }
}
impl NativePolicy {
    pub(super) fn check(&self, cancellation: &CancellationToken) -> Result<()> {
        self.inventory.check()?;
        if cancellation.is_cancelled() {
            return Err(self.inventory.cancelled());
        }
        Ok(())
    }
    pub(super) async fn verify_manifest(
        &self,
        manifest: &CheckpointManifest,
        guard: &Operation<'_>,
        session: Option<&parking_lot::Mutex<SessionSegments>>,
        cancellation: &CancellationToken,
    ) -> std::result::Result<(), NativeRecoveryError> {
        for (node, entry) in manifest.operators() {
            if let Some(record) = self.inventory.record(node) {
                self.verify_entry(record, entry, guard, session, cancellation)
                    .await
                    .map_err(|error| NativeRecoveryError::Native(record.failure(error)))?;
            }
        }
        Ok(())
    }
    async fn verify_entry(
        &self,
        record: &crate::array::NativeCheckpointRecord,
        entry: &crate::OperatorManifestEntry,
        guard: &Operation<'_>,
        session: Option<&parking_lot::Mutex<SessionSegments>>,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        record.admit_retained(entry)?;
        for handle in record.handles(entry)? {
            if !session.is_some_and(|state| state.lock().verified.contains(handle)) {
                let verified: crate::array::NativeIoCompletion = record
                    .verify(
                        Arc::clone(&self.local),
                        guard.shared(),
                        handle,
                        cancellation,
                    )
                    .await?;
                if let Some(session) = session {
                    session.lock().verified.insert(verified.handle().clone());
                }
            }
        }
        Ok(())
    }
}
impl ManifestTransaction {
    pub(crate) fn native_reader(
        &self,
        node: &str,
        entry: &crate::OperatorManifestEntry,
    ) -> Result<(
        Arc<LocalStateLineageBackend>,
        crate::state::NativeSegmentReader,
    )> {
        let policy = self
            .native
            .as_ref()
            .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                message: "native:array_sequence@1 state.reader: requires trusted admission".into(),
            })?;
        let record =
            policy
                .inventory
                .record(node)
                .ok_or_else(|| {
                    CalcFlowError::CheckpointMismatch {
                message:
                    "native:array_sequence@1 state.reader: operator is absent from native admission"
                        .into(),
            }
                })?;
        record.admit_entry(entry)?;
        let handle =
            record
                .handles(entry)?
                .first()
                .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                    message: format!(
                        "operator {node:?} state.reader: empty snapshot has no segment to read"
                    ),
                })?;
        let reader = policy.local.native_reader(handle)?;
        Ok((Arc::clone(&policy.local), reader))
    }

    pub(crate) fn native_mark_verified(&self, handle: &StateHandle) {
        self.session_segments.lock().verified.insert(handle.clone());
    }

    #[cfg(test)]
    pub(crate) fn native_is_verified(&self, handle: &StateHandle) -> bool {
        self.session_segments.lock().verified.contains(handle)
    }

    pub(crate) fn with_native_sequences(
        mut self,
        inventory: Arc<NativeSequenceInventory>,
        local: Arc<LocalStateLineageBackend>,
    ) -> std::result::Result<Self, NativeSequenceFailure> {
        if Arc::as_ptr(&self.lineage).cast::<()>() != Arc::as_ptr(&local).cast::<()>() {
            return Err(inventory.first_failure(
                inventory.mismatch("reader", "reader must own the actual transaction lineage"),
            ));
        }
        if !local.supports_prepaid_load() {
            return Err(inventory
                .first_failure(inventory.mismatch("reader", "requires a certified local reader")));
        }
        self.native = Some(Arc::new(NativePolicy { inventory, local }));
        Ok(self)
    }
    pub(super) async fn operation(
        &self,
        cancellation: &CancellationToken,
        name: &str,
    ) -> Result<Operation<'_>> {
        if let Some(policy) = &self.native {
            let guard = self
                .native_load_guard(|| policy.check(cancellation))
                .await?;
            return Ok(Operation::Native {
                guard: Arc::new(guard),
            });
        }
        let guard = owner_settled(cancellation, name, async {
            Ok(self.operation.lock().await)
        })
        .await?;
        Ok(Operation::Legacy { _guard: guard })
    }
    pub(super) fn verified_manifest_handles(
        &self,
        manifest: &CheckpointManifest,
    ) -> Vec<StateHandle> {
        let mut handles = Vec::new();
        for source in manifest.sources().values() {
            if let Some(history) = &source.history {
                handles.extend(history.segments.iter().cloned());
            }
        }
        for (node, entry) in manifest.operators() {
            if !self
                .native
                .as_ref()
                .is_some_and(|policy| policy.inventory.contains(node))
            {
                handles.extend(entry.segments.iter().cloned());
            }
        }
        for sink in manifest.sinks().values() {
            handles.extend(sink.segments.iter().cloned());
        }
        handles
    }
    pub(super) async fn validate_candidate_segments(
        &self,
        manifest: &CheckpointManifest,
        read: &CandidateRead<'_, '_>,
    ) -> std::result::Result<(), NativeRecoveryError> {
        let inventory = self.native.as_ref().map(|policy| policy.inventory.as_ref());
        owner_settled(
            read.cancellation,
            &format!("{}-segments", read.name),
            super::validate_manifest_segments_scoped(
                self.lineage.as_ref(),
                manifest,
                read.session,
                inventory,
            ),
        )
        .await?;
        if matches!(read.validation, Validation::Maintenance)
            && let Some(policy) = &self.native
        {
            policy
                .verify_manifest(manifest, read.operation, read.session, read.cancellation)
                .await?;
        }
        Ok(())
    }
    pub(super) fn reject_native_raw_prepared(
        &self,
        segments: &std::collections::BTreeMap<StateHandle, Vec<u8>>,
    ) -> Result<()> {
        for handle in segments.keys() {
            if self
                .native
                .as_ref()
                .is_some_and(|policy| policy.inventory.contains(handle.operator_id()))
            {
                return Err(CalcFlowError::CheckpointMismatch {
                    message: format!(
                        "native:array_sequence@1 operator {:?} state.segments: requires captured immutable segment ownership",
                        handle.operator_id()
                    ),
                });
            }
        }
        Ok(())
    }
    pub(super) fn operation_failure(&self, error: CalcFlowError) -> NativeRecoveryError {
        match &self.native {
            Some(policy) => NativeRecoveryError::Native(policy.inventory.first_failure(error)),
            None => NativeRecoveryError::Legacy(error),
        }
    }
    pub(super) fn native_admission(
        &self,
        manifest: &CheckpointManifest,
        validation: Validation,
    ) -> std::result::Result<(), NativeRecoveryError> {
        if let Some(policy) = &self.native {
            match validation {
                Validation::Recovery => policy.inventory.admit_manifest(manifest),
                Validation::Maintenance => policy.inventory.admit_retained_manifest(manifest),
            }
            .map_err(NativeRecoveryError::Native)?;
        }
        Ok(())
    }
}
