use std::sync::Arc;

use super::super::{
    ManifestTransaction, SegmentPublication, StagedOperatorState, WorkingStatePins, digest,
};
use super::{NativePolicy, Operation};
use crate::{
    CalcFlowError, CancellationToken, Epoch, OperatorStateSnapshot, Result, StateHandle,
    StateSegment,
};

impl ManifestTransaction {
    pub(in crate::state::transaction) async fn stage_native_state(
        &self,
        policy: &NativePolicy,
        operator: &str,
        epoch: Epoch,
        snapshot: OperatorStateSnapshot,
        cancellation: &CancellationToken,
        guard: &Operation<'_>,
    ) -> Result<StagedOperatorState> {
        let record = policy
            .inventory
            .record(operator)
            .expect("native stage is selected by intrinsic inventory");
        record.admit_snapshot(&snapshot)?;
        let mut handles = Vec::with_capacity(snapshot.segments.len());
        for (segment_id, segment) in &snapshot.segments {
            let (handle, status) = self.native_handle(operator, epoch, segment_id, segment)?;
            if status != SegmentPublication::Carried {
                let _complete = record
                    .stage_publish(
                        Arc::clone(&policy.local),
                        guard.shared(),
                        &handle,
                        segment,
                        cancellation,
                    )
                    .await?;
                self.remember_native(&handle);
            }
            handles.push(handle);
        }
        #[cfg(test)]
        self.inject_fault(super::super::ManifestTransactionFaultPoint::StateStage)?;
        Ok(StagedOperatorState {
            working: WorkingStatePins::acquire(
                self.session_segments.clone(),
                self.lineage.clone(),
                &handles,
            )?,
            inline_metadata: snapshot.inline_metadata,
            segments: handles,
        })
    }
    fn native_handle(
        &self,
        operator: &str,
        epoch: Epoch,
        segment_id: &str,
        segment: &StateSegment,
    ) -> Result<(StateHandle, SegmentPublication)> {
        let length = u64::try_from(segment.bytes().len()).map_err(|_| {
            CalcFlowError::CheckpointMismatch {
                message: "native:array_sequence@1 state.bytes: encoded length exceeds u64".into(),
            }
        })?;
        let key = (operator.to_owned(), segment_id.to_owned());
        let carried = self
            .session_segments
            .lock()
            .carried
            .get(&key)
            .filter(|handle| handle.sha256() == segment.sha256() && handle.byte_len() == length)
            .cloned();
        if let Some(handle) = carried {
            return Ok((handle, SegmentPublication::Carried));
        }
        let path = format!(
            "committed/{}/{}/{}-{}.segment",
            self.lineage_hash,
            digest(operator),
            epoch.as_u64(),
            digest(segment_id)
        );
        StateHandle::new(operator, epoch, segment_id, &path, length, segment.sha256())
            .map(|handle| (handle, SegmentPublication::Staged))
    }
    fn remember_native(&self, handle: &StateHandle) {
        let mut session = self.session_segments.lock();
        session.carried.insert(
            (
                handle.operator_id().to_owned(),
                handle.segment_id().to_owned(),
            ),
            handle.clone(),
        );
        session.verified.insert(handle.clone());
    }
}
