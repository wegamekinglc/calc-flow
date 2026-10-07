use std::{
    collections::BTreeMap,
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use datafusion::execution::memory_pool::MemoryReservation;

use super::LoadOwner;
use crate::{
    CalcFlowError, CancellationToken, OperatorManifestEntry, OperatorStateSnapshot, Result,
    StreamJoinOperator, state::ManifestTransaction,
};

mod inventory;

pub(super) async fn load_snapshot(
    transaction: &Arc<ManifestTransaction>,
    loads: &LoadOwner,
    operator: &StreamJoinOperator,
    operator_id: &str,
    entry: &OperatorManifestEntry,
    cancellation: &CancellationToken,
    reader: &Arc<crate::state::LocalStateLineageBackend>,
) -> Result<OperatorStateSnapshot> {
    validate_handles(operator_id, entry)?;
    if entry.segments.is_empty() {
        return transaction
            .load_operator_state_cancellable(operator_id, entry, cancellation)
            .await;
    }
    let bytes = inventory::request_bytes(operator_id, entry)
        .and_then(|bytes| {
            entry.segments.iter().try_fold(bytes, |bytes, handle| {
                bytes.checked_add(reader.prepaid_load_controls(handle)?)
            })
        })
        .ok_or_else(oversized)?;
    let owner = Arc::new(operator.reserve_checkpoint_preload(bytes)?);
    let request = Request {
        transaction: transaction.clone(),
        reader: reader.clone(),
        operator_id: operator_id.into(),
        entry: OperatorManifestEntry {
            progress: BTreeMap::new(),
            inline_metadata: entry.inline_metadata.clone(),
            segments: entry.segments.clone(),
        },
        cancellation: cancellation.clone(),
    };
    let future = request.load(owner.clone());
    loads.load(FundedLoad::new(future, owner)?).await
}

fn validate_handles(operator_id: &str, entry: &OperatorManifestEntry) -> Result<()> {
    for handle in &entry.segments {
        handle.validate_owner(operator_id)?;
    }
    for (position, handle) in entry.segments.iter().enumerate() {
        if entry.segments[..position]
            .iter()
            .any(|previous| previous.segment_id() == handle.segment_id())
        {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "operator {operator_id:?} repeats state segment {:?}",
                    handle.segment_id()
                ),
            });
        }
    }
    Ok(())
}

fn oversized() -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: "Join checkpoint preload inventory exceeds addressable memory".into(),
    }
}

struct Request {
    transaction: Arc<ManifestTransaction>,
    reader: Arc<crate::state::LocalStateLineageBackend>,
    operator_id: String,
    entry: OperatorManifestEntry,
    cancellation: CancellationToken,
}

impl Request {
    async fn load(self, owner: Arc<MemoryReservation>) -> Result<OperatorStateSnapshot> {
        self.transaction
            .load_operator_state_prepaid_local(
                &self.operator_id,
                &self.entry,
                &self.cancellation,
                &self.reader,
                owner,
            )
            .await
    }
}

// The owned loading future and all partial data are destroyed before its last credit.
struct FundedLoad<F> {
    future: Pin<Box<F>>,
    _credit: Arc<MemoryReservation>,
}

impl<F: Future<Output = Result<OperatorStateSnapshot>>> FundedLoad<F> {
    fn new(future: F, credit: Arc<MemoryReservation>) -> Result<Self> {
        let bytes = inventory::future_bytes::<F>();
        let result = bytes.ok_or_else(oversized).and_then(|bytes| {
            credit
                .try_grow(bytes)
                .map_err(|error| CalcFlowError::DataFusion {
                    node_id: Some("stream-join-preload".into()),
                    message: error.to_string(),
                })
        });
        if let Err(error) = result {
            drop(future);
            drop(credit);
            return Err(error);
        }
        Ok(Self {
            future: Box::pin(future),
            _credit: credit,
        })
    }
}

impl<F: Future<Output = Result<OperatorStateSnapshot>>> Future for FundedLoad<F> {
    type Output = Result<OperatorStateSnapshot>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut().future.as_mut().poll(context)
    }
}

#[cfg(test)]
mod ownership_tests;
