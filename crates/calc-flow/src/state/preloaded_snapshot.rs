//! Private recovery ownership: cloned snapshots share their admission receipt.

use std::{
    any::Any,
    mem::{align_of, size_of},
    ops::Deref,
    sync::Arc,
};

use crate::{CalcFlowError, OperatorStateSnapshot, Result};

pub(crate) struct PreloadedSnapshot {
    snapshot: OperatorStateSnapshot,
    receipt: Option<Arc<dyn Any + Send + Sync>>,
}

impl PreloadedSnapshot {
    pub(crate) fn new(
        snapshot: OperatorStateSnapshot,
        receipt: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Self {
        Self { snapshot, receipt }
    }

    pub(crate) fn uncredited(snapshot: OperatorStateSnapshot) -> Self {
        Self::new(snapshot, None)
    }

    pub(crate) fn allocation_bytes() -> u64 {
        let bytes = size_of::<Self>() + 2 * size_of::<usize>() + align_of::<Self>() - 1;
        u64::try_from(bytes).expect("fixed snapshot descriptor fits u64")
    }

    pub(crate) fn map(
        self,
        transform: impl FnOnce(OperatorStateSnapshot) -> Result<OperatorStateSnapshot>,
    ) -> Result<Self> {
        let Self { snapshot, receipt } = self;
        let snapshot = transform(snapshot)?;
        Ok(Self { snapshot, receipt })
    }

    pub(crate) fn snapshot_mut(&mut self) -> &mut OperatorStateSnapshot {
        &mut self.snapshot
    }

    #[cfg(test)]
    pub(crate) fn clone_legacy(&self) -> Result<OperatorStateSnapshot> {
        self.require_uncredited()?;
        Ok(self.snapshot.clone())
    }

    pub(crate) fn take_legacy(snapshot: &mut Arc<Self>) -> Result<OperatorStateSnapshot> {
        Ok(std::mem::take(Self::legacy_mut(snapshot)?))
    }

    pub(crate) fn legacy_mut(snapshot: &mut Arc<Self>) -> Result<&mut OperatorStateSnapshot> {
        snapshot.require_uncredited()?;
        if Arc::get_mut(snapshot).is_none() {
            *snapshot = Arc::new(Self::uncredited(snapshot.snapshot.clone()));
        }
        Ok(&mut Arc::get_mut(snapshot)
            .expect("unique legacy snapshot")
            .snapshot)
    }

    fn require_uncredited(&self) -> Result<()> {
        if self.receipt.is_some() {
            return Err(CalcFlowError::Internal {
                message: "credited native recovery snapshots must remain shared".into(),
            });
        }
        Ok(())
    }
}

impl std::fmt::Debug for PreloadedSnapshot {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreloadedSnapshot")
            .field("segments", &self.snapshot.segments.len())
            .field("credited", &self.receipt.is_some())
            .finish_non_exhaustive()
    }
}

impl Deref for PreloadedSnapshot {
    type Target = OperatorStateSnapshot;

    fn deref(&self) -> &Self::Target {
        &self.snapshot
    }
}

#[cfg(test)]
#[path = "preloaded_snapshot/tests.rs"]
mod tests;
