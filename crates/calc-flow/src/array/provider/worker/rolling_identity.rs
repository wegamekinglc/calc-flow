//! Shared diagnostic identity keeps its admission owner through worker settlement.

use std::{
    alloc::Layout,
    any::Any,
    ops::Deref,
    sync::{Arc, atomic::AtomicUsize},
};

pub(crate) struct RollingWorkerIdentity {
    node: String,
    _lease: Arc<dyn Any + Send + Sync>,
}
impl RollingWorkerIdentity {
    pub(crate) fn allocation_bytes(node: &str) -> Option<u64> {
        let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Self>())
            .ok()?;
        let bytes = layout.pad_to_align().size().checked_add(node.len())?;
        u64::try_from(bytes).ok()
    }
    pub(crate) fn new(node: &str, lease: Arc<dyn Any + Send + Sync>) -> Arc<Self> {
        Arc::new(Self {
            node: node.into(),
            _lease: lease,
        })
    }
    pub(crate) fn as_str(&self) -> &str {
        &self.node
    }
}
impl Deref for RollingWorkerIdentity {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}
