use super::super::{
    columnar::{Quantum, RowPayload},
    materialization::JoinOutput,
};
use crate::{Result, StreamOperatorContext};
use datafusion::{arrow::array::UInt64Array, execution::memory_pool::MemoryReservation};
use std::{ops::Range, sync::Arc};

pub(super) struct Inputs {
    pub(super) left: Option<RowPayload>,
    pub(super) right: Option<RowPayload>,
    pub(super) left_indices: Option<UInt64Array>,
    pub(super) right_indices: Option<UInt64Array>,
    pub(super) credit: Option<Arc<MemoryReservation>>,
    pub(super) released: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Inputs {
    pub(super) async fn capture(
        materializer: &JoinOutput<'_>,
        range: Range<usize>,
        credit: Arc<MemoryReservation>,
        released: tokio::sync::oneshot::Sender<()>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Self> {
        let first = materializer.payloads(&materializer.matched[range.start]);
        let mut inputs = Self {
            left: Some(first.0.clone()),
            right: Some(first.1.clone()),
            left_indices: None,
            right_indices: None,
            credit: Some(credit),
            released: Some(released),
        };
        let mut left_indices = Vec::with_capacity(range.len());
        let mut right_indices = Vec::with_capacity(range.len());
        let mut quantum = Quantum::default();
        for pair in &materializer.matched[range] {
            quantum.step(context, 1, 16).await?;
            let (left, right) = materializer.payloads(pair);
            left_indices.push(u64::try_from(left.offset()).expect("Arrow row offset fits u64"));
            right_indices.push(u64::try_from(right.offset()).expect("Arrow row offset fits u64"));
        }
        inputs.left_indices = Some(UInt64Array::from(left_indices));
        inputs.right_indices = Some(UInt64Array::from(right_indices));
        Ok(inputs)
    }
}

impl Drop for Inputs {
    fn drop(&mut self) {
        drop(self.left.take());
        drop(self.right.take());
        drop(self.left_indices.take());
        drop(self.right_indices.take());
        if let Some(released) = self.released.take() {
            let _ = released.send(());
        }
        drop(self.credit.take());
    }
}
