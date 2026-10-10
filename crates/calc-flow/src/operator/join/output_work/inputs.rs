use super::super::{
    columnar::{Quantum, RowPayload},
    materialization::JoinOutput,
};
use crate::{Result, StreamOperatorContext};
use datafusion::execution::memory_pool::MemoryReservation;
use std::{ops::Range, sync::Arc};

#[derive(Clone, Copy)]
pub(super) struct Selection {
    pub(super) left: u64,
    pub(super) right: u64,
}

pub(super) struct Inputs {
    pub(super) left: Option<RowPayload>,
    pub(super) right: Option<RowPayload>,
    pub(super) rows: Vec<Selection>,
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
            rows: Vec::with_capacity(range.len()),
            credit: Some(credit),
            released: Some(released),
        };
        let mut quantum = Quantum::default();
        for pair in &materializer.matched[range] {
            quantum.step(context, 1, 16).await?;
            let (left, right) = materializer.payloads(pair);
            inputs.rows.push(Selection {
                left: u64::try_from(left.offset()).expect("Arrow row offset fits u64"),
                right: u64::try_from(right.offset()).expect("Arrow row offset fits u64"),
            });
        }
        Ok(inputs)
    }
}

impl Drop for Inputs {
    fn drop(&mut self) {
        drop(self.left.take());
        drop(self.right.take());
        drop(std::mem::take(&mut self.rows));
        if let Some(released) = self.released.take() {
            let _ = released.send(());
        }
        drop(self.credit.take());
    }
}
