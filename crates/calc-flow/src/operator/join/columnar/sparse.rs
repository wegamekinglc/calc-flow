use super::{CopySelection, PayloadChunk, Quantum, RowPayload, SelectedRow};
use crate::operator::join::{RetainedRows, StreamJoinOperator};
use crate::{Result, StreamOperatorContext, time::EventTime};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[cfg(test)]
mod tests;

pub(super) struct ChunkRow {
    pub row_id: u64,
    pub time: EventTime,
    pub bytes: usize,
    pub dense: AtomicUsize,
}

#[derive(Default)]
pub(in crate::operator::join) struct SparseQueue {
    head: Option<Arc<PayloadChunk>>,
    tail: Option<Arc<PayloadChunk>>,
    count: usize,
}

impl SparseQueue {
    fn push(&mut self, chunk: Arc<PayloadChunk>) {
        if chunk.queued.swap(true, Ordering::Relaxed) {
            return;
        }
        if let Some(tail) = &self.tail {
            *tail.next.lock() = Some(Arc::clone(&chunk));
        } else {
            self.head = Some(Arc::clone(&chunk));
        }
        self.tail = Some(chunk);
        self.count += 1;
    }

    fn pop(&mut self) -> Option<Arc<PayloadChunk>> {
        let chunk = self.head.take()?;
        self.head = chunk.next.lock().take();
        if self.head.is_none() {
            self.tail = None;
        }
        chunk.queued.store(false, Ordering::Relaxed);
        self.count -= 1;
        Some(chunk)
    }

    pub(in crate::operator::join) fn clear(&mut self) {
        while self.pop().is_some() {}
    }

    pub(in crate::operator::join) fn remove(&mut self, payload: &RowPayload) {
        if let RowPayload::Shared { chunk, row } = payload {
            chunk.remove_live(*row);
            self.push(Arc::clone(chunk));
        }
    }

    pub(in crate::operator::join) fn enqueue_if_due(&mut self, payload: &RowPayload) {
        if let RowPayload::Shared { chunk, .. } = payload
            && chunk.replacement_due()
        {
            self.push(Arc::clone(chunk));
        }
    }
}

impl Drop for SparseQueue {
    fn drop(&mut self) {
        self.clear();
    }
}

impl PayloadChunk {
    fn add_live(&self, offset: usize, dense: usize) {
        let row = &self.inventory[offset];
        if row.dense.swap(dense, Ordering::Relaxed) == usize::MAX {
            self.live_count.fetch_add(1, Ordering::Relaxed);
            self.live_bytes.fetch_add(row.bytes, Ordering::Relaxed);
        }
    }

    fn remove_live(&self, offset: usize) {
        let row = &self.inventory[offset];
        if row.dense.swap(usize::MAX, Ordering::Relaxed) != usize::MAX {
            self.live_count.fetch_sub(1, Ordering::Relaxed);
            self.live_bytes.fetch_sub(row.bytes, Ordering::Relaxed);
        }
    }

    fn replacement_due(&self) -> bool {
        let live = self
            .live_bytes
            .load(Ordering::Relaxed)
            .saturating_add(self.terminal_offsets);
        self.live_count.load(Ordering::Relaxed) != 0
            && self.backing_bytes > 4_096
            && self.backing_bytes > live.saturating_mul(4)
    }
}

impl RowPayload {
    pub(in crate::operator::join) fn mark_live(&self, dense: usize) {
        if let Self::Shared { chunk, row } = self {
            chunk.add_live(*row, dense);
        }
    }

    pub(in crate::operator::join) fn mark_live_at(&self, offset: usize, dense: usize) {
        if let Self::Shared { chunk, .. } = self {
            chunk.add_live(offset, dense);
        }
    }
    fn belongs_to(&self, chunk: &Arc<PayloadChunk>, offset: usize) -> bool {
        matches!(self, Self::Shared { chunk: current, row } if *row == offset && Arc::ptr_eq(current, chunk))
    }
}

impl StreamJoinOperator {
    pub(in crate::operator::join) fn has_sparse_candidates(&self) -> bool {
        self.state.left.2.count != 0 || self.state.right.2.count != 0
    }

    fn sparse_rows(&self, port: usize) -> &RetainedRows {
        if port == 0 {
            &self.state.left
        } else {
            &self.state.right
        }
    }

    fn sparse_rows_mut(&mut self, port: usize) -> &mut RetainedRows {
        if port == 0 {
            &mut self.state.left
        } else {
            &mut self.state.right
        }
    }

    fn sparse_position(
        &self,
        port: usize,
        chunk: &Arc<PayloadChunk>,
        source: usize,
    ) -> Option<usize> {
        let entry = chunk.inventory.get(source)?;
        let dense = entry.dense.load(Ordering::Relaxed);
        let retained = self.sparse_rows(port).get(dense)?;
        (retained.row_id == entry.row_id
            && retained.event_time == entry.time
            && retained.record.belongs_to(chunk, source))
        .then_some(dense)
    }

    async fn sparse_selection(
        &mut self,
        port: usize,
        chunk: &Arc<PayloadChunk>,
        context: &StreamOperatorContext<'_>,
        quantum: &mut Quantum,
    ) -> Result<Option<CopySelection>> {
        let Some(mut selected) =
            CopySelection::reserve(self, chunk.live_count.load(Ordering::Relaxed))?
        else {
            return Ok(None);
        };
        for (source, row) in chunk.inventory.iter().enumerate() {
            quantum.step(context, 8, 0).await?;
            if self.sparse_position(port, chunk, source).is_some() {
                selected.rows.push(SelectedRow {
                    source,
                    row_id: row.row_id,
                    time: row.time,
                });
            }
        }
        Ok(Some(selected))
    }

    fn install_sparse_chunk(
        &mut self,
        port: usize,
        original: &Arc<PayloadChunk>,
        replacement: &Arc<PayloadChunk>,
        selection: &[SelectedRow],
    ) {
        for (offset, selected) in selection.iter().enumerate() {
            let index = self
                .sparse_position(port, original, selected.source)
                .expect("owned sparse selection remains live until installation");
            let payload = RowPayload::Shared {
                chunk: Arc::clone(replacement),
                row: offset,
            };
            payload.mark_live(index);
            self.sparse_rows_mut(port)
                .replace_payload(index, payload.clone());
            // The derived capture reads the live state at encode time, so
            // the repaired payload binds automatically (spec FR3).
            original.remove_live(selected.source);
        }
    }

    async fn replace_sparse_chunk(
        &mut self,
        port: usize,
        chunk: &Arc<PayloadChunk>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<bool> {
        if !chunk.replacement_due() {
            return Ok(true);
        }
        let mut quantum = Quantum::default();
        let Some(selected) = self
            .sparse_selection(port, chunk, context, &mut quantum)
            .await?
        else {
            return Ok(false);
        };
        let Some(replacement) = self
            .owned_payload_columns(
                &chunk.columns,
                port,
                (&selected.rows).into(),
                context,
                &mut quantum,
            )
            .await?
        else {
            return Ok(false);
        };
        context.check_cancelled()?;
        self.install_sparse_chunk(port, chunk, &replacement, &selected.rows);
        Ok(true)
    }

    async fn repair_sparse_side(
        &mut self,
        port: usize,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let attempts = self.sparse_rows(port).2.count;
        for _ in 0..attempts {
            let chunk = Arc::clone(
                self.sparse_rows(port)
                    .2
                    .head
                    .as_ref()
                    .expect("queued candidate"),
            );
            let finished = self.replace_sparse_chunk(port, &chunk, context).await?;
            self.sparse_rows_mut(port).2.pop();
            if !finished {
                self.sparse_rows_mut(port).2.push(chunk);
            }
        }
        Ok(())
    }

    pub(in crate::operator::join) async fn repair_sparse_chunks(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        self.repair_sparse_side(0, context).await?;
        self.repair_sparse_side(1, context).await?;
        self.ingress_progress = context.ingress_progress().clone();
        Ok(())
    }
}
