use super::{CopySelection, PayloadChunk, Quantum, RowPayload, SelectedRow};
use crate::operator::join::{JoinSide, PendingOp, RetainedRows, StreamJoinOperator};
use crate::{Result, StreamOperatorContext, time::EventTime};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub(super) struct ChunkRow {
    pub row_id: u64,
    pub time: EventTime,
    pub bytes: usize,
    pub live: AtomicBool,
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
    fn add_live(&self, offset: usize) {
        let row = &self.inventory[offset];
        if !row.live.swap(true, Ordering::Relaxed) {
            self.live_count.fetch_add(1, Ordering::Relaxed);
            self.live_bytes.fetch_add(row.bytes, Ordering::Relaxed);
        }
    }

    fn remove_live(&self, offset: usize) {
        let row = &self.inventory[offset];
        if row.live.swap(false, Ordering::Relaxed) {
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
    pub(in crate::operator::join) fn mark_live(&self) {
        if let Self::Shared { chunk, row } = self {
            chunk.add_live(*row);
        }
    }

    fn belongs_to(&self, chunk: &Arc<PayloadChunk>) -> bool {
        matches!(self, Self::Shared { chunk: current, .. } if Arc::ptr_eq(current, chunk))
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

    fn sparse_position(&self, port: usize, time: EventTime, id: u64) -> Option<usize> {
        let expirations = if port == 0 {
            &self.state.left_expirations
        } else {
            &self.state.right_expirations
        };
        expirations
            .entries
            .get(&(time, id))
            .map(|(index, _)| *index)
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
            if row.live.load(Ordering::Relaxed)
                && self
                    .sparse_position(port, row.time, row.row_id)
                    .is_some_and(|index| self.sparse_rows(port)[index].record.belongs_to(chunk))
            {
                selected.rows.push(SelectedRow {
                    source,
                    row_id: row.row_id,
                    time: row.time,
                    retain: true,
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
        let side = if port == 0 {
            JoinSide::Left
        } else {
            JoinSide::Right
        };
        for (offset, selected) in selection.iter().enumerate() {
            let index = self
                .sparse_position(port, selected.time, selected.row_id)
                .expect("owned sparse selection remains live until installation");
            let payload = RowPayload::Shared {
                chunk: Arc::clone(replacement),
                row: offset,
            };
            payload.mark_live();
            self.sparse_rows_mut(port)[index].record = payload.clone();
            self.state
                .deltas
                .pending
                .rebind_payload(side, selected.row_id, payload);
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
            .owned_payload_columns(&chunk.columns, port, &selected.rows, context, &mut quantum)
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

impl crate::operator::join::PendingLog {
    fn rebind_payload(&mut self, side: JoinSide, id: u64, payload: RowPayload) {
        if let Some(slot) = self.upserts.get(&(side, id))
            && let Some(entry) = &mut self.slots[*slot]
            && let PendingOp::Upsert { record, .. } = &mut entry.op
        {
            *record = payload;
        }
    }
}
