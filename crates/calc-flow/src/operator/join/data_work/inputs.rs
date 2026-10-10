use super::super::{
    AdmittedRow, JoinTimeBounds, StoredRow,
    checkpoint_v2::ContainerFunding,
    native_lookup::{NativeIndex, NativeKeys},
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::Arc;

pub(super) struct ProbeInputs {
    pub(super) admitted: Vec<AdmittedRow>,
    pub(super) keys: Option<NativeKeys>,
    pub(super) slots: Vec<Option<u32>>,
    pub(super) opposite: Option<Arc<Vec<StoredRow>>>,
    pub(super) index: Option<Arc<NativeIndex>>,
    pub(super) containers: Option<Arc<ContainerFunding>>,
    pub(super) bounds: JoinTimeBounds,
    pub(super) incoming_is_left: bool,
    pub(super) control: Option<Arc<MemoryReservation>>,
    pub(super) released: Option<tokio::sync::oneshot::Sender<()>>,
    #[cfg(test)]
    pub(super) hook: Option<super::TestHook>,
}

impl ProbeInputs {
    pub(super) fn slot(&self, position: usize) -> Option<u32> {
        self.slots[self.keys.as_ref().expect("owned probe keys").id(position) as usize]
    }

    pub(super) fn range(&self, position: usize) -> (crate::EventTime, crate::EventTime) {
        super::super::native_lookup::time_range(
            self.bounds,
            self.incoming_is_left,
            self.admitted[position].event_time,
        )
    }

    pub(super) fn count(&self, position: usize) -> usize {
        self.slot(position).map_or(0, |id| {
            self.index
                .as_ref()
                .expect("owned native index")
                .count_window_by_id(id, self.range(position))
        })
    }

    pub(super) fn matches(&self, position: usize) -> impl Iterator<Item = usize> + '_ {
        self.slot(position).into_iter().flat_map(move |id| {
            self.index
                .as_ref()
                .expect("owned native index")
                .range_by_id(id, self.range(position))
        })
    }
}

impl Drop for ProbeInputs {
    fn drop(&mut self) {
        drop(std::mem::take(&mut self.admitted));
        drop(self.keys.take());
        drop(std::mem::take(&mut self.slots));
        drop(self.opposite.take());
        drop(self.index.take());
        drop(self.containers.take());
        if let Some(released) = self.released.take() {
            let _ = released.send(());
        }
        drop(self.control.take());
    }
}
