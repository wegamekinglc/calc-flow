use super::super::{
    AdmittedBatch, AdmittedRows, JoinTimeBounds,
    checkpoint_v2::ContainerFunding,
    native_lookup::{NativeIndex, NativeKeys, WindowBounds, WindowCursor},
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::Arc;

pub(super) struct ProbeInputs {
    pub(super) admitted: AdmittedBatch,
    pub(super) keys: Option<NativeKeys>,
    pub(super) slots: Vec<Option<u32>>,
    pub(super) opposite: Option<Arc<crate::operator::join::StateRows>>,
    pub(super) index: Option<Arc<NativeIndex>>,
    pub(super) containers: Option<Arc<ContainerFunding>>,
    pub(super) bounds: JoinTimeBounds,
    pub(super) incoming_is_left: bool,
    pub(super) control: Option<Arc<MemoryReservation>>,
    pub(super) released: Option<tokio::sync::oneshot::Sender<()>>,
    #[cfg(test)]
    pub(super) hook: Option<super::TestHook>,
    #[cfg(test)]
    pub(super) unit_hook: Option<super::UnitTestHook>,
}

impl ProbeInputs {
    pub(super) fn slot(&self, position: usize) -> Option<u32> {
        self.slots
            .get(self.keys.as_ref().expect("owned probe keys").id(position) as usize)
            .copied()
            .flatten()
    }

    pub(super) fn range(&self, position: usize) -> (crate::EventTime, crate::EventTime) {
        super::super::native_lookup::time_range(
            self.bounds,
            self.incoming_is_left,
            self.admitted.event_time(position),
        )
    }

    /// Unit-local window cursors, one per distinct probe key. Each gather unit
    /// owns its own set; cursors are never shared between concurrent units.
    pub(super) fn cursors(&self) -> Vec<Option<WindowCursor>> {
        vec![None; self.slots.len()]
    }

    /// Exact window bounds for one probe row, advancing the unit-local cursor
    /// so consecutive probes for a key reuse its previous position.
    pub(super) fn window(
        &self,
        position: usize,
        cursors: &mut [Option<WindowCursor>],
    ) -> WindowBounds {
        let Some(id) = self.slot(position) else {
            return WindowBounds::EMPTY;
        };
        let slot = self.keys.as_ref().expect("owned probe keys").id(position) as usize;
        let range = self.range(position);
        let bounds = self
            .index
            .as_ref()
            .expect("owned native index")
            .window_bounds_by_id(id, range, cursors[slot]);
        cursors[slot] = Some(WindowCursor::new(range, bounds));
        bounds
    }

    /// Dense opposite indices for one probe row's window, advancing the cursor.
    pub(super) fn matches_window(
        &self,
        position: usize,
        cursors: &mut [Option<WindowCursor>],
    ) -> impl Iterator<Item = usize> + '_ {
        let bounds = self.window(position, cursors);
        self.slot(position).into_iter().flat_map(move |id| {
            self.index
                .as_ref()
                .expect("owned native index")
                .range_by_bounds(id, bounds)
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
