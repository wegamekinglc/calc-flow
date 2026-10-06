use super::{SessionSegments, StateHandle, StateLineageBackend};
use crate::{CalcFlowError, Result};
use std::sync::Arc;

pub(crate) struct WorkingStatePins {
    session: Arc<parking_lot::Mutex<SessionSegments>>,
    handles: Vec<StateHandle>,
    _lineage: Arc<dyn StateLineageBackend>,
}

impl WorkingStatePins {
    pub(super) fn acquire(
        session: Arc<parking_lot::Mutex<SessionSegments>>,
        lineage: Arc<dyn StateLineageBackend>,
        handles: &[StateHandle],
    ) -> Result<Option<Arc<Self>>> {
        if handles.is_empty() {
            return Ok(None);
        }
        let mut state = session.lock();
        for handle in handles {
            if state.working.get(handle) == Some(&usize::MAX) {
                return Err(CalcFlowError::Internal {
                    message: "working state reference count overflowed".into(),
                });
            }
        }
        for handle in handles {
            *state.working.entry(handle.clone()).or_default() += 1;
        }
        drop(state);
        Ok(Some(Arc::new(Self {
            session,
            handles: handles.to_vec(),
            _lineage: lineage,
        })))
    }
}

impl Drop for WorkingStatePins {
    fn drop(&mut self) {
        let mut state = self.session.lock();
        for handle in &self.handles {
            if let Some(count) = state.working.get_mut(handle) {
                *count -= 1;
                if *count == 0 {
                    state.working.remove(handle);
                }
            }
        }
    }
}
