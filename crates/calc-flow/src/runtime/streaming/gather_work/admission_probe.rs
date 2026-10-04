use super::{GatherOperatorId, JobGatherOwner, TaskId};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionStage {
    Attempt,
    Generation,
}

pub(crate) struct AdmissionEvent {
    pub(crate) stage: AdmissionStage,
    pub(crate) operator: String,
    pub(crate) task: Option<TaskId>,
    pub(crate) fee: usize,
    pub(crate) available: usize,
}

pub(crate) struct AdmissionProbe {
    stage: AdmissionStage,
    pool: Arc<dyn MemoryPool>,
    limit: usize,
    fired: AtomicBool,
    event: Mutex<Option<AdmissionEvent>>,
}

impl AdmissionProbe {
    pub(crate) fn install(
        owner: &JobGatherOwner,
        stage: AdmissionStage,
        pool: Arc<dyn MemoryPool>,
        limit: usize,
    ) -> Arc<Self> {
        let probe = Arc::new(Self {
            stage,
            pool,
            limit,
            fired: AtomicBool::new(false),
            event: Mutex::new(None),
        });
        owner.0.home.state.lock().admission_probe = Some(probe.clone());
        probe
    }
    pub(crate) fn take_event(&self) -> Option<AdmissionEvent> {
        self.event.lock().take()
    }
}

pub(super) fn before(
    probe: Option<&Arc<AdmissionProbe>>,
    stage: AdmissionStage,
    fee: usize,
    operator: &GatherOperatorId,
) -> crate::Result<Option<MemoryReservation>> {
    let Some(probe) = probe.filter(|probe| probe.stage == stage) else {
        return Ok(None);
    };
    if probe.fired.swap(true, Ordering::AcqRel) {
        return Ok(None);
    }
    let pressure = MemoryConsumer::new("real-gather-admission-pressure").register(&probe.pool);
    let headroom = probe
        .limit
        .checked_sub(probe.pool.reserved())
        .expect("fixture limit");
    let available = fee.checked_sub(1).expect("nonzero native execution fee");
    pressure
        .try_grow(
            headroom
                .checked_sub(available)
                .expect("fixture has native fee headroom"),
        )
        .map_err(|error| super::internal(&error.to_string()))?;
    *probe.event.lock() = Some(AdmissionEvent {
        stage,
        operator: operator.name.to_string(),
        task: operator.task,
        fee,
        available,
    });
    Ok(Some(pressure))
}
