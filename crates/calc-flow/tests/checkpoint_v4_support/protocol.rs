use super::table::{
    FakePersistence, FakeTable, PartitionImage, PersistedTable, PersistenceFailure,
    PersistenceFault, TableFreeze,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fault {
    Freeze,
    Write,
    Checksum,
    DiskFullState,
    DiskFullManifestAbsent,
    ManifestUncertain,
    ManifestInstalled,
    WorkerPanic,
    SinkCommit,
    SourceAck,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FailureKind {
    Freeze,
    Write,
    Checksum,
    DiskFull,
    WorkerPanic,
    Conflict,
    Cancelled,
    ProcessExit,
    Stopped,
    SinkCommit,
    SourceAck,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Recovery {
    PreviousManifest,
    RecoveryRequired,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SinkDecision {
    #[default]
    None,
    Prepared,
    Abort,
    Preserve,
    Committed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Event {
    SourceCut(u64),
    CaptureAck(u64),
    Barrier(u64),
    TerminalCapture(u64),
    Data(u64),
    Persisted(u64),
    SinkPrecommitted(u64),
    ManifestDurable(u64),
    ManifestInstalled(u64),
    SinkCommitted(u64),
    SourceAcked(u64),
    Completed(u64),
    Failed(FailureKind),
    SinkAbort,
    SinkPreserve,
    Stopped,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProtocolError {
    Backpressure(u64),
    Prerequisite(&'static str),
    Epoch,
    Conflict,
    Failed(FailureKind),
}

#[allow(
    clippy::struct_excessive_bools,
    reason = "the harness independently tracks the four protocol publication phases"
)]
#[derive(Debug)]
struct ActiveEpoch {
    freeze: TableFreeze,
    persisted: Option<PersistedTable>,
    fault: Option<Fault>,
    terminal: bool,
    sink_precommitted: bool,
    manifest_durable: bool,
    sink_committed: bool,
}

#[derive(Clone, Copy)]
enum Publication {
    Absent,
    Uncertain,
    Installed,
}

#[derive(Debug)]
pub struct ProtocolHarness {
    table: FakeTable,
    persistence: FakePersistence,
    active: Option<ActiveEpoch>,
    events: Vec<Event>,
    source_acks: Vec<u64>,
    last_completed: Option<u64>,
    manifest_epoch: Option<u64>,
    recovery: Option<Recovery>,
    sink_decision: SinkDecision,
    reservation_bytes: u64,
    workers_joined: bool,
    lease_released: bool,
    stopped: bool,
}

impl Default for ProtocolHarness {
    fn default() -> Self {
        Self {
            table: FakeTable::seed(true, &[]),
            persistence: FakePersistence::default(),
            active: None,
            events: Vec::new(),
            source_acks: Vec::new(),
            last_completed: None,
            manifest_epoch: None,
            recovery: None,
            sink_decision: SinkDecision::None,
            reservation_bytes: 0,
            workers_joined: false,
            lease_released: false,
            stopped: false,
        }
    }
}

impl ProtocolHarness {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn request(&mut self, epoch: u64) -> Result<(), ProtocolError> {
        self.begin(epoch, false, None)
    }

    pub fn request_with_fault(&mut self, epoch: u64, fault: Fault) -> Result<(), ProtocolError> {
        self.begin(epoch, false, Some(fault))
    }

    pub fn request_terminal(&mut self, epoch: u64) -> Result<(), ProtocolError> {
        self.begin(epoch, true, None)
    }

    fn begin(
        &mut self,
        epoch: u64,
        terminal: bool,
        fault: Option<Fault>,
    ) -> Result<(), ProtocolError> {
        if self.stopped {
            return Err(ProtocolError::Failed(FailureKind::Stopped));
        }
        if let Some(active) = &self.active {
            return Err(ProtocolError::Backpressure(active.freeze.epoch));
        }
        if epoch != self.last_completed.unwrap_or(0) + 1 {
            return Err(ProtocolError::Epoch);
        }
        self.events.push(Event::SourceCut(epoch));
        if fault == Some(Fault::Freeze) {
            self.fail(FailureKind::Freeze, Publication::Absent);
            return Err(ProtocolError::Failed(FailureKind::Freeze));
        }
        let Ok(freeze) = self.table.freeze(epoch) else {
            self.fail(FailureKind::Freeze, Publication::Absent);
            return Err(ProtocolError::Failed(FailureKind::Freeze));
        };
        self.reservation_bytes = freeze.reservation_bytes();
        self.active = Some(ActiveEpoch {
            freeze,
            persisted: None,
            fault,
            terminal,
            sink_precommitted: false,
            manifest_durable: false,
            sink_committed: false,
        });
        if terminal {
            self.events.push(Event::TerminalCapture(epoch));
        }
        self.events.push(Event::CaptureAck(epoch));
        if !terminal {
            self.events.push(Event::Barrier(epoch));
        }
        Ok(())
    }

    pub fn receive_barrier(&mut self, epoch: u64) -> Result<(), ProtocolError> {
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.freeze.epoch != epoch)
        {
            let active_epoch = self
                .active
                .as_ref()
                .map_or(epoch, |active| active.freeze.epoch);
            let publication = self.publication_for_active(active_epoch)?;
            self.fail(FailureKind::Conflict, publication);
            return Err(ProtocolError::Conflict);
        }
        if self.active.is_none() && epoch != self.last_completed.unwrap_or(0) + 1 {
            return Err(ProtocolError::Epoch);
        }
        Ok(())
    }

    pub fn append_data(
        &mut self,
        epoch: u64,
        group: u32,
        slot: u32,
        bytes: &[u8],
    ) -> Result<(), ProtocolError> {
        let active = self.active_epoch(epoch)?;
        if active.terminal {
            return Err(ProtocolError::Prerequisite("terminal ingress is ended"));
        }
        self.table.begin();
        self.table.append(group, slot, bytes);
        self.table.commit();
        self.events.push(Event::Data(epoch));
        Ok(())
    }

    pub fn precommit_sink(&mut self, epoch: u64) -> Result<(), ProtocolError> {
        let active = self.active_epoch(epoch)?;
        if active.sink_precommitted {
            return Ok(());
        }
        self.active_mut(epoch)?.sink_precommitted = true;
        self.sink_decision = SinkDecision::Prepared;
        self.events.push(Event::SinkPrecommitted(epoch));
        Ok(())
    }

    pub fn resolve_persistence(&mut self, epoch: u64) -> Result<(), ProtocolError> {
        let active = self.active_epoch(epoch)?;
        if active.persisted.is_some() {
            return Ok(());
        }
        let freeze = active.freeze.clone();
        let fault = persistence_fault(active.fault);
        match self.persistence.persist(&freeze, fault) {
            Ok(persisted) => {
                self.active_mut(epoch)?.persisted = Some(persisted);
                self.events.push(Event::Persisted(epoch));
                Ok(())
            }
            Err(error) => {
                let kind = persistence_failure(error);
                self.fail(kind, Publication::Absent);
                Err(ProtocolError::Failed(kind))
            }
        }
    }

    pub fn submit_persistence(
        &mut self,
        epoch: u64,
        evidence_digest: &str,
    ) -> Result<(), ProtocolError> {
        let Some(active) = &self.active else {
            return Err(ProtocolError::Epoch);
        };
        if active.freeze.epoch != epoch {
            let active_epoch = active.freeze.epoch;
            let publication = self.publication_for_active(active_epoch)?;
            self.fail(FailureKind::Conflict, publication);
            return Err(ProtocolError::Conflict);
        }
        let expected = active
            .persisted
            .as_ref()
            .map(PersistedTable::evidence_digest)
            .ok_or(ProtocolError::Prerequisite("persistence has not resolved"))?
            .to_owned();
        if evidence_digest != expected {
            let publication = self.publication_for_active(epoch)?;
            self.fail(FailureKind::Conflict, publication);
            return Err(ProtocolError::Conflict);
        }
        Ok(())
    }

    pub fn persistence_digest(&self, epoch: u64) -> Result<String, ProtocolError> {
        self.active_epoch(epoch)?
            .persisted
            .as_ref()
            .map(|persisted| persisted.evidence_digest().to_owned())
            .ok_or(ProtocolError::Prerequisite("persistence has not resolved"))
    }

    pub fn publish_manifest(&mut self, epoch: u64) -> Result<(), ProtocolError> {
        let active = self.active_epoch(epoch)?;
        if active.persisted.is_none() {
            return Err(ProtocolError::Prerequisite("persistence has not resolved"));
        }
        if !active.sink_precommitted {
            return Err(ProtocolError::Prerequisite("sink has not pre-committed"));
        }
        if active.manifest_durable {
            return Ok(());
        }
        match active.fault {
            Some(Fault::DiskFullManifestAbsent) => {
                self.fail(FailureKind::DiskFull, Publication::Absent);
                Err(ProtocolError::Failed(FailureKind::DiskFull))
            }
            Some(Fault::ManifestUncertain) => {
                self.fail(FailureKind::Write, Publication::Uncertain);
                Err(ProtocolError::Failed(FailureKind::Write))
            }
            Some(Fault::ManifestInstalled) => {
                self.install_manifest(epoch, false);
                self.events.push(Event::ManifestInstalled(epoch));
                self.fail(FailureKind::Write, Publication::Installed);
                Err(ProtocolError::Failed(FailureKind::Write))
            }
            _ => {
                self.install_manifest(epoch, true);
                self.events.push(Event::ManifestDurable(epoch));
                Ok(())
            }
        }
    }

    fn install_manifest(&mut self, epoch: u64, durable: bool) {
        let image = self
            .active
            .as_ref()
            .and_then(|active| active.persisted.as_ref())
            .map(|persisted| persisted.image.clone())
            .expect("persisted image exists before publication");
        self.table.publish(image);
        self.manifest_epoch = Some(epoch);
        if let Some(active) = &mut self.active {
            active.manifest_durable = durable;
        }
    }

    pub fn commit_sink(&mut self, epoch: u64) -> Result<(), ProtocolError> {
        let active = self.active_epoch(epoch)?;
        if !active.manifest_durable {
            return Err(ProtocolError::Prerequisite("manifest is not durable"));
        }
        if active.sink_committed {
            return Ok(());
        }
        if active.fault == Some(Fault::SinkCommit) {
            self.fail(FailureKind::SinkCommit, Publication::Installed);
            return Err(ProtocolError::Failed(FailureKind::SinkCommit));
        }
        self.active_mut(epoch)?.sink_committed = true;
        self.sink_decision = SinkDecision::Committed;
        self.events.push(Event::SinkCommitted(epoch));
        Ok(())
    }

    pub fn acknowledge_source(&mut self, epoch: u64) -> Result<(), ProtocolError> {
        let active = self.active_epoch(epoch)?;
        if !active.sink_committed {
            return Err(ProtocolError::Prerequisite("sink has not committed"));
        }
        if active.fault == Some(Fault::SourceAck) {
            self.fail(FailureKind::SourceAck, Publication::Installed);
            self.sink_decision = SinkDecision::Committed;
            return Err(ProtocolError::Failed(FailureKind::SourceAck));
        }
        self.source_acks.push(epoch);
        self.last_completed = Some(epoch);
        self.recovery = None;
        self.sink_decision = SinkDecision::Committed;
        self.events.push(Event::SourceAcked(epoch));
        self.events.push(Event::Completed(epoch));
        self.active = None;
        self.table.settle();
        self.reservation_bytes = 0;
        self.workers_joined = true;
        Ok(())
    }

    pub fn complete_epoch(&mut self, epoch: u64) -> Result<(), ProtocolError> {
        self.precommit_sink(epoch)?;
        self.resolve_persistence(epoch)?;
        self.publish_manifest(epoch)?;
        self.commit_sink(epoch)?;
        self.acknowledge_source(epoch)
    }

    pub fn cancel(&mut self, epoch: u64) -> Result<(), ProtocolError> {
        let publication = self.publication_for_active(epoch)?;
        self.fail(FailureKind::Cancelled, publication);
        Err(ProtocolError::Failed(FailureKind::Cancelled))
    }

    pub fn process_exit(&mut self, epoch: u64) -> Result<(), ProtocolError> {
        let publication = self.publication_for_active(epoch)?;
        self.fail(FailureKind::ProcessExit, publication);
        Err(ProtocolError::Failed(FailureKind::ProcessExit))
    }

    fn publication_for_active(&self, epoch: u64) -> Result<Publication, ProtocolError> {
        let active = self.active_epoch(epoch)?;
        Ok(if active.manifest_durable {
            Publication::Installed
        } else {
            Publication::Absent
        })
    }

    pub fn stop(&mut self) {
        if let Some(active) = &self.active {
            let publication = if active.manifest_durable {
                Publication::Installed
            } else {
                Publication::Absent
            };
            self.fail(FailureKind::Stopped, publication);
        } else {
            self.table.settle();
            self.reservation_bytes = 0;
            self.workers_joined = true;
            self.lease_released = true;
            self.stopped = true;
        }
        self.events.push(Event::Stopped);
    }

    fn fail(&mut self, kind: FailureKind, publication: Publication) {
        let preserve = matches!(publication, Publication::Uncertain | Publication::Installed);
        self.active = None;
        self.table.settle();
        self.reservation_bytes = 0;
        self.workers_joined = true;
        self.lease_released = true;
        self.stopped = true;
        self.recovery = Some(if preserve {
            Recovery::RecoveryRequired
        } else {
            Recovery::PreviousManifest
        });
        self.sink_decision = if preserve {
            SinkDecision::Preserve
        } else {
            SinkDecision::Abort
        };
        self.events.push(Event::Failed(kind));
        self.events.push(if preserve {
            Event::SinkPreserve
        } else {
            Event::SinkAbort
        });
    }

    fn active_epoch(&self, epoch: u64) -> Result<&ActiveEpoch, ProtocolError> {
        self.active
            .as_ref()
            .filter(|active| active.freeze.epoch == epoch)
            .ok_or(ProtocolError::Epoch)
    }

    fn active_mut(&mut self, epoch: u64) -> Result<&mut ActiveEpoch, ProtocolError> {
        self.active
            .as_mut()
            .filter(|active| active.freeze.epoch == epoch)
            .ok_or(ProtocolError::Epoch)
    }

    pub fn events(&self) -> &[Event] {
        &self.events
    }

    pub fn source_acks(&self) -> &[u64] {
        &self.source_acks
    }

    pub const fn last_completed(&self) -> Option<u64> {
        self.last_completed
    }

    pub const fn manifest_epoch(&self) -> Option<u64> {
        self.manifest_epoch
    }

    pub const fn recovery(&self) -> Option<Recovery> {
        self.recovery
    }

    pub const fn sink_decision(&self) -> SinkDecision {
        self.sink_decision
    }

    pub const fn reservation_bytes(&self) -> u64 {
        self.reservation_bytes
    }

    pub const fn workers_joined(&self) -> bool {
        self.workers_joined
    }

    pub const fn lease_released(&self) -> bool {
        self.lease_released
    }

    pub const fn stopped(&self) -> bool {
        self.stopped
    }

    pub fn durable_image(&self) -> PartitionImage {
        self.table.durable_image()
    }

    pub fn table(&self) -> &FakeTable {
        &self.table
    }

    pub fn table_mut(&mut self) -> &mut FakeTable {
        &mut self.table
    }

    pub fn persistence(&self) -> &FakePersistence {
        &self.persistence
    }
}

const fn persistence_fault(fault: Option<Fault>) -> Option<PersistenceFault> {
    match fault {
        Some(Fault::Write) => Some(PersistenceFault::Write),
        Some(Fault::Checksum) => Some(PersistenceFault::Checksum),
        Some(Fault::DiskFullState) => Some(PersistenceFault::DiskFull),
        Some(Fault::WorkerPanic) => Some(PersistenceFault::WorkerPanic),
        _ => None,
    }
}

fn persistence_failure(error: PersistenceFailure) -> FailureKind {
    match error {
        PersistenceFailure::Write => FailureKind::Write,
        PersistenceFailure::Checksum => FailureKind::Checksum,
        PersistenceFailure::DiskFull => FailureKind::DiskFull,
        PersistenceFailure::WorkerPanic => FailureKind::WorkerPanic,
    }
}
