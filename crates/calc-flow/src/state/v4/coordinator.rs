//! Capture-freeze-forward coordination for one in-flight epoch.
//!
//! The coordinator owns ordering, not persistence. Capture and barrier delivery
//! never wait for object writes; durable publication, sink commit and source
//! acknowledgement happen only in their specified order.

use super::{OperatorV4, Result, V4Manifest, V4Store, mismatch, opaque::V4OpaqueAdapter};
use crate::state::{
    StateTable,
    table::v4::{V4PersistedTable, V4TableCapture},
};

#[allow(dead_code, reason = "P3.4 runner wiring uses these recovery states")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::state) enum V4Recovery {
    PreviousManifest,
    RecoveryRequired,
}

#[allow(
    clippy::struct_excessive_bools,
    reason = "the epoch protocol independently tracks barrier, sink, manifest and commit phases"
)]
#[derive(Debug)]
struct ActiveEpoch {
    epoch: u64,
    freeze: Option<crate::state::table::TableFreeze>,
    persisted: Option<V4PersistedTable>,
    manifest: Option<V4Manifest>,
    manifest_path: Option<std::path::PathBuf>,
    opaque: Option<V4OpaqueAdapter>,
    opaque_persisted: bool,
    terminal: bool,
    barrier_forwarded: bool,
    sink_precommitted: bool,
    manifest_durable: bool,
    sink_committed: bool,
}

impl ActiveEpoch {
    fn new(freeze: crate::state::table::TableFreeze, terminal: bool, manifest: V4Manifest) -> Self {
        Self {
            epoch: freeze.epoch(),
            freeze: Some(freeze),
            persisted: None,
            manifest: Some(manifest),
            manifest_path: None,
            opaque: None,
            opaque_persisted: false,
            terminal,
            barrier_forwarded: false,
            sink_precommitted: false,
            manifest_durable: false,
            sink_committed: false,
        }
    }
}

#[derive(Debug)]
pub(in crate::state) struct V4Coordinator {
    store: Option<V4Store>,
    active: Option<ActiveEpoch>,
    previous: Option<V4PersistedTable>,
    last_completed: Option<u64>,
    recovery: Option<V4Recovery>,
    stopped: bool,
    max_inflight_bytes: usize,
}

#[allow(
    dead_code,
    reason = "P3.4 runner wiring uses stop, recovery and active-epoch inspection"
)]
impl Default for V4Coordinator {
    fn default() -> Self {
        Self {
            store: None,
            active: None,
            previous: None,
            last_completed: None,
            recovery: None,
            stopped: false,
            max_inflight_bytes: 1 << 30,
        }
    }
}

#[allow(
    dead_code,
    reason = "P3.4 runner wiring uses stop, recovery and active-epoch inspection"
)]
impl V4Coordinator {
    pub(in crate::state) fn new(store: V4Store) -> Self {
        Self {
            store: Some(store),
            ..Self::default()
        }
    }

    pub(in crate::state) fn with_limit(store: V4Store, max_inflight_bytes: usize) -> Self {
        Self {
            store: Some(store),
            max_inflight_bytes,
            ..Self::default()
        }
    }

    pub(in crate::state) fn begin(
        &mut self,
        table: &mut StateTable,
        epoch: u64,
        terminal: bool,
        manifest: V4Manifest,
    ) -> Result<()> {
        if self.stopped {
            return Err(mismatch("checkpoint coordinator is stopped".into()));
        }
        if self.active.is_some() {
            return Err(mismatch(format!(
                "epoch {epoch} overlaps the active checkpoint"
            )));
        }
        if epoch != self.last_completed.unwrap_or(0) + 1 {
            return Err(mismatch("checkpoint epoch is not the next epoch".into()));
        }
        if manifest.epoch != epoch {
            return Err(mismatch(
                "manifest epoch differs from checkpoint epoch".into(),
            ));
        }
        let freeze = table.freeze(epoch)?;
        let reservation = freeze.reservation_bytes();
        if reservation > self.max_inflight_bytes {
            table.abort_freeze(epoch)?;
            return Err(mismatch(format!(
                "checkpoint in-flight reservation {reservation} exceeds {} bytes",
                self.max_inflight_bytes
            )));
        }
        self.active = Some(ActiveEpoch::new(freeze, terminal, manifest));
        Ok(())
    }

    pub(in crate::state) fn begin_with_opaque(
        &mut self,
        adapter: V4OpaqueAdapter,
        epoch: u64,
    ) -> Result<()> {
        self.active_mut(epoch)?.opaque = Some(adapter);
        Ok(())
    }

    pub(in crate::state) fn forward_barrier(&mut self, epoch: u64) -> Result<()> {
        let active = self.active_mut(epoch)?;
        if active.terminal {
            return Err(mismatch(
                "terminal capture has no downstream barrier".into(),
            ));
        }
        active.barrier_forwarded = true;
        Ok(())
    }

    pub(in crate::state) async fn persist(&mut self, epoch: u64) -> Result<()> {
        let Some((freeze, manifest)) = self.take_capture(epoch)? else {
            return Ok(());
        };
        self.persist_table(epoch, freeze, &manifest).await?;
        self.persist_opaque(epoch).await
    }

    fn take_capture(
        &mut self,
        epoch: u64,
    ) -> Result<Option<(crate::state::table::TableFreeze, V4Manifest)>> {
        let active = self.active_mut(epoch)?;
        if active.persisted.is_some() {
            return Ok(None);
        }
        let freeze = active
            .freeze
            .take()
            .ok_or_else(|| mismatch("checkpoint capture is absent".into()))?;
        let manifest = active
            .manifest
            .take()
            .ok_or_else(|| mismatch("checkpoint manifest is absent".into()))?;
        Ok(Some((freeze, manifest)))
    }

    async fn persist_table(
        &mut self,
        epoch: u64,
        freeze: crate::state::table::TableFreeze,
        manifest: &V4Manifest,
    ) -> Result<()> {
        let previous = self.previous.as_ref();
        let (persisted, manifest) = V4TableCapture::new(freeze, previous)
            .persist_objects(self.store_mut()?, manifest)
            .await?;
        let active = self.active_mut(epoch)?;
        active.persisted = Some(persisted);
        active.manifest = Some(manifest);
        Ok(())
    }

    async fn persist_opaque(&mut self, epoch: u64) -> Result<()> {
        let adapter = self.active_mut(epoch)?.opaque.take();
        let Some(adapter) = adapter else {
            return Ok(());
        };
        let owner = adapter.owner().to_owned();
        let (operator, _) = adapter.persist(self.store_mut()?, epoch).await?;
        let active = self.active_mut(epoch)?;
        Self::record_opaque_manifest(active, owner, operator)
    }

    fn record_opaque_manifest(
        active: &mut ActiveEpoch,
        owner: String,
        operator: OperatorV4,
    ) -> Result<()> {
        let manifest = active
            .manifest
            .as_mut()
            .expect("manifest is present after persistence");
        manifest.operators.insert(owner, operator);
        let fingerprint = crate::state::v4::manifest_fingerprint(manifest)?;
        manifest.manifest_fingerprint = fingerprint;
        active.opaque_persisted = true;
        Ok(())
    }

    pub(in crate::state) fn precommit_sink(&mut self, epoch: u64) -> Result<()> {
        self.active_mut(epoch)?.sink_precommitted = true;
        Ok(())
    }

    pub(in crate::state) fn publish_manifest(&mut self, epoch: u64) -> Result<()> {
        let Some(manifest) = self.manifest_for_publication(epoch)? else {
            return Ok(());
        };
        let path = self.store_mut()?.publish_manifest(&manifest)?;
        self.record_manifest_publication(epoch, path)
    }

    fn manifest_for_publication(&mut self, epoch: u64) -> Result<Option<V4Manifest>> {
        let active = self.active_mut(epoch)?;
        if active.manifest_durable {
            return Ok(None);
        }
        if active.persisted.is_none() {
            return Err(mismatch("manifest publication precedes persistence".into()));
        }
        if !active.sink_precommitted {
            return Err(mismatch(
                "manifest publication precedes sink pre-commit".into(),
            ));
        }
        active
            .manifest
            .clone()
            .map(Some)
            .ok_or_else(|| mismatch("manifest contents are absent".into()))
    }

    fn record_manifest_publication(&mut self, epoch: u64, path: std::path::PathBuf) -> Result<()> {
        let active = self.active_mut(epoch)?;
        if let Some(persisted) = &mut active.persisted {
            persisted.manifest_path.clone_from(&path);
        }
        active.manifest_path = Some(path);
        active.manifest_durable = true;
        Ok(())
    }

    pub(in crate::state) fn commit_sink(&mut self, epoch: u64) -> Result<()> {
        let active = self.active_mut(epoch)?;
        if !active.manifest_durable {
            return Err(mismatch("sink commit precedes manifest durability".into()));
        }
        active.sink_committed = true;
        Ok(())
    }

    pub(in crate::state) fn acknowledge_source(
        &mut self,
        table: &mut StateTable,
        epoch: u64,
    ) -> Result<()> {
        let previous = self.persisted_for_acknowledgement(epoch)?.lineage.clone();
        table.settle_freeze(epoch, previous)?;
        let active = self
            .active
            .take()
            .expect("validated active epoch remains present");
        self.last_completed = Some(epoch);
        self.previous = active.persisted;
        self.recovery = None;
        Ok(())
    }

    fn persisted_for_acknowledgement(&self, epoch: u64) -> Result<&V4PersistedTable> {
        let active = self
            .active
            .as_ref()
            .ok_or_else(|| mismatch("source acknowledgement has no active epoch".into()))?;
        if active.epoch != epoch || active.manifest_path.is_none() || !active.sink_committed {
            return Err(mismatch(
                "source acknowledgement precedes sink commit".into(),
            ));
        }
        active
            .persisted
            .as_ref()
            .ok_or_else(|| mismatch("source acknowledgement has no persisted lineage".into()))
    }

    pub(in crate::state) fn stop(&mut self) {
        if let Some(active) = self.active.take() {
            self.recovery = Some(if active.manifest_durable {
                V4Recovery::RecoveryRequired
            } else {
                V4Recovery::PreviousManifest
            });
        }
        self.stopped = true;
    }

    pub(in crate::state) const fn last_completed(&self) -> Option<u64> {
        self.last_completed
    }

    pub(in crate::state) const fn recovery(&self) -> Option<V4Recovery> {
        self.recovery
    }

    pub(in crate::state) fn active_epoch(&self) -> Option<u64> {
        self.active.as_ref().map(|active| active.epoch)
    }

    fn active_mut(&mut self, epoch: u64) -> Result<&mut ActiveEpoch> {
        match self.active.as_mut() {
            Some(active) if active.epoch == epoch => Ok(active),
            _ => Err(mismatch(
                "checkpoint epoch does not match the active capture".into(),
            )),
        }
    }

    fn store_mut(&mut self) -> Result<&mut V4Store> {
        self.store
            .as_mut()
            .ok_or_else(|| mismatch("checkpoint store is absent".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{EntityKey, ValueKind, table::KeyGroupId};

    fn schema() -> StateTable {
        let mut table = StateTable::with_key_groups(true, &[KeyGroupId(0), KeyGroupId(1)])
            .expect("test key groups");
        table.add_keyed_column("counter", ValueKind::UInt64);
        table.add_numeric_rows_column("numeric", 2);
        table
    }

    fn seed(table: &mut StateTable) {
        table.begin().unwrap();
        let entity = table
            .resolve_or_append(EntityKey::new(b"a".to_vec()))
            .unwrap();
        table.set_u64(0, entity, 2).unwrap();
        table.push_numeric_rows(0, entity, Some(1.0)).unwrap();
        table.push_numeric_rows(0, entity, Some(3.0)).unwrap();
        table.commit().unwrap();
    }

    fn manifest() -> V4Manifest {
        V4Manifest::from_bytes(
            include_str!("../../../tests/fixtures/checkpoint-v4/manifest-v4.json").as_bytes(),
        )
        .unwrap()
    }

    fn manifest_at(epoch: u64) -> V4Manifest {
        let mut manifest = manifest();
        manifest.epoch = epoch;
        manifest.manifest_fingerprint = crate::state::v4::manifest_fingerprint(&manifest).unwrap();
        manifest
    }

    fn store() -> (tempfile::TempDir, V4Store) {
        let temporary = tempfile::tempdir().unwrap();
        let store = V4Store::new(temporary.path().join("lineage"));
        (temporary, store)
    }

    #[tokio::test]
    async fn forwards_before_persistence_then_commits_in_order() {
        let (temporary, store) = store();
        let mut coordinator = V4Coordinator::new(store);
        let mut table = schema();
        seed(&mut table);
        coordinator
            .begin(&mut table, 1, false, manifest_at(1))
            .unwrap();
        coordinator.forward_barrier(1).unwrap();
        assert!(
            !temporary
                .path()
                .join("lineage/manifests/epoch-1.json")
                .exists()
        );
        coordinator.persist(1).await.unwrap();
        assert!(
            !temporary
                .path()
                .join("lineage/manifests/epoch-1.json")
                .exists()
        );
        assert!(coordinator.acknowledge_source(&mut table, 1).is_err());
        assert_eq!(coordinator.active_epoch(), Some(1));
        coordinator.precommit_sink(1).unwrap();
        coordinator.publish_manifest(1).unwrap();
        assert!(
            temporary
                .path()
                .join("lineage/manifests/epoch-1.json")
                .exists()
        );
        coordinator.commit_sink(1).unwrap();
        coordinator.acknowledge_source(&mut table, 1).unwrap();
        assert_eq!(coordinator.last_completed(), Some(1));
    }

    #[tokio::test]
    async fn rejects_overlap_and_supports_terminal_epoch() {
        let (_temporary, store) = store();
        let mut coordinator = V4Coordinator::new(store);
        let mut table = schema();
        seed(&mut table);
        coordinator
            .begin(&mut table, 1, false, manifest_at(1))
            .unwrap();
        assert!(
            coordinator
                .begin(&mut table, 2, false, manifest_at(2))
                .is_err()
        );
        coordinator.persist(1).await.unwrap();
        coordinator.precommit_sink(1).unwrap();
        coordinator.publish_manifest(1).unwrap();
        coordinator.commit_sink(1).unwrap();
        coordinator.acknowledge_source(&mut table, 1).unwrap();
        coordinator
            .begin(&mut table, 2, true, manifest_at(2))
            .unwrap();
        assert!(coordinator.forward_barrier(2).is_err());
        coordinator.persist(2).await.unwrap();
        coordinator.precommit_sink(2).unwrap();
        coordinator.publish_manifest(2).unwrap();
        coordinator.commit_sink(2).unwrap();
        coordinator.acknowledge_source(&mut table, 2).unwrap();
        assert_eq!(coordinator.last_completed(), Some(2));
    }

    #[tokio::test]
    async fn rejects_inflight_reservation_before_capture_ack() {
        let (_temporary, store) = store();
        let mut coordinator = V4Coordinator::with_limit(store, 0);
        let mut table = schema();
        seed(&mut table);
        let error = coordinator
            .begin(&mut table, 1, false, manifest_at(1))
            .unwrap_err();
        assert!(error.to_string().contains("in-flight reservation"));
        assert_eq!(coordinator.active_epoch(), None);
        table
            .freeze(1)
            .expect("rejected capture releases its state-table freeze");
        table.abort_freeze(1).unwrap();
    }

    #[tokio::test]
    async fn opaque_adapter_persists_and_restores_snapshot() {
        let (_temporary, store) = store();
        let mut coordinator = V4Coordinator::new(store);
        let mut table = schema();
        seed(&mut table);
        let snapshot = crate::OperatorStateSnapshot {
            inline_metadata: crate::JsonMap::from([("epoch".into(), serde_json::Value::from(1))]),
            segments: std::collections::BTreeMap::from([(
                "base".into(),
                crate::StateSegment::new(b"state".to_vec()),
            )]),
        };
        coordinator
            .begin(&mut table, 1, false, manifest_at(1))
            .unwrap();
        coordinator
            .begin_with_opaque(V4OpaqueAdapter::from_snapshot("rolling", &snapshot), 1)
            .unwrap();
        coordinator.forward_barrier(1).unwrap();
        coordinator.persist(1).await.unwrap();
        coordinator.precommit_sink(1).unwrap();
        coordinator.publish_manifest(1).unwrap();
        coordinator.commit_sink(1).unwrap();
        coordinator.acknowledge_source(&mut table, 1).unwrap();
        let path = coordinator.previous.as_ref().unwrap().manifest_path.clone();
        let stored = V4Store::read_manifest(&path).unwrap();
        let restored =
            V4OpaqueAdapter::restore(coordinator.store.as_mut().unwrap(), &stored, "rolling")
                .unwrap();
        assert_eq!(
            restored.segments["base"].bytes(),
            snapshot.segments["base"].bytes()
        );
    }
}
