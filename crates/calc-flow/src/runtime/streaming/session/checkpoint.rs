//! Synchronous durable session checkpoints.

use std::path::PathBuf;

use super::{
    BTreeMap, Duration, EventTime, GeneratedWatermarkState, Lifecycle, Result, SessionInput,
    SessionWatermark, StreamExecutionPlan, StreamRuntimeConfig, StreamSession, WatermarkMode,
    session_block_on, session_error, session_runtime,
};
use crate::{
    Epoch, SourceWatermarkManifestState,
    state::{
        StateTable,
        v4::{
            V4Store,
            coordinator::V4Coordinator,
            manifest::{JobKind, SourceV4, V4Manifest, manifest_fingerprint},
            opaque::V4OpaqueAdapter,
        },
    },
};

pub(super) struct SessionCheckpoints {
    _opened: crate::runtime::streaming::checkpoint::OpenedManagedCheckpointRuntime,
    coordinator: V4Coordinator,
    runtime: tokio::runtime::Runtime,
}

impl SessionCheckpoints {
    fn new(
        opened: crate::runtime::streaming::checkpoint::OpenedManagedCheckpointRuntime,
        last_completed: u64,
    ) -> Result<Self> {
        let store = V4Store::new(opened.manifest_root().join("lineage"));
        Ok(Self {
            _opened: opened,
            coordinator: V4Coordinator::with_completed(store, last_completed),
            runtime: session_runtime()?,
        })
    }

    fn manifest_dir(
        opened: &crate::runtime::streaming::checkpoint::OpenedManagedCheckpointRuntime,
    ) -> PathBuf {
        opened.manifest_root().join("lineage/manifests")
    }

    fn latest_manifest(
        opened: &crate::runtime::streaming::checkpoint::OpenedManagedCheckpointRuntime,
    ) -> Result<(V4Manifest, PathBuf)> {
        let directory = Self::manifest_dir(opened);
        let entries = std::fs::read_dir(&directory)
            .map_err(|_| session_error("no restorable v4 session checkpoint manifest was found"))?;
        let mut selected = None;
        for entry in entries {
            let entry =
                entry.map_err(|_| session_error("cannot read the v4 manifest directory"))?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some(epoch) = name
                .strip_prefix("epoch-")
                .and_then(|name| name.strip_suffix(".json"))
                .and_then(|name| name.parse::<u64>().ok())
            else {
                continue;
            };
            if selected
                .as_ref()
                .is_none_or(|(previous, _)| epoch > *previous)
            {
                selected = Some((epoch, path));
            }
        }
        let (epoch, path) = selected.ok_or_else(|| {
            session_error("no restorable v4 session checkpoint manifest was found")
        })?;
        let manifest = V4Store::read_manifest(&path)?;
        if manifest.job_kind != JobKind::Session {
            return Err(session_error(
                "checkpoint manifest job kind runner cannot restore a session; expected session",
            ));
        }
        if manifest.epoch != epoch {
            return Err(session_error(
                "checkpoint manifest epoch differs from its filename",
            ));
        }
        Ok((manifest, path))
    }

    fn ensure_fresh(
        opened: &crate::runtime::streaming::checkpoint::OpenedManagedCheckpointRuntime,
    ) -> Result<()> {
        let directory = Self::manifest_dir(opened);
        if directory.exists()
            && std::fs::read_dir(&directory)
                .map_err(|_| session_error("cannot read the v4 manifest directory"))?
                .next()
                .is_some()
        {
            return Err(session_error(
                "session checkpoint lineage already contains a manifest; restore it or use a new lineage",
            ));
        }
        Ok(())
    }

    fn capture(&mut self, session: &mut StreamSession) -> Result<Epoch> {
        let epoch = self.coordinator.last_completed().unwrap_or(0) + 1;
        let result = {
            let SessionCheckpoints {
                coordinator,
                runtime,
                ..
            } = self;
            session_block_on(runtime, Self::capture_async(coordinator, session, epoch))
        };
        if result.is_err() {
            if self.coordinator.active_epoch().is_some() {
                session.lifecycle = Lifecycle::Failed;
            }
            self.coordinator.stop();
        }
        result
    }

    async fn capture_async(
        coordinator: &mut V4Coordinator,
        session: &mut StreamSession,
        epoch: u64,
    ) -> Result<Epoch> {
        let mut table = StateTable::with_checkpointing(true);
        let manifest = session.manifest_for_checkpoint(epoch)?;
        coordinator.begin(
            &mut table,
            epoch,
            session.lifecycle == Lifecycle::Finished,
            manifest,
        )?;
        for node in &mut session.nodes {
            let snapshot =
                node.checkpoint(Epoch::new(epoch).expect("checkpoint epoch is nonzero"))?;
            coordinator.begin_with_opaque(
                V4OpaqueAdapter::from_snapshot(node.node_id(), &snapshot),
                epoch,
            )?;
        }
        if session.lifecycle != Lifecycle::Finished {
            coordinator.forward_barrier(epoch)?;
        }
        coordinator.persist(epoch).await?;
        coordinator.precommit_sink(epoch)?;
        coordinator.publish_manifest(epoch)?;
        coordinator.commit_sink(epoch)?;
        coordinator.acknowledge_source(&mut table, epoch)?;
        Epoch::new(epoch).ok_or_else(|| session_error("checkpoint epoch is zero"))
    }
}

impl StreamSession {
    fn open_storage(
        &self,
        checkpoints: crate::ManagedCheckpointRuntime,
    ) -> Result<crate::runtime::streaming::checkpoint::OpenedManagedCheckpointRuntime> {
        let runtime = session_runtime()?;
        session_block_on(&runtime, checkpoints.open(self.job.cancellation()))
    }

    /// # Errors
    ///
    /// Returns an error when checkpointing is disabled, the lineage already
    /// contains a manifest, storage preparation fails, or graph admission fails.
    /// Starts a fresh durable session lineage.
    pub fn new_with_checkpoints(
        plan: StreamExecutionPlan,
        config: &StreamRuntimeConfig,
        watermarks: BTreeMap<String, SessionWatermark>,
        checkpoints: crate::ManagedCheckpointRuntime,
    ) -> Result<Self> {
        if !config.checkpointing {
            return Err(session_error(
                "checkpoint storage requires checkpointing=true",
            ));
        }
        let mut session = Self::new(plan, config, watermarks)?;
        session.job = session.job.clone().with_checkpointing(true);
        let opened = Self::open_storage(&session, checkpoints)?;
        SessionCheckpoints::ensure_fresh(&opened)?;
        session.checkpoints = Some(SessionCheckpoints::new(opened, 0)?);
        Ok(session)
    }

    /// # Errors
    ///
    /// Returns an error when checkpointing is disabled, no complete session
    /// manifest exists, the job kind or fingerprint differs, or state restoration fails.
    /// Restores the highest complete durable v4 session checkpoint.
    pub fn restore(
        plan: StreamExecutionPlan,
        config: &StreamRuntimeConfig,
        watermarks: BTreeMap<String, SessionWatermark>,
        checkpoints: crate::ManagedCheckpointRuntime,
    ) -> Result<Self> {
        if !config.checkpointing {
            return Err(session_error(
                "checkpoint storage requires checkpointing=true",
            ));
        }
        let mut session = Self::new(plan, config, watermarks)?;
        session.job = session.job.clone().with_checkpointing(true);
        let opened = Self::open_storage(&session, checkpoints)?;
        let (manifest, _) = SessionCheckpoints::latest_manifest(&opened)?;
        if manifest.recovery_fingerprint != session.job.fingerprint() {
            return Err(session_error(
                "checkpoint recovery fingerprint differs from the plan",
            ));
        }
        let input_id = session.input.binding_id.clone();
        let source = manifest.sources.get(&input_id).ok_or_else(|| {
            session_error("checkpoint manifest does not contain the session input")
        })?;
        session.restore_input(source);
        for node in &mut session.nodes {
            let mut store = V4Store::new(opened.manifest_root().join("lineage"));
            let snapshot = V4OpaqueAdapter::restore(&mut store, &manifest, node.node_id())?;
            node.restore(&snapshot)?;
        }
        session.lifecycle = if manifest.sources[&input_id].ended {
            Lifecycle::Finished
        } else {
            Lifecycle::Open
        };
        session.checkpoints = Some(SessionCheckpoints::new(opened, manifest.epoch)?);
        Ok(session)
    }

    /// # Errors
    ///
    /// Returns an error when the session has no durable storage, is already
    /// finished or failed, or capture, persistence or publication fails.
    /// Synchronously persists the current session state and returns its epoch.
    pub fn checkpoint(&mut self) -> Result<Epoch> {
        self.check_callable()?;
        let mut checkpoints = self.checkpoints.take().ok_or_else(|| {
            session_error(
                "invalid session.checkpoints: checkpoint requires managed checkpoint storage",
            )
        })?;
        let result = checkpoints.capture(self);
        self.checkpoints = Some(checkpoints);
        result
    }

    fn manifest_for_checkpoint(&self, epoch: u64) -> Result<V4Manifest> {
        let source = SourceV4 {
            cursor: None,
            identity_hash: "session".into(),
            next_sequence: self.input.progress.next_sequence.unwrap_or(0),
            ended: self.lifecycle == Lifecycle::Finished,
            watermark_policy: self.input.watermark_manifest_state(),
            history: None,
        };
        let manifest = V4Manifest {
            format_version: 4,
            pipeline_name: "session".into(),
            pipeline_fingerprint: self.job.fingerprint().to_owned(),
            runtime_config_hash: String::new(),
            epoch,
            created_at: chrono::Utc::now(),
            recovery_status: "final".into(),
            job_kind: JobKind::Session,
            recovery_fingerprint: self.job.fingerprint().to_owned(),
            sources: BTreeMap::from([(self.input.binding_id.clone(), source)]),
            operators: BTreeMap::new(),
            sinks: BTreeMap::new(),
            static_inputs: BTreeMap::new(),
            manifest_fingerprint: String::new(),
        };
        Ok(V4Manifest {
            manifest_fingerprint: manifest_fingerprint(&manifest)?,
            ..manifest
        })
    }

    fn restore_input(&mut self, source: &SourceV4) {
        self.input.progress.next_sequence = Some(source.next_sequence);
        self.input.progress.watermark = source.watermark_policy.last_emitted_micros();
        self.input.progress.max_event_time = source
            .watermark_policy
            .observed_max_micros()
            .map(EventTime::as_micros);
        if let WatermarkMode::Generated { event_time, .. } = &self.input.mode {
            let (observed, last) = match source.watermark_policy {
                SourceWatermarkManifestState::BoundedOutOfOrderness {
                    observed_max_micros,
                    last_emitted_micros,
                    ..
                } => (observed_max_micros, last_emitted_micros),
                _ => (None, source.watermark_policy.last_emitted_micros()),
            };
            let delay = self
                .input
                .progress
                .generator
                .as_ref()
                .map_or(Duration::from_nanos(0), |generator| {
                    generator.max_out_of_orderness()
                });
            self.input.progress.generator = Some(GeneratedWatermarkState::restore(
                event_time.clone(),
                delay,
                observed,
                last,
            ));
        }
    }
}

trait SessionWatermarkManifest {
    fn last_emitted_micros(&self) -> Option<EventTime>;
    fn observed_max_micros(&self) -> Option<EventTime>;
}

impl SessionWatermarkManifest for SourceWatermarkManifestState {
    fn last_emitted_micros(&self) -> Option<EventTime> {
        match self {
            Self::SourceProvided {
                last_emitted_micros,
                ..
            }
            | Self::BoundedOutOfOrderness {
                last_emitted_micros,
                ..
            } => *last_emitted_micros,
            Self::Disabled { .. } => None,
        }
    }

    fn observed_max_micros(&self) -> Option<EventTime> {
        match self {
            Self::BoundedOutOfOrderness {
                observed_max_micros,
                ..
            } => *observed_max_micros,
            _ => None,
        }
    }
}

impl SessionInput {
    fn watermark_manifest_state(&self) -> SourceWatermarkManifestState {
        match &self.mode {
            WatermarkMode::Explicit => SourceWatermarkManifestState::SourceProvided {
                last_emitted_micros: self.progress.watermark,
                idle: false,
            },
            WatermarkMode::Generated { .. } => {
                SourceWatermarkManifestState::BoundedOutOfOrderness {
                    observed_max_micros: self.progress.max_event_time.map(EventTime::from_micros),
                    last_emitted_micros: self.progress.watermark,
                    idle: false,
                }
            }
            WatermarkMode::Disabled => SourceWatermarkManifestState::Disabled { idle: false },
        }
    }
}
