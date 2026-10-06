use super::index_v3::{
    log::{
        self, BucketCut,
        chain::{self, Descriptor, Kind},
        journal::{Change, Identity, Journal, Version},
    },
    owners::{OwnerReader, OwnerWriter},
};
use super::{index_v3, mismatch, payload_segments};
use crate::{
    Result, StateSegment, StreamOperatorContext,
    operator::asof::{
        StreamAsofJoinOperator, checked,
        state::{
            BatchKey, ChunkData, Encoding, Inventory, LeftPrefix, PayloadBatch, PreparedLeftDrain,
            State,
        },
    },
};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Payload {
    pub key: BatchKey,
    pub sha256: String,
    pub bytes: u64,
    pub charge_bytes: u64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Control {
    pub frames: Vec<Descriptor>,
    pub payloads: Vec<Payload>,
    pub generation: u64,
    pub owner_capacity: usize,
    pub base_payloads: Vec<BatchKey>,
}

struct CapturedPayloads {
    payloads: BTreeMap<BatchKey, StateSegment>,
    payload_charges: BTreeMap<BatchKey, u64>,
    base_payloads: Vec<BatchKey>,
}

struct CapturedFrames {
    frames: Vec<Descriptor>,
    segments: BTreeMap<String, StateSegment>,
    owners: OwnerWriter,
    body_credit: Option<MemoryReservation>,
}

struct RecoveredRows {
    state: State,
    owners: OwnerReader,
    workspaces: Vec<MemoryReservation>,
}

struct PaidControl {
    control: Control,
    workspace: MemoryReservation,
}

struct LogRecoveryInput {
    direct_wire: Arc<MemoryReservation>,
    workspace: MemoryReservation,
    batches: BTreeMap<BatchKey, Arc<PayloadBatch>>,
}

pub(in crate::operator::asof) struct Body {
    pub kind: Kind,
    pub segment: StateSegment,
    pub owners: OwnerWriter,
    pub credit: MemoryReservation,
    pub records: u64,
}

pub(in crate::operator::asof) struct LogState {
    pub journal: Journal,
    pub frames: Vec<Descriptor>,
    pub segments: BTreeMap<String, StateSegment>,
    pub payloads: BTreeMap<BatchKey, StateSegment>,
    pub payload_charges: BTreeMap<BatchKey, u64>,
    pub retention_bytes: u64,
    pub registry: Option<OwnerWriter>,
    pub credit: Option<Arc<MemoryReservation>>,
    pub force_base: bool,
    pub dirty_cut: bool,
    pub generation: u64,
    pub pending: Option<Body>,
    pub base_payloads: Vec<BatchKey>,
}

impl Default for LogState {
    fn default() -> Self {
        Self {
            journal: Journal::default(),
            frames: Vec::new(),
            segments: BTreeMap::new(),
            payloads: BTreeMap::new(),
            payload_charges: BTreeMap::new(),
            retention_bytes: 0,
            registry: None,
            credit: None,
            force_base: true,
            dirty_cut: false,
            generation: 0,
            pending: None,
            base_payloads: Vec::new(),
        }
    }
}

impl LogState {
    pub fn bytes(&self) -> u64 {
        self.journal.bytes()
            + self.retention_bytes
            + self
                .segments
                .values()
                .map(|segment| segment.bytes_arc().capacity() as u64 + 256)
                .sum::<u64>()
    }
    pub fn has_owner(&self, address: usize) -> bool {
        self.registry
            .as_ref()
            .is_some_and(|registry| registry.contains_address(address))
    }
    pub fn keeps_delta(&self) -> bool {
        !self.force_base && !self.frames.is_empty()
    }
    pub fn install_journal(&mut self, prepared: Journal) {
        self.force_base |= prepared.requires_base();
        self.journal.install(prepared);
    }
    pub fn credit_bytes(
        &self,
        owns: impl Fn(usize) -> bool,
        payload_live: impl Fn(&BatchKey) -> bool,
    ) -> Result<u64> {
        credit_bytes(
            &self.frames,
            &self.payloads,
            self.registry.as_ref(),
            &self.payload_charges,
            false,
            owns,
            payload_live,
        )
    }
}

fn credit_bytes(
    frames: &[Descriptor],
    payloads: &BTreeMap<BatchKey, StateSegment>,
    registry: Option<&OwnerWriter>,
    charges: &BTreeMap<BatchKey, u64>,
    only_unowned: bool,
    owns: impl Fn(usize) -> bool,
    payload_live: impl Fn(&BatchKey) -> bool,
) -> Result<u64> {
    let mut bytes = if frames.is_empty() && payloads.is_empty() && registry.is_none() {
        0
    } else {
        4096 + (frames.len() + payloads.len()) as u64 * 1024
    };
    bytes = registry_credit_bytes(bytes, registry, &owns)?;
    payload_credit_bytes(bytes, payloads, charges, only_unowned, &payload_live)
}

fn registry_credit_bytes(
    mut bytes: u64,
    registry: Option<&OwnerWriter>,
    owns: &impl Fn(usize) -> bool,
) -> Result<u64> {
    if let Some(registry) = registry {
        bytes = checked("asof", bytes, registry.metadata_bytes())?;
        for (address, allocation) in registry.allocations() {
            if !owns(address) {
                bytes = checked("asof", bytes, allocation)?;
            }
        }
    }
    Ok(bytes)
}

fn payload_credit_bytes(
    mut bytes: u64,
    payloads: &BTreeMap<BatchKey, StateSegment>,
    charges: &BTreeMap<BatchKey, u64>,
    only_unowned: bool,
    payload_live: &impl Fn(&BatchKey) -> bool,
) -> Result<u64> {
    for (key, segment) in payloads {
        if !payload_live(key) && (!only_unowned || !segment.has_owner()) {
            let capacity = charges
                .get(key)
                .copied()
                .ok_or_else(|| mismatch("ASOF historical payload charge is missing"))?;
            bytes = checked("asof", bytes, capacity + 256)?;
        }
    }
    Ok(bytes)
}

fn admission_changes(
    admission: &super::super::admission::Admission,
    side: usize,
    count: usize,
) -> Vec<Change> {
    let mut edits = Vec::with_capacity(count);
    if side == 0 {
        for chunk in admission
            .left_chunks
            .as_ref()
            .expect("prepared left chunks")
        {
            edits.push(Change {
                identity: Identity::Left(chunk.batch_key()),
                before: None,
                after: Some(chunk.journal_version()),
            });
        }
    } else {
        for (identity, reference) in &admission.rows {
            let batch = &admission.batches[reference.batch_index];
            edits.push(Change {
                identity: Identity::Right(identity.clone()),
                before: None,
                after: Some(Version::Right {
                    tag: 1,
                    payload: Some((batch.key, reference.row)),
                }),
            });
        }
    }
    edits
}

fn delta_has_new_left(changes: &[Change], batch: BatchKey) -> bool {
    changes
        .binary_search_by(|change| change.identity.cmp(&Identity::Left(batch)))
        .ok()
        .is_some_and(|index| changes[index].before.is_none())
}

fn reserve(pool: &Arc<dyn MemoryPool>, bytes: u64, name: &str) -> Result<MemoryReservation> {
    let bytes = usize::try_from(bytes).map_err(|_| {
        crate::operator::asof::reason(
            name,
            crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
            "ASOF workspace exceeds address domain",
        )
    })?;
    let credit = MemoryConsumer::new("asof-owned-workspace").register(pool);
    credit.try_grow(bytes).map_err(|_| {
        crate::operator::asof::reason(
            name,
            crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
            "ASOF aggregate workspace exceeds max_state_bytes",
        )
    })?;
    Ok(credit)
}

struct OwnedDelta {
    capacities: [usize; 16],
    counts: [usize; 3],
    kinds: [super::super::state::SequenceKind; 2],
    changes: Vec<Change>,
    left: Vec<(BatchKey, Arc<ChunkData>, usize)>,
    buckets: Vec<BucketCut>,
    registry: OwnerWriter,
    live: BTreeSet<usize>,
    workspace: MemoryReservation,
}

impl OwnedDelta {
    fn encode(self, pool: &Arc<dyn MemoryPool>, limit: u64, name: &str) -> Result<Body> {
        self.encode_checked(pool, limit, name, &|| Ok(()))
    }

    fn encode_checked(
        self,
        pool: &Arc<dyn MemoryPool>,
        limit: u64,
        name: &str,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<Body> {
        cancel()?;
        debug_assert!(self.workspace.size() >= 4096);
        let left = self
            .left
            .iter()
            .map(|(batch, data, head)| (*batch, data.as_ref(), *head))
            .collect::<Vec<_>>();
        let input = log::Input {
            capacities: self.capacities,
            counts: self.counts,
            kinds: self.kinds,
            changes: &self.changes,
            left: &left,
            buckets: &self.buckets,
        };
        let encoded = log::encode_checked(
            &input,
            &self.registry,
            |address| self.live.contains(&address),
            |bytes| reserve(pool, bytes, name),
            limit,
            name,
            cancel,
        )?;
        cancel()?;
        let body = Body {
            kind: Kind::Delta,
            segment: encoded.segment,
            owners: encoded.owners,
            credit: encoded.owner_credit,
            records: self.changes.len() as u64,
        };
        drop(left);
        drop(self);
        Ok(body)
    }
}

struct BaseWork {
    snapshot: index_v3::owned::Index,
    workspace: MemoryReservation,
    pool: Arc<dyn MemoryPool>,
    name: String,
    limit: usize,
    length: u64,
    rows: u64,
}

impl crate::runtime::streaming::gather_work::OwnedCpuWork for BaseWork {
    type Output = Body;

    fn run(self, stop: &crate::runtime::streaming::gather_work::GatherStop) -> Result<Body> {
        stop.check()?;
        let (segment, owners) =
            self.snapshot
                .encode_registered_checked(self.length, self.limit, &|| stop.check())?;
        let wire = self.workspace.split(segment.bytes_arc().capacity() + 256);
        let credit = reserve(&self.pool, owners.metadata_bytes(), &self.name)?;
        stop.check()?;
        Ok(Body {
            kind: Kind::Base,
            segment: segment.with_owner(Arc::new(wire)),
            owners,
            credit,
            records: self.rows,
        })
    }
}

struct DeltaWork {
    input: OwnedDelta,
    pool: Arc<dyn MemoryPool>,
    limit: u64,
    name: String,
}

impl crate::runtime::streaming::gather_work::OwnedCpuWork for DeltaWork {
    type Output = Body;

    fn run(self, stop: &crate::runtime::streaming::gather_work::GatherStop) -> Result<Body> {
        self.input
            .encode_checked(&self.pool, self.limit, &self.name, &|| stop.check())
    }
}

impl StreamAsofJoinOperator {
    fn log_prefers_base(&self, changes: u64) -> bool {
        let rows = changes.saturating_add(self.checkpoint_log.journal.changes().len() as u64);
        rows >= 4096 && rows >= self.status.state_rows / 8
    }

    pub(in crate::operator::asof) fn log_projection(
        &self,
        mut inventory: Inventory,
        length: u64,
        journal: &Journal,
        retention_bytes: u64,
    ) -> Result<Inventory> {
        inventory.bytes = inventory
            .bytes
            .checked_sub(if length == 0 { 0 } else { length + 256 })
            .ok_or_else(|| mismatch("ASOF virtual index accounting differs"))?;
        let frames = self
            .checkpoint_log
            .segments
            .values()
            .map(|segment| segment.bytes_arc().capacity() as u64 + 256)
            .sum::<u64>();
        inventory.bytes = checked(
            &self.name,
            inventory.bytes,
            checked(
                &self.name,
                frames + journal.bytes() + self.replay_bytes(),
                retention_bytes,
            )?,
        )?;
        self.check_inventory_limits(&inventory)?;
        Ok(inventory)
    }

    pub(in crate::operator::asof) fn prepare_log_retention(
        &self,
        removals: &super::super::state::OwnerRemovals,
        batches: &BTreeMap<BatchKey, usize>,
    ) -> Result<(Arc<MemoryReservation>, u64)> {
        let owns = |address| self.state.keeps_encoding(address, removals);
        let live = |key: &BatchKey| {
            self.state.batches.references(key) > batches.get(key).copied().unwrap_or(0)
        };
        let log = &self.checkpoint_log;
        let gauge = credit_bytes(
            &log.frames,
            &log.payloads,
            log.registry.as_ref(),
            &log.payload_charges,
            false,
            owns,
            live,
        )?;
        let paid = credit_bytes(
            &log.frames,
            &log.payloads,
            log.registry.as_ref(),
            &log.payload_charges,
            true,
            owns,
            live,
        )?;
        Ok((Arc::new(self.reserve_workspace(paid)?), gauge))
    }

    pub(in crate::operator::asof) fn prepare_log_admission(
        &self,
        admission: &super::super::admission::Admission,
        side: usize,
        owners: &super::super::state::OwnerUpdates,
    ) -> Result<Journal> {
        if !self.checkpoint_log.keeps_delta() {
            return Ok(Journal::default());
        }
        let count = if side == 0 {
            admission.left_chunks.as_ref().map_or(0, Vec::len)
        } else {
            admission.rows.len()
        };
        if self.log_prefers_base(count as u64) {
            return Ok(Journal::base());
        }
        let _workspace = self.reserve_workspace(count as u64 * size_of::<Change>() as u64 + 256)?;
        let edits = admission_changes(admission, side, count);
        self.checkpoint_log.journal.prepare(
            &edits,
            |address| {
                self.state.owns_encoding(address)
                    || owners.contains(address)
                    || self.checkpoint_log.has_owner(address)
            },
            |bytes| self.reserve_workspace(bytes),
            &self.name,
        )
    }

    pub(in crate::operator::asof) fn prepare_log_prefix(
        &self,
        prefix: &LeftPrefix,
        drain: &PreparedLeftDrain,
    ) -> Result<Journal> {
        if !self.checkpoint_log.keeps_delta() {
            return Ok(Journal::default());
        }
        if self.log_prefers_base(prefix.batches.len() as u64) {
            return Ok(Journal::base());
        }
        let _workspace = self.reserve_workspace(
            prefix.batches.len() as u64 * (size_of::<Change>() + 512) as u64 + 4096,
        )?;
        let edits = self
            .state
            .left
            .journal_prefix(prefix, drain, &self.state.batches);
        self.checkpoint_log.journal.prepare(
            &edits,
            |address| {
                self.state.keeps_encoding(address, &prefix.owners)
                    || self.checkpoint_log.has_owner(address)
            },
            |bytes| self.reserve_workspace(bytes),
            &self.name,
        )
    }

    pub(in crate::operator::asof) fn prepare_log_eviction(
        &self,
        preview: &super::super::state::EvictionPreview,
    ) -> Result<Journal> {
        if !self.checkpoint_log.keeps_delta() {
            return Ok(Journal::default());
        }
        let count = preview.evicted_payloads
            + preview.removed_identity_only
            + preview.selected.len() as u64;
        if self.log_prefers_base(count) {
            return Ok(Journal::base());
        }
        let _workspace =
            self.reserve_workspace(count * (size_of::<Change>() + 512) as u64 * 2 + 4096)?;
        let mut edits = Vec::with_capacity(usize::try_from(count).expect("bounded eviction rows"));
        let threshold = super::super::state::retention_threshold(&self.state, &self.status);
        for handle in &preview.selected {
            let (key, bucket) = self.state.right.indexed_bucket(*handle);
            edits.extend(bucket.journal_eviction(
                key,
                &self.state.batches,
                &self.status,
                self.spec.tolerance_micros(),
                threshold,
            ));
        }
        self.checkpoint_log.journal.prepare(
            &edits,
            |address| {
                self.state.keeps_encoding(address, &preview.owners)
                    || self.checkpoint_log.has_owner(address)
            },
            |bytes| self.reserve_workspace(bytes),
            &self.name,
        )
    }

    fn owned_delta(&self) -> Result<OwnedDelta> {
        let registry = self
            .checkpoint_log
            .registry
            .as_ref()
            .ok_or_else(|| mismatch("ASOF delta registry is missing"))?;
        let changes = self.checkpoint_log.journal.changes();
        let bound = self.delta_workspace_bound(registry, changes)?;
        let workspace = self.reserve_workspace(bound)?;
        let mut left = self
            .state
            .left
            .checkpoint_owned_chunks(&self.state.batches)
            .filter(|(batch, _, _)| delta_has_new_left(changes, *batch))
            .collect::<Vec<_>>();
        left.sort_unstable_by_key(|(batch, _, _)| *batch);
        let keys = self.checkpoint_log.journal.keys().to_vec();
        let buckets = keys
            .into_iter()
            .map(|key| {
                let state = self
                    .state
                    .right
                    .get(&key)
                    .map(|bucket| (bucket.len() as u64, bucket.checkpoint_capacities()));
                BucketCut { key, state }
            })
            .collect();
        Ok(OwnedDelta {
            capacities: log::model::capacities(&self.state),
            counts: log::model::counts(&self.state),
            kinds: self.sequence_kinds(),
            changes: changes.to_vec(),
            left,
            buckets,
            registry: registry.clone(),
            live: self.state.encoding_addresses().collect(),
            workspace,
        })
    }

    fn delta_workspace_bound(&self, registry: &OwnerWriter, changes: &[Change]) -> Result<u64> {
        let (buffers, metadata) = self.state.encoding_owner_allocation();
        let mut bound = checked(
            &self.name,
            buffers,
            metadata
                + registry.metadata_bytes()
                + (changes.len() + self.checkpoint_log.journal.keys().len()) as u64
                    * (size_of::<Change>() + 192) as u64
                + 4096,
        )?;
        for (batch, data, _) in self.state.left.checkpoint_chunks(&self.state.batches) {
            if delta_has_new_left(changes, batch) {
                bound = checked(&self.name, bound, data.retained_input_bytes(&self.name)?)?;
            }
        }
        for (_, bytes) in registry.allocations() {
            bound = checked(&self.name, bound, bytes)?;
        }
        Ok(bound)
    }

    fn needs_log_base(&self) -> bool {
        self.checkpoint_log.force_base
            || self.checkpoint_log.frames.is_empty()
            || self.checkpoint_log.frames.len() > chain::MAX_DELTAS
    }

    async fn encode_log_base_async(&self, context: &StreamOperatorContext<'_>) -> Result<Body> {
        let length = index_v3::encoded_length(&self.state, &self.name)?.max(index_v3::BASE_BYTES);
        let workspace = self.reserve_workspace(index_v3::workspace_bytes(
            &self.state,
            length,
            true,
            &self.name,
        )?)?;
        let snapshot = index_v3::owned::Index::capture(&self.state, context).await?;
        let pool = self.runtime.pool.clone();
        let name = self.name.clone();
        let limit =
            usize::try_from(self.spec.limits().max_state_bytes()).expect("validated byte limit");
        let rows = self.status.state_rows;
        self.run_cpu_work(
            BaseWork {
                snapshot,
                workspace,
                pool,
                name,
                limit,
                length,
                rows,
            },
            context,
        )
        .await
    }

    fn row_log_ready(&self) -> bool {
        self.checkpoint_log.pending.is_some()
            || (!self.needs_log_base()
                && self.checkpoint_log.journal.is_empty()
                && !self.checkpoint_log.dirty_cut)
    }

    async fn encode_log_body_async(&self, context: &StreamOperatorContext<'_>) -> Result<Body> {
        if self.needs_log_base() {
            self.encode_log_base_async(context).await
        } else {
            let input = self.owned_delta()?;
            let pool = self.runtime.pool.clone();
            let name = self.name.clone();
            let limit = self.spec.limits().max_state_bytes();
            self.run_cpu_work(
                DeltaWork {
                    input,
                    pool,
                    limit,
                    name,
                },
                context,
            )
            .await
        }
    }

    async fn compact_log_body_async(
        &self,
        body: Body,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Body> {
        if self.log_compaction_needed(&body)? {
            drop(body);
            self.encode_log_base_async(context).await
        } else {
            Ok(body)
        }
    }

    pub(super) async fn prepare_row_log_async(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        context.check_cancelled()?;
        if self.empty_row_log()? {
            return Ok(());
        }
        if self.row_log_ready() {
            return Ok(());
        }
        let body = self.encode_log_body_async(context).await?;
        let body = self.compact_log_body_async(body, context).await?;
        context.check_cancelled()?;
        self.checkpoint_log.pending = Some(body);
        Ok(())
    }

    pub(super) fn prepare_row_log_sync(&mut self) -> Result<()> {
        if self.empty_row_log()? {
            return Ok(());
        }
        if self.row_log_ready() {
            return Ok(());
        }
        let body = if self.needs_log_base() {
            self.encode_log_base_sync()?
        } else {
            self.owned_delta()?.encode(
                &self.runtime.pool,
                self.spec.limits().max_state_bytes(),
                &self.name,
            )?
        };
        self.checkpoint_log.pending = Some(body);
        Ok(())
    }
}

impl StreamAsofJoinOperator {
    fn log_compaction_needed(&self, body: &Body) -> Result<bool> {
        if body.kind == Kind::Base {
            return Ok(false);
        }
        let native = self.state.capacity_inventory(None, &self.name)?.bytes;
        let retired = body
            .owners
            .allocations()
            .filter(|(address, _)| !self.state.owns_encoding(*address))
            .try_fold(0, |bytes, (_, next)| checked(&self.name, bytes, next))?;
        let metadata = self.checkpoint_log.credit_bytes(
            |address| self.state.owns_encoding(address),
            |key| self.state.batches.references(key) != 0,
        )?;
        Ok(chain::requires_compaction(
            &self.checkpoint_log.frames,
            body.segment.bytes().len() as u64 + chain::HEADER_BYTES as u64,
            metadata,
            retired,
            index_v3::encoded_length(&self.state, &self.name)?.max(index_v3::BASE_BYTES),
            self.spec.limits().max_state_bytes().saturating_sub(native),
        ))
    }

    fn encode_log_base_sync(&self) -> Result<Body> {
        let length = index_v3::encoded_length(&self.state, &self.name)?.max(index_v3::BASE_BYTES);
        let workspace = self.reserve_workspace(index_v3::workspace_bytes(
            &self.state,
            length,
            false,
            &self.name,
        )?)?;
        let owners = index_v3::source_owners(&self.state);
        let credit = self.reserve_workspace(owners.metadata_bytes())?;
        let segment = index_v3::encode_sync(
            &self.state,
            length,
            usize::try_from(self.spec.limits().max_state_bytes()).expect("validated byte limit"),
        )?;
        let wire = workspace.split(segment.bytes_arc().capacity() + 256);
        Ok(Body {
            kind: Kind::Base,
            segment: segment.with_owner(Arc::new(wire)),
            owners,
            credit,
            records: self.status.state_rows,
        })
    }

    pub(super) fn empty_row_log(&self) -> Result<bool> {
        Ok(self.state.capacity_inventory(None, &self.name)?.bytes == 0
            && self.checkpoint_log.frames.is_empty())
    }

    pub(super) fn capture_row_log(
        &mut self,
        epoch: crate::Epoch,
    ) -> Result<crate::OperatorStateSnapshot> {
        if self.empty_row_log()? {
            return self.capture_empty_log(epoch);
        }
        let pending = self.checkpoint_log.pending.take();
        let pending = self.compact_pending_log(pending, epoch)?;
        let count = self.checkpoint_log.frames.len()
            + self.checkpoint_log.payloads.len()
            + self.state.batches.len()
            + 1;
        let _metadata_workspace = self.reserve_workspace(8192 + count as u64 * 2048)?;
        let is_base = pending.as_ref().is_some_and(|body| body.kind == Kind::Base);
        let generation = self.captured_log_generation(is_base)?;
        let CapturedPayloads {
            payloads,
            payload_charges,
            base_payloads,
        } = self.capture_log_payloads(is_base);
        let CapturedFrames {
            frames,
            segments,
            owners,
            body_credit: _body_credit,
        } = self.capture_log_frames(epoch, pending, is_base, generation)?;
        let (bytes, credit) =
            self.captured_log_credit(&frames, &payloads, &owners, &payload_charges)?;
        let log = LogState {
            journal: Journal::default(),
            frames,
            segments,
            payloads,
            payload_charges,
            retention_bytes: bytes,
            registry: Some(owners),
            credit: Some(credit.clone()),
            force_base: false,
            dirty_cut: false,
            generation,
            pending: None,
            base_payloads,
        };
        self.publish_captured_log(log, epoch)
    }

    fn publish_captured_log(
        &mut self,
        log: LogState,
        epoch: crate::Epoch,
    ) -> Result<crate::OperatorStateSnapshot> {
        let mut inventory = self.state.capacity_inventory(None, &self.name)?;
        inventory.bytes = checked(&self.name, inventory.bytes, log.bytes())?;
        self.check_inventory_limits(&inventory)?;
        let snapshot = self.capture_log_snapshot(&log, epoch, &inventory)?;
        let index_length = index_v3::encoded_length(&self.state, &self.name)?;
        self.checkpoint_log = log;
        self.status.state_bytes = inventory.bytes;
        self.prepared = None;
        self.deferred_index_len = Some(index_length);
        Ok(snapshot)
    }

    fn capture_empty_log(&self, epoch: crate::Epoch) -> Result<crate::OperatorStateSnapshot> {
        let mut inline_metadata = self.capture_metadata(epoch, 10)?;
        if self.payload_projection.is_none() {
            inline_metadata.remove("retained_payloads");
        }
        inline_metadata.insert("checkpoint_log".into(), serde_json::json!({"frames": [], "payloads": [], "generation": 0, "owner_capacity": 0, "base_payloads": []}));
        Ok(crate::OperatorStateSnapshot {
            inline_metadata,
            segments: BTreeMap::new(),
        })
    }

    fn compact_pending_log(
        &self,
        pending: Option<Body>,
        epoch: crate::Epoch,
    ) -> Result<Option<Body>> {
        if self.log_epoch_requires_base(epoch, pending.is_some()) {
            drop(pending);
            return Ok(Some(self.encode_log_base_sync()?));
        }
        match pending {
            Some(body) if self.log_compaction_needed(&body)? => {
                drop(body);
                Ok(Some(self.encode_log_base_sync()?))
            }
            other => Ok(other),
        }
    }

    fn log_epoch_requires_base(&self, epoch: crate::Epoch, pending: bool) -> bool {
        self.checkpoint_log.frames.last().is_some_and(|previous| {
            previous.epoch > epoch.as_u64() || (pending && previous.epoch == epoch.as_u64())
        })
    }

    fn captured_log_generation(&self, is_base: bool) -> Result<u64> {
        if is_base {
            self.checkpoint_log
                .generation
                .checked_add(1)
                .ok_or_else(|| mismatch("ASOF log generation overflowed"))
        } else {
            Ok(self.checkpoint_log.generation)
        }
    }

    fn captured_log_credit(
        &self,
        frames: &[Descriptor],
        payloads: &BTreeMap<BatchKey, StateSegment>,
        owners: &OwnerWriter,
        payload_charges: &BTreeMap<BatchKey, u64>,
    ) -> Result<(u64, Arc<MemoryReservation>)> {
        let bytes = credit_bytes(
            frames,
            payloads,
            Some(owners),
            payload_charges,
            false,
            |address| self.state.owns_encoding(address),
            |key| self.state.batches.references(key) != 0,
        )?;
        let paid = credit_bytes(
            frames,
            payloads,
            Some(owners),
            payload_charges,
            true,
            |address| self.state.owns_encoding(address),
            |key| self.state.batches.references(key) != 0,
        )?;
        let credit = Arc::new(self.reserve_workspace(paid)?);
        Ok((bytes, credit))
    }

    fn capture_log_payloads(&self, is_base: bool) -> CapturedPayloads {
        let mut payloads = if is_base {
            BTreeMap::new()
        } else {
            self.checkpoint_log.payloads.clone()
        };
        let mut payload_charges = if is_base {
            BTreeMap::new()
        } else {
            self.checkpoint_log.payload_charges.clone()
        };
        for (key, (batch, _)) in self.state.batches.iter() {
            payload_charges.insert(*key, batch.encoded_charge_bytes);
            payloads
                .entry(*key)
                .or_insert_with(|| batch.encoded.get().expect("prepared ASOF payload").clone());
        }
        let base_payloads = if is_base {
            let mut keys = self
                .state
                .batches
                .iter()
                .map(|(key, _)| *key)
                .collect::<Vec<_>>();
            keys.sort_unstable();
            keys
        } else {
            self.checkpoint_log.base_payloads.clone()
        };
        CapturedPayloads {
            payloads,
            payload_charges,
            base_payloads,
        }
    }

    fn capture_log_frames(
        &self,
        epoch: crate::Epoch,
        pending: Option<Body>,
        is_base: bool,
        generation: u64,
    ) -> Result<CapturedFrames> {
        let mut frames = if is_base {
            Vec::with_capacity(chain::MAX_DELTAS + 1)
        } else {
            let mut frames = Vec::with_capacity(chain::MAX_DELTAS + 1);
            frames.extend_from_slice(&self.checkpoint_log.frames);
            frames
        };
        let mut segments = if is_base {
            BTreeMap::new()
        } else {
            self.checkpoint_log.segments.clone()
        };
        let (owners, body_credit) = if let Some(body) = pending {
            let Body {
                kind: _,
                segment,
                owners,
                credit,
                records,
            } = body;
            let length = segment.bytes().len() as u64 + chain::HEADER_BYTES as u64 + 256;
            let wire = self.reserve_workspace(length)?;
            let (descriptor, segment) = chain::encode(
                generation,
                epoch.as_u64(),
                frames.last(),
                &self.fingerprint,
                records,
                segment.bytes(),
                wire,
            )?;
            segments.insert(descriptor.name(), segment);
            frames.push(descriptor);
            (owners, Some(credit))
        } else {
            (
                self.checkpoint_log
                    .registry
                    .as_ref()
                    .ok_or_else(|| mismatch("ASOF log registry is missing"))?
                    .clone(),
                None,
            )
        };
        Ok(CapturedFrames {
            frames,
            segments,
            owners,
            body_credit,
        })
    }

    fn capture_log_snapshot(
        &self,
        log: &LogState,
        epoch: crate::Epoch,
        inventory: &Inventory,
    ) -> Result<crate::OperatorStateSnapshot> {
        let mut inline_metadata = self.capture_metadata(epoch, 10)?;
        let credit = log.credit.as_ref().expect("captured log credit");
        inline_metadata
            .get_mut("metrics")
            .expect("serialized ASOF metrics")["state_bytes"] = serde_json::json!(inventory.bytes);
        let control = Control {
            frames: log.frames.clone(),
            payloads: log
                .payloads
                .iter()
                .map(|(key, segment)| Payload {
                    key: *key,
                    sha256: segment.sha256().into(),
                    bytes: segment.bytes().len() as u64,
                    charge_bytes: log.payload_charges[key],
                })
                .collect(),
            generation: log.generation,
            owner_capacity: log.registry.as_ref().expect("prepared registry").capacity(),
            base_payloads: log.base_payloads.clone(),
        };
        inline_metadata.insert(
            "checkpoint_log".into(),
            serde_json::to_value(control).map_err(|error| mismatch(&error.to_string()))?,
        );
        let mut snapshot_segments = log
            .segments
            .iter()
            .map(|(name, segment)| (name.clone(), segment.clone().with_owner(credit.clone())))
            .collect::<BTreeMap<_, _>>();
        for (key, segment) in &log.payloads {
            snapshot_segments.insert(
                payload_segments::batch_segment(*key),
                segment.clone().with_owner(credit.clone()),
            );
        }
        Ok(crate::OperatorStateSnapshot {
            inline_metadata,
            segments: snapshot_segments,
        })
    }

    fn decode_empty_log(
        &self,
        snapshot: &crate::OperatorStateSnapshot,
        metadata: &super::Metadata<'_>,
        raw: &serde_json::Value,
    ) -> Result<super::DecodedSnapshot> {
        if raw
            != &serde_json::json!({"frames": [], "payloads": [], "generation": 0, "owner_capacity": 0, "base_payloads": []})
            || snapshot.inline_metadata.contains_key("retained_payloads")
                != self.payload_projection.is_some()
        {
            return Err(mismatch("ASOF empty log control differs"));
        }
        let mut state = State::empty_tracked();
        state.sequence_kinds = self.sequence_kinds();
        let inventory = state.capacity_inventory(None, &self.name)?;
        super::validate_gauges(&inventory, 0, &metadata.metrics)?;
        self.validate_restored_limits(&inventory, metadata)?;
        Ok(super::DecodedSnapshot {
            state,
            metrics: metadata.metrics.clone(),
            terminal: metadata.terminal,
            sequence: metadata.next_output_sequence,
            prepared: None,
            checkpoint_log: LogState::default(),
            _workspaces: Vec::new(),
        })
    }

    fn log_payload_workspace(
        &self,
        control: &Control,
        snapshot: &crate::OperatorStateSnapshot,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<u64> {
        let mut payload_workspace_bytes = 0;
        let mut previous = None;
        for (ordinal, payload) in control.payloads.iter().enumerate() {
            index_v3::check_step(ordinal, cancel)?;
            if previous.is_some_and(|previous| previous >= payload.key) || payload.key.0 > 1 {
                return Err(mismatch("ASOF log payload order differs"));
            }
            previous = Some(payload.key);
            let body = validated_payload_body(payload, snapshot)?;
            payload_workspace_bytes = checked(
                &self.name,
                payload_workspace_bytes,
                body + 1152
                    + self
                        .physical_schema(usize::from(payload.key.0))
                        .fields()
                        .len() as u64
                        * 320,
            )?;
        }
        Ok(payload_workspace_bytes)
    }

    fn decode_log_payloads(
        &self,
        control: &Control,
        snapshot: &crate::OperatorStateSnapshot,
        direct_wire: &Arc<MemoryReservation>,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<BTreeMap<BatchKey, Arc<PayloadBatch>>> {
        let mut batches = BTreeMap::new();
        for (ordinal, payload) in control.payloads.iter().enumerate() {
            index_v3::check_step(ordinal, cancel)?;
            let segment = &snapshot.segments[&payload_segments::batch_segment(payload.key)];
            let segment = if segment.has_owner() {
                segment.clone()
            } else {
                segment.clone().with_owner(direct_wire.clone())
            };
            let batch = self.decode_payload_batch(payload.key, &segment, true)?;
            if payload.charge_bytes != batch.encoded_charge_bytes
                || payload.charge_bytes < payload.bytes
            {
                return Err(mismatch("ASOF historical payload charge differs"));
            }
            batches.insert(payload.key, batch);
        }
        Ok(batches)
    }

    fn decode_log_rows(
        &self,
        control: &Control,
        frames: &[chain::Frame<'_>],
        batches: &BTreeMap<BatchKey, Arc<PayloadBatch>>,
        workspace: &MemoryReservation,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<RecoveredRows> {
        let (mut state, mut owners) =
            self.decode_log_base_rows(control, frames, batches, cancel)?;
        let mut workspaces = Vec::with_capacity(36);
        if frames.len() > 1 {
            let recovered = self.apply_log_frames(state, owners, frames, batches, cancel)?;
            state = recovered.state;
            owners = recovered.owners;
            workspaces = recovered.workspaces;
        } else {
            lease_recovered_index(&mut state, workspace)?;
        }
        self.validate_indexed_rows(&state)?;
        Ok(RecoveredRows {
            state,
            owners,
            workspaces,
        })
    }

    fn decode_log_base_rows(
        &self,
        control: &Control,
        frames: &[chain::Frame<'_>],
        batches: &BTreeMap<BatchKey, Arc<PayloadBatch>>,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<(State, OwnerReader)> {
        if control
            .base_payloads
            .windows(2)
            .any(|keys| keys[0] >= keys[1])
        {
            return Err(mismatch("ASOF base payload order differs"));
        }
        let base_batches = control
            .base_payloads
            .iter()
            .map(|key| {
                batches
                    .get(key)
                    .cloned()
                    .map(|batch| (*key, batch))
                    .ok_or_else(|| mismatch("ASOF base payload is missing"))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let (state, owners) = index_v3::decode_registered_bytes_checked(
            frames[0].body,
            &base_batches,
            self.spec.limits().max_state_rows(),
            self.spec.limits().max_state_bytes(),
            self.sequence_kinds(),
            cancel,
        )?;
        let base_inventory = state.capacity_inventory(None, &self.name)?;
        if frames[0].records != base_inventory.identities {
            return Err(mismatch("ASOF base row census differs"));
        }
        Ok((state, owners))
    }

    fn apply_log_frames(
        &self,
        state: State,
        mut owners: OwnerReader,
        frames: &[chain::Frame<'_>],
        batches: &BTreeMap<BatchKey, Arc<PayloadBatch>>,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<RecoveredRows> {
        let base_inventory = state.capacity_inventory(None, &self.name)?;
        let workspaces = Vec::with_capacity(36);
        let model_bytes = log_model_bytes(&state, &base_inventory)?;
        let mut model = log::model::Model::from_state(
            &state,
            self.reserve_workspace(model_bytes as u64)?,
            cancel,
        )?;
        drop(state);
        owners = self.apply_model_frames(&mut model, owners, frames, batches, cancel)?;
        self.materialize_log_model(model, owners, batches, workspaces, cancel)
    }

    fn apply_model_frames(
        &self,
        model: &mut log::model::Model,
        mut owners: OwnerReader,
        frames: &[chain::Frame<'_>],
        batches: &BTreeMap<BatchKey, Arc<PayloadBatch>>,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<OwnerReader> {
        for frame in frames.iter().skip(1) {
            cancel()?;
            let delta = self.decode_log_frame(frame, &owners, batches, cancel)?;
            owners = model.apply(delta, cancel)?;
        }
        Ok(owners)
    }

    fn decode_log_frame(
        &self,
        frame: &chain::Frame<'_>,
        owners: &OwnerReader,
        batches: &BTreeMap<BatchKey, Arc<PayloadBatch>>,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<log::DecodedDelta> {
        let bytes = log::restore_charge(
            frame.body,
            owners.count(),
            self.sequence_kinds(),
            self.spec.limits().max_state_rows(),
            self.spec.limits().max_state_bytes(),
        )?;
        let delta = log::decode(
            frame.body,
            owners,
            batches,
            self.sequence_kinds(),
            &self.spec.limits(),
            self.reserve_workspace(bytes)?,
            cancel,
        )?;
        if delta.changes.len() as u64 != frame.records {
            return Err(mismatch("ASOF delta row census differs"));
        }
        Ok(delta)
    }

    fn materialize_log_model(
        &self,
        model: log::model::Model,
        owners: OwnerReader,
        batches: &BTreeMap<BatchKey, Arc<PayloadBatch>>,
        mut workspaces: Vec<MemoryReservation>,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<RecoveredRows> {
        let candidate = self.reserve_workspace(model.native_charge(cancel)?)?;
        let mut state = model.materialize(
            batches,
            &candidate,
            self.spec.limits().max_state_rows(),
            cancel,
        )?;
        lease_recovered_index(&mut state, &candidate)?;
        workspaces.extend(model.into_workspaces());
        workspaces.push(candidate);
        Ok(RecoveredRows {
            state,
            owners,
            workspaces,
        })
    }

    fn restore_log_inventory(
        &self,
        control: Control,
        snapshot: &crate::OperatorStateSnapshot,
        state: &State,
        owners: OwnerReader,
        direct_wire: &Arc<MemoryReservation>,
    ) -> Result<(LogState, MemoryReservation)> {
        let registry_work = self.reserve_workspace(
            4096 + control.owner_capacity as u64 * (size_of::<Encoding>() + 256) as u64,
        )?;
        let registry = owners.into_writer(control.owner_capacity)?;
        let mut segments = BTreeMap::new();
        for frame in &control.frames {
            let segment = &snapshot.segments[&frame.name()];
            segments.insert(frame.name(), segment_with_wire_owner(segment, direct_wire));
        }
        let payloads = control
            .payloads
            .iter()
            .map(|payload| {
                (payload.key, {
                    let segment = &snapshot.segments[&payload_segments::batch_segment(payload.key)];
                    segment_with_wire_owner(segment, direct_wire)
                })
            })
            .collect::<BTreeMap<_, _>>();
        let payload_charges = control
            .payloads
            .iter()
            .map(|payload| (payload.key, payload.charge_bytes))
            .collect::<BTreeMap<_, _>>();
        let bytes = credit_bytes(
            &control.frames,
            &payloads,
            Some(&registry),
            &payload_charges,
            false,
            |address| state.owns_encoding(address),
            |key| state.batches.references(key) != 0,
        )?;
        let paid = credit_bytes(
            &control.frames,
            &payloads,
            Some(&registry),
            &payload_charges,
            true,
            |address| state.owns_encoding(address),
            |key| state.batches.references(key) != 0,
        )?;
        let credit = Arc::new(self.reserve_workspace(paid)?);
        let checkpoint_log = LogState {
            journal: Journal::default(),
            frames: control.frames,
            segments,
            payloads,
            payload_charges,
            retention_bytes: bytes,
            registry: Some(registry),
            credit: Some(credit),
            force_base: false,
            dirty_cut: false,
            generation: control.generation,
            pending: None,
            base_payloads: control.base_payloads,
        };
        Ok((checkpoint_log, registry_work))
    }

    pub(super) fn decode_row_log(
        &self,
        snapshot: &crate::OperatorStateSnapshot,
        metadata: &super::Metadata<'_>,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<super::DecodedSnapshot> {
        cancel()?;
        let raw = snapshot
            .inline_metadata
            .get("checkpoint_log")
            .ok_or_else(|| mismatch("ASOF log control is missing"))?;
        if snapshot.segments.is_empty() {
            return self.decode_empty_log(snapshot, metadata, raw);
        }
        self.decode_nonempty_row_log(snapshot, metadata, raw, cancel)
    }

    fn decode_nonempty_row_log(
        &self,
        snapshot: &crate::OperatorStateSnapshot,
        metadata: &super::Metadata<'_>,
        raw: &serde_json::Value,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<super::DecodedSnapshot> {
        let paid = self.decode_log_control(raw, snapshot)?;
        let control = &paid.control;
        let frames = chain::validate_chain(
            &control.frames,
            &snapshot.segments,
            metadata.epoch,
            &self.fingerprint,
            &paid.workspace,
            cancel,
        )?;
        let input = self.prepare_log_recovery(control, snapshot, &frames, cancel)?;
        super::validate_batch_ranges(&input.batches, &metadata.metrics)?;
        let recovered =
            self.decode_log_rows(control, &frames, &input.batches, &input.workspace, cancel)?;
        self.finish_log_restore(paid, snapshot, metadata, recovered, input, cancel)
    }

    fn decode_log_control(
        &self,
        raw: &serde_json::Value,
        snapshot: &crate::OperatorStateSnapshot,
    ) -> Result<PaidControl> {
        let control_workspace = self.reserve_workspace(control_bound(raw, 0)? + 8192)?;
        let serialized =
            serde_json::to_string(raw).map_err(|error| mismatch(&error.to_string()))?;
        let control: Control =
            serde_json::from_str(&serialized).map_err(|error| mismatch(&error.to_string()))?;
        validate_log_control(&control, snapshot)?;
        Ok(PaidControl {
            control,
            workspace: control_workspace,
        })
    }

    fn prepare_log_recovery(
        &self,
        control: &Control,
        snapshot: &crate::OperatorStateSnapshot,
        frames: &[chain::Frame<'_>],
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<LogRecoveryInput> {
        let payload_workspace_bytes = self.log_payload_workspace(control, snapshot, cancel)?;
        let wire_bytes = snapshot
            .segments
            .values()
            .filter(|segment| !segment.has_owner())
            .try_fold(0, |bytes, segment| {
                checked(
                    &self.name,
                    bytes,
                    segment.bytes_arc().capacity() as u64 + 256,
                )
            })?;
        let direct_wire = Arc::new(self.reserve_workspace(wire_bytes)?);
        let workspace = self.reserve_workspace(
            payload_workspace_bytes
                + index_v3::restore_charge_checked(
                    frames[0].body,
                    self.spec.limits().max_state_rows(),
                    self.spec.limits().max_state_bytes(),
                    cancel,
                )?,
        )?;
        let batches = self.decode_log_payloads(control, snapshot, &direct_wire, cancel)?;
        Ok(LogRecoveryInput {
            direct_wire,
            workspace,
            batches,
        })
    }

    fn finish_log_restore(
        &self,
        paid: PaidControl,
        snapshot: &crate::OperatorStateSnapshot,
        metadata: &super::Metadata<'_>,
        recovered: RecoveredRows,
        input: LogRecoveryInput,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<super::DecodedSnapshot> {
        let RecoveredRows {
            state,
            owners,
            mut workspaces,
        } = recovered;
        let (checkpoint_log, registry_work) =
            self.restore_log_inventory(paid.control, snapshot, &state, owners, &input.direct_wire)?;
        let mut base_inventory = state.capacity_inventory(None, &self.name)?;
        base_inventory.bytes = checked(&self.name, base_inventory.bytes, checkpoint_log.bytes())?;
        super::validate_gauges(&base_inventory, state.left.len() as u64, &metadata.metrics)?;
        self.validate_restored_limits(&base_inventory, metadata)?;
        workspaces.push(input.workspace);
        workspaces.push(paid.workspace);
        workspaces.push(registry_work);
        cancel()?;
        Ok(super::DecodedSnapshot {
            state,
            metrics: metadata.metrics.clone(),
            terminal: metadata.terminal,
            sequence: metadata.next_output_sequence,
            prepared: None,
            checkpoint_log,
            _workspaces: workspaces,
        })
    }
}

fn validated_payload_body(
    payload: &Payload,
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<u64> {
    let encoded = snapshot
        .segments
        .get(&payload_segments::batch_segment(payload.key))
        .ok_or_else(|| mismatch("ASOF historical payload is missing"))?;
    if encoded.sha256() != payload.sha256 || encoded.bytes().len() as u64 != payload.bytes {
        return Err(mismatch("ASOF historical payload digest or length differs"));
    }
    let body = super::super::codec::payload_body_bytes(encoded.bytes())
        .map_err(|_| mismatch("ASOF invalid Arrow batch framing"))?;
    Ok(body)
}

fn segment_with_wire_owner(segment: &StateSegment, wire: &Arc<MemoryReservation>) -> StateSegment {
    if segment.has_owner() {
        segment.clone()
    } else {
        segment.clone().with_owner(wire.clone())
    }
}

fn validate_log_control(control: &Control, snapshot: &crate::OperatorStateSnapshot) -> Result<()> {
    if control.frames.is_empty()
        || control.frames.len() > chain::MAX_DELTAS + 1
        || control.generation == 0
        || control
            .frames
            .iter()
            .any(|frame| frame.generation != control.generation)
        || control.payloads.len() + control.frames.len() != snapshot.segments.len()
    {
        return Err(mismatch("ASOF log control inventory differs"));
    }
    Ok(())
}

fn lease_recovered_index(state: &mut State, workspace: &MemoryReservation) -> Result<()> {
    let auxiliary = state.right.auxiliary_bytes();
    if auxiliary > workspace.size() {
        return Err(mismatch("ASOF recovery index exceeds prepaid workspace"));
    }
    state
        .right
        .install_recovery_lease(workspace.split(auxiliary));
    Ok(())
}

fn log_model_bytes(state: &State, base_inventory: &Inventory) -> Result<usize> {
    let model_bytes = usize::try_from(base_inventory.identities)
        .map_err(|_| mismatch("ASOF model exceeds address domain"))?
        .saturating_mul(size_of::<((i64, Encoding), Version)>())
        .saturating_add(state.right.len().saturating_mul(512))
        .saturating_add(
            state
                .left
                .checkpoint_chunks(&state.batches)
                .len()
                .saturating_mul(512),
        )
        .saturating_add(4096);
    Ok(model_bytes)
}

fn control_bound(value: &serde_json::Value, depth: usize) -> Result<u64> {
    if depth > 8 {
        return Err(mismatch("ASOF log control nesting differs"));
    }
    match value {
        serde_json::Value::String(value) => checked("asof", 256, value.len() as u64 * 8),
        serde_json::Value::Array(values) => values.iter().try_fold(256, |bytes, value| {
            checked("asof", bytes, control_bound(value, depth + 1)?)
        }),
        serde_json::Value::Object(values) => values.iter().try_fold(256, |bytes, (key, value)| {
            checked(
                "asof",
                bytes,
                checked(
                    "asof",
                    key.len() as u64 * 8,
                    control_bound(value, depth + 1)?,
                )?,
            )
        }),
        _ => Ok(256),
    }
}
