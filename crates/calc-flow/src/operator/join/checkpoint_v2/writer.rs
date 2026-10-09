use std::sync::Arc;

use datafusion::{arrow::datatypes::SchemaRef, execution::memory_pool::MemoryReservation};

use super::super::{PendingOp, StoredRow, StreamJoinOperator};
use super::{
    budget as restore_budget,
    ipc::accounting::{self, sum},
};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherScope, GatherStop, ObservedTicket, OwnedCpuWork,
    cleanup_control_bytes,
};
use crate::{Epoch, OperatorStateSnapshot, Result, StreamOperatorContext};

mod budget;
mod buffer;
mod compact;
mod cut;
mod encode;
mod metadata;

#[cfg(test)]
pub(in crate::operator::join) type WriterTestHook = Arc<dyn Fn(&MemoryReservation) + Send + Sync>;

#[derive(Default)]
pub(in crate::operator::join) struct WriterState {
    tracker: cut::Tracker,
    prepared: Option<PreparedBase>,
    prepared_delta: Option<PreparedBase>,
    deltas: Vec<metadata::DeltaEntry>,
    base: Option<encode::Base>,
    base_epoch: u64,
    dirty_epochs: u32,
    migration_required: bool,
}

struct PreparedBase {
    cut: cut::Cut,
    base: encode::Base,
}

impl WriterState {
    pub(super) fn restored(
        snapshot: &OperatorStateSnapshot,
        inventory: &super::inventory::Inventory<'_>,
        workspace: &MemoryReservation,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        let history = encode::restored::decode(snapshot, inventory, workspace, check)?;
        Ok(Self {
            base: Some(history.base),
            deltas: history.deltas,
            base_epoch: history.base_epoch,
            dirty_epochs: history.dirty_epochs,
            ..Self::default()
        })
    }

    pub(super) fn bind_owner(&mut self, previous: &Self) -> Result<()> {
        self.tracker = previous.tracker.next_owner()?;
        Ok(())
    }

    pub(in crate::operator::join) fn changed(&mut self, dirty: bool) -> Result<()> {
        self.tracker = self.tracker.advanced(dirty)?;
        Ok(())
    }

    pub(in crate::operator::join) fn next_owner(&self, migration_required: bool) -> Result<Self> {
        Ok(Self {
            tracker: self.tracker.next_owner()?,
            migration_required,
            ..Self::default()
        })
    }
}

struct Inputs {
    rows: [Option<Arc<Vec<StoredRow>>>; 2],
    pending: Option<Vec<PendingOp>>,
    _containers: Option<Arc<super::ContainerFunding>>,
    released: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Drop for Inputs {
    fn drop(&mut self) {
        drop(self.rows[0].take());
        drop(self.rows[1].take());
        drop(self.pending.take());
        self._containers = None;
        if let Some(released) = self.released.take() {
            let _ = released.send(());
        }
    }
}

struct SubmittedWriter {
    cut: cut::Cut,
    kind: Preparation,
    _scope: GatherScope,
    ticket: Option<ObservedTicket<encode::Base>>,
    _caller_credit: MemoryReservation,
}

struct WriterWork {
    inputs: Inputs,
    schemas: [SchemaRef; 2],
    name: String,
    #[cfg(test)]
    hook: Option<WriterTestHook>,
    #[cfg(test)]
    base_hook: Option<WriterTestHook>,
    workspace: MemoryReservation,
}

#[derive(Clone, Copy)]
enum Preparation {
    Base,
    Delta,
}

impl OwnedCpuWork for WriterWork {
    type Output = encode::Base;

    fn control_bytes(&self) -> Result<usize> {
        sum(&[
            size_of::<Self>(),
            self.name.capacity(),
            cleanup_control_bytes::<Self::Output>(),
            super::super::metadata_validation::inventory::caller_controls(&self.name)
                .ok_or_else(|| buffer::error("V2 checkpoint work controls overflow"))?,
            restore_budget::registration_bytes()?,
        ])
    }

    fn run(self, stop: &GatherStop) -> Result<Self::Output> {
        stop.check()?;
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(&self.workspace);
        }
        let result = self.encode(stop);
        stop.check()?;
        result
    }
}

impl WriterWork {
    fn encode(&self, stop: &GatherStop) -> Result<encode::Base> {
        if let Some(pending) = &self.inputs.pending {
            return encode::pending_owned(
                pending,
                &self.schemas,
                Epoch::INITIAL,
                &self.workspace,
                &|| stop.check(),
            );
        }
        let sides = self
            .inputs
            .rows
            .each_ref()
            .map(|rows| rows.as_ref().expect("owned V2 checkpoint input").as_slice());
        #[cfg(test)]
        if let Some(hook) = &self.base_hook {
            hook(&self.workspace);
        }
        encode::base(sides, &self.schemas, &self.workspace, &|| stop.check())
    }
}

impl StreamJoinOperator {
    pub(in crate::operator::join) async fn prepare_v2_checkpoint(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        self.begin_v2_preparation(context).await?;
        self.prepare_v2_kind(Preparation::Base, context).await
    }

    pub(in crate::operator::join) async fn prepare_v2_checkpoint_automatic(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        self.begin_v2_preparation(context).await?;
        let Some(kind) = self.v2_preparation_kind() else {
            return context.check_cancelled();
        };
        self.prepare_v2_kind(kind, context).await
    }

    fn v2_preparation_kind(&self) -> Option<Preparation> {
        if self.state.last_checkpoint_epoch.is_none()
            || self.v2_writer.migration_required
            || self.v2_writer.dirty_epochs >= 4
        {
            Some(Preparation::Base)
        } else if !self.state.deltas.pending.is_empty() {
            Some(Preparation::Delta)
        } else {
            None
        }
    }

    async fn prepare_v2_kind(
        &mut self,
        kind: Preparation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let mut submitted = self.submit_v2_writer(kind, context).await?;
        let prepared = submitted
            .ticket
            .take()
            .expect("one submitted writer")
            .finish()
            .await?;
        self.await_snapshot_release(context).await?;
        prepared.install(|base| {
            self.install_v2_writer_base(base, submitted.cut, submitted.kind, context)
        })?;
        self.await_compaction_release(context).await?;
        context.check_cancelled()
    }

    async fn begin_v2_preparation(&mut self, context: &StreamOperatorContext<'_>) -> Result<()> {
        context.check_cancelled()?;
        self.await_compaction_release(context).await?;
        self.check_v2_prepare_epoch()
    }

    fn check_v2_prepare_epoch(&self) -> Result<()> {
        if self
            .state
            .last_checkpoint_epoch
            .is_some_and(|epoch| epoch.as_u64() == u64::MAX)
        {
            return Err(buffer::error("V2 checkpoint has no advancing epoch"));
        }
        Ok(())
    }

    fn install_v2_writer_base(
        &mut self,
        base: encode::Base,
        cut: cut::Cut,
        kind: Preparation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        context.check_cancelled()?;
        if cut
            != self
                .v2_writer
                .tracker
                .snapshot(self.state.last_checkpoint_epoch)
        {
            return Ok(());
        }
        if matches!(kind, Preparation::Delta) {
            self.v2_writer.prepared_delta = Some(PreparedBase { cut, base });
            return Ok(());
        }
        if let Some(epoch) = self.state.last_checkpoint_epoch {
            let tracker = self.v2_writer.tracker.advanced(true)?;
            self.v2_writer.deltas = Vec::new();
            self.v2_writer.base = Some(base);
            self.v2_writer.base_epoch = epoch.as_u64();
            self.v2_writer.dirty_epochs = 0;
            self.v2_writer.prepared = None;
            self.v2_writer.prepared_delta = None;
            self.v2_writer.migration_required = false;
            self.state.deltas.pending.clear();
            self.v2_writer.tracker = tracker;
        } else {
            self.v2_writer.prepared = Some(PreparedBase { cut, base });
        }
        Ok(())
    }

    async fn submit_v2_writer(
        &mut self,
        kind: Preparation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<SubmittedWriter> {
        let cut = self
            .v2_writer
            .tracker
            .snapshot(self.state.last_checkpoint_epoch);
        let workspace = self.v2_writer_workspace()?;
        let scope = context
            .gather_client(GatherOperatorId::new(Arc::from(self.name.as_str())))
            .scope()?;
        let retirement = context.job().gather_owner().retain_retirement()?;
        let credit = workspace.new_empty();
        let work = self.v2_writer_work(kind, workspace.new_empty(), context)?;
        let ticket = scope
            .submit_observed_work(
                work,
                credit,
                GatherStop::from_job(context.job()),
                retirement,
                &mut self.compaction_cleanup,
            )
            .await
            .map_err(writer_admission)?;
        Ok(SubmittedWriter {
            cut,
            kind,
            _scope: scope,
            ticket: Some(ticket),
            _caller_credit: workspace,
        })
    }

    fn v2_writer_workspace(&mut self) -> Result<MemoryReservation> {
        let workspace = self
            .runtime
            .runtime()?
            .incremental_reservation("stream-join-v2-write");
        accounting::reserve(
            &workspace,
            sum(&[
                self.name.len(),
                size_of::<SubmittedWriter>(),
                budget::diagnostic_bytes(),
                super::super::metadata_validation::inventory::caller_controls(&self.name)
                    .ok_or_else(|| buffer::error("V2 checkpoint caller controls overflow"))?,
                restore_budget::registration_bytes()?,
            ])?,
        )?;
        Ok(workspace)
    }

    fn v2_writer_work(
        &mut self,
        kind: Preparation,
        workspace: MemoryReservation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<WriterWork> {
        accounting::reserve(
            &workspace,
            sum(&[
                self.name.len(),
                size_of::<WriterWork>(),
                release_channel_bytes()?,
                budget::diagnostic_bytes(),
                restore_budget::registration_bytes()?,
            ])?,
        )?;
        let mut inputs = self.v2_writer_inputs(kind, &workspace, &|| context.check_cancelled())?;
        let (released, receiver) = tokio::sync::oneshot::channel();
        self.compaction_release = Some(receiver);
        inputs.released = Some(released);
        Ok(WriterWork {
            inputs,
            schemas: [
                Arc::clone(self.input_schema(0)),
                Arc::clone(self.input_schema(1)),
            ],
            name: self.name.clone(),
            #[cfg(test)]
            hook: self.checkpoint_writer_test_hook.clone(),
            #[cfg(test)]
            base_hook: self.checkpoint_writer_base_test_hook.clone(),
            workspace,
        })
    }

    fn v2_writer_inputs(
        &self,
        kind: Preparation,
        workspace: &MemoryReservation,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Inputs> {
        let mut inputs = Inputs {
            rows: [None, None],
            pending: None,
            _containers: self.v2_containers.clone(),
            released: None,
        };
        match kind {
            Preparation::Base => {
                inputs.rows = [
                    Some(Arc::clone(&self.state.left.0)),
                    Some(Arc::clone(&self.state.right.0)),
                ];
            }
            Preparation::Delta => {
                inputs.pending = Some(encode::copy_pending(
                    &self.state.deltas.pending,
                    workspace,
                    check,
                )?);
            }
        }
        Ok(inputs)
    }

    pub(in crate::operator::join) fn capture_v2_checkpoint(
        &mut self,
        epoch: Epoch,
    ) -> Result<OperatorStateSnapshot> {
        self.check_v2_capture_epoch(epoch)?;
        let tracker = self
            .v2_writer
            .tracker
            .advanced(!self.state.deltas.pending.is_empty())?;
        let snapshot = if self.fresh_v2_preparation_matches() {
            self.capture_initial_prepared(epoch)?
        } else {
            self.capture_v2_dirty(epoch)?
        };
        self.state.deltas.pending.clear();
        self.state.last_checkpoint_epoch = Some(epoch);
        self.v2_writer.tracker = tracker;
        self.v2_writer.prepared = None;
        self.v2_writer.prepared_delta = None;
        Ok(snapshot)
    }

    fn check_v2_capture_epoch(&self, epoch: Epoch) -> Result<()> {
        if self
            .state
            .last_checkpoint_epoch
            .is_some_and(|previous| epoch <= previous)
        {
            return Err(buffer::error(
                "stream Join checkpoint epoch did not advance strictly",
            ));
        }
        if self.v2_writer.migration_required {
            return Err(buffer::error(
                "V2 checkpoint requires successful migration preparation",
            ));
        }
        Ok(())
    }

    fn fresh_v2_preparation_matches(&self) -> bool {
        self.state.last_checkpoint_epoch.is_none()
            && self
                .v2_writer
                .prepared
                .as_ref()
                .is_some_and(|prepared| prepared.cut == self.v2_writer.tracker.snapshot(None))
    }

    fn capture_initial_prepared(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        let prepared = self
            .v2_writer
            .prepared
            .as_ref()
            .expect("matching fresh preparation");
        let anchor = if prepared.base.payloads.is_empty() {
            0
        } else {
            epoch.as_u64()
        };
        let snapshot = metadata::encode(
            self,
            epoch,
            anchor,
            &[],
            &prepared.base.payloads,
            prepared.base.credit(),
        )?;
        let snapshot = OperatorStateSnapshot {
            inline_metadata: snapshot,
            segments: clone_segments(&prepared.base, None, prepared.base.credit())?,
        };
        let prepared = self
            .v2_writer
            .prepared
            .take()
            .expect("matching fresh preparation");
        self.v2_writer.base = Some(prepared.base);
        self.v2_writer.base_epoch = anchor;
        Ok(snapshot)
    }

    fn capture_v2_dirty(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        let workspace = self.v2_writer_workspace()?;
        let schemas = [
            Arc::clone(self.input_schema(0)),
            Arc::clone(self.input_schema(1)),
        ];
        let empty = self.v2_empty_base(&schemas, &workspace)?;
        if self.state.deltas.pending.is_empty() {
            self.capture_v2_clean(epoch, empty)
        } else {
            self.capture_v2_pending(epoch, &schemas, &workspace, empty)
        }
    }

    fn v2_empty_base(
        &self,
        schemas: &[SchemaRef; 2],
        workspace: &MemoryReservation,
    ) -> Result<Option<encode::Base>> {
        if self.v2_writer.base.is_none() {
            Ok(Some(encode::base([&[], &[]], schemas, workspace, &|| {
                Ok(())
            })?))
        } else {
            Ok(None)
        }
    }

    fn current_v2_base<'a>(&'a self, empty: &'a Option<encode::Base>) -> &'a encode::Base {
        self.v2_writer
            .base
            .as_ref()
            .or(empty.as_ref())
            .expect("one current base")
    }

    fn capture_v2_clean(
        &mut self,
        epoch: Epoch,
        empty: Option<encode::Base>,
    ) -> Result<OperatorStateSnapshot> {
        let base = self.current_v2_base(&empty);
        let inline_metadata = metadata::encode(
            self,
            epoch,
            self.v2_writer.base_epoch,
            &self.v2_writer.deltas,
            &base.payloads,
            base.credit(),
        )?;
        let snapshot = OperatorStateSnapshot {
            inline_metadata,
            segments: clone_segments(base, None, base.credit())?,
        };
        if let Some(empty) = empty {
            self.v2_writer.base = Some(empty);
        }
        Ok(snapshot)
    }

    fn capture_v2_pending(
        &mut self,
        epoch: Epoch,
        schemas: &[SchemaRef; 2],
        workspace: &MemoryReservation,
        empty: Option<encode::Base>,
    ) -> Result<OperatorStateSnapshot> {
        let dirty_epochs = self.next_v2_dirty_epoch()?;
        let delta = self.v2_pending_encoding(schemas, epoch, workspace)?;
        let deltas = self.v2_delta_entries(epoch, &delta)?;
        let snapshot =
            self.snapshot_with_v2_delta(epoch, self.current_v2_base(&empty), &delta, &deltas)?;
        self.install_v2_delta(empty, delta)?;
        self.v2_writer.deltas = deltas;
        self.v2_writer.dirty_epochs = dirty_epochs;
        Ok(snapshot)
    }

    fn v2_pending_encoding(
        &mut self,
        schemas: &[SchemaRef; 2],
        epoch: Epoch,
        workspace: &MemoryReservation,
    ) -> Result<encode::Base> {
        let cut = self
            .v2_writer
            .tracker
            .snapshot(self.state.last_checkpoint_epoch);
        if self
            .v2_writer
            .prepared_delta
            .as_ref()
            .is_some_and(|prepared| prepared.cut == cut)
        {
            let prepared = self
                .v2_writer
                .prepared_delta
                .take()
                .expect("matching dirty preparation");
            return encode::retarget::at_epoch(prepared.base, epoch);
        }
        encode::pending(
            &self.state.deltas.pending,
            schemas,
            epoch,
            workspace,
            &|| Ok(()),
        )
    }

    fn next_v2_dirty_epoch(&self) -> Result<u32> {
        self.v2_writer
            .dirty_epochs
            .checked_add(1)
            .ok_or_else(|| buffer::error("V2 dirty epoch counter overflow"))
    }

    fn install_v2_delta(&mut self, empty: Option<encode::Base>, delta: encode::Base) -> Result<()> {
        if let Some(mut empty) = empty {
            empty.merge(delta)?;
            self.v2_writer.base = Some(empty);
        } else {
            self.v2_writer
                .base
                .as_mut()
                .expect("one current base")
                .merge(delta)?;
        }
        Ok(())
    }

    fn v2_delta_entries(
        &self,
        epoch: Epoch,
        delta: &encode::Base,
    ) -> Result<Vec<metadata::DeltaEntry>> {
        let count = accounting::add(self.v2_writer.deltas.len(), 1)?;
        accounting::reserve(
            delta.credit(),
            sum(&[
                bulk_control_bytes::<metadata::DeltaEntry>(count)?,
                accounting::product(count, 4 * size_of::<&str>())?,
            ])?,
        )?;
        let mut deltas = self
            .v2_writer
            .deltas
            .iter()
            .map(|entry| metadata::DeltaEntry {
                epoch: entry.epoch,
                sides: entry.sides.clone(),
            })
            .collect::<Vec<_>>();
        let sides = [("left", "left-delta-"), ("right", "right-delta-")]
            .into_iter()
            .filter_map(|(side, prefix)| {
                delta
                    .segments
                    .keys()
                    .any(|name| name.starts_with(prefix))
                    .then_some(side)
            })
            .collect();
        deltas.push(metadata::DeltaEntry {
            epoch: epoch.as_u64(),
            sides,
        });
        Ok(deltas)
    }

    fn snapshot_with_v2_delta(
        &self,
        epoch: Epoch,
        base: &encode::Base,
        delta: &encode::Base,
        deltas: &[metadata::DeltaEntry],
    ) -> Result<OperatorStateSnapshot> {
        let count = accounting::add(base.payloads.len(), delta.payloads.len())?;
        let text = base
            .payloads
            .iter()
            .chain(&delta.payloads)
            .try_fold(0, |bytes, entry| accounting::add(bytes, entry.sha256.len()))?;
        accounting::reserve(
            delta.credit(),
            sum(&[bulk_control_bytes::<encode::PayloadEntry>(count)?, text])?,
        )?;
        let mut payloads = base
            .payloads
            .iter()
            .chain(&delta.payloads)
            .cloned()
            .collect::<Vec<_>>();
        encode::canonical_payloads(&mut payloads);
        let inline_metadata = metadata::encode(
            self,
            epoch,
            self.v2_writer.base_epoch,
            deltas,
            &payloads,
            delta.credit(),
        )?;
        let segments = clone_segments(base, Some(delta), delta.credit())?;
        Ok(OperatorStateSnapshot {
            inline_metadata,
            segments,
        })
    }
}

fn bulk_control_bytes<T>(count: usize) -> Result<usize> {
    accounting::product(accounting::product(count, 3)?.max(4), size_of::<T>())
}

fn clone_segments(
    base: &encode::Base,
    delta: Option<&encode::Base>,
    credit: &MemoryReservation,
) -> Result<std::collections::BTreeMap<String, crate::StateSegment>> {
    let count = accounting::add(
        base.segments.len(),
        delta.map_or(0, |delta| delta.segments.len()),
    )?;
    let entries = || {
        base.segments
            .iter()
            .chain(delta.into_iter().flat_map(|delta| delta.segments.iter()))
    };
    let strings = entries().try_fold(0, |bytes, (name, segment)| {
        sum(&[bytes, name.len(), segment.sha256().len()])
    })?;
    accounting::reserve(
        credit,
        sum(&[
            restore_budget::tree::<String, crate::StateSegment>(count)?,
            strings,
        ])?,
    )?;
    let mut segments = std::collections::BTreeMap::new();
    for (name, segment) in entries() {
        segments.insert(name.clone(), segment.clone());
    }
    Ok(segments)
}

fn writer_admission(failure: AdmissionFailure) -> crate::CalcFlowError {
    match failure {
        AdmissionFailure::Runtime(error) => error,
        AdmissionFailure::Budget { .. } => {
            buffer::error("V2 checkpoint native-work admission failed")
        }
    }
}

fn release_channel_bytes() -> Result<usize> {
    // Pinned Tokio oneshot Inner<()>: atomic state, optional value and two waker slots.
    accounting::arc::<(
        std::sync::atomic::AtomicUsize,
        Option<()>,
        [std::mem::MaybeUninit<std::task::Waker>; 2],
    )>()
}
