//! Copy shared right columns on blocking workers before the commit boundary.
//! The reservation owns both immutable inputs and replacements until release.

use super::{
    StreamAsofJoinOperator, checked,
    state::{BatchKey, Encoding, PayloadRemoval, PreparedPayloadRemoval, RightBucket, RightState},
};
use crate::{CalcFlowError, Result, StreamOperatorContext};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::Arc;

type Buckets = Vec<(u32, Arc<RightBucket>)>;

pub(super) struct PreparedRightCopies {
    originals: Buckets,
    replacements: Buckets,
    workspace: Option<MemoryReservation>,
}

impl PreparedRightCopies {
    pub fn install(mut self, state: &mut RightState) {
        for (id, bucket) in self.replacements.drain(..) {
            // `originals` keeps the previous allocation alive, so replacing
            // the state pointer cannot drop a large column on the executor.
            state.install_prepared_bucket(id, bucket);
        }
    }
}

impl Drop for PreparedRightCopies {
    fn drop(&mut self) {
        let originals = std::mem::take(&mut self.originals);
        let replacements = std::mem::take(&mut self.replacements);
        let workspace = self.workspace.take();
        if originals.is_empty() && replacements.is_empty() {
            return;
        }
        let release = move || drop((originals, replacements, workspace));
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn_blocking(release);
        } else {
            release();
        }
    }
}

impl StreamAsofJoinOperator {
    pub(super) async fn prepare_pool_compaction(
        &self,
        removals: &std::collections::BTreeMap<BatchKey, usize>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedPayloadRemoval> {
        let layout = self.state.batches.project_remove(removals, &self.name)?;
        if !layout.replace {
            return Ok(PreparedPayloadRemoval::empty());
        }
        let bytes = self
            .pool_compaction_workspace(removals, layout.metadata_bytes, layout.remaining, context)
            .await?;
        let workspace = self.reserve_workspace(bytes)?;
        let prepared = self
            .capture_pool_inputs(removals, &layout, workspace, context)
            .await?;
        populate_pool(prepared, layout, context).await
    }

    async fn capture_pool_inputs(
        &self,
        removals: &std::collections::BTreeMap<BatchKey, usize>,
        layout: &PayloadRemoval,
        workspace: MemoryReservation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedPayloadRemoval> {
        let mut prepared = PreparedPayloadRemoval::capture(&self.state.batches, layout, workspace);
        for (ordinal, (id, batch, references)) in
            self.state.batches.compaction_entries(removals).enumerate()
        {
            cooperate(ordinal, context).await?;
            prepared.retain(id, batch, references);
        }
        Ok(prepared)
    }

    async fn pool_compaction_workspace(
        &self,
        removals: &std::collections::BTreeMap<BatchKey, usize>,
        replacement_bytes: u64,
        count: usize,
        context: &StreamOperatorContext<'_>,
    ) -> Result<u64> {
        let descriptors = count
            .checked_mul(size_of::<(u32, Arc<super::state::PayloadBatch>, usize)>())
            .ok_or_else(|| {
                super::reason(
                    &self.name,
                    crate::StreamingFailureReason::AsofCounterOverflow,
                    "ASOF pool copy workspace overflowed",
                )
            })? as u64;
        let mut bytes = checked(
            &self.name,
            self.state.batches.metadata_bytes(),
            checked(&self.name, replacement_bytes, descriptors + 512)?,
        )?;
        for (ordinal, (_, batch, _)) in self.state.batches.compaction_entries(removals).enumerate()
        {
            cooperate(ordinal, context).await?;
            bytes = checked(
                &self.name,
                bytes,
                super::state::capacity_batch_allocation(batch, &self.name)?,
            )?;
        }
        Ok(bytes)
    }

    pub(super) async fn prepare_right_admission_copies(
        &self,
        additions: &[(Encoding, usize)],
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedRightCopies> {
        self.prepare_right_copies(
            self.state.right.shared_admission_buckets(additions),
            context,
        )
        .await
    }

    pub(super) async fn prepare_right_eviction_copies(
        &self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedRightCopies> {
        let threshold = super::state::retention_threshold(&self.state, &self.status);
        self.prepare_right_copies(
            self.state.right.shared_eviction_buckets(
                &self.status,
                self.spec.tolerance_micros(),
                threshold,
            ),
            context,
        )
        .await
    }

    async fn prepare_right_copies<'a>(
        &self,
        selected: impl Iterator<Item = (u32, &'a Arc<RightBucket>)> + Clone,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedRightCopies> {
        let (count, bytes) = copy_extent(selected.clone(), &self.name)?;
        if count == 0 {
            return Ok(PreparedRightCopies {
                originals: Vec::new(),
                replacements: Vec::new(),
                workspace: None,
            });
        }
        let buffers = self
            .right_copy_buffer_bytes(selected.clone(), context)
            .await?;
        let workspace = self.reserve_workspace(checked(&self.name, bytes, buffers)?)?;
        let originals = capture_right_buckets(selected, count, context).await?;
        copy_right_buckets(originals, workspace, context).await
    }

    fn right_copy_scratch_bytes<'a>(
        &self,
        mut selected: impl Iterator<Item = (u32, &'a Arc<RightBucket>)>,
    ) -> Result<u64> {
        let rows = selected.try_fold(0, |rows, (_, bucket)| {
            checked(&self.name, rows, bucket.len() as u64)
        })?;
        let selected_metadata = rows
            .checked_mul(128)
            .and_then(|bytes| bytes.checked_add(384))
            .ok_or_else(|| {
                super::reason(
                    &self.name,
                    crate::StreamingFailureReason::AsofCounterOverflow,
                    "ASOF selected owner scratch overflowed",
                )
            })?;
        checked(
            &self.name,
            self.state
                .encoding_owner_allocation()
                .1
                .min(selected_metadata),
            256,
        )
    }

    async fn right_copy_buffer_bytes<'a>(
        &self,
        selected: impl Iterator<Item = (u32, &'a Arc<RightBucket>)> + Clone,
        context: &StreamOperatorContext<'_>,
    ) -> Result<u64> {
        // The global owner count bounds address-set scratch, but unrelated
        // buffers are not retained by a right-column copy and need no lease.
        let _scratch = self.reserve_workspace(self.right_copy_scratch_bytes(selected.clone())?)?;
        let mut owners = std::collections::BTreeSet::new();
        let mut bytes = 0;
        let mut ordinal = 0;
        for (_, bucket) in selected {
            for ((_, sequence), _) in bucket.as_ref() {
                cooperate(ordinal, context).await?;
                ordinal += 1;
                if let Some((address, retained)) = sequence.as_ref().allocation() {
                    if owners.insert(address) {
                        bytes = checked(&self.name, bytes, retained)?;
                    }
                }
            }
        }
        Ok(bytes)
    }
}

async fn populate_pool(
    mut prepared: PreparedPayloadRemoval,
    layout: PayloadRemoval,
    context: &StreamOperatorContext<'_>,
) -> Result<PreparedPayloadRemoval> {
    context.check_cancelled()?;
    let worker = tokio::task::spawn_blocking(move || {
        prepared.populate(&layout);
        prepared
    });
    let prepared = tokio::select! {
        result = worker => result.map_err(|error| CalcFlowError::Internal { message: format!("ASOF pool compaction task failed: {error}") })?,
        () = context.job().cancellation().cancelled() => {
        context.check_cancelled()?;
        unreachable!("cancelled ASOF pool compaction")
        }
    };
    context.check_cancelled()?;
    Ok(prepared)
}

async fn capture_right_buckets<'a>(
    selected: impl Iterator<Item = (u32, &'a Arc<RightBucket>)>,
    count: usize,
    context: &StreamOperatorContext<'_>,
) -> Result<Buckets> {
    let mut originals = Vec::with_capacity(count);
    for (ordinal, (id, bucket)) in selected.enumerate() {
        if ordinal % 256 == 0 {
            context.check_cancelled()?;
        }
        if ordinal > 0 && ordinal % 8_192 == 0 {
            tokio::task::yield_now().await;
        }
        originals.push((id, bucket.clone()));
    }
    Ok(originals)
}

async fn copy_right_buckets(
    originals: Buckets,
    workspace: MemoryReservation,
    context: &StreamOperatorContext<'_>,
) -> Result<PreparedRightCopies> {
    let cancellation = context.job().cancellation().clone();
    let run_id = context.job().job_id();
    let deadline = context.job().deadline().copied();
    context.check_cancelled()?;
    let worker = copy_worker(originals, workspace, move || {
        if cancellation.is_cancelled()
            || deadline.is_some_and(|deadline| chrono::Utc::now() >= deadline)
        {
            Err(CalcFlowError::Cancelled {
                run_id: run_id.to_string(),
            })
        } else {
            Ok(())
        }
    });
    let prepared = tokio::select! {
        result = worker => result?,
        () = context.job().cancellation().cancelled() => {
        context.check_cancelled()?;
        unreachable!("cancelled ASOF right column copy")
        }
    };
    context.check_cancelled()?;
    Ok(prepared)
}

async fn cooperate(ordinal: usize, context: &StreamOperatorContext<'_>) -> Result<()> {
    if ordinal.is_multiple_of(256) {
        context.check_cancelled()?;
    }
    if ordinal > 0 && ordinal.is_multiple_of(8_192) {
        tokio::task::yield_now().await;
    }
    Ok(())
}

fn copy_extent<'a>(
    mut selected: impl Iterator<Item = (u32, &'a Arc<RightBucket>)>,
    name: &str,
) -> Result<(usize, u64)> {
    selected.try_fold((0, 256), |(count, bytes), (_, bucket)| {
        let retained = checked(
            name,
            bucket.metadata_bytes(),
            (size_of::<RightBucket>() + 2 * size_of::<usize>()) as u64,
        )?;
        let columns = retained.checked_mul(2).ok_or_else(|| {
            super::reason(
                name,
                crate::StreamingFailureReason::AsofCounterOverflow,
                "ASOF right copy workspace overflowed",
            )
        })?;
        let descriptors = (2 * size_of::<(u32, Arc<RightBucket>)>()) as u64;
        Ok((
            count + 1,
            checked(name, bytes, checked(name, columns, descriptors)?)?,
        ))
    })
}

async fn copy_worker(
    originals: Buckets,
    workspace: MemoryReservation,
    check: impl Fn() -> Result<()> + Send + 'static,
) -> Result<PreparedRightCopies> {
    tokio::task::spawn_blocking(move || {
        let mut prepared = PreparedRightCopies {
            replacements: Vec::with_capacity(originals.len()),
            originals,
            workspace: Some(workspace),
        };
        copy_columns(&mut prepared, check)?;
        Ok(prepared)
    })
    .await
    .map_err(|error| CalcFlowError::Internal {
        message: format!("ASOF right copy task failed: {error}"),
    })?
}

fn copy_columns(prepared: &mut PreparedRightCopies, check: impl Fn() -> Result<()>) -> Result<()> {
    for (id, bucket) in &prepared.originals {
        check()?;
        prepared
            .replacements
            .push((*id, Arc::new(bucket.as_ref().clone())));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::asof::state::SequenceKind;
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};

    fn bucket(kind: SequenceKind) -> Arc<RightBucket> {
        let mut bucket = RightBucket::with_sequence_kind(kind);
        let sequence = kind.width().map_or_else(
            || Encoding::from_slice(b"a shared generic sequence"),
            |width| kind.decode_integer(&[0; 8][..width]),
        );
        for time in 0..8_192 {
            bucket.insert((time, sequence.clone()), None);
        }
        Arc::new(bucket)
    }

    #[tokio::test]
    async fn selected_right_copy_does_not_fund_unrelated_encoding_buffers() {
        use crate::{
            AsofJoinSide, AsofStateLimits, CancellationToken, JsonMap, StreamAsofJoinSpec,
            StreamJobContext,
        };
        use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Utf8, false),
        ]));
        let side = |prefix: &str| {
            AsofJoinSide::new(
                vec!["key".into()],
                "time".into(),
                vec!["seq".into()],
                prefix.into(),
            )
            .unwrap()
        };
        let spec = StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            std::time::Duration::ZERO,
            AsofStateLimits::new(10_000, 8 << 20).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamAsofJoinOperator::new("asof", schema.clone(), schema, spec).unwrap();
        let source = bucket(SequenceKind::Canonical);
        let unrelated = Encoding::from_slice(&vec![7; 1 << 20]);
        operator
            .state
            .right
            .insert(Encoding::from_slice(b"selected"), source.as_ref().clone());
        let mut untouched = RightBucket::with_sequence_kind(SequenceKind::Canonical);
        untouched.insert((0, Encoding::from_slice(b"untouched")), None);
        operator.state.right.insert(unrelated, untouched);
        operator.state.rebuild_encoding_owners();
        let (_, columns) = copy_extent([(0, &source)].into_iter(), "asof").unwrap();
        operator.runtime.pool = Arc::new(GreedyMemoryPool::new(columns as usize + 4_096));
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "asof", None);
        let prepared = operator
            .prepare_right_copies([(0, &source)].into_iter(), &context)
            .await
            .unwrap();
        assert_eq!(
            prepared.replacements[0].1.checkpoint_capacities(),
            source.checkpoint_capacities()
        );
        drop(prepared);
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while operator.runtime.pool.reserved() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[test]
    fn right_copy_allocation_peak_fits_its_derived_reservation() {
        for kind in [
            SequenceKind::Canonical,
            SequenceKind::Signed(1),
            SequenceKind::Unsigned(8),
        ] {
            let source = bucket(kind);
            let (_, bytes) = copy_extent([(0, &source)].into_iter(), "asof").unwrap();
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(bytes as usize));
            let workspace = MemoryConsumer::new("right-copy").register(&pool);
            workspace.try_grow(bytes as usize).unwrap();
            let mut prepared = PreparedRightCopies {
                originals: vec![(0, source)],
                replacements: Vec::with_capacity(1),
                workspace: Some(workspace),
            };
            let allocations =
                allocation_counter::measure(|| copy_columns(&mut prepared, || Ok(())).unwrap());
            assert!(
                allocations.bytes_max <= bytes,
                "{kind:?}: {allocations:?}, reserved={bytes}"
            );
            assert_eq!(
                prepared.replacements[0].1.checkpoint_capacities(),
                prepared.originals[0].1.checkpoint_capacities()
            );
            drop(prepared);
            assert_eq!(pool.reserved(), 0);
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dropped_right_copy_keeps_its_column_lease_until_worker_exit() {
        let source = bucket(SequenceKind::Signed(8));
        let weak = Arc::downgrade(&source);
        let (_, bytes) = copy_extent([(0, &source)].into_iter(), "asof").unwrap();
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(bytes as usize));
        let workspace = MemoryConsumer::new("right-copy").register(&pool);
        workspace.try_grow(bytes as usize).unwrap();
        let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let worker_gate = gate.clone();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let mut operation = Box::pin(copy_worker(
            vec![(0, source.clone())],
            workspace,
            move || {
                started_tx.send(()).unwrap();
                let (flag, changed) = worker_gate.as_ref();
                let mut released = flag.lock().unwrap();
                while !*released {
                    released = changed.wait(released).unwrap();
                }
                Ok(())
            },
        ));
        assert!(futures::poll!(operation.as_mut()).is_pending());
        started_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        drop(operation);
        drop(source);
        assert!(weak.upgrade().is_some());
        assert_eq!(pool.reserved(), bytes as usize);
        let (flag, changed) = gate.as_ref();
        *flag.lock().unwrap() = true;
        changed.notify_all();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while pool.reserved() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(weak.upgrade().is_none());
    }
}
