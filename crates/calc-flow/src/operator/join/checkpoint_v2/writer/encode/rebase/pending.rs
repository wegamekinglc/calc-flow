use std::sync::Arc;

use super::super::{SchemaRef, clone_pending_op, pending_owned};
use super::{
    Base, HistoryInput, IndexInput, MemoryReservation, PayloadInput, Result, StreamJoinOperator,
    StreamOperatorContext, accounting, error, funding, input,
};
use crate::Epoch;
use crate::operator::join::{PendingOp, checkpoint_v2::ContainerFunding, columnar::Quantum};

pub(in crate::operator::join::checkpoint_v2::writer) struct PendingInput {
    pub(in crate::operator::join::checkpoint_v2::writer) identity:
        super::super::super::HistoryIdentity,
    rows: Vec<PendingOp>,
    _containers: Option<Arc<ContainerFunding>>,
    funding: MemoryReservation,
}

impl PendingInput {
    fn allocate(
        operator: &StreamJoinOperator,
        count: usize,
        workspace: &MemoryReservation,
    ) -> Result<Self> {
        let credit = workspace.new_empty();
        accounting::reserve(
            &credit,
            accounting::sum(&[
                accounting::product(count, size_of::<PendingOp>())?,
                super::super::budget::registration_bytes()?,
            ])?,
        )?;
        let pending = Self {
            identity: operator.checkpoint_identity,
            rows: funding::vector(count)?,
            _containers: operator.v2_containers.clone(),
            funding: credit,
        };
        Ok(pending)
    }

    pub(in crate::operator::join::checkpoint_v2::writer) async fn copy(
        operator: &StreamJoinOperator,
        workspace: &MemoryReservation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Self> {
        context.check_cancelled()?;
        let mut quantum = Quantum::default();
        let count = pending_count(operator, &mut quantum, context).await?;
        let mut pending = Self::allocate(operator, count, workspace)?;
        for op in operator.state.deltas.pending.iter() {
            quantum.step(context, 1, size_of::<PendingOp>()).await?;
            pending.rows.push(clone_pending_op(op, &pending.funding)?);
        }
        context.check_cancelled()?;
        Ok(pending)
    }
}

async fn pending_count(
    operator: &StreamJoinOperator,
    quantum: &mut Quantum,
    context: &StreamOperatorContext<'_>,
) -> Result<usize> {
    let mut count = 0;
    for op in operator.state.deltas.pending.iter() {
        quantum.step(context, 1, 0).await?;
        if !matches!(op, PendingOp::Upsert { .. }) {
            return Err(error("V2 rebase pending input contains a tombstone"));
        }
        count = accounting::add(count, 1)?;
    }
    Ok(count)
}

pub(super) fn encode(
    history: HistoryInput,
    pending: PendingInput,
    schemas: &[SchemaRef; 2],
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<Base> {
    check()?;
    if pending.identity != history.identity {
        return Err(error("checkpoint pending layout differs from history"));
    }
    let dirty = pending_owned(&pending.rows, schemas, Epoch::INITIAL, workspace, check)?;
    drop(pending);
    check()?;
    let history = combine(history, &dirty, workspace, check)?;
    drop(dirty);
    check()?;
    super::encode(&history, workspace, check)
}

fn combine(
    history: HistoryInput,
    dirty: &Base,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<HistoryInput> {
    let mut combined = allocate_combined(history, dirty, workspace)?;
    for (name, segment) in &dirty.segments {
        check()?;
        append_segment(&mut combined, dirty, name, segment)?;
    }
    combined
        .payloads
        .sort_unstable_by_key(|payload| (payload.side, payload.digest));
    validate_payload_ids(&combined.payloads, check)?;
    check()?;
    Ok(combined)
}

fn combined_counts(history: &HistoryInput, dirty: &Base) -> Result<(usize, usize)> {
    let dirty_indexes = dirty
        .segments
        .len()
        .checked_sub(dirty.payloads.len())
        .ok_or_else(|| error("V2 rebase pending segment count overflow"))?;
    let indexes = accounting::add(history.indexes.len(), dirty_indexes)?;
    let payloads = accounting::add(history.payloads.len(), dirty.payloads.len())?;
    Ok((indexes, payloads))
}

fn allocate_combined(
    history: HistoryInput,
    dirty: &Base,
    workspace: &MemoryReservation,
) -> Result<HistoryInput> {
    let (indexes, payloads) = combined_counts(&history, dirty)?;
    accounting::reserve(workspace, funding::inputs(indexes, payloads)?)?;
    let mut combined = HistoryInput {
        identity: history.identity,
        indexes: funding::vector(indexes)?,
        payloads: funding::vector(payloads)?,
        expected: history.expected,
    };
    combined.indexes.extend(history.indexes);
    combined.payloads.extend(history.payloads);
    Ok(combined)
}

fn append_segment(
    combined: &mut HistoryInput,
    dirty: &Base,
    name: &str,
    segment: &super::StateSegment,
) -> Result<()> {
    match input::classify(name)? {
        input::Segment::Index(side, Some(epoch)) if epoch == Epoch::INITIAL.as_u64() => {
            combined.indexes.push(IndexInput {
                side,
                epoch: Some(epoch),
                pending: true,
                segment: segment.clone(),
            });
        }
        input::Segment::Payload(side, digest) => {
            let entry = input::payload_entry(dirty, side, digest)?;
            combined.payloads.push(PayloadInput {
                side,
                digest: super::super::super::super::geometry::checked(
                    super::super::super::super::inventory::digest(digest),
                )?,
                rows: usize::try_from(entry.rows)
                    .map_err(|_| error("V2 rebase pending payload row count overflow"))?,
                segment: segment.clone(),
            });
        }
        input::Segment::Index(_, _) => {
            return Err(error("V2 rebase pending index epoch differs"));
        }
    }
    Ok(())
}

fn validate_payload_ids(payloads: &[PayloadInput], check: &dyn Fn() -> Result<()>) -> Result<()> {
    for pair in payloads.windows(2) {
        check()?;
        if (pair[0].side, pair[0].digest) == (pair[1].side, pair[1].digest) {
            return Err(error("V2 rebase payload identity is duplicated"));
        }
    }
    Ok(())
}
