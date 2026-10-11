use std::sync::Arc;

use super::super::{SchemaRef, clone_pending_op, pending_owned};
use super::{
    Base, HistoryInput, IndexInput, MemoryReservation, PayloadInput, Result, StreamJoinOperator,
    StreamOperatorContext, accounting, error, funding, input,
};
use crate::Epoch;
use crate::operator::join::{PendingOp, checkpoint_v2::ContainerFunding, columnar::Quantum};

pub(in crate::operator::join::checkpoint_v2::writer) struct PendingInput {
    rows: Vec<PendingOp>,
    _containers: Option<Arc<ContainerFunding>>,
    funding: MemoryReservation,
}

impl PendingInput {
    pub(in crate::operator::join::checkpoint_v2::writer) async fn copy(
        operator: &StreamJoinOperator,
        workspace: &MemoryReservation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Self> {
        context.check_cancelled()?;
        let mut quantum = Quantum::default();
        let mut count = 0;
        for op in operator.state.deltas.pending.iter() {
            quantum.step(context, 1, 0).await?;
            if !matches!(op, PendingOp::Upsert { .. }) {
                return Err(error("V2 rebase pending input contains a tombstone"));
            }
            count = accounting::add(count, 1)?;
        }
        let credit = workspace.new_empty();
        accounting::reserve(
            &credit,
            accounting::sum(&[
                accounting::product(count, size_of::<PendingOp>())?,
                super::super::budget::registration_bytes()?,
            ])?,
        )?;
        let mut pending = Self {
            rows: funding::vector(count)?,
            _containers: operator.v2_containers.clone(),
            funding: credit,
        };
        for op in operator.state.deltas.pending.iter() {
            quantum.step(context, 1, size_of::<PendingOp>()).await?;
            pending.rows.push(clone_pending_op(op, &pending.funding)?);
        }
        context.check_cancelled()?;
        Ok(pending)
    }
}

pub(super) fn encode(
    history: HistoryInput,
    pending: PendingInput,
    schemas: &[SchemaRef; 2],
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<Base> {
    check()?;
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
    let dirty_indexes = dirty
        .segments
        .len()
        .checked_sub(dirty.payloads.len())
        .ok_or_else(|| error("V2 rebase pending segment count overflow"))?;
    let indexes = accounting::add(history.indexes.len(), dirty_indexes)?;
    let payloads = accounting::add(history.payloads.len(), dirty.payloads.len())?;
    accounting::reserve(workspace, funding::inputs(indexes, payloads)?)?;
    let mut combined = HistoryInput {
        indexes: funding::vector(indexes)?,
        payloads: funding::vector(payloads)?,
        expected: history.expected,
    };
    combined.indexes.extend(history.indexes);
    combined.payloads.extend(history.payloads);
    for (name, segment) in &dirty.segments {
        check()?;
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
    }
    combined
        .payloads
        .sort_unstable_by_key(|payload| (payload.side, payload.digest));
    for pair in combined.payloads.windows(2) {
        check()?;
        if (pair[0].side, pair[0].digest) == (pair[1].side, pair[1].digest) {
            return Err(error("V2 rebase payload identity is duplicated"));
        }
    }
    check()?;
    Ok(combined)
}
