use std::sync::Arc;

use super::super::delta::SideDelta;
use super::super::{SchemaRef, copy_delta, dirty};
use super::{
    Base, HistoryInput, IndexInput, MemoryReservation, PayloadInput, Result, StreamJoinOperator,
    StreamOperatorContext, accounting, error, funding, input,
};
use crate::Epoch;
use crate::operator::join::{checkpoint_v2::ContainerFunding, columnar::Quantum};

/// The derived dirty window of a rebase-eligible operator (spec FR7): the
/// lists are empty by eligibility, so the input is the shared row state plus
/// its watermarks and O(1) upsert counts.
pub(in crate::operator::join::checkpoint_v2::writer) struct PendingInput {
    pub(in crate::operator::join::checkpoint_v2::writer) identity:
        super::super::super::HistoryIdentity,
    delta: super::super::CopiedDelta,
    rows: [Arc<crate::operator::join::StateRows>; 2],
    _containers: Option<Arc<ContainerFunding>>,
    _funding: MemoryReservation,
}

impl PendingInput {
    fn allocate(operator: &StreamJoinOperator, workspace: &MemoryReservation) -> Result<Self> {
        let credit = workspace.new_empty();
        accounting::reserve(
            &credit,
            accounting::sum(&[
                accounting::product(2, size_of::<Arc<crate::operator::join::StateRows>>())?,
                super::super::budget::registration_bytes()?,
            ])?,
        )?;
        let pending = Self {
            identity: operator.checkpoint_identity,
            delta: copy_delta(&operator.state.deltas, &credit, &|| {
                // Synchronous copy: the cooperative steps happen per entry.
                Ok(())
            })?,
            rows: [
                Arc::clone(&operator.state.left.0),
                Arc::clone(&operator.state.right.0),
            ],
            _containers: operator.v2_containers.clone(),
            _funding: credit,
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
        let entries: usize = operator
            .state
            .deltas
            .tombstones
            .iter()
            .map(crate::operator::join::Tombstones::len)
            .sum();
        for _ in 0..entries {
            quantum.step(context, 1, 0).await?;
        }
        let pending = Self::allocate(operator, workspace)?;
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
    if pending.identity != history.identity {
        return Err(error("checkpoint pending layout differs from history"));
    }
    let rows = pending
        .rows
        .each_ref()
        .map(|rows| rows.as_ref() as &dyn crate::operator::join::Rows);
    let sides = [
        SideDelta {
            rows: rows[0],
            watermark: pending.delta.watermark[0],
            upserts: pending.delta.upserts[0],
            tombstones: &pending.delta.tombstones[0],
        },
        SideDelta {
            rows: rows[1],
            watermark: pending.delta.watermark[1],
            upserts: pending.delta.upserts[1],
            tombstones: &pending.delta.tombstones[1],
        },
    ];
    let dirty = dirty(&sides, schemas, Epoch::INITIAL, workspace, check)?;
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
