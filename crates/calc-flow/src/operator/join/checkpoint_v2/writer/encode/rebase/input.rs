use super::super::super::super::{geometry::checked, inventory};
use super::{
    Base, HistoryInput, IndexInput, JoinSide, MemoryReservation, PayloadEntry, PayloadInput,
    Result, SideExpected, StateSegment, StreamJoinOperator, StreamOperatorContext, accounting,
    error, frame, funding,
};
use crate::operator::join::columnar::Quantum;

pub(super) enum Segment<'a> {
    Index(JoinSide, Option<u64>),
    Payload(JoinSide, &'a str),
}

pub(super) async fn eligible(
    operator: &StreamJoinOperator,
    context: &StreamOperatorContext<'_>,
) -> Result<bool> {
    context.check_cancelled()?;
    let Some((base, epoch)) = eligible_base(operator, context) else {
        return Ok(false);
    };
    let mut quantum = Quantum::default();
    // Rebase combines only upserts: any recorded tombstone disqualifies the
    // window (spec FR7), exactly as a tombstone-bearing pending log did.
    if operator
        .state
        .deltas
        .tombstones
        .iter()
        .any(|list| !list.is_empty())
    {
        return Ok(false);
    }
    validate_inventory(operator, base, epoch, &mut quantum, context).await?;
    let eligible = validate_segments(operator, base, &mut quantum, context).await?;
    context.check_cancelled()?;
    Ok(eligible)
}

fn eligible_base<'a>(
    operator: &'a StreamJoinOperator,
    context: &StreamOperatorContext<'_>,
) -> Option<(&'a Base, u64)> {
    let writer = &operator.v2_writer;
    let epoch = operator.state.last_checkpoint_epoch?;
    let base = writer.base.as_ref()?;
    if !context.job().checkpointing()
        || !writer.history_trusted
        || writer.migration_required
        || !operator.v2_history_matches()
    {
        return None;
    }
    Some((base, epoch.as_u64()))
}

async fn validate_inventory(
    operator: &StreamJoinOperator,
    base: &Base,
    epoch: u64,
    quantum: &mut Quantum,
    context: &StreamOperatorContext<'_>,
) -> Result<()> {
    let writer = &operator.v2_writer;
    let indexes = validate_lists(operator, base, epoch, quantum, context).await?;
    if base.segments.len() != accounting::add(indexes, base.payloads.len())?
        || !base.segments.contains_key("left-base")
        || !base.segments.contains_key("right-base")
        || writer.base_epoch > epoch
    {
        return Err(error("V2 rebase segment inventory differs"));
    }
    Ok(())
}

async fn validate_segments(
    operator: &StreamJoinOperator,
    base: &Base,
    quantum: &mut Quantum,
    context: &StreamOperatorContext<'_>,
) -> Result<bool> {
    let mut eligible = true;
    for (name, segment) in &base.segments {
        quantum.step(context, 1, 64).await?;
        eligible &= validate_segment(operator, base, name, segment)?;
    }
    Ok(eligible)
}

fn validate_segment(
    operator: &StreamJoinOperator,
    base: &Base,
    name: &str,
    segment: &StateSegment,
) -> Result<bool> {
    checked(inventory::digest(segment.sha256()))?;
    if !segment.has_owner() {
        return Err(error("V2 rebase segment has no backing owner"));
    }
    match classify(name)? {
        Segment::Index(side, epoch) => {
            return validate_index(operator, segment, side, epoch);
        }
        Segment::Payload(side, digest) => {
            validate_payload(base, segment, side, digest)?;
        }
    }
    Ok(true)
}

fn validate_index(
    operator: &StreamJoinOperator,
    segment: &StateSegment,
    side: JoinSide,
    epoch: Option<u64>,
) -> Result<bool> {
    let index = index(segment, side, epoch)?;
    if let Some(epoch) = epoch {
        validate_delta(operator, side, epoch)?;
    } else if operator.v2_writer.base_epoch == 0 && index.upserts != 0 {
        return Err(error("V2 rebase epoch-zero base must be empty"));
    }
    Ok(index.tombstones == 0)
}

async fn validate_lists(
    operator: &StreamJoinOperator,
    base: &Base,
    captured_epoch: u64,
    quantum: &mut Quantum,
    context: &StreamOperatorContext<'_>,
) -> Result<usize> {
    let writer = &operator.v2_writer;
    let mut previous = writer.base_epoch;
    let mut indexes = 2;
    for delta in &writer.deltas {
        quantum.step(context, 1, 0).await?;
        if delta.epoch <= previous
            || delta.epoch > captured_epoch
            || !matches!(
                delta.sides.as_slice(),
                ["left" | "right"] | ["left", "right"]
            )
        {
            return Err(error("V2 rebase delta inventory is inconsistent"));
        }
        previous = delta.epoch;
        indexes = accounting::add(indexes, delta.sides.len())?;
    }
    validate_payload_list(base, quantum, context).await?;
    Ok(indexes)
}

async fn validate_payload_list(
    base: &Base,
    quantum: &mut Quantum,
    context: &StreamOperatorContext<'_>,
) -> Result<()> {
    let mut previous = None;
    for payload in &base.payloads {
        quantum.step(context, 1, 64).await?;
        let side = side(payload.side)?;
        let digest = checked(inventory::digest(&payload.sha256))?;
        if payload.rows == 0 || previous.is_some_and(|previous| (side, digest) <= previous) {
            return Err(error("V2 rebase payload inventory is inconsistent"));
        }
        previous = Some((side, digest));
    }
    Ok(())
}

fn validate_delta(operator: &StreamJoinOperator, side: JoinSide, epoch: u64) -> Result<()> {
    let deltas = &operator.v2_writer.deltas;
    let index = deltas
        .binary_search_by_key(&epoch, |delta| delta.epoch)
        .map_err(|_| error("V2 rebase delta is missing from inventory"))?;
    if !deltas[index].sides.contains(&side.as_str()) {
        return Err(error("V2 rebase delta side is missing from inventory"));
    }
    Ok(())
}

pub(super) fn payload_entry<'a>(
    base: &'a Base,
    side: JoinSide,
    digest: &str,
) -> Result<&'a PayloadEntry> {
    let index = base
        .payloads
        .binary_search_by(|entry| (entry.side, entry.sha256.as_str()).cmp(&(side.as_str(), digest)))
        .map_err(|_| error("V2 rebase payload is missing from inventory"))?;
    Ok(&base.payloads[index])
}

fn validate_payload(
    base: &Base,
    segment: &StateSegment,
    side: JoinSide,
    digest: &str,
) -> Result<()> {
    checked(inventory::digest(digest))?;
    let payload = payload_entry(base, side, digest)?;
    let frame = checked(frame::Payload::decode(segment.bytes(), side))?;
    if segment.sha256() != digest
        || usize::try_from(payload.rows).ok() != Some(frame.rows)
        || usize::try_from(payload.bytes).ok() != Some(segment.bytes().len())
    {
        return Err(error("V2 rebase payload frame differs from inventory"));
    }
    Ok(())
}

pub(super) async fn copy(
    operator: &StreamJoinOperator,
    workspace: &MemoryReservation,
    context: &StreamOperatorContext<'_>,
) -> Result<HistoryInput> {
    context.check_cancelled()?;
    let base = operator
        .v2_writer
        .base
        .as_ref()
        .expect("eligible rebase Base");
    let mut history = allocate_history(operator, base, workspace)?;
    let mut quantum = Quantum::default();
    for (name, segment) in &base.segments {
        quantum.step(context, 1, 64).await?;
        copy_segment(&mut history, base, name, segment)?;
    }
    context.check_cancelled()?;
    Ok(history)
}

fn allocate_history(
    operator: &StreamJoinOperator,
    base: &Base,
    workspace: &MemoryReservation,
) -> Result<HistoryInput> {
    let indexes = base
        .segments
        .len()
        .checked_sub(base.payloads.len())
        .ok_or_else(|| error("V2 rebase input count overflow"))?;
    let expected = expectations(operator)?;
    accounting::reserve(workspace, funding::inputs(indexes, base.payloads.len())?)?;
    let history = HistoryInput {
        identity: operator.checkpoint_identity,
        indexes: funding::vector(indexes)?,
        payloads: funding::vector(base.payloads.len())?,
        expected,
    };
    Ok(history)
}

fn copy_segment(
    history: &mut HistoryInput,
    base: &Base,
    name: &str,
    segment: &StateSegment,
) -> Result<()> {
    match classify(name)? {
        Segment::Index(side, epoch) => history.indexes.push(IndexInput {
            side,
            epoch,
            pending: false,
            segment: segment.clone(),
        }),
        Segment::Payload(side, digest) => {
            let entry = payload_entry(base, side, digest)?;
            history.payloads.push(PayloadInput {
                side,
                digest: checked(inventory::digest(digest))?,
                rows: usize::try_from(entry.rows)
                    .map_err(|_| error("V2 rebase input row count overflow"))?,
                segment: segment.clone(),
            });
        }
    }
    Ok(())
}

fn expectations(operator: &StreamJoinOperator) -> Result<[SideExpected; 2]> {
    let state = &operator.state;
    for (rows, metrics) in [
        (&state.left, &state.metrics.left),
        (&state.right, &state.metrics.right),
    ] {
        if u64::try_from(rows.len()).ok() != Some(metrics.retained_rows) {
            return Err(error("V2 rebase live row count differs from metrics"));
        }
    }
    Ok([
        SideExpected {
            rows: state.left.len(),
            charge: state.metrics.left.retained_bytes,
            next_row_id: state.next_left_row_id,
        },
        SideExpected {
            rows: state.right.len(),
            charge: state.metrics.right.retained_bytes,
            next_row_id: state.next_right_row_id,
        },
    ])
}

pub(super) fn index(
    segment: &StateSegment,
    side: JoinSide,
    epoch: Option<u64>,
) -> Result<frame::Index<'_>> {
    checked(match epoch {
        Some(epoch) => frame::Index::delta(segment.bytes(), side, epoch),
        None => frame::Index::base(segment.bytes(), side),
    })
}

pub(super) fn classify(name: &str) -> Result<Segment<'_>> {
    if name.len() > "right-payload-".len() + 64 {
        return Err(error("V2 rebase segment name is too long"));
    }
    let (side, suffix) = checked(inventory::split_side(name))?;
    if suffix == "base" {
        return Ok(Segment::Index(side, None));
    }
    if let Some(digest) = suffix.strip_prefix("payload-") {
        return Ok(Segment::Payload(side, digest));
    }
    if let Some(epoch) = suffix.strip_prefix("delta-") {
        let epoch = canonical_epoch(epoch)?;
        return Ok(Segment::Index(side, Some(epoch)));
    }
    Err(error("V2 rebase segment name is invalid"))
}

fn canonical_epoch(epoch: &str) -> Result<u64> {
    if epoch.starts_with('0') || !epoch.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(error("V2 rebase delta epoch is not canonical"));
    }
    let epoch = epoch
        .parse::<u64>()
        .map_err(|_| error("V2 rebase delta epoch is invalid"))?;
    Ok(epoch)
}

fn side(side: &str) -> Result<JoinSide> {
    match side {
        "left" => Ok(JoinSide::Left),
        "right" => Ok(JoinSide::Right),
        _ => Err(error("V2 rebase payload side is invalid")),
    }
}
