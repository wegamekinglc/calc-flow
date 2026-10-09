use super::super::super::super::PendingOp;
use super::*;

const DELTA_MAGIC: &[u8; 8] = b"CFJDIX2\0";

pub(super) fn encode(
    pending: &[PendingOp],
    schemas: &[SchemaRef; 2],
    epoch: crate::Epoch,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<Base> {
    let mut encoder = Encoder::new(workspace)?;
    for (index, side) in [JoinSide::Left, JoinSide::Right].into_iter().enumerate() {
        check()?;
        encode_side(
            &mut encoder,
            pending,
            Arc::clone(&schemas[index]),
            side,
            epoch,
            workspace,
            check,
        )?;
    }
    check()?;
    Ok(encoder.output)
}

fn encode_side(
    encoder: &mut Encoder,
    pending: &[PendingOp],
    schema: SchemaRef,
    side: JoinSide,
    epoch: crate::Epoch,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    let (rows, tombstones) = pending_rows(pending, side, workspace, check)?;
    if rows.is_empty() && tombstones.is_empty() {
        return Ok(());
    }
    let ordered = ordered_rows(&rows, workspace)?;
    let digest = if ordered.is_empty() {
        [0; 32]
    } else {
        encoder.payload(&ordered, schema, side, workspace, check)?
    };
    let bytes = index(&ordered, &tombstones, side, epoch, &digest, encoder, check)?;
    install_delta(encoder, side, epoch, bytes)
}

fn pending_rows<'a>(
    pending: &'a [PendingOp],
    side: JoinSide,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<(Vec<StoredRow>, Vec<&'a PendingOp>)> {
    let rows = upserts(pending, side, workspace, check)?;
    let tombstones = removals(pending, side, workspace, check)?;
    Ok((rows, tombstones))
}

fn install_delta(
    encoder: &mut Encoder,
    side: JoinSide,
    epoch: crate::Epoch,
    bytes: Vec<u8>,
) -> Result<()> {
    // At most 32 name bytes; 3L covers the formatter's old/new backing overlap.
    ensure(&encoder.funding, add(encoder.retained, 96)?)?;
    encoder.install(format!("{}-delta-{}", side.as_str(), epoch.as_u64()), bytes)
}

fn upserts(
    pending: &[PendingOp],
    side: JoinSide,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<StoredRow>> {
    let count = count_ops(pending, side, true, check)?;
    accounting::reserve(workspace, bulk_bytes::<StoredRow>(count)?)?;
    let mut rows = Vec::with_capacity(count);
    for op in pending.iter() {
        check()?;
        if let PendingOp::Upsert {
            side: actual,
            row_id,
            event_time,
            encoded_key,
            record,
            charge,
        } = op
        {
            if *actual == side {
                admit_payload_clone(record, workspace)?;
                rows.push(StoredRow {
                    record: record.clone(),
                    event_time: *event_time,
                    row_id: *row_id,
                    charge: *charge,
                    encoded_key: Arc::clone(encoded_key),
                });
            }
        }
    }
    rows.sort_unstable_by_key(|row| row.row_id);
    Ok(rows)
}

fn removals<'a>(
    pending: &'a [PendingOp],
    side: JoinSide,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<&'a PendingOp>> {
    let count = count_ops(pending, side, false, check)?;
    accounting::reserve(workspace, bulk_bytes::<&PendingOp>(count)?)?;
    let mut rows = Vec::with_capacity(count);
    for op in pending.iter() {
        check()?;
        if matches!(op, PendingOp::Tombstone { side: actual, .. } if *actual == side) {
            rows.push(op);
        }
    }
    rows.sort_unstable_by_key(|op| op.identity().1);
    Ok(rows)
}

fn count_ops(
    pending: &[PendingOp],
    side: JoinSide,
    upsert: bool,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    pending.iter().try_fold(0, |count, op| {
        check()?;
        let selected = op.identity().0 == side && matches!(op, PendingOp::Upsert { .. }) == upsert;
        add(count, usize::from(selected))
    })
}

fn index(
    rows: &[&StoredRow],
    tombstones: &[&PendingOp],
    side: JoinSide,
    epoch: crate::Epoch,
    digest: &[u8; 32],
    encoder: &Encoder,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<u8>> {
    let mut bytes = PaidBuffer::new(&encoder.funding, encoder.retained);
    bytes.append(&index_header(side, epoch, rows.len(), tombstones.len())?)?;
    for (position, row) in rows.iter().enumerate() {
        check()?;
        upsert(&mut bytes, row, digest, position)?;
    }
    append_tombstones(&mut bytes, tombstones, check)?;
    Ok(bytes.into_bytes())
}

fn append_tombstones(
    bytes: &mut PaidBuffer<'_>,
    tombstones: &[&PendingOp],
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for row in tombstones {
        check()?;
        tombstone(bytes, row)?;
    }
    Ok(())
}

fn index_header(
    side: JoinSide,
    epoch: crate::Epoch,
    upserts: usize,
    removals: usize,
) -> Result<[u8; 40]> {
    let mut fixed = [0; 40];
    fixed[..16].copy_from_slice(&header(DELTA_MAGIC, side));
    fixed[16..24].copy_from_slice(&epoch.as_u64().to_le_bytes());
    fixed[24..32].copy_from_slice(&integer(upserts)?);
    fixed[32..40].copy_from_slice(&integer(removals)?);
    Ok(fixed)
}

fn tombstone(bytes: &mut PaidBuffer<'_>, op: &PendingOp) -> Result<()> {
    let PendingOp::Tombstone {
        row_id,
        event_time,
        encoded_key,
        ..
    } = op
    else {
        unreachable!("delta removal selection contains tombstones")
    };
    let mut fixed = [0; 24];
    fixed[..8].copy_from_slice(&row_id.to_le_bytes());
    fixed[8..16].copy_from_slice(&event_time.as_micros().to_le_bytes());
    fixed[16..24].copy_from_slice(&integer(encoded_key.len())?);
    bytes.append(&fixed)?;
    bytes.append(encoded_key)
}
