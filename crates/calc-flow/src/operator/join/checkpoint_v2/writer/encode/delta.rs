use super::super::super::super::PendingOp;
use super::{
    Arc, Base, CheckpointRow, Encoder, JoinSide, MemoryReservation, PaidBuffer, PendingUpsert,
    Result, SchemaRef, accounting, add, ensure, header, integer, sum, upsert, upsert_bytes,
};

const DELTA_MAGIC: &[u8; 8] = b"CFJDIX2\0";

pub(super) fn encode<'a, I: Iterator<Item = &'a PendingOp>>(
    pending: impl Fn() -> I,
    schemas: &[SchemaRef; 2],
    epoch: crate::Epoch,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<Base> {
    check()?;
    let mut encoder = Encoder::new(workspace)?;
    for (index, side) in [JoinSide::Left, JoinSide::Right].into_iter().enumerate() {
        check()?;
        encode_side(
            &mut encoder,
            &pending,
            Arc::clone(&schemas[index]),
            side,
            epoch,
            workspace,
            check,
        )?;
    }
    check()?;
    Ok(encoder.finish())
}

fn encode_side<'a, I: Iterator<Item = &'a PendingOp>>(
    encoder: &mut Encoder,
    pending: &impl Fn() -> I,
    schema: SchemaRef,
    side: JoinSide,
    epoch: crate::Epoch,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    let selected = PendingRows::select(pending, side, workspace, check)?;
    if selected.rows.is_empty() && selected.tombstones.is_empty() {
        return Ok(());
    }
    let digest = if selected.rows.is_empty() {
        [0; 32]
    } else {
        encoder.payload(&selected.rows, schema, side, &selected.funding, check)?
    };
    let bytes = index(
        &selected.rows,
        &selected.tombstones,
        side,
        epoch,
        &digest,
        encoder,
        check,
    )?;
    install_delta(encoder, side, epoch, bytes)
}

// Field order releases the actual selection allocations before their credit.
struct PendingRows<'a> {
    rows: Vec<PendingUpsert<'a>>,
    tombstones: Vec<&'a PendingOp>,
    funding: MemoryReservation,
}

impl<'a> PendingRows<'a> {
    fn allocate(upserts: usize, removals: usize, workspace: &MemoryReservation) -> Result<Self> {
        let funding = workspace.new_empty();
        accounting::reserve(
            &funding,
            sum(&[
                accounting::product(upserts, size_of::<PendingUpsert<'_>>())?,
                accounting::product(removals, size_of::<&PendingOp>())?,
            ])?,
        )?;
        let selected = Self {
            rows: exact_selection(upserts)?,
            tombstones: exact_selection(removals)?,
            funding,
        };
        Ok(selected)
    }

    fn select<I: Iterator<Item = &'a PendingOp>>(
        pending: &impl Fn() -> I,
        side: JoinSide,
        workspace: &MemoryReservation,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        let (upserts, removals) = pending_counts(pending, side, check)?;
        let mut selected = Self::allocate(upserts, removals, workspace)?;
        for op in pending() {
            check()?;
            if op.identity().0 == side {
                if let Some(row) = PendingUpsert::new(op) {
                    selected.rows.push(row);
                } else {
                    selected.tombstones.push(op);
                }
            }
        }
        selected.rows.sort_unstable_by_key(|row| row.row().row_id);
        selected
            .tombstones
            .sort_unstable_by_key(|op| op.identity().1);
        check()?;
        Ok(selected)
    }
}

fn pending_counts<'a, I: Iterator<Item = &'a PendingOp>>(
    pending: &impl Fn() -> I,
    side: JoinSide,
    check: &dyn Fn() -> Result<()>,
) -> Result<(usize, usize)> {
    let (upserts, removals) =
        pending().try_fold((0, 0), |(upserts, removals), op| -> Result<_> {
            check()?;
            if op.identity().0 != side {
                Ok((upserts, removals))
            } else if matches!(op, PendingOp::Upsert { .. }) {
                Ok((add(upserts, 1)?, removals))
            } else {
                Ok((upserts, add(removals, 1)?))
            }
        })?;
    Ok((upserts, removals))
}

fn exact_selection<T>(count: usize) -> Result<Vec<T>> {
    let bytes = accounting::product(count, size_of::<T>())?;
    isize::try_from(bytes)
        .map_err(|_| super::error("V2 checkpoint selection capacity overflow"))?;
    let mut selected = Vec::new();
    selected
        .try_reserve_exact(count)
        .map_err(|_| super::error("V2 checkpoint selection allocation failed"))?;
    if selected.capacity() > count {
        return Err(super::error(
            "V2 checkpoint selection exceeded paid capacity",
        ));
    }
    Ok(selected)
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

fn index(
    rows: &[impl CheckpointRow],
    tombstones: &[&PendingOp],
    side: JoinSide,
    epoch: crate::Epoch,
    digest: &[u8; 32],
    encoder: &Encoder,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<u8>> {
    let capacity = index_capacity(rows, tombstones, check)?;
    check()?;
    let mut bytes = PaidBuffer::exact(&encoder.funding, encoder.retained, capacity)?;
    bytes.append(&index_header(side, epoch, rows.len(), tombstones.len())?)?;
    append_upserts(&mut bytes, rows, digest, check)?;
    append_tombstones(&mut bytes, tombstones, check)?;
    bytes.into_bytes()
}

fn append_upserts(
    bytes: &mut PaidBuffer<'_>,
    rows: &[impl CheckpointRow],
    digest: &[u8; 32],
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for (position, row) in rows.iter().enumerate() {
        check()?;
        upsert(bytes, row, digest, position)?;
    }
    Ok(())
}

fn index_capacity(
    rows: &[impl CheckpointRow],
    tombstones: &[&PendingOp],
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    let capacity =
        tombstones
            .iter()
            .try_fold(add(40, upsert_bytes(rows, check)?)?, |bytes, op| {
                check()?;
                let PendingOp::Tombstone { encoded_key, .. } = op else {
                    unreachable!("delta removal selection contains tombstones")
                };
                sum(&[bytes, 24, encoded_key.len()])
            })?;
    Ok(capacity)
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
    fixed[..16].copy_from_slice(&header(*DELTA_MAGIC, side));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::join::columnar::RowPayload;
    use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};

    #[test]
    fn test_pending_selection_exact_credit_retires_between_sides() {
        let record = RecordBatch::try_from_iter([(
            "key",
            Arc::new(Int64Array::from(vec![9])) as datafusion::arrow::array::ArrayRef,
        )])
        .unwrap();
        let key = Arc::new(vec![1, 9].into());
        let pending = [
            PendingOp::Upsert {
                side: JoinSide::Left,
                row_id: 5,
                event_time: crate::EventTime::from_micros(3),
                encoded_key: Arc::clone(&key),
                record: RowPayload::Legacy(record),
                charge: 99,
            },
            PendingOp::Tombstone {
                side: JoinSide::Right,
                row_id: 8,
                event_time: crate::EventTime::from_micros(-3),
                encoded_key: Arc::clone(&key),
            },
            PendingOp::Tombstone {
                side: JoinSide::Left,
                row_id: 7,
                event_time: crate::EventTime::from_micros(1),
                encoded_key: key,
            },
        ];
        assert_eq!(size_of::<PendingUpsert<'_>>(), size_of::<&PendingOp>());
        for (side, upserts, removals) in [(JoinSide::Left, 1, 1), (JoinSide::Right, 0, 1)] {
            let paid = (upserts + removals) * size_of::<&PendingOp>();
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(paid));
            let workspace = MemoryConsumer::new("pending-selection").register(&pool);
            let mut result = None;
            let allocations = allocation_counter::measure(|| {
                result = Some(PendingRows::select(
                    &|| pending.iter(),
                    side,
                    &workspace,
                    &|| Ok(()),
                ));
            });
            let selected = result.unwrap().unwrap();
            assert_eq!(
                (selected.rows.len(), selected.rows.capacity()),
                (upserts, upserts)
            );
            assert_eq!(
                (selected.tombstones.len(), selected.tombstones.capacity()),
                (removals, removals)
            );
            assert_eq!(selected.funding.size(), paid);
            assert_eq!(allocations.bytes_current, i64::try_from(paid).unwrap());
            assert!(allocations.bytes_max <= u64::try_from(paid).unwrap());
            drop(selected);
            assert_eq!(pool.reserved(), 0);
            let pressure = workspace.new_empty();
            pressure.try_grow(1).unwrap();
            assert!(PendingRows::select(&|| pending.iter(), side, &workspace, &|| Ok(())).is_err());
            assert_eq!(pool.reserved(), 1);
        }
    }
}
