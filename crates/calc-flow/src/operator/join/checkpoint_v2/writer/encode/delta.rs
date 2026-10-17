#[cfg(test)]
use super::super::super::super::PendingOp;
use super::super::super::super::columnar::FramedKey;
use super::super::super::super::{JoinSide, Tombstones};
use super::{
    Arc, Base, CheckpointRow, Encoder, MemoryReservation, PaidBuffer, Reference, Result, SchemaRef,
    accounting, add, ensure, error, header, integer, sum, upsert, upsert_bytes,
};
use crate::EventTime;

const DELTA_MAGIC: &[u8; 8] = b"CFJDIX2\0";

/// Read access to one side's recorded tombstones (spec FR2): either the live
/// list or an owned copy made for asynchronous delta work.
pub(in crate::operator::join::checkpoint_v2::writer) trait TombstoneList {
    fn entries(&self) -> usize;
    fn entry(&self, index: usize) -> (u64, EventTime, &Arc<FramedKey>);
}

impl TombstoneList for Tombstones {
    fn entries(&self) -> usize {
        Tombstones::len(self)
    }
    fn entry(&self, index: usize) -> (u64, EventTime, &Arc<FramedKey>) {
        Tombstones::entry(self, index)
    }
}

/// Derived per-side delta input (spec FR3): the live rows at or above the
/// watermark provide the upserts and the recorded list the tombstones.
pub(in crate::operator::join::checkpoint_v2::writer) struct SideDelta<'a> {
    pub(in crate::operator::join::checkpoint_v2::writer) rows: &'a dyn crate::operator::join::Rows,
    pub(in crate::operator::join::checkpoint_v2::writer) watermark: u64,
    pub(in crate::operator::join::checkpoint_v2::writer) upserts: usize,
    pub(in crate::operator::join::checkpoint_v2::writer) tombstones: &'a dyn TombstoneList,
}

pub(super) fn encode(
    sides: &[SideDelta<'_>; 2],
    schemas: &[SchemaRef; 2],
    epoch: crate::Epoch,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<Base> {
    check()?;
    let mut encoder = Encoder::new(workspace)?;
    for (index, side) in [JoinSide::Left, JoinSide::Right].into_iter().enumerate() {
        check()?;
        let selected = select(&sides[index], workspace, check)?;
        emit_side(
            &mut encoder,
            selected,
            Arc::clone(&schemas[index]),
            side,
            epoch,
            check,
        )?;
    }
    check()?;
    Ok(encoder.finish())
}

/// One wire tombstone selected for encoding.
pub(super) struct TombstoneEntry<'a> {
    pub(super) row_id: u64,
    pub(super) event_time: EventTime,
    pub(super) encoded_key: &'a Arc<FramedKey>,
}

/// Per-side selected upserts and tombstones, both ascending by `row_id`.
pub(super) struct SideSelection<'a, C: CheckpointRow> {
    rows: Vec<C>,
    tombstones: Vec<TombstoneEntry<'a>>,
    funding: MemoryReservation,
}

impl<C: CheckpointRow> SideSelection<'_, C> {
    fn allocate(upserts: usize, removals: usize, workspace: &MemoryReservation) -> Result<Self> {
        let funding = workspace.new_empty();
        accounting::reserve(
            &funding,
            sum(&[
                accounting::product(upserts, size_of::<C>())?,
                accounting::product(removals, size_of::<TombstoneEntry<'_>>())?,
            ])?,
        )?;
        Ok(Self {
            rows: exact_selection(upserts)?,
            tombstones: exact_selection(removals)?,
            funding,
        })
    }
}

/// Derives one side's selection from the live rows at or above its
/// watermark; the paid count comes from the O(1) dirty counter (spec FR8).
fn select<'a>(
    source: &SideDelta<'a>,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<SideSelection<'a, Reference<'a>>> {
    let mut selected =
        SideSelection::allocate(source.upserts, source.tombstones.entries(), workspace)?;
    for index in source.rows.references_from(source.watermark) {
        check()?;
        selected.rows.push(Reference {
            rows: source.rows,
            index,
        });
    }
    if selected.rows.len() != source.upserts {
        return Err(error(
            "V2 checkpoint derived selection differs from its paid count",
        ));
    }
    selected.rows.sort_unstable_by_key(|row| row.row().row_id);
    for entry in 0..source.tombstones.entries() {
        check()?;
        let (row_id, event_time, encoded_key) = source.tombstones.entry(entry);
        selected.tombstones.push(TombstoneEntry {
            row_id,
            event_time,
            encoded_key,
        });
    }
    selected
        .tombstones
        .sort_unstable_by_key(|entry| entry.row_id);
    check()?;
    Ok(selected)
}

/// Selects one side from an old-style pending-op list: the byte-equivalence
/// oracle (spec FR4) and encoder-test constructor.
#[cfg(test)]
pub(super) fn select_ops<'a>(
    pending: &'a [PendingOp],
    side: JoinSide,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<SideSelection<'a, super::PendingUpsert<'a>>> {
    let (upserts, removals) = pending_counts(pending, side, check)?;
    let mut selected = SideSelection::allocate(upserts, removals, workspace)?;
    for op in pending {
        check()?;
        if op.identity().0 == side {
            if let Some(row) = super::PendingUpsert::new(op) {
                selected.rows.push(row);
            } else {
                let PendingOp::Tombstone {
                    row_id,
                    event_time,
                    encoded_key,
                    ..
                } = op
                else {
                    unreachable!("delta removal selection contains tombstones")
                };
                selected.tombstones.push(TombstoneEntry {
                    row_id: *row_id,
                    event_time: *event_time,
                    encoded_key,
                });
            }
        }
    }
    selected.rows.sort_unstable_by_key(|row| row.row().row_id);
    selected
        .tombstones
        .sort_unstable_by_key(|entry| entry.row_id);
    check()?;
    Ok(selected)
}

#[cfg(test)]
fn pending_counts(
    pending: &[PendingOp],
    side: JoinSide,
    check: &dyn Fn() -> Result<()>,
) -> Result<(usize, usize)> {
    pending
        .iter()
        .try_fold((0, 0), |(upserts, removals), op| -> Result<_> {
            check()?;
            if op.identity().0 != side {
                Ok((upserts, removals))
            } else if matches!(op, PendingOp::Upsert { .. }) {
                Ok((add(upserts, 1)?, removals))
            } else {
                Ok((upserts, add(removals, 1)?))
            }
        })
}

pub(super) fn emit_side<C: CheckpointRow>(
    encoder: &mut Encoder,
    selected: SideSelection<'_, C>,
    schema: SchemaRef,
    side: JoinSide,
    epoch: crate::Epoch,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    let SideSelection {
        rows,
        tombstones,
        funding,
    } = selected;
    if rows.is_empty() && tombstones.is_empty() {
        return Ok(());
    }
    debug_assert!(
        rows.windows(2)
            .all(|pair| pair[0].row().row_id < pair[1].row().row_id),
        "delta upsert row ids are unique and ascending"
    );
    debug_assert!(
        tombstones
            .windows(2)
            .all(|pair| pair[0].row_id < pair[1].row_id),
        "delta tombstone row ids are unique and ascending"
    );
    let digest = if rows.is_empty() {
        [0; 32]
    } else {
        encoder.payload(&rows, schema, side, &funding, check)?
    };
    let bytes = index(&rows, &tombstones, side, epoch, &digest, encoder, check)?;
    install_delta(encoder, side, epoch, bytes)
}

fn exact_selection<T>(count: usize) -> Result<Vec<T>> {
    let bytes = accounting::product(count, size_of::<T>())?;
    isize::try_from(bytes).map_err(|_| error("V2 checkpoint selection capacity overflow"))?;
    let mut selected = Vec::new();
    selected
        .try_reserve_exact(count)
        .map_err(|_| error("V2 checkpoint selection allocation failed"))?;
    if selected.capacity() > count {
        return Err(error("V2 checkpoint selection exceeded paid capacity"));
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

fn index<C: CheckpointRow>(
    rows: &[C],
    tombstones: &[TombstoneEntry<'_>],
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

fn append_upserts<C: CheckpointRow>(
    bytes: &mut PaidBuffer<'_>,
    rows: &[C],
    digest: &[u8; 32],
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for (position, row) in rows.iter().enumerate() {
        check()?;
        upsert(bytes, row, digest, position)?;
    }
    Ok(())
}

fn index_capacity<C: CheckpointRow>(
    rows: &[C],
    tombstones: &[TombstoneEntry<'_>],
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    let capacity =
        tombstones
            .iter()
            .try_fold(add(40, upsert_bytes(rows, check)?)?, |bytes, entry| {
                check()?;
                sum(&[bytes, 24, entry.encoded_key.len()])
            })?;
    Ok(capacity)
}

fn append_tombstones(
    bytes: &mut PaidBuffer<'_>,
    tombstones: &[TombstoneEntry<'_>],
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for entry in tombstones {
        check()?;
        tombstone(bytes, entry)?;
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

fn tombstone(bytes: &mut PaidBuffer<'_>, entry: &TombstoneEntry<'_>) -> Result<()> {
    let mut fixed = [0; 24];
    fixed[..8].copy_from_slice(&entry.row_id.to_le_bytes());
    fixed[8..16].copy_from_slice(&entry.event_time.as_micros().to_le_bytes());
    fixed[16..24].copy_from_slice(&integer(entry.encoded_key.len())?);
    bytes.append(&fixed)?;
    bytes.append(entry.encoded_key)
}
