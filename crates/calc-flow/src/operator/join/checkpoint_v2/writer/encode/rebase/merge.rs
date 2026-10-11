use std::{cmp::Reverse, collections::BinaryHeap};

use super::super::super::super::{budget, geometry::checked};
use super::{HistoryInput, JoinSide, MemoryReservation, Result, accounting, error, frame, funding};

struct Cursor<'a> {
    records: frame::Records<'a>,
    head: Option<frame::Upsert<'a>>,
    pending: bool,
}

impl Cursor<'_> {
    fn advance(&mut self) -> Result<()> {
        self.head = match checked(self.records.next())? {
            Some(frame::Record::Upsert(row)) => Some(row),
            Some(frame::Record::Tombstone(_)) => {
                return Err(error("V2 rebase index contains a tombstone"));
            }
            None => None,
        };
        Ok(())
    }
}

struct PayloadView<'a> {
    frame: frame::Payload<'a>,
    seen: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct Counts {
    pub(super) rows: usize,
    charge: u64,
    key_bytes: usize,
}

impl Counts {
    pub(super) fn index_bytes(self) -> Result<usize> {
        accounting::sum(&[32, accounting::product(self.rows, 72)?, self.key_bytes])
    }

    fn append(self, row: &frame::Upsert<'_>) -> Result<Self> {
        Ok(Self {
            rows: accounting::add(self.rows, 1)?,
            charge: self
                .charge
                .checked_add(row.charge)
                .ok_or_else(|| error("V2 rebase row charges overflow"))?,
            key_bytes: accounting::add(self.key_bytes, row.key.len())?,
        })
    }
}

pub(super) fn scan(
    history: &HistoryInput,
    side: JoinSide,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
    mut visit: impl FnMut(&frame::Upsert<'_>) -> Result<()>,
) -> Result<Counts> {
    let (count, _credit) = scan_credit(history, side, workspace, check)?;
    let mut cursors = cursors(history, side, count, check)?;
    let mut payloads = payloads(history, check)?;
    let mut heap = heap(&cursors, check)?;
    let counts = scan_rows(
        history,
        side,
        &mut cursors,
        &mut payloads,
        &mut heap,
        check,
        &mut visit,
    )?;
    validate_scan(history, side, &payloads, counts, check)?;
    check()?;
    Ok(counts)
}

fn scan_credit(
    history: &HistoryInput,
    side: JoinSide,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<(usize, MemoryReservation)> {
    check()?;
    let count = history.indexes.iter().try_fold(0, |count, input| {
        check()?;
        accounting::add(count, usize::from(input.side == side))
    })?;
    let credit = workspace.new_empty();
    accounting::reserve(
        &credit,
        scan_workspace_bytes(count, history.payloads.len())?,
    )?;
    Ok((count, credit))
}

fn scan_workspace_bytes(count: usize, payloads: usize) -> Result<usize> {
    accounting::sum(&[
        budget::registration_bytes()?,
        accounting::vector_peak::<Cursor<'_>>(count)?,
        accounting::vector_peak::<Reverse<(u64, usize)>>(count)?,
        accounting::vector_peak::<PayloadView<'_>>(payloads)?,
    ])
}

fn scan_rows(
    history: &HistoryInput,
    side: JoinSide,
    cursors: &mut [Cursor<'_>],
    payloads: &mut [PayloadView<'_>],
    heap: &mut BinaryHeap<Reverse<(u64, usize)>>,
    check: &dyn Fn() -> Result<()>,
    visit: &mut impl FnMut(&frame::Upsert<'_>) -> Result<()>,
) -> Result<Counts> {
    let expected = &history.expected[usize::from(side == JoinSide::Right)];
    let mut counts = Counts::default();
    let mut previous = None;
    let mut pending_started = false;
    while let Some(Reverse((id, index))) = heap.pop() {
        check()?;
        let cursor = &mut cursors[index];
        let row = cursor.head.take().expect("one heap entry per ready cursor");
        validate_order(
            previous,
            pending_started,
            id,
            cursor.pending,
            expected.next_row_id,
        )?;
        previous = Some(id);
        pending_started |= cursor.pending;
        locate(history, payloads, side, &row)?;
        counts = counts.append(&row)?;
        visit(&row)?;
        advance_cursor(cursor, heap, index)?;
    }
    Ok(counts)
}

fn validate_order(
    previous: Option<u64>,
    pending_started: bool,
    id: u64,
    pending: bool,
    next_row_id: u64,
) -> Result<()> {
    if previous.is_some_and(|previous| id <= previous) || id >= next_row_id {
        return Err(error("V2 rebase row IDs are not globally unique and valid"));
    }
    if pending_started && !pending {
        return Err(error(
            "V2 rebase pending rows do not follow captured history",
        ));
    }
    Ok(())
}

fn advance_cursor(
    cursor: &mut Cursor<'_>,
    heap: &mut BinaryHeap<Reverse<(u64, usize)>>,
    index: usize,
) -> Result<()> {
    cursor.advance()?;
    if let Some(row) = &cursor.head {
        heap.push(Reverse((row.row_id, index)));
    }
    Ok(())
}

fn validate_scan(
    history: &HistoryInput,
    side: JoinSide,
    payloads: &[PayloadView<'_>],
    counts: Counts,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    let expected = &history.expected[usize::from(side == JoinSide::Right)];
    if counts.rows != expected.rows || counts.charge != expected.charge {
        return Err(error("V2 rebase history differs from retained row totals"));
    }
    for (input, payload) in history.payloads.iter().zip(payloads) {
        check()?;
        if input.side == side && payload.seen != input.rows {
            return Err(error("V2 rebase payload is not wholly referenced"));
        }
    }
    Ok(())
}

fn cursors<'a>(
    history: &'a HistoryInput,
    side: JoinSide,
    count: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<Cursor<'a>>> {
    let mut cursors = funding::vector(count)?;
    for input in history.indexes.iter().filter(|input| input.side == side) {
        check()?;
        let index = input.index()?;
        if index.tombstones != 0 {
            return Err(error("V2 rebase index contains tombstones"));
        }
        let mut cursor = Cursor {
            records: index.records(),
            head: None,
            pending: input.pending,
        };
        cursor.advance()?;
        cursors.push(cursor);
    }
    Ok(cursors)
}

fn payloads<'a>(
    history: &'a HistoryInput,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<PayloadView<'a>>> {
    let mut views = funding::vector(history.payloads.len())?;
    for payload in &history.payloads {
        check()?;
        let frame = checked(frame::Payload::decode(
            payload.segment.bytes(),
            payload.side,
        ))?;
        if frame.rows != payload.rows {
            return Err(error("V2 rebase payload row count differs"));
        }
        views.push(PayloadView { frame, seen: 0 });
    }
    Ok(views)
}

fn heap(
    cursors: &[Cursor<'_>],
    check: &dyn Fn() -> Result<()>,
) -> Result<BinaryHeap<Reverse<(u64, usize)>>> {
    let mut heap = BinaryHeap::new();
    heap.try_reserve_exact(cursors.len())
        .map_err(|_| error("V2 rebase cursor heap allocation failed"))?;
    if heap.capacity() > cursors.len() {
        return Err(error("V2 rebase cursor heap exceeded paid capacity"));
    }
    for (index, cursor) in cursors.iter().enumerate() {
        check()?;
        if let Some(row) = &cursor.head {
            heap.push(Reverse((row.row_id, index)));
        }
    }
    Ok(heap)
}

fn locate(
    history: &HistoryInput,
    payloads: &mut [PayloadView<'_>],
    side: JoinSide,
    row: &frame::Upsert<'_>,
) -> Result<()> {
    let index = history
        .payloads
        .binary_search_by_key(&(side, row.digest), |payload| {
            (payload.side, payload.digest)
        })
        .map_err(|_| error("V2 rebase locator names an unknown payload"))?;
    let payload = &mut payloads[index];
    let position = usize::try_from(row.payload_row)
        .map_err(|_| error("V2 rebase payload position overflow"))?;
    if checked(payload.frame.row_id(position))? != row.row_id {
        return Err(error("V2 rebase payload row ID differs from its locator"));
    }
    payload.seen = accounting::add(payload.seen, 1)?;
    Ok(())
}
