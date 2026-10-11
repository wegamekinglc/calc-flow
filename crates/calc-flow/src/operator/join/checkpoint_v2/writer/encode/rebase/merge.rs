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
    check()?;
    let count = history.indexes.iter().try_fold(0, |count, input| {
        check()?;
        accounting::add(count, usize::from(input.side == side))
    })?;
    let credit = workspace.new_empty();
    accounting::reserve(
        &credit,
        accounting::sum(&[
            budget::registration_bytes()?,
            accounting::vector_peak::<Cursor<'_>>(count)?,
            accounting::vector_peak::<Reverse<(u64, usize)>>(count)?,
            accounting::vector_peak::<PayloadView<'_>>(history.payloads.len())?,
        ])?,
    )?;
    let mut cursors = cursors(history, side, count, check)?;
    let mut payloads = payloads(history, check)?;
    let mut heap = heap(&cursors, check)?;
    let expected = &history.expected[usize::from(side == JoinSide::Right)];
    let mut counts = Counts::default();
    let mut previous = None;
    let mut pending_started = false;
    while let Some(Reverse((id, index))) = heap.pop() {
        check()?;
        let cursor = &mut cursors[index];
        let row = cursor.head.take().expect("one heap entry per ready cursor");
        if previous.is_some_and(|previous| id <= previous) || id >= expected.next_row_id {
            return Err(error("V2 rebase row IDs are not globally unique and valid"));
        }
        previous = Some(id);
        if pending_started && !cursor.pending {
            return Err(error(
                "V2 rebase pending rows do not follow captured history",
            ));
        }
        pending_started |= cursor.pending;
        locate(history, &mut payloads, side, &row)?;
        counts = counts.append(&row)?;
        visit(&row)?;
        cursor.advance()?;
        if let Some(row) = &cursor.head {
            heap.push(Reverse((row.row_id, index)));
        }
    }
    if counts.rows != expected.rows || counts.charge != expected.charge {
        return Err(error("V2 rebase history differs from retained row totals"));
    }
    for (input, payload) in history.payloads.iter().zip(&payloads) {
        check()?;
        if input.side == side && payload.seen != input.rows {
            return Err(error("V2 rebase payload is not wholly referenced"));
        }
    }
    check()?;
    Ok(counts)
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
