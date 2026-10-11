use std::{cmp::Reverse, collections::BinaryHeap};

use datafusion::execution::memory_pool::MemoryReservation;

use super::super::frame::{Record, Records};
use super::{History, JoinSide, Locator, Result, checked, invalid, product, reserve, sum};

type Head = Reverse<(u64, Option<u64>, usize)>;

struct Cursor<'a> {
    records: Records<'a>,
    head: Option<Record<'a>>,
    epoch: Option<u64>,
}

pub(super) struct Scan<'a> {
    cursors: Vec<Cursor<'a>>,
    heap: BinaryHeap<Head>,
    _credit: MemoryReservation,
}

impl<'a> Scan<'a> {
    pub(super) fn new(
        history: &History<'a>,
        side: JoinSide,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        let count = product(
            history
                .indexes
                .iter()
                .filter(|index| index.side == side)
                .count(),
            2,
        )?;
        let credit = history.credit.new_empty();
        reserve(
            &credit,
            sum(&[
                product(count, size_of::<Cursor<'_>>())?,
                product(count, size_of::<Head>())?,
            ])?,
        )?;
        let mut scan = Self {
            cursors: Vec::with_capacity(count),
            heap: BinaryHeap::with_capacity(count),
            _credit: credit,
        };
        if scan.cursors.capacity() != count || scan.heap.capacity() != count {
            return Err(invalid(
                "V2 history cursor capacity differs from its funding",
            ));
        }
        for index in history.indexes.iter().filter(|index| index.side == side) {
            for records in index.groups {
                check()?;
                scan.cursors.push(Cursor {
                    records,
                    head: None,
                    epoch: index.epoch,
                });
                scan.advance(scan.cursors.len() - 1)?;
            }
        }
        Ok(scan)
    }

    fn advance(&mut self, index: usize) -> Result<()> {
        let cursor = &mut self.cursors[index];
        cursor.head = checked(cursor.records.next())?;
        if let Some(head) = &cursor.head {
            let id = match head {
                Record::Upsert(row) => row.row_id,
                Record::Tombstone(row) => row.row_id,
            };
            self.heap.push(Reverse((id, cursor.epoch, index)));
        }
        Ok(())
    }

    fn pop(&mut self) -> Result<Option<(Record<'a>, Option<u64>)>> {
        let Some(Reverse((_, epoch, index))) = self.heap.pop() else {
            return Ok(None);
        };
        let record = self.cursors[index]
            .head
            .take()
            .expect("queued history head");
        self.advance(index)?;
        Ok(Some((record, epoch)))
    }

    pub(super) fn next(
        &mut self,
        next_id: u64,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Option<(Locator<'a>, bool)>> {
        check()?;
        let Some((record, introduced)) = self.pop()? else {
            return Ok(None);
        };
        let Record::Upsert(row) = record else {
            return Err(invalid("V2 tombstone does not identify a live row"));
        };
        if row.row_id >= next_id {
            return Err(invalid(
                "V2 historical row ID is reused or exceeds its next ID",
            ));
        }
        let mut live = true;
        while self
            .heap
            .peek()
            .is_some_and(|Reverse((id, _, _))| *id == row.row_id)
        {
            check()?;
            let (record, epoch) = self.pop()?.expect("matching history head");
            let Record::Tombstone(tombstone) = record else {
                return Err(invalid(
                    "V2 historical row ID is reused or exceeds its next ID",
                ));
            };
            if !live {
                return Err(invalid("V2 tombstone does not identify a live row"));
            }
            if epoch <= introduced {
                return Err(invalid("V2 index upsert and tombstone groups overlap"));
            }
            if row.time != tombstone.time || row.key != tombstone.key {
                return Err(invalid("V2 tombstone identity differs from its live row"));
            }
            live = false;
        }
        Ok(Some((
            Locator {
                row_id: row.row_id,
                time: row.time,
                charge: row.charge,
                digest: row.digest,
                payload_row: row.payload_row,
                key: row.key,
            },
            live,
        )))
    }
}
