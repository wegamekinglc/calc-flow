use datafusion::execution::memory_pool::MemoryReservation;

use super::super::{JoinSide, OperatorStateSnapshot};
use super::frame::{Payload, Records};
use super::geometry::{self, Geometry, checked, invalid};
use super::inventory;
use super::ipc::accounting::{product, reserve, sum};
use crate::Result;

mod merge;

#[derive(Clone, Copy)]
pub(super) struct Locator<'a> {
    pub(super) row_id: u64,
    pub(super) time: i64,
    pub(super) charge: u64,
    pub(super) digest: [u8; 32],
    pub(super) payload_row: u64,
    pub(super) key: &'a [u8],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct LiveSummary {
    pub(super) rows: usize,
    pub(super) bytes: u64,
}

impl LiveSummary {
    fn add(&mut self, locator: &Locator<'_>) -> Result<()> {
        self.rows = self
            .rows
            .checked_add(1)
            .ok_or_else(|| invalid("V2 retained row count overflow"))?;
        self.bytes = self
            .bytes
            .checked_add(locator.charge)
            .ok_or_else(|| invalid("V2 retained byte count overflow"))?;
        Ok(())
    }
}

struct IndexGroups<'a> {
    side: JoinSide,
    epoch: Option<u64>,
    groups: [Records<'a>; 2],
}

struct PayloadFrame<'a> {
    side: JoinSide,
    digest: [u8; 32],
    frame: Payload<'a>,
}

pub(super) struct History<'a> {
    indexes: Vec<IndexGroups<'a>>,
    payloads: Vec<PayloadFrame<'a>>,
    credit: MemoryReservation,
}

impl<'a> History<'a> {
    pub(super) fn new(
        snapshot: &'a OperatorStateSnapshot,
        geometry: &Geometry,
        workspace: &MemoryReservation,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        let indexes = snapshot
            .segments
            .len()
            .checked_sub(geometry.payloads)
            .ok_or_else(|| invalid("V2 index count underflow"))?;
        let mut history = Self::allocate(indexes, geometry.payloads, workspace)?;
        for (name, segment) in &snapshot.segments {
            check()?;
            history.add_segment(name, segment.bytes(), check)?;
        }
        Ok(history)
    }

    fn allocate(indexes: usize, payloads: usize, workspace: &MemoryReservation) -> Result<Self> {
        let credit = workspace.new_empty();
        reserve(
            &credit,
            sum(&[
                product(indexes, size_of::<IndexGroups<'_>>())?,
                product(payloads, size_of::<PayloadFrame<'_>>())?,
            ])?,
        )?;
        let history = Self {
            indexes: Vec::with_capacity(indexes),
            payloads: Vec::with_capacity(payloads),
            credit,
        };
        if history.indexes.capacity() != indexes || history.payloads.capacity() != payloads {
            return Err(invalid(
                "V2 history descriptor capacity differs from its funding",
            ));
        }
        Ok(history)
    }

    fn add_segment(
        &mut self,
        name: &str,
        bytes: &'a [u8],
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let (side, suffix) = checked(inventory::split_side(name))?;
        if let Some(digest) = suffix.strip_prefix("payload-") {
            self.payloads.push(PayloadFrame {
                side,
                digest: checked(inventory::digest(digest))?,
                frame: checked(Payload::decode(bytes, side))?,
            });
        } else {
            let index = geometry::index(bytes, suffix, side)?;
            self.indexes.push(IndexGroups {
                side,
                epoch: index.epoch,
                groups: index.groups(check)?,
            });
        }
        Ok(())
    }

    pub(super) fn scan(
        &self,
        side: JoinSide,
        next_id: u64,
        check: &dyn Fn() -> Result<()>,
        mut visit: impl FnMut(&Locator<'a>, bool) -> Result<()>,
    ) -> Result<LiveSummary> {
        let mut scan = merge::Scan::new(self, side, check)?;
        let mut summary = LiveSummary::default();
        while let Some((locator, live)) = scan.next(next_id, check)? {
            self.locate(side, &locator)?;
            if live {
                summary.add(&locator)?;
            }
            visit(&locator, live)?;
        }
        Ok(summary)
    }

    fn locate(&self, side: JoinSide, locator: &Locator<'_>) -> Result<()> {
        let index = self
            .payloads
            .binary_search_by_key(&(side, locator.digest), |payload| {
                (payload.side, payload.digest)
            })
            .map_err(|_| invalid("V2 upsert references an unknown payload"))?;
        let row = usize::try_from(locator.payload_row)
            .map_err(|_| invalid("V2 payload row does not fit usize"))?;
        if checked(self.payloads[index].frame.row_id(row))? != locator.row_id {
            return Err(invalid("V2 payload row ID differs from its upsert"));
        }
        Ok(())
    }
}

pub(super) fn tree<K, V>(count: usize) -> Option<usize> {
    let alignment = align_of::<K>()
        .max(align_of::<V>())
        .max(align_of::<usize>());
    let node = tree_node::<K, V>(alignment)?;
    count.checked_add(1)?.checked_mul(node)
}

fn tree_node<K, V>(alignment: usize) -> Option<usize> {
    let fields = size_of::<usize>() + 2 * size_of::<u16>() + 12 * size_of::<usize>();
    let slots = 11_usize.checked_mul(size_of::<K>().checked_add(size_of::<V>())?)?;
    fields
        .checked_add(slots)?
        .checked_add(6 * (alignment - 1))?
        .checked_add(alignment - 1)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}
