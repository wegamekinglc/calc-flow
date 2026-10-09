use std::collections::{BTreeMap, BTreeSet};

use datafusion::execution::memory_pool::MemoryReservation;

use super::super::{JoinSide, OperatorStateSnapshot, metadata_validation::ValidatedMetadata};
use super::frame::{Index, Record, Tombstone, Upsert};
use super::geometry::{Geometry, checked, invalid};
use super::inventory::{self, Inventory};
use crate::{CalcFlowError, Result};

#[derive(Clone, Copy)]
pub(super) struct Locator<'a> {
    pub(super) row_id: u64,
    pub(super) time: i64,
    pub(super) charge: u64,
    pub(super) digest: [u8; 32],
    pub(super) payload_row: u64,
    pub(super) key: &'a [u8],
    introduced_in: Option<u64>,
}

#[derive(Default)]
pub(super) struct SideHistory<'a> {
    pub(super) live: BTreeMap<u64, Locator<'a>>,
    pub(super) locators: BTreeSet<([u8; 32], u64)>,
    pub(super) historical: BTreeMap<u64, Locator<'a>>,
}

pub(super) fn fold<'a>(
    snapshot: &'a OperatorStateSnapshot,
    inventory: &Inventory<'_>,
    metadata: &ValidatedMetadata,
    geometry: &Geometry,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<[SideHistory<'a>; 2]> {
    reserve(workspace, geometry)?;
    check()?;
    let left = fold_side(
        snapshot,
        inventory,
        JoinSide::Left,
        metadata.next_left_row_id,
        check,
    )?;
    check()?;
    let right = fold_side(
        snapshot,
        inventory,
        JoinSide::Right,
        metadata.next_right_row_id,
        check,
    )?;
    check()?;
    Ok([left, right])
}

fn fold_side<'a>(
    snapshot: &'a OperatorStateSnapshot,
    inventory: &Inventory<'_>,
    side: JoinSide,
    next_id: u64,
    check: &dyn Fn() -> Result<()>,
) -> Result<SideHistory<'a>> {
    let mut history = SideHistory::default();
    history.apply(&base_index(snapshot, side)?, next_id, check)?;
    fold_deltas(snapshot, inventory, side, next_id, check, &mut history)?;
    Ok(history)
}

fn fold_deltas<'a>(
    snapshot: &'a OperatorStateSnapshot,
    inventory: &Inventory<'_>,
    side: JoinSide,
    next_id: u64,
    check: &dyn Fn() -> Result<()>,
    history: &mut SideHistory<'a>,
) -> Result<()> {
    for delta in inventory.delta_sides() {
        check()?;
        let (epoch, sides) = checked(delta)?;
        if !listed_side(sides, side)? {
            continue;
        }
        history.apply(&delta_index(snapshot, side, epoch)?, next_id, check)?;
    }
    Ok(())
}

fn delta_index(snapshot: &OperatorStateSnapshot, side: JoinSide, epoch: u64) -> Result<Index<'_>> {
    let segment = delta_segment(snapshot, side, epoch)?;
    checked(Index::delta(segment, side, epoch))
}

fn base_index(snapshot: &OperatorStateSnapshot, side: JoinSide) -> Result<Index<'_>> {
    let name = match side {
        JoinSide::Left => "left-base",
        JoinSide::Right => "right-base",
    };
    let base = snapshot
        .segments
        .get(name)
        .ok_or_else(|| invalid("V2 base segment is missing"))?;
    checked(Index::base(base.bytes(), side))
}

impl<'a> SideHistory<'a> {
    fn apply(
        &mut self,
        index: &Index<'a>,
        next_id: u64,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let mut records = index.records();
        loop {
            check()?;
            let Some(record) = checked(records.next())? else {
                return Ok(());
            };
            match record {
                Record::Upsert(row) => self.upsert(&row, next_id, index.epoch)?,
                Record::Tombstone(row) => self.remove(&row, index.epoch)?,
            }
        }
    }

    fn upsert(&mut self, row: &Upsert<'a>, next_id: u64, epoch: Option<u64>) -> Result<()> {
        if row.row_id >= next_id || self.historical.contains_key(&row.row_id) {
            return Err(invalid(
                "V2 historical row ID is reused or exceeds its next ID",
            ));
        }
        if !self.locators.insert((row.digest, row.payload_row)) {
            return Err(invalid("V2 payload row is introduced more than once"));
        }
        let locator = Locator {
            row_id: row.row_id,
            time: row.time,
            charge: row.charge,
            digest: row.digest,
            payload_row: row.payload_row,
            key: row.key,
            introduced_in: epoch,
        };
        self.historical.insert(row.row_id, locator);
        self.live.insert(row.row_id, locator);
        Ok(())
    }

    fn remove(&mut self, row: &Tombstone<'_>, epoch: Option<u64>) -> Result<()> {
        let live = self
            .live
            .get(&row.row_id)
            .ok_or_else(|| invalid("V2 tombstone does not identify a live row"))?;
        if epoch.is_some() && live.introduced_in == epoch {
            return Err(invalid("V2 index upsert and tombstone groups overlap"));
        }
        if live.time != row.time || live.key != row.key {
            return Err(invalid("V2 tombstone identity differs from its live row"));
        }
        self.live.remove(&row.row_id);
        Ok(())
    }
}

fn listed_side(sides: &[serde_json::Value], expected: JoinSide) -> Result<bool> {
    for value in sides {
        if checked(inventory::side(value))? == expected {
            return Ok(true);
        }
    }
    Ok(false)
}

fn delta_segment(snapshot: &OperatorStateSnapshot, side: JoinSide, epoch: u64) -> Result<&[u8]> {
    let prefix = match side {
        JoinSide::Left => "left-delta-",
        JoinSide::Right => "right-delta-",
    };
    snapshot
        .segments
        .iter()
        .find(|(name, _)| {
            name.strip_prefix(prefix)
                .and_then(|text| text.parse::<u64>().ok())
                == Some(epoch)
        })
        .map(|(_, segment)| segment.bytes())
        .ok_or_else(|| invalid("V2 listed delta segment is missing"))
}

fn reserve(workspace: &MemoryReservation, geometry: &Geometry) -> Result<()> {
    let count = geometry.upserts;
    let bytes = tree::<u64, Locator<'_>>(count)
        .and_then(|bytes| bytes.checked_add(tree::<u64, Locator<'_>>(count)?))
        .and_then(|bytes| bytes.checked_add(tree::<([u8; 32], u64), ()>(count)?))
        .and_then(|bytes| bytes.checked_add(2 * size_of::<SideHistory<'_>>()))
        .ok_or_else(|| invalid("V2 history workspace charge overflow"))?;
    workspace
        .try_grow(bytes)
        .map_err(|_| CalcFlowError::Internal {
            message: "V2 history workspace credit admission failed".into(),
        })
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
