use std::{any::Any, sync::Arc};

use super::super::super::{geometry, inventory};
use super::super::metadata::DeltaEntry;
use super::{
    Base, Encoder, MemoryReservation, PayloadEntry, Result, StateSegment, accounting, add, budget,
    ensure, error, product, sum,
};
use crate::OperatorStateSnapshot;

type Owner = Arc<dyn Any + Send + Sync>;

pub(in crate::operator::join::checkpoint_v2::writer) struct RestoredHistory {
    pub(in crate::operator::join::checkpoint_v2::writer) deltas: Vec<DeltaEntry>,
    pub(in crate::operator::join::checkpoint_v2::writer) base_epoch: u64,
    pub(in crate::operator::join::checkpoint_v2::writer) dirty_epochs: u32,
    pub(in crate::operator::join::checkpoint_v2::writer) base: Base,
}

struct Counts {
    payloads: usize,
    deltas: usize,
    bytes: usize,
}

pub(in crate::operator::join::checkpoint_v2::writer) fn decode(
    snapshot: &OperatorStateSnapshot,
    inventory: &inventory::Inventory<'_>,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<RestoredHistory> {
    check()?;
    let encoder = Encoder::new(workspace)?;
    let counts = required(snapshot, inventory, check)?;
    ensure(&encoder.funding, add(encoder.retained, counts.bytes)?)?;
    copy_history(snapshot, inventory, &counts, encoder, check)
}

fn copy_history(
    snapshot: &OperatorStateSnapshot,
    inventory: &inventory::Inventory<'_>,
    counts: &Counts,
    mut encoder: Encoder,
    check: &dyn Fn() -> Result<()>,
) -> Result<RestoredHistory> {
    let dirty_epochs =
        u32::try_from(counts.deltas).map_err(|_| error("V2 restored dirty epochs overflow"))?;
    let deltas = copy_deltas(inventory, counts.deltas, check)?;
    encoder.output.payloads = copy_payloads(inventory, counts.payloads, check)?;
    copy_segments(snapshot, &mut encoder, check)?;
    check()?;
    Ok(RestoredHistory {
        deltas,
        base_epoch: inventory.base_epoch,
        dirty_epochs,
        base: encoder.output,
    })
}

fn required(
    snapshot: &OperatorStateSnapshot,
    inventory: &inventory::Inventory<'_>,
    check: &dyn Fn() -> Result<()>,
) -> Result<Counts> {
    let counts = Counts {
        payloads: 0,
        deltas: 0,
        bytes: segment_bytes(snapshot, check)?,
    };
    let counts = payload_bytes(counts, inventory, check)?;
    let counts = delta_bytes(counts, inventory, check)?;
    finish_counts(counts)
}

fn segment_bytes(
    snapshot: &OperatorStateSnapshot,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    let mut bytes = budget::tree::<String, StateSegment>(snapshot.segments.len())?;
    for (name, segment) in &snapshot.segments {
        check()?;
        bytes = sum(&[
            bytes,
            name.len(),
            segment.sha256().len(),
            segment_owner_bytes(segment)?,
        ])?;
    }
    Ok(bytes)
}

fn payload_bytes(
    mut counts: Counts,
    inventory: &inventory::Inventory<'_>,
    check: &dyn Fn() -> Result<()>,
) -> Result<Counts> {
    for payload in inventory.payloads() {
        check()?;
        geometry::checked(payload)?;
        counts.payloads = add(counts.payloads, 1)?;
        counts.bytes = add(counts.bytes, 64)?;
    }
    Ok(counts)
}

fn delta_bytes(
    mut counts: Counts,
    inventory: &inventory::Inventory<'_>,
    check: &dyn Fn() -> Result<()>,
) -> Result<Counts> {
    for delta in inventory.delta_sides() {
        check()?;
        let (_, sides) = geometry::checked(delta)?;
        counts.deltas = add(counts.deltas, 1)?;
        counts.bytes = add(
            counts.bytes,
            product(sides.len(), size_of::<&'static str>())?,
        )?;
    }
    Ok(counts)
}

fn finish_counts(mut counts: Counts) -> Result<Counts> {
    counts.bytes = sum(&[
        counts.bytes,
        product(counts.payloads, size_of::<PayloadEntry>())?,
        product(counts.deltas, size_of::<DeltaEntry>())?,
    ])?;
    Ok(counts)
}

fn segment_owner_bytes(segment: &StateSegment) -> Result<usize> {
    if segment.has_owner() {
        accounting::arc::<(Owner, Owner)>()
    } else {
        add(
            accounting::arc::<Vec<u8>>()?,
            segment.bytes_arc().capacity(),
        )
    }
}

fn copy_deltas(
    inventory: &inventory::Inventory<'_>,
    count: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<DeltaEntry>> {
    let mut deltas = Vec::with_capacity(count);
    for delta in inventory.delta_sides() {
        check()?;
        let (epoch, values) = geometry::checked(delta)?;
        let mut sides = Vec::with_capacity(values.len());
        for value in values {
            check()?;
            sides.push(geometry::checked(inventory::side(value))?.as_str());
        }
        deltas.push(DeltaEntry { epoch, sides });
    }
    Ok(deltas)
}

fn copy_payloads(
    inventory: &inventory::Inventory<'_>,
    count: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<PayloadEntry>> {
    let mut payloads = Vec::with_capacity(count);
    for payload in inventory.payloads() {
        check()?;
        let payload = geometry::checked(payload)?;
        payloads.push(PayloadEntry {
            side: payload.side.as_str(),
            sha256: hex::encode(payload.digest),
            rows: payload.rows,
            bytes: payload.bytes,
        });
    }
    Ok(payloads)
}

fn copy_segments(
    snapshot: &OperatorStateSnapshot,
    encoder: &mut Encoder,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for (name, segment) in &snapshot.segments {
        check()?;
        encoder.output.segments.insert(
            name.clone(),
            segment.clone().with_owner(encoder.funding.clone()),
        );
    }
    Ok(())
}
