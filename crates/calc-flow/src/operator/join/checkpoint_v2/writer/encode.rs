use std::{collections::BTreeMap, sync::Arc};

use datafusion::arrow::{
    array::ArrayRef,
    datatypes::SchemaRef,
    ipc::{
        MetadataVersion,
        writer::{IpcWriteOptions, StreamWriter},
    },
};
use datafusion::execution::memory_pool::MemoryReservation;
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::super::super::{JoinSide, StoredRow};
use super::super::{
    budget,
    ipc::accounting::{self, add, product, sum},
};
use super::{
    buffer::{PaidBuffer, ensure, error},
    compact,
};
use crate::{Result, StateSegment};

mod delta;
pub(super) mod restored;
pub(super) mod retarget;

const BASE_MAGIC: &[u8; 8] = b"CFJIDX2\0";
const PAYLOAD_MAGIC: &[u8; 8] = b"CFJPAY2\0";

#[derive(Clone, Serialize)]
pub(super) struct PayloadEntry {
    pub(super) side: &'static str,
    pub(super) sha256: String,
    pub(super) rows: u64,
    pub(super) bytes: u64,
}

pub(super) struct Base {
    pub(super) segments: BTreeMap<String, StateSegment>,
    pub(super) payloads: Vec<PayloadEntry>,
    owners: Vec<Arc<MemoryReservation>>,
    funding: Arc<MemoryReservation>,
}

impl Base {
    pub(super) fn credit(&self) -> &MemoryReservation {
        &self.funding
    }

    pub(super) fn merge(&mut self, mut incoming: Self) -> Result<()> {
        self.admit_merge(&incoming)?;
        self.segments.append(&mut incoming.segments);
        self.payloads.append(&mut incoming.payloads);
        canonical_payloads(&mut self.payloads);
        self.owners.append(&mut incoming.owners);
        self.owners.push(incoming.funding);
        Ok(())
    }
    fn admit_merge(&self, incoming: &Self) -> Result<()> {
        let segments = add(self.segments.len(), incoming.segments.len())?;
        let payloads = add(self.payloads.len(), incoming.payloads.len())?;
        let owners = sum(&[self.owners.len(), incoming.owners.len(), 1])?;
        accounting::reserve(
            &self.funding,
            sum(&[
                budget::tree::<String, StateSegment>(segments)?,
                bulk_bytes::<PayloadEntry>(payloads)?,
                bulk_bytes::<Arc<MemoryReservation>>(owners)?,
            ])?,
        )
    }
}

pub(super) fn canonical_payloads(payloads: &mut Vec<PayloadEntry>) {
    payloads.sort_unstable_by(|left, right| {
        left.side
            .cmp(right.side)
            .then_with(|| left.sha256.cmp(&right.sha256))
    });
    payloads.dedup_by(|left, right| left.side == right.side && left.sha256 == right.sha256);
}

fn bulk_bytes<T>(length: usize) -> Result<usize> {
    product(product(length, 3)?.max(4), size_of::<T>())
}

pub(super) fn pending(
    pending: &super::super::super::PendingLog,
    schemas: &[SchemaRef; 2],
    epoch: crate::Epoch,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<Base> {
    let owned = copy_pending(pending, workspace, check)?;
    pending_owned(&owned, schemas, epoch, workspace, check)
}

pub(super) fn copy_pending(
    pending: &super::super::super::PendingLog,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<super::super::super::PendingOp>> {
    let count = pending.iter().try_fold(0, |count, _| {
        check()?;
        add(count, 1)
    })?;
    accounting::reserve(
        workspace,
        bulk_bytes::<super::super::super::PendingOp>(count)?,
    )?;
    let mut owned = Vec::with_capacity(count);
    for op in pending.iter() {
        check()?;
        owned.push(clone_pending_op(op, workspace)?);
    }
    Ok(owned)
}

fn clone_pending_op(
    op: &super::super::super::PendingOp,
    workspace: &MemoryReservation,
) -> Result<super::super::super::PendingOp> {
    if let super::super::super::PendingOp::Upsert { record, .. } = op {
        admit_payload_clone(record, workspace)?;
    }
    Ok(op.clone())
}

pub(super) fn admit_payload_clone(
    record: &super::super::super::columnar::RowPayload,
    workspace: &MemoryReservation,
) -> Result<()> {
    if let super::super::super::columnar::RowPayload::Legacy(record) = record {
        accounting::reserve(
            workspace,
            product(record.num_columns(), size_of::<ArrayRef>())?,
        )?;
    }
    Ok(())
}

pub(super) fn pending_owned(
    pending: &[super::super::super::PendingOp],
    schemas: &[SchemaRef; 2],
    epoch: crate::Epoch,
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<Base> {
    delta::encode(pending, schemas, epoch, workspace, check)
}

struct Encoder {
    output: Base,
    funding: Arc<MemoryReservation>,
    retained: usize,
}

pub(super) fn base(
    sides: [&[StoredRow]; 2],
    schemas: &[SchemaRef; 2],
    workspace: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<Base> {
    let mut encoder = Encoder::new(workspace)?;
    for (index, side) in [JoinSide::Left, JoinSide::Right].into_iter().enumerate() {
        check()?;
        encoder.side(
            sides[index],
            Arc::clone(&schemas[index]),
            side,
            workspace,
            check,
        )?;
    }
    check()?;
    Ok(encoder.output)
}

impl Encoder {
    fn new(workspace: &MemoryReservation) -> Result<Self> {
        let credit = workspace.new_empty();
        let seed = sum(&[
            accounting::arc::<MemoryReservation>()?,
            super::budget::diagnostic_bytes(),
            budget::registration_bytes()?,
        ])?;
        ensure(&credit, seed)?;
        let funding = Arc::new(credit);
        Ok(Self {
            output: Base {
                segments: BTreeMap::new(),
                payloads: Vec::new(),
                owners: Vec::new(),
                funding: Arc::clone(&funding),
            },
            funding,
            retained: seed,
        })
    }

    fn side(
        &mut self,
        rows: &[StoredRow],
        schema: SchemaRef,
        side: JoinSide,
        workspace: &MemoryReservation,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let ordered = ordered_rows(rows, workspace)?;
        let digest = if ordered.is_empty() {
            [0; 32]
        } else {
            self.payload(&ordered, schema, side, workspace, check)?
        };
        let bytes = encode_index(&ordered, side, &digest, &self.funding, self.retained, check)?;
        ensure(&self.funding, add(self.retained, 16)?)?;
        self.install(format!("{}-base", side.as_str()), bytes)
    }

    fn payload(
        &mut self,
        rows: &[&StoredRow],
        schema: SchemaRef,
        side: JoinSide,
        workspace: &MemoryReservation,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<[u8; 32]> {
        let compacted = compact::compact(rows, schema, workspace, check)?;
        let scratch = super::budget::stream_scratch(&compacted.record)?;
        ensure(&self.funding, add(self.retained, scratch)?)?;
        let mut ipc = PaidBuffer::new(&self.funding, add(self.retained, scratch)?);
        write_stream(&mut ipc, &compacted.record, check)?;
        let ipc = ipc.into_bytes();
        drop(compacted);
        self.frame_payload(rows, side, ipc, check)
    }

    fn frame_payload(
        &mut self,
        rows: &[&StoredRow],
        side: JoinSide,
        ipc: Vec<u8>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<[u8; 32]> {
        let mut bytes = PaidBuffer::new(&self.funding, add(self.retained, ipc.capacity())?);
        append_payload(&mut bytes, rows, side, &ipc, check)?;
        let bytes = bytes.into_bytes();
        drop(ipc);
        check()?;
        self.admit_payload_entry(bytes.capacity())?;
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        let sha256 = hex::encode(digest);
        let entry = payload_entry(side, &sha256, rows.len(), bytes.len())?;
        self.install(format!("{}-payload-{sha256}", side.as_str()), bytes)?;
        self.output.payloads.push(entry);
        Ok(digest)
    }

    fn admit_payload_entry(&self, capacity: usize) -> Result<()> {
        ensure(
            &self.funding,
            sum(&[
                self.retained,
                capacity,
                256,
                accounting::vector_peak::<PayloadEntry>(2)?,
            ])?,
        )
    }

    fn install(&mut self, name: String, bytes: Vec<u8>) -> Result<()> {
        let overhead = sum(&[
            budget::tree::<String, StateSegment>(1)?,
            name.capacity(),
            accounting::arc::<Vec<u8>>()?,
            64,
            accounting::vector_peak::<PayloadEntry>(2)?,
            128,
        ])?;
        let retained = sum(&[self.retained, bytes.capacity(), overhead])?;
        ensure(&self.funding, retained)?;
        let segment = StateSegment::new(bytes).with_owner(self.funding.clone());
        self.output.segments.insert(name, segment);
        self.retained = retained;
        Ok(())
    }
}

fn header(magic: [u8; 8], side: JoinSide) -> [u8; 16] {
    let mut header = [0; 16];
    header[..8].copy_from_slice(&magic);
    header[8..12].copy_from_slice(&2_u32.to_le_bytes());
    header[12] = u8::from(side == JoinSide::Right);
    header
}

fn integer(value: usize) -> Result<[u8; 8]> {
    Ok(u64::try_from(value)
        .map_err(|_| error("V2 checkpoint count overflow"))?
        .to_le_bytes())
}

fn upsert(
    buffer: &mut PaidBuffer<'_>,
    row: &StoredRow,
    digest: &[u8; 32],
    position: usize,
) -> Result<()> {
    let mut fixed = [0_u8; 72];
    fixed[..8].copy_from_slice(&row.row_id.to_le_bytes());
    fixed[8..16].copy_from_slice(&row.event_time.as_micros().to_le_bytes());
    fixed[16..24].copy_from_slice(&row.charge.to_le_bytes());
    fixed[24..56].copy_from_slice(digest);
    fixed[56..64].copy_from_slice(&integer(position)?);
    fixed[64..72].copy_from_slice(&integer(row.encoded_key.len())?);
    buffer.append(&fixed)?;
    buffer.append(&row.encoded_key)
}

fn ordered_rows<'a>(
    rows: &'a [StoredRow],
    workspace: &MemoryReservation,
) -> Result<Vec<&'a StoredRow>> {
    accounting::reserve(
        workspace,
        accounting::vector_peak::<&StoredRow>(rows.len())?,
    )?;
    let mut ordered = rows.iter().collect::<Vec<_>>();
    ordered.sort_unstable_by_key(|row| row.row_id);
    Ok(ordered)
}

fn encode_index(
    rows: &[&StoredRow],
    side: JoinSide,
    digest: &[u8; 32],
    credit: &MemoryReservation,
    retained: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<u8>> {
    let mut index = PaidBuffer::new(credit, retained);
    let mut fixed = [0; 32];
    fixed[..16].copy_from_slice(&header(*BASE_MAGIC, side));
    fixed[16..24].copy_from_slice(&integer(rows.len())?);
    index.append(&fixed)?;
    for (position, row) in rows.iter().enumerate() {
        check()?;
        upsert(&mut index, row, digest, position)?;
    }
    Ok(index.into_bytes())
}

fn write_stream(
    output: &mut PaidBuffer<'_>,
    record: &datafusion::arrow::record_batch::RecordBatch,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    let options = IpcWriteOptions::try_new(8, false, MetadataVersion::V5)
        .map_err(|_| error("V2 checkpoint IPC options are invalid"))?;
    let mut writer = StreamWriter::try_new_with_options(output, record.schema_ref(), options)
        .map_err(|_| error("V2 checkpoint IPC schema encoding failed"))?;
    check()?;
    writer
        .write(record)
        .map_err(|_| error("V2 checkpoint IPC payload encoding failed"))?;
    check()?;
    writer
        .finish()
        .map_err(|_| error("V2 checkpoint IPC finish failed"))
}

fn append_payload(
    bytes: &mut PaidBuffer<'_>,
    rows: &[&StoredRow],
    side: JoinSide,
    ipc: &[u8],
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    let mut fixed = [0; 32];
    fixed[..16].copy_from_slice(&header(*PAYLOAD_MAGIC, side));
    fixed[16..24].copy_from_slice(&integer(rows.len())?);
    fixed[24..32].copy_from_slice(&integer(ipc.len())?);
    bytes.append(&fixed)?;
    for row in rows {
        check()?;
        bytes.append(&row.row_id.to_le_bytes())?;
    }
    bytes.append(ipc)
}

fn payload_entry(side: JoinSide, sha256: &str, rows: usize, bytes: usize) -> Result<PayloadEntry> {
    Ok(PayloadEntry {
        side: side.as_str(),
        sha256: sha256.into(),
        rows: u64::try_from(rows).map_err(|_| error("V2 payload count overflow"))?,
        bytes: u64::try_from(bytes).map_err(|_| error("V2 payload length overflow"))?,
    })
}
