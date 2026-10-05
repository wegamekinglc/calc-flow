pub(in crate::operator::asof) mod chain;
pub(in crate::operator::asof) mod journal;
pub(in crate::operator::asof) mod model;
mod read;
#[cfg(test)]
mod tests;
mod write;

use super::{
    BatchKey, ChunkData, Cursor, Encoding, PayloadBatch, PreparedLeftChunk, SequenceKind, mismatch,
    owners::{OwnerReader, OwnerWriter},
};
use crate::{Result, StateSegment};
use datafusion::execution::memory_pool::MemoryReservation;
use journal::{Change, Identity, Version};
use std::{collections::BTreeMap, sync::Arc};

const MAGIC: &[u8; 8] = b"CFASRW10";
const HEADER_BYTES: u64 = 208;

pub(in crate::operator::asof) struct BucketCut {
    pub key: Encoding,
    pub state: Option<(u64, [usize; 5])>,
}

pub(in crate::operator::asof) struct Input<'a> {
    pub capacities: [usize; 16],
    pub counts: [usize; 3],
    pub kinds: [SequenceKind; 2],
    pub changes: &'a [Change],
    pub left: &'a [(BatchKey, &'a ChunkData, usize)],
    pub buckets: &'a [BucketCut],
}

pub(in crate::operator::asof) struct EncodedDelta {
    pub segment: StateSegment,
    pub owners: OwnerWriter,
    pub owner_credit: MemoryReservation,
}

pub(in crate::operator::asof) struct DecodedDelta {
    pub capacities: [usize; 16],
    pub counts: [usize; 3],
    pub changes: Vec<Change>,
    pub left: Vec<(BatchKey, PreparedLeftChunk)>,
    pub buckets: Vec<BucketCut>,
    pub owners: OwnerReader,
    pub workspace: MemoryReservation,
}

#[cfg(test)]
pub(in crate::operator::asof) fn encode(
    input: &Input<'_>,
    previous: &OwnerWriter,
    live_allocation: impl Fn(usize) -> bool,
    reserve: impl FnMut(u64) -> Result<MemoryReservation>,
    limit: u64,
    name: &str,
) -> Result<EncodedDelta> {
    encode_checked(
        input,
        previous,
        live_allocation,
        reserve,
        limit,
        name,
        &|| Ok(()),
    )
}

pub(in crate::operator::asof) fn encode_checked(
    input: &Input<'_>,
    previous: &OwnerWriter,
    live_allocation: impl Fn(usize) -> bool,
    reserve: impl FnMut(u64) -> Result<MemoryReservation>,
    limit: u64,
    name: &str,
    cancel: &dyn Fn() -> Result<()>,
) -> Result<EncodedDelta> {
    write::encode(
        input,
        previous,
        live_allocation,
        reserve,
        limit,
        name,
        cancel,
    )
}

pub(in crate::operator::asof) fn restore_charge(
    bytes: &[u8],
    previous_owners: usize,
    kinds: [SequenceKind; 2],
    max_rows: u64,
    max_bytes: u64,
) -> Result<u64> {
    read::restore_charge(bytes, previous_owners, kinds, max_rows, max_bytes)
}

pub(in crate::operator::asof) fn decode(
    bytes: &[u8],
    owners: &OwnerReader,
    batches: &BTreeMap<BatchKey, Arc<PayloadBatch>>,
    kinds: [SequenceKind; 2],
    limits: &crate::AsofStateLimits,
    workspace: MemoryReservation,
    check_cancelled: impl FnMut() -> Result<()>,
) -> Result<DecodedDelta> {
    read::decode(
        bytes,
        owners,
        batches,
        kinds,
        limits,
        workspace,
        check_cancelled,
    )
}

fn version_bytes(version: Option<Version>) -> u64 {
    1 + match version {
        None => 0,
        Some(Version::Left { .. }) => 56,
        Some(Version::Right { payload, .. }) => 1 + if payload.is_some() { 12 } else { 0 },
    }
}

fn read_version(
    cursor: &mut Cursor<'_>,
    identity: &Identity,
    kind: SequenceKind,
) -> Result<Option<Version>> {
    match cursor.byte()? {
        0 => Ok(None),
        1 => match identity {
            Identity::Left(_) => {
                let rows = cursor.integer()?;
                let capacities = cursor.capacities([
                    8,
                    4,
                    size_of::<Option<Encoding>>(),
                    8,
                    4,
                    kind.storage_bytes(),
                ])?;
                Ok(Some(Version::Left { rows, capacities }))
            }
            Identity::Right(_) => read_right_version(cursor),
        },
        _ => Err(mismatch("ASOF log version tag differs")),
    }
}

fn read_right_version(cursor: &mut Cursor<'_>) -> Result<Option<Version>> {
    let tag = cursor.byte()?;
    let payload = match tag {
        0 | 2 => None,
        1 => Some(((1, cursor.integer()?), cursor.small()?)),
        _ => return Err(mismatch("ASOF log storage tag differs")),
    };
    Ok(Some(Version::Right { tag, payload }))
}
