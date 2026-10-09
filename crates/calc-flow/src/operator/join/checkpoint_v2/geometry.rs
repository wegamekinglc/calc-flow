use super::super::{JoinSide, OperatorStateSnapshot};
use super::frame::{Index, InvalidFrame, Payload, Record};
use super::inventory::Inventory;
use crate::{CalcFlowError, Result};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Default)]
pub(super) struct Geometry {
    pub(super) upserts: usize,
    pub(super) tombstones: usize,
    pub(super) key_bytes: usize,
    pub(super) longest_key: usize,
    pub(super) payload_rows: usize,
    pub(super) payloads: usize,
}

impl Geometry {
    pub(super) fn inspect(
        snapshot: &OperatorStateSnapshot,
        inventory: &Inventory<'_>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        let mut counts = Self::default();
        for (name, segment) in &snapshot.segments {
            check()?;
            counts.inspect_segment(name, segment.bytes(), inventory, check)?;
        }
        if counts.upserts != counts.payload_rows {
            return Err(invalid(
                "V2 payload rows and historical upsert counts differ",
            ));
        }
        Ok(counts)
    }

    fn inspect_segment(
        &mut self,
        name: &str,
        bytes: &[u8],
        inventory: &Inventory<'_>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let (side, suffix) = checked(super::inventory::split_side(name))?;
        if let Some(digest) = suffix.strip_prefix("payload-") {
            self.inspect_payload(bytes, side, digest, inventory, check)
        } else {
            self.inspect_index_segment(bytes, suffix, side, inventory.base_epoch, check)
        }
    }

    fn inspect_index_segment(
        &mut self,
        bytes: &[u8],
        suffix: &str,
        side: JoinSide,
        base_epoch: u64,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let index = index(bytes, suffix, side)?;
        if base_epoch == 0 && index.epoch.is_none() && index.upserts != 0 {
            return Err(invalid("V2 epoch-zero bases must be empty"));
        }
        self.inspect_index(&index, check)
    }

    fn inspect_payload(
        &mut self,
        bytes: &[u8],
        side: JoinSide,
        digest: &str,
        inventory: &Inventory<'_>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let digest = checked(super::inventory::digest(digest))?;
        validate_digest(bytes, &digest, check)?;
        let payload = checked(Payload::decode(bytes, side))?;
        validate_payload_count(inventory, side, digest, payload.rows, check)?;
        validate_payload_ids(&payload, check)?;
        self.payloads = sum(self.payloads, 1)?;
        self.payload_rows = sum(self.payload_rows, payload.rows)?;
        Ok(())
    }

    fn inspect_index(&mut self, index: &Index<'_>, check: &dyn Fn() -> Result<()>) -> Result<()> {
        self.upserts = sum(self.upserts, index.upserts)?;
        self.tombstones = sum(self.tombstones, index.tombstones)?;
        let mut records = index.records();
        loop {
            check()?;
            let Some(record) = checked(records.next())? else {
                return Ok(());
            };
            let key = match record {
                Record::Upsert(row) => row.key,
                Record::Tombstone(row) => row.key,
            };
            self.key_bytes = sum(self.key_bytes, key.len())?;
            self.longest_key = self.longest_key.max(key.len());
        }
    }
}

fn validate_payload_ids(payload: &Payload<'_>, check: &dyn Fn() -> Result<()>) -> Result<()> {
    let mut previous = None;
    for row in 0..payload.rows {
        check()?;
        let id = checked(payload.row_id(row))?;
        if previous.is_some_and(|previous| id <= previous) {
            return Err(invalid("V2 payload IDs must be strictly increasing"));
        }
        previous = Some(id);
    }
    Ok(())
}

fn validate_digest(
    bytes: &[u8],
    expected: &[u8; 32],
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    let mut digest = Sha256::new();
    for chunk in bytes.chunks(4096) {
        check()?;
        digest.update(chunk);
    }
    if digest.finalize().as_slice() != expected {
        return Err(invalid("V2 payload SHA-256 differs from its inventory"));
    }
    Ok(())
}

fn validate_payload_count(
    inventory: &Inventory<'_>,
    side: JoinSide,
    digest: [u8; 32],
    rows: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for entry in inventory.payloads() {
        check()?;
        let entry = checked(entry)?;
        if entry.side == side && entry.digest == digest {
            let rows = u64::try_from(rows).map_err(|_| invalid("V2 payload rows exceed u64"))?;
            return if entry.rows == rows {
                Ok(())
            } else {
                Err(invalid("V2 payload row count differs from its inventory"))
            };
        }
    }
    Err(invalid("V2 payload is not listed in its inventory"))
}

pub(super) fn index<'a>(bytes: &'a [u8], suffix: &str, side: JoinSide) -> Result<Index<'a>> {
    if suffix == "base" {
        return checked(Index::base(bytes, side));
    }
    let epoch = suffix
        .strip_prefix("delta-")
        .and_then(|text| text.parse().ok())
        .ok_or_else(|| invalid("V2 index identity is invalid"))?;
    checked(Index::delta(bytes, side, epoch))
}

fn sum(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid("V2 history count overflow"))
}

pub(super) fn checked<T>(value: std::result::Result<T, InvalidFrame>) -> Result<T> {
    value.map_err(|InvalidFrame(message)| invalid(message))
}

pub(super) fn invalid(message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: message.into(),
    }
}
