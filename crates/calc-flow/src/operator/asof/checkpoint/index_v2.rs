//! Columnar ASOF checkpoint index. Arrow payload batches live in separate,
//! immutable snapshot segments and are referenced by (side, batch id, row).

use super::{
    super::{
        checked,
        codec::BoundedWriter,
        state::{BatchKey, Encoding, LeftOrder, RowPayload, State},
    },
    mismatch,
};
use crate::{CalcFlowError, Result, StateSegment, StreamOperatorContext};
use datafusion::execution::memory_pool::MemoryReservation;
use std::collections::BTreeMap;

const MAGIC: &[u8; 8] = b"CFASOF02";
pub(super) const INDEX_SEGMENT: &str = "asof-index-v2";
const CHECK_EVERY: usize = 256;
const YIELD_EVERY: usize = 8_192;

pub(super) fn batch_segment(key: BatchKey) -> String {
    format!("asof-batch-{}-{}", key.0, key.1)
}

pub(super) fn parse_batch_segment(name: &str) -> Result<BatchKey> {
    let Some(rest) = name.strip_prefix("asof-batch-") else {
        return Err(mismatch("unexpected ASOF segment name"));
    };
    let Some((side, id)) = rest.split_once('-') else {
        return Err(mismatch("malformed ASOF batch segment name"));
    };
    let side = side
        .parse::<u8>()
        .map_err(|_| mismatch("invalid ASOF batch side"))?;
    let id = id
        .parse::<u64>()
        .map_err(|_| mismatch("invalid ASOF batch id"))?;
    if side > 1 || batch_segment((side, id)) != name {
        return Err(mismatch("noncanonical ASOF batch segment name"));
    }
    Ok((side, id))
}

pub(in super::super) fn encoded_length(state: &State, name: &str) -> Result<u64> {
    if state.left.is_empty() && state.right.is_empty() {
        return Ok(0);
    }
    let mut size = 24_u64;
    for (_, key, sequence) in state.left.keys() {
        size = checked(name, size, 41 + key.len() as u64 + sequence.len() as u64)?;
    }
    for (key, bucket) in &state.right {
        size = checked(name, size, 16 + key.len() as u64)?;
        for ((_, sequence), row) in bucket {
            size = checked(
                name,
                size,
                17 + sequence.len() as u64 + if row.is_some() { 17 } else { 0 },
            )?;
        }
    }
    Ok(size)
}

pub(in super::super) fn left_prefix_length(state: &State, count: usize, name: &str) -> Result<u64> {
    state
        .left
        .iter()
        .take(count)
        .try_fold(0, |size, ((_, key, sequence), _)| {
            checked(name, size, 41 + key.len() as u64 + sequence.len() as u64)
        })
}

pub(super) async fn encode(
    state: &State,
    length: u64,
    limit: usize,
    context: &StreamOperatorContext<'_>,
    workspace: MemoryReservation,
) -> Result<(StateSegment, MemoryReservation)> {
    let capacity = usize::try_from(length).expect("reserved address domain");
    let mut bytes = Vec::with_capacity(capacity);
    {
        let mut writer = BoundedWriter::with_capacity(&mut bytes, capacity, limit);
        write_header(&mut writer, state)?;
        context.check_cancelled()?;
        tokio::task::yield_now().await;
        for (ordinal, ((time, key, sequence), payload)) in state.left.iter().enumerate() {
            write_left(&mut writer, *time, key, sequence, payload)?;
            if ordinal % CHECK_EVERY == CHECK_EVERY - 1 {
                context.check_cancelled()?;
            }
            if ordinal % YIELD_EVERY == YIELD_EVERY - 1 {
                tokio::task::yield_now().await;
            }
        }
        let mut ordinal = 0;
        for (key, bucket) in &state.right {
            write_bucket_header(&mut writer, key, bucket.len())?;
            for ((time, sequence), payload) in bucket {
                write_right(&mut writer, *time, sequence, payload.as_ref())?;
                ordinal += 1;
                if ordinal % CHECK_EVERY == 0 {
                    context.check_cancelled()?;
                }
                if ordinal % YIELD_EVERY == 0 {
                    tokio::task::yield_now().await;
                }
            }
        }
        context.check_cancelled()?;
    }
    if bytes.len() != capacity {
        return Err(mismatch("ASOF index encoded length differs"));
    }
    let result = tokio::task::spawn_blocking(move || (StateSegment::new(bytes), workspace))
        .await
        .map_err(|error| CalcFlowError::Internal {
            message: format!("ASOF index digest task failed: {error}"),
        })?;
    context.check_cancelled()?;
    Ok(result)
}

pub(super) fn encode_sync(state: &State, length: u64, limit: usize) -> Result<StateSegment> {
    let capacity = usize::try_from(length).expect("reserved address domain");
    let mut bytes = Vec::with_capacity(capacity);
    {
        let mut writer = BoundedWriter::with_capacity(&mut bytes, capacity, limit);
        write_header(&mut writer, state)?;
        for ((time, key, sequence), payload) in &state.left {
            write_left(&mut writer, *time, key, sequence, payload)?;
        }
        for (key, bucket) in &state.right {
            write_bucket_header(&mut writer, key, bucket.len())?;
            for ((time, sequence), payload) in bucket {
                write_right(&mut writer, *time, sequence, payload.as_ref())?;
            }
        }
    }
    if bytes.len() != capacity {
        return Err(mismatch("ASOF index encoded length differs"));
    }
    Ok(StateSegment::new(bytes))
}

fn write_header(writer: &mut BoundedWriter<'_>, state: &State) -> Result<()> {
    let left = (state.left.len() as u64).to_le_bytes();
    let right = (state.right.len() as u64).to_le_bytes();
    write_parts(writer, &[MAGIC, &left, &right])
}

fn write_left(
    writer: &mut BoundedWriter<'_>,
    time: i64,
    key: &Encoding,
    sequence: &Encoding,
    payload: &RowPayload,
) -> Result<()> {
    let time = time.to_le_bytes();
    let key_len = (key.len() as u64).to_le_bytes();
    let sequence_len = (sequence.len() as u64).to_le_bytes();
    let reference = ref_bytes(payload);
    write_parts(
        writer,
        &[
            &time,
            &key_len,
            key.as_slice(),
            &sequence_len,
            sequence.as_slice(),
            &reference,
        ],
    )
}

fn write_bucket_header(writer: &mut BoundedWriter<'_>, key: &Encoding, rows: usize) -> Result<()> {
    let key_len = (key.len() as u64).to_le_bytes();
    let rows = (rows as u64).to_le_bytes();
    write_parts(writer, &[&key_len, key.as_slice(), &rows])
}

fn write_right(
    writer: &mut BoundedWriter<'_>,
    time: i64,
    sequence: &Encoding,
    payload: Option<&RowPayload>,
) -> Result<()> {
    let time = time.to_le_bytes();
    let sequence_len = (sequence.len() as u64).to_le_bytes();
    let flag = [u8::from(payload.is_some())];
    if let Some(payload) = payload {
        let reference = ref_bytes(payload);
        write_parts(
            writer,
            &[&time, &sequence_len, sequence.as_slice(), &flag, &reference],
        )
    } else {
        write_parts(writer, &[&time, &sequence_len, sequence.as_slice(), &flag])
    }
}

fn ref_bytes(row: &RowPayload) -> [u8; 17] {
    let mut bytes = [0; 17];
    bytes[0] = row.batch.key.0;
    bytes[1..9].copy_from_slice(&row.batch.key.1.to_le_bytes());
    bytes[9..17].copy_from_slice(&(row.row as u64).to_le_bytes());
    bytes
}

fn write_parts(writer: &mut BoundedWriter<'_>, parts: &[&[u8]]) -> Result<()> {
    writer
        .write_parts(parts)
        .map_err(|error| mismatch(&error.to_string()))
}

struct Reader<'a> {
    bytes: &'a [u8],
    max_rows: u64,
    rows: u64,
}
impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if count > self.bytes.len() {
            return Err(mismatch("ASOF index is truncated"));
        }
        let (value, rest) = self.bytes.split_at(count);
        self.bytes = rest;
        Ok(value)
    }
    fn integer(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
    fn count(&mut self) -> Result<u64> {
        let count = self.integer()?;
        if count > self.max_rows {
            return Err(mismatch("ASOF index count exceeds row limit"));
        }
        Ok(count)
    }
    fn blob(&mut self) -> Result<Encoding> {
        let count = usize::try_from(self.integer()?)
            .map_err(|_| mismatch("ASOF index field exceeds address domain"))?;
        Ok(std::sync::Arc::new(self.take(count)?.to_vec()))
    }
    fn skip_blob(&mut self) -> Result<()> {
        let count = usize::try_from(self.integer()?)
            .map_err(|_| mismatch("ASOF index field exceeds address domain"))?;
        self.take(count)?;
        Ok(())
    }
    fn time(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
    fn row(&mut self) -> Result<()> {
        self.rows = self
            .rows
            .checked_add(1)
            .filter(|rows| *rows <= self.max_rows)
            .ok_or_else(|| mismatch("ASOF index has too many rows"))?;
        Ok(())
    }
    fn row_ref(
        &mut self,
        batches: &BTreeMap<BatchKey, std::sync::Arc<super::super::state::PayloadBatch>>,
        side: u8,
    ) -> Result<RowPayload> {
        let actual_side = self.take(1)?[0];
        let key = (actual_side, self.integer()?);
        let row = usize::try_from(self.integer()?)
            .map_err(|_| mismatch("ASOF row offset exceeds address domain"))?;
        let batch = batches
            .get(&key)
            .ok_or_else(|| mismatch("ASOF index references a missing batch"))?;
        if actual_side != side || row >= batch.record.num_rows() {
            return Err(mismatch("ASOF index references an invalid batch row"));
        }
        Ok(RowPayload {
            batch: batch.clone(),
            row,
        })
    }
}

/// Scan untrusted index bytes before any row/map allocation. The index bytes
/// cover copied identity buffers; per-row and bucket headroom is counted from
/// the actual encoded entries, independent of snapshot metrics.
pub(super) fn restore_charge(bytes: &[u8], max_rows: u64) -> Result<u64> {
    let mut reader = Reader {
        bytes,
        max_rows,
        rows: 0,
    };
    if reader.take(8)? != MAGIC {
        return Err(mismatch("ASOF index magic differs"));
    }
    let left = reader.count()?;
    let buckets = reader.count()?;
    for _ in 0..left {
        reader.row()?;
        reader.take(8)?;
        reader.skip_blob()?;
        reader.skip_blob()?;
        reader.take(17)?;
    }
    for _ in 0..buckets {
        reader.skip_blob()?;
        let count = reader.count()?;
        if count == 0 {
            return Err(mismatch("ASOF empty right bucket"));
        }
        for _ in 0..count {
            reader.row()?;
            reader.take(8)?;
            reader.skip_blob()?;
            let marker = reader.take(1)?[0];
            match marker {
                0 => {}
                1 => {
                    reader.take(17)?;
                }
                _ => return Err(mismatch("ASOF right payload marker differs")),
            }
        }
    }
    if !reader.bytes.is_empty() {
        return Err(mismatch("ASOF index has trailing data"));
    }
    let rows = reader.rows;
    let charge = (bytes.len() as u64)
        .checked_mul(2)
        .and_then(|value| value.checked_add(rows.checked_mul(384)?))
        .and_then(|value| value.checked_add(buckets.checked_mul(64)?))
        .ok_or_else(|| mismatch("ASOF index restore charge overflowed"))?;
    Ok(charge)
}

pub(super) fn decode(
    bytes: &[u8],
    batches: &BTreeMap<BatchKey, std::sync::Arc<super::super::state::PayloadBatch>>,
    max_rows: u64,
) -> Result<State> {
    let mut reader = Reader {
        bytes,
        max_rows,
        rows: 0,
    };
    if reader.take(8)? != MAGIC {
        return Err(mismatch("ASOF index magic differs"));
    }
    let left_count = reader.count()?;
    let bucket_count = reader.count()?;
    let mut state = State::default();
    for _ in 0..left_count {
        reader.row()?;
        let identity: LeftOrder = (reader.time()?, reader.blob()?, reader.blob()?);
        let payload = reader.row_ref(batches, 0)?;
        if state
            .left
            .last_key_value()
            .is_some_and(|(last, _)| last >= &identity)
        {
            return Err(mismatch("ASOF left index order is not strict"));
        }
        state.attach(&payload);
        state.left.insert(identity, payload);
    }
    for _ in 0..bucket_count {
        let key = reader.blob()?;
        if state
            .right
            .last_key_value()
            .is_some_and(|(last, _)| last >= &key)
        {
            return Err(mismatch("ASOF right bucket order is not strict"));
        }
        let count = reader.count()?;
        if count == 0 {
            return Err(mismatch("ASOF empty right bucket"));
        }
        let mut bucket = BTreeMap::new();
        for _ in 0..count {
            reader.row()?;
            let identity = (reader.time()?, reader.blob()?);
            let marker = reader.take(1)?[0];
            let payload = match marker {
                0 => None,
                1 => Some(reader.row_ref(batches, 1)?),
                _ => return Err(mismatch("ASOF right payload marker differs")),
            };
            if bucket
                .last_key_value()
                .is_some_and(|(last, _)| last >= &identity)
            {
                return Err(mismatch("ASOF right index order is not strict"));
            }
            if let Some(payload) = &payload {
                state.attach(payload);
            }
            bucket.insert(identity, payload);
        }
        state.right.insert(key, bucket);
    }
    if !reader.bytes.is_empty() || state.batches.len() != batches.len() {
        return Err(mismatch("ASOF index has trailing data or unused batches"));
    }
    Ok(state)
}
