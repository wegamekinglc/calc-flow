//! Columnar ASOF checkpoint index. Arrow payload batches live in separate,
//! immutable snapshot segments and are referenced by (side, batch id, row).

use super::{
    super::{
        checked,
        codec::BoundedWriter,
        state::{BatchKey, Encoding, LeftOrder, RightBucket, RowPayload, State},
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
type OwnedKeys = Vec<(Encoding, u32)>;

pub(super) fn workspace_bytes(state: &State, length: u64, owned: bool, name: &str) -> Result<u64> {
    let slot = if owned {
        size_of::<(Encoding, u32)>()
    } else {
        size_of::<u32>()
    };
    let sorting = (state.right.len() as u64)
        .checked_mul(slot as u64)
        .ok_or_else(|| {
            super::super::reason(
                name,
                crate::StreamingFailureReason::AsofCounterOverflow,
                "ASOF checkpoint key workspace overflowed",
            )
        })?;
    checked(name, length, sorting)
}

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
    let left = state
        .left
        .keys()
        .try_fold(0_u64, |size, (_, key, sequence)| {
            checked(name, size, 41 + key.len() as u64 + sequence.len() as u64)
        })?;
    checked(
        name,
        checked(name, 24, left)?,
        right_encoded_length(state, name)?,
    )
}

fn right_encoded_length(state: &State, name: &str) -> Result<u64> {
    let mut size = 0_u64;
    for (key, bucket) in &state.right {
        size = checked(name, size, 16 + key.len() as u64)?;
        size = checked(name, size, right_bucket_length(bucket, name)?)?;
    }
    Ok(size)
}

fn right_bucket_length(bucket: &RightBucket, name: &str) -> Result<u64> {
    bucket.iter().try_fold(0, |size, ((_, sequence), row)| {
        checked(
            name,
            size,
            17 + sequence.len() as u64 + if row.is_some() { 17 } else { 0 },
        )
    })
}

pub(super) async fn encode(
    state: &State,
    length: u64,
    limit: usize,
    context: &StreamOperatorContext<'_>,
    workspace: MemoryReservation,
) -> Result<(StateSegment, MemoryReservation)> {
    let (keys, workspace) = sort_right_keys(state, context, workspace).await?;
    let bytes = encode_bytes(state, length, limit, context, &keys).await?;
    drop(keys);
    let result = tokio::task::spawn_blocking(move || (StateSegment::new(bytes), workspace))
        .await
        .map_err(|error| CalcFlowError::Internal {
            message: format!("ASOF index digest task failed: {error}"),
        })?;
    context.check_cancelled()?;
    Ok(result)
}

async fn sort_right_keys(
    state: &State,
    context: &StreamOperatorContext<'_>,
    workspace: MemoryReservation,
) -> Result<(OwnedKeys, MemoryReservation)> {
    let mut keys = Vec::with_capacity(state.right.len());
    for (ordinal, (id, key)) in state.right.indexed_keys().enumerate() {
        keys.push((key.clone(), id));
        checkpoint_tick(ordinal + 1, context).await?;
    }
    context.check_cancelled()?;
    let result = sort_owned_keys(keys, workspace, |keys| {
        keys.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    })
    .await?;
    context.check_cancelled()?;
    Ok(result)
}

async fn sort_owned_keys(
    mut keys: OwnedKeys,
    workspace: MemoryReservation,
    sort: impl FnOnce(&mut OwnedKeys) + Send + 'static,
) -> Result<(OwnedKeys, MemoryReservation)> {
    tokio::task::spawn_blocking(move || {
        sort(&mut keys);
        (keys, workspace)
    })
    .await
    .map_err(|error| CalcFlowError::Internal {
        message: format!("ASOF checkpoint key sorting task failed: {error}"),
    })
}

async fn encode_bytes(
    state: &State,
    length: u64,
    limit: usize,
    context: &StreamOperatorContext<'_>,
    keys: &[(Encoding, u32)],
) -> Result<Vec<u8>> {
    let capacity = usize::try_from(length).expect("reserved address domain");
    let mut bytes = Vec::with_capacity(capacity);
    {
        let mut writer = BoundedWriter::with_capacity(&mut bytes, capacity, limit);
        write_header(&mut writer, state)?;
        context.check_cancelled()?;
        tokio::task::yield_now().await;
        write_left_async(&mut writer, state, context).await?;
        write_right_async(&mut writer, state, context, keys).await?;
        context.check_cancelled()?;
    }
    if bytes.len() != capacity {
        return Err(mismatch("ASOF index encoded length differs"));
    }
    Ok(bytes)
}

async fn write_left_async(
    writer: &mut BoundedWriter<'_>,
    state: &State,
    context: &StreamOperatorContext<'_>,
) -> Result<()> {
    for (ordinal, ((time, key, sequence), payload)) in state.left.iter().enumerate() {
        write_left(writer, *time, key, sequence, payload)?;
        checkpoint_tick(ordinal + 1, context).await?;
    }
    Ok(())
}

async fn write_right_async(
    writer: &mut BoundedWriter<'_>,
    state: &State,
    context: &StreamOperatorContext<'_>,
    keys: &[(Encoding, u32)],
) -> Result<()> {
    let mut ordinal = 0;
    for (key, id) in keys {
        let bucket = state.right.bucket_by_id(*id);
        write_bucket_header(writer, key, bucket.len())?;
        for ((time, sequence), payload) in bucket {
            write_right(writer, *time, sequence, payload)?;
            ordinal += 1;
            checkpoint_tick(ordinal, context).await?;
        }
    }
    Ok(())
}

async fn checkpoint_tick(ordinal: usize, context: &StreamOperatorContext<'_>) -> Result<()> {
    if ordinal % CHECK_EVERY == 0 {
        context.check_cancelled()?;
    }
    if ordinal % YIELD_EVERY == 0 {
        tokio::task::yield_now().await;
    }
    Ok(())
}

pub(super) fn encode_sync(state: &State, length: u64, limit: usize) -> Result<StateSegment> {
    let capacity = usize::try_from(length).expect("reserved address domain");
    let mut bytes = Vec::with_capacity(capacity);
    {
        let mut writer = BoundedWriter::with_capacity(&mut bytes, capacity, limit);
        write_header(&mut writer, state)?;
        write_state_sync(&mut writer, state)?;
    }
    if bytes.len() != capacity {
        return Err(mismatch("ASOF index encoded length differs"));
    }
    Ok(StateSegment::new(bytes))
}

fn write_state_sync(writer: &mut BoundedWriter<'_>, state: &State) -> Result<()> {
    for ((time, key, sequence), payload) in &state.left {
        write_left(writer, *time, key, sequence, payload)?;
    }
    for (key, bucket) in state.right.ordered_iter() {
        write_bucket_header(writer, key, bucket.len())?;
        for ((time, sequence), payload) in bucket {
            write_right(writer, *time, sequence, payload)?;
        }
    }
    Ok(())
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
        Ok(Encoding::from_slice(self.take(count)?))
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
        validate_row_ref(actual_side, side, row, batch.record.num_rows())?;
        Ok(RowPayload {
            batch: batch.clone(),
            row,
        })
    }
}

fn validate_row_ref(actual_side: u8, expected_side: u8, row: usize, rows: usize) -> Result<()> {
    if actual_side != expected_side || row >= rows {
        return Err(mismatch("ASOF index references an invalid batch row"));
    }
    Ok(())
}

/// Scan untrusted index bytes before any row/map allocation. The index bytes
/// cover copied identity buffers; per-row and bucket headroom is counted from
/// the actual encoded entries, independent of snapshot metrics.
pub(super) fn restore_charge(bytes: &[u8], max_rows: u64) -> Result<u64> {
    let (mut reader, left, buckets) = read_index_header(bytes, max_rows)?;
    scan_left_charge(&mut reader, left)?;
    scan_right_charge(&mut reader, buckets)?;
    ensure_consumed(&reader)?;
    index_restore_charge(bytes.len() as u64, reader.rows, buckets)
}

fn read_index_header(bytes: &[u8], max_rows: u64) -> Result<(Reader<'_>, u64, u64)> {
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
    if buckets > u64::from(u32::MAX) {
        return Err(mismatch("ASOF key count exceeds handle domain"));
    }
    Ok((reader, left, buckets))
}

fn scan_left_charge(reader: &mut Reader<'_>, left: u64) -> Result<()> {
    for _ in 0..left {
        reader.row()?;
        reader.take(8)?;
        reader.skip_blob()?;
        reader.skip_blob()?;
        reader.take(17)?;
    }
    Ok(())
}

fn scan_right_charge(reader: &mut Reader<'_>, buckets: u64) -> Result<()> {
    for _ in 0..buckets {
        reader.skip_blob()?;
        let count = reader.count()?;
        if count == 0 {
            return Err(mismatch("ASOF empty right bucket"));
        }
        scan_right_rows(reader, count)?;
    }
    Ok(())
}

fn scan_right_rows(reader: &mut Reader<'_>, count: u64) -> Result<()> {
    for _ in 0..count {
        reader.row()?;
        reader.take(8)?;
        reader.skip_blob()?;
        match reader.take(1)?[0] {
            0 => {}
            1 => {
                reader.take(17)?;
            }
            _ => return Err(mismatch("ASOF right payload marker differs")),
        }
    }
    Ok(())
}

fn ensure_consumed(reader: &Reader<'_>) -> Result<()> {
    if !reader.bytes.is_empty() {
        return Err(mismatch("ASOF index has trailing data"));
    }
    Ok(())
}

fn index_restore_charge(bytes: u64, rows: u64, buckets: u64) -> Result<u64> {
    let charge = bytes
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
    let (mut reader, left_count, bucket_count) = read_index_header(bytes, max_rows)?;
    let mut state = State::default();
    decode_left_index(&mut reader, left_count, batches, &mut state)?;
    decode_right_index(&mut reader, bucket_count, batches, &mut state)?;
    if !reader.bytes.is_empty() || state.batches.len() != batches.len() {
        return Err(mismatch("ASOF index has trailing data or unused batches"));
    }
    Ok(state)
}

fn decode_left_index(
    reader: &mut Reader<'_>,
    count: u64,
    batches: &BTreeMap<BatchKey, std::sync::Arc<super::super::state::PayloadBatch>>,
    state: &mut State,
) -> Result<()> {
    for _ in 0..count {
        let (identity, payload) = read_left_entry(reader, batches)?;
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
    Ok(())
}

fn read_left_entry(
    reader: &mut Reader<'_>,
    batches: &BTreeMap<BatchKey, std::sync::Arc<super::super::state::PayloadBatch>>,
) -> Result<(LeftOrder, RowPayload)> {
    reader.row()?;
    let identity: LeftOrder = (reader.time()?, reader.blob()?, reader.blob()?);
    Ok((identity, reader.row_ref(batches, 0)?))
}

fn decode_right_index(
    reader: &mut Reader<'_>,
    count: u64,
    batches: &BTreeMap<BatchKey, std::sync::Arc<super::super::state::PayloadBatch>>,
    state: &mut State,
) -> Result<()> {
    for _ in 0..count {
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
        let bucket = decode_right_bucket(reader, count, batches, state)?;
        state.right.insert(key, bucket);
    }
    Ok(())
}

fn decode_right_bucket(
    reader: &mut Reader<'_>,
    count: u64,
    batches: &BTreeMap<BatchKey, std::sync::Arc<super::super::state::PayloadBatch>>,
    state: &mut State,
) -> Result<RightBucket> {
    let mut bucket = RightBucket::new();
    for _ in 0..count {
        let (identity, payload) = read_right_entry(reader, batches)?;
        if bucket
            .last_key_value()
            .is_some_and(|(last, _)| last >= (&identity.0, &identity.1))
        {
            return Err(mismatch("ASOF right index order is not strict"));
        }
        if let Some(payload) = &payload {
            state.attach(payload);
        }
        bucket.insert(identity, payload);
    }
    Ok(bucket)
}

fn read_right_entry(
    reader: &mut Reader<'_>,
    batches: &BTreeMap<BatchKey, std::sync::Arc<super::super::state::PayloadBatch>>,
) -> Result<((i64, Encoding), Option<RowPayload>)> {
    reader.row()?;
    let identity = (reader.time()?, reader.blob()?);
    let payload = match reader.take(1)?[0] {
        0 => None,
        1 => Some(reader.row_ref(batches, 1)?),
        _ => return Err(mismatch("ASOF right payload marker differs")),
    };
    Ok((identity, payload))
}

#[cfg(test)]
mod sorting_tests {
    use super::*;
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
    use std::{
        sync::{Arc, Barrier},
        time::Duration,
    };

    #[tokio::test(flavor = "current_thread")]
    async fn dropped_key_sort_keeps_workspace_reserved_until_worker_exit() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4_096));
        let workspace = MemoryConsumer::new("key-sort-test").register(&pool);
        workspace.try_grow(1_024).unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let gate = Arc::new(Barrier::new(2));
        let worker_gate = gate.clone();
        let mut future = Box::pin(sort_owned_keys(vec![], workspace, move |_| {
            started_tx.send(()).unwrap();
            worker_gate.wait();
        }));
        assert!(futures::poll!(future.as_mut()).is_pending());
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(future);
        let while_running = pool.reserved();
        gate.wait();
        assert_eq!(while_running, 1_024);
        tokio::time::timeout(Duration::from_secs(5), async {
            while pool.reserved() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}
