//! Columnar indexes preserve owned capacities and canonical buffer aliases.
//! Capacity hints are validated before allocation and are never accepted as
//! accounting evidence: restored buffers are measured again by the runtime.

mod owned;
mod owners;

use super::mismatch;
use crate::operator::asof::{
    checked,
    state::{
        BatchKey, ChunkData, Encoding, EncodingOwners, PayloadBatch, PayloadPool,
        PreparedLeftChunk, RightBucket, RightState, RowPayload, RowRef, SequenceColumn,
        SequenceKind, State,
    },
};
use crate::{Result, StateSegment};
use owners::{OwnerReader, OwnerWriter};
use std::{collections::BTreeMap, sync::Arc};

pub(super) const INDEX_SEGMENT: &str = "asof-index-v3";
const MAGIC: &[u8; 8] = b"CFASOF03";
const BASE_BYTES: u64 = 80;
const LEFT_HEADER: u64 = 73;
const RIGHT_HEADER: u64 = 65;

pub(super) struct Cursor<'a> {
    bytes: &'a [u8],
    limit: u64,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], limit: u64) -> Self {
        Self { bytes, limit }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        if count > self.bytes.len() {
            return Err(mismatch("ASOF truncated v3 index"));
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

    fn small(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn address(&mut self) -> Result<usize> {
        usize::try_from(self.integer()?)
            .map_err(|_| mismatch("ASOF v3 capacity exceeds address domain"))
    }

    fn capacity(&mut self, width: usize) -> Result<usize> {
        let capacity = self.address()?;
        allocation(capacity, width, self.limit)?;
        Ok(capacity)
    }

    fn finish(&self) -> Result<()> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err(mismatch("ASOF v3 index contains trailing data"))
        }
    }
}

fn allocation(capacity: usize, width: usize, limit: u64) -> Result<u64> {
    (capacity as u64)
        .checked_mul(width as u64)
        .filter(|bytes| *bytes <= limit)
        .ok_or_else(|| mismatch("ASOF v3 allocation exceeds limits"))
}

fn put(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn source_owners(state: &State) -> OwnerWriter {
    let mut owners = OwnerWriter::default();
    for (_, data, head) in state.left.checkpoint_chunks(&state.batches) {
        let mut keys = data
            .keys
            .iter()
            .filter_map(Option::as_ref)
            .collect::<Vec<_>>();
        keys.sort_unstable();
        for key in keys {
            owners.register(key);
        }
        if data.sequences.kind() == SequenceKind::Canonical {
            for sequence in data.sequences.iter().skip(head) {
                owners.register(sequence.as_ref());
            }
        }
    }
    for (key, bucket) in state.right.ordered_iter() {
        owners.register(key);
        if state.sequence_kinds[1] == SequenceKind::Canonical {
            for ((_, sequence), _) in bucket {
                owners.register(sequence.as_ref());
            }
        }
    }
    owners
}

pub(in super::super) fn encoded_length(state: &State, name: &str) -> Result<u64> {
    if state.left.is_empty() && state.right.is_empty() {
        return Ok(0);
    }
    let owners = state
        .owners_encoded_length()
        .unwrap_or_else(|| source_owners(state).encoded_length());
    let mut bytes = checked(name, BASE_BYTES, owners)?;
    for (_, data, head) in state.left.checkpoint_chunks(&state.batches) {
        let rows = (data.sequences.len() - head) as u64;
        let keys = data.keys.iter().filter(|key| key.is_some()).count() as u64;
        let columns = rows
            .checked_mul(16 + state.sequence_kinds[0].reference_bytes())
            .ok_or_else(|| mismatch("ASOF v3 left index length overflowed"))?;
        bytes = checked(name, bytes, LEFT_HEADER + 16 * keys)?;
        bytes = checked(name, bytes, columns)?;
    }
    for bucket in state.right.values() {
        let rows = bucket.len() as u64;
        let columns = rows
            .checked_mul(9 + state.sequence_kinds[1].reference_bytes())
            .ok_or_else(|| mismatch("ASOF v3 right index length overflowed"))?;
        bytes = checked(name, bytes, RIGHT_HEADER)?;
        bytes = checked(
            name,
            bytes,
            checked(name, columns, bucket.payload_len() as u64 * 12)?,
        )?;
    }
    Ok(bytes)
}

pub(super) fn workspace_bytes(state: &State, length: u64, owned: bool, name: &str) -> Result<u64> {
    owned::workspace_bytes(state, length, owned, name)
}

pub(super) async fn encode(
    state: &State,
    length: u64,
    limit: usize,
    context: &crate::StreamOperatorContext<'_>,
    workspace: datafusion::execution::memory_pool::MemoryReservation,
) -> Result<(
    StateSegment,
    datafusion::execution::memory_pool::MemoryReservation,
)> {
    let snapshot = owned::Index::capture(state, context).await?;
    let (result, workspace) = tokio::task::spawn_blocking(move || {
        let result = snapshot.encode(length, limit);
        (result, workspace)
    })
    .await
    .map_err(|error| crate::CalcFlowError::Internal {
        message: format!("ASOF v3 index task failed: {error}"),
    })?;
    context.check_cancelled()?;
    Ok((result?, workspace))
}

pub(super) fn encode_sync(state: &State, length: u64, limit: usize) -> Result<StateSegment> {
    if !state.left.legacy.is_empty() {
        return Err(mismatch("ASOF v3 capture requires columnar left chunks"));
    }
    let capacity =
        usize::try_from(length).map_err(|_| mismatch("ASOF v3 index exceeds address domain"))?;
    if capacity > limit {
        return Err(mismatch("ASOF v3 index exceeds limits"));
    }
    let mut bytes = Vec::with_capacity(capacity);
    let owners = source_owners(state);
    bytes.extend_from_slice(MAGIC);
    put(
        &mut bytes,
        state.left.checkpoint_chunks(&state.batches).len() as u64,
    );
    put(&mut bytes, state.right.len() as u64);
    put(&mut bytes, state.left.len() as u64);
    let pool = state.batches.backing_buckets();
    let right = state.right.checkpoint_capacities();
    for value in [
        pool.0,
        pool.1,
        right[0],
        right[1],
        state.left.chunk_capacity(),
    ] {
        put(&mut bytes, value as u64);
    }
    owners.write(&mut bytes);
    for (batch, data, head) in state.left.checkpoint_chunks(&state.batches) {
        write_left(
            &mut bytes,
            batch,
            data,
            head,
            state.sequence_kinds[0],
            &owners,
        );
    }
    for (key, bucket) in state.right.ordered_iter() {
        write_right(&mut bytes, key, bucket, state, &owners);
    }
    if bytes.len() != capacity {
        return Err(mismatch("ASOF v3 encoded length differs"));
    }
    Ok(StateSegment::new(bytes))
}

fn write_sequence(
    bytes: &mut Vec<u8>,
    kind: SequenceKind,
    sequence: &Encoding,
    owners: &OwnerWriter,
) {
    if let Some(width) = kind.width() {
        bytes.extend_from_slice(&kind.integer_bytes(sequence)[..width]);
    } else {
        owners.reference(bytes, sequence);
    }
}

fn write_left(
    bytes: &mut Vec<u8>,
    batch: BatchKey,
    data: &ChunkData,
    head: usize,
    kind: SequenceKind,
    owners: &OwnerWriter,
) {
    let mut keys = data
        .keys
        .iter()
        .enumerate()
        .filter_map(|(id, key)| key.as_ref().map(|key| (id, key)))
        .collect::<Vec<_>>();
    keys.sort_unstable_by(|left, right| left.1.cmp(right.1));
    let mut remap = vec![0_u32; data.keys.len()];
    put(bytes, batch.1);
    put(bytes, (data.sequences.len() - head) as u64);
    put(bytes, keys.len() as u64);
    bytes.push(kind.flag());
    let capacities = [
        data.times.inner().capacity() / 8,
        data.positions.as_ref().map_or(0, Vec::capacity),
        data.keys.capacity(),
        data.key_counts.capacity(),
        data.key_ids.capacity(),
        data.sequences.capacity(),
    ];
    for capacity in capacities {
        put(bytes, capacity as u64);
    }
    for (id, (old, key)) in keys.into_iter().enumerate() {
        remap[old] = u32::try_from(id).expect("preflighted key domain");
        owners.reference(bytes, key);
    }
    #[cfg(target_endian = "little")]
    bytes.extend_from_slice(&data.times.inner().as_slice()[head * 8..]);
    #[cfg(target_endian = "big")]
    for time in &data.times[head..] {
        bytes.extend_from_slice(&time.to_le_bytes());
    }
    for key in &data.key_ids[head..] {
        bytes.extend_from_slice(&remap[*key as usize].to_le_bytes());
    }
    if let Some(values) = data.sequences.integer_slice(head..data.sequences.len()) {
        bytes.extend_from_slice(values);
    } else {
        for sequence in data.sequences.iter().skip(head) {
            write_sequence(bytes, kind, sequence.as_ref(), owners);
        }
    }
    for ordinal in head..data.sequences.len() {
        bytes.extend_from_slice(&data.position(ordinal).to_le_bytes());
    }
}

fn write_right(
    bytes: &mut Vec<u8>,
    key: &Encoding,
    bucket: &RightBucket,
    state: &State,
    owners: &OwnerWriter,
) {
    write_right_columns(bytes, key, bucket, state.sequence_kinds[1], owners, |row| {
        state.batches.key(row).1
    });
}

fn write_right_columns(
    bytes: &mut Vec<u8>,
    key: &Encoding,
    bucket: &RightBucket,
    kind: SequenceKind,
    owners: &OwnerWriter,
    batch_id: impl Fn(RowRef) -> u64,
) {
    owners.reference(bytes, key);
    put(bytes, bucket.len() as u64);
    bytes.push(kind.flag());
    for capacity in bucket.checkpoint_capacities() {
        put(bytes, capacity as u64);
    }
    if let Some((times, sequences)) = bucket.checkpoint_integer_columns() {
        for time in times {
            bytes.extend_from_slice(&time.to_le_bytes());
        }
        bytes.extend_from_slice(sequences);
        bytes.resize(
            bytes.len() + bucket.len(),
            u8::from(bucket.payload_len() != 0),
        );
        if let Some(rows) = bucket.checkpoint_payload_refs() {
            for row in rows.iter().flatten() {
                put(bytes, batch_id(*row));
                bytes.extend_from_slice(&row.row.to_le_bytes());
            }
        }
        return;
    }
    for ((time, _), _) in bucket {
        bytes.extend_from_slice(&time.to_le_bytes());
    }
    for ((_, sequence), _) in bucket {
        write_sequence(bytes, kind, sequence.as_ref(), owners);
    }
    for (_, _, tag) in bucket.checkpoint_rows() {
        bytes.push(tag);
    }
    for (_, row) in bucket {
        if let Some(row) = row {
            put(bytes, batch_id(*row));
            bytes.extend_from_slice(&row.row.to_le_bytes());
        }
    }
}

fn sequence_kind(cursor: &mut Cursor<'_>, expected: SequenceKind) -> Result<SequenceKind> {
    let kind = SequenceKind::from_flag(cursor.byte()?)
        .ok_or_else(|| mismatch("ASOF v3 sequence codec differs"))?;
    if kind != expected {
        return Err(mismatch("ASOF v3 sequence type differs"));
    }
    Ok(kind)
}

fn sequence(
    cursor: &mut Cursor<'_>,
    kind: SequenceKind,
    owners: &mut OwnerReader,
) -> Result<Encoding> {
    if let Some(width) = kind.width() {
        Ok(kind.decode_integer(cursor.take(width)?))
    } else {
        owners.reference(cursor)
    }
}

fn require_capacity(capacity: usize, count: usize) -> Result<()> {
    if capacity < count {
        Err(mismatch("ASOF v3 capacity is smaller than its live column"))
    } else {
        Ok(())
    }
}

struct Header {
    chunks: usize,
    buckets: usize,
    left_rows: usize,
    capacities: [usize; 5],
}

fn read_header(cursor: &mut Cursor<'_>, max_rows: u64) -> Result<Header> {
    if cursor.take(8)? != MAGIC {
        return Err(mismatch("ASOF v3 index magic differs"));
    }
    let chunks = cursor.address()?;
    let buckets = cursor.address()?;
    let left_rows = cursor.address()?;
    if left_rows as u64 > max_rows || chunks > left_rows || buckets as u64 > max_rows {
        return Err(mismatch("ASOF v3 counts exceed row limits"));
    }
    let mut capacities = [0; 5];
    for (capacity, width) in capacities.iter_mut().zip([25, 25, 40, 5, 32]) {
        *capacity = cursor.capacity(width)?;
    }
    validate_hash_capacity(capacities[0])?;
    validate_hash_capacity(capacities[1])?;
    validate_hash_capacity(capacities[3])?;
    require_capacity(capacities[2], buckets)?;
    require_capacity(capacities[4], chunks)?;
    if capacities[3] < buckets {
        return Err(mismatch("ASOF v3 hash capacity is too small"));
    }
    Ok(Header {
        chunks,
        buckets,
        left_rows,
        capacities,
    })
}

fn validate_hash_capacity(capacity: usize) -> Result<()> {
    if capacity == 0 || (capacity >= 4 && capacity.is_power_of_two()) {
        Ok(())
    } else {
        Err(mismatch("ASOF v3 hash backing capacity differs"))
    }
}

/// Scan all declared allocations without allocating index columns. The caller
/// reserves this checked total before decoding either indexes or payloads.
pub(super) fn restore_charge(bytes: &[u8], max_rows: u64, max_bytes: u64) -> Result<u64> {
    let mut cursor = Cursor::new(bytes, max_bytes);
    let header = read_header(&mut cursor, max_rows)?;
    let mut charge = 0;
    for (capacity, width) in header.capacities.into_iter().zip([25, 25, 40, 5, 32]) {
        charge = restore_add(charge, allocation(capacity, width, max_bytes)?)?;
    }
    charge = restore_add(
        charge,
        restore_add(512, allocation(header.buckets, 256, u64::MAX)?)?,
    )?;
    charge = restore_add(charge, owners::restore_charge(&mut cursor)?)?;
    let mut rows = 0_u64;
    for _ in 0..header.chunks {
        let (count, allocation) = scan_left(&mut cursor, max_rows)?;
        rows = restore_add(rows, count)?;
        charge = restore_add(charge, allocation)?;
    }
    if rows != header.left_rows as u64 {
        return Err(mismatch("ASOF v3 left count differs"));
    }
    for _ in 0..header.buckets {
        let (count, allocation) = scan_right(&mut cursor, max_rows - rows)?;
        rows = restore_add(rows, count)?;
        if rows > max_rows {
            return Err(mismatch("ASOF v3 row count exceeds limits"));
        }
        charge = restore_add(charge, allocation)?;
    }
    cursor.finish()?;
    Ok(charge)
}

fn restore_add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| mismatch("ASOF v3 decode workspace overflowed"))
}

fn skip_rows(cursor: &mut Cursor<'_>, rows: usize, width: usize) -> Result<()> {
    let bytes = rows
        .checked_mul(width)
        .ok_or_else(|| mismatch("ASOF v3 column bytes overflowed"))?;
    cursor.take(bytes)?;
    Ok(())
}

fn scan_kind(cursor: &mut Cursor<'_>) -> Result<SequenceKind> {
    SequenceKind::from_flag(cursor.byte()?)
        .ok_or_else(|| mismatch("ASOF v3 sequence codec differs"))
}

fn scan_left(cursor: &mut Cursor<'_>, remaining: u64) -> Result<(u64, u64)> {
    cursor.integer()?;
    let rows = cursor.address()?;
    let keys = cursor.address()?;
    let kind = scan_kind(cursor)?;
    if rows == 0 || rows as u64 > remaining || keys == 0 || keys > rows {
        return Err(mismatch("ASOF v3 left counts differ"));
    }
    let mut charge = 1_024_u64;
    for width in [
        8,
        4,
        size_of::<Option<Encoding>>(),
        8,
        4,
        kind.storage_bytes(),
    ] {
        let capacity = cursor.capacity(width)?;
        charge = restore_add(charge, allocation(capacity, width, cursor.limit)?)?;
    }
    skip_rows(cursor, keys, 16)?;
    skip_rows(cursor, rows, 16 + kind.storage_bytes())?;
    Ok((rows as u64, charge))
}

fn scan_right(cursor: &mut Cursor<'_>, remaining: u64) -> Result<(u64, u64)> {
    cursor.take(16)?;
    let rows = cursor.address()?;
    let kind = scan_kind(cursor)?;
    if rows == 0 || rows as u64 > remaining {
        return Err(mismatch("ASOF v3 right row count exceeds limits"));
    }
    let mut charge = 256_u64;
    for width in [8, kind.storage_bytes(), 8, 8, kind.storage_bytes()] {
        let capacity = cursor.capacity(width)?;
        charge = restore_add(charge, allocation(capacity, width, cursor.limit)?)?;
    }
    skip_rows(cursor, rows, 8 + kind.storage_bytes())?;
    let tags = cursor.take(rows)?;
    if tags.iter().any(|tag| *tag > 2) {
        return Err(mismatch("ASOF v3 storage tag differs"));
    }
    let payloads = tags
        .iter()
        .fold(0, |count, &tag| count + usize::from(tag == 1));
    let general = tags
        .iter()
        .fold(0, |count, &tag| count + usize::from(tag == 2));
    charge = restore_add(charge, allocation(general, 512, u64::MAX)?)?;
    skip_rows(cursor, payloads, 12)?;
    Ok((rows as u64, charge))
}

pub(super) fn decode(
    segment: &StateSegment,
    batches: &BTreeMap<BatchKey, Arc<PayloadBatch>>,
    max_rows: u64,
    max_bytes: u64,
    kinds: [SequenceKind; 2],
) -> Result<State> {
    let mut cursor = Cursor::new(segment.bytes(), max_bytes);
    let header = read_header(&mut cursor, max_rows)?;
    let mut owners = OwnerReader::read(&mut cursor)?;
    let mut state = State::empty_tracked();
    state.sequence_kinds = kinds;
    state.batches = PayloadPool::with_backing_buckets(header.capacities[0], header.capacities[1]);
    state.right = RightState::with_capacities(header.capacities[2], header.capacities[3]);
    state.left.reserve_chunks_exact(header.capacities[4]);
    let mut previous = None;
    for _ in 0..header.chunks {
        let chunk = read_left(
            &mut cursor,
            &mut owners,
            batches,
            kinds[0],
            max_rows,
            &mut previous,
        )?;
        state.left.install(vec![chunk], &mut state.batches);
    }
    if state.left.len() != header.left_rows {
        return Err(mismatch("ASOF v3 left row count differs"));
    }
    let mut rows = header.left_rows as u64;
    for _ in 0..header.buckets {
        let (key, bucket) = read_right(
            &mut cursor,
            &mut owners,
            batches,
            &mut state.batches,
            kinds[1],
            max_rows - rows,
        )?;
        rows += bucket.len() as u64;
        if state
            .right
            .last_key_value()
            .is_some_and(|(last, _)| last >= &key)
        {
            return Err(mismatch("ASOF v3 right key order is not strict"));
        }
        state.right.insert(key, bucket);
    }
    cursor.finish()?;
    owners.finish()?;
    if state.batches.len() != batches.len() {
        return Err(mismatch("ASOF v3 contains unreferenced payload batches"));
    }
    validate_left_order(&state)?;
    state.rebuild_encoding_owners();
    state.rebuild_right_minima();
    if state.batches.backing_buckets() != (header.capacities[0], header.capacities[1])
        || state.right.checkpoint_capacities() != [header.capacities[2], header.capacities[3]]
        || state.left.chunk_capacity() != header.capacities[4]
    {
        return Err(mismatch("ASOF v3 capacity reconstruction differs"));
    }
    Ok(state)
}

fn validate_left_order(state: &State) -> Result<()> {
    let mut previous = None;
    for (identity, _) in state.left.iter() {
        if previous.is_some_and(|last| last >= identity) {
            return Err(mismatch("ASOF v3 left identity order is not strict"));
        }
        previous = Some(identity);
    }
    Ok(())
}

fn left_index_inventory(
    times: &[i64],
    keys: &[Option<Encoding>],
    key_ids: &[u32],
    sequences: &SequenceColumn,
) -> Result<(u64, EncodingOwners)> {
    let mut encodings = EncodingOwners::default();
    for key in keys.iter().flatten() {
        encodings.attach(key);
    }
    let mut index_bytes = 0;
    for ordinal in 0..times.len() {
        let identity = (
            &times[ordinal],
            keys[key_ids[ordinal] as usize]
                .as_ref()
                .expect("validated key"),
            sequences.get(ordinal).expect("validated sequence"),
        );
        if ordinal > 0 {
            let last = (
                &times[ordinal - 1],
                keys[key_ids[ordinal - 1] as usize]
                    .as_ref()
                    .expect("validated key"),
                sequences.get(ordinal - 1).expect("validated sequence"),
            );
            if last >= identity {
                return Err(mismatch("ASOF v3 chunk identity order is not strict"));
            }
        }
        encodings.attach(identity.2.as_ref());
        index_bytes += 41 + identity.1.len() as u64 + identity.2.len() as u64;
    }
    Ok((index_bytes, encodings))
}

fn read_left_positions(
    cursor: &mut Cursor<'_>,
    rows: usize,
    capacity: usize,
    payload_rows: usize,
) -> Result<(Option<Vec<u32>>, u32)> {
    let mut positions = (capacity != 0).then(|| Vec::with_capacity(capacity));
    let mut start = 0;
    for ordinal in 0..rows {
        let row = cursor.small()?;
        if row as usize >= payload_rows {
            return Err(mismatch("ASOF v3 left payload row differs"));
        }
        if ordinal == 0 {
            start = row;
        }
        if let Some(positions) = &mut positions {
            positions.push(row);
        } else if u32::try_from(ordinal)
            .ok()
            .and_then(|ordinal| start.checked_add(ordinal))
            != Some(row)
        {
            return Err(mismatch("ASOF v3 implicit payload positions differ"));
        }
    }
    Ok((positions, start))
}

fn read_left(
    cursor: &mut Cursor<'_>,
    owners: &mut OwnerReader,
    batches: &BTreeMap<BatchKey, Arc<PayloadBatch>>,
    expected: SequenceKind,
    max_rows: u64,
    previous: &mut Option<BatchKey>,
) -> Result<PreparedLeftChunk> {
    let batch = (0, cursor.integer()?);
    if previous.is_some_and(|last| last >= batch) {
        return Err(mismatch("ASOF v3 left batch order is not strict"));
    }
    *previous = Some(batch);
    let rows = cursor.address()?;
    let count = cursor.address()?;
    if rows == 0 || rows as u64 > max_rows || count == 0 || count > rows {
        return Err(mismatch("ASOF v3 left chunk counts differ"));
    }
    let kind = sequence_kind(cursor, expected)?;
    let mut capacities = [0; 6];
    for (capacity, width) in capacities.iter_mut().zip([
        8,
        4,
        size_of::<Option<Encoding>>(),
        8,
        4,
        kind.storage_bytes(),
    ]) {
        *capacity = cursor.capacity(width)?;
    }
    for index in [0, 4, 5] {
        require_capacity(capacities[index], rows)?;
    }
    require_capacity(capacities[2], count)?;
    require_capacity(capacities[3], count)?;
    if capacities[1] != 0 {
        require_capacity(capacities[1], rows)?;
    }
    let mut keys = Vec::with_capacity(capacities[2]);
    let mut key_counts = Vec::with_capacity(capacities[3]);
    for _ in 0..count {
        let key = owners.reference(cursor)?;
        if keys
            .last()
            .and_then(Option::as_ref)
            .is_some_and(|last| last >= &key)
        {
            return Err(mismatch("ASOF v3 left key order is not strict"));
        }
        keys.push(Some(key));
        key_counts.push(0);
    }
    let mut times = Vec::with_capacity(capacities[0]);
    for _ in 0..rows {
        times.push(i64::from_le_bytes(
            cursor.take(8)?.try_into().expect("eight bytes"),
        ));
    }
    let mut key_ids = Vec::with_capacity(capacities[4]);
    for _ in 0..rows {
        let id = cursor.small()?;
        let references = key_counts
            .get_mut(id as usize)
            .ok_or_else(|| mismatch("ASOF v3 left key reference differs"))?;
        *references += 1;
        key_ids.push(id);
    }
    if key_counts.contains(&0) {
        return Err(mismatch("ASOF v3 contains an unused left key"));
    }
    let sequences = read_sequence_column(cursor, rows, capacities[5], kind, owners)?;
    let owner = batches
        .get(&batch)
        .ok_or_else(|| mismatch("ASOF v3 left payload batch is missing"))?;
    let (positions, start) =
        read_left_positions(cursor, rows, capacities[1], owner.record.num_rows())?;
    let (index_bytes, encodings) = left_index_inventory(&times, &keys, &key_ids, &sequences)?;
    Ok(PreparedLeftChunk::from_index(
        owner.clone(),
        ChunkData {
            times: times.into(),
            positions,
            start: if capacities[1] == 0 { start } else { 0 },
            keys,
            key_counts,
            key_ids,
            sequences,
            index_bytes,
            owners: encodings,
        },
    ))
}

fn read_sequence_column(
    cursor: &mut Cursor<'_>,
    rows: usize,
    capacity: usize,
    kind: SequenceKind,
    owners: &mut OwnerReader,
) -> Result<SequenceColumn> {
    if let Some(width) = kind.width() {
        let length = rows
            .checked_mul(width)
            .ok_or_else(|| mismatch("ASOF v3 sequence column length overflowed"))?;
        return Ok(SequenceColumn::from_integer_slice(
            cursor.take(length)?,
            capacity,
            kind,
        ));
    }
    let mut column = SequenceColumn::with_capacity(capacity, kind);
    for _ in 0..rows {
        column.push(owners.reference(cursor)?);
    }
    Ok(column)
}

fn read_right(
    cursor: &mut Cursor<'_>,
    owners: &mut OwnerReader,
    batches: &BTreeMap<BatchKey, Arc<PayloadBatch>>,
    pool: &mut PayloadPool,
    expected: SequenceKind,
    remaining: u64,
) -> Result<(Encoding, RightBucket)> {
    let key = owners.reference(cursor)?;
    let count = cursor.address()?;
    if count == 0 || count as u64 > remaining {
        return Err(mismatch("ASOF v3 right row count exceeds limits"));
    }
    let kind = sequence_kind(cursor, expected)?;
    let mut capacities = [0; 5];
    for (capacity, width) in
        capacities
            .iter_mut()
            .zip([8, kind.storage_bytes(), 8, 8, kind.storage_bytes()])
    {
        *capacity = cursor.capacity(width)?;
    }
    let times = cursor.take(
        count
            .checked_mul(8)
            .ok_or_else(|| mismatch("ASOF v3 time column length overflowed"))?,
    )?;
    let encoded = cursor.take(
        count
            .checked_mul(kind.storage_bytes())
            .ok_or_else(|| mismatch("ASOF v3 sequence column length overflowed"))?,
    )?;
    let mut sequences = Cursor::new(encoded, cursor.limit);
    let tags = cursor.take(count)?;
    let payloads = tags
        .iter()
        .fold(0, |count, &tag| count + usize::from(tag == 1));
    let identities = tags
        .iter()
        .fold(0, |count, &tag| count + usize::from(tag == 0));
    if tags.iter().any(|tag| *tag > 2) {
        return Err(mismatch("ASOF v3 storage tag differs"));
    }
    for capacity in &capacities[..3] {
        require_capacity(*capacity, payloads)?;
    }
    for capacity in &capacities[3..] {
        require_capacity(*capacity, identities)?;
    }
    if payloads == 0 && capacities[..3] != [0; 3] {
        return Err(mismatch("ASOF v3 contains empty payload storage"));
    }
    let mut bucket = RightBucket::with_index_capacities(capacities, kind);
    let mut previous = None;
    for (time, &tag) in times.chunks_exact(8).zip(tags) {
        let time = i64::from_le_bytes(time.try_into().expect("eight bytes"));
        let sequence = sequence(&mut sequences, kind, owners)?;
        let order = (time, sequence);
        if previous.as_ref().is_some_and(|last| last >= &order) {
            return Err(mismatch("ASOF v3 right identity order is not strict"));
        }
        previous = Some(order.clone());
        let row = if tag == 1 {
            let id = cursor.integer()?;
            let row = cursor.small()? as usize;
            let batch = batches
                .get(&(1, id))
                .ok_or_else(|| mismatch("ASOF v3 right payload batch is missing"))?;
            if row >= batch.record.num_rows() {
                return Err(mismatch("ASOF v3 right payload row differs"));
            }
            Some(pool.attach(&RowPayload {
                batch: batch.clone(),
                row,
            }))
        } else {
            None
        };
        bucket.push_index(order, tag, row);
    }
    sequences.finish()?;
    Ok((key, bucket))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::asof::{AsofJoinSide, state::encode_columns};
    use datafusion::arrow::{
        array::{Int64Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use std::sync::OnceLock;

    fn fixture() -> (State, BTreeMap<BatchKey, Arc<PayloadBatch>>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("time", DataType::Int64, false),
            Field::new("key", DataType::Int64, false),
            Field::new("sequence", DataType::Utf8, false),
        ]));
        let record = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![0, 1, 2])),
                Arc::new(Int64Array::from(vec![1, 2, 1])),
                Arc::new(StringArray::from(vec![
                    "a very long first sequence",
                    "another long second sequence",
                    "a long surviving sequence",
                ])),
            ],
        )
        .unwrap();
        let keys = encode_columns(&record, &["key".into()]).unwrap();
        let sequences = encode_columns(&record, &["sequence".into()]).unwrap();
        let mut state = State::empty_tracked();
        let mut batches = BTreeMap::new();
        let side = AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["sequence".into()],
            "left_".into(),
        )
        .unwrap();
        for index in 0..2 {
            let batch = Arc::new(PayloadBatch {
                key: (index, 0),
                record: Arc::new(record.clone()),
                encoded: OnceLock::new(),
                encoded_charge_bytes: 0,
                body_bytes: 0,
            });
            batches.insert(batch.key, batch.clone());
            let rows = (0..3)
                .map(|row| {
                    (
                        (row as i64, keys.row(row), sequences.row(row)),
                        RowPayload {
                            batch: batch.clone(),
                            row,
                        },
                    )
                })
                .collect::<Vec<_>>();
            if index == 0 {
                let chunks = PreparedLeftChunk::prepare(&rows, &side, "test").unwrap();
                state.left.install(chunks, &mut state.batches);
            } else {
                for ((time, key, sequence), payload) in rows {
                    let reference = state.attach(&payload);
                    state
                        .right
                        .bucket_mut_or_default(key)
                        .insert((time, sequence), Some(reference));
                }
            }
        }
        state.rebuild_encoding_owners();
        (state, batches)
    }

    #[test]
    fn columnar_prescan_rejects_overflowing_row_counts_without_panicking() {
        for right in [false, true] {
            let mut bytes = MAGIC.to_vec();
            for count in [u64::from(!right), u64::from(right), u64::from(!right)] {
                put(&mut bytes, count);
            }
            for capacity in [
                0,
                0,
                u64::from(right),
                if right { 4 } else { 0 },
                u64::from(!right),
            ] {
                put(&mut bytes, capacity);
            }
            put(&mut bytes, 0);
            if right {
                bytes.extend_from_slice(&[0; 16]);
            } else {
                put(&mut bytes, 0);
            }
            put(&mut bytes, u64::MAX);
            if !right {
                put(&mut bytes, 1);
            }
            bytes.push(0);
            for _ in 0..if right { 5 } else { 6 } {
                put(&mut bytes, 0);
            }
            assert!(restore_charge(&bytes, u64::MAX, u64::MAX).is_err());
        }
    }

    #[test]
    fn columnar_index_preserves_capacities_and_shared_owners() {
        let (state, batches) = fixture();
        let length = encoded_length(&state, "test").unwrap();
        let segment = encode_sync(&state, length, 1_000_000).unwrap();
        assert_eq!(&segment.bytes()[..8], MAGIC);
        let restored = decode(
            &segment,
            &batches,
            100,
            1_000_000,
            [SequenceKind::Canonical; 2],
        )
        .unwrap();
        assert_eq!(
            restored.capacity_inventory(None, "test").unwrap().bytes,
            state.capacity_inventory(None, "test").unwrap().bytes
        );
        let encoded = encode_sync(
            &restored,
            encoded_length(&restored, "test").unwrap(),
            1_000_000,
        )
        .unwrap();
        assert_eq!(encoded.bytes(), segment.bytes());
        let source = state
            .left
            .iter()
            .map(|(identity, row)| (*identity.0, identity.1.clone(), identity.2.clone(), row.row))
            .collect::<Vec<_>>();
        let actual = restored
            .left
            .iter()
            .map(|(identity, row)| (*identity.0, identity.1.clone(), identity.2.clone(), row.row))
            .collect::<Vec<_>>();
        assert_eq!(actual, source);
        let owners = restored
            .left
            .iter()
            .map(|(identity, _)| identity.2.allocation().unwrap().0)
            .collect::<Vec<_>>();
        assert!(owners.iter().all(|owner| *owner == owners[0]));
    }

    #[test]
    fn columnar_index_restores_a_partly_consumed_chunk_without_charging_dead_rows() {
        let (mut state, batches) = fixture();
        state.commit_left_prefix(1);
        state.rebuild_encoding_owners();
        let segment =
            encode_sync(&state, encoded_length(&state, "test").unwrap(), 1_000_000).unwrap();
        let restored = decode(
            &segment,
            &batches,
            100,
            1_000_000,
            [SequenceKind::Canonical; 2],
        )
        .unwrap();
        assert_eq!(restored.left.len(), 2);
        assert_eq!(
            restored
                .left
                .iter()
                .map(|(_, row)| row.row)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(
            restored.capacity_inventory(None, "test").unwrap().bytes,
            state.capacity_inventory(None, "test").unwrap().bytes
        );
        assert_eq!(
            encode_sync(
                &restored,
                encoded_length(&restored, "test").unwrap(),
                1_000_000
            )
            .unwrap()
            .bytes(),
            segment.bytes()
        );
    }

    #[test]
    fn capacity_admission_projection_matches_installed_shared_buffers() {
        let (source, batches) = fixture();
        let mut state = State::empty_tracked();
        let side = AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["sequence".into()],
            "left_".into(),
        )
        .unwrap();
        let rows = source
            .left
            .iter()
            .map(|((time, key, sequence), row)| {
                (
                    (*time, key.clone(), sequence.into_owned()),
                    RowPayload {
                        batch: batches[&(0, 0)].clone(),
                        row: row.row as usize,
                    },
                )
            })
            .collect::<Vec<_>>();
        let chunks = PreparedLeftChunk::prepare(&rows, &side, "test").unwrap();
        let (length, projected, owners) = state
            .project_capacity_admission(
                state.capacity_snapshot("test"),
                &rows,
                Some(&chunks),
                &[],
                &[rows[0].1.batch.clone()],
                "test",
            )
            .unwrap();
        state.left.install(chunks, &mut state.batches);
        state.install_encoding_owners(owners);
        assert_eq!(length, encoded_length(&state, "test").unwrap());
        assert_eq!(
            projected.bytes,
            state.capacity_inventory(None, "test").unwrap().bytes + length + 256
        );
        let mut counts = BTreeMap::<Encoding, usize>::new();
        let rows = rows
            .into_iter()
            .map(|(identity, mut row)| {
                *counts.entry(identity.1.clone()).or_default() += 1;
                row.batch = batches[&(1, 0)].clone();
                (identity, row)
            })
            .collect::<Vec<_>>();
        let counts = counts.into_iter().collect::<Vec<_>>();
        let (length, projected, owners) = state
            .project_capacity_admission(
                state.capacity_snapshot("test"),
                &rows,
                None,
                &counts,
                &[rows[0].1.batch.clone()],
                "test",
            )
            .unwrap();
        for (key, count) in counts {
            state
                .right
                .bucket_mut_or_default(key)
                .reserve_payloads(count);
        }
        for ((time, key, sequence), row) in rows {
            let reference = state.attach(&row);
            state
                .right
                .bucket_mut_or_default(key)
                .insert_admitted((time, sequence), reference);
        }
        state.install_encoding_owners(owners);
        assert_eq!(length, encoded_length(&state, "test").unwrap());
        assert_eq!(
            projected.bytes,
            state.capacity_inventory(None, "test").unwrap().bytes + length + 256
        );
    }

    #[test]
    fn capacity_prefix_projection_matches_commit_without_serializing_rows() {
        let (mut state, _) = fixture();
        for count in [1, 2] {
            let mut prefix = crate::operator::asof::state::LeftPrefix::default();
            for (identity, row) in state.left.iter().take(count) {
                prefix
                    .visit(&identity, row, &state.batches, "test")
                    .unwrap();
            }
            let drain = state
                .left
                .drain_input(&prefix.batches, &state.batches)
                .prepare();
            let (length, projected, _) = state
                .project_capacity_prefix(state.capacity_snapshot("test"), &prefix, &drain, "test")
                .unwrap();
            state.commit_matched_left_prefix(&prefix, drain);
            assert_eq!(length, encoded_length(&state, "test").unwrap());
            assert_eq!(
                projected.bytes,
                state.capacity_inventory(None, "test").unwrap().bytes + length + 256
            );
        }
    }

    #[test]
    fn capacity_eviction_projection_matches_payload_and_identity_transitions() {
        use crate::EventTime;
        use crate::operator::asof::StreamAsofJoinStatus;
        for (left, right) in [(0, 0), (1, 0), (3, 0), (3, 1), (3, 3), (i64::MAX, i64::MAX)] {
            let (mut state, _) = fixture();
            state.commit_left_prefix(3);
            let mut status = StreamAsofJoinStatus::default();
            status.left.watermark_micros = Some(EventTime::from_micros(left));
            status.right.watermark_micros = Some(EventTime::from_micros(right));
            let preview = state.preview_eviction(&status, 0, "test").unwrap();
            let (length, projected, _) = state
                .project_capacity_eviction(
                    state.capacity_snapshot("test"),
                    &preview,
                    &status,
                    0,
                    "test",
                )
                .unwrap();
            state.evict(&status, 0);
            assert_eq!(
                length,
                encoded_length(&state, "test").unwrap(),
                "frontiers=({left},{right})"
            );
            assert_eq!(
                projected.bytes,
                state.capacity_inventory(None, "test").unwrap().bytes
                    + if length == 0 { 0 } else { length + 256 },
                "frontiers=({left},{right})"
            );
        }
    }

    #[test]
    fn columnar_restore_reservation_bounds_its_peak_index_allocations() {
        let (state, batches) = fixture();
        let segment =
            encode_sync(&state, encoded_length(&state, "test").unwrap(), 1_000_000).unwrap();
        let charge = restore_charge(segment.bytes(), 100, 1_000_000).unwrap();
        let mut restored = None;
        let measured = allocation_counter::measure(|| {
            restored = Some(
                decode(
                    &segment,
                    &batches,
                    100,
                    1_000_000,
                    [SequenceKind::Canonical; 2],
                )
                .unwrap(),
            );
        });
        assert!(
            measured.bytes_max as u64 <= charge,
            "charge={charge}, {measured:?}"
        );
        assert!(restored.is_some());
    }

    #[test]
    fn columnar_index_rejects_padding_truncation_and_unknown_sequence_codecs() {
        let (state, batches) = fixture();
        let segment =
            encode_sync(&state, encoded_length(&state, "test").unwrap(), 1_000_000).unwrap();
        let mut padding = segment.bytes().to_vec();
        padding[81] = 1;
        assert!(
            decode(
                &StateSegment::new(padding),
                &batches,
                100,
                1_000_000,
                [SequenceKind::Canonical; 2]
            )
            .is_err()
        );
        assert!(
            restore_charge(
                &segment.bytes()[..segment.bytes().len() - 1],
                100,
                1_000_000
            )
            .is_err()
        );
        let mut extra = segment.bytes().to_vec();
        extra.push(0);
        assert!(
            decode(
                &StateSegment::new(extra),
                &batches,
                100,
                1_000_000,
                [SequenceKind::Canonical; 2]
            )
            .is_err()
        );
        for flag in [9, 127, 255] {
            assert!(SequenceKind::from_flag(flag).is_none());
        }
    }

    #[test]
    fn columnar_index_rejects_huge_capacity_before_allocating_it() {
        let (state, batches) = fixture();
        let segment =
            encode_sync(&state, encoded_length(&state, "test").unwrap(), 1_000_000).unwrap();
        let mut bytes = segment.bytes().to_vec();
        bytes[32..40].copy_from_slice(&u64::MAX.to_le_bytes());
        let allocations = allocation_counter::measure(|| {
            assert!(
                decode(
                    &StateSegment::new(bytes.clone()),
                    &batches,
                    100,
                    1_000_000,
                    [SequenceKind::Canonical; 2]
                )
                .is_err()
            );
        });
        assert!(allocations.bytes_max < 16_384, "{allocations:?}");
    }
}
