use super::{
    Arc, BatchKey, BucketCut, Change, ChunkData, DecodedDelta, Encoding, Identity,
    MemoryReservation, OwnerReader, PayloadBatch, PreparedLeftChunk, Result, SequenceKind, Version,
    mismatch,
};
use crate::operator::asof::state::{
    EncodingOwners, PayloadPool, RightBucket, RightState, RowPayload, SequenceColumn, State,
};
use std::collections::{BTreeMap, BTreeSet};

struct Left {
    owner: Arc<PayloadBatch>,
    data: Arc<ChunkData>,
    head: usize,
    capacities: [usize; 6],
}

struct Right {
    rows: Vec<((i64, Encoding), Version)>,
    edits: BTreeMap<(i64, Encoding), Option<Version>>,
    len: usize,
    capacities: [usize; 5],
}

impl Right {
    fn get(&self, order: &(i64, Encoding)) -> Option<Version> {
        self.edits.get(order).copied().unwrap_or_else(|| {
            self.rows
                .binary_search_by(|(key, _)| key.cmp(order))
                .ok()
                .map(|index| self.rows[index].1)
        })
    }
    fn iter(&self) -> impl Iterator<Item = (&(i64, Encoding), Version)> {
        let mut rows = self.rows.iter().peekable();
        let mut edits = self.edits.iter().peekable();
        std::iter::from_fn(move || {
            loop {
                match (rows.peek(), edits.peek()) {
                    (None, None) => return None,
                    (Some(_), None) => {
                        return rows.next().map(|(order, version)| (order, *version));
                    }
                    (None, Some(_)) => {
                        let (order, version) = edits.next().expect("peeked edit");
                        if let Some(version) = version {
                            return Some((order, *version));
                        }
                    }
                    (Some((row, _)), Some((edit, _))) => match row.cmp(*edit) {
                        std::cmp::Ordering::Less => {
                            return rows.next().map(|(order, version)| (order, *version));
                        }
                        std::cmp::Ordering::Equal => {
                            rows.next();
                            let (order, version) = edits.next().expect("peeked edit");
                            if let Some(version) = version {
                                return Some((order, *version));
                            }
                        }
                        std::cmp::Ordering::Greater => {
                            let (order, version) = edits.next().expect("peeked edit");
                            if let Some(version) = version {
                                return Some((order, *version));
                            }
                        }
                    },
                }
            }
        })
    }
}

pub(in crate::operator::asof) struct Model {
    capacities: [usize; 16],
    counts: [usize; 3],
    kinds: [SequenceKind; 2],
    left: BTreeMap<BatchKey, Left>,
    right: BTreeMap<Encoding, Right>,
    workspaces: Vec<MemoryReservation>,
}

pub(in crate::operator::asof) fn capacities(state: &State) -> [usize; 16] {
    let pool = state.batches.backing_buckets();
    let right = state.right.checkpoint_capacities();
    let heaps = state.right.heap_capacities();
    let mut values = [0; 16];
    values[..8].copy_from_slice(&[
        pool.0,
        pool.1,
        right[0],
        right[1],
        state.left.chunk_capacity(),
        heaps[0],
        heaps[1],
        heaps[2],
    ]);
    values[8..].copy_from_slice(&state.right.shard_capacities());
    values
}

pub(in crate::operator::asof) fn counts(state: &State) -> [usize; 3] {
    [
        state.left.checkpoint_chunks(&state.batches).len(),
        state.right.len(),
        state.left.len(),
    ]
}

impl Model {
    pub fn from_state(
        state: &State,
        workspace: MemoryReservation,
        cancel: &dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        cancel()?;
        validate_model_workspace(state, &workspace)?;
        let left = capture_model_left(state, cancel)?;
        let right = capture_model_right(state, cancel)?;
        let mut workspaces = Vec::with_capacity(33);
        workspaces.push(workspace);
        Ok(Self {
            capacities: capacities(state),
            counts: counts(state),
            kinds: state.sequence_kinds,
            left,
            right,
            workspaces,
        })
    }

    fn apply_left(
        &mut self,
        batch: BatchKey,
        before: Option<Version>,
        after: Option<Version>,
        installed: &mut BTreeMap<BatchKey, PreparedLeftChunk>,
    ) -> Result<()> {
        let actual = self.left.get(&batch).map(|chunk| Version::Left {
            rows: (chunk.data.sequences.len() - chunk.head) as u64,
            capacities: chunk.capacities,
        });
        if actual != before {
            return Err(mismatch("ASOF log left predecessor differs"));
        }
        match after {
            None => {
                self.left.remove(&batch);
            }
            Some(Version::Left { rows, capacities }) => {
                if let Some(chunk) = self.left.get_mut(&batch) {
                    chunk.head = chunk.data.sequences.len()
                        - usize::try_from(rows)
                            .map_err(|_| mismatch("ASOF log left suffix exceeds address domain"))?;
                    chunk.capacities = capacities;
                } else {
                    let (owner, data) = installed
                        .remove(&batch)
                        .ok_or_else(|| mismatch("ASOF log installed left chunk is missing"))?
                        .into_parts();
                    self.left.insert(
                        batch,
                        Left {
                            owner,
                            data: Arc::new(data),
                            head: 0,
                            capacities,
                        },
                    );
                }
            }
            _ => return Err(mismatch("ASOF log left version differs")),
        }
        Ok(())
    }

    pub fn apply(
        &mut self,
        delta: DecodedDelta,
        mut cancelled: impl FnMut() -> Result<()>,
    ) -> Result<OwnerReader> {
        cancelled()?;
        let DecodedDelta {
            capacities,
            counts,
            changes,
            left,
            buckets,
            owners,
            workspace,
        } = delta;
        self.workspaces.push(workspace);
        let mut installed = left.into_iter().collect::<BTreeMap<_, _>>();
        self.apply_changes(changes, &mut installed, &mut cancelled)?;
        self.apply_cuts(buckets, &mut cancelled)?;
        self.capacities = capacities;
        self.counts = counts;
        cancelled()?;
        Ok(owners)
    }

    fn apply_changes(
        &mut self,
        changes: Vec<Change>,
        installed: &mut BTreeMap<BatchKey, PreparedLeftChunk>,
        cancelled: &mut impl FnMut() -> Result<()>,
    ) -> Result<()> {
        for (ordinal, change) in changes.into_iter().enumerate() {
            if ordinal.is_multiple_of(128) {
                cancelled()?;
            }
            self.apply_change(change, installed)?;
        }
        if !installed.is_empty() {
            return Err(mismatch("ASOF log has extra installed left chunks"));
        }
        Ok(())
    }

    fn apply_change(
        &mut self,
        change: Change,
        installed: &mut BTreeMap<BatchKey, PreparedLeftChunk>,
    ) -> Result<()> {
        match change.identity {
            Identity::Left(batch) => {
                self.apply_left(batch, change.before, change.after, installed)?;
            }
            Identity::Right((time, key, sequence)) => {
                self.apply_right((time, sequence), key, change.before, change.after)?;
            }
        }
        Ok(())
    }

    fn apply_right(
        &mut self,
        order: (i64, Encoding),
        key: Encoding,
        before: Option<Version>,
        after: Option<Version>,
    ) -> Result<()> {
        let actual = self.right.get(&key).and_then(|bucket| bucket.get(&order));
        if actual != before {
            return Err(mismatch("ASOF log right predecessor differs"));
        }
        match after {
            None => {
                let bucket = self
                    .right
                    .get_mut(&key)
                    .ok_or_else(|| mismatch("ASOF log deleted right bucket is missing"))?;
                bucket.edits.insert(order, None);
                bucket.len -= 1;
            }
            Some(version) => {
                let bucket = self.right.entry(key).or_insert_with(|| Right {
                    rows: Vec::new(),
                    edits: BTreeMap::new(),
                    len: 0,
                    capacities: [0; 5],
                });
                if actual.is_none() {
                    bucket.len += 1;
                }
                bucket.edits.insert(order, Some(version));
            }
        }
        Ok(())
    }

    fn apply_cuts(
        &mut self,
        buckets: Vec<BucketCut>,
        cancelled: &mut impl FnMut() -> Result<()>,
    ) -> Result<()> {
        for (ordinal, cut) in buckets.into_iter().enumerate() {
            if ordinal.is_multiple_of(128) {
                cancelled()?;
            }
            self.apply_bucket_cut(&cut, cancelled)?;
        }
        Ok(())
    }

    fn apply_bucket_cut(
        &mut self,
        cut: &BucketCut,
        cancelled: &mut impl FnMut() -> Result<()>,
    ) -> Result<()> {
        let actual = self.right.get(&cut.key);
        match cut.state {
            None if actual.is_none_or(|bucket| bucket.len == 0) => {
                self.right.remove(&cut.key);
            }
            Some((rows, capacities)) if actual.is_some_and(|bucket| bucket.len as u64 == rows) => {
                let counts = tag_counts(actual.expect("validated bucket"), cancelled)?;
                super::super::validate_right_capacities(capacities, counts)?;
                self.right
                    .get_mut(&cut.key)
                    .expect("validated bucket")
                    .capacities = capacities;
            }
            _ => return Err(mismatch("ASOF log final bucket census differs")),
        }
        Ok(())
    }

    pub fn into_workspaces(self) -> Vec<MemoryReservation> {
        let Self { workspaces, .. } = self;
        workspaces
    }

    pub fn native_charge(&self, mut cancel: impl FnMut() -> Result<()>) -> Result<u64> {
        cancel()?;
        let mut bytes = 4096;
        for (capacity, width) in self
            .capacities
            .into_iter()
            .zip(super::super::CAPACITY_WIDTHS)
        {
            bytes = super::super::restore_add(
                bytes,
                super::super::allocation(capacity, width, u64::MAX)?,
            )?;
        }
        let mut allocations = BTreeSet::new();
        for (key, bucket) in &self.right {
            if let Some((id, _)) = key.allocation() {
                allocations.insert(id);
            }
            let counts = tag_counts(bucket, &mut cancel)?;
            super::super::validate_right_capacities(bucket.capacities, counts)?;
            bytes = super::super::restore_add(bytes, 256 + counts[2] as u64 * 512)?;
            for (capacity, width) in bucket.capacities.into_iter().zip([
                8,
                self.kinds[1].storage_bytes(),
                8,
                8,
                self.kinds[1].storage_bytes(),
            ]) {
                bytes = super::super::restore_add(
                    bytes,
                    super::super::allocation(capacity, width, u64::MAX)?,
                )?;
            }
            for (ordinal, ((_, sequence), _)) in bucket.iter().enumerate() {
                if ordinal.is_multiple_of(128) {
                    cancel()?;
                }
                if let Some((id, _)) = sequence.allocation() {
                    allocations.insert(id);
                }
            }
        }
        for chunk in self.left.values() {
            for (capacity, width) in chunk.capacities.into_iter().zip([
                8,
                4,
                size_of::<Option<Encoding>>(),
                8,
                4,
                self.kinds[0].storage_bytes(),
            ]) {
                bytes = super::super::restore_add(
                    bytes,
                    super::super::allocation(capacity, width, u64::MAX)?,
                )?;
            }
            let rows = chunk.data.sequences.len() - chunk.head;
            bytes = super::super::restore_add(bytes, 1024 + rows as u64 * 256)?;
            for (ordinal, key) in chunk.data.keys.iter().flatten().enumerate() {
                if ordinal.is_multiple_of(128) {
                    cancel()?;
                }
                if let Some((id, _)) = key.allocation() {
                    allocations.insert(id);
                }
            }
            for (ordinal, sequence) in chunk.data.sequences.iter().skip(chunk.head).enumerate() {
                if ordinal.is_multiple_of(128) {
                    cancel()?;
                }
                if let Some((id, _)) = sequence.allocation() {
                    allocations.insert(id);
                }
            }
        }
        super::super::restore_add(bytes, allocations.len() as u64 * 256)
    }

    pub fn materialize(
        &self,
        batches: &BTreeMap<BatchKey, Arc<PayloadBatch>>,
        workspace: &MemoryReservation,
        max_rows: u64,
        cancelled: &dyn Fn() -> Result<()>,
    ) -> Result<State> {
        cancelled()?;
        if (workspace.size() as u64) < self.native_charge(&mut || cancelled())? {
            return Err(mismatch("ASOF native replay exceeds prepaid workspace"));
        }
        if self.left.len() != self.counts[0] || self.right.len() != self.counts[1] {
            return Err(mismatch("ASOF final native container census differs"));
        }
        let rows = self
            .left
            .values()
            .map(|chunk| chunk.data.sequences.len() - chunk.head)
            .sum::<usize>();
        let right_rows = self.right.values().map(|bucket| bucket.len).sum::<usize>();
        if rows != self.counts[2] || rows as u64 + right_rows as u64 > max_rows {
            return Err(mismatch("ASOF final native row census exceeds limits"));
        }
        let mut state = State::empty_tracked();
        state.sequence_kinds = self.kinds;
        state.batches = PayloadPool::with_backing_buckets(self.capacities[0], self.capacities[1]);
        state.right = RightState::with_index_capacities(
            self.capacities[2],
            self.capacities[3],
            self.capacities[5..8].try_into().expect("three heaps"),
            self.capacities[8..].try_into().expect("eight shards"),
        );
        state.left.reserve_chunks_exact(self.capacities[4]);
        let mut left = Vec::with_capacity(self.left.len());
        for chunk in self.left.values() {
            cancelled()?;
            left.push(PreparedLeftChunk::from_index(
                chunk.owner.clone(),
                copy_left(chunk, self.kinds[0], cancelled)?,
            ));
        }
        state.left.install(left, &mut state.batches);
        for (key, source) in &self.right {
            cancelled()?;
            let mut bucket = RightBucket::with_index_capacities(source.capacities, self.kinds[1]);
            for (ordinal, (order, version)) in source.iter().enumerate() {
                if ordinal.is_multiple_of(128) {
                    cancelled()?;
                }
                let Version::Right { tag, payload } = version else {
                    return Err(mismatch("ASOF model right version differs"));
                };
                let row = payload
                    .map(|(batch, row)| {
                        let batch = batches
                            .get(&batch)
                            .ok_or_else(|| mismatch("ASOF model payload batch is missing"))?;
                        if row as usize >= batch.record.num_rows() {
                            return Err(mismatch("ASOF model payload position differs"));
                        }
                        Ok(state.batches.attach(&RowPayload {
                            batch: batch.clone(),
                            row: row as usize,
                        }))
                    })
                    .transpose()?;
                bucket.push_index(order.clone(), tag, row);
            }
            if !state.right.can_insert_restored(key) {
                return Err(mismatch("ASOF log right shard capacity is too small"));
            }
            state.right.insert_restored(key.clone(), bucket);
        }
        state.rebuild_encoding_owners_checked(&mut || cancelled())?;
        [
            state.right_payload_min,
            state.right_identity_min,
            state.right_dominance_min,
        ] = state.right.minima();
        if capacities(&state) != self.capacities || counts(&state) != self.counts {
            return Err(mismatch(
                "ASOF native replay capacity reconstruction differs",
            ));
        }
        super::super::validate_left_order(&state, cancelled)?;
        cancelled()?;
        Ok(state)
    }
}

fn validate_model_workspace(state: &State, workspace: &MemoryReservation) -> Result<()> {
    let rows = state
        .right
        .values()
        .map(RightBucket::len)
        .sum::<usize>()
        .checked_add(state.left.len())
        .ok_or_else(|| mismatch("ASOF model row count overflowed"))?;
    if workspace.size()
        < rows
            .saturating_mul(size_of::<((i64, Encoding), Version)>())
            .saturating_add(state.right.len().saturating_mul(512))
            .saturating_add(
                state
                    .left
                    .checkpoint_chunks(&state.batches)
                    .len()
                    .saturating_mul(512),
            )
            .saturating_add(4096)
    {
        return Err(mismatch("ASOF model exceeds prepaid workspace"));
    }
    Ok(())
}

fn capture_model_left(
    state: &State,
    cancel: &dyn Fn() -> Result<()>,
) -> Result<BTreeMap<BatchKey, Left>> {
    let left = state
        .left
        .checkpoint_owned_chunks(&state.batches)
        .enumerate()
        .map(|(ordinal, (batch, data, head))| {
            super::super::check_step(ordinal, cancel)?;
            let capacities = data.checkpoint_capacities();
            Ok((
                batch,
                Left {
                    owner: state.batches[&batch].0.clone(),
                    data,
                    head,
                    capacities,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    Ok(left)
}

fn capture_model_right(
    state: &State,
    cancel: &dyn Fn() -> Result<()>,
) -> Result<BTreeMap<Encoding, Right>> {
    let mut right = BTreeMap::new();
    for (key, bucket) in state.right.ordered_iter() {
        cancel()?;
        let mut rows = Vec::with_capacity(bucket.len());
        for (ordinal, ((time, sequence), payload, tag)) in bucket.checkpoint_rows().enumerate() {
            super::super::check_step(ordinal, cancel)?;
            let payload = payload.map(|row| (state.batches.key(*row), row.row));
            rows.push((
                (*time, sequence.into_owned()),
                Version::Right { tag, payload },
            ));
        }
        right.insert(
            key.clone(),
            Right {
                rows,
                edits: BTreeMap::new(),
                len: bucket.len(),
                capacities: bucket.checkpoint_capacities(),
            },
        );
    }
    Ok(right)
}

fn tag_counts(bucket: &Right, cancelled: &mut impl FnMut() -> Result<()>) -> Result<[usize; 3]> {
    let mut counts = [0; 3];
    for (ordinal, (_, version)) in bucket.iter().enumerate() {
        if ordinal.is_multiple_of(128) {
            cancelled()?;
        }
        let Version::Right { tag, .. } = version else {
            return Err(mismatch("ASOF model right storage differs"));
        };
        *counts
            .get_mut(usize::from(tag))
            .ok_or_else(|| mismatch("ASOF model right tag differs"))? += 1;
    }
    Ok(counts)
}

fn copy_left(
    chunk: &Left,
    kind: SequenceKind,
    cancelled: &dyn Fn() -> Result<()>,
) -> Result<ChunkData> {
    let rows = chunk.data.sequences.len() - chunk.head;
    let capacities = chunk.capacities;
    let mut dictionary = BTreeMap::<Encoding, u32>::new();
    for ordinal in chunk.head..chunk.data.sequences.len() {
        if ordinal.is_multiple_of(128) {
            cancelled()?;
        }
        let key = chunk.data.keys[chunk.data.key_ids[ordinal] as usize]
            .as_ref()
            .ok_or_else(|| mismatch("ASOF model left key is missing"))?;
        dictionary.entry(key.clone()).or_default();
    }
    for (id, value) in dictionary.values_mut().enumerate() {
        *value = u32::try_from(id)
            .map_err(|_| mismatch("ASOF model left key count exceeds handle domain"))?;
    }
    for index in [0, 4, 5] {
        super::super::require_capacity(capacities[index], rows)?;
    }
    for index in [2, 3] {
        super::super::require_capacity(capacities[index], dictionary.len())?;
    }
    if capacities[1] != 0 {
        super::super::require_capacity(capacities[1], rows)?;
    }
    let mut times = Vec::with_capacity(capacities[0]);
    let mut positions = (capacities[1] != 0).then(|| Vec::with_capacity(capacities[1]));
    let mut keys = Vec::with_capacity(capacities[2]);
    keys.extend(dictionary.keys().cloned().map(Some));
    let mut key_counts = Vec::with_capacity(capacities[3]);
    key_counts.resize(keys.len(), 0);
    let mut key_ids = Vec::with_capacity(capacities[4]);
    let mut sequences = SequenceColumn::with_capacity(capacities[5], kind);
    let start = chunk.data.position(chunk.head);
    for (offset, ordinal) in (chunk.head..chunk.data.sequences.len()).enumerate() {
        if offset.is_multiple_of(128) {
            cancelled()?;
        }
        let key = chunk.data.keys[chunk.data.key_ids[ordinal] as usize]
            .as_ref()
            .expect("validated key");
        let id = dictionary[key];
        times.push(chunk.data.times[ordinal]);
        key_ids.push(id);
        key_counts[id as usize] += 1;
        sequences.push(
            chunk
                .data
                .sequences
                .get(ordinal)
                .expect("validated sequence")
                .into_owned(),
        );
        let position = chunk.data.position(ordinal);
        if let Some(positions) = &mut positions {
            positions.push(position);
        } else if start.checked_add(
            u32::try_from(offset)
                .map_err(|_| mismatch("ASOF model implicit position exceeds domain"))?,
        ) != Some(position)
        {
            return Err(mismatch("ASOF model implicit payload position differs"));
        }
    }
    let owners: EncodingOwners =
        super::super::left_encoding_owners(&times, &keys, &key_ids, &sequences, cancelled)?;
    Ok(ChunkData {
        times: times.into(),
        positions,
        start: if capacities[1] == 0 { start } else { 0 },
        keys,
        key_counts,
        key_ids,
        sequences,
        owners,
    })
}
