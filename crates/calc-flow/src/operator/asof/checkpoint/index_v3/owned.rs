//! Detached capture retains index columns and shared canonical buffers only.
//! Arrow payloads remain in their separately charged immutable IPC segments.

use super::{
    ChunkData, Encoding, OwnerWriter, RightBucket, SequenceKind, State, StateSegment, checked,
    mismatch, put, write_left, write_right_columns,
};
use crate::{Result, StreamOperatorContext};
use std::sync::Arc;

const CHECK_EVERY: usize = 256;
const YIELD_EVERY: usize = 8_192;
type LeftChunk = (super::BatchKey, Arc<ChunkData>, usize);

struct Bucket {
    key: Encoding,
    rows: Arc<RightBucket>,
}

pub(super) struct Index {
    capacities: [usize; 5],
    kinds: [SequenceKind; 2],
    left_rows: usize,
    left: Vec<LeftChunk>,
    right: Vec<Bucket>,
    batches: Vec<(u32, u64)>,
}

pub(super) fn workspace_bytes(state: &State, length: u64, owned: bool, name: &str) -> Result<u64> {
    let (buffers, metadata) = state.encoding_owner_allocation();
    // OwnerWriter's ordered address map and handle vector, plus one chunk's
    // sorting/remap vectors, are live while the output buffer is written.
    let sorting = state
        .left
        .checkpoint_chunks(&state.batches)
        .map(|(_, data, _)| data.keys.len() as u64 * 32 + 128)
        .max()
        .unwrap_or(0);
    // Owner metadata already bounds the address map and handle vector. The
    // 256-byte fixed allowance matches the committed index's owner/control
    // fee; captured columns and all sorting descriptors are funded below.
    let mut bytes = checked(name, length, checked(name, metadata + 256, sorting)?)?;
    if owned {
        bytes = checked(name, bytes, buffers)?;
        bytes = checked(name, bytes, state.left.capacity_bytes(name)?)?;
        bytes = checked(
            name,
            bytes,
            (state.left.checkpoint_chunks(&state.batches).len() * size_of::<LeftChunk>()) as u64,
        )?;
        bytes = checked(
            name,
            bytes,
            (state.right.len() * size_of::<Bucket>()) as u64,
        )?;
        bytes = checked(
            name,
            bytes,
            (state.batches.len() * size_of::<(u32, u64)>()) as u64,
        )?;
        for bucket in state.right.values() {
            bytes = checked(
                name,
                bytes,
                bucket.metadata_bytes()
                    + (size_of::<RightBucket>() + 2 * size_of::<usize>()) as u64,
            )?;
        }
    } else {
        bytes = checked(name, bytes, state.right.len() as u64 * 4)?;
    }
    Ok(bytes)
}

async fn cooperate(count: usize, context: &StreamOperatorContext<'_>) -> Result<()> {
    if count.is_multiple_of(CHECK_EVERY) {
        context.check_cancelled()?;
    }
    if count.is_multiple_of(YIELD_EVERY) {
        tokio::task::yield_now().await;
    }
    Ok(())
}

impl Index {
    pub async fn capture(state: &State, context: &StreamOperatorContext<'_>) -> Result<Self> {
        context.check_cancelled()?;
        if !state.left.legacy.is_empty() {
            return Err(mismatch("ASOF legacy rows must migrate before v3 capture"));
        }
        let pool = state.batches.backing_buckets();
        let right_capacity = state.right.checkpoint_capacities();
        let mut left = Vec::with_capacity(state.left.checkpoint_chunks(&state.batches).len());
        for chunk in state.left.checkpoint_owned_chunks(&state.batches) {
            left.push(chunk);
            cooperate(left.len(), context).await?;
        }
        let mut right = Vec::with_capacity(state.right.len());
        let mut count = left.len();
        for (key, rows) in state.right.owned_buckets() {
            right.push(Bucket { key, rows });
            count += 1;
            cooperate(count, context).await?;
        }
        let mut batches = Vec::with_capacity(state.batches.len());
        for batch in state.batches.checkpoint_right_handles() {
            batches.push(batch);
            count += 1;
            cooperate(count, context).await?;
        }
        context.check_cancelled()?;
        Ok(Self {
            capacities: [
                pool.0,
                pool.1,
                right_capacity[0],
                right_capacity[1],
                state.left.chunk_capacity(),
            ],
            kinds: state.sequence_kinds,
            left_rows: state.left.len(),
            left,
            right,
            batches,
        })
    }

    fn owners(&self) -> OwnerWriter {
        let mut owners = OwnerWriter::default();
        for (_, data, head) in &self.left {
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
                for sequence in data.sequences.iter().skip(*head) {
                    owners.register(sequence.as_ref());
                }
            }
        }
        for bucket in &self.right {
            owners.register(&bucket.key);
            if self.kinds[1] == SequenceKind::Canonical {
                for ((_, sequence), _) in bucket.rows.as_ref() {
                    owners.register(sequence.as_ref());
                }
            }
        }
        owners
    }

    pub fn encode(mut self, length: u64, limit: usize) -> Result<StateSegment> {
        let capacity = usize::try_from(length)
            .map_err(|_| mismatch("ASOF v3 index exceeds address domain"))?;
        if capacity > limit {
            return Err(mismatch("ASOF v3 index exceeds limits"));
        }
        self.right
            .sort_unstable_by(|left, right| left.key.cmp(&right.key));
        self.batches.sort_unstable_by_key(|(handle, _)| *handle);
        let owners = self.owners();
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(super::MAGIC);
        for count in [self.left.len(), self.right.len(), self.left_rows] {
            put(&mut bytes, count as u64);
        }
        for capacity in self.capacities {
            put(&mut bytes, capacity as u64);
        }
        owners.write(&mut bytes);
        for (batch, data, head) in &self.left {
            write_left(&mut bytes, *batch, data, *head, self.kinds[0], &owners);
        }
        for bucket in &self.right {
            write_right_columns(
                &mut bytes,
                &bucket.key,
                &bucket.rows,
                self.kinds[1],
                &owners,
                |row| {
                    let index = self
                        .batches
                        .binary_search_by_key(&row.batch_handle(), |(handle, _)| *handle)
                        .expect("captured batch handle");
                    self.batches[index].1
                },
            );
        }
        if bytes.len() != capacity {
            return Err(mismatch("ASOF v3 encoded length differs"));
        }
        Ok(StateSegment::new(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CancellationToken, JsonMap, StreamAsofJoinStatus, StreamJobContext};

    use crate::operator::asof::{
        AsofJoinSide,
        state::{BatchKey, PayloadBatch, PreparedLeftChunk, RowPayload, encode_columns},
    };
    use datafusion::arrow::{
        array::{Int64Array, StringArray},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use std::{collections::BTreeMap, sync::OnceLock};

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

    #[tokio::test]
    async fn owned_capture_preserves_columns_after_mutation_and_releases_arrow_payloads() {
        let (mut state, batches) = fixture();
        let payloads = batches.values().map(Arc::downgrade).collect::<Vec<_>>();
        let length = super::super::encoded_length(&state, "asof").unwrap();
        let expected = super::super::encode_sync(&state, length, 1 << 20).unwrap();
        let budget = workspace_bytes(&state, length, true, "asof").unwrap();
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "asof", None);
        let frozen = Index::capture(&state, &context).await.unwrap();
        // Retained checkpoint work forces the same funded copy path that a
        // cancelled blocking task uses when the live operator resumes.
        state.commit_left_prefix(1);
        let mut status = StreamAsofJoinStatus::default();
        status.left.ended = true;
        status.right.ended = true;
        assert_eq!(state.evict(&status, 0), 1);
        drop(state);
        drop(batches);
        assert!(payloads.iter().all(|payload| payload.upgrade().is_none()));
        let mut actual = None;
        let allocations = allocation_counter::measure(|| {
            actual = Some(frozen.encode(length, 1 << 20).unwrap());
        });
        assert!(
            allocations.bytes_max <= budget,
            "{allocations:?}, workspace={budget}"
        );
        assert_eq!(actual.unwrap(), expected);
    }
}
