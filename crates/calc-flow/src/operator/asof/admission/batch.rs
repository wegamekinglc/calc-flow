//! General batch admission. IDs identify keys; ranks order their canonical bytes.

use super::{
    Admission, AdmissionRef, AdmissionWorkspace, Arc, Batch, InputKeys, LeftOrder, RecordBatch,
    ReservedIdentities, Result, StreamAsofJoinOperator, StreamOperatorContext, UInt64Array,
    ValidatedInput, can_share_batch, check_input_cancellation, encode_payload, state, take, times,
};
use datafusion::arrow::{
    array::{Array, TimestampMicrosecondArray},
    datatypes::Schema,
};
use std::{cmp::Ordering, collections::HashMap};

#[cfg(test)]
#[derive(Default, Debug, PartialEq, Eq)]
pub(super) struct Work {
    pub records: usize,
    pub rows: usize,
    pub ranks: usize,
    pub sorted_rows: usize,
    pub sequence_fallbacks: usize,
    pub canonical_rows: usize,
}

#[cfg(test)]
thread_local! {
    static WORK: std::cell::RefCell<Work> = std::cell::RefCell::default();
}

#[cfg(test)]
pub(super) fn take_work() -> Work {
    WORK.with(|work| std::mem::take(&mut *work.borrow_mut()))
}

pub(super) struct Record {
    source: usize,
    selection: Option<UInt64Array>,
    times: TimestampMicrosecondArray,
    sequences: Sequences,
    keys: Vec<u32>,
    order: Option<Vec<u32>>,
}

pub(in super::super) struct Columns {
    records: Vec<Record>,
    order: Vec<(usize, u32)>,
}

pub(super) struct Plan {
    columns: Columns,
    keys: InputKeys,
    duplicates: u64,
}

/// Preserve each accepted run's canonical sequence owner, including mixed-late
/// inputs. Integer sequences address one typed column without row encodings.
struct Sequences(Vec<(usize, state::EncodedColumns)>);

impl Sequences {
    fn with_row<R>(&self, row: usize, f: impl FnOnce(&[u8]) -> R) -> R {
        let run = self.0.partition_point(|(start, _)| *start <= row) - 1;
        let (start, columns) = &self.0[run];
        columns.with_row(row - start, f)
    }

    fn row(&self, row: usize) -> state::Encoding {
        let run = self.0.partition_point(|(start, _)| *start <= row) - 1;
        let (start, columns) = &self.0[run];
        columns.row(row - start)
    }

    fn prepare(
        record: &RecordBatch,
        selected: &RecordBatch,
        input: ValidatedInput,
        side: &super::AsofJoinSide,
    ) -> Result<Self> {
        if state::SequenceKind::for_side(&record.schema(), side)
            .width()
            .is_some()
        {
            return Ok(Self(vec![(
                0,
                state::encode_columns(selected, side.sequence_by())?,
            )]));
        }
        if selected.num_rows() == record.num_rows() {
            return Ok(Self(vec![(
                0,
                state::encode_columns(record, side.sequence_by())?,
            )]));
        }
        let mut runs = Vec::new();
        let mut ordinal = 0;
        let mut next = 0;
        let times = times(record, side);
        while next < record.num_rows() {
            if input.is_late(times.value(next)) {
                next += 1;
                continue;
            }
            let start = next;
            next += times.values()[start..]
                .iter()
                .take_while(|time| !input.is_late(**time))
                .count();
            runs.push((
                ordinal,
                state::encode_columns(&record.slice(start, next - start), side.sequence_by())?,
            ));
            ordinal += next - start;
        }
        Ok(Self(runs))
    }
}

pub(super) fn plan(
    operator: &StreamAsofJoinOperator,
    batch: &Batch,
    input: ValidatedInput,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<Plan>> {
    #[cfg(test)]
    if super::identity_tests::columnar::legacy_requested() {
        return Ok(None);
    }
    let mut keys = InputKeys::default();
    let records = prepare_records(operator, batch, input, &mut keys, context)?;
    let ranks = canonical_ranks(&keys);
    let mut columns = Columns {
        records,
        order: Vec::new(),
    };
    columns.prove_order(&ranks, context)?;
    // Reuse the delivered ordered left constructor rather than replacing it.
    let ordered_append = columns.is_ordered_append(operator, input.index, &keys);
    if operator.columnar_left_input(batch, input, columns.order.len(), ordered_append) {
        return Ok(None);
    }
    let duplicates = columns.duplicates(operator, input.index, &keys, &ranks, context)?;
    Ok(Some(Plan {
        columns,
        keys,
        duplicates,
    }))
}

fn prepare_records(
    operator: &StreamAsofJoinOperator,
    batch: &Batch,
    input: ValidatedInput,
    keys: &mut InputKeys,
    context: &StreamOperatorContext<'_>,
) -> Result<Vec<Record>> {
    let mut records = Vec::new();
    for (source, record) in batch.table_payload()?.batches().iter().enumerate() {
        if let Some(record) = Record::prepare(operator, record, source, input, keys, context)? {
            records.push(record);
        }
    }
    Ok(records)
}

fn canonical_ranks(keys: &InputKeys) -> Vec<u32> {
    #[cfg(test)]
    WORK.with_borrow_mut(|work| work.ranks += keys.values.len());
    let mut ids = (0..keys.values.len()).collect::<Vec<_>>();
    ids.sort_unstable_by(|a, b| keys.values[*a].encoding.cmp(&keys.values[*b].encoding));
    let mut ranks = vec![0; ids.len()];
    for (rank, id) in ids.into_iter().enumerate() {
        ranks[id] = u32::try_from(rank).expect("validated ASOF key count");
    }
    ranks
}

impl Record {
    fn prepare(
        operator: &StreamAsofJoinOperator,
        record: &RecordBatch,
        source: usize,
        input: ValidatedInput,
        keys: &mut InputKeys,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<Self>> {
        let side = input.side(&operator.spec);
        let selection = input.watermark.map(|_| {
            UInt64Array::from_iter_values(
                times(record, side)
                    .values()
                    .iter()
                    .enumerate()
                    .filter(|(_, time)| !input.is_late(**time))
                    .map(|(row, _)| row as u64),
            )
        });
        let count = selection.as_ref().map_or(record.num_rows(), Array::len);
        if count == 0 {
            return Ok(None);
        }
        #[cfg(test)]
        WORK.with_borrow_mut(|work| {
            work.records += 1;
            work.rows += count;
        });
        u32::try_from(count).map_err(|_| {
            super::reason(
                &operator.name,
                crate::StreamingFailureReason::AsofCounterOverflow,
                "ASOF compact row range exceeded",
            )
        })?;
        let selection = selection.filter(|rows| rows.len() != record.num_rows());
        let identity = select_identity(record, side, selection.as_ref())?;
        context.check_cancelled()?;
        let encoded = state::encode_key_columns(&identity, side.keys(), |bytes| {
            operator.reserve_workspace(bytes)
        })?;
        let key_ids = key_ids(operator, &encoded, count, keys, context)?;
        Ok(Some(Self {
            source,
            selection,
            times: times(&identity, side).clone(),
            sequences: Sequences::prepare(record, &identity, input, side)?,
            keys: key_ids,
            order: None,
        }))
    }

    fn compare(&self, a: usize, other: &Self, b: usize, ranks: &[u32]) -> Ordering {
        self.times
            .value(a)
            .cmp(&other.times.value(b))
            .then_with(|| ranks[self.keys[a] as usize].cmp(&ranks[other.keys[b] as usize]))
            .then_with(|| {
                self.sequences
                    .with_row(a, |a| other.sequences.with_row(b, |b| a.cmp(b)))
            })
    }

    fn identity(&self, row: usize, keys: &[(state::Encoding, usize)]) -> LeftOrder {
        (
            self.times.value(row),
            keys[self.keys[row] as usize].0.clone(),
            self.sequences.row(row),
        )
    }

    fn chunk(
        &self,
        keys: &InputKeys,
        owner: Arc<state::PayloadBatch>,
        kind: state::SequenceKind,
        cancelled: &dyn Fn() -> Result<()>,
    ) -> Result<state::PreparedLeftChunk> {
        let mut data = state::ChunkData {
            times: Vec::<i64>::new().into(),
            positions: self.order.clone(),
            start: 0,
            keys: Vec::with_capacity(1),
            key_counts: Vec::with_capacity(1),
            key_ids: Vec::with_capacity(self.keys.len()),
            sequences: state::SequenceColumn::with_capacity(self.keys.len(), kind),
            owners: state::EncodingOwners::default(),
        };
        let mut times = Vec::with_capacity(self.keys.len());
        let mut interned = HashMap::<u32, u32>::new();
        for ordinal in 0..self.keys.len() {
            if ordinal.is_multiple_of(128) {
                cancelled()?;
            }
            let row = self
                .order
                .as_ref()
                .map_or(ordinal, |order| order[ordinal] as usize);
            let key = self.keys[row];
            let id = *interned.entry(key).or_insert_with(|| {
                let id = u32::try_from(data.keys.len()).expect("validated ASOF key count");
                if data.keys.len() == data.keys.capacity() {
                    data.keys.reserve_exact(data.keys.len().max(1));
                    data.key_counts.reserve_exact(data.key_counts.len().max(1));
                }
                let encoding = &keys.values[key as usize].encoding;
                data.keys.push(Some(encoding.clone()));
                data.key_counts.push(0);
                data.owners.attach(encoding);
                id
            });
            times.push(self.times.value(row));
            data.key_counts[id as usize] += 1;
            data.key_ids.push(id);
            let sequence = self.sequences.row(row);
            data.owners.attach(&sequence);
            data.sequences.push(sequence);
        }
        data.times = times.into();
        Ok(state::PreparedLeftChunk::from_index(owner, data))
    }
}

fn key_ids(
    operator: &StreamAsofJoinOperator,
    encoded: &state::EncodedColumns,
    count: usize,
    keys: &mut InputKeys,
    context: &StreamOperatorContext<'_>,
) -> Result<Vec<u32>> {
    if let state::EncodedColumns::StringKeys { rows, ids } = encoded {
        let mapping = (0..rows.len())
            .map(|id| {
                check_input_cancellation(id, context)?;
                let (_, id) = keys.intern(rows.row(id), None, operator)?;
                keys.values[id as usize].rows -= 1;
                Ok(id)
            })
            .collect::<Result<Vec<_>>>()?;
        ids.iter()
            .enumerate()
            .map(|(row, id)| {
                check_input_cancellation(row, context)?;
                let id = mapping[*id as usize];
                keys.values[id as usize].rows += 1;
                Ok(id)
            })
            .collect()
    } else {
        (0..count)
            .map(|row| {
                check_input_cancellation(row, context)?;
                encoded.with_row(row, |bytes| {
                    keys.intern(bytes, None, operator).map(|(_, id)| id)
                })
            })
            .collect()
    }
}

fn select_identity(
    record: &RecordBatch,
    side: &super::AsofJoinSide,
    selection: Option<&UInt64Array>,
) -> Result<RecordBatch> {
    let indices = super::identity_column_names(side)
        .map(|name| {
            record
                .schema()
                .index_of(name)
                .expect("validated ASOF identity column")
        })
        .collect::<Vec<_>>();
    // Identity encoding needs fields and values, not payload schema metadata.
    // Sharing fields avoids copying metadata before the payload reservation.
    let schema = Schema::new(
        indices
            .iter()
            .map(|&index| record.schema_ref().fields()[index].clone())
            .collect::<Vec<_>>(),
    );
    let columns = indices
        .iter()
        .map(|&index| record.column(index).clone())
        .collect();
    let identity = RecordBatch::try_new(Arc::new(schema), columns)
        .map_err(|e| super::super::arrow_error(&e))?;
    select(&identity, selection)
}

fn select(record: &RecordBatch, selection: Option<&UInt64Array>) -> Result<RecordBatch> {
    if selection.is_none() && can_share_batch(record)? {
        return Ok(record.clone());
    }
    let full;
    let indices = if let Some(selection) = selection {
        selection
    } else {
        full = UInt64Array::from_iter_values(0..record.num_rows() as u64);
        &full
    };
    let columns = record
        .columns()
        .iter()
        .map(|column| take(column, indices, None).map_err(|e| super::super::arrow_error(&e)))
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new_with_options(
        record.schema(),
        columns,
        &datafusion::arrow::record_batch::RecordBatchOptions::new()
            .with_row_count(Some(indices.len())),
    )
    .map_err(|e| super::super::arrow_error(&e))
}

impl Columns {
    fn can_prepare_inline(&self, keys: &InputKeys) -> bool {
        let count = self.order.len();
        if count <= 256 {
            return true;
        }
        if count > 4_096 {
            return false;
        }
        let physical = self.records.iter().enumerate().flat_map(|(batch, record)| {
            (0..record.keys.len())
                .map(move |row| (batch, u32::try_from(row).expect("validated compact row")))
        });
        if !physical.eq(self.order.iter().copied()) {
            return false;
        }
        self.records.iter().fold(0_usize, |bytes, record| {
            record
                .keys
                .iter()
                .enumerate()
                .fold(bytes, |bytes, (row, key)| {
                    bytes
                        .saturating_add(8)
                        .saturating_add(keys.values[*key as usize].encoding.as_slice().len())
                        .saturating_add(record.sequences.with_row(row, <[u8]>::len))
                })
        }) <= 128 * 1_024
    }

    fn compatibility_rows(
        &self,
        keys: &InputKeys,
        side: usize,
        canonical: bool,
    ) -> Vec<(LeftOrder, AdmissionRef)> {
        if canonical {
            #[cfg(test)]
            WORK.with_borrow_mut(|work| work.sequence_fallbacks += 1);
            let keys = keys
                .values
                .iter()
                .map(|key| (key.encoding.clone(), key.rows))
                .collect::<Vec<_>>();
            let mut left_ids = Vec::with_capacity(keys.len());
            let mut owned = 0;
            for (key, _) in &keys {
                left_ids.push(if state::Encoding::fits_inline(key.as_slice()) {
                    0
                } else {
                    let id = owned;
                    owned += 1;
                    id
                });
            }
            let rows = self
                .records
                .iter()
                .enumerate()
                .flat_map(|(batch, record)| {
                    let keys = &keys;
                    let left_ids = &left_ids;
                    (0..record.keys.len()).map(move |row| {
                        let key_index = record.keys[row] as usize;
                        (
                            record.identity(row, keys),
                            AdmissionRef {
                                batch_index: batch,
                                row: u32::try_from(row).expect("validated ASOF compact row"),
                                key_index: if side == 0 {
                                    left_ids[key_index]
                                } else {
                                    record.keys[row]
                                },
                            },
                        )
                    })
                })
                .collect::<Vec<_>>();
            #[cfg(test)]
            WORK.with_borrow_mut(|work| work.canonical_rows += rows.len());
            super::ordered_admission_rows(rows, side)
        } else {
            Vec::new()
        }
    }

    fn payloads(
        &self,
        operator: &StreamAsofJoinOperator,
        batch: &Batch,
        input: ValidatedInput,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Vec<Arc<state::PayloadBatch>>> {
        let mut batches = Vec::with_capacity(self.records.len());
        let retained = operator
            .payload_projection
            .as_ref()
            .map(|p| &p.columns[input.index]);
        let mut position = operator.admission_base(input.index);
        for record in &self.records {
            context.check_cancelled()?;
            let source = &batch.table_payload()?.batches()[record.source];
            let projected = retained
                .map(|columns| source.project(columns))
                .transpose()
                .map_err(|e| super::super::arrow_error(&e))?;
            let compact = select(
                projected.as_ref().unwrap_or(source),
                record.selection.as_ref(),
            )?;
            batches.push(encode_payload(
                compact,
                input.index,
                position,
                operator.physical_header(input.index),
                &operator.name,
            )?);
            position += record.keys.len() as u64;
        }
        Ok(batches)
    }

    fn chunks(
        &self,
        keys: &InputKeys,
        batches: &[Arc<state::PayloadBatch>],
        kind: state::SequenceKind,
        cancelled: &dyn Fn() -> Result<()>,
    ) -> Result<Vec<state::PreparedLeftChunk>> {
        self.records
            .iter()
            .zip(batches)
            .map(|(record, owner)| {
                cancelled()?;
                record.chunk(keys, owner.clone(), kind, cancelled)
            })
            .collect()
    }

    fn prove_order(&mut self, ranks: &[u32], context: &StreamOperatorContext<'_>) -> Result<()> {
        for (batch, record) in self.records.iter_mut().enumerate() {
            context.check_cancelled()?;
            let count = u32::try_from(record.keys.len()).expect("validated ASOF row range");
            let mut order = (0..count).collect::<Vec<_>>();
            if !order.windows(2).all(|p| {
                record
                    .compare(p[0] as usize, record, p[1] as usize, ranks)
                    .is_lt()
            }) {
                order.sort_unstable_by(|a, b| {
                    record.compare(*a as usize, record, *b as usize, ranks)
                });
                #[cfg(test)]
                WORK.with_borrow_mut(|work| work.sorted_rows += order.len());
                record.order = Some(order.clone());
            }
            self.order.extend(order.into_iter().map(|row| (batch, row)));
        }
        context.check_cancelled()
    }

    fn duplicates(
        &mut self,
        operator: &StreamAsofJoinOperator,
        side: usize,
        keys: &InputKeys,
        ranks: &[u32],
        context: &StreamOperatorContext<'_>,
    ) -> Result<u64> {
        let mut sorted = self.order.clone();
        let compare = |a: &(usize, u32), b: &(usize, u32)| {
            self.records[a.0].compare(a.1 as usize, &self.records[b.0], b.1 as usize, ranks)
        };
        if !sorted.windows(2).all(|p| compare(&p[0], &p[1]).is_le()) {
            sorted.sort_unstable_by(compare);
            #[cfg(test)]
            WORK.with_borrow_mut(|work| work.sorted_rows += sorted.len());
        }
        let skip_resident = sorted.first().is_none_or(|&(batch, row)| {
            let record = &self.records[batch];
            let identity = (
                record.times.value(row as usize),
                keys.values[record.keys[row as usize] as usize]
                    .encoding
                    .clone(),
                record.sequences.row(row as usize),
            );
            super::identities_are_after_state(&operator.state, side, std::iter::once(&identity))
        });
        let mut duplicates = 0;
        let mut previous = None;
        for (ordinal, &(batch, row)) in sorted.iter().enumerate() {
            check_input_cancellation(ordinal, context)?;
            let record = &self.records[batch];
            let identity = (
                record.times.value(row as usize),
                keys.values[record.keys[row as usize] as usize]
                    .encoding
                    .clone(),
                record.sequences.row(row as usize),
            );
            if previous.is_some_and(|last| compare(&last, &(batch, row)).is_eq())
                || (!skip_resident && operator.state.contains_identity(side, &identity))
            {
                duplicates += 1;
            }
            previous = Some((batch, row));
        }
        context.check_cancelled()?;
        // Reuse the global proof as the right installation permutation, so
        // independently reversed records cannot cause repeated vector shifts.
        self.order = sorted;
        Ok(duplicates)
    }

    fn is_ordered_append(
        &self,
        operator: &StreamAsofJoinOperator,
        side: usize,
        keys: &InputKeys,
    ) -> bool {
        if side != 0 || self.records.iter().any(|r| r.order.is_some()) {
            return false;
        }
        let mut previous = operator
            .state
            .left
            .last_key_value()
            .map(|((t, k, s), _)| (*t, k.clone(), s.into_owned()));
        for record in &self.records {
            let first = (
                record.times.value(0),
                keys.values[record.keys[0] as usize].encoding.clone(),
                record.sequences.row(0),
            );
            if previous.as_ref().is_some_and(|last| last >= &first) {
                return false;
            }
            let row = record.keys.len() - 1;
            previous = Some((
                record.times.value(row),
                keys.values[record.keys[row] as usize].encoding.clone(),
                record.sequences.row(row),
            ));
        }
        true
    }

    pub(super) fn identities<'a>(
        &'a self,
        keys: &'a [(state::Encoding, usize)],
    ) -> impl Iterator<Item = (LeftOrder, AdmissionRef)> + 'a {
        self.order.iter().map(move |&(batch, row)| {
            let record = &self.records[batch];
            (
                record.identity(row as usize, keys),
                AdmissionRef {
                    batch_index: batch,
                    row,
                    key_index: record.keys[row as usize],
                },
            )
        })
    }

    pub(super) fn install(
        self,
        state: &mut state::RightState,
        handles: &[u32],
        payloads: &[state::RowRef],
    ) {
        for (batch, row) in self.order {
            let record = &self.records[batch];
            state.insert_reserved(
                handles[record.keys[row as usize] as usize],
                (
                    record.times.value(row as usize),
                    record.sequences.row(row as usize),
                ),
                payloads[batch].with_row(row),
            );
        }
    }
}

impl Plan {
    pub(super) async fn prepare(
        mut self,
        operator: &mut StreamAsofJoinOperator,
        batch: &Batch,
        input: ValidatedInput,
        identity: ReservedIdentities,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Admission> {
        operator.record_duplicates(input.index, self.duplicates)?;
        let accepted = operator.check_admission_rows(input.index, identity.rows as u64)?;
        let payload = operator.input_workspace(batch, input)?;
        let batches = self.columns.payloads(operator, batch, input, context)?;
        let workspace = AdmissionWorkspace {
            _identity: identity.reservation,
            _payload: payload,
            _keys: self.keys.workspace.take(),
        };
        // The canonical owner ledger and right installer still accept row
        // descriptors. Materialize only at that compatibility boundary; all
        // selection, encoding, ranks and duplicate proofs above remain batched.
        let canonical = operator.state.sequence_kinds[input.index].width().is_none();
        let rows = self
            .columns
            .compatibility_rows(&self.keys, input.index, canonical);
        let (left_chunks, right_capacities, right_columns, workspace) = if input.index == 0 {
            let (_, chunks, workspace) = self
                .prepare_left(operator, &batches, workspace, context)
                .await?;
            (chunks, Vec::new(), None, workspace)
        } else {
            (
                None,
                super::admitted_right_capacities(self.keys.values, 1),
                (!canonical).then_some(super::right::Columns::General(self.columns)),
                workspace,
            )
        };
        context.check_cancelled()?;
        Ok(Admission {
            rows,
            batches,
            accepted,
            left_chunks,
            right_capacities,
            right_buckets: None,
            right_columns,
            _workspace: workspace,
        })
    }

    async fn prepare_left(
        self,
        operator: &mut StreamAsofJoinOperator,
        batches: &[Arc<state::PayloadBatch>],
        workspace: AdmissionWorkspace,
        context: &StreamOperatorContext<'_>,
    ) -> Result<super::PreparedInput> {
        let kind = operator.state.sequence_kinds[0];
        if self.columns.can_prepare_inline(&self.keys) {
            let chunks = self
                .columns
                .chunks(&self.keys, batches, kind, &|| context.check_cancelled())?;
            Ok((Vec::new(), Some(chunks), workspace))
        } else {
            let descriptor = operator.reserve_left_work(batches.len())?;
            let work = ChunkWork {
                columns: self.columns,
                keys: self.keys,
                batches: batches.to_vec(),
                kind,
                workspace,
                _descriptor: descriptor,
            };
            operator.run_cpu_work(work, context).await
        }
    }
}

struct ChunkWork {
    columns: Columns,
    keys: InputKeys,
    batches: Vec<Arc<state::PayloadBatch>>,
    kind: state::SequenceKind,
    workspace: AdmissionWorkspace,
    _descriptor: super::MemoryReservation,
}

impl super::OwnedCpuWork for ChunkWork {
    type Output = super::PreparedInput;

    fn run(self, stop: &super::GatherStop) -> Result<Self::Output> {
        stop.check()?;
        let chunks = self
            .columns
            .chunks(&self.keys, &self.batches, self.kind, &|| stop.check())?;
        stop.check()?;
        Ok((Vec::new(), Some(chunks), self.workspace))
    }
}
