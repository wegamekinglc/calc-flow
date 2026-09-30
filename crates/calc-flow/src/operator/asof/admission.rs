use super::{
    AsofJoinSide, AsofLatePolicy, StreamAsofJoinOperator, StreamAsofJoinSideStatus,
    StreamAsofJoinSpec, StreamAsofJoinStatus, reason,
    state::{self, LeftOrder, RowPayload},
};
use crate::{Batch, Result, StreamOperatorContext, StreamingFailureReason};
use ahash::RandomState;
use datafusion::arrow::{
    array::{Array, TimestampMicrosecondArray, UInt64Array},
    compute::take,
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::MemoryReservation;
use hashbrown::HashTable;
use std::{collections::HashSet, hash::BuildHasher, sync::Arc};

#[derive(Clone, Copy)]
pub(super) struct ValidatedInput {
    pub index: usize,
    pub watermark: Option<i64>,
}

type PreparedInput = (
    Vec<(LeftOrder, RowPayload)>,
    Option<Vec<state::PreparedLeftChunk>>,
    AdmissionWorkspace,
);

type InputRow<'a> = (LeftOrder, &'a RecordBatch, usize);

pub(super) struct Admission {
    pub rows: Vec<(LeftOrder, RowPayload)>,
    pub batches: Vec<Arc<state::PayloadBatch>>,
    pub accepted: u64,
    pub left_chunks: Option<Vec<state::PreparedLeftChunk>>,
    pub right_capacities: Vec<(state::Encoding, usize)>,
    _workspace: AdmissionWorkspace,
}

struct AdmissionWorkspace {
    _identity: MemoryReservation,
    _payload: MemoryReservation,
    _keys: Option<MemoryReservation>,
}

struct AdmittedKey {
    encoding: state::Encoding,
    hash: u64,
    rows: usize,
}

#[derive(Default)]
struct InputKeys {
    index: HashTable<u32>,
    values: Vec<AdmittedKey>,
    workspace: Option<MemoryReservation>,
}

impl InputKeys {
    fn intern(
        &mut self,
        bytes: &[u8],
        hash: Option<u64>,
        operator: &StreamAsofJoinOperator,
    ) -> Result<state::Encoding> {
        let resident = &operator.state.right;
        let name = &operator.name;
        let hash = hash.unwrap_or_else(|| resident.hasher().hash_one(bytes));
        if let Some(id) = self.index.find(hash, |id| {
            self.values[*id as usize].encoding.as_slice() == bytes
        }) {
            let key = &mut self.values[*id as usize];
            key.rows += 1;
            return Ok(key.encoding.clone());
        }
        state::validate_key_count(self.values.len() as u64 + 1, name)?;
        let id = u32::try_from(self.values.len()).expect("validated ASOF key count");
        let encoding = if let Some(existing) = resident.encoding_hashed(hash, bytes) {
            existing.clone()
        } else {
            self.copy_owned_key(bytes, operator)?
        };
        self.values.push(AdmittedKey {
            encoding: encoding.clone(),
            hash,
            rows: 1,
        });
        self.index
            .insert_unique(hash, id, |id| self.values[*id as usize].hash);
        Ok(encoding)
    }

    fn copy_owned_key(
        &mut self,
        bytes: &[u8],
        operator: &StreamAsofJoinOperator,
    ) -> Result<state::Encoding> {
        if !state::Encoding::fits_inline(bytes) {
            if self.workspace.is_none() {
                self.workspace = Some(operator.reserve_workspace(0)?);
            }
            self.workspace
                .as_ref()
                .expect("ASOF key copy workspace")
                .try_grow(bytes.len())
                .map_err(|_| {
                    reason(
                        &operator.name,
                        StreamingFailureReason::AsofWorkspaceLimitExceeded,
                        "ASOF owned key copies exceed max_state_bytes workspace",
                    )
                })?;
        }
        Ok(state::Encoding::from_slice(bytes))
    }
}

/// Owns the batch converters and their optional, separately reserved hash
/// vector. Mixed-late input encodes only each accepted immutable slice.
struct InputEncodings<'a> {
    operator: &'a StreamAsofJoinOperator,
    batch: &'a RecordBatch,
    side: &'a AsofJoinSide,
    columns: Option<(state::EncodedColumns, state::EncodedColumns)>,
    hashes: Option<Vec<u64>>,
    _hash_workspace: Option<MemoryReservation>,
    input: ValidatedInput,
    range: std::ops::Range<usize>,
}

impl<'a> InputEncodings<'a> {
    fn new(
        operator: &'a StreamAsofJoinOperator,
        batch: &'a RecordBatch,
        input: ValidatedInput,
    ) -> Result<Option<Self>> {
        let side = input.side(&operator.spec);
        let accepted = input.watermark.map_or(batch.num_rows(), |_| {
            times(batch, side)
                .values()
                .iter()
                .filter(|time| !input.is_late(**time))
                .count()
        });
        if accepted == 0 {
            return Ok(None);
        }
        let columns = (accepted == batch.num_rows())
            .then(|| Self::encode_identity_columns(operator, batch, side))
            .transpose()?;
        let (hashes, hash_workspace) =
            Self::batch_key_hashes(operator, batch, input, columns.as_ref())?;
        let range = if columns.is_some() {
            0..batch.num_rows()
        } else {
            0..0
        };
        Ok(Some(Self {
            operator,
            batch,
            side,
            columns,
            hashes,
            _hash_workspace: hash_workspace,
            input,
            range,
        }))
    }

    fn batch_key_hashes(
        operator: &StreamAsofJoinOperator,
        batch: &RecordBatch,
        input: ValidatedInput,
        columns: Option<&(state::EncodedColumns, state::EncodedColumns)>,
    ) -> Result<(Option<Vec<u64>>, Option<MemoryReservation>)> {
        let workspace = columns
            .filter(|(keys, _)| {
                input.index == 1
                    && matches!(
                        keys,
                        state::EncodedColumns::Binary(_) | state::EncodedColumns::StringKeys { .. }
                    )
            })
            .and_then(|_| {
                operator
                    .reserve_workspace((batch.num_rows() as u64).saturating_mul(8))
                    .ok()
            });
        let hashes = match (columns, &workspace) {
            (Some((keys, _)), Some(_)) => {
                Some(keys.hashes(operator.state.right.hasher(), batch.num_rows())?)
            }
            _ => None,
        };
        Ok((hashes, workspace))
    }

    fn with_row<R>(
        &mut self,
        row: usize,
        use_identity: impl FnOnce(&[u8], Option<u64>, state::Encoding) -> Result<R>,
    ) -> Result<R> {
        if !self.range.contains(&row) {
            self.columns = None;
            let event_times = times(self.batch, self.side);
            let count = event_times.values()[row..]
                .iter()
                .take_while(|time| !self.input.is_late(**time))
                .count();
            let slice = self.batch.slice(row, count);
            self.columns = Some(Self::encode_identity_columns(
                self.operator,
                &slice,
                self.side,
            )?);
            self.range = row..row + count;
        }
        let (keys, sequences) = self.columns.as_ref().expect("accepted ASOF encodings");
        let index = row - self.range.start;
        keys.with_row(index, |bytes| {
            use_identity(
                bytes,
                self.hashes.as_ref().map(|hashes| hashes[row]),
                sequences.row(index),
            )
        })
    }

    fn encode_identity_columns(
        operator: &StreamAsofJoinOperator,
        batch: &RecordBatch,
        side: &AsofJoinSide,
    ) -> Result<(state::EncodedColumns, state::EncodedColumns)> {
        let keys = if state::SequenceKind::for_side(&batch.schema(), side)
            .width()
            .is_some()
        {
            state::encode_key_columns(batch, side.keys(), |bytes| {
                operator.reserve_workspace(bytes)
            })?
        } else {
            state::encode_columns(batch, side.keys())?
        };
        Ok((keys, state::encode_columns(batch, side.sequence_by())?))
    }
}

struct InputIdentities<'a> {
    rows: Vec<InputRow<'a>>,
    duplicates: u64,
    right_capacities: Vec<(state::Encoding, usize)>,
    key_workspace: Option<MemoryReservation>,
}

impl StreamAsofJoinOperator {
    pub(super) fn validate_admission(
        &mut self,
        ingress: &str,
        batch: &Batch,
    ) -> Result<ValidatedInput> {
        let index = ingress_index(ingress, &self.name)?;
        self.inputs[index].validate(batch, ingress).map_err(|_| {
            reason(
                &self.name,
                StreamingFailureReason::AsofInvalidInput,
                "input does not match the declared exact table schema",
            )
        })?;
        let status = side_status(&mut self.status, index);
        if self.terminal || status.ended {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofProtocolError,
                "input received data after end-of-input",
            ));
        }
        let input = ValidatedInput {
            index,
            watermark: status.watermark_micros.map(crate::EventTime::as_micros),
        };
        let batches = batch.table_payload()?.batches();
        validate_nulls(batches, input.side(&self.spec), &self.name, ingress)?;
        self.validate_late_rows(batches, input)?;
        Ok(input)
    }

    fn validate_late_rows(&mut self, batches: &[RecordBatch], input: ValidatedInput) -> Result<()> {
        let late = batches
            .iter()
            .map(|batch| {
                times(batch, input.side(&self.spec))
                    .values()
                    .iter()
                    .filter(|time| input.is_late(**time))
                    .count() as u64
            })
            .sum();
        let status = side_status(&mut self.status, input.index);
        status.late_rows = super::checked(&self.name, status.late_rows, late)?;
        if late > 0 && self.spec.late_policy() == AsofLatePolicy::Error {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofLateRow,
                "input contains event time below its accepted watermark",
            ));
        }
        Ok(())
    }

    #[tracing::instrument(
        name = "asof.admission",
        level = "debug",
        skip_all,
        fields(operator = %self.name, side = input.index, rows = batch.num_rows())
    )]
    pub(super) async fn prepare_admission(
        &mut self,
        input: ValidatedInput,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Admission> {
        let identity_workspace = self
            .reserve_admission_identities(batch, input, context)
            .await?;
        let InputIdentities {
            rows,
            duplicates,
            right_capacities,
            key_workspace,
        } = self
            .validated_input_identities(batch, input, context)
            .await?;
        self.record_duplicates(input.index, duplicates)?;
        let accepted = self.check_admission_rows(input.index, rows.len() as u64)?;
        let payload_workspace = self.input_workspace(batch, input)?;
        let base = if input.index == 0 {
            self.status.left.accepted_rows
        } else {
            self.status.right.accepted_rows
        };
        let (rows, batches) = encode_rows(
            rows,
            input.index,
            base,
            self.payload_header_bytes[input.index],
            &self.name,
        )?;
        let rows = ordered_admission_rows(rows, input.index);
        let workspace = AdmissionWorkspace {
            _identity: identity_workspace,
            _payload: payload_workspace,
            _keys: key_workspace,
        };
        let (rows, left_chunks, workspace) = self
            .prepare_input_chunks(rows, workspace, input.index, context)
            .await?;
        Ok(Admission {
            rows,
            batches,
            left_chunks,
            accepted,
            right_capacities,
            _workspace: workspace,
        })
    }

    async fn validated_input_identities<'a>(
        &mut self,
        batch: &'a Batch,
        input: ValidatedInput,
        context: &StreamOperatorContext<'_>,
    ) -> Result<InputIdentities<'a>> {
        match self.admission_identities(batch.table_payload()?.batches(), input, context) {
            Ok(identities) => Ok(identities),
            Err(error)
                if matches!(
                    &error,
                    crate::CalcFlowError::OperatorReason {
                        reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
                        ..
                    }
                ) =>
            {
                self.validate_duplicates_without_workspace(batch, input, context)
                    .await?;
                Err(error)
            }
            Err(error) => Err(error),
        }
    }

    async fn prepare_input_chunks(
        &self,
        rows: Vec<(LeftOrder, RowPayload)>,
        workspace: AdmissionWorkspace,
        side: usize,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedInput> {
        Ok(if side == 0 {
            let side = self.spec.left().clone();
            let name = self.name.clone();
            context.check_cancelled()?;
            let work = chunk_worker(workspace, move || {
                let chunks = state::PreparedLeftChunk::prepare(&rows, &side, &name)?;
                Ok((rows, chunks))
            });
            let ((rows, chunks), workspace) = tokio::select! {
                result = work => result?,
                () = context.job().cancellation().cancelled() => {
                    context.check_cancelled()?;
                    unreachable!("cancelled ASOF admission")
                }
            };
            context.check_cancelled()?;
            (rows, Some(chunks), workspace)
        } else {
            (rows, None, workspace)
        })
    }

    async fn reserve_admission_identities(
        &mut self,
        batch: &Batch,
        input: ValidatedInput,
        context: &StreamOperatorContext<'_>,
    ) -> Result<MemoryReservation> {
        match self.identity_workspace(batch, input) {
            Ok(reservation) => Ok(reservation),
            Err(error) => {
                self.validate_duplicates_without_workspace(batch, input, context)
                    .await?;
                Err(error)
            }
        }
    }

    pub(super) fn record_duplicates(&mut self, index: usize, duplicates: u64) -> Result<()> {
        let status = side_status(&mut self.status, index);
        status.duplicate_rows = super::checked(&self.name, status.duplicate_rows, duplicates)?;
        if duplicates > 0 {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofDuplicateIdentity,
                "input contains a duplicate key/event-time/sequence identity",
            ));
        }
        Ok(())
    }

    fn check_admission_rows(&mut self, index: usize, rows: u64) -> Result<u64> {
        let accepted = {
            let status = side_status(&mut self.status, index);
            super::checked(&self.name, status.accepted_rows, rows)?
        };
        // The committed identity count comes from the maintained gauges; the
        // clone-prepare-install transaction revalidates the full charge
        // before install, so admission only needs the fail-closed row bound.
        if super::checked(&self.name, self.status.state_rows, rows)?
            > self.spec.limits().max_state_rows()
        {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofStateLimitExceeded,
                "stream_asof_join.limits.max_state_rows exceeded",
            ));
        }
        Ok(accepted)
    }

    fn input_identity(
        &self,
        encodings: &mut InputEncodings<'_>,
        keys: &mut InputKeys,
        input: ValidatedInput,
        time: i64,
        row: usize,
    ) -> Result<LeftOrder> {
        encodings.with_row(row, |bytes, hash, sequence| {
            let key = if input.index == 0 && state::Encoding::fits_inline(bytes) {
                state::Encoding::from_slice(bytes)
            } else {
                keys.intern(bytes, hash, self)?
            };
            Ok((time, key, sequence))
        })
    }

    fn admission_identities<'a>(
        &self,
        batches: &'a [RecordBatch],
        input: ValidatedInput,
        context: &StreamOperatorContext<'_>,
    ) -> Result<InputIdentities<'a>> {
        let side = input.side(&self.spec);
        let capacity = if input.watermark.is_none() {
            batches.iter().map(RecordBatch::num_rows).sum()
        } else {
            0
        };
        let mut keys = InputKeys::default();
        let mut rows = Vec::with_capacity(capacity);
        for batch in batches {
            let Some(mut encodings) = InputEncodings::new(self, batch, input)? else {
                continue;
            };
            let event_times = times(batch, side);
            for row in 0..batch.num_rows() {
                check_input_cancellation(row, context)?;
                let time = event_times.value(row);
                if input.is_late(time) {
                    continue;
                }
                let identity = self.input_identity(&mut encodings, &mut keys, input, time, row)?;
                rows.push((identity, batch, row));
            }
        }
        let duplicates = count_duplicate_identities(
            &self.state,
            input.index,
            rows.iter().map(|(identity, _, _)| identity),
            context,
        )?;
        let right_capacities = if input.index == 1 {
            keys.values
                .into_iter()
                .map(|key| (key.encoding, key.rows))
                .collect()
        } else {
            Vec::new()
        };
        Ok(InputIdentities {
            rows,
            duplicates,
            right_capacities,
            key_workspace: keys.workspace,
        })
    }
}

fn check_input_cancellation(row: usize, context: &StreamOperatorContext<'_>) -> Result<()> {
    if row.is_multiple_of(1_024) {
        context.check_cancelled()?;
    }
    Ok(())
}

fn count_duplicate_identities<'a>(
    state: &state::State,
    side: usize,
    identities: impl ExactSizeIterator<Item = &'a LeftOrder> + Clone,
    context: &StreamOperatorContext<'_>,
) -> Result<u64> {
    let (skip_resident, mut seen) = prepare_duplicate_probes(state, side, &identities, context)?;
    let mut previous = None;
    let mut duplicates = 0;
    for (position, identity) in identities.enumerate() {
        if position % 1_024 == 0 {
            context.check_cancelled()?;
        }
        if repeated_identity(state, side, identity, &mut seen, previous, skip_resident) {
            duplicates += 1;
        }
        previous = Some(identity);
    }
    context.check_cancelled()?;
    Ok(duplicates)
}

fn prepare_duplicate_probes<'a>(
    state: &state::State,
    side: usize,
    identities: &(impl ExactSizeIterator<Item = &'a LeftOrder> + Clone),
    context: &StreamOperatorContext<'_>,
) -> Result<(bool, Option<HashSet<LeftOrder, RandomState>>)> {
    let sorted = identities_are_sorted((*identities).clone(), context)?;
    let skip_resident = sorted && identities_are_after_state(state, side, (*identities).clone());
    let seen =
        (!sorted).then(|| HashSet::with_capacity_and_hasher(identities.len(), RandomState::new()));
    Ok((skip_resident, seen))
}

fn repeated_identity(
    state: &state::State,
    side: usize,
    identity: &LeftOrder,
    seen: &mut Option<HashSet<LeftOrder, RandomState>>,
    previous: Option<&LeftOrder>,
    skip_resident: bool,
) -> bool {
    let repeated = if let Some(seen) = seen {
        !seen.insert(identity.clone())
    } else {
        previous == Some(identity)
    };
    repeated || (!skip_resident && state.contains_identity(side, identity))
}

fn identities_are_sorted<'a>(
    identities: impl Iterator<Item = &'a LeftOrder>,
    context: &StreamOperatorContext<'_>,
) -> Result<bool> {
    let mut previous = None;
    for (position, identity) in identities.enumerate() {
        if position % 1_024 == 0 {
            context.check_cancelled()?;
        }
        if previous.is_some_and(|last| last > identity) {
            return Ok(false);
        }
        previous = Some(identity);
    }
    Ok(true)
}

fn identities_are_after_state<'a>(
    state: &state::State,
    side: usize,
    mut identities: impl Iterator<Item = &'a LeftOrder>,
) -> bool {
    let Some(first) = identities.next() else {
        return true;
    };
    if side == 0 {
        state.left.last_key_value().is_none_or(|(last, _)| {
            last < (&first.0, &first.1, std::borrow::Cow::Borrowed(&first.2))
        })
    } else {
        state
            .right
            .values()
            .filter_map(|bucket| bucket.last_key_value().map(|((time, _), _)| *time))
            .max()
            .is_none_or(|latest| latest < first.0)
    }
}

impl Admission {
    pub fn install(
        &mut self,
        ingress: &str,
        state: &mut state::State,
        status: &mut StreamAsofJoinStatus,
    ) {
        if ingress == "left" {
            state.left.install(
                self.left_chunks.take().expect("prepared left chunks"),
                &mut state.batches,
            );
            self.rows.clear();
            status.left.accepted_rows = self.accepted;
        } else {
            for (key, count) in self.right_capacities.drain(..) {
                state
                    .right
                    .bucket_mut_or_kind(key, state.sequence_kinds[1])
                    .reserve_payloads(count);
            }
            // Each compact payload owns exactly its admitted rows. Attach its
            // reference count once, then resolve row handles without two pool
            // lookups for every row. Admission batches are ordered by key.
            let payload_refs = self
                .batches
                .iter()
                .map(|batch| state.batches.attach_batch(batch, batch.record.num_rows()))
                .collect::<Vec<_>>();
            for (identity, payload) in self.rows.drain(..) {
                let batch = self
                    .batches
                    .partition_point(|batch| batch.key < payload.batch.key);
                let payload = payload_refs[batch].with_row(
                    u32::try_from(payload.row).expect("preflighted ASOF payload row index"),
                );
                state.right_payload_min = Some(
                    state
                        .right_payload_min
                        .map_or(identity.0, |previous| previous.min(identity.0)),
                );
                state
                    .right
                    .bucket_mut_or_default(identity.1)
                    .insert_admitted((identity.0, identity.2), payload);
            }
            status.right.accepted_rows = self.accepted;
        }
    }
}

/// Detached blocking work owns its reservations until every retained input
/// and temporary sort buffer has been released, even if admission is dropped.
async fn chunk_worker<R: Send + 'static>(
    workspace: AdmissionWorkspace,
    work: impl FnOnce() -> Result<R> + Send + 'static,
) -> Result<(R, AdmissionWorkspace)> {
    tokio::task::spawn_blocking(move || Ok((work()?, workspace)))
        .await
        .map_err(|error| crate::CalcFlowError::Internal {
            message: format!("ASOF chunk preparation task failed: {error}"),
        })?
}

/// Identity columns in declaration order: keys, sequence columns, event time.
pub(super) fn identity_column_names(side: &AsofJoinSide) -> impl Iterator<Item = &str> {
    side.keys()
        .iter()
        .chain(side.sequence_by())
        .map(String::as_str)
        .chain(std::iter::once(side.event_time()))
}

/// Reports whether one identity column of a validated batch contains nulls.
pub(super) fn identity_column_nulls(batch: &RecordBatch, column: &str) -> bool {
    batch
        .column(batch.schema().index_of(column).expect("validated"))
        .null_count()
        != 0
}

fn validate_nulls(
    batches: &[RecordBatch],
    side: &AsofJoinSide,
    node: &str,
    ingress: &str,
) -> Result<()> {
    for column in identity_column_names(side) {
        for batch in batches {
            if identity_column_nulls(batch, column) {
                return Err(reason(
                    node,
                    StreamingFailureReason::AsofInvalidInput,
                    &format!("{ingress} identity column {column:?} contains null values"),
                ));
            }
        }
    }
    Ok(())
}
pub(super) fn times<'a>(
    batch: &'a RecordBatch,
    side: &AsofJoinSide,
) -> &'a TimestampMicrosecondArray {
    batch
        .column(
            batch
                .schema()
                .index_of(side.event_time())
                .expect("validated"),
        )
        .as_any()
        .downcast_ref()
        .expect("validated timestamp")
}

impl ValidatedInput {
    pub(super) fn side(self, spec: &StreamAsofJoinSpec) -> &AsofJoinSide {
        if self.index == 0 {
            spec.left()
        } else {
            spec.right()
        }
    }

    pub(super) fn is_late(self, time: i64) -> bool {
        self.watermark.is_some_and(|watermark| time < watermark)
    }
}

fn ingress_index(ingress: &str, node: &str) -> Result<usize> {
    match ingress {
        "left" => Ok(0),
        "right" => Ok(1),
        _ => Err(reason(
            node,
            StreamingFailureReason::AsofInvalidInput,
            "unknown ingress",
        )),
    }
}

fn side_status(status: &mut StreamAsofJoinStatus, index: usize) -> &mut StreamAsofJoinSideStatus {
    if index == 0 {
        &mut status.left
    } else {
        &mut status.right
    }
}

type EncodedInput = (Vec<(LeftOrder, RowPayload)>, Vec<Arc<state::PayloadBatch>>);

fn ordered_admission_rows(
    mut rows: Vec<(LeftOrder, RowPayload)>,
    side: usize,
) -> Vec<(LeftOrder, RowPayload)> {
    if side == 1 && !rows.windows(2).all(|pair| pair[0].0 <= pair[1].0) {
        // Keep each admitted run ordered so a watermark-local reversal
        // does not repeatedly shift a whole per-key right vector.
        rows.sort_unstable_by(|left, right| {
            left.0
                .1
                .cmp(&right.0.1)
                .then_with(|| left.0.0.cmp(&right.0.0))
                .then_with(|| left.0.2.cmp(&right.0.2))
        });
    }
    rows
}

fn encode_rows(
    rows: Vec<InputRow<'_>>,
    side: usize,
    base: u64,
    header: u64,
    name: &str,
) -> Result<EncodedInput> {
    let mut payloads = Vec::new();
    let mut position = 0;
    while position < rows.len() {
        let batch = rows[position].1;
        let end = position
            + rows[position..]
                .iter()
                .take_while(|(_, candidate, _)| std::ptr::eq(*candidate, batch))
                .count();
        let compact = compact_accepted_rows(&rows[position..end], batch)?;
        let payload = encode_payload(compact, side, base + position as u64, header, name)?;
        payloads.push((end - position, payload));
        position = end;
    }
    let batches = payloads
        .iter()
        .map(|(_, payload)| payload.clone())
        .collect();
    let mut input = rows.into_iter();
    let mut result = Vec::with_capacity(input.len());
    for (count, payload) in payloads {
        for ordinal in 0..count {
            let (identity, _, _) = input.next().expect("partitioned ASOF input row");
            result.push((
                identity,
                RowPayload {
                    batch: Arc::clone(&payload),
                    row: ordinal,
                },
            ));
        }
    }
    debug_assert!(input.next().is_none());
    Ok((result, batches))
}

fn compact_accepted_rows(rows: &[InputRow<'_>], batch: &RecordBatch) -> Result<RecordBatch> {
    if rows.len() == batch.num_rows() && can_share_batch(batch)? {
        return Ok(batch.clone());
    }
    let indices = UInt64Array::from(
        rows.iter()
            .map(|(_, _, row)| *row as u64)
            .collect::<Vec<_>>(),
    );
    let columns = batch
        .columns()
        .iter()
        .map(|column| take(column, &indices, None).map_err(|error| super::arrow_error(&error)))
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(batch.schema(), columns).map_err(|error| super::arrow_error(&error))
}

fn encode_payload(
    record: RecordBatch,
    side: usize,
    id: u64,
    header: u64,
    name: &str,
) -> Result<Arc<state::PayloadBatch>> {
    let (encoded_charge_bytes, body_bytes) =
        super::workspace::payload_bound_with_header(&record, header, name)?;
    Ok(Arc::new(state::PayloadBatch {
        key: (u8::try_from(side).expect("validated two-sided ingress"), id),
        record: Arc::new(record),
        encoded: std::sync::OnceLock::new(),
        encoded_charge_bytes,
        body_bytes,
    }))
}

/// Reuse a complete Arrow batch only when its backing buffers contain no
/// uncharged bytes outside the accepted slice. Even a small tail can otherwise
/// bypass a tight state-memory limit across many admitted batches.
fn can_share_batch(batch: &RecordBatch) -> Result<bool> {
    let excess = batch.columns().iter().try_fold(0_u64, |total, column| {
        let data = column.to_data();
        let logical = data
            .get_slice_memory_size()
            .map_err(|error| super::arrow_error(&error))? as u64;
        Ok::<_, crate::CalcFlowError>(
            total.saturating_add((data.get_buffer_memory_size() as u64).saturating_sub(logical)),
        )
    })?;
    Ok(excess == 0)
}

#[cfg(test)]
mod identity_tests {
    use super::*;
    use crate::{CancellationToken, JsonMap, StreamJobContext};

    #[tokio::test(flavor = "current_thread")]
    async fn chunk_worker_leaves_timers_running_and_keeps_workspace_until_exit() {
        let (operator, _) = identity_fixture();
        let pool = operator.runtime.pool.clone();
        let workspace = AdmissionWorkspace {
            _identity: operator.reserve_workspace(4_096).unwrap(),
            _payload: operator.reserve_workspace(0).unwrap(),
            _keys: None,
        };
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let gate = Arc::new(std::sync::Barrier::new(2));
        let worker_gate = gate.clone();
        // A timeout releases the gate even when the synchronous red version
        // blocks polling, so the regression fails instead of hanging.
        let releaser = std::thread::spawn(move || {
            let _ = release_rx.recv_timeout(std::time::Duration::from_secs(1));
            gate.wait();
        });
        let mut future = Box::pin(chunk_worker(workspace, move || {
            started_tx.send(()).unwrap();
            worker_gate.wait();
            Ok(())
        }));
        assert!(futures::poll!(future.as_mut()).is_pending());
        started_rx
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        drop(future);
        assert_eq!(pool.reserved(), 4_096);
        release_tx.send(()).unwrap();
        tokio::task::spawn_blocking(move || releaser.join().unwrap())
            .await
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while pool.reserved() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn duplicate_identity_precedes_new_key_copy_workspace_failure() {
        use datafusion::arrow::array::{Int64Array, StringArray};
        let (mut operator, schema) = identity_fixture();
        let key = "k".repeat(1_000_000);
        let record = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![key.as_str(); 2])),
                Arc::new(TimestampMicrosecondArray::from(vec![10; 2]).with_timezone("UTC")),
                Arc::new(Int64Array::from(vec![1; 2])),
            ],
        )
        .unwrap();
        let batch = Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
        let input = operator.validate_admission("right", &batch).unwrap();
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "asof", None);
        let error = operator
            .prepare_admission(input, &batch, &context)
            .await
            .err()
            .expect("duplicate");
        assert!(matches!(
            error,
            crate::CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofDuplicateIdentity,
                ..
            }
        ));
        assert_eq!(operator.status.right.duplicate_rows, 1);
        assert_eq!(operator.status.right.accepted_rows, 0);
        assert_eq!(operator.runtime.pool.reserved(), 0);
    }

    fn identity_fixture() -> (
        StreamAsofJoinOperator,
        datafusion::arrow::datatypes::SchemaRef,
    ) {
        use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("seq", DataType::Int64, false),
        ]));
        let side = |prefix: &str| {
            AsofJoinSide::new(
                vec!["key".into()],
                "time".into(),
                vec!["seq".into()],
                prefix.into(),
            )
            .unwrap()
        };
        let spec = StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            std::time::Duration::ZERO,
            super::super::AsofStateLimits::new(10_000, 2 << 20).unwrap(),
        )
        .unwrap();
        (
            StreamAsofJoinOperator::new("asof", schema.clone(), schema.clone(), spec).unwrap(),
            schema,
        )
    }

    #[test]
    fn repeated_long_keys_charge_one_retained_copy() {
        use datafusion::arrow::array::{Int64Array, StringArray};
        let (operator, schema) = identity_fixture();
        let key = "k".repeat(48_000);
        let record = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from_iter_values((0..32).map(|_| key.as_str()))),
                Arc::new(
                    TimestampMicrosecondArray::from_iter_values((0..32).map(|_| 10))
                        .with_timezone("UTC"),
                ),
                Arc::new(Int64Array::from_iter_values(0..32)),
            ],
        )
        .unwrap();
        let batch = Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
        let input = ValidatedInput {
            index: 1,
            watermark: None,
        };
        let _workspace = operator
            .identity_workspace(&batch, input)
            .expect("32 copies of the canonical batch buffer fit 2 MiB");
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "asof", None);
        let identities = operator
            .admission_identities(batch.table_payload().unwrap().batches(), input, &context)
            .unwrap();
        assert_eq!(identities.rows.len(), 32);
        let first = identities.rows[0].0.1.as_slice().as_ptr();
        assert!(
            identities
                .rows
                .iter()
                .all(|row| row.0.1.as_slice().as_ptr() == first)
        );
    }

    #[test]
    fn typed_string_dictionary_reserves_unique_value_copy_before_arrow_take() {
        use datafusion::arrow::array::{Int64Array, StringArray};
        let (operator, schema) = identity_fixture();
        let keys = (0..3)
            .map(|row| format!("{row}{}", "x".repeat(512_000)))
            .collect::<Vec<_>>();
        let record = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(keys)),
                Arc::new(TimestampMicrosecondArray::from(vec![10; 3]).with_timezone("UTC")),
                Arc::new(Int64Array::from(vec![1, 2, 3])),
            ],
        )
        .unwrap();
        let batch = Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
        let input = ValidatedInput {
            index: 1,
            watermark: None,
        };
        let identity = operator.identity_workspace(&batch, input).unwrap();
        assert!(
            identity.size() < 2 << 20,
            "the original identity workspace fits"
        );
        let error = InputEncodings::new(
            &operator,
            &batch.table_payload().unwrap().batches()[0],
            input,
        )
        .err()
        .expect("typed key value copies require their own workspace");
        assert!(matches!(
            error,
            crate::CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
                ..
            }
        ));
        assert_eq!(operator.runtime.pool.reserved(), identity.size());
        drop(identity);
        assert_eq!(operator.runtime.pool.reserved(), 0);
    }

    #[test]
    fn admitted_key_handles_distinguish_full_hash_collisions() {
        let (operator, _) = identity_fixture();
        let mut keys = InputKeys::default();
        for bytes in [b"one".as_slice(), b"two", b"one"] {
            assert_eq!(
                keys.intern(bytes, Some(0), &operator).unwrap().as_slice(),
                bytes
            );
        }
        assert_eq!(keys.values.len(), 2);
        assert_eq!(keys.values[0].rows, 2);
        assert_eq!(keys.values[1].rows, 1);
    }

    #[test]
    fn one_late_row_keeps_batch_identity_conversion() {
        use datafusion::arrow::array::{Int64Array, StringArray};

        let (operator, schema) = identity_fixture();
        let record = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from_iter_values((0..1_000).map(|_| "kept"))),
                Arc::new(
                    TimestampMicrosecondArray::from_iter_values(
                        (0..1_000).map(|row| if row == 500 { 0 } else { 10 }),
                    )
                    .with_timezone("UTC"),
                ),
                Arc::new(Int64Array::from_iter_values(0..1_000)),
            ],
        )
        .unwrap();
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut retained = None;
        let allocation = allocation_counter::measure(|| {
            retained = Some(
                operator
                    .admission_identities(
                        std::slice::from_ref(&record),
                        ValidatedInput {
                            index: 1,
                            watermark: Some(1),
                        },
                        &context,
                    )
                    .unwrap()
                    .rows
                    .len(),
            );
        });
        assert_eq!(retained, Some(999));
        assert!(
            allocation.count_total <= 128,
            "batch converters were rebuilt per row: {allocation:?}"
        );
    }

    #[test]
    fn discarded_large_keys_do_not_allocate_identity_buffers() {
        use datafusion::arrow::array::{Int64Array, StringArray};

        let (operator, schema) = identity_fixture();
        let discarded = "x".repeat(4_096);
        let record = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from_iter_values((0..1_000).map(|row| {
                    if row == 999 {
                        "kept"
                    } else {
                        discarded.as_str()
                    }
                }))),
                Arc::new(
                    TimestampMicrosecondArray::from_iter_values(
                        (0..1_000).map(|row| if row == 999 { 10 } else { 0 }),
                    )
                    .with_timezone("UTC"),
                ),
                Arc::new(Int64Array::from_iter_values(0..1_000)),
            ],
        )
        .unwrap();
        let batch = Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "asof", None);
        for frontier in [1, 11] {
            let input = ValidatedInput {
                index: 1,
                watermark: Some(frontier),
            };
            let reservation = operator.identity_workspace(&batch, input).unwrap();
            let charge = reservation.size() as u64;
            let mut retained = None;
            let allocation = allocation_counter::measure(|| {
                retained = Some(
                    operator
                        .admission_identities(
                            batch.table_payload().unwrap().batches(),
                            input,
                            &context,
                        )
                        .unwrap(),
                );
            });
            assert!(
                allocation.bytes_max <= charge,
                "frontier={frontier}, peak={}, reserved={charge}",
                allocation.bytes_max
            );
            assert_eq!(retained.unwrap().rows.len(), usize::from(frontier == 1));
        }
    }

    #[test]
    fn long_unique_key_buffers_fit_identity_workspace() {
        use datafusion::arrow::array::{Int64Array, StringArray};

        let (operator, schema) = identity_fixture();
        let key = "k".repeat(48_000);
        let record = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec![key.as_str()])),
                Arc::new(TimestampMicrosecondArray::from(vec![10]).with_timezone("UTC")),
                Arc::new(Int64Array::from(vec![1])),
            ],
        )
        .unwrap();
        let batch = Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
        let input = ValidatedInput {
            index: 1,
            watermark: None,
        };
        let reservation = operator.identity_workspace(&batch, input).unwrap();
        let charge = reservation.size() as u64;
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "asof", None);
        let mut retained = None;
        let allocation = allocation_counter::measure(|| {
            retained = Some(
                operator
                    .admission_identities(batch.table_payload().unwrap().batches(), input, &context)
                    .unwrap(),
            );
        });
        let key_copies = retained
            .as_ref()
            .unwrap()
            .key_workspace
            .as_ref()
            .map_or(0, |reservation| reservation.size() as u64);
        let hash_vector = size_of::<u64>() as u64;
        let total_workspace = charge + key_copies + hash_vector;
        assert!(
            allocation.bytes_max <= total_workspace,
            "peak={}, reserved={total_workspace}",
            allocation.bytes_max
        );
        assert_eq!(retained.unwrap().rows.len(), 1);
    }

    #[test]
    fn ordered_and_unordered_duplicates_count_each_rejected_row_once() {
        let key = state::Encoding::from_slice(&[1]);
        let sequence = state::Encoding::from_slice(&[2]);
        let mut state = state::State::default();
        state
            .right
            .bucket_mut_or_default(key.clone())
            .insert((1, sequence.clone()), None);
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "asof", None);
        let identity = |time| (time, key.clone(), sequence.clone());

        let ordered = [identity(1), identity(1), identity(2), identity(3)];
        assert_eq!(
            count_duplicate_identities(&state, 1, ordered.iter(), &context).unwrap(),
            2
        );
        let appended = [identity(4), identity(5)];
        assert_eq!(
            count_duplicate_identities(&state, 1, appended.iter(), &context).unwrap(),
            0
        );
        let shuffled = [identity(3), identity(2), identity(3)];
        assert_eq!(
            count_duplicate_identities(&state, 1, shuffled.iter(), &context).unwrap(),
            1
        );
    }
}
