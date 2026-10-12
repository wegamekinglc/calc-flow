//! Ordered inline-string left admission without per-row identity descriptors.

#[cfg(test)]
use super::identity_tests;
use super::{
    Admission, AdmissionWorkspace, Arc, Array, AsofJoinSide, Batch, GatherStop, MemoryReservation,
    OwnedCpuWork, PreparedInput, RecordBatch, ReservedIdentities, Result, StreamAsofJoinOperator,
    StreamOperatorContext, UInt64Array, ValidatedInput, can_share_batch, check_input_cancellation,
    encode_payload, state, take, times,
};
use datafusion::arrow::{array::StringArray, datatypes::DataType};

#[cfg(test)]
mod tests;

pub(super) struct IdentityPlan {
    keys: Vec<state::EncodedColumns>,
    rows: usize,
}

pub(super) fn plan(
    operator: &StreamAsofJoinOperator,
    batch: &Batch,
    input: ValidatedInput,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<IdentityPlan>> {
    #[cfg(test)]
    if identity_tests::columnar::legacy_requested() {
        return Ok(None);
    }
    if input.index != 0 || !eligible(operator, batch, input) {
        return Ok(None);
    }
    let side = operator.spec.left();
    let records = batch.table_payload()?.batches();
    let Some(keys) = encode_ordered_keys(operator, records, side, input, context)? else {
        return Ok(None);
    };
    let first = records
        .iter()
        .find(|record| record.num_rows() != 0)
        .expect("nonempty direct admission");
    let first_time = times(first, side).value(0);
    let first_key = keys[0].row(0);
    let first_sequence = state::encode_columns(first, side.sequence_by())?.row(0);
    if operator
        .state
        .left
        .last_key_value()
        .is_some_and(|(last, _)| {
            last >= (
                &first_time,
                &first_key,
                std::borrow::Cow::Borrowed(&first_sequence),
            )
        })
    {
        return Ok(None);
    }
    context.check_cancelled()?;
    Ok(Some(IdentityPlan {
        keys,
        rows: batch.num_rows(),
    }))
}

pub(super) fn eligible(
    operator: &StreamAsofJoinOperator,
    batch: &Batch,
    input: ValidatedInput,
) -> bool {
    let side = input.side(&operator.spec);
    let schema = &operator.schemas[input.index];
    let ([key], [sequence]) = (side.keys(), side.sequence_by()) else {
        return false;
    };
    if batch.num_rows() == 0
        || schema
            .field_with_name(key)
            .expect("validated key")
            .data_type()
            != &DataType::Utf8
        || schema
            .field_with_name(sequence)
            .expect("validated sequence")
            .data_type()
            != &DataType::UInt64
    {
        return false;
    }
    true
}

fn encode_ordered_keys(
    operator: &StreamAsofJoinOperator,
    records: &[RecordBatch],
    side: &AsofJoinSide,
    input: ValidatedInput,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<Vec<state::EncodedColumns>>> {
    if !strictly_ordered(records, side, input, context)? {
        return Ok(None);
    }
    let mut keys = Vec::new();
    for record in records.iter().filter(|record| record.num_rows() != 0) {
        context.check_cancelled()?;
        let encoded = state::encode_key_columns(record, side.keys(), |bytes| {
            operator.reserve_workspace(bytes)
        })?;
        if !encoded.is_inline_string_dictionary() {
            return Ok(None);
        }
        keys.push(encoded);
    }
    Ok(Some(keys))
}

pub(super) fn strictly_ordered(
    records: &[RecordBatch],
    side: &AsofJoinSide,
    input: ValidatedInput,
    context: &StreamOperatorContext<'_>,
) -> Result<bool> {
    let mut previous = None;
    for record in records {
        let keys = record
            .column_by_name(&side.keys()[0])
            .expect("validated key")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("direct Utf8 key");
        let sequences = record
            .column_by_name(&side.sequence_by()[0])
            .expect("validated sequence")
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("direct unsigned sequence");
        let times = times(record, side);
        for row in 0..record.num_rows() {
            check_input_cancellation(row, context)?;
            let identity = (times.value(row), keys.value(row), sequences.value(row));
            if input.is_late(identity.0) || previous.is_some_and(|last| last >= identity) {
                return Ok(false);
            }
            previous = Some(identity);
        }
    }
    context.check_cancelled()?;
    Ok(true)
}

pub(super) struct ChunkWork {
    keys: Vec<state::EncodedColumns>,
    batches: Vec<Arc<state::PayloadBatch>>,
    workspace: AdmissionWorkspace,
    side: AsofJoinSide,
    name: String,
    _descriptor: MemoryReservation,
}

impl OwnedCpuWork for ChunkWork {
    type Output = PreparedInput;

    fn run(self, stop: &GatherStop) -> Result<PreparedInput> {
        stop.check()?;
        let chunks = state::PreparedLeftChunk::prepare_inline_strings(
            self.keys,
            &self.batches,
            &self.side,
            &self.name,
            &|| stop.check(),
        )?;
        stop.check()?;
        Ok((Vec::new(), Some(chunks), self.workspace))
    }
}

impl IdentityPlan {
    pub(super) async fn prepare(
        self,
        operator: &mut StreamAsofJoinOperator,
        batch: &Batch,
        input: ValidatedInput,
        identity: ReservedIdentities,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Admission> {
        debug_assert_eq!(identity.rows, self.rows);
        operator.record_duplicates(0, 0)?;
        let accepted = operator.check_admission_rows(0, self.rows as u64)?;
        let payload = operator.input_workspace(batch, input)?;
        let batches = encode_complete_payloads(operator, batch)?;
        let workspace = AdmissionWorkspace {
            _identity: identity.reservation,
            _payload: payload,
            _keys: None,
        };
        context.check_cancelled()?;
        let expected_rows = self.rows;
        let (rows, left_chunks, workspace) = self
            .prepare_chunks(operator, &batches, workspace, context)
            .await?;
        let admission = Admission {
            rows,
            batches,
            accepted,
            left_chunks,
            right_capacities: Vec::new(),
            right_buckets: None,
            right_columns: None,
            _workspace: workspace,
        };
        debug_assert_eq!(admission.len(), expected_rows);
        Ok(admission)
    }

    async fn prepare_chunks(
        self,
        operator: &mut StreamAsofJoinOperator,
        batches: &[Arc<state::PayloadBatch>],
        workspace: AdmissionWorkspace,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedInput> {
        // Inline canonical keys retain the existing 4096-row /128KiB work threshold.
        if self.rows <= 4_096 {
            let chunks = state::PreparedLeftChunk::prepare_inline_strings(
                self.keys,
                batches,
                operator.spec.left(),
                &operator.name,
                &|| context.check_cancelled(),
            )?;
            context.check_cancelled()?;
            Ok((Vec::new(), Some(chunks), workspace))
        } else {
            let descriptor = operator.reserve_left_work(batches.len())?;
            let work = ChunkWork {
                keys: self.keys,
                batches: batches.to_vec(),
                workspace,
                side: operator.spec.left().clone(),
                name: operator.name.clone(),
                _descriptor: descriptor,
            };
            operator.run_cpu_work(work, context).await
        }
    }
}

fn encode_complete_payloads(
    operator: &StreamAsofJoinOperator,
    input: &Batch,
) -> Result<Vec<Arc<state::PayloadBatch>>> {
    encode_complete_payloads_for(operator, input, 0)
}

pub(super) fn encode_complete_payloads_for(
    operator: &StreamAsofJoinOperator,
    input: &Batch,
    side: usize,
) -> Result<Vec<Arc<state::PayloadBatch>>> {
    let retained = operator
        .payload_projection
        .as_ref()
        .map(|plan| &plan.columns[side]);
    let base = operator.admission_base(side);
    let mut position = 0;
    input
        .table_payload()?
        .batches()
        .iter()
        .filter(|record| record.num_rows() != 0)
        .map(|record| {
            let projected = retained
                .filter(|columns| columns.len() != record.num_columns())
                .map(|columns| {
                    record
                        .project(columns)
                        .map_err(|error| super::super::arrow_error(&error))
                })
                .transpose()?;
            let compact = compact_complete_record(projected.as_ref().unwrap_or(record))?;
            let payload = encode_payload(
                compact,
                side,
                base + position,
                operator.physical_header(side),
                &operator.name,
            )?;
            position += record.num_rows() as u64;
            Ok(payload)
        })
        .collect()
}

fn compact_complete_record(record: &RecordBatch) -> Result<RecordBatch> {
    if can_share_batch(record)? {
        return Ok(record.clone());
    }
    let indices = UInt64Array::from_iter_values(0..record.num_rows() as u64);
    let columns = record
        .columns()
        .iter()
        .map(|column| {
            take(column, &indices, None).map_err(|error| super::super::arrow_error(&error))
        })
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(record.schema(), columns)
        .map_err(|error| super::super::arrow_error(&error))
}
