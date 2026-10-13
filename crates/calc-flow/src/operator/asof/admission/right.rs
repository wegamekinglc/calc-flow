//! Append-only right admission with typed columns and batch-local key IDs.

#[cfg(test)]
use super::identity_tests;
use super::{
    Admission, AdmissionRef, AdmissionWorkspace, Array, Batch, InputKeys, LeftOrder, RecordBatch,
    ReservedIdentities, Result, StreamAsofJoinOperator, StreamOperatorContext,
    StreamingFailureReason, TimestampMicrosecondArray, UInt64Array, ValidatedInput,
    admitted_right_capacities, check_input_cancellation, direct, reason, state, times,
};

struct RecordColumns {
    times: TimestampMicrosecondArray,
    sequences: UInt64Array,
    keys: Vec<u32>,
}

pub(in super::super) struct Columns {
    records: Vec<RecordColumns>,
}

pub(super) struct Plan {
    columns: Columns,
    capacities: Vec<(state::Encoding, usize)>,
}

pub(super) fn plan(
    operator: &StreamAsofJoinOperator,
    batch: &Batch,
    input: ValidatedInput,
    context: &StreamOperatorContext<'_>,
) -> Result<Option<Plan>> {
    if !eligible(operator, batch, input) {
        return Ok(None);
    }
    let side = operator.spec.right();
    let records = batch.table_payload()?.batches();
    if !direct::strictly_ordered(records, side, input, context)? {
        return Ok(None);
    }
    let mut keys = InputKeys::default();
    let mut columns = Vec::new();
    for record in records.iter().filter(|record| record.num_rows() != 0) {
        let Some(record) = RecordColumns::prepare(operator, record, &mut keys, context)? else {
            return Ok(None);
        };
        columns.push(record);
    }
    context.check_cancelled()?;
    Ok(Some(Plan {
        columns: Columns { records: columns },
        capacities: admitted_right_capacities(keys.values, 1),
    }))
}

fn eligible(operator: &StreamAsofJoinOperator, batch: &Batch, input: ValidatedInput) -> bool {
    #[cfg(test)]
    if identity_tests::columnar::legacy_requested() {
        return false;
    }
    input.index == 1 && direct::eligible(operator, batch, input)
}

fn dictionary_mapping(
    operator: &StreamAsofJoinOperator,
    rows: &state::EncodedBatch,
    keys: &mut InputKeys,
    context: &StreamOperatorContext<'_>,
) -> Result<Vec<u32>> {
    (0..rows.len())
        .map(|id| {
            check_input_cancellation(id, context)?;
            let (_, index) = keys.intern(rows.row(id), None, operator)?;
            // The dictionary lookup does not itself admit a row.
            keys.values[index as usize].rows -= 1;
            Ok(index)
        })
        .collect()
}

impl RecordColumns {
    fn prepare(
        operator: &StreamAsofJoinOperator,
        record: &RecordBatch,
        keys: &mut InputKeys,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<Self>> {
        context.check_cancelled()?;
        let side = operator.spec.right();
        let encoded = state::encode_key_columns(record, side.keys(), |bytes| {
            operator.reserve_workspace(bytes)
        })?;
        if !encoded.is_inline_string_dictionary() {
            return Ok(None);
        }
        let state::EncodedColumns::StringKeys { rows, ids } = encoded else {
            unreachable!("proven inline string dictionary")
        };
        u32::try_from(ids.len() - 1).map_err(|_| {
            reason(
                &operator.name,
                StreamingFailureReason::AsofCounterOverflow,
                "ASOF payload row exceeds compact reference range",
            )
        })?;
        let mapping = dictionary_mapping(operator, &rows, keys, context)?;
        let times = times(record, side);
        let sequences = record
            .column_by_name(&side.sequence_by()[0])
            .expect("validated sequence")
            .as_any()
            .downcast_ref::<UInt64Array>()
            .expect("direct unsigned sequence");
        let mut columns = Self {
            times: times.clone(),
            sequences: sequences.clone(),
            keys: ids,
        };
        if !columns.remap(operator, &mapping, keys, context)? {
            return Ok(None);
        }
        Ok(Some(columns))
    }

    fn remap(
        &mut self,
        operator: &StreamAsofJoinOperator,
        mapping: &[u32],
        keys: &mut InputKeys,
        context: &StreamOperatorContext<'_>,
    ) -> Result<bool> {
        for (row, id) in self.keys.iter_mut().enumerate() {
            check_input_cancellation(row, context)?;
            *id = mapping[*id as usize];
            let key = &mut keys.values[*id as usize];
            if key.rows == 0
                && !follows_retained(
                    operator,
                    &key.encoding,
                    self.times.value(row),
                    self.sequences.value(row),
                )
            {
                return Ok(false);
            }
            key.rows += 1;
        }
        Ok(true)
    }
}

fn follows_retained(
    operator: &StreamAsofJoinOperator,
    key: &state::Encoding,
    time: i64,
    sequence: u64,
) -> bool {
    let sequence = sequence_encoding(sequence);
    !operator
        .state
        .right
        .get(key)
        .and_then(state::RightBucket::last_key_value)
        .is_some_and(|((previous_time, previous), _)| {
            (previous_time, previous.as_ref()) >= (&time, &sequence)
        })
}

impl Plan {
    pub(super) fn prepare(
        self,
        operator: &mut StreamAsofJoinOperator,
        batch: &Batch,
        input: ValidatedInput,
        identity: ReservedIdentities,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Admission> {
        operator.record_duplicates(1, 0)?;
        let accepted = operator.check_admission_rows(1, identity.rows as u64)?;
        let payload = operator.input_workspace(batch, input)?;
        let batches = direct::encode_complete_payloads_for(operator, batch, 1)?;
        context.check_cancelled()?;
        Ok(Admission {
            rows: Vec::new(),
            batches,
            accepted,
            left_chunks: None,
            right_capacities: self.capacities,
            right_buckets: None,
            right_columns: Some(self.columns),
            _workspace: AdmissionWorkspace {
                _identity: identity.reservation,
                _payload: payload,
                _keys: None,
            },
        })
    }
}

impl Columns {
    pub(in super::super) fn install(
        self,
        state: &mut state::RightState,
        handles: &[u32],
        payloads: &[state::RowRef],
    ) {
        const STACK_BUCKETS: usize = 64;
        if handles.len() <= STACK_BUCKETS {
            // COW and capacity preparation already made these buckets unique.
            // Keep their borrows on the stack without new workspace charges.
            let mut buckets = state.reserved_unsigned_buckets::<STACK_BUCKETS>(handles);
            self.visit_rows(payloads, |key, time, sequence, row| {
                buckets[key as usize]
                    .as_deref_mut()
                    .expect("admitted ASOF key")
                    .push_admitted_unsigned(time, sequence, row);
            });
        } else {
            self.visit_rows(payloads, |key, time, sequence, row| {
                state.push_reserved_unsigned(handles[key as usize], time, sequence, row);
            });
        }
    }

    fn visit_rows(
        self,
        payloads: &[state::RowRef],
        mut install: impl FnMut(u32, i64, u64, state::RowRef),
    ) {
        for (record, payload) in self.records.into_iter().zip(payloads) {
            for (row, key) in record.keys.into_iter().enumerate() {
                install(
                    key,
                    record.times.value(row),
                    record.sequences.value(row),
                    payload.with_row(u32::try_from(row).expect("preflighted compact reference")),
                );
            }
        }
    }

    pub(in super::super) fn identities<'a>(
        &'a self,
        keys: &'a [(state::Encoding, usize)],
    ) -> impl Iterator<Item = (LeftOrder, AdmissionRef)> + 'a {
        self.records
            .iter()
            .enumerate()
            .flat_map(move |(batch_index, record)| {
                record.keys.iter().enumerate().map(move |(row, key)| {
                    (
                        (
                            record.times.value(row),
                            keys[*key as usize].0.clone(),
                            sequence_encoding(record.sequences.value(row)),
                        ),
                        AdmissionRef {
                            batch_index,
                            row: u32::try_from(row).expect("preflighted compact reference"),
                            key_index: *key,
                        },
                    )
                })
            })
    }
}

fn sequence_encoding(value: u64) -> state::Encoding {
    let mut bytes = [1; 9];
    bytes[1..].copy_from_slice(&value.to_be_bytes());
    state::Encoding::from_slice(&bytes)
}
