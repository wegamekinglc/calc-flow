use super::{
    AsofJoinSide, AsofLatePolicy, StreamAsofJoinOperator, StreamAsofJoinSideStatus,
    StreamAsofJoinSpec, columnar, reason,
};
use crate::operator::keyed::{KeySchema, MORSEL_ROWS, hash_rows};
use crate::{Batch, Result, StreamingFailureReason};
use datafusion::arrow::{
    array::{Array, ArrayRef, BooleanArray, TimestampMicrosecondArray},
    compute::interleave_record_batch,
    datatypes::Field,
    record_batch::RecordBatch,
};
use datafusion::{common::ScalarValue, execution::memory_pool::MemoryReservation};
use std::{collections::HashMap, sync::Arc};

pub(super) enum PreparedRecord<'a> {
    Input(&'a RecordBatch),
    Assembled {
        record: RecordBatch,
        workspace: MemoryReservation,
    },
}

#[derive(Clone, Copy)]
pub(super) struct ValidatedInput {
    pub index: usize,
    pub watermark: Option<i64>,
    pub accepted: u64,
}

pub(super) struct AcceptedRows {
    pub(super) count: usize,
    pub(super) mask: Option<Vec<bool>>,
}

impl AcceptedRows {
    pub(super) fn all(count: usize) -> Self {
        Self { count, mask: None }
    }

    pub(super) fn for_record(
        record: &RecordBatch,
        side: &AsofJoinSide,
        input: ValidatedInput,
    ) -> Self {
        let count = usize::try_from(input.accepted).expect("validated input row count");
        if count == record.num_rows() {
            return Self::all(record.num_rows());
        }
        let mask = times(record, side)
            .values()
            .iter()
            .map(|time| !input.is_late(*time))
            .collect::<Vec<_>>();
        Self {
            count,
            mask: Some(mask),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct IngressFacts {
    watermark: Option<crate::EventTime>,
    late: u64,
}

impl IngressFacts {
    pub(in crate::operator) fn watermark(&self) -> Option<i64> {
        self.watermark.map(crate::EventTime::as_micros)
    }
}

pub(crate) struct IngressPlan {
    node: String,
    ingress: String,
    port: crate::Port,
    side: AsofJoinSide,
}

impl IngressPlan {
    pub(crate) async fn prepare(
        &self,
        batch: &Batch,
        watermark: Option<crate::EventTime>,
    ) -> Result<IngressFacts> {
        self.port.validate(batch, &self.ingress)?;
        let batches = batch.table_payload()?.batches();
        validate_nulls(batches, &self.side, &self.node, &self.ingress)?;
        let mut late = 0;
        for record in batches {
            for values in times(record, &self.side).values().chunks(4096) {
                if let Some(watermark) = watermark {
                    late += values
                        .iter()
                        .filter(|time| **time < watermark.as_micros())
                        .count() as u64;
                }
                tokio::task::yield_now().await;
            }
        }
        Ok(IngressFacts { watermark, late })
    }
}

impl StreamAsofJoinOperator {
    pub(crate) fn ingress_plan(&self, ingress: &str) -> Result<IngressPlan> {
        let index = ingress_index(ingress, &self.name)?;
        Ok(IngressPlan {
            node: self.name.clone(),
            ingress: ingress.into(),
            port: self.inputs[index].clone(),
            side: if index == 0 {
                self.spec.left()
            } else {
                self.spec.right()
            }
            .clone(),
        })
    }

    pub(super) fn validate_admission_with_facts(
        &mut self,
        ingress: &str,
        batch: &Batch,
        prepared: Option<Result<IngressFacts>>,
    ) -> Result<ValidatedInput> {
        let input = self.admission_input(ingress, batch)?;
        let batches = batch.table_payload()?.batches();
        let late = self.admission_late_count(batches, input, ingress, prepared)?;
        self.validate_late_count(late, input)?;
        let accepted = u64::try_from(batch.num_rows())
            .expect("Arrow row count fits the ASOF counter domain")
            - late;
        Ok(ValidatedInput { accepted, ..input })
    }

    fn admission_input(&mut self, ingress: &str, batch: &Batch) -> Result<ValidatedInput> {
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
        Ok(ValidatedInput {
            index,
            watermark: status.watermark_micros.map(crate::EventTime::as_micros),
            accepted: 0,
        })
    }

    fn admission_late_count(
        &self,
        batches: &[RecordBatch],
        input: ValidatedInput,
        ingress: &str,
        prepared: Option<Result<IngressFacts>>,
    ) -> Result<u64> {
        if let Some(facts) = prepared
            .transpose()?
            .filter(|facts| facts.watermark() == input.watermark)
        {
            return Ok(facts.late);
        }
        validate_nulls(batches, input.side(&self.spec), &self.name, ingress)?;
        Ok(batches
            .iter()
            .map(|batch| {
                times(batch, input.side(&self.spec))
                    .values()
                    .iter()
                    .filter(|time| input.is_late(**time))
                    .count() as u64
            })
            .sum())
    }

    fn validate_late_count(&mut self, late: u64, input: ValidatedInput) -> Result<()> {
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

    pub(super) fn record_duplicate(&mut self, side: usize) -> Result<()> {
        let status = side_status(&mut self.status, side);
        status.duplicate_rows = super::checked(&self.name, status.duplicate_rows, 1)?;
        Ok(())
    }
}

pub(super) fn contains_duplicate_identity(
    record: &RecordBatch,
    side: &AsofJoinSide,
    input: ValidatedInput,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<bool> {
    contains_duplicate_identity_in_records(std::slice::from_ref(record), side, input, check)
}

pub(super) fn contains_duplicate_identity_in_records(
    records: &[RecordBatch],
    side: &AsofJoinSide,
    input: ValidatedInput,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<bool> {
    let Some(first) = records.first() else {
        return Ok(false);
    };
    let mut search = DuplicateIdentitySearch::new(records, side, first, input)?;
    for (record_index, record) in records.iter().enumerate() {
        if search.contains_in_record(record_index, record, check)? {
            return Ok(true);
        }
    }
    Ok(false)
}

struct DuplicateIdentitySearch<'a> {
    records: &'a [RecordBatch],
    side: &'a AsofJoinSide,
    schema: KeySchema,
    input: ValidatedInput,
    seen: HashMap<u64, Vec<(usize, usize)>>,
}

impl<'a> DuplicateIdentitySearch<'a> {
    fn new(
        records: &'a [RecordBatch],
        side: &'a AsofJoinSide,
        first: &RecordBatch,
        input: ValidatedInput,
    ) -> Result<Self> {
        let fields = identity_columns(first, side)?
            .iter()
            .map(|array| Arc::new(Field::new("identity", array.data_type().clone(), false)))
            .collect();
        Ok(Self {
            records,
            side,
            schema: KeySchema::groupable(fields)?,
            input,
            seen: HashMap::new(),
        })
    }

    fn contains_in_record(
        &mut self,
        record_index: usize,
        record: &RecordBatch,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<bool> {
        let columns = identity_columns(record, self.side)?;
        for (morsel, times) in times(record, self.side)
            .values()
            .chunks(MORSEL_ROWS)
            .enumerate()
        {
            check()?;
            let first = morsel * MORSEL_ROWS;
            let kept = self.input.accepted_indices(times).count();
            if kept == 0 {
                continue;
            }
            let selected = (kept < times.len()).then(|| {
                times
                    .iter()
                    .map(|&time| Some(!self.input.is_late(time)))
                    .collect::<BooleanArray>()
            });
            let part = columns
                .iter()
                .map(|column| column.slice(first, times.len()))
                .collect::<Vec<_>>();
            let hashes = hash_rows(&self.schema, &part, selected.as_ref(), check)?;
            for (offset, hash) in hashes.into_iter().enumerate() {
                if selected.as_ref().is_none_or(|mask| mask.value(offset))
                    && self.already_seen(record_index, first + offset, hash, &columns)?
                {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn already_seen(
        &mut self,
        record: usize,
        row: usize,
        hash: u64,
        columns: &[ArrayRef],
    ) -> Result<bool> {
        if let Some(prior) = self.seen.get(&hash)
            && self.contains_prior(prior, columns, row)?
        {
            return Ok(true);
        }
        self.seen.entry(hash).or_default().push((record, row));
        Ok(false)
    }

    fn contains_prior(
        &self,
        prior: &[(usize, usize)],
        columns: &[ArrayRef],
        row: usize,
    ) -> Result<bool> {
        for &(prior_record, prior_row) in prior {
            let prior_columns = identity_columns(&self.records[prior_record], self.side)?;
            if same_identity(&prior_columns, prior_row, columns, row)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

fn identity_columns(record: &RecordBatch, side: &AsofJoinSide) -> Result<Vec<ArrayRef>> {
    side.keys()
        .iter()
        .map(String::as_str)
        .chain(std::iter::once(side.event_time()))
        .chain(side.sequence_by().iter().map(String::as_str))
        .map(|name| {
            record
                .schema()
                .index_of(name)
                .map(|index| record.column(index).clone())
                .map_err(|error| super::arrow_error(&error))
        })
        .collect()
}

fn same_identity(
    left: &[ArrayRef],
    left_row: usize,
    right: &[ArrayRef],
    right_row: usize,
) -> Result<bool> {
    for (left, right) in left.iter().zip(right) {
        let left_value = ScalarValue::try_from_array(left, left_row).map_err(|error| {
            crate::CalcFlowError::Format {
                message: format!("ASOF duplicate comparison failed: {error}"),
            }
        })?;
        let right_value = ScalarValue::try_from_array(right, right_row).map_err(|error| {
            crate::CalcFlowError::Format {
                message: format!("ASOF duplicate comparison failed: {error}"),
            }
        })?;
        if left_value != right_value {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Identity columns in declaration order: keys, sequence columns, event time.
pub(super) fn identity_column_names(side: &AsofJoinSide) -> impl Iterator<Item = &str> {
    side.keys()
        .iter()
        .chain(side.sequence_by())
        .map(String::as_str)
        .chain(std::iter::once(side.event_time()))
}

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

    pub(super) fn record_charge(
        self,
        record: &RecordBatch,
        side: &AsofJoinSide,
        full: bool,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<usize> {
        if full {
            check()?;
            Ok(columnar::logical_record_selection_charge(
                record,
                &AcceptedRows::all(record.num_rows()),
            ))
        } else {
            columnar::logical_filtered_record_charge(
                record,
                times(record, side).values(),
                self,
                check,
            )
        }
    }

    pub(super) fn gather_records(
        self,
        batches: &[RecordBatch],
        side: &AsofJoinSide,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<RecordBatch> {
        let mut positions =
            Vec::with_capacity(usize::try_from(self.accepted).expect("validated input row count"));
        for (batch, record) in batches.iter().enumerate() {
            for (morsel, times) in times(record, side).values().chunks(MORSEL_ROWS).enumerate() {
                check()?;
                let start = morsel * MORSEL_ROWS;
                positions.extend(self.accepted_indices(times).map(|row| (batch, start + row)));
            }
        }
        interleave_record_batch(&batches.iter().collect::<Vec<_>>(), &positions)
            .map_err(|error| super::arrow_error(&error))
    }

    pub(super) fn accepted_indices(
        self,
        times: &[i64],
    ) -> impl Iterator<Item = usize> + Clone + '_ {
        times
            .iter()
            .enumerate()
            .filter_map(move |(row, time)| (!self.is_late(*time)).then_some(row))
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

fn side_status(
    status: &mut super::StreamAsofJoinStatus,
    index: usize,
) -> &mut StreamAsofJoinSideStatus {
    if index == 0 {
        &mut status.left
    } else {
        &mut status.right
    }
}
