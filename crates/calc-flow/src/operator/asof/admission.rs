use super::{
    AsofJoinSide, AsofLatePolicy, StreamAsofJoinOperator, StreamAsofJoinSideStatus,
    StreamAsofJoinSpec, StreamAsofJoinStatus, reason,
    state::{self, LeftOrder},
};
use crate::{Batch, Result, StateSegment, StreamOperatorContext, StreamingFailureReason};
use datafusion::arrow::{
    array::{Array, TimestampMicrosecondArray},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
pub(super) struct ValidatedInput {
    pub index: usize,
    pub watermark: Option<i64>,
}

type InputRow<'a> = (LeftOrder, &'a RecordBatch, usize);

pub(super) struct Admission {
    pub rows: Vec<(LeftOrder, StateSegment)>,
    pub accepted: u64,
    _identity_workspace: MemoryReservation,
    _payload_workspace: MemoryReservation,
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

    pub(super) async fn prepare_admission(
        &mut self,
        input: ValidatedInput,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Admission> {
        let identity_workspace = self
            .reserve_admission_identities(batch, input, context)
            .await?;
        let (rows, duplicates) =
            self.admission_identities(batch.table_payload()?.batches(), input, context)?;
        self.record_duplicates(input.index, duplicates)?;
        let accepted = self.check_admission_rows(input.index, rows.len() as u64)?;
        let payload_workspace = self.input_workspace(batch, input)?;
        let limit = usize::try_from(self.spec.limits().max_state_bytes()).expect("validated");
        let rows = encode_rows(rows, limit)?;
        Ok(Admission {
            rows,
            accepted,
            _identity_workspace: identity_workspace,
            _payload_workspace: payload_workspace,
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
        let status = side_status(&mut self.status, index);
        let accepted = super::checked(&self.name, status.accepted_rows, rows)?;
        let retained = self.state.inventory(None, &self.name)?.identities;
        if super::checked(&self.name, retained, rows)? > self.spec.limits().max_state_rows() {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofStateLimitExceeded,
                "stream_asof_join.limits.max_state_rows exceeded",
            ));
        }
        Ok(accepted)
    }

    fn admission_identities<'a>(
        &self,
        batches: &'a [RecordBatch],
        input: ValidatedInput,
        context: &StreamOperatorContext<'_>,
    ) -> Result<(Vec<InputRow<'a>>, u64)> {
        let side = input.side(&self.spec);
        let mut seen = BTreeSet::new();
        let mut rows = Vec::new();
        let mut duplicates = 0;
        for batch in batches {
            for row in 0..batch.num_rows() {
                context.check_cancelled()?;
                if input.is_late(times(batch, side).value(row)) {
                    continue;
                }
                let identity = encoded_identity(batch, row, side)?;
                let exists = self.state.contains_identity(input.index, &identity);
                if exists || !seen.insert(identity.clone()) {
                    duplicates += 1;
                }
                rows.push((identity, batch, row));
            }
        }
        Ok((rows, duplicates))
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
            for (identity, payload) in self.rows.drain(..) {
                state.left.insert(identity, payload);
            }
            status.left.accepted_rows = self.accepted;
        } else {
            for (identity, payload) in self.rows.drain(..) {
                state
                    .right
                    .entry(identity.1)
                    .or_default()
                    .insert((identity.0, identity.2), Some(payload));
            }
            status.right.accepted_rows = self.accepted;
        }
    }
}

fn validate_nulls(
    batches: &[RecordBatch],
    side: &AsofJoinSide,
    node: &str,
    ingress: &str,
) -> Result<()> {
    for column in side
        .keys()
        .iter()
        .chain(side.sequence_by())
        .map(String::as_str)
        .chain(std::iter::once(side.event_time()))
    {
        for batch in batches {
            if batch
                .column(batch.schema().index_of(column).expect("validated"))
                .null_count()
                != 0
            {
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

fn encoded_identity(batch: &RecordBatch, row: usize, side: &AsofJoinSide) -> Result<LeftOrder> {
    Ok((
        times(batch, side).value(row),
        state::encoded_columns(batch, row, side.keys())?,
        state::encoded_columns(batch, row, side.sequence_by())?,
    ))
}

fn encode_rows(rows: Vec<InputRow<'_>>, limit: usize) -> Result<Vec<(LeftOrder, StateSegment)>> {
    rows.into_iter()
        .map(|(identity, batch, row)| {
            Ok((
                identity,
                StateSegment::new(super::codec::encode_batch(&batch.slice(row, 1), limit)?),
            ))
        })
        .collect()
}
