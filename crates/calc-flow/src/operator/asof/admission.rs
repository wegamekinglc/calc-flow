use super::{
    AsofJoinSide, AsofLatePolicy, StreamAsofJoinOperator, reason,
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
        let index = match ingress {
            "left" => 0,
            "right" => 1,
            _ => {
                return Err(reason(
                    &self.name,
                    StreamingFailureReason::AsofInvalidInput,
                    "unknown ingress",
                ));
            }
        };
        self.inputs[index].validate(batch, ingress).map_err(|_| {
            reason(
                &self.name,
                StreamingFailureReason::AsofInvalidInput,
                "input does not match the declared exact table schema",
            )
        })?;
        let side = if index == 0 {
            self.spec.left()
        } else {
            self.spec.right()
        };
        let side_status = if index == 0 {
            &mut self.status.left
        } else {
            &mut self.status.right
        };
        if self.terminal || side_status.ended {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofProtocolError,
                "input received data after end-of-input",
            ));
        }
        let batches = batch.table_payload()?.batches();
        validate_nulls(batches, side, &self.name, ingress)?;
        let watermark = side_status
            .watermark_micros
            .map(crate::EventTime::as_micros);
        let late = batches
            .iter()
            .map(|batch| {
                times(batch, side)
                    .values()
                    .iter()
                    .filter(|time| watermark.is_some_and(|wm| **time < wm))
                    .count() as u64
            })
            .sum();
        side_status.late_rows = super::checked(&self.name, side_status.late_rows, late)?;
        if late > 0 && self.spec.late_policy() == AsofLatePolicy::Error {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofLateRow,
                "input contains event time below its accepted watermark",
            ));
        }
        Ok(ValidatedInput { index, watermark })
    }

    pub(super) async fn prepare_admission(
        &mut self,
        input: ValidatedInput,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Admission> {
        let identity_workspace = match self.identity_workspace(batch, input) {
            Ok(reservation) => reservation,
            Err(error) => {
                self.validate_duplicates_without_workspace(batch, input, context)
                    .await?;
                return Err(error);
            }
        };
        let index = input.index;
        let (rows, duplicates) =
            self.admission_identities(batch.table_payload()?.batches(), input, context)?;
        let side_status = if index == 0 {
            &mut self.status.left
        } else {
            &mut self.status.right
        };
        side_status.duplicate_rows =
            super::checked(&self.name, side_status.duplicate_rows, duplicates)?;
        if duplicates > 0 {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofDuplicateIdentity,
                "input contains a duplicate key/event-time/sequence identity",
            ));
        }
        let accepted = super::checked(&self.name, side_status.accepted_rows, rows.len() as u64)?;
        let retained = self.state.inventory(None, &self.name)?.identities;
        if super::checked(&self.name, retained, rows.len() as u64)?
            > self.spec.limits().max_state_rows()
        {
            return Err(reason(
                &self.name,
                StreamingFailureReason::AsofStateLimitExceeded,
                "stream_asof_join.limits.max_state_rows exceeded",
            ));
        }
        let payload_workspace = self.input_workspace(batch, input)?;
        let limit = usize::try_from(self.spec.limits().max_state_bytes()).expect("validated");
        let rows = rows
            .into_iter()
            .map(|(identity, batch, row)| {
                Ok((
                    identity,
                    StateSegment::new(super::codec::encode_batch(&batch.slice(row, 1), limit)?),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Admission {
            rows,
            accepted,
            _identity_workspace: identity_workspace,
            _payload_workspace: payload_workspace,
        })
    }
    fn admission_identities<'a>(
        &self,
        batches: &'a [RecordBatch],
        input: ValidatedInput,
        context: &StreamOperatorContext<'_>,
    ) -> Result<(Vec<InputRow<'a>>, u64)> {
        let side = if input.index == 0 {
            self.spec.left()
        } else {
            self.spec.right()
        };
        let mut seen = BTreeSet::new();
        let mut rows = Vec::new();
        let mut duplicates = 0;
        for batch in batches {
            for row in 0..batch.num_rows() {
                context.check_cancelled()?;
                if input
                    .watermark
                    .is_some_and(|wm| times(batch, side).value(row) < wm)
                {
                    continue;
                }
                let identity = (
                    times(batch, side).value(row),
                    state::encoded_columns(batch, row, side.keys())?,
                    state::encoded_columns(batch, row, side.sequence_by())?,
                );
                let exists = if input.index == 0 {
                    self.state.left.contains_key(&identity)
                } else {
                    self.state.right.get(&identity.1).is_some_and(|bucket| {
                        bucket.contains_key(&(identity.0, identity.2.clone()))
                    })
                };
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
        status: &mut super::StreamAsofJoinStatus,
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
