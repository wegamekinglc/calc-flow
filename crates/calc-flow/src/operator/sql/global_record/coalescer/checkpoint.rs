use super::super::super::super::{
    SqlCheckpointInput, decode_sql_state, encode_sql_state_async, ipc, sql_state_error,
};
use super::super::super::{
    IncrementalSql, compact_state::NativeStateDescriptor, predicate::InputPredicate,
};
use super::{RecordBatch, Result, State, checked_bytes, df_error};
use crate::{Batch, BatchMetadata, OperatorStateSnapshot, StateSegment, StreamOperatorContext};
use datafusion::{
    arrow::datatypes::SchemaRef, common::ScalarValue, execution::memory_pool::MemoryReservation,
};
use serde::{Deserialize, Serialize};

type SavedStates = Vec<Vec<ScalarValue>>;

struct CoalescerRestoreInput {
    complete: Batch,
    tail: Batch,
    descriptor: NativeStateDescriptor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub(in crate::operator::sql) enum Inventory {
    None,
    Filter {
        complete: String,
        tail: String,
        tail_rows: u64,
    },
}

pub(in crate::operator::sql) struct Capture {
    pub complete: StateSegment,
    pub tail: StateSegment,
    pub tail_rows: u64,
}

impl Capture {
    pub(in crate::operator::sql) fn inventory(&self) -> Inventory {
        Inventory::Filter {
            complete: self.complete.sha256().into(),
            tail: self.tail.sha256().into(),
            tail_rows: self.tail_rows,
        }
    }
}

impl Inventory {
    pub(in crate::operator::sql) fn validate(
        &self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<()> {
        match self {
            Self::None
                if !snapshot.segments.contains_key("global-complete")
                    && !snapshot.segments.contains_key("global-tail") =>
            {
                Ok(())
            }
            Self::Filter {
                complete,
                tail,
                tail_rows,
            } if *tail_rows < super::ROWS as u64
                && snapshot
                    .segments
                    .get("global-complete")
                    .is_some_and(|segment| segment.sha256() == complete)
                && snapshot
                    .segments
                    .get("global-tail")
                    .is_some_and(|segment| segment.sha256() == tail) =>
            {
                Ok(())
            }
            _ => Err(sql_state_error("SQL global coalescer inventory is invalid")),
        }
    }
}

impl IncrementalSql {
    fn coalescer_records(
        &self,
        name: &str,
    ) -> Result<Option<([RecordBatch; 2], MemoryReservation)>> {
        if !self.requires_global_coalescer() {
            return Ok(None);
        }
        let state = self
            .global_records
            .as_ref()
            .and_then(|proof| proof.coalesced.as_ref())
            .ok_or_else(|| df_error(name, "global coalescer state is missing"))?;
        let descriptor = self.native_descriptor(name)?;
        let reservation = self.coalescer_capture_credit(state, name)?;
        let complete = state.complete_record(descriptor.wire_schema, name)?;
        Ok(Some(([complete, state.tail.clone()], reservation)))
    }

    fn coalescer_capture_credit(&self, state: &State, name: &str) -> Result<MemoryReservation> {
        let reservation = self.reservation.new_empty();
        let bytes = state
            .tail
            .columns()
            .iter()
            .try_fold(16384, |bytes, array| {
                checked_bytes(bytes, [(array.get_array_memory_size(), 2)], name)
            })?;
        super::super::super::ensure_reservation(
            &reservation,
            checked_bytes(bytes, [(self.aggregates.len(), 4096)], name)?,
            name,
        )?;
        Ok(reservation)
    }

    pub(in crate::operator::sql) fn capture_coalescer(
        &self,
        name: &str,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<Option<Capture>> {
        let Some(([complete, tail], reservation)) = self.coalescer_records(name)? else {
            return Ok(None);
        };
        check()?;
        let tail_rows = tail.num_rows() as u64;
        let complete = ipc::encode(
            &Batch::table(vec![complete], BatchMetadata::default())?,
            reservation.new_empty(),
            check,
        )?;
        let tail = ipc::encode(
            &Batch::table(vec![tail], BatchMetadata::default())?,
            reservation.new_empty(),
            check,
        )?;
        Ok(Some(Capture {
            complete: complete.segment.clone(),
            tail: tail.segment.clone(),
            tail_rows,
        }))
    }

    pub(in crate::operator::sql) async fn capture_coalescer_async(
        &self,
        name: &str,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<Capture>> {
        let Some(([complete, tail], reservation)) = self.coalescer_records(name)? else {
            return Ok(None);
        };
        context.check_cancelled()?;
        let tail_rows = tail.num_rows() as u64;
        let complete = encode(complete, &reservation, context, name).await?;
        let tail = encode(tail, &reservation, context, name).await?;
        Ok(Some(Capture {
            complete,
            tail,
            tail_rows,
        }))
    }

    pub(in crate::operator::sql) fn restore_coalescer(
        &mut self,
        inventory: &Inventory,
        snapshot: &OperatorStateSnapshot,
        rows: u64,
        check: &dyn Fn() -> Result<()>,
        name: &str,
    ) -> Result<()> {
        let Some(tail_rows) = self.coalescer_tail_rows(inventory, snapshot, rows, name)? else {
            return Ok(());
        };
        check()?;
        let input = self.coalescer_restore_input(snapshot, tail_rows, name)?;
        let (states, tail, reservation) =
            self.prepare_coalescer_restore(&input, rows, check, name)?;
        check()?;
        self.global_records
            .as_mut()
            .expect("global coalescer proof")
            .coalesced = Some(State::restore(states, tail, reservation, name)?);
        Ok(())
    }

    fn coalescer_tail_rows(
        &self,
        inventory: &Inventory,
        snapshot: &OperatorStateSnapshot,
        rows: u64,
        name: &str,
    ) -> Result<Option<u64>> {
        inventory.validate(snapshot)?;
        let Inventory::Filter { tail_rows, .. } = inventory else {
            return if self.requires_global_coalescer() {
                Err(df_error(name, "global coalescer checkpoint is missing"))
            } else {
                Ok(None)
            };
        };
        if !self.requires_global_coalescer() || *tail_rows > rows {
            return Err(df_error(
                name,
                "global coalescer checkpoint differs from plan",
            ));
        }
        Ok(Some(*tail_rows))
    }

    fn coalescer_restore_input(
        &self,
        snapshot: &OperatorStateSnapshot,
        tail_rows: u64,
        name: &str,
    ) -> Result<CoalescerRestoreInput> {
        let complete = decode_sql_state(snapshot.segments["global-complete"].bytes())?;
        let tail = decode_sql_state(snapshot.segments["global-tail"].bytes())?;
        let descriptor = self.native_descriptor(name)?;
        validate_complete_record(&complete, &descriptor.wire_schema, name)?;
        validate_tail_record(&tail, &self.schema, tail_rows, name)?;
        Ok(CoalescerRestoreInput {
            complete,
            tail,
            descriptor,
        })
    }

    fn prepare_coalescer_restore(
        &self,
        input: &CoalescerRestoreInput,
        rows: u64,
        check: &dyn Fn() -> Result<()>,
        name: &str,
    ) -> Result<(SavedStates, RecordBatch, MemoryReservation)> {
        let record = &input.complete.table_payload()?.batches()[0];
        let (reservation, predicate) = self.coalescer_restore_workspace(&input.tail, name)?;
        let states = complete_states(record, &input.descriptor, self.aggregates.len(), name)?;
        let tail = input.tail.table_payload()?.batches()[0].clone();
        validate_tail_selection(predicate, &tail, name)?;
        self.validate_coalescer_counts(&states, rows, tail.num_rows(), name)?;
        self.validate_coalescer_prefix(&states, &tail, check, name)?;
        Ok((states, tail, reservation))
    }

    fn coalescer_restore_workspace(
        &self,
        tail: &Batch,
        name: &str,
    ) -> Result<(MemoryReservation, &InputPredicate)> {
        let reservation = self.reservation.new_empty();
        let bytes = tail.table_payload()?.batches()[0]
            .columns()
            .iter()
            .try_fold(
                checked_bytes(32768, [(self.aggregates.len(), 8192)], name)?,
                |bytes, array| checked_bytes(bytes, [(array.get_array_memory_size(), 8)], name),
            )?;
        let predicate = self.predicate.as_ref().expect("filtered model");
        let expression_width = checked_bytes(
            0,
            [(self.input_nodes.saturating_sub(self.aggregates.len()), 64)],
            name,
        )?;
        super::super::super::ensure_reservation(
            &reservation,
            checked_bytes(
                bytes,
                [
                    (
                        predicate.workspace(tail.num_rows(), self.aggregates.len(), name)?,
                        1,
                    ),
                    (tail.num_rows(), expression_width),
                ],
                name,
            )?,
            name,
        )?;
        Ok((reservation, predicate))
    }

    fn validate_coalescer_counts(
        &self,
        states: &[Vec<ScalarValue>],
        rows: u64,
        tail_rows: usize,
        name: &str,
    ) -> Result<()> {
        let proof = self
            .global_records
            .as_ref()
            .expect("global coalescer proof");
        for (kind, state) in proof.kinds.iter().zip(states) {
            if super::super::saved_count(kind, state, name)? > rows - tail_rows as u64 {
                return Err(df_error(
                    name,
                    "global completed count exceeds input history",
                ));
            }
        }
        Ok(())
    }

    fn validate_coalescer_prefix(
        &self,
        states: &[Vec<ScalarValue>],
        tail: &RecordBatch,
        check: &dyn Fn() -> Result<()>,
        name: &str,
    ) -> Result<()> {
        let work = super::super::RecordWork {
            records: Vec::new(),
            expressions: self.aggregates.clone(),
            filters: self.aggregate_filters.clone(),
            states: Vec::new(),
            predicate: None,
            input_checks: self.input_checks.clone(),
            previous: None,
            coalesced_scratch: None,
            coalesced_credit: None,
            batch_size: super::ROWS,
            name: name.to_owned(),
            _input_owner: None,
        };
        let mut accumulators = work.accumulators(states, check)?;
        if tail.num_rows() != 0 {
            work.update_batch(tail, &mut accumulators, check)?;
        }
        let actual = work.values(&mut accumulators, check)?;
        let expected = &self.groups[0].states;
        if actual.len() != expected.len()
            || actual.iter().zip(expected).any(|(actual, expected)| {
                actual.len() != expected.len()
                    || actual
                        .iter()
                        .zip(expected)
                        .any(|(a, b)| !scalar_equal(a, b))
            })
        {
            return Err(df_error(
                name,
                "global completed state and tail disagree with prefix state",
            ));
        }
        Ok(())
    }
}

fn validate_complete_record(complete: &Batch, schema: &SchemaRef, name: &str) -> Result<()> {
    if complete.num_rows() != 1
        || complete.table_payload()?.schema() != schema
        || complete.table_payload()?.batches().len() != 1
    {
        return Err(df_error(
            name,
            "global coalescer checkpoint schema or rows are invalid",
        ));
    }
    Ok(())
}

fn validate_tail_record(tail: &Batch, schema: &SchemaRef, rows: u64, name: &str) -> Result<()> {
    if u64::try_from(tail.num_rows()).ok() != Some(rows)
        || tail.table_payload()?.schema() != schema
        || tail.table_payload()?.batches().len() != 1
    {
        return Err(df_error(
            name,
            "global coalescer checkpoint schema or rows are invalid",
        ));
    }
    Ok(())
}

fn complete_states(
    record: &RecordBatch,
    descriptor: &NativeStateDescriptor,
    count: usize,
    name: &str,
) -> Result<SavedStates> {
    let mut column = 0;
    let mut states = Vec::with_capacity(count);
    for fields in &descriptor.state_fields {
        let state = (column..column + fields.len())
            .map(|index| ScalarValue::try_from_array(record.column(index), 0))
            .collect::<datafusion::common::Result<Vec<_>>>()
            .map_err(|error| df_error(name, error))?;
        column += fields.len();
        states.push(state);
    }
    Ok(states)
}

fn validate_tail_selection(
    predicate: &InputPredicate,
    tail: &RecordBatch,
    name: &str,
) -> Result<()> {
    let mask = predicate.evaluate(tail, name)?;
    if mask.iter().any(|selected| selected != Some(true)) {
        return Err(df_error(
            name,
            "global coalescer tail contains rejected rows",
        ));
    }
    Ok(())
}

fn scalar_equal(a: &ScalarValue, b: &ScalarValue) -> bool {
    use ScalarValue;
    match (a, b) {
        (ScalarValue::Float32(a), ScalarValue::Float32(b)) => {
            a.map(f32::to_bits) == b.map(f32::to_bits)
        }
        (ScalarValue::Float64(a), ScalarValue::Float64(b)) => {
            a.map(f64::to_bits) == b.map(f64::to_bits)
        }
        _ => a == b,
    }
}

async fn encode(
    record: RecordBatch,
    credit: &MemoryReservation,
    context: &StreamOperatorContext<'_>,
    name: &str,
) -> Result<StateSegment> {
    context.check_cancelled()?;
    let reservation = credit.new_empty();
    super::super::super::ensure_reservation(
        &reservation,
        checked_bytes(8192, [(record.num_columns(), 512)], name)?,
        name,
    )?;
    let input_reservation = credit.new_empty();
    let backing = record.columns().iter().try_fold(8192, |bytes, array| {
        checked_bytes(bytes, [(array.get_array_memory_size(), 1)], name)
    })?;
    super::super::super::ensure_reservation(&input_reservation, backing, name)?;
    let input = SqlCheckpointInput {
        records: vec![record],
        metadata: BatchMetadata::default(),
        native: None,
        #[cfg(test)]
        native_name: None,
        reservation,
        input_reservation,
    };
    let attempt = tokio_util::sync::CancellationToken::new();
    let _cancel_on_drop = attempt.clone().drop_guard();
    let (segment, _, _) =
        encode_sql_state_async(Some(input), None, context.job().clone(), attempt).await?;
    context.check_cancelled()?;
    Ok(segment.expect("encoded coalescer record").segment.clone())
}
