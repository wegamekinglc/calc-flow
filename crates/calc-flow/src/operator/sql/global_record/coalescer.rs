use super::{
    Arc, DataType, GatherStop, MemoryReservation, RecordBatch, RecordWork, Result, ScalarValue,
    Update, checked_bytes, df_error,
};
use datafusion::arrow::{
    array::{Array, GenericStringArray, OffsetSizeTrait},
    compute::BatchCoalescer,
};

#[path = "coalescer/checkpoint.rs"]
mod checkpoint;
pub(in crate::operator::sql) use checkpoint::{Capture, Inventory};

pub(in crate::operator::sql) const ROWS: usize = 8192;

pub(in crate::operator::sql) struct State {
    pub(super) complete: Vec<Vec<ScalarValue>>,
    pub(super) tail: RecordBatch,
    _reservation: Arc<MemoryReservation>,
}

fn capacity(
    records: &[RecordBatch],
    previous: Option<&State>,
    stop: &GatherStop,
    name: &str,
) -> Result<usize> {
    let schema = records[0].schema();
    let base = checked_bytes(
        16384,
        [(super::super::super::ipc::schema_bytes(&schema)?, 4)],
        name,
    )?;
    schema
        .fields()
        .iter()
        .enumerate()
        .try_fold(base, |bytes, (column, field)| {
            stop.check()?;
            let backing =
                column_capacity(field.data_type(), records, previous, column, stop, name)?;
            checked_bytes(
                bytes,
                [(backing, 1), (ROWS.div_ceil(8), 1), (1, 1024)],
                name,
            )
        })
}

fn column_capacity(
    dtype: &DataType,
    records: &[RecordBatch],
    previous: Option<&State>,
    column: usize,
    stop: &GatherStop,
    name: &str,
) -> Result<usize> {
    match dtype {
        DataType::Utf8 => string_capacity::<i32>(records, previous, column, stop, name),
        DataType::LargeUtf8 => string_capacity::<i64>(records, previous, column, stop, name),
        dtype => {
            let width = dtype
                .primitive_width()
                .or_else(|| matches!(dtype, DataType::Boolean | DataType::Null).then_some(1))
                .ok_or_else(|| df_error(name, "global coalescer column is unsupported"))?;
            checked_bytes(0, [(ROWS, width)], name)
        }
    }
}

fn string_capacity<O: OffsetSizeTrait>(
    records: &[RecordBatch],
    previous: Option<&State>,
    column: usize,
    stop: &GatherStop,
    name: &str,
) -> Result<usize> {
    let mut total = 0;
    let mut largest = 0;
    for record in records.iter().chain(previous.map(|state| &state.tail)) {
        stop.check()?;
        let array = record
            .column(column)
            .as_any()
            .downcast_ref::<GenericStringArray<O>>()
            .ok_or_else(|| df_error(name, "global coalescer string type differs"))?;
        let offsets = array.value_offsets();
        let bytes = (offsets[array.len()] - offsets[0]).as_usize();
        total = checked_bytes(total, [(bytes, 1)], name)?;
        largest = largest.max(largest_string(offsets, stop)?);
    }
    let values = checked_bytes(0, [(ROWS, largest)], name)?.min(total);
    checked_bytes(values, [(ROWS + 1, size_of::<O>())], name)
}

fn largest_string<O: OffsetSizeTrait>(offsets: &[O], stop: &GatherStop) -> Result<usize> {
    let mut largest = 0;
    for (index, pair) in offsets.windows(2).enumerate() {
        if index % ROWS == 0 {
            stop.check()?;
        }
        largest = largest.max((pair[1] - pair[0]).as_usize());
    }
    Ok(largest)
}

pub(super) fn run(
    work: &RecordWork,
    predicate: &super::super::predicate::InputPredicate,
    stop: &GatherStop,
) -> Result<Update> {
    reserve_buffers(work, stop)?;
    let saved = work
        .previous
        .as_ref()
        .map_or(work.states.as_slice(), |state| state.complete.as_slice());
    let check = || stop.check();
    let mut accumulators = work.accumulators(saved, &check)?;
    let schema = work.records[0].schema();
    let mut coalescer =
        BatchCoalescer::new(schema.clone(), ROWS).with_biggest_coalesce_batch_size(Some(ROWS / 2));
    restore_tail(work, &mut coalescer, stop)?;
    update_records(
        work,
        predicate,
        &mut coalescer,
        &mut accumulators,
        stop,
        &check,
    )?;
    finish_update(
        work,
        &mut accumulators,
        &mut coalescer,
        schema,
        stop,
        &check,
    )
}

fn reserve_buffers(work: &RecordWork, stop: &GatherStop) -> Result<()> {
    let bound = capacity(&work.records, work.previous.as_deref(), stop, &work.name)?;
    super::super::ensure_reservation(
        work.coalesced_scratch
            .as_ref()
            .expect("coalescer scratch owner"),
        checked_bytes(0, [(bound, 8)], &work.name)?,
        &work.name,
    )?;
    super::super::ensure_reservation(
        work.coalesced_credit
            .as_ref()
            .expect("coalescer state owner"),
        checked_bytes(bound, [(work.expressions.len(), 4096)], &work.name)?,
        &work.name,
    )?;
    Ok(())
}

fn restore_tail(
    work: &RecordWork,
    coalescer: &mut BatchCoalescer,
    stop: &GatherStop,
) -> Result<()> {
    if let Some(previous) = &work.previous {
        for offset in (0..previous.tail.num_rows()).step_by(ROWS / 2) {
            stop.check()?;
            coalescer
                .push_batch(
                    previous
                        .tail
                        .slice(offset, (ROWS / 2).min(previous.tail.num_rows() - offset)),
                )
                .map_err(|error| df_error(&work.name, error))?;
        }
    }
    Ok(())
}

type Accumulators = [Box<dyn datafusion::logical_expr::Accumulator>];

fn update_records(
    work: &RecordWork,
    predicate: &super::super::predicate::InputPredicate,
    coalescer: &mut BatchCoalescer,
    accumulators: &mut Accumulators,
    stop: &GatherStop,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for record in &work.records {
        stop.check()?;
        for offset in (0..record.num_rows()).step_by(work.batch_size) {
            stop.check()?;
            let batch = record.slice(offset, work.batch_size.min(record.num_rows() - offset));
            push_filtered(work, predicate, batch, coalescer, accumulators, check)?;
        }
    }
    Ok(())
}

fn push_filtered(
    work: &RecordWork,
    predicate: &super::super::predicate::InputPredicate,
    batch: RecordBatch,
    coalescer: &mut BatchCoalescer,
    accumulators: &mut Accumulators,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    let mask = predicate.evaluate(&batch, &work.name)?;
    coalescer
        .push_batch_with_filter(batch, &mask)
        .map_err(|error| df_error(&work.name, error))?;
    while let Some(batch) = coalescer.next_completed_batch() {
        work.update_batch(&batch, accumulators, check)?;
    }
    Ok(())
}

fn finish_update(
    work: &RecordWork,
    accumulators: &mut Accumulators,
    coalescer: &mut BatchCoalescer,
    schema: super::SchemaRef,
    stop: &GatherStop,
    check: &dyn Fn() -> Result<()>,
) -> Result<Update> {
    let complete = work.values(accumulators, check)?;
    let tail = finish_tail(coalescer, schema, &work.name)?;
    if tail.num_rows() != 0 {
        work.update_batch(&tail, accumulators, check)?;
    }
    let values = work.values(accumulators, check)?;
    let state = finish_state(work, complete, tail, stop)?;
    Ok(Update {
        values,
        coalesced: Some(state),
    })
}

fn finish_tail(
    coalescer: &mut BatchCoalescer,
    schema: super::SchemaRef,
    name: &str,
) -> Result<RecordBatch> {
    coalescer
        .finish_buffered_batch()
        .map_err(|error| df_error(name, error))?;
    let tail = coalescer
        .next_completed_batch()
        .unwrap_or_else(|| RecordBatch::new_empty(schema));
    if tail.num_rows() >= ROWS || coalescer.next_completed_batch().is_some() {
        return Err(df_error(
            name,
            "global coalescer tail exceeds one partial batch",
        ));
    }
    Ok(tail)
}

fn finish_state(
    work: &RecordWork,
    complete: Vec<Vec<ScalarValue>>,
    tail: RecordBatch,
    stop: &GatherStop,
) -> Result<Arc<State>> {
    let reservation = work
        .coalesced_credit
        .as_ref()
        .expect("paid coalescer state")
        .clone();
    let charge = retained_bytes(&complete, &tail, &work.name)?;
    super::super::ensure_reservation(&reservation, charge, &work.name)?;
    if reservation.size() > charge {
        reservation.shrink(reservation.size() - charge);
    }
    stop.check()?;
    Ok(Arc::new(State {
        complete,
        tail,
        _reservation: reservation,
    }))
}

fn retained_bytes(complete: &[Vec<ScalarValue>], tail: &RecordBatch, name: &str) -> Result<usize> {
    let bytes = checked_bytes(
        8192,
        [
            (complete.len(), 4096),
            (super::super::super::ipc::schema_bytes(&tail.schema())?, 4),
        ],
        name,
    )?;
    tail.columns().iter().try_fold(bytes, |bytes, array| {
        checked_bytes(bytes, [(array.get_array_memory_size(), 1)], name)
    })
}

impl State {
    pub(in crate::operator::sql) fn complete_record(
        &self,
        schema: super::SchemaRef,
        name: &str,
    ) -> Result<RecordBatch> {
        let columns = self
            .complete
            .iter()
            .flatten()
            .map(ScalarValue::to_array)
            .collect::<datafusion::common::Result<Vec<_>>>()
            .map_err(|error| df_error(name, error))?;
        RecordBatch::try_new(schema, columns).map_err(|error| df_error(name, error))
    }

    pub(in crate::operator::sql) fn restore(
        complete: Vec<Vec<ScalarValue>>,
        tail: RecordBatch,
        reservation: MemoryReservation,
        name: &str,
    ) -> Result<Arc<Self>> {
        let charge = retained_bytes(&complete, &tail, name)?;
        super::super::ensure_reservation(&reservation, charge, name)?;
        if reservation.size() > charge {
            reservation.shrink(reservation.size() - charge);
        }
        Ok(Arc::new(Self {
            complete,
            tail,
            _reservation: Arc::new(reservation),
        }))
    }
}
