use std::{collections::VecDeque, mem::size_of, time::Instant};

use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, Float64Builder, StringArray, TimestampMicrosecondArray,
    },
    compute::{concat_batches, sum as arrow_sum},
    datatypes::{DataType, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use sha2::{Digest, Sha256};

use super::{DataFusionRollingMetrics, DataFusionRollingWindow, RollingExecutionBatch};

#[derive(Clone, Debug)]
pub(super) struct SqlSumRangeKernel {
    partition_index: usize,
    event_time_index: usize,
    input_index: usize,
    preceding_micros: u64,
    fingerprint: String,
}

#[derive(Clone, Debug, Default)]
pub(super) struct SqlSumRangeState {
    entity: Option<String>,
    samples: VecDeque<TimedValue>,
    sum: Option<f64>,
    count: u64,
    pending: Option<RecordBatch>,
}

#[derive(Clone, Copy, Debug)]
struct TimedValue {
    event_time: i64,
    value: Option<f64>,
}

#[derive(Clone, Copy)]
struct InputColumns<'a> {
    entities: &'a StringArray,
    times: &'a TimestampMicrosecondArray,
    values: &'a Float64Array,
}

fn supports_schema(
    schema: &Schema,
    partition_index: usize,
    event_time_index: usize,
    input_index: usize,
) -> bool {
    supports_partition(schema.field(partition_index))
        && supports_event_time(schema.field(event_time_index))
        && schema.field(input_index).data_type() == &DataType::Float64
}

fn supports_partition(field: &datafusion::arrow::datatypes::Field) -> bool {
    field.data_type() == &DataType::Utf8 && !field.is_nullable()
}

fn supports_event_time(field: &datafusion::arrow::datatypes::Field) -> bool {
    !field.is_nullable()
        && matches!(
            field.data_type(),
            DataType::Timestamp(TimeUnit::Microsecond, timezone)
                if timezone.as_deref().is_none_or(|timezone| timezone == "UTC")
        )
}

impl SqlSumRangeKernel {
    pub(super) fn compile(
        schema: &Schema,
        partition_indices: &[usize],
        order_indices: &[usize],
        windows: &[DataFusionRollingWindow],
    ) -> Option<Self> {
        let ([window], [partition_index], [event_time_index]) =
            (windows, partition_indices, order_indices)
        else {
            return None;
        };
        let preceding_micros = window.duration_micros.filter(|micros| *micros > 0)?;
        if !window.is_sum
            || window.is_count
            || !supports_schema(
                schema,
                *partition_index,
                *event_time_index,
                window.input_index,
            )
        {
            return None;
        }
        let fingerprint = format!(
            "sql-sum-range-v1:{:x}",
            Sha256::digest(format!(
                "{partition_indices:?}:{order_indices:?}:{windows:?}"
            ))
        );
        Some(Self {
            partition_index: *partition_index,
            event_time_index: *event_time_index,
            input_index: window.input_index,
            preceding_micros,
            fingerprint,
        })
    }

    pub(super) fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub(super) fn estimated_state_bytes_per_entity() -> usize {
        size_of::<SqlSumRangeState>() + size_of::<TimedValue>()
    }

    pub(super) fn update_and_fill(
        &self,
        state: SqlSumRangeState,
        input: &RecordBatch,
    ) -> crate::Result<RollingExecutionBatch> {
        self.transition(state, input, false)
    }

    pub(super) fn finish(
        &self,
        state: SqlSumRangeState,
        input: &RecordBatch,
    ) -> crate::Result<Option<RollingExecutionBatch>> {
        if state.pending.is_none() {
            return Ok(None);
        }
        self.transition(state, input, true).map(Some)
    }

    fn transition(
        &self,
        mut state: SqlSumRangeState,
        input: &RecordBatch,
        finish: bool,
    ) -> crate::Result<RollingExecutionBatch> {
        let started = Instant::now();
        self.columns(input)?;
        let input_validation_ns = elapsed_ns(started);
        let started = Instant::now();
        let combined = combine_pending(state.pending.take(), input)?;
        let columns = self.columns(&combined)?;
        let held = if finish {
            0
        } else {
            trailing_peer_rows(columns.entities, columns.times)
        };
        let processed = combined.num_rows() - held;
        let pending = (held != 0).then(|| combined.slice(processed, held));
        let output_input = combined.slice(0, processed);
        let entity_encode_ns = elapsed_ns(started);
        let started = Instant::now();
        let mut output = Float64Builder::with_capacity(processed);
        self.fill_range(&mut state, columns, processed, &mut output)?;
        let kernel_ns = elapsed_ns(started);
        let started = Instant::now();
        let columns = vec![std::sync::Arc::new(output.finish()) as ArrayRef];
        let output_build_ns = elapsed_ns(started);
        state.pending = pending;
        let state_bytes = state.estimated_bytes();
        let entities = usize::from(state.entity.is_some());
        Ok(RollingExecutionBatch {
            input: output_input,
            columns,
            state: super::RollingExecutionState::SqlSumRange(state),
            metrics: DataFusionRollingMetrics {
                input_validation_ns,
                entity_encode_ns,
                kernel_ns,
                output_build_ns,
                input_rows: input.num_rows(),
                entities,
                state_bytes,
                ..DataFusionRollingMetrics::default()
            },
        })
    }

    fn columns<'a>(&self, input: &'a RecordBatch) -> crate::Result<InputColumns<'a>> {
        let entities: &StringArray = typed_column(
            input,
            self.partition_index,
            "SQL range partition type changed",
        )?;
        let times: &TimestampMicrosecondArray = typed_column(
            input,
            self.event_time_index,
            "SQL range event-time type changed",
        )?;
        let values: &Float64Array =
            typed_column(input, self.input_index, "SQL range input type changed")?;
        if entities.null_count() != 0 || times.null_count() != 0 {
            return Err(internal("SQL range required input contains nulls"));
        }
        Ok(InputColumns {
            entities,
            times,
            values,
        })
    }

    fn fill_range(
        &self,
        state: &mut SqlSumRangeState,
        input: InputColumns<'_>,
        rows: usize,
        output: &mut Float64Builder,
    ) -> crate::Result<()> {
        let mut row = 0;
        while row < rows {
            let entity = input.entities.value(row);
            state.start_entity(entity);
            let event_time = input.times.value(row);
            let end = peer_group_end(input, row, rows, entity, event_time);
            state.add_group(input.values, row, end, event_time)?;
            state.expire(event_time, self.preceding_micros)?;
            let result = (state.count != 0).then_some(state.sum).flatten();
            for _ in row..end {
                output.append_option(result);
            }
            row = end;
        }
        Ok(())
    }
}

impl SqlSumRangeState {
    fn start_entity(&mut self, entity: &str) {
        if self.entity.as_deref() != Some(entity) {
            self.entity = Some(entity.to_owned());
            self.samples.clear();
            self.sum = None;
            self.count = 0;
        }
    }

    fn add_group(
        &mut self,
        values: &Float64Array,
        start: usize,
        end: usize,
        event_time: i64,
    ) -> crate::Result<()> {
        let group = values.slice(start, end - start);
        let valid = group.len() - group.null_count();
        self.count = self
            .count
            .checked_add(u64::try_from(valid).unwrap_or(u64::MAX))
            .ok_or_else(|| internal("SQL range count overflowed"))?;
        if let Some(total) = arrow_sum(&group) {
            *self.sum.get_or_insert(0.0) += total;
        }
        for row in start..end {
            let value = (!values.is_null(row)).then(|| values.value(row));
            self.samples.push_back(TimedValue { event_time, value });
        }
        Ok(())
    }

    fn expire(&mut self, event_time: i64, preceding_micros: u64) -> crate::Result<()> {
        let lower = i128::from(event_time) - i128::from(preceding_micros);
        let expired = self
            .samples
            .iter()
            .take_while(|sample| i128::from(sample.event_time) < lower)
            .count();
        if expired == 0 {
            return Ok(());
        }
        let values = self
            .samples
            .drain(..expired)
            .map(|sample| sample.value)
            .collect::<Float64Array>();
        if let Some(total) = arrow_sum(&values) {
            self.sum = Some(self.sum.unwrap_or(0.0) - total);
        }
        let valid = values.len() - values.null_count();
        self.count = self
            .count
            .checked_sub(u64::try_from(valid).unwrap_or(u64::MAX))
            .ok_or_else(|| internal("SQL range count underflowed"))?;
        Ok(())
    }

    fn estimated_bytes(&self) -> usize {
        size_of::<Self>()
            + self.samples.capacity() * size_of::<TimedValue>()
            + self.entity.as_ref().map_or(0, String::capacity)
            + self.pending.as_ref().map_or(0, |pending| {
                pending
                    .columns()
                    .iter()
                    .map(Array::get_array_memory_size)
                    .sum()
            })
    }
}

fn peer_group_end(
    input: InputColumns<'_>,
    row: usize,
    rows: usize,
    entity: &str,
    event_time: i64,
) -> usize {
    let mut end = row + 1;
    while end < rows && input.entities.value(end) == entity && input.times.value(end) == event_time
    {
        end += 1;
    }
    end
}

fn typed_column<'a, T: 'static>(
    input: &'a RecordBatch,
    index: usize,
    message: &'static str,
) -> crate::Result<&'a T> {
    input
        .column(index)
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| internal(message))
}

fn combine_pending(
    pending: Option<RecordBatch>,
    input: &RecordBatch,
) -> crate::Result<RecordBatch> {
    match pending {
        Some(pending) if input.num_rows() != 0 => {
            concat_batches(&pending.schema(), [&pending, input])
                .map_err(|error| internal(&format!("SQL range pending input failed: {error}")))
        }
        Some(pending) => Ok(pending),
        None => Ok(input.clone()),
    }
}

fn trailing_peer_rows(entities: &StringArray, times: &TimestampMicrosecondArray) -> usize {
    let Some(last) = entities.len().checked_sub(1) else {
        return 0;
    };
    let entity = entities.value(last);
    let event_time = times.value(last);
    let mut first = last;
    while first != 0 && entities.value(first - 1) == entity && times.value(first - 1) == event_time
    {
        first -= 1;
    }
    entities.len() - first
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn internal(message: &str) -> crate::CalcFlowError {
    crate::CalcFlowError::Internal {
        message: message.to_owned(),
    }
}
