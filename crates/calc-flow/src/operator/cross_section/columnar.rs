use std::{ops::Range, sync::Arc};

use datafusion::{
    arrow::{
        array::{
            Array, ArrayRef, BooleanBuilder, Float64Array, Float64Builder, StringArray,
            TimestampMicrosecondArray, UInt64Array,
        },
        compute::{concat_batches, take_record_batch},
        datatypes::{DataType, Schema, SchemaRef},
        record_batch::RecordBatch,
    },
    common::ScalarValue,
};

use super::{
    CompiledCrossSectionSpec, CompiledEvaluation, EventTime, PreparedCrossSectionColumn, Result,
    StatisticAccumulator, compute_group_outputs, operator_error, regression_residual_pairs,
};

pub(super) fn supports_payload(schema: &Schema) -> bool {
    // Only flat scalar columns have a bounded future row representation
    // for checkpoints and transitions back to map-based stream state.
    schema.fields().iter().all(|field| {
        matches!(
            field.data_type(),
            DataType::Null
                | DataType::Boolean
                | DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Date32
                | DataType::Date64
                | DataType::Timestamp(_, _)
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Binary
                | DataType::LargeBinary
        )
    })
}

struct Keys<'a> {
    times: &'a TimestampMicrosecondArray,
    entities: &'a StringArray,
    sequences: &'a UInt64Array,
}

impl<'a> Keys<'a> {
    fn read(record: &'a RecordBatch, compiled: &CompiledCrossSectionSpec) -> Self {
        Self {
            times: times(record, compiled),
            entities: record
                .column(compiled.entity_columns[0].index)
                .as_any()
                .downcast_ref()
                .expect("columnar cross-section has a UTF-8 entity"),
            sequences: record
                .column(compiled.sequence_columns[0].index)
                .as_any()
                .downcast_ref()
                .expect("columnar cross-section has a UInt64 sequence"),
        }
    }

    fn at(&self, index: usize) -> (i64, Option<&'a str>, u64) {
        (
            self.times.value(index),
            self.entities
                .is_valid(index)
                .then(|| self.entities.value(index)),
            self.sequences.value(index),
        )
    }

    fn is_canonical(&self, watermark: Option<EventTime>) -> bool {
        let len = self.times.len();
        len != 0
            && self.times.null_count() == 0
            && self.sequences.null_count() == 0
            && watermark.is_none_or(|value| self.times.value(0) > value.as_micros())
            && (1..len).all(|index| self.at(index - 1) < self.at(index))
    }
}

fn times<'a>(
    record: &'a RecordBatch,
    compiled: &CompiledCrossSectionSpec,
) -> &'a TimestampMicrosecondArray {
    record
        .column(compiled.event_time_index)
        .as_any()
        .downcast_ref()
        .expect("columnar cross-section has microsecond event time")
}

pub(super) fn canonical_record(
    records: &[RecordBatch],
    compiled: &CompiledCrossSectionSpec,
    watermark: Option<EventTime>,
) -> Option<RecordBatch> {
    let mut nonempty = records.iter().filter(|record| record.num_rows() != 0);
    let first = nonempty.next()?;
    let first_keys = Keys::read(first, compiled);
    if !first_keys.is_canonical(watermark) {
        return None;
    }
    let mut last = first_keys.at(first.num_rows() - 1);
    let mut multiple = false;
    for record in nonempty {
        let keys = Keys::read(record, compiled);
        if !keys.is_canonical(None) || last >= keys.at(0) {
            return None;
        }
        last = keys.at(record.num_rows() - 1);
        multiple = true;
    }
    if multiple {
        concat_batches(first.schema_ref(), records).ok()
    } else {
        Some(first.clone())
    }
}

pub(super) fn join_advancing(
    pending: &RecordBatch,
    incoming: &RecordBatch,
    compiled: &CompiledCrossSectionSpec,
) -> Option<RecordBatch> {
    let previous = Keys::read(pending, compiled);
    let next = Keys::read(incoming, compiled);
    let last = previous.at(pending.num_rows() - 1);
    // Only carry one open group into an advancing input. Repeated
    // concatenation of a large open group would copy its rows quadratically.
    if pending.num_rows() > incoming.num_rows()
        || previous.times.value(0) != last.0
        || last >= next.at(0)
        || last.0 >= next.times.value(incoming.num_rows() - 1)
    {
        return None;
    }
    concat_batches(incoming.schema_ref(), [pending, incoming]).ok()
}

pub(super) fn closed_rows(
    input: &RecordBatch,
    compiled: &CompiledCrossSectionSpec,
    watermark: EventTime,
) -> usize {
    times(input, compiled)
        .values()
        .partition_point(|&time| time <= watermark.as_micros())
}

pub(super) fn compact_last_group(
    input: &RecordBatch,
    first_open: usize,
    compiled: &CompiledCrossSectionSpec,
    node_id: &str,
) -> Result<Option<RecordBatch>> {
    let times = times(input, compiled).values();
    if times[first_open] != times[times.len() - 1] {
        return Ok(None);
    }
    // Take releases the closed rows' backing buffers from retained state.
    let indices = UInt64Array::from_iter_values(first_open as u64..times.len() as u64);
    take_record_batch(input, &indices)
        .map(Some)
        .map_err(|error| operator_error(node_id, &error.to_string()))
}

pub(super) fn build_record(
    input: &RecordBatch,
    compiled: &CompiledCrossSectionSpec,
    output_schema: &SchemaRef,
    node_id: &str,
) -> Result<RecordBatch> {
    let times = times(input, compiled).values();
    let derived = if let [output] = compiled.outputs.as_slice()
        && let CompiledEvaluation::Mean { min_samples } = output.evaluation
    {
        vec![mean_column(input, times, output.input_index, min_samples)]
    } else {
        grouped_columns(input, times, compiled)
    };
    let columns = input.columns().iter().cloned().chain(derived).collect();
    RecordBatch::try_new(Arc::clone(output_schema), columns)
        .map_err(|error| operator_error(node_id, &error.to_string()))
}

fn groups(times: &[i64]) -> impl Iterator<Item = Range<usize>> + '_ {
    times
        .chunk_by(|left, right| left == right)
        .scan(0, |start, group| {
            let end = *start + group.len();
            let range = *start..end;
            *start = end;
            Some(range)
        })
}

fn float_column(input: &RecordBatch, index: usize) -> &Float64Array {
    input
        .column(index)
        .as_any()
        .downcast_ref()
        .expect("columnar cross-section operands are Float64")
}

fn mean_column(input: &RecordBatch, times: &[i64], index: usize, min_samples: u64) -> ArrayRef {
    let values = float_column(input, index);
    let mut means = Float64Builder::with_capacity(input.num_rows());
    for group in groups(times) {
        let slice = values.slice(group.start, group.len());
        let mut accumulator = StatisticAccumulator::default();
        for value in slice.iter().flatten().filter(|value| !value.is_nan()) {
            accumulator.add(value);
        }
        if accumulator.count >= min_samples {
            means.append_value_n(accumulator.classified_mean(), group.len());
        } else {
            means.append_nulls(group.len());
        }
    }
    Arc::new(means.finish())
}

fn grouped_columns(
    input: &RecordBatch,
    times: &[i64],
    compiled: &CompiledCrossSectionSpec,
) -> Vec<ArrayRef> {
    let mut builders = compiled
        .outputs
        .iter()
        .map(|output| OutputBuilder::new(&output.data_type, input.num_rows()))
        .collect::<Vec<_>>();
    for group in groups(times) {
        let operand = |index| float_column(input, index).slice(group.start, group.len());
        let columns = compute_group_outputs(
            compiled,
            |index, ordered| {
                let values = operand(index).iter().map(ScalarValue::Float64).collect();
                PreparedCrossSectionColumn::from_values(values, ordered)
            },
            |dependent, independent, min_samples| {
                let x = operand(independent);
                let y = operand(dependent);
                let pairs = x
                    .iter()
                    .zip(&y)
                    .map(|(x, y)| x.zip(y).filter(|(x, y)| !x.is_nan() && !y.is_nan()))
                    .collect();
                regression_residual_pairs(pairs, min_samples)
            },
        );
        for (builder, column) in builders.iter_mut().zip(columns) {
            builder.append(column);
        }
    }
    builders.into_iter().map(OutputBuilder::finish).collect()
}

enum OutputBuilder {
    Float(Float64Builder),
    Boolean(BooleanBuilder),
}

impl OutputBuilder {
    fn new(data_type: &DataType, capacity: usize) -> Self {
        match data_type {
            DataType::Float64 => Self::Float(Float64Builder::with_capacity(capacity)),
            DataType::Boolean => Self::Boolean(BooleanBuilder::with_capacity(capacity)),
            _ => unreachable!("columnar cross-section outputs are Float64 or Boolean"),
        }
    }

    fn append(&mut self, values: Vec<ScalarValue>) {
        match self {
            Self::Float(builder) => builder.extend(values.into_iter().map(|value| match value {
                ScalarValue::Float64(value) => value,
                _ => unreachable!("Float64 cross-section output has a different scalar type"),
            })),
            Self::Boolean(builder) => builder.extend(values.into_iter().map(|value| match value {
                ScalarValue::Boolean(value) => value,
                _ => unreachable!("Boolean cross-section output has a different scalar type"),
            })),
        }
    }

    fn finish(self) -> ArrayRef {
        match self {
            Self::Float(mut builder) => Arc::new(builder.finish()),
            Self::Boolean(mut builder) => Arc::new(builder.finish()),
        }
    }
}
