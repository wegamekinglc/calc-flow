use std::sync::Arc;

use ahash::RandomState;
use datafusion::{
    arrow::{
        array::{
            Array, ArrayRef, Float64Array, StringArray, UInt64Array,
            cast::AsArray,
            types::{
                TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
                TimestampSecondType,
            },
        },
        compute::{concat_batches, take},
        datatypes::{DataType, SchemaRef, TimeUnit},
        error::ArrowError,
        record_batch::RecordBatch,
    },
    common::{DataFusionError, Result as DataFusionResult},
};

use super::{AsofSpec, BandSpec, JoinSpec, NativeInput};

pub(super) fn execute(spec: &JoinSpec, schema: SchemaRef) -> DataFusionResult<RecordBatch> {
    match spec {
        JoinSpec::Asof(spec) => asof(spec, schema),
        JoinSpec::Band(spec) => band(spec, schema),
    }
}

fn arrow(error: ArrowError) -> DataFusionError {
    DataFusionError::ArrowError(Box::new(error), None)
}

fn concat(input: &NativeInput) -> DataFusionResult<RecordBatch> {
    concat_batches(&input.schema, input.batches.iter()).map_err(arrow)
}

fn timestamps(array: &ArrayRef) -> DataFusionResult<Vec<Option<i64>>> {
    macro_rules! values {
        ($type:ty) => {
            Ok(array.as_primitive::<$type>().iter().collect())
        };
    }
    match array.data_type() {
        DataType::Timestamp(TimeUnit::Second, _) => values!(TimestampSecondType),
        DataType::Timestamp(TimeUnit::Millisecond, _) => values!(TimestampMillisecondType),
        DataType::Timestamp(TimeUnit::Microsecond, _) => values!(TimestampMicrosecondType),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => values!(TimestampNanosecondType),
        other => Err(DataFusionError::Internal(format!(
            "native join expected a timestamp column, got {other}"
        ))),
    }
}

fn asof(spec: &AsofSpec, schema: SchemaRef) -> DataFusionResult<RecordBatch> {
    let left = concat(&spec.left)?;
    let right = concat(&spec.right)?;
    let right_keys = right
        .column(spec.right_key)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| DataFusionError::Internal("native ASOF key must be Utf8".into()))?;
    let right_times = timestamps(right.column(spec.right_time))?;
    let right_sequences = right
        .column(spec.right_sequence)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| DataFusionError::Internal("native ASOF sequence must be UInt64".into()))?;
    let mut latest = std::collections::HashMap::with_hasher(RandomState::new());
    for (row, time) in right_times.iter().copied().enumerate() {
        if right_keys.is_null(row) || time.is_none() {
            continue;
        }
        let sequence = (!right_sequences.is_null(row)).then(|| right_sequences.value(row));
        let time = time.expect("checked non-null timestamp");
        let by_time = latest
            .entry(right_keys.value(row))
            .or_insert_with(|| std::collections::HashMap::with_hasher(RandomState::new()));
        let replace = by_time
            .get(&time)
            .is_none_or(|(current, _)| match (sequence, *current) {
                (Some(next), Some(current)) => next > current,
                (None, Some(_)) => true,
                _ => false,
            });
        if replace {
            by_time.insert(time, (sequence, row));
        }
    }

    let left_keys = left
        .column(spec.left_key)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| DataFusionError::Internal("native ASOF key must be Utf8".into()))?;
    let left_times = timestamps(left.column(spec.left_time))?;
    let right_indices = (0..left.num_rows())
        .map(|row| {
            if left_keys.is_null(row) {
                return None;
            }
            let time = left_times[row]?;
            latest
                .get(left_keys.value(row))
                .and_then(|by_time| by_time.get(&time))
                .and_then(|(_, right_row)| u64::try_from(*right_row).ok())
        })
        .collect::<Vec<_>>();
    let left_indices = UInt64Array::from_iter_values(0..left.num_rows() as u64);
    let right_indices = UInt64Array::from_iter(right_indices);
    let columns = vec![
        take(left.column(spec.left_output), &left_indices, None).map_err(arrow)?,
        take(right.column(spec.right_output), &right_indices, None).map_err(arrow)?,
    ];
    RecordBatch::try_new(schema, columns).map_err(arrow)
}

#[derive(Clone, Copy, Debug)]
struct BandEntry {
    time: i64,
    sequence: Option<u64>,
    price: Option<f64>,
    row: usize,
}

fn band(spec: &BandSpec, schema: SchemaRef) -> DataFusionResult<RecordBatch> {
    let left = concat(&spec.left)?;
    let right = concat(&spec.right)?;
    let right_keys = right
        .column(spec.right_key)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| DataFusionError::Internal("native band key must be Utf8".into()))?;
    let right_times = timestamps(right.column(spec.right_time))?;
    let right_sequences = right
        .column(spec.right_sequence)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| DataFusionError::Internal("native band sequence must be UInt64".into()))?;
    let right_prices = right
        .column(spec.right_price)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| DataFusionError::Internal("native band price must be Float64".into()))?;
    let mut by_key = std::collections::HashMap::with_hasher(RandomState::new());
    for (row, time) in right_times.iter().copied().enumerate() {
        if right_keys.is_null(row) || time.is_none() {
            continue;
        }
        let entry = BandEntry {
            time: time.expect("checked non-null timestamp"),
            sequence: (!right_sequences.is_null(row)).then(|| right_sequences.value(row)),
            price: (!right_prices.is_null(row)).then(|| right_prices.value(row)),
            row,
        };
        by_key
            .entry(right_keys.value(row))
            .or_insert_with(Vec::new)
            .push(entry);
    }
    for entries in by_key.values_mut() {
        entries.sort_unstable_by_key(|entry| (entry.time, entry.row));
    }

    let left_keys = left
        .column(spec.left_key)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| DataFusionError::Internal("native band key must be Utf8".into()))?;
    let left_times = timestamps(left.column(spec.left_time))?;
    let left_sequences = left
        .column(spec.left_sequence)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| DataFusionError::Internal("native band sequence must be UInt64".into()))?;
    let left_prices = left
        .column(spec.left_price)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| DataFusionError::Internal("native band price must be Float64".into()))?;
    let capacity = left
        .num_rows()
        .saturating_mul(spec.offsets.len())
        .min(16_000_000);
    let mut sequences = Vec::with_capacity(capacity);
    let mut right_sequences_output = Vec::with_capacity(capacity);
    let mut values = Vec::with_capacity(capacity);
    for (row, time) in left_times.iter().copied().enumerate() {
        if left_keys.is_null(row) || time.is_none() {
            continue;
        }
        let Some(entries) = by_key.get(left_keys.value(row)) else {
            continue;
        };
        let left_sequence = (!left_sequences.is_null(row)).then(|| left_sequences.value(row));
        let left_price = (!left_prices.is_null(row)).then(|| left_prices.value(row));
        for offset in spec.offsets.iter() {
            let target = time
                .expect("checked non-null timestamp")
                .checked_add(*offset)
                .ok_or_else(|| DataFusionError::Execution("timestamp out of range".into()))?;
            let first = entries.partition_point(|entry| entry.time < target);
            for entry in &entries[first..] {
                if entry.time != target {
                    break;
                }
                sequences.push(left_sequence);
                right_sequences_output.push(entry.sequence);
                values.push(match (left_price, entry.price) {
                    (Some(left), Some(right)) => Some(left * right),
                    _ => None,
                });
            }
        }
    }
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt64Array::from_iter(sequences)),
        Arc::new(UInt64Array::from_iter(right_sequences_output)),
        Arc::new(Float64Array::from_iter(values)),
    ];
    RecordBatch::try_new(schema, columns).map_err(arrow)
}
