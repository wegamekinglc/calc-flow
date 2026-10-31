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
    physical_plan::{
        SendableRecordBatchStream, metrics::BaselineMetrics, stream::RecordBatchStreamAdapter,
    },
};
use futures::stream;

use super::{AsofSpec, BandSpec, JoinSpec, NativeInput};

pub(super) fn execute(
    spec: &JoinSpec,
    schema: SchemaRef,
    batch_size: usize,
    metrics: BaselineMetrics,
    partition: usize,
    partitions: usize,
) -> SendableRecordBatchStream {
    let state = StreamState::New(Box::new(NewState {
        spec: spec.clone(),
        schema: Arc::clone(&schema),
        batch_size,
        partition,
        partitions,
    }));
    let metrics = Arc::new(metrics);
    let output = stream::try_unfold(state, move |state| {
        let metrics = Arc::clone(&metrics);
        async move {
            let timer = metrics.elapsed_compute().timer();
            let (state, batch) = match state {
                StreamState::New(state) => {
                    let NewState {
                        spec,
                        schema,
                        batch_size,
                        partition,
                        partitions,
                    } = *state;
                    let mut state = State::new(&spec, schema, batch_size, partition, partitions)?;
                    let batch = state.next_batch()?;
                    (state, batch)
                }
                StreamState::Running(mut state) => {
                    let batch = state.next_batch()?;
                    (state, batch)
                }
            };
            if let Some(batch) = &batch {
                metrics.record_output(batch.num_rows());
            }
            timer.done();
            Ok(batch.map(|batch| (batch, StreamState::Running(state))))
        }
    });
    Box::pin(RecordBatchStreamAdapter::new(schema, output))
}

struct NewState {
    spec: JoinSpec,
    schema: SchemaRef,
    batch_size: usize,
    partition: usize,
    partitions: usize,
}

enum StreamState {
    New(Box<NewState>),
    Running(State),
}

enum State {
    Asof(Box<AsofState>),
    Band(Box<BandState>),
}

impl State {
    fn new(
        spec: &JoinSpec,
        schema: SchemaRef,
        batch_size: usize,
        partition: usize,
        partitions: usize,
    ) -> DataFusionResult<Self> {
        let batch_size = batch_size.clamp(1, 65_536);
        Ok(match spec {
            JoinSpec::Asof(spec) => Self::Asof(Box::new(AsofState::new(spec, schema, batch_size)?)),
            JoinSpec::Band(spec) => Self::Band(Box::new(BandState::new(
                spec, schema, batch_size, partition, partitions,
            )?)),
        })
    }

    fn next_batch(&mut self) -> DataFusionResult<Option<RecordBatch>> {
        match self {
            Self::Asof(state) => state.next_batch(),
            Self::Band(state) => state.next_batch(),
        }
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

#[cfg(test)]
thread_local! {
    static ASOF_OUTPUT_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn take_asof_output_rows() -> usize {
    ASOF_OUTPUT_ROWS.with(|count| count.replace(0))
}

type AsofEntries = std::collections::HashMap<i64, (Option<u64>, usize), RandomState>;
type AsofMap = std::collections::HashMap<String, AsofEntries, RandomState>;

struct AsofState {
    schema: SchemaRef,
    left: RecordBatch,
    right: RecordBatch,
    left_keys: StringArray,
    left_times: Vec<Option<i64>>,
    latest: AsofMap,
    left_output: usize,
    right_output: usize,
    position: usize,
    batch_size: usize,
}

impl AsofState {
    fn new(spec: &AsofSpec, schema: SchemaRef, batch_size: usize) -> DataFusionResult<Self> {
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
            .ok_or_else(|| {
                DataFusionError::Internal("native ASOF sequence must be UInt64".into())
            })?;
        let mut latest: AsofMap = std::collections::HashMap::with_hasher(RandomState::new());
        for (row, time) in right_times.iter().copied().enumerate() {
            if right_keys.is_null(row) || time.is_none() {
                continue;
            }
            let sequence = (!right_sequences.is_null(row)).then(|| right_sequences.value(row));
            let time = time.expect("checked non-null timestamp");
            let key = right_keys.value(row);
            let by_time = if let Some(by_time) = latest.get_mut(key) {
                by_time
            } else {
                latest
                    .entry(key.to_owned())
                    .or_insert_with(|| std::collections::HashMap::with_hasher(RandomState::new()))
            };
            let replace =
                by_time
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
            .ok_or_else(|| DataFusionError::Internal("native ASOF key must be Utf8".into()))?
            .clone();
        let left_times = timestamps(left.column(spec.left_time))?;
        Ok(Self {
            schema,
            left,
            right,
            left_keys,
            left_times,
            latest,
            left_output: spec.left_output,
            right_output: spec.right_output,
            position: 0,
            batch_size,
        })
    }

    fn next_batch(&mut self) -> DataFusionResult<Option<RecordBatch>> {
        if self.position == self.left.num_rows() {
            return Ok(None);
        }
        let end = (self.position + self.batch_size).min(self.left.num_rows());
        let right_indices = (self.position..end)
            .map(|row| {
                if self.left_keys.is_null(row) {
                    return None;
                }
                let time = self.left_times[row]?;
                self.latest
                    .get(self.left_keys.value(row))
                    .and_then(|by_time| by_time.get(&time))
                    .and_then(|(_, right_row)| u64::try_from(*right_row).ok())
            })
            .collect::<Vec<_>>();
        #[cfg(test)]
        ASOF_OUTPUT_ROWS.with(|count| count.set(count.get() + end - self.position));
        let left_indices = UInt64Array::from_iter_values(self.position as u64..end as u64);
        let right_indices = UInt64Array::from_iter(right_indices);
        let columns = vec![
            take(self.left.column(self.left_output), &left_indices, None).map_err(arrow)?,
            take(self.right.column(self.right_output), &right_indices, None).map_err(arrow)?,
        ];
        let output = RecordBatch::try_new(Arc::clone(&self.schema), columns).map_err(arrow)?;
        self.position = end;
        Ok(Some(output))
    }
}

#[derive(Clone, Copy, Debug)]
struct BandEntry {
    time: i64,
    sequence: Option<u64>,
    price: Option<f64>,
}

#[derive(Debug)]
struct KeyBand {
    entries: Vec<BandEntry>,
    dense_step: Option<i64>,
    offset_indices: Vec<Option<i64>>,
}

fn band_step(entries: &[BandEntry]) -> Option<i64> {
    let step = entries.get(1)?.time.checked_sub(entries.first()?.time)?;
    (step > 0
        && entries
            .windows(2)
            .all(|pair| pair[1].time.checked_sub(pair[0].time) == Some(step)))
    .then_some(step)
}

#[derive(Debug)]
struct BandMap {
    indices: std::collections::HashMap<String, usize, RandomState>,
    entries: Vec<KeyBand>,
}

fn build_bands(right: &RecordBatch, spec: &BandSpec) -> DataFusionResult<BandMap> {
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
    let mut by_key: std::collections::HashMap<String, Vec<BandEntry>, RandomState> =
        std::collections::HashMap::with_hasher(RandomState::new());
    for (row, time) in right_times.iter().copied().enumerate() {
        if right_keys.is_null(row) || time.is_none() {
            continue;
        }
        let entry = BandEntry {
            time: time.expect("checked non-null timestamp"),
            sequence: (!right_sequences.is_null(row)).then(|| right_sequences.value(row)),
            price: (!right_prices.is_null(row)).then(|| right_prices.value(row)),
        };
        let key = right_keys.value(row);
        if let Some(entries) = by_key.get_mut(key) {
            entries.push(entry);
        } else {
            by_key.insert(key.to_owned(), vec![entry]);
        }
    }
    for entries in by_key.values_mut() {
        if !entries.is_sorted_by_key(|entry| entry.time) {
            entries.sort_by_key(|entry| entry.time);
        }
    }
    let mut indices =
        std::collections::HashMap::with_capacity_and_hasher(by_key.len(), RandomState::new());
    let entries = by_key
        .into_iter()
        .enumerate()
        .map(|(index, (key, entries))| {
            indices.insert(key, index);
            let dense_step = band_step(&entries);
            let offset_indices = dense_step.map_or_else(Vec::new, |step| {
                spec.offsets
                    .iter()
                    .map(|offset| (offset % step == 0).then(|| offset / step))
                    .collect()
            });
            KeyBand {
                entries,
                dense_step,
                offset_indices,
            }
        })
        .collect();
    Ok(BandMap { indices, entries })
}

fn band_first(
    band: &KeyBand,
    cursors: &mut [usize],
    offset_index: usize,
    target: i64,
    next_match: Option<usize>,
    dense_base: Option<i128>,
) -> usize {
    if let Some(first) = next_match {
        return first;
    }
    if let Some(step) = band.dense_step {
        let index = if let Some(offset) = band.offset_indices[offset_index] {
            dense_base.map(|base| base + i128::from(offset))
        } else {
            let step = i128::from(step);
            let delta = i128::from(target) - i128::from(band.entries[0].time);
            (delta >= 0 && delta % step == 0).then(|| delta / step)
        };
        return index
            .and_then(|index| usize::try_from(index).ok())
            .filter(|index| {
                band.entries
                    .get(*index)
                    .is_some_and(|entry| entry.time == target)
            })
            .unwrap_or(band.entries.len());
    }
    let cursor = &mut cursors[offset_index];
    if band
        .entries
        .get(*cursor)
        .is_none_or(|entry| target < entry.time)
    {
        *cursor = band.entries.partition_point(|entry| entry.time < target);
    } else {
        while band
            .entries
            .get(*cursor)
            .is_some_and(|entry| entry.time < target)
        {
            *cursor += 1;
        }
    }
    *cursor
}

fn band_target(time: i64, offset: i64, bounds: (i64, i64)) -> DataFusionResult<i64> {
    if !(bounds.0..=bounds.1).contains(&time) {
        return Err(DataFusionError::Execution("Timestamp out of range".into()));
    }
    time.checked_add(offset)
        .filter(|target| (bounds.0..=bounds.1).contains(target))
        .ok_or_else(|| DataFusionError::Execution("Timestamp out of range".into()))
}

#[derive(Debug, Default)]
pub(super) struct BandShared {
    prepared: std::sync::OnceLock<Result<Arc<BandPrepared>, String>>,
}

#[derive(Debug)]
pub(super) struct BandPrepared {
    bands: Arc<BandMap>,
    left_keys: StringArray,
    left_times: Arc<[Option<i64>]>,
    left_sequences: UInt64Array,
    left_prices: Float64Array,
    time_bounds: (i64, i64),
}

impl BandPrepared {
    fn new(spec: &BandSpec) -> DataFusionResult<Self> {
        let left = concat(&spec.left)?;
        let right = concat(&spec.right)?;
        let bands = Arc::new(build_bands(&right, spec)?);
        let left_keys = left
            .column(spec.left_key)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| DataFusionError::Internal("native band key must be Utf8".into()))?
            .clone();
        let left_times = timestamps(left.column(spec.left_time))?.into();
        let left_sequences = left
            .column(spec.left_sequence)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| DataFusionError::Internal("native band sequence must be UInt64".into()))?
            .clone();
        let left_prices = left
            .column(spec.left_price)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| DataFusionError::Internal("native band price must be Float64".into()))?
            .clone();
        let DataType::Timestamp(unit, timezone) = left.column(spec.left_time).data_type() else {
            return Err(DataFusionError::Internal(
                "native band time must be a timestamp".into(),
            ));
        };
        if let Some(zone) = timezone {
            zone.parse::<datafusion::arrow::array::timezone::Tz>()
                .map_err(|error| DataFusionError::Execution(error.to_string()))?;
        }
        let time_bounds = crate::datafusion_interval::representable_bounds(*unit);
        Ok(Self {
            bands,
            left_keys,
            left_times,
            left_sequences,
            left_prices,
            time_bounds,
        })
    }
}

struct BandState {
    schema: SchemaRef,
    bands: Arc<BandMap>,
    cursors: Vec<Vec<usize>>,
    left_keys: StringArray,
    left_times: Arc<[Option<i64>]>,
    left_sequences: UInt64Array,
    left_prices: Float64Array,
    offsets: Arc<[i64]>,
    time_bounds: (i64, i64),
    batch_size: usize,
    partition_step: usize,
    row: usize,
    offset_index: usize,
    next_match: Option<usize>,
    band_row: usize,
    current_band: Option<usize>,
    dense_row: usize,
    dense_base: Option<i128>,
}

impl BandState {
    fn new(
        spec: &BandSpec,
        schema: SchemaRef,
        batch_size: usize,
        partition: usize,
        partitions: usize,
    ) -> DataFusionResult<Self> {
        let prepared = spec
            .shared
            .prepared
            .get_or_init(|| {
                BandPrepared::new(spec)
                    .map(Arc::new)
                    .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|error| DataFusionError::Execution(error.clone()))?;
        let cursors = prepared
            .bands
            .entries
            .iter()
            .map(|_| vec![0; spec.offsets.len()])
            .collect();
        Ok(Self {
            schema,
            bands: Arc::clone(&prepared.bands),
            cursors,
            left_keys: prepared.left_keys.clone(),
            left_times: Arc::clone(&prepared.left_times),
            left_sequences: prepared.left_sequences.clone(),
            left_prices: prepared.left_prices.clone(),
            offsets: Arc::clone(&spec.offsets),
            time_bounds: prepared.time_bounds,
            batch_size,
            partition_step: partitions.max(1),
            row: partition,
            offset_index: 0,
            next_match: None,
            band_row: usize::MAX,
            current_band: None,
            dense_row: usize::MAX,
            dense_base: None,
        })
    }

    fn next_batch(&mut self) -> DataFusionResult<Option<RecordBatch>> {
        let mut sequences = Vec::with_capacity(self.batch_size);
        let mut right_sequences = Vec::with_capacity(self.batch_size);
        let mut values = Vec::with_capacity(self.batch_size);
        while sequences.len() < self.batch_size && self.row < self.left_times.len() {
            if self.offset_index == self.offsets.len() {
                self.row += self.partition_step;
                self.offset_index = 0;
                self.next_match = None;
                continue;
            }
            if self.left_times[self.row].is_none() {
                self.row += self.partition_step;
                self.offset_index = 0;
                self.next_match = None;
                continue;
            }
            let time = self.left_times[self.row].expect("checked non-null timestamp");
            let target = band_target(time, self.offsets[self.offset_index], self.time_bounds)?;
            if self.left_keys.is_null(self.row) {
                self.offset_index += 1;
                self.next_match = None;
                continue;
            }
            let band_index = if self.band_row == self.row {
                self.current_band
            } else {
                let index = self
                    .bands
                    .indices
                    .get(self.left_keys.value(self.row))
                    .copied();
                self.band_row = self.row;
                self.current_band = index;
                index
            };
            let Some(band_index) = band_index else {
                self.offset_index += 1;
                self.next_match = None;
                continue;
            };
            let band = &self.bands.entries[band_index];
            if self.dense_row != self.row {
                self.dense_row = self.row;
                self.dense_base = band.dense_step.and_then(|step| {
                    let step = i128::from(step);
                    let delta = i128::from(time) - i128::from(band.entries[0].time);
                    (delta % step == 0).then(|| delta / step)
                });
            }
            let first = band_first(
                band,
                &mut self.cursors[band_index],
                self.offset_index,
                target,
                self.next_match,
                self.dense_base,
            );
            let mut next = first;
            while sequences.len() < self.batch_size
                && band
                    .entries
                    .get(next)
                    .is_some_and(|entry| entry.time == target)
            {
                let entry = band.entries[next];
                sequences.push(
                    (!self.left_sequences.is_null(self.row))
                        .then(|| self.left_sequences.value(self.row)),
                );
                right_sequences.push(entry.sequence);
                let left_price =
                    (!self.left_prices.is_null(self.row)).then(|| self.left_prices.value(self.row));
                values.push(match (left_price, entry.price) {
                    (Some(left), Some(right)) => Some(left * right),
                    _ => None,
                });
                next += 1;
            }
            self.next_match = Some(next);
            if band
                .entries
                .get(next)
                .is_none_or(|entry| entry.time != target)
            {
                self.offset_index += 1;
                self.next_match = None;
            }
        }
        if sequences.is_empty() {
            return Ok(None);
        }
        let columns: Vec<ArrayRef> = vec![
            Arc::new(UInt64Array::from_iter(sequences)),
            Arc::new(UInt64Array::from_iter(right_sequences)),
            Arc::new(Float64Array::from_iter(values)),
        ];
        RecordBatch::try_new(Arc::clone(&self.schema), columns)
            .map(Some)
            .map_err(arrow)
    }
}
