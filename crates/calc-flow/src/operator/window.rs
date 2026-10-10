use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    io::Cursor,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use datafusion::arrow::{
    array::{
        Array, ArrayRef, BooleanArray, Date32Array, Date64Array, FixedSizeBinaryArray,
        Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
        LargeBinaryArray, LargeStringArray, StringArray, TimestampMicrosecondArray,
        TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt8Array,
        UInt16Array, UInt32Array, UInt64Array,
    },
    buffer::NullBuffer,
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    ipc::{
        convert::IpcSchemaEncoder,
        reader::FileReader,
        writer::{DictionaryTracker, FileWriter},
    },
    record_batch::RecordBatch,
};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    Batch, BatchKind, CalcFlowError, EventTime, JsonMap, Port, Result, StateHandle,
    StreamCollector, StreamOperator, StreamOperatorContext, canonical_json,
    state::{SegmentDescriptor, SegmentKind, StateInventory, StateOperation, fold_state_segments},
};

use super::checkpoint::{checkpoint_mismatch, compile_error, internal_error, state_format};
use super::{
    LateMetricDelta, OperatorMetadata, StateBudget, accumulate_late_metrics, validate_operator_name,
};

/// Maximum number of concrete hopping-window assignments for one input row.
pub const MAX_WINDOW_OVERLAP: u64 = 1_024;

pub(crate) const WINDOW_STATE_LAYOUT_VERSION: u32 = 1;
const WINDOW_SEGMENT_LAYOUT_VERSION: u32 = 2;
const MAX_GROUP_KEY_BYTES: usize = 65_536;
const MAX_WINDOW_DELTA_SEGMENTS: usize = 32;

#[path = "window/kernels.rs"]
mod kernels;
mod keys;

#[cfg(test)]
#[path = "window/group_tests.rs"]
mod group_tests;

#[cfg(test)]
#[path = "window/kernel_tests.rs"]
mod kernel_tests;

#[cfg(test)]
#[path = "window/key_tests.rs"]
mod key_tests;

/// Aggregate function supported by the first built-in window operator.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AggregateFunction {
    /// Count non-null input values.
    Count,
    /// Sum numeric input values.
    Sum,
    /// Select the minimum supported scalar.
    Min,
    /// Select the maximum supported scalar.
    Max,
    /// Compute the arithmetic mean of numeric input values.
    Avg,
}

/// One declared window aggregate and its output column name.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AggregateSpec {
    /// Aggregate function.
    pub function: AggregateFunction,
    /// Input column name.
    pub column: String,
    /// Output column name.
    pub output: String,
}

/// Fixed UTC window geometry represented in exact microseconds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum WindowGeometry {
    /// Non-overlapping windows of one fixed size.
    Tumbling {
        /// Window size in exact microseconds.
        size_micros: u64,
    },
    /// Fixed-size windows beginning at every slide coordinate.
    Hopping {
        /// Window size in exact microseconds.
        size_micros: u64,
        /// Window slide in exact microseconds.
        slide_micros: u64,
    },
}

/// Data-only declaration of one event-time window aggregation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WindowSpec {
    /// Input timestamp column used for window assignment.
    pub event_time_column: String,
    /// Group columns in semantic declaration order.
    pub group_by: Vec<String>,
    /// Fixed tumbling or hopping geometry.
    pub geometry: WindowGeometry,
    /// Aggregates in semantic declaration order.
    pub aggregates: Vec<AggregateSpec>,
}

impl WindowSpec {
    /// Creates a tumbling-window declaration.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the event-time column
    /// is empty or `size` is zero, non-integral in microseconds, or outside
    /// the serialized microsecond range.
    pub fn tumbling(event_time_column: &str, size: Duration) -> Result<Self> {
        let size_micros = exact_duration_micros(size, "window.geometry.size")?;
        let spec = Self {
            event_time_column: event_time_column.into(),
            group_by: Vec::new(),
            geometry: WindowGeometry::Tumbling { size_micros },
            aggregates: Vec::new(),
        };
        spec.validate_arguments()?;
        Ok(spec)
    }

    /// Creates a hopping-window declaration.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when either duration is
    /// invalid, the size is not an exact multiple of the slide, or one row
    /// would receive more than [`MAX_WINDOW_OVERLAP`] assignments.
    pub fn hopping(event_time_column: &str, size: Duration, slide: Duration) -> Result<Self> {
        let size_micros = exact_duration_micros(size, "window.geometry.size")?;
        let slide_micros = exact_duration_micros(slide, "window.geometry.slide")?;
        let spec = Self {
            event_time_column: event_time_column.into(),
            group_by: Vec::new(),
            geometry: WindowGeometry::Hopping {
                size_micros,
                slide_micros,
            },
            aggregates: Vec::new(),
        };
        spec.validate_arguments()?;
        Ok(spec)
    }

    /// Returns a declaration with the exact ordered grouping columns.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for an empty, duplicate, or
    /// reserved group name, or a collision with an aggregate output.
    pub fn group_by<I, S>(mut self, columns: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.group_by = columns.into_iter().map(Into::into).collect();
        self.validate_arguments()?;
        Ok(self)
    }

    /// Appends one aggregate in semantic declaration order.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for an empty input/output
    /// name or a duplicate, reserved, or group-column output name.
    pub fn aggregate(
        mut self,
        function: AggregateFunction,
        column: &str,
        output: &str,
    ) -> Result<Self> {
        self.aggregates.push(AggregateSpec {
            function,
            column: column.into(),
            output: output.into(),
        });
        self.validate_arguments()?;
        Ok(self)
    }

    fn validate_arguments(&self) -> Result<()> {
        if self.event_time_column.is_empty() {
            return Err(invalid_argument(
                "window.event_time_column",
                "must not be empty",
            ));
        }
        validate_geometry(self.geometry)?;

        let group_names = validate_group_names(&self.group_by)?;
        validate_aggregate_names(&self.aggregates, &group_names)
    }
}

fn validate_group_names(columns: &[String]) -> Result<BTreeSet<&String>> {
    let mut names = BTreeSet::new();
    for (index, column) in columns.iter().enumerate() {
        let field = format!("window.group_by[{index}]");
        if column.is_empty() {
            return Err(invalid_argument(&field, "must not be empty"));
        }
        if is_reserved_output(column) {
            return Err(invalid_argument(
                &field,
                "collides with a reserved window output name",
            ));
        }
        if !names.insert(column) {
            return Err(invalid_argument(
                &field,
                "duplicates an earlier group column",
            ));
        }
    }
    Ok(names)
}

fn validate_aggregate_names(
    aggregates: &[AggregateSpec],
    group_names: &BTreeSet<&String>,
) -> Result<()> {
    let mut outputs = BTreeSet::new();
    for (index, aggregate) in aggregates.iter().enumerate() {
        if aggregate.column.is_empty() {
            return Err(invalid_argument(
                &format!("window.aggregates[{index}].column"),
                "must not be empty",
            ));
        }
        let output_field = format!("window.aggregates[{index}].output");
        if aggregate.output.is_empty() {
            return Err(invalid_argument(&output_field, "must not be empty"));
        }
        if is_reserved_output(&aggregate.output) || group_names.contains(&aggregate.output) {
            return Err(invalid_argument(
                &output_field,
                "collides with a reserved or group-column output name",
            ));
        }
        if !outputs.insert(&aggregate.output) {
            return Err(invalid_argument(
                &output_field,
                "duplicates an earlier aggregate output",
            ));
        }
    }
    Ok(())
}

/// Built-in stream-only event-time window aggregation operator.
pub struct WindowAggregateOperator {
    name: String,
    spec: WindowSpec,
    input_ports: [Port; 1],
    output_ports: [Port; 1],
    compiled: CompiledWindowSpec,
    state: WindowState,
    state_budget: StateBudget,
}

#[derive(Clone)]
struct CompiledWindowSpec {
    event_time_index: usize,
    group_columns: Vec<CompiledGroupColumn>,
    aggregates: Vec<CompiledAggregate>,
    geometry: CompiledWindowGeometry,
    #[allow(
        dead_code,
        reason = "the M4 persistence work package records the compiled configuration hash"
    )]
    configuration_hash: String,
    state_schema_fingerprint: String,
}

#[derive(Clone)]
struct CompiledGroupColumn {
    index: usize,
    data_type: DataType,
}

#[derive(Clone)]
struct CompiledAggregate {
    input_index: usize,
    input_type: DataType,
    output_type: DataType,
}

#[derive(Clone, Copy)]
struct CompiledWindowGeometry {
    size_micros: u64,
    slide_micros: u64,
    overlap: u64,
}

#[derive(Default)]
struct WindowState {
    accumulators: BTreeMap<WindowKey, AccumulatorRow>,
    accumulator_bytes: u64,
    dirty: BTreeSet<WindowKey>,
    emitted_pending_snapshot: BTreeSet<WindowKey>,
    last_input_watermark: Option<EventTime>,
    next_output_sequence: u64,
    ended: bool,
    metrics: LateMetricDelta,
    prepared_segments: Vec<PreparedStateSegment>,
    retained_inventory: StateInventory,
    retained_segments: BTreeMap<String, crate::StateSegment>,
    replace_retained_on_checkpoint: bool,
    last_checkpoint_epoch: Option<crate::Epoch>,
    pipeline_fingerprint: Option<String>,
    operator_id: Option<String>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct WindowKey {
    start: EventTime,
    end: EventTime,
    stable_group_key: Arc<[u8]>,
}

#[derive(Clone)]
struct AccumulatorRow {
    group_values: Vec<Option<ScalarValue>>,
    aggregates: Vec<AccumulatorValue>,
}

#[derive(Clone, Debug)]
enum ScalarValue {
    Boolean(bool),
    Signed(i64),
    Unsigned(u64),
    Float32(u32),
    Float64(u64),
    String(String),
    Date32(i32),
    Date64(i64),
    Timestamp(i64),
}

#[derive(Clone)]
enum AccumulatorValue {
    Count(u64),
    SignedSum(Option<i128>),
    UnsignedSum(Option<u128>),
    FloatSum(Option<CompensatedSum>),
    Min(Option<ScalarValue>),
    Max(Option<ScalarValue>),
    SignedAverage { sum: i128, count: u64 },
    UnsignedAverage { sum: u128, count: u64 },
    FloatAverage { sum: CompensatedSum, count: u64 },
}

#[derive(Clone, Copy)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    const fn new(sum: f64) -> Self {
        Self {
            sum,
            correction: 0.0,
        }
    }

    fn add(&mut self, value: f64) {
        let next = canonicalize_float(self.sum + value);
        if self.sum.is_finite() && value.is_finite() && next.is_finite() {
            let correction = if self.sum.abs() >= value.abs() {
                (self.sum - next) + value
            } else {
                (value - next) + self.sum
            };
            self.correction = canonicalize_float(self.correction + correction);
        } else {
            self.correction = 0.0;
        }
        self.sum = next;
    }

    fn total(self) -> f64 {
        canonicalize_float(self.sum + self.correction)
    }
}

#[derive(Clone)]
struct StateOperationRow {
    key: WindowKey,
    entry: AccumulatorRow,
    tombstone: bool,
}

struct PreparedStateSegment {
    kind: SegmentKind,
    bytes: Vec<u8>,
}

struct PreparedSnapshotSegments {
    descriptors: Vec<SegmentDescriptor>,
    bytes: BTreeMap<String, crate::StateSegment>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WindowSnapshotMetadata {
    state_layout_version: u32,
    configuration_hash: String,
    state_schema_fingerprint: String,
    epoch: crate::Epoch,
    #[serde(deserialize_with = "deserialize_required_option")]
    pipeline_fingerprint: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    operator_id: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    last_input_watermark: Option<EventTime>,
    next_output_sequence: u64,
    ended: bool,
    metrics: LateMetricDelta,
    segment_inventory: Vec<SegmentDescriptor>,
}

struct InputBatchUpdate {
    accumulators: BTreeMap<WindowKey, AccumulatorRow>,
    usage: WindowStateUsage,
    metrics: LateMetricDelta,
}

#[derive(Clone, Copy)]
struct WindowStateUsage {
    rows: u64,
    bytes: u64,
}

/// One open window of one interned batch group.
#[derive(Clone, Copy)]
struct WindowSlot {
    start: EventTime,
    end: EventTime,
    group: usize,
}

#[derive(Clone, Copy)]
struct CachedWindowSlot {
    key: (i64, usize),
    index: usize,
}

/// Per-batch accumulator scratch: hashed while rows stream in, then sorted
/// once into the deterministic window-key order installs and encodes observe.
struct BatchScratch<'a> {
    group_ids: HashMap<Arc<[u8]>, usize>,
    string_group_ids: HashMap<Option<&'a str>, usize>,
    integer_group_ids: HashMap<keys::IntegerKey, usize>,
    group_keys: Vec<Arc<[u8]>>,
    // Keyed by (window start, group); the fixed geometry determines the end.
    slots: HashMap<(i64, usize), usize>,
    slot_cache: [Option<CachedWindowSlot>; 64],
    slot_cache_ways: usize,
    entries: Vec<(WindowKey, AccumulatorRow)>,
    // Reused by the general composite-key path, without owned row scalars.
    encoded_group: Vec<u8>,
    usage: WindowStateUsage,
    metrics: PreparedInputMetrics,
}

impl<'a> BatchScratch<'a> {
    fn new(usage: WindowStateUsage, overlap: u64) -> Self {
        let slot_cache_ways = match overlap {
            1 => 1,
            2 => 2,
            3 | 4 => 4,
            _ => 0,
        };
        Self {
            group_ids: HashMap::new(),
            string_group_ids: HashMap::new(),
            integer_group_ids: HashMap::new(),
            group_keys: Vec::new(),
            slots: HashMap::new(),
            slot_cache: [None; 64],
            slot_cache_ways,
            entries: Vec::new(),
            encoded_group: Vec::new(),
            usage,
            metrics: PreparedInputMetrics::default(),
        }
    }

    /// Borrow single string keys directly from Arrow. Canonical keys are
    /// created only for a distinct batch group.
    fn intern_group(
        &mut self,
        columns: &RecordColumns<'a>,
        row: usize,
        operator_id: &str,
        names: &[String],
    ) -> Result<usize> {
        let integer_key = keys::IntegerKey::read(columns, row);
        if let Some(key) = integer_key
            && let Some(&group) = self.integer_group_ids.get(&key)
        {
            return Ok(group);
        }
        let string_key = match columns.groups.as_slice() {
            [(column, _)] => column.borrowed_string(row),
            _ => BorrowedString::Other,
        };
        if let BorrowedString::Value(key) = string_key
            && let Some(&group) = self.string_group_ids.get(&key)
        {
            return Ok(group);
        }
        encode_group_key(columns, row, operator_id, names, &mut self.encoded_group)?;
        if integer_key.is_none()
            && matches!(string_key, BorrowedString::Other)
            && let Some(&group) = self.group_ids.get(self.encoded_group.as_slice())
        {
            return Ok(group);
        }
        let key = Arc::<[u8]>::from(self.encoded_group.as_slice());
        let group = self.group_keys.len();
        self.group_keys.push(Arc::clone(&key));
        if let Some(value) = integer_key {
            self.integer_group_ids.insert(value, group);
        } else if let BorrowedString::Value(value) = string_key {
            self.string_group_ids.insert(value, group);
        } else {
            self.group_ids.insert(key, group);
        }
        Ok(group)
    }

    fn into_update(self) -> InputBatchUpdate {
        InputBatchUpdate {
            accumulators: self.entries.into_iter().collect(),
            usage: self.usage,
            metrics: self.metrics.into_delta(),
        }
    }

    fn cached_slot(&self, key: (i64, usize)) -> Option<usize> {
        if self.slot_cache_ways == 0 {
            return None;
        }
        let position = key.1.wrapping_mul(self.slot_cache_ways) & (self.slot_cache.len() - 1);
        let cached = self.slot_cache[position]
            .filter(|entry| {
                #[cfg(test)]
                group_tests::CACHE_KEY_COMPARISONS.with(|calls| calls.set(calls.get() + 1));
                entry.key == key
            })
            .map(|entry| entry.index);
        if cached.is_some() || self.slot_cache_ways == 1 {
            return cached;
        }
        self.slot_cache[position + 1..position + self.slot_cache_ways]
            .iter()
            .flatten()
            .find(|entry| {
                #[cfg(test)]
                group_tests::CACHE_KEY_COMPARISONS.with(|calls| calls.set(calls.get() + 1));
                entry.key == key
            })
            .map(|entry| entry.index)
    }

    fn cache_slot(&mut self, key: (i64, usize), index: usize) {
        if self.slot_cache_ways == 0 {
            return;
        }
        let position = key.1.wrapping_mul(self.slot_cache_ways) & (self.slot_cache.len() - 1);
        let offset = if self.slot_cache_ways == 1 {
            0
        } else {
            self.slot_cache[position..position + self.slot_cache_ways]
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.map(|entry| entry.index))
                .map(|(offset, _)| offset)
                .unwrap_or_default()
        };
        self.slot_cache[position + offset] = Some(CachedWindowSlot { key, index });
    }
}

/// Input columns of one record, downcast once so per-row reads skip dynamic
/// type dispatch.
struct RecordColumns<'a> {
    event_time: EventTimeColumn<'a>,
    groups: Vec<(ScalarColumn<'a>, &'a DataType)>,
    aggregates: Vec<(ScalarColumn<'a>, kernels::UpdateKernel)>,
}

impl<'a> RecordColumns<'a> {
    fn new(
        record: &'a RecordBatch,
        spec: &WindowSpec,
        compiled: &'a CompiledWindowSpec,
        operator_id: &str,
    ) -> Result<Self> {
        let groups = compiled
            .group_columns
            .iter()
            .map(|column| {
                ScalarColumn::new(
                    record.column(column.index).as_ref(),
                    &column.data_type,
                    operator_id,
                )
                .map(|values| (values, &column.data_type))
            })
            .collect::<Result<_>>()?;
        let aggregates = spec
            .aggregates
            .iter()
            .zip(&compiled.aggregates)
            .map(|(aggregate, compiled)| {
                aggregate_column(record, aggregate.function, compiled, operator_id).map(|column| {
                    (
                        column,
                        kernels::select(aggregate.function, &compiled.input_type),
                    )
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            event_time: EventTimeColumn::new(record, compiled.event_time_index, operator_id)?,
            groups,
            aggregates,
        })
    }
}

/// `count` accepts any input type and only reads validity, so its column
/// stays opaque; every other aggregate reads typed scalars.
fn aggregate_column<'a>(
    record: &'a RecordBatch,
    function: AggregateFunction,
    aggregate: &CompiledAggregate,
    operator_id: &str,
) -> Result<ScalarColumn<'a>> {
    let array = record.column(aggregate.input_index).as_ref();
    if function == AggregateFunction::Count {
        return Ok(ScalarColumn::opaque(array));
    }
    ScalarColumn::new(array, &aggregate.input_type, operator_id)
}

// Stable logical prices for retained map entries and scalar slots; variable
// key and string lengths are charged separately from these fixed costs.
const WINDOW_ENTRY_BASE_BYTES: u64 = 128;
const WINDOW_GROUP_VALUE_BYTES: u64 = 64;
const WINDOW_AGGREGATE_BYTES: u64 = 64;

fn logical_length(length: usize) -> u64 {
    u64::try_from(length).unwrap_or(u64::MAX)
}

fn scalar_dynamic_bytes(value: &ScalarValue) -> u64 {
    match value {
        ScalarValue::String(value) => logical_length(value.len()),
        _ => 0,
    }
}

fn aggregate_dynamic_bytes(entry: &AccumulatorRow) -> u64 {
    entry
        .aggregates
        .iter()
        .filter_map(|aggregate| match aggregate {
            AccumulatorValue::Min(value) | AccumulatorValue::Max(value) => value.as_ref(),
            _ => None,
        })
        .map(scalar_dynamic_bytes)
        .fold(0_u64, u64::saturating_add)
}

fn window_entry_bytes(key_bytes: u64, entry: &AccumulatorRow) -> u64 {
    let group_bytes = logical_length(entry.group_values.len())
        .saturating_mul(WINDOW_GROUP_VALUE_BYTES)
        .saturating_add(
            entry
                .group_values
                .iter()
                .flatten()
                .map(scalar_dynamic_bytes)
                .fold(0_u64, u64::saturating_add),
        );
    let aggregate_bytes = logical_length(entry.aggregates.len())
        .saturating_mul(WINDOW_AGGREGATE_BYTES)
        .saturating_add(aggregate_dynamic_bytes(entry));
    WINDOW_ENTRY_BASE_BYTES
        .saturating_add(key_bytes)
        .saturating_add(group_bytes)
        .saturating_add(aggregate_bytes)
}

/// Charges one updated accumulator against the running usage: a new key adds
/// its whole entry, an existing key replaces its previous dynamic bytes.
fn charge_accumulator(
    usage: WindowStateUsage,
    previous_dynamic_bytes: Option<u64>,
    key_bytes: u64,
    accumulator: &AccumulatorRow,
) -> Result<WindowStateUsage> {
    let Some(previous_dynamic_bytes) = previous_dynamic_bytes else {
        return Ok(WindowStateUsage {
            rows: usage.rows.saturating_add(1),
            bytes: usage
                .bytes
                .saturating_add(window_entry_bytes(key_bytes, accumulator)),
        });
    };
    let bytes = usage
        .bytes
        .checked_sub(previous_dynamic_bytes)
        .ok_or_else(|| internal_error("window accumulator charge underflowed"))?
        .saturating_add(aggregate_dynamic_bytes(accumulator));
    Ok(WindowStateUsage {
        rows: usage.rows,
        bytes,
    })
}

fn window_state_usage(accumulators: &BTreeMap<WindowKey, AccumulatorRow>) -> WindowStateUsage {
    WindowStateUsage {
        rows: logical_length(accumulators.len()),
        bytes: accumulators
            .iter()
            .map(|(key, entry)| {
                window_entry_bytes(logical_length(key.stable_group_key.len()), entry)
            })
            .fold(0_u64, u64::saturating_add),
    }
}

#[derive(Default)]
struct PreparedInputMetrics {
    late_rows: u64,
    max_lateness_micros: Option<u64>,
    null_event_time_rows: u64,
}

impl PreparedInputMetrics {
    fn into_delta(self) -> LateMetricDelta {
        LateMetricDelta {
            late_rows: self.late_rows,
            affected_batches: u64::from(self.late_rows > 0),
            max_lateness_micros: self.max_lateness_micros,
            null_event_time_rows: self.null_event_time_rows,
            null_event_time_batches: u64::from(self.null_event_time_rows > 0),
        }
    }
}

impl fmt::Debug for WindowAggregateOperator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WindowAggregateOperator")
            .field("name", &self.name)
            .field("spec", &self.spec)
            .field("input_ports", &self.input_ports)
            .field("output_ports", &self.output_ports)
            .finish_non_exhaustive()
    }
}

impl WindowAggregateOperator {
    /// Compiles one window declaration against an exact Arrow input schema.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for invalid declaration
    /// fields and [`CalcFlowError::Compile`] for missing, ambiguous, or
    /// unsupported input columns and aggregate combinations.
    pub fn new(name: &str, input_schema: SchemaRef, spec: WindowSpec) -> Result<Self> {
        validate_operator_name(name)?;
        spec.validate_arguments()?;
        let configuration = configuration(&spec)?;
        let compiled = compile_spec(&input_schema, &spec, &configuration)?;
        let output_schema = output_schema(&input_schema, &spec, &compiled);
        Ok(Self {
            name: name.into(),
            spec,
            input_ports: [Port::with_schema_ref(
                "input",
                BatchKind::Table,
                true,
                Some(input_schema),
            )?],
            output_ports: [Port::with_schema_ref(
                "output",
                BatchKind::Table,
                true,
                Some(output_schema),
            )?],
            compiled,
            state: WindowState::default(),
            state_budget: StateBudget::default(),
        })
    }

    /// Sets the logical row and byte limit for retained window accumulators.
    ///
    /// The default is one million windows and 256 MiB of logical charge.
    /// Existing state must fit the replacement budget. Checkpoint segment
    /// copies and process RSS are outside this limit.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] if existing state exceeds
    /// the requested budget.
    ///
    /// # Examples
    ///
    /// ```
    /// use calc_flow::{StateBudget, WindowAggregateOperator};
    /// fn configure(operator: &mut WindowAggregateOperator) -> calc_flow::Result<()> {
    ///     operator.set_state_budget(StateBudget::new(10_000, 64 << 20)?)
    /// }
    /// ```
    pub fn set_state_budget(&mut self, budget: StateBudget) -> Result<()> {
        if !budget.allows(
            u64::try_from(self.state.accumulators.len()).unwrap_or(u64::MAX),
            self.state.accumulator_bytes,
        ) {
            return Err(invalid_argument(
                "window.state_budget",
                "existing accumulator state exceeds the requested budget",
            ));
        }
        self.state_budget = budget;
        Ok(())
    }

    fn prepare_input_batch(
        &self,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<InputBatchUpdate> {
        if self.state.ended {
            return Err(operator_error(
                context.operator_id(),
                "received data after end-of-input",
            ));
        }
        let table = batch.table_payload()?;
        let mut scratch = BatchScratch::new(
            WindowStateUsage {
                rows: logical_length(self.state.accumulators.len()),
                bytes: self.state.accumulator_bytes,
            },
            self.compiled.geometry.overlap,
        );

        for record in table.batches() {
            let columns =
                RecordColumns::new(record, &self.spec, &self.compiled, context.operator_id())?;
            for row in 0..record.num_rows() {
                self.prepare_row(&columns, row, context, &mut scratch)?;
            }
        }

        Ok(scratch.into_update())
    }

    fn prepare_row<'a>(
        &self,
        columns: &RecordColumns<'a>,
        row: usize,
        context: &StreamOperatorContext<'_>,
        scratch: &mut BatchScratch<'a>,
    ) -> Result<()> {
        let operator_id = context.operator_id();
        let Some(event_time) =
            columns
                .event_time
                .at(row, operator_id, &self.spec.event_time_column)?
        else {
            return record_null_event_time(&mut scratch.metrics, operator_id);
        };
        let assignments = window_assignments(event_time, self.compiled.geometry)
            .map_err(|message| operator_error(operator_id, &message))?;
        let watermark = context.input_watermark();
        if !record_late_assignments(
            assignments.clone(),
            watermark,
            &mut scratch.metrics,
            operator_id,
        )? {
            return Ok(());
        }
        let group = scratch.intern_group(columns, row, operator_id, &self.spec.group_by)?;
        assignments
            .filter(|&(_, end)| is_open_assignment(end, watermark))
            .try_for_each(|(start, end)| {
                let slot = WindowSlot { start, end, group };
                self.prepare_assignment(columns, row, slot, scratch, operator_id)
            })
    }

    fn prepare_assignment(
        &self,
        columns: &RecordColumns<'_>,
        row: usize,
        slot: WindowSlot,
        scratch: &mut BatchScratch<'_>,
        operator_id: &str,
    ) -> Result<()> {
        let (index, previous_dynamic_bytes) = self.accumulator_slot(scratch, slot, columns, row);
        let (key, accumulator) = &mut scratch.entries[index];
        update_accumulators(accumulator, columns, row, &self.spec, operator_id)?;
        let next = charge_accumulator(
            scratch.usage,
            previous_dynamic_bytes,
            logical_length(key.stable_group_key.len()),
            accumulator,
        )?;
        if !self.state_budget.allows(next.rows, next.bytes) {
            return Err(operator_error(operator_id, "window state budget exceeded"));
        }
        scratch.usage = next;
        Ok(())
    }

    /// Finds or creates the batch scratch entry for one window slot, seeding a
    /// new entry from live state. Returns the entry index and the dynamic
    /// bytes already charged for it, or `None` when the key is new to state.
    fn accumulator_slot(
        &self,
        scratch: &mut BatchScratch<'_>,
        slot: WindowSlot,
        columns: &RecordColumns<'_>,
        row: usize,
    ) -> (usize, Option<u64>) {
        let slot_key = (slot.start.as_micros(), slot.group);
        if let Some(index) = scratch.cached_slot(slot_key) {
            return (
                index,
                Some(aggregate_dynamic_bytes(&scratch.entries[index].1)),
            );
        }
        #[cfg(test)]
        group_tests::SLOT_LOOKUPS.with(|calls| calls.set(calls.get() + 1));
        if let Some(&index) = scratch.slots.get(&slot_key) {
            scratch.cache_slot(slot_key, index);
            return (
                index,
                Some(aggregate_dynamic_bytes(&scratch.entries[index].1)),
            );
        }
        let key = WindowKey {
            start: slot.start,
            end: slot.end,
            stable_group_key: Arc::clone(&scratch.group_keys[slot.group]),
        };
        let existing = self.state.accumulators.get(&key);
        let previous_dynamic_bytes = existing.map(aggregate_dynamic_bytes);
        let entry = existing.cloned().unwrap_or_else(|| {
            let values = columns
                .groups
                .iter()
                .map(|(column, _)| column.scalar_at(row))
                .collect();
            new_accumulator_row(&self.spec, &self.compiled, values)
        });
        let index = scratch.entries.len();
        scratch.entries.push((key, entry));
        scratch.slots.insert(slot_key, index);
        scratch.cache_slot(slot_key, index);
        (index, previous_dynamic_bytes)
    }

    fn observe_context(&self, context: &StreamOperatorContext<'_>) -> Result<()> {
        if self
            .state
            .pipeline_fingerprint
            .as_deref()
            .is_some_and(|value| value != context.job().fingerprint())
        {
            return Err(operator_error(
                context.operator_id(),
                "window state was used with a different pipeline fingerprint",
            ));
        }
        if self
            .state
            .operator_id
            .as_deref()
            .is_some_and(|value| value != context.operator_id())
        {
            return Err(operator_error(
                context.operator_id(),
                "window state was used with a different operator ID",
            ));
        }
        Ok(())
    }

    fn install_context_identity(&mut self, context: &StreamOperatorContext<'_>) {
        self.state
            .pipeline_fingerprint
            .get_or_insert_with(|| context.job().fingerprint().to_owned());
        self.state
            .operator_id
            .get_or_insert_with(|| context.operator_id().to_owned());
    }

    async fn encode_operations(
        &self,
        operations: Vec<StateOperationRow>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<Vec<u8>>> {
        if operations.is_empty() {
            return Ok(None);
        }
        let spec = self.spec.clone();
        let compiled = self.compiled.clone();
        let pipeline_fingerprint = context.job().fingerprint().to_owned();
        let operator_id = context.operator_id().to_owned();
        tokio::task::spawn_blocking(move || {
            encode_state_segment(
                &operations,
                &spec,
                &compiled,
                &pipeline_fingerprint,
                &operator_id,
            )
        })
        .await
        .map_err(|error| internal_error(format!("window state encoder task failed: {error}")))?
        .map(Some)
    }

    fn upsert_operations(update: &InputBatchUpdate) -> Vec<StateOperationRow> {
        update
            .accumulators
            .iter()
            .map(|(key, entry)| StateOperationRow {
                key: key.clone(),
                entry: entry.clone(),
                tombstone: false,
            })
            .collect()
    }

    fn tombstone_operations(&self, keys: &[WindowKey]) -> Vec<StateOperationRow> {
        keys.iter()
            .map(|key| StateOperationRow {
                key: key.clone(),
                entry: self.state.accumulators[key].clone(),
                tombstone: true,
            })
            .collect()
    }

    async fn compact_prepared_if_needed(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let retained_delta_count = self
            .state
            .retained_inventory
            .segments()
            .iter()
            .filter(|segment| segment.kind == SegmentKind::Delta)
            .count();
        let prepared_delta_count = self
            .state
            .prepared_segments
            .iter()
            .filter(|segment| segment.kind == SegmentKind::Delta)
            .count();
        let retained_requires_compaction = self.state.retained_inventory.needs_compaction(
            MAX_WINDOW_DELTA_SEGMENTS,
            crate::MAX_MANIFEST_DOCUMENT_BYTES,
        )?;
        if !retained_requires_compaction
            && retained_delta_count
                .checked_add(prepared_delta_count)
                .is_some_and(|count| count <= MAX_WINDOW_DELTA_SEGMENTS)
        {
            return Ok(());
        }

        let operations = self
            .state
            .accumulators
            .iter()
            .filter(|(key, _)| !self.state.emitted_pending_snapshot.contains(*key))
            .map(|(key, entry)| StateOperationRow {
                key: key.clone(),
                entry: entry.clone(),
                tombstone: false,
            })
            .collect();
        let compacted = self.encode_operations(operations, context).await?;
        self.state.prepared_segments = compacted
            .map(|bytes| {
                vec![PreparedStateSegment {
                    kind: SegmentKind::Base,
                    bytes,
                }]
            })
            .unwrap_or_default();
        self.state.replace_retained_on_checkpoint = true;
        Ok(())
    }

    async fn emit_keys(
        &mut self,
        keys: &[WindowKey],
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        let record = build_output_record(
            keys,
            &self.state.accumulators,
            &self.spec,
            &self.compiled,
            self.output_ports[0]
                .schema()
                .expect("window output always has an exact schema"),
            context.operator_id(),
        )?;
        let batches = chunk_output_record(
            &record,
            context.operator_id(),
            self.state.next_output_sequence,
            context.output_budget(),
        )?;
        for batch in batches {
            output.emit("output", batch).await?;
            self.state.next_output_sequence = self
                .state
                .next_output_sequence
                .checked_add(1)
                .expect("all output sequences were prevalidated");
        }
        if context.job().checkpointing() {
            self.state
                .emitted_pending_snapshot
                .extend(keys.iter().cloned());
        } else {
            for key in keys {
                if let Some(entry) = self.state.accumulators.remove(key) {
                    self.state.accumulator_bytes =
                        self.state
                            .accumulator_bytes
                            .saturating_sub(window_entry_bytes(
                                logical_length(key.stable_group_key.len()),
                                &entry,
                            ));
                }
            }
        }
        Ok(())
    }
}

/// Splits one window output record into edge-budget-sized messages via the
/// shared operator chunker.
fn chunk_output_record(
    record: &RecordBatch,
    operator_id: &str,
    first_sequence: u64,
    budget: crate::EdgeBudget,
) -> Result<Vec<Batch>> {
    super::output_chunk::chunk_output_record(
        record,
        operator_id,
        first_sequence,
        budget,
        super::output_chunk::OutputChunkErrors::WINDOW,
    )
}

impl OperatorMetadata for WindowAggregateOperator {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_ports(&self) -> &[Port] {
        &self.input_ports
    }

    fn output_ports(&self) -> &[Port] {
        &self.output_ports
    }

    fn configuration(&self) -> JsonMap {
        configuration(&self.spec).expect("validated window configuration remains serializable")
    }
}

#[async_trait]
impl StreamOperator for WindowAggregateOperator {
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.validate_process_input(ingress, &batch, context)?;
        self.compact_prepared_if_needed(context).await?;
        let update = self.prepare_input_batch(&batch, context)?;
        let next_metrics = accumulate_late_metrics(self.state.metrics, update.metrics)?;
        let encoded = if context.job().checkpointing() {
            self.encode_operations(Self::upsert_operations(&update), context)
                .await?
        } else {
            None
        };
        context.record_window_metrics(
            update.metrics.late_rows,
            update.metrics.max_lateness_micros,
            update.metrics.null_event_time_rows,
        )?;
        self.install_input_update(update, next_metrics, encoded, context);
        Ok(())
    }

    async fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.observe_context(context)?;
        if self
            .state
            .last_input_watermark
            .is_some_and(|previous| watermark <= previous)
        {
            return Err(operator_error(
                context.operator_id(),
                "input watermark did not advance strictly",
            ));
        }
        self.compact_prepared_if_needed(context).await?;
        let keys = self
            .state
            .accumulators
            .keys()
            .filter(|key| {
                key.end <= watermark && !self.state.emitted_pending_snapshot.contains(*key)
            })
            .cloned()
            .collect::<Vec<_>>();
        let tombstones = if context.job().checkpointing() {
            self.encode_operations(self.tombstone_operations(&keys), context)
                .await?
        } else {
            None
        };
        self.emit_keys(&keys, context, output).await?;
        if let Some(tombstones) = tombstones {
            self.state.prepared_segments.push(PreparedStateSegment {
                kind: SegmentKind::Delta,
                bytes: tombstones,
            });
        }
        self.install_context_identity(context);
        self.state.last_input_watermark = Some(watermark);
        Ok(())
    }

    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.observe_context(context)?;
        if self.state.ended {
            return Ok(());
        }
        self.compact_prepared_if_needed(context).await?;
        let keys = self
            .state
            .accumulators
            .keys()
            .filter(|key| !self.state.emitted_pending_snapshot.contains(*key))
            .cloned()
            .collect::<Vec<_>>();
        let tombstones = if context.job().checkpointing() {
            self.encode_operations(self.tombstone_operations(&keys), context)
                .await?
        } else {
            None
        };
        self.emit_keys(&keys, context, output).await?;
        if let Some(tombstones) = tombstones {
            self.state.prepared_segments.push(PreparedStateSegment {
                kind: SegmentKind::Delta,
                bytes: tombstones,
            });
        }
        self.install_context_identity(context);
        self.state.ended = true;
        Ok(())
    }

    fn checkpoint(&mut self, epoch: crate::Epoch) -> Result<crate::OperatorStateSnapshot> {
        if self
            .state
            .last_checkpoint_epoch
            .is_some_and(|previous| epoch <= previous)
        {
            return Err(checkpoint_mismatch(
                "window checkpoint epoch did not advance strictly",
            ));
        }
        let prepared = self.prepare_snapshot_segments(epoch)?;
        let retained_inventory = self.next_snapshot_inventory(prepared.descriptors)?;
        let retained_segments = self.next_snapshot_segments(&retained_inventory, prepared.bytes)?;
        let metadata = WindowSnapshotMetadata {
            state_layout_version: WINDOW_SEGMENT_LAYOUT_VERSION,
            configuration_hash: self.compiled.configuration_hash.clone(),
            state_schema_fingerprint: self.compiled.state_schema_fingerprint.clone(),
            epoch,
            pipeline_fingerprint: self.state.pipeline_fingerprint.clone(),
            operator_id: self.state.operator_id.clone(),
            last_input_watermark: self.state.last_input_watermark,
            next_output_sequence: self.state.next_output_sequence,
            ended: self.state.ended,
            metrics: self.state.metrics,
            segment_inventory: retained_inventory.segments().to_vec(),
        };
        let Value::Object(inline_metadata) =
            serde_json::to_value(metadata).map_err(|error| format_error(&error))?
        else {
            return Err(internal_error(
                "window snapshot metadata did not serialize as an object",
            ));
        };
        self.state.prepared_segments.clear();
        for key in std::mem::take(&mut self.state.emitted_pending_snapshot) {
            if let Some(entry) = self.state.accumulators.remove(&key) {
                self.state.accumulator_bytes =
                    self.state
                        .accumulator_bytes
                        .saturating_sub(window_entry_bytes(
                            logical_length(key.stable_group_key.len()),
                            &entry,
                        ));
            }
            self.state.dirty.remove(&key);
        }
        self.state.dirty.clear();
        self.state.retained_inventory = retained_inventory;
        self.state.retained_segments = retained_segments.clone();
        self.state.replace_retained_on_checkpoint = false;
        self.state.last_checkpoint_epoch = Some(epoch);
        Ok(crate::OperatorStateSnapshot {
            inline_metadata: inline_metadata.into_iter().collect(),
            segments: retained_segments,
        })
    }

    fn restore(&mut self, snapshot: &crate::OperatorStateSnapshot) -> Result<()> {
        if snapshot.inline_metadata.is_empty() && snapshot.segments.is_empty() {
            return self.reset();
        }
        let metadata = parse_snapshot_metadata(snapshot)?;
        let inventory =
            validate_snapshot_metadata(&metadata, &self.spec, &self.compiled, snapshot)?;
        let decoded = self.decode_snapshot_segments(snapshot, &metadata)?;
        self.install_restored_state(metadata, inventory, snapshot.segments.clone(), decoded)
    }

    fn reset(&mut self) -> Result<()> {
        self.state = WindowState::default();
        Ok(())
    }
}

impl WindowAggregateOperator {
    fn prepare_snapshot_segments(&self, epoch: crate::Epoch) -> Result<PreparedSnapshotSegments> {
        let mut descriptors = Vec::with_capacity(self.state.prepared_segments.len());
        let mut segments = BTreeMap::new();
        for (ordinal, segment) in self.state.prepared_segments.iter().enumerate() {
            let kind = match segment.kind {
                SegmentKind::Base => "base",
                SegmentKind::Delta => "delta",
            };
            let segment_id = format!("{kind}-{:020}-{ordinal:08}", epoch.as_u64());
            // One shared allocation and one digest serve both the snapshot and
            // the manifest descriptor; nothing re-encodes or re-hashes here.
            let snapshot_segment = crate::StateSegment::new(segment.bytes.clone());
            let descriptor = self.snapshot_segment_descriptor(
                epoch,
                &segment_id,
                segment.kind,
                &snapshot_segment,
            )?;
            descriptors.push(descriptor);
            segments.insert(segment_id, snapshot_segment);
        }
        Ok(PreparedSnapshotSegments {
            descriptors,
            bytes: segments,
        })
    }

    fn snapshot_segment_descriptor(
        &self,
        epoch: crate::Epoch,
        segment_id: &str,
        kind: SegmentKind,
        segment: &crate::StateSegment,
    ) -> Result<SegmentDescriptor> {
        let operator_id = self.state.operator_id.as_deref().ok_or_else(|| {
            checkpoint_mismatch("window segment is missing its operator identity")
        })?;
        let relative_path = format!(
            "committed/{operator_id}/{:020}-{segment_id}.arrow",
            epoch.as_u64()
        );
        let byte_len = u64::try_from(segment.bytes().len())
            .map_err(|_| internal_error("window segment length does not fit u64"))?;
        Ok(SegmentDescriptor {
            kind,
            state_layout_version: WINDOW_SEGMENT_LAYOUT_VERSION,
            schema_fingerprint: self.compiled.state_schema_fingerprint.clone(),
            handle: StateHandle::new(
                operator_id,
                epoch,
                segment_id,
                &relative_path,
                byte_len,
                segment.sha256(),
            )?,
        })
    }

    fn next_snapshot_inventory(
        &self,
        new_descriptors: Vec<SegmentDescriptor>,
    ) -> Result<StateInventory> {
        if !self.state.replace_retained_on_checkpoint {
            let mut retained = self.state.retained_inventory.segments().to_vec();
            retained.extend(new_descriptors);
            return StateInventory::new(retained);
        }
        let Some((base, later)) = new_descriptors.split_first() else {
            return Ok(StateInventory::default());
        };
        if base.kind != SegmentKind::Base {
            return StateInventory::new(new_descriptors);
        }
        if self
            .state
            .retained_inventory
            .segments()
            .first()
            .is_some_and(|retained| {
                retained.state_layout_version != base.state_layout_version
                    || retained.schema_fingerprint != base.schema_fingerprint
            })
        {
            return StateInventory::new(new_descriptors);
        }
        let replacement = self
            .state
            .retained_inventory
            .replacement_after_full_compaction(base.clone())?;
        let mut retained = replacement.segments().to_vec();
        retained.extend_from_slice(later);
        StateInventory::new(retained)
    }

    fn next_snapshot_segments(
        &self,
        inventory: &StateInventory,
        new_segments: BTreeMap<String, crate::StateSegment>,
    ) -> Result<BTreeMap<String, crate::StateSegment>> {
        let mut retained = if self.state.replace_retained_on_checkpoint {
            BTreeMap::new()
        } else {
            self.state.retained_segments.clone()
        };
        for (segment_id, segment) in new_segments {
            if retained.insert(segment_id, segment).is_some() {
                return Err(checkpoint_mismatch(
                    "window checkpoint produced a duplicate segment ID",
                ));
            }
        }
        let expected_ids = inventory
            .segments()
            .iter()
            .map(|descriptor| descriptor.handle.segment_id())
            .collect::<Vec<_>>();
        let actual_ids = retained.keys().map(String::as_str).collect::<Vec<_>>();
        if expected_ids != actual_ids {
            return Err(checkpoint_mismatch(
                "window checkpoint segment data does not match its inventory",
            ));
        }
        Ok(retained)
    }

    fn validate_process_input(
        &self,
        ingress: &str,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        if ingress != "input" {
            return Err(CalcFlowError::Operator {
                node_id: self.name.clone(),
                message: format!("unknown ingress {ingress:?}; expected \"input\""),
            });
        }
        self.input_ports[0].validate(batch, &format!("{}.input", self.name))?;
        self.observe_context(context)
    }

    fn install_input_update(
        &mut self,
        update: InputBatchUpdate,
        next_metrics: LateMetricDelta,
        encoded: Option<Vec<u8>>,
        context: &StreamOperatorContext<'_>,
    ) {
        self.install_context_identity(context);
        self.state.metrics = next_metrics;
        self.state.accumulator_bytes = update.usage.bytes;
        for (key, accumulator) in update.accumulators {
            if context.job().checkpointing() {
                self.state.dirty.insert(key.clone());
            }
            self.state.accumulators.insert(key, accumulator);
        }
        if let Some(encoded) = encoded {
            self.state.prepared_segments.push(PreparedStateSegment {
                kind: SegmentKind::Delta,
                bytes: encoded,
            });
        }
    }

    fn decode_snapshot_segments(
        &self,
        snapshot: &crate::OperatorStateSnapshot,
        metadata: &WindowSnapshotMetadata,
    ) -> Result<BTreeMap<WindowKey, AccumulatorRow>> {
        let segments = snapshot_segments(snapshot, &metadata.segment_inventory)?;
        let spec = self.spec.clone();
        let compiled = self.compiled.clone();
        let pipeline_fingerprint = metadata.pipeline_fingerprint.clone();
        let operator_id = metadata.operator_id.clone();
        std::thread::spawn(move || {
            decode_state_segments(
                &segments,
                &spec,
                &compiled,
                pipeline_fingerprint.as_deref(),
                operator_id.as_deref(),
            )
        })
        .join()
        .map_err(|_| internal_error("window state decoder worker panicked"))?
    }

    fn install_restored_state(
        &mut self,
        metadata: WindowSnapshotMetadata,
        inventory: StateInventory,
        retained_segments: BTreeMap<String, crate::StateSegment>,
        decoded: BTreeMap<WindowKey, AccumulatorRow>,
    ) -> Result<()> {
        let usage = window_state_usage(&decoded);
        if !self.state_budget.allows(usage.rows, usage.bytes) {
            return Err(checkpoint_mismatch("window state budget exceeded"));
        }
        let migrate_legacy = metadata.state_layout_version == WINDOW_STATE_LAYOUT_VERSION;
        let prepared_segments = if migrate_legacy && !decoded.is_empty() {
            let pipeline_fingerprint =
                metadata.pipeline_fingerprint.as_deref().ok_or_else(|| {
                    checkpoint_mismatch("window legacy state is missing its pipeline fingerprint")
                })?;
            let operator_id = metadata.operator_id.as_deref().ok_or_else(|| {
                checkpoint_mismatch("window legacy state is missing its operator ID")
            })?;
            let operations = decoded
                .iter()
                .map(|(key, entry)| StateOperationRow {
                    key: key.clone(),
                    entry: entry.clone(),
                    tombstone: false,
                })
                .collect::<Vec<_>>();
            vec![PreparedStateSegment {
                kind: SegmentKind::Base,
                bytes: encode_state_segment(
                    &operations,
                    &self.spec,
                    &self.compiled,
                    pipeline_fingerprint,
                    operator_id,
                )?,
            }]
        } else {
            Vec::new()
        };
        self.state = WindowState {
            accumulators: decoded,
            accumulator_bytes: usage.bytes,
            last_input_watermark: metadata.last_input_watermark,
            next_output_sequence: metadata.next_output_sequence,
            ended: metadata.ended,
            metrics: metadata.metrics,
            retained_inventory: inventory,
            retained_segments,
            prepared_segments,
            replace_retained_on_checkpoint: migrate_legacy,
            last_checkpoint_epoch: Some(metadata.epoch),
            pipeline_fingerprint: metadata.pipeline_fingerprint,
            operator_id: metadata.operator_id,
            ..WindowState::default()
        };
        Ok(())
    }
}

fn parse_snapshot_metadata(
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<WindowSnapshotMetadata> {
    serde_json::from_value::<WindowSnapshotMetadata>(Value::Object(
        snapshot.inline_metadata.clone().into_iter().collect(),
    ))
    .map_err(|error| format_error(&error))
}

fn snapshot_segments(
    snapshot: &crate::OperatorStateSnapshot,
    inventory: &[SegmentDescriptor],
) -> Result<Vec<(u32, Arc<Vec<u8>>)>> {
    inventory
        .iter()
        .map(|descriptor| {
            let segment_id = descriptor.handle.segment_id();
            let segment = snapshot.segments.get(segment_id).ok_or_else(|| {
                checkpoint_mismatch(format!("window snapshot is missing segment {segment_id:?}"))
            })?;
            validate_snapshot_segment_bytes(descriptor, segment.bytes())?;
            Ok((descriptor.state_layout_version, segment.bytes_arc()))
        })
        .collect()
}

fn validate_snapshot_segment_bytes(descriptor: &SegmentDescriptor, bytes: &[u8]) -> Result<()> {
    if u64::try_from(bytes.len()).ok() != Some(descriptor.handle.byte_len()) {
        return Err(checkpoint_mismatch(
            "window snapshot segment byte length does not match its handle",
        ));
    }
    if hex::encode(Sha256::digest(bytes)) != descriptor.handle.sha256() {
        return Err(checkpoint_mismatch(
            "window snapshot segment checksum does not match its handle",
        ));
    }
    Ok(())
}

fn record_null_event_time(metrics: &mut PreparedInputMetrics, operator_id: &str) -> Result<()> {
    metrics.null_event_time_rows = metrics
        .null_event_time_rows
        .checked_add(1)
        .ok_or_else(|| operator_error(operator_id, "null event-time row counter overflowed"))?;
    Ok(())
}

/// Records every late assignment and reports whether any assignment is open.
fn record_late_assignments(
    assignments: impl Iterator<Item = (EventTime, EventTime)>,
    watermark: Option<EventTime>,
    metrics: &mut PreparedInputMetrics,
    operator_id: &str,
) -> Result<bool> {
    let mut any_open = false;
    for (_, end) in assignments {
        if let Some(closing_watermark) = watermark.filter(|value| end <= *value) {
            record_late_assignment(metrics, closing_watermark, end, operator_id)?;
        } else {
            any_open = true;
        }
    }
    Ok(any_open)
}

fn is_open_assignment(end: EventTime, watermark: Option<EventTime>) -> bool {
    watermark.is_none_or(|value| end > value)
}

fn record_late_assignment(
    metrics: &mut PreparedInputMetrics,
    watermark: EventTime,
    end: EventTime,
    operator_id: &str,
) -> Result<()> {
    metrics.late_rows = metrics
        .late_rows
        .checked_add(1)
        .ok_or_else(|| operator_error(operator_id, "late row counter overflowed"))?;
    let lateness =
        u64::try_from(i128::from(watermark.as_micros()) - i128::from(end.as_micros()))
            .map_err(|_| operator_error(operator_id, "late assignment distance overflowed"))?;
    metrics.max_lateness_micros = Some(
        metrics
            .max_lateness_micros
            .map_or(lateness, |maximum| maximum.max(lateness)),
    );
    Ok(())
}

/// The event-time column downcast once per record; every supported unit
/// stores `i64` values.
struct EventTimeColumn<'a> {
    array: &'a dyn Array,
    data_type: &'a DataType,
    values: &'a [i64],
}

impl<'a> EventTimeColumn<'a> {
    fn new(record: &'a RecordBatch, index: usize, operator_id: &str) -> Result<Self> {
        let array = record.column(index).as_ref();
        let data_type = record.schema_ref().field(index).data_type();
        let values = match data_type {
            DataType::Timestamp(TimeUnit::Second, _) => downcast_array::<TimestampSecondArray>(
                array,
                operator_id,
                "event-time timestamp(second)",
            )
            .map(|array| array.values().as_ref()),
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                downcast_array::<TimestampMillisecondArray>(
                    array,
                    operator_id,
                    "event-time timestamp(millisecond)",
                )
                .map(|array| array.values().as_ref())
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                downcast_array::<TimestampMicrosecondArray>(
                    array,
                    operator_id,
                    "event-time timestamp(microsecond)",
                )
                .map(|array| array.values().as_ref())
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                downcast_array::<TimestampNanosecondArray>(
                    array,
                    operator_id,
                    "event-time timestamp(nanosecond)",
                )
                .map(|array| array.values().as_ref())
            }
            _ => Err(operator_error(
                operator_id,
                "compiled event-time column is not a timestamp",
            )),
        }?;
        Ok(Self {
            array,
            data_type,
            values,
        })
    }

    fn at(&self, row: usize, operator_id: &str, column: &str) -> Result<Option<EventTime>> {
        if self.array.is_null(row) {
            return Ok(None);
        }
        EventTime::import_timestamp(self.values[row], self.data_type, column)
            .map(Some)
            .map_err(|error| {
                operator_error(
                    operator_id,
                    &format!("event-time conversion failed: {error}"),
                )
            })
    }
}

/// Returns the event's window assignments, earliest window first, without
/// allocating. Starts and ends grow monotonically with the window, so
/// validating the two extremes up front validates every assignment and keeps
/// the eager error precedence.
fn window_assignments(
    event_time: EventTime,
    geometry: CompiledWindowGeometry,
) -> std::result::Result<impl Iterator<Item = (EventTime, EventTime)> + Clone, String> {
    let time = i128::from(event_time.as_micros());
    let size = i128::from(geometry.size_micros);
    let slide = i128::from(geometry.slide_micros);
    let latest_start = time.div_euclid(slide) * slide;
    let assignment = move |offset| window_assignment(latest_start, slide, size, offset);
    assignment(geometry.overlap - 1)?;
    assignment(0)?;
    Ok((0..geometry.overlap)
        .rev()
        .map(move |offset| assignment(offset).expect("window assignment extremes were validated")))
}

fn window_assignment(
    latest_start: i128,
    slide: i128,
    size: i128,
    offset: u64,
) -> std::result::Result<(EventTime, EventTime), String> {
    let start = latest_start
        .checked_sub(i128::from(offset) * slide)
        .ok_or_else(|| "window assignment start overflowed".to_string())?;
    let end = start
        .checked_add(size)
        .ok_or_else(|| "window assignment end overflowed".to_string())?;
    let start = i64::try_from(start)
        .map_err(|_| "window assignment start is outside EventTime".to_string())?;
    let end =
        i64::try_from(end).map_err(|_| "window assignment end is outside EventTime".to_string())?;
    Ok((EventTime::from_micros(start), EventTime::from_micros(end)))
}

/// Encodes the row's stable group key without allocating string scalars.
fn encode_group_key(
    columns: &RecordColumns<'_>,
    row: usize,
    operator_id: &str,
    names: &[String],
    encoded: &mut Vec<u8>,
) -> Result<()> {
    #[cfg(test)]
    key_tests::GROUP_ENCODINGS.with(|calls| calls.set(calls.get() + 1));
    encoded.clear();
    for (ordinal, (column, data_type)) in columns.groups.iter().enumerate() {
        let result = if let BorrowedString::Value(value) = column.borrowed_string(row) {
            match value {
                None => extend_group_encoding(encoded, &[0x00]),
                Some(value) => extend_group_encoding(encoded, &[0x01])
                    .and_then(|()| encode_group_string_bytes(encoded, value)),
            }
        } else {
            encode_group_scalar(encoded, data_type, column.scalar_at(row).as_ref())
        };
        result.map_err(|message| {
            operator_error(
                operator_id,
                &format!(
                    "window.group_by[{ordinal}] ({:?}) encoding failed: {message}",
                    names[ordinal]
                ),
            )
        })?;
    }
    Ok(())
}

fn encode_group_scalar(
    encoded: &mut Vec<u8>,
    data_type: &DataType,
    value: Option<&ScalarValue>,
) -> std::result::Result<(), String> {
    let Some(value) = value else {
        extend_group_encoding(encoded, &[0x00])?;
        return Ok(());
    };
    extend_group_encoding(encoded, &[0x01])?;
    match data_type {
        DataType::Boolean => encode_boolean_group(encoded, value),
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            encode_signed_group(encoded, data_type, value)
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            encode_unsigned_group(encoded, data_type, value)
        }
        DataType::Float32 | DataType::Float64 => encode_float_group(encoded, data_type, value),
        DataType::Utf8 | DataType::LargeUtf8 => encode_string_group(encoded, value),
        DataType::Date32 | DataType::Date64 | DataType::Timestamp(TimeUnit::Microsecond, _) => {
            encode_temporal_group(encoded, data_type, value)
        }
        _ => Err("compiled group scalar type mismatch".into()),
    }
}

fn encode_boolean_group(
    encoded: &mut Vec<u8>,
    value: &ScalarValue,
) -> std::result::Result<(), String> {
    let ScalarValue::Boolean(value) = value else {
        return Err("compiled group scalar type mismatch".into());
    };
    extend_group_encoding(encoded, &[u8::from(*value)])
}

fn encode_signed_group(
    encoded: &mut Vec<u8>,
    data_type: &DataType,
    value: &ScalarValue,
) -> std::result::Result<(), String> {
    let ScalarValue::Signed(value) = value else {
        return Err("compiled group scalar type mismatch".into());
    };
    match data_type {
        DataType::Int8 => encode_group_i8(encoded, *value),
        DataType::Int16 => encode_group_i16(encoded, *value),
        DataType::Int32 => encode_group_i32(encoded, *value),
        DataType::Int64 => extend_group_encoding(encoded, &ordered_i64(*value)),
        _ => Err("compiled group scalar type mismatch".into()),
    }
}

fn encode_group_i8(encoded: &mut Vec<u8>, value: i64) -> std::result::Result<(), String> {
    let value = i8::try_from(value).map_err(|_| "Int8 group scalar escaped its compiled range")?;
    extend_group_encoding(encoded, &[value.to_be_bytes()[0] ^ 0x80])
}

fn encode_group_i16(encoded: &mut Vec<u8>, value: i64) -> std::result::Result<(), String> {
    let value =
        i16::try_from(value).map_err(|_| "Int16 group scalar escaped its compiled range")?;
    let mut bytes = value.to_be_bytes();
    bytes[0] ^= 0x80;
    extend_group_encoding(encoded, &bytes)
}

fn encode_group_i32(encoded: &mut Vec<u8>, value: i64) -> std::result::Result<(), String> {
    let value =
        i32::try_from(value).map_err(|_| "Int32 group scalar escaped its compiled range")?;
    let mut bytes = value.to_be_bytes();
    bytes[0] ^= 0x80;
    extend_group_encoding(encoded, &bytes)
}

fn ordered_i64(value: i64) -> [u8; 8] {
    let mut bytes = value.to_be_bytes();
    bytes[0] ^= 0x80;
    bytes
}

fn encode_unsigned_group(
    encoded: &mut Vec<u8>,
    data_type: &DataType,
    value: &ScalarValue,
) -> std::result::Result<(), String> {
    let ScalarValue::Unsigned(value) = value else {
        return Err("compiled group scalar type mismatch".into());
    };
    match data_type {
        DataType::UInt8 => encode_group_u8(encoded, *value),
        DataType::UInt16 => encode_group_u16(encoded, *value),
        DataType::UInt32 => encode_group_u32(encoded, *value),
        DataType::UInt64 => extend_group_encoding(encoded, &value.to_be_bytes()),
        _ => Err("compiled group scalar type mismatch".into()),
    }
}

fn encode_group_u8(encoded: &mut Vec<u8>, value: u64) -> std::result::Result<(), String> {
    let value = u8::try_from(value).map_err(|_| "UInt8 group scalar escaped its compiled range")?;
    extend_group_encoding(encoded, &[value])
}

fn encode_group_u16(encoded: &mut Vec<u8>, value: u64) -> std::result::Result<(), String> {
    let value =
        u16::try_from(value).map_err(|_| "UInt16 group scalar escaped its compiled range")?;
    extend_group_encoding(encoded, &value.to_be_bytes())
}

fn encode_group_u32(encoded: &mut Vec<u8>, value: u64) -> std::result::Result<(), String> {
    let value =
        u32::try_from(value).map_err(|_| "UInt32 group scalar escaped its compiled range")?;
    extend_group_encoding(encoded, &value.to_be_bytes())
}

fn encode_float_group(
    encoded: &mut Vec<u8>,
    data_type: &DataType,
    value: &ScalarValue,
) -> std::result::Result<(), String> {
    match (data_type, value) {
        (DataType::Float32, ScalarValue::Float32(bits)) => {
            extend_group_encoding(encoded, &ordered_float32(*bits).to_be_bytes())
        }
        (DataType::Float64, ScalarValue::Float64(bits)) => {
            extend_group_encoding(encoded, &ordered_float64(*bits).to_be_bytes())
        }
        _ => Err("compiled group scalar type mismatch".into()),
    }
}

fn ordered_float32(bits: u32) -> u32 {
    if bits & (1 << 31) != 0 {
        !bits
    } else {
        bits | (1 << 31)
    }
}

fn ordered_float64(bits: u64) -> u64 {
    if bits & (1 << 63) != 0 {
        !bits
    } else {
        bits | (1 << 63)
    }
}

fn encode_string_group(
    encoded: &mut Vec<u8>,
    value: &ScalarValue,
) -> std::result::Result<(), String> {
    let ScalarValue::String(value) = value else {
        return Err("compiled group scalar type mismatch".into());
    };
    encode_group_string_bytes(encoded, value)
}

fn encode_group_string_bytes(
    encoded: &mut Vec<u8>,
    value: &str,
) -> std::result::Result<(), String> {
    for byte in value.as_bytes() {
        let escaped = if *byte == 0 {
            &[0x00, 0xff][..]
        } else {
            std::slice::from_ref(byte)
        };
        extend_group_encoding(encoded, escaped)?;
    }
    extend_group_encoding(encoded, &[0x00, 0x00])
}

fn encode_temporal_group(
    encoded: &mut Vec<u8>,
    data_type: &DataType,
    value: &ScalarValue,
) -> std::result::Result<(), String> {
    match (data_type, value) {
        (DataType::Date32, ScalarValue::Date32(value)) => {
            let mut bytes = value.to_be_bytes();
            bytes[0] ^= 0x80;
            extend_group_encoding(encoded, &bytes)
        }
        (DataType::Date64, ScalarValue::Date64(value))
        | (DataType::Timestamp(TimeUnit::Microsecond, _), ScalarValue::Timestamp(value)) => {
            extend_group_encoding(encoded, &ordered_i64(*value))
        }
        _ => Err("compiled group scalar type mismatch".into()),
    }
}

fn extend_group_encoding(encoded: &mut Vec<u8>, bytes: &[u8]) -> std::result::Result<(), String> {
    if encoded
        .len()
        .checked_add(bytes.len())
        .is_none_or(|length| length > MAX_GROUP_KEY_BYTES)
    {
        return Err(format!(
            "stable key exceeds the {MAX_GROUP_KEY_BYTES}-byte limit"
        ));
    }
    encoded.extend_from_slice(bytes);
    Ok(())
}

fn new_accumulator_row(
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    group_values: Vec<Option<ScalarValue>>,
) -> AccumulatorRow {
    let aggregates = spec
        .aggregates
        .iter()
        .zip(&compiled.aggregates)
        .map(|(aggregate, compiled)| match aggregate.function {
            AggregateFunction::Count => AccumulatorValue::Count(0),
            AggregateFunction::Sum => match compiled.output_type {
                DataType::Int64 => AccumulatorValue::SignedSum(None),
                DataType::UInt64 => AccumulatorValue::UnsignedSum(None),
                DataType::Float64 => AccumulatorValue::FloatSum(None),
                _ => unreachable!("aggregate matrix validated at construction"),
            },
            AggregateFunction::Min => AccumulatorValue::Min(None),
            AggregateFunction::Max => AccumulatorValue::Max(None),
            AggregateFunction::Avg => match compiled.input_type {
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                    AccumulatorValue::SignedAverage { sum: 0, count: 0 }
                }
                DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                    AccumulatorValue::UnsignedAverage { sum: 0, count: 0 }
                }
                DataType::Float32 | DataType::Float64 => AccumulatorValue::FloatAverage {
                    sum: CompensatedSum::new(0.0),
                    count: 0,
                },
                _ => unreachable!("aggregate matrix validated at construction"),
            },
        })
        .collect();
    AccumulatorRow {
        group_values,
        aggregates,
    }
}

fn update_accumulators(
    row: &mut AccumulatorRow,
    columns: &RecordColumns<'_>,
    row_index: usize,
    spec: &WindowSpec,
    operator_id: &str,
) -> Result<()> {
    for (ordinal, ((_, (column, update)), accumulator)) in spec
        .aggregates
        .iter()
        .zip(&columns.aggregates)
        .zip(&mut row.aggregates)
        .enumerate()
    {
        if column.is_null(row_index) {
            continue;
        }
        update(column, row_index, accumulator).map_err(|message| {
            operator_error(
                operator_id,
                &format!("window.aggregates[{ordinal}] update failed: {message}"),
            )
        })?;
    }
    Ok(())
}

#[cfg(test)]
fn aggregate_input(
    function: AggregateFunction,
    column: &ScalarColumn<'_>,
    row: usize,
) -> Option<ScalarValue> {
    if column.is_null(row) {
        None
    } else if function == AggregateFunction::Count {
        Some(ScalarValue::Unsigned(0))
    } else {
        Some(column.value(row))
    }
}

fn update_accumulator(
    accumulator: &mut AccumulatorValue,
    function: AggregateFunction,
    value: ScalarValue,
) -> std::result::Result<(), String> {
    match function {
        AggregateFunction::Count => update_count(accumulator),
        AggregateFunction::Sum => update_sum(accumulator, &value),
        AggregateFunction::Min => update_extreme(accumulator, value, Ordering::Less),
        AggregateFunction::Max => update_extreme(accumulator, value, Ordering::Greater),
        AggregateFunction::Avg => update_average(accumulator, &value),
    }
}

fn update_count(accumulator: &mut AccumulatorValue) -> std::result::Result<(), String> {
    let AccumulatorValue::Count(count) = accumulator else {
        return Err("compiled aggregate accumulator type mismatch".into());
    };
    *count = count
        .checked_add(1)
        .ok_or_else(|| "count overflowed UInt64".to_string())?;
    Ok(())
}

fn update_sum(
    accumulator: &mut AccumulatorValue,
    value: &ScalarValue,
) -> std::result::Result<(), String> {
    match accumulator {
        AccumulatorValue::SignedSum(sum) => update_signed_sum(sum, value),
        AccumulatorValue::UnsignedSum(sum) => update_unsigned_sum(sum, value),
        AccumulatorValue::FloatSum(sum) => {
            sum.get_or_insert(CompensatedSum::new(0.0))
                .add(float_value(value)?);
            Ok(())
        }
        _ => Err("compiled aggregate accumulator type mismatch".into()),
    }
}

fn update_signed_sum(
    sum: &mut Option<i128>,
    value: &ScalarValue,
) -> std::result::Result<(), String> {
    let updated = sum
        .unwrap_or(0)
        .checked_add(i128::from(signed_value(value)?))
        .ok_or_else(|| "signed sum overflowed its widened state".to_string())?;
    i64::try_from(updated).map_err(|_| "signed sum overflowed Int64".to_string())?;
    *sum = Some(updated);
    Ok(())
}

fn update_unsigned_sum(
    sum: &mut Option<u128>,
    value: &ScalarValue,
) -> std::result::Result<(), String> {
    let updated = sum
        .unwrap_or(0)
        .checked_add(u128::from(unsigned_value(value)?))
        .ok_or_else(|| "unsigned sum overflowed its widened state".to_string())?;
    u64::try_from(updated).map_err(|_| "unsigned sum overflowed UInt64".to_string())?;
    *sum = Some(updated);
    Ok(())
}

fn update_extreme(
    accumulator: &mut AccumulatorValue,
    value: ScalarValue,
    ordering: Ordering,
) -> std::result::Result<(), String> {
    let (AccumulatorValue::Min(current) | AccumulatorValue::Max(current)) = accumulator else {
        return Err("compiled aggregate accumulator type mismatch".into());
    };
    if current
        .as_ref()
        .is_none_or(|current| scalar_total_cmp(&value, current) == ordering)
    {
        *current = Some(value);
    }
    Ok(())
}

fn update_average(
    accumulator: &mut AccumulatorValue,
    value: &ScalarValue,
) -> std::result::Result<(), String> {
    match accumulator {
        AccumulatorValue::SignedAverage { sum, count } => update_signed_average(sum, count, value),
        AccumulatorValue::UnsignedAverage { sum, count } => {
            update_unsigned_average(sum, count, value)
        }
        AccumulatorValue::FloatAverage { sum, count } => {
            sum.add(float_value(value)?);
            increment_average_count(count)
        }
        _ => Err("compiled aggregate accumulator type mismatch".into()),
    }
}

fn update_signed_average(
    sum: &mut i128,
    count: &mut u64,
    value: &ScalarValue,
) -> std::result::Result<(), String> {
    *sum = sum
        .checked_add(i128::from(signed_value(value)?))
        .ok_or_else(|| "signed average sum overflowed Int128".to_string())?;
    increment_average_count(count)
}

fn update_unsigned_average(
    sum: &mut u128,
    count: &mut u64,
    value: &ScalarValue,
) -> std::result::Result<(), String> {
    *sum = sum
        .checked_add(u128::from(unsigned_value(value)?))
        .ok_or_else(|| "unsigned average sum overflowed UInt128".to_string())?;
    increment_average_count(count)
}

fn increment_average_count(count: &mut u64) -> std::result::Result<(), String> {
    *count = count
        .checked_add(1)
        .ok_or_else(|| "average count overflowed UInt64".to_string())?;
    Ok(())
}

fn build_output_record(
    keys: &[WindowKey],
    state: &BTreeMap<WindowKey, AccumulatorRow>,
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    schema: &SchemaRef,
    operator_id: &str,
) -> Result<RecordBatch> {
    let starts = keys
        .iter()
        .map(|key| key.start.as_micros())
        .collect::<Vec<_>>();
    let ends = keys
        .iter()
        .map(|key| key.end.as_micros())
        .collect::<Vec<_>>();
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(TimestampMicrosecondArray::from(starts).with_timezone("UTC")),
        Arc::new(TimestampMicrosecondArray::from(ends).with_timezone("UTC")),
    ];

    for (ordinal, group) in compiled.group_columns.iter().enumerate() {
        let values = keys
            .iter()
            .map(|key| state[key].group_values[ordinal].clone())
            .collect::<Vec<_>>();
        arrays.push(scalar_array(&group.data_type, &values, operator_id)?);
    }
    for (ordinal, (aggregate, compiled_aggregate)) in
        spec.aggregates.iter().zip(&compiled.aggregates).enumerate()
    {
        let values = keys
            .iter()
            .map(|key| finalize_accumulator(&state[key].aggregates[ordinal]))
            .collect::<Result<Vec<_>>>()?;
        arrays.push(scalar_array(
            &compiled_aggregate.output_type,
            &values,
            operator_id,
        )?);
        debug_assert_eq!(
            schema
                .field(2 + compiled.group_columns.len() + ordinal)
                .name(),
            &aggregate.output
        );
    }
    RecordBatch::try_new(Arc::clone(schema), arrays).map_err(|error| {
        operator_error(
            operator_id,
            &format!("window output RecordBatch construction failed: {error}"),
        )
    })
}

fn finalize_accumulator(accumulator: &AccumulatorValue) -> Result<Option<ScalarValue>> {
    match accumulator {
        AccumulatorValue::Count(value) => Ok(Some(ScalarValue::Unsigned(*value))),
        AccumulatorValue::SignedSum(value) => value
            .map(|value| {
                i64::try_from(value)
                    .map(ScalarValue::Signed)
                    .map_err(|_| internal_error("signed sum escaped its output range"))
            })
            .transpose(),
        AccumulatorValue::UnsignedSum(value) => value
            .map(|value| {
                u64::try_from(value)
                    .map(ScalarValue::Unsigned)
                    .map_err(|_| internal_error("unsigned sum escaped its output range"))
            })
            .transpose(),
        AccumulatorValue::FloatSum(value) => {
            Ok(value.map(|value| ScalarValue::Float64(value.total().to_bits())))
        }
        AccumulatorValue::Min(value) | AccumulatorValue::Max(value) => Ok(value.clone()),
        AccumulatorValue::SignedAverage { sum, count } => {
            Ok((*count != 0).then(|| ScalarValue::Float64(signed_average(*sum, *count).to_bits())))
        }
        AccumulatorValue::UnsignedAverage { sum, count } => {
            Ok((*count != 0)
                .then(|| ScalarValue::Float64(unsigned_average(*sum, *count).to_bits())))
        }
        AccumulatorValue::FloatAverage { sum, count } => Ok((*count != 0)
            .then(|| ScalarValue::Float64(float_average(sum.total(), *count).to_bits()))),
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen average output type is Float64"
)]
fn signed_average(sum: i128, count: u64) -> f64 {
    sum as f64 / count as f64
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen average output type is Float64"
)]
fn unsigned_average(sum: u128, count: u64) -> f64 {
    sum as f64 / count as f64
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen average output type is Float64"
)]
fn float_average(sum: f64, count: u64) -> f64 {
    canonicalize_float(sum / count as f64)
}

fn canonicalize_float(value: f64) -> f64 {
    if value.is_nan() {
        f64::from_bits(0x7ff8_0000_0000_0000)
    } else {
        value
    }
}

#[allow(
    clippy::match_same_arms,
    reason = "distinct scalar variants intentionally preserve their logical Arrow types"
)]
fn scalar_total_cmp(left: &ScalarValue, right: &ScalarValue) -> Ordering {
    match (left, right) {
        (ScalarValue::Boolean(left), ScalarValue::Boolean(right)) => left.cmp(right),
        (ScalarValue::Signed(left), ScalarValue::Signed(right)) => left.cmp(right),
        (ScalarValue::Unsigned(left), ScalarValue::Unsigned(right)) => left.cmp(right),
        (ScalarValue::Float32(left), ScalarValue::Float32(right)) => {
            f32::from_bits(*left).total_cmp(&f32::from_bits(*right))
        }
        (ScalarValue::Float64(left), ScalarValue::Float64(right)) => {
            f64::from_bits(*left).total_cmp(&f64::from_bits(*right))
        }
        (ScalarValue::String(left), ScalarValue::String(right)) => left.cmp(right),
        (ScalarValue::Date32(left), ScalarValue::Date32(right)) => left.cmp(right),
        (ScalarValue::Date64(left), ScalarValue::Date64(right)) => left.cmp(right),
        (ScalarValue::Timestamp(left), ScalarValue::Timestamp(right)) => left.cmp(right),
        _ => unreachable!("compiled min/max scalars share one type"),
    }
}

fn signed_value(value: &ScalarValue) -> std::result::Result<i64, String> {
    if let ScalarValue::Signed(value) = value {
        Ok(*value)
    } else {
        Err("expected a signed integer scalar".into())
    }
}

fn unsigned_value(value: &ScalarValue) -> std::result::Result<u64, String> {
    if let ScalarValue::Unsigned(value) = value {
        Ok(*value)
    } else {
        Err("expected an unsigned integer scalar".into())
    }
}

fn float_value(value: &ScalarValue) -> std::result::Result<f64, String> {
    match value {
        ScalarValue::Float32(bits) => Ok(f64::from(f32::from_bits(*bits))),
        ScalarValue::Float64(bits) => Ok(f64::from_bits(*bits)),
        _ => Err("expected a floating-point scalar".into()),
    }
}

fn scalar_at(
    array: &dyn Array,
    data_type: &DataType,
    row: usize,
    operator_id: &str,
) -> Result<Option<ScalarValue>> {
    if array.is_null(row) {
        return Ok(None);
    }
    ScalarColumn::new(array, data_type, operator_id).map(|column| Some(column.value(row)))
}

/// One scalar column downcast once, so per-row reads skip dynamic type
/// dispatch. `Opaque` columns answer only validity checks.
#[derive(Clone, Copy)]
struct ScalarColumn<'a> {
    nulls: Option<&'a NullBuffer>,
    values: TypedValues<'a>,
}

#[derive(Clone, Copy)]
enum BorrowedString<'a> {
    Other,
    Value(Option<&'a str>),
}

#[derive(Clone, Copy)]
enum TypedValues<'a> {
    Opaque,
    Boolean(&'a BooleanArray),
    Int8(&'a Int8Array),
    Int16(&'a Int16Array),
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    UInt8(&'a UInt8Array),
    UInt16(&'a UInt16Array),
    UInt32(&'a UInt32Array),
    UInt64(&'a UInt64Array),
    Float32(&'a Float32Array),
    Float64(&'a Float64Array),
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
    Date32(&'a Date32Array),
    Date64(&'a Date64Array),
    Timestamp(&'a TimestampMicrosecondArray),
}

impl<'a> ScalarColumn<'a> {
    /// Distinguish string columns, including their null group, from other types.
    fn borrowed_string(&self, row: usize) -> BorrowedString<'a> {
        let value = match self.values {
            TypedValues::Utf8(array) => (!self.is_null(row)).then(|| array.value(row)),
            TypedValues::LargeUtf8(array) => (!self.is_null(row)).then(|| array.value(row)),
            _ => return BorrowedString::Other,
        };
        BorrowedString::Value(value)
    }

    fn opaque(array: &'a dyn Array) -> Self {
        Self {
            nulls: array.nulls(),
            values: TypedValues::Opaque,
        }
    }

    fn new(array: &'a dyn Array, data_type: &DataType, operator_id: &str) -> Result<Self> {
        Ok(Self {
            nulls: array.nulls(),
            values: TypedValues::new(array, data_type, operator_id)?,
        })
    }

    fn is_null(&self, row: usize) -> bool {
        self.nulls.is_some_and(|nulls| nulls.is_null(row))
    }

    fn scalar_at(&self, row: usize) -> Option<ScalarValue> {
        (!self.is_null(row)).then(|| self.value(row))
    }

    fn value(&self, row: usize) -> ScalarValue {
        self.values.value(row)
    }
}

impl<'a> TypedValues<'a> {
    fn new(array: &'a dyn Array, data_type: &DataType, operator_id: &str) -> Result<Self> {
        match data_type {
            DataType::Boolean => downcast_array(array, operator_id, "Boolean").map(Self::Boolean),
            DataType::Int8 => downcast_array(array, operator_id, "Int8").map(Self::Int8),
            DataType::Int16 => downcast_array(array, operator_id, "Int16").map(Self::Int16),
            DataType::Int32 => downcast_array(array, operator_id, "Int32").map(Self::Int32),
            DataType::Int64 => downcast_array(array, operator_id, "Int64").map(Self::Int64),
            DataType::UInt8 => downcast_array(array, operator_id, "UInt8").map(Self::UInt8),
            DataType::UInt16 => downcast_array(array, operator_id, "UInt16").map(Self::UInt16),
            DataType::UInt32 => downcast_array(array, operator_id, "UInt32").map(Self::UInt32),
            DataType::UInt64 => downcast_array(array, operator_id, "UInt64").map(Self::UInt64),
            DataType::Float32 => downcast_array(array, operator_id, "Float32").map(Self::Float32),
            DataType::Float64 => downcast_array(array, operator_id, "Float64").map(Self::Float64),
            DataType::Utf8 => downcast_array(array, operator_id, "Utf8").map(Self::Utf8),
            DataType::LargeUtf8 => {
                downcast_array(array, operator_id, "LargeUtf8").map(Self::LargeUtf8)
            }
            DataType::Date32 => downcast_array(array, operator_id, "Date32").map(Self::Date32),
            DataType::Date64 => downcast_array(array, operator_id, "Date64").map(Self::Date64),
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                downcast_array(array, operator_id, "Timestamp(Microsecond)").map(Self::Timestamp)
            }
            _ => Err(operator_error(
                operator_id,
                &format!("compiled scalar type {data_type} is unsupported"),
            )),
        }
    }

    fn value(self, row: usize) -> ScalarValue {
        match self {
            Self::Opaque => unreachable!("opaque columns only answer validity checks"),
            Self::Boolean(array) => ScalarValue::Boolean(array.value(row)),
            Self::Int8(array) => ScalarValue::Signed(i64::from(array.value(row))),
            Self::Int16(array) => ScalarValue::Signed(i64::from(array.value(row))),
            Self::Int32(array) => ScalarValue::Signed(i64::from(array.value(row))),
            Self::Int64(array) => ScalarValue::Signed(array.value(row)),
            Self::UInt8(array) => ScalarValue::Unsigned(u64::from(array.value(row))),
            Self::UInt16(array) => ScalarValue::Unsigned(u64::from(array.value(row))),
            Self::UInt32(array) => ScalarValue::Unsigned(u64::from(array.value(row))),
            Self::UInt64(array) => ScalarValue::Unsigned(array.value(row)),
            Self::Float32(array) => ScalarValue::Float32(array.value(row).to_bits()),
            Self::Float64(array) => ScalarValue::Float64(array.value(row).to_bits()),
            Self::Utf8(array) => ScalarValue::String(array.value(row).into()),
            Self::LargeUtf8(array) => ScalarValue::String(array.value(row).into()),
            Self::Date32(array) => ScalarValue::Date32(array.value(row)),
            Self::Date64(array) => ScalarValue::Date64(array.value(row)),
            Self::Timestamp(array) => ScalarValue::Timestamp(array.value(row)),
        }
    }
}

macro_rules! primitive_scalar_array {
    ($values:expr, $array:ty, $pattern:pat => $value:expr) => {{
        let values = $values
            .iter()
            .map(|value| match value {
                None => Ok(None),
                Some($pattern) => Ok(Some($value)),
                Some(_) => Err(internal_error("output scalar type mismatch")),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(<$array>::from(values)) as ArrayRef)
    }};
}

fn scalar_array(
    data_type: &DataType,
    values: &[Option<ScalarValue>],
    operator_id: &str,
) -> Result<ArrayRef> {
    match data_type {
        DataType::Boolean => {
            primitive_scalar_array!(values, BooleanArray, ScalarValue::Boolean(value) => *value)
        }
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            signed_scalar_array(data_type, values)
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            unsigned_scalar_array(data_type, values)
        }
        DataType::Float32 | DataType::Float64 => float_scalar_array(data_type, values),
        DataType::Utf8 | DataType::LargeUtf8 => string_scalar_array(data_type, values),
        DataType::Date32 | DataType::Date64 | DataType::Timestamp(TimeUnit::Microsecond, _) => {
            temporal_scalar_array(data_type, values)
        }
        _ => Err(operator_error(
            operator_id,
            &format!("cannot build window output array for type {data_type}"),
        )),
    }
}

fn signed_scalar_array(data_type: &DataType, values: &[Option<ScalarValue>]) -> Result<ArrayRef> {
    match data_type {
        DataType::Int8 => primitive_scalar_array!(values, Int8Array, ScalarValue::Signed(value) =>
            i8::try_from(*value).map_err(|_| internal_error("Int8 output scalar overflowed"))?),
        DataType::Int16 => {
            primitive_scalar_array!(values, Int16Array, ScalarValue::Signed(value) =>
            i16::try_from(*value).map_err(|_| internal_error("Int16 output scalar overflowed"))?)
        }
        DataType::Int32 => {
            primitive_scalar_array!(values, Int32Array, ScalarValue::Signed(value) =>
            i32::try_from(*value).map_err(|_| internal_error("Int32 output scalar overflowed"))?)
        }
        DataType::Int64 => {
            primitive_scalar_array!(values, Int64Array, ScalarValue::Signed(value) => *value)
        }
        _ => Err(internal_error("compiled signed output type mismatch")),
    }
}

fn unsigned_scalar_array(data_type: &DataType, values: &[Option<ScalarValue>]) -> Result<ArrayRef> {
    match data_type {
        DataType::UInt8 => {
            primitive_scalar_array!(values, UInt8Array, ScalarValue::Unsigned(value) =>
            u8::try_from(*value).map_err(|_| internal_error("UInt8 output scalar overflowed"))?)
        }
        DataType::UInt16 => {
            primitive_scalar_array!(values, UInt16Array, ScalarValue::Unsigned(value) =>
            u16::try_from(*value).map_err(|_| internal_error("UInt16 output scalar overflowed"))?)
        }
        DataType::UInt32 => {
            primitive_scalar_array!(values, UInt32Array, ScalarValue::Unsigned(value) =>
            u32::try_from(*value).map_err(|_| internal_error("UInt32 output scalar overflowed"))?)
        }
        DataType::UInt64 => {
            primitive_scalar_array!(values, UInt64Array, ScalarValue::Unsigned(value) => *value)
        }
        _ => Err(internal_error("compiled unsigned output type mismatch")),
    }
}

fn float_scalar_array(data_type: &DataType, values: &[Option<ScalarValue>]) -> Result<ArrayRef> {
    match data_type {
        DataType::Float32 => {
            primitive_scalar_array!(values, Float32Array, ScalarValue::Float32(value) => f32::from_bits(*value))
        }
        DataType::Float64 => {
            primitive_scalar_array!(values, Float64Array, ScalarValue::Float64(value) => f64::from_bits(*value))
        }
        _ => Err(internal_error("compiled float output type mismatch")),
    }
}

fn string_scalar_array(data_type: &DataType, values: &[Option<ScalarValue>]) -> Result<ArrayRef> {
    let values = values
        .iter()
        .map(|value| match value {
            None => Ok(None),
            Some(ScalarValue::String(value)) => Ok(Some(value.as_str())),
            Some(_) => Err(internal_error("string output scalar type mismatch")),
        })
        .collect::<Result<Vec<_>>>()?;
    match data_type {
        DataType::Utf8 => Ok(Arc::new(StringArray::from(values))),
        DataType::LargeUtf8 => Ok(Arc::new(LargeStringArray::from(values))),
        _ => Err(internal_error("compiled string output type mismatch")),
    }
}

fn temporal_scalar_array(data_type: &DataType, values: &[Option<ScalarValue>]) -> Result<ArrayRef> {
    match data_type {
        DataType::Date32 => {
            primitive_scalar_array!(values, Date32Array, ScalarValue::Date32(value) => *value)
        }
        DataType::Date64 => {
            primitive_scalar_array!(values, Date64Array, ScalarValue::Date64(value) => *value)
        }
        DataType::Timestamp(TimeUnit::Microsecond, timezone) => {
            timestamp_scalar_array(values, timezone.clone())
        }
        _ => Err(internal_error("compiled temporal output type mismatch")),
    }
}

fn timestamp_scalar_array(
    values: &[Option<ScalarValue>],
    timezone: Option<Arc<str>>,
) -> Result<ArrayRef> {
    let values = values
        .iter()
        .map(|value| match value {
            None => Ok(None),
            Some(ScalarValue::Timestamp(value)) => Ok(Some(*value)),
            Some(_) => Err(internal_error("timestamp output scalar type mismatch")),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(
        TimestampMicrosecondArray::from(values).with_timezone_opt(timezone),
    ))
}

fn downcast_array<'a, T: 'static>(
    array: &'a dyn Array,
    operator_id: &str,
    expected: &str,
) -> Result<&'a T> {
    array.as_any().downcast_ref::<T>().ok_or_else(|| {
        operator_error(
            operator_id,
            &format!("compiled {expected} column has a different physical array"),
        )
    })
}

fn operator_error(node_id: &str, message: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node_id.into(),
        message: message.into(),
    }
}

fn compile_spec(
    input_schema: &Schema,
    spec: &WindowSpec,
    configuration: &JsonMap,
) -> Result<CompiledWindowSpec> {
    let event_time_index = exact_field_index(input_schema, &spec.event_time_column)?;
    validate_event_time_type(
        input_schema.field(event_time_index).data_type(),
        &spec.event_time_column,
    )?;
    let group_columns = compile_group_columns(input_schema, spec)?;
    let aggregates = compile_aggregates(input_schema, spec)?;
    let geometry = compile_geometry(spec.geometry);
    let canonical = canonical_json(&Value::Object(configuration.clone().into_iter().collect()))?;
    let mut compiled = CompiledWindowSpec {
        event_time_index,
        group_columns,
        aggregates,
        geometry,
        configuration_hash: hex::encode(Sha256::digest(canonical.as_bytes())),
        state_schema_fingerprint: String::new(),
    };
    compiled.state_schema_fingerprint = state_schema_fingerprint(spec, &compiled);
    Ok(compiled)
}

fn compile_group_columns(
    input_schema: &Schema,
    spec: &WindowSpec,
) -> Result<Vec<CompiledGroupColumn>> {
    spec.group_by
        .iter()
        .map(|column| {
            let index = exact_field_index(input_schema, column)?;
            let data_type = input_schema.field(index).data_type().clone();
            if !supports_group_type(&data_type) {
                return Err(compile_error(format!(
                    "window group column {column:?} has unsupported type {data_type}"
                )));
            }
            Ok(CompiledGroupColumn { index, data_type })
        })
        .collect()
}

fn compile_aggregates(input_schema: &Schema, spec: &WindowSpec) -> Result<Vec<CompiledAggregate>> {
    spec.aggregates
        .iter()
        .map(|aggregate| {
            let input_index = exact_field_index(input_schema, &aggregate.column)?;
            let input_type = input_schema.field(input_index).data_type().clone();
            let output_type =
                aggregate_output_type(aggregate.function, &input_type).ok_or_else(|| {
                    compile_error(format!(
                        "window aggregate {:?} does not support column {:?} with type {input_type}",
                        aggregate.function, aggregate.column
                    ))
                })?;
            Ok(CompiledAggregate {
                input_index,
                input_type,
                output_type,
            })
        })
        .collect()
}

fn compile_geometry(geometry: WindowGeometry) -> CompiledWindowGeometry {
    let (size_micros, slide_micros) = match geometry {
        WindowGeometry::Tumbling { size_micros } => (size_micros, size_micros),
        WindowGeometry::Hopping {
            size_micros,
            slide_micros,
        } => (size_micros, slide_micros),
    };
    CompiledWindowGeometry {
        size_micros,
        slide_micros,
        overlap: size_micros / slide_micros,
    }
}

fn state_schema_fingerprint(spec: &WindowSpec, compiled: &CompiledWindowSpec) -> String {
    state_schema_fingerprint_for_version(spec, compiled, WINDOW_SEGMENT_LAYOUT_VERSION)
}

fn state_schema_fingerprint_for_version(
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    version: u32,
) -> String {
    let schema = Schema::new(state_fields_for_version(spec, compiled, version));
    let mut dictionary_tracker = DictionaryTracker::new(true);
    let encoded = IpcSchemaEncoder::new()
        .with_dictionary_tracker(&mut dictionary_tracker)
        .schema_to_fb(&schema);
    hex::encode(Sha256::digest(encoded.finished_data()))
}

fn state_schema(
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    pipeline_fingerprint: &str,
    operator_id: &str,
) -> Schema {
    state_schema_for_version(
        spec,
        compiled,
        pipeline_fingerprint,
        operator_id,
        WINDOW_SEGMENT_LAYOUT_VERSION,
    )
}

fn state_schema_for_version(
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    pipeline_fingerprint: &str,
    operator_id: &str,
    version: u32,
) -> Schema {
    let schema_fingerprint = if version == WINDOW_SEGMENT_LAYOUT_VERSION {
        compiled.state_schema_fingerprint.clone()
    } else {
        state_schema_fingerprint_for_version(spec, compiled, version)
    };
    Schema::new_with_metadata(
        state_fields_for_version(spec, compiled, version),
        HashMap::from([
            ("calc_flow.state_layout_version".into(), version.to_string()),
            (
                "calc_flow.pipeline_fingerprint".into(),
                pipeline_fingerprint.into(),
            ),
            ("calc_flow.operator_id".into(), operator_id.into()),
            (
                "calc_flow.operator_configuration_hash".into(),
                compiled.configuration_hash.clone(),
            ),
            (
                "calc_flow.state_schema_fingerprint".into(),
                schema_fingerprint,
            ),
            ("calc_flow.group_key_encoding".into(), "g1".into()),
        ]),
    )
}

fn state_fields_for_version(
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    version: u32,
) -> Vec<Field> {
    let utc_timestamp = DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC")));
    let mut fields = vec![
        Field::new("_operation", DataType::UInt8, false),
        Field::new("window_start", utc_timestamp.clone(), false),
        Field::new("window_end", utc_timestamp, false),
        Field::new("_stable_group_key", DataType::LargeBinary, false),
    ];
    fields.extend(
        spec.group_by
            .iter()
            .zip(&compiled.group_columns)
            .map(|(name, column)| Field::new(name, column.data_type.clone(), true)),
    );
    for (ordinal, (aggregate, compiled_aggregate)) in
        spec.aggregates.iter().zip(&compiled.aggregates).enumerate()
    {
        let value_name = format!("_agg_{ordinal:04}_value");
        match aggregate.function {
            AggregateFunction::Count => {
                fields.push(Field::new(value_name, DataType::UInt64, true));
            }
            AggregateFunction::Sum => {
                fields.push(Field::new(
                    value_name,
                    compiled_aggregate.output_type.clone(),
                    true,
                ));
                if version >= WINDOW_SEGMENT_LAYOUT_VERSION
                    && compiled_aggregate.output_type == DataType::Float64
                {
                    fields.push(Field::new(
                        format!("_agg_{ordinal:04}_correction"),
                        DataType::Float64,
                        true,
                    ));
                }
            }
            AggregateFunction::Min | AggregateFunction::Max => {
                fields.push(Field::new(
                    value_name,
                    compiled_aggregate.input_type.clone(),
                    true,
                ));
            }
            AggregateFunction::Avg => {
                let state_type = match compiled_aggregate.input_type {
                    DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64 => DataType::FixedSizeBinary(16),
                    DataType::Float32 | DataType::Float64 => DataType::Float64,
                    _ => unreachable!("average input matrix validated at construction"),
                };
                fields.push(Field::new(value_name, state_type, true));
                fields.push(Field::new(
                    format!("_agg_{ordinal:04}_count"),
                    DataType::UInt64,
                    true,
                ));
                if version >= WINDOW_SEGMENT_LAYOUT_VERSION
                    && matches!(
                        compiled_aggregate.input_type,
                        DataType::Float32 | DataType::Float64
                    )
                {
                    fields.push(Field::new(
                        format!("_agg_{ordinal:04}_correction"),
                        DataType::Float64,
                        true,
                    ));
                }
            }
        }
    }
    fields
}

fn encode_state_segment(
    operations: &[StateOperationRow],
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    pipeline_fingerprint: &str,
    operator_id: &str,
) -> Result<Vec<u8>> {
    validate_state_operations(operations)?;
    let schema = state_schema(spec, compiled, pipeline_fingerprint, operator_id);
    let mut arrays = state_key_arrays(operations);
    append_group_state_arrays(&mut arrays, operations, compiled, operator_id)?;
    append_aggregate_state_arrays(&mut arrays, operations, spec, compiled, operator_id)?;
    write_state_ipc(&schema, arrays)
}

fn validate_state_operations(operations: &[StateOperationRow]) -> Result<()> {
    if operations.is_empty() {
        return Err(internal_error(
            "cannot encode an empty window state segment",
        ));
    }
    if operations.windows(2).any(|pair| pair[0].key >= pair[1].key) {
        return Err(internal_error(
            "window state operations are not in strict key order",
        ));
    }
    Ok(())
}

fn state_key_arrays(operations: &[StateOperationRow]) -> Vec<ArrayRef> {
    vec![
        Arc::new(UInt8Array::from(
            operations
                .iter()
                .map(|row| u8::from(row.tombstone))
                .collect::<Vec<_>>(),
        )),
        Arc::new(
            TimestampMicrosecondArray::from(
                operations
                    .iter()
                    .map(|row| row.key.start.as_micros())
                    .collect::<Vec<_>>(),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(
            TimestampMicrosecondArray::from(
                operations
                    .iter()
                    .map(|row| row.key.end.as_micros())
                    .collect::<Vec<_>>(),
            )
            .with_timezone("UTC"),
        ),
        Arc::new(LargeBinaryArray::from_iter_values(
            operations.iter().map(|row| &*row.key.stable_group_key),
        )),
    ]
}

fn append_group_state_arrays(
    arrays: &mut Vec<ArrayRef>,
    operations: &[StateOperationRow],
    compiled: &CompiledWindowSpec,
    operator_id: &str,
) -> Result<()> {
    for (ordinal, group) in compiled.group_columns.iter().enumerate() {
        let values = operations
            .iter()
            .map(|row| row.entry.group_values[ordinal].clone())
            .collect::<Vec<_>>();
        arrays.push(scalar_array(&group.data_type, &values, operator_id)?);
    }
    Ok(())
}

fn append_aggregate_state_arrays(
    arrays: &mut Vec<ArrayRef>,
    operations: &[StateOperationRow],
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    operator_id: &str,
) -> Result<()> {
    for (ordinal, (aggregate, compiled_aggregate)) in
        spec.aggregates.iter().zip(&compiled.aggregates).enumerate()
    {
        append_accumulator_state_array(
            arrays,
            operations,
            ordinal,
            aggregate.function,
            compiled_aggregate,
            operator_id,
        )?;
    }
    Ok(())
}

fn write_state_ipc(schema: &Schema, arrays: Vec<ArrayRef>) -> Result<Vec<u8>> {
    let record = RecordBatch::try_new(Arc::new(schema.clone()), arrays)
        .map_err(|error| state_format(format!("window state batch is invalid: {error}")))?;
    let mut bytes = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut bytes, schema)
            .map_err(|error| state_format(format!("window state IPC header failed: {error}")))?;
        writer
            .write(&record)
            .map_err(|error| state_format(format!("window state IPC write failed: {error}")))?;
        writer
            .finish()
            .map_err(|error| state_format(format!("window state IPC finish failed: {error}")))?;
    }
    Ok(bytes)
}

fn append_accumulator_state_array(
    arrays: &mut Vec<ArrayRef>,
    operations: &[StateOperationRow],
    ordinal: usize,
    function: AggregateFunction,
    compiled: &CompiledAggregate,
    operator_id: &str,
) -> Result<()> {
    match function {
        AggregateFunction::Count
        | AggregateFunction::Sum
        | AggregateFunction::Min
        | AggregateFunction::Max => {
            let state_type = match function {
                AggregateFunction::Count => &DataType::UInt64,
                AggregateFunction::Sum => &compiled.output_type,
                AggregateFunction::Min | AggregateFunction::Max => &compiled.input_type,
                AggregateFunction::Avg => unreachable!(),
            };
            let values = operations
                .iter()
                .map(|row| {
                    if row.tombstone {
                        Ok(None)
                    } else {
                        accumulator_state_scalar(&row.entry.aggregates[ordinal])
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            arrays.push(scalar_array(state_type, &values, operator_id)?);
            if function == AggregateFunction::Sum && compiled.output_type == DataType::Float64 {
                arrays.push(float_correction_state_array(
                    operations,
                    ordinal,
                    operator_id,
                )?);
            }
        }
        AggregateFunction::Avg => append_average_state_arrays(
            arrays,
            operations,
            ordinal,
            &compiled.input_type,
            operator_id,
        )?,
    }
    Ok(())
}

fn append_average_state_arrays(
    arrays: &mut Vec<ArrayRef>,
    operations: &[StateOperationRow],
    ordinal: usize,
    input_type: &DataType,
    operator_id: &str,
) -> Result<()> {
    match input_type {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            arrays.push(average_state_array(
                operations,
                ordinal,
                AverageStateKind::Signed,
            )?);
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            arrays.push(average_state_array(
                operations,
                ordinal,
                AverageStateKind::Unsigned,
            )?);
        }
        DataType::Float32 | DataType::Float64 => {
            arrays.push(float_average_state_array(operations, ordinal, operator_id)?);
        }
        _ => unreachable!("average input matrix validated at construction"),
    }
    arrays.push(average_count_array(operations, ordinal)?);
    if matches!(input_type, DataType::Float32 | DataType::Float64) {
        arrays.push(float_correction_state_array(
            operations,
            ordinal,
            operator_id,
        )?);
    }
    Ok(())
}

/// Which fixed-width average family a state column carries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AverageStateKind {
    Signed,
    Unsigned,
}

impl AverageStateKind {
    fn label(self) -> &'static str {
        match self {
            Self::Signed => "signed",
            Self::Unsigned => "unsigned",
        }
    }
}

fn average_state_array(
    operations: &[StateOperationRow],
    ordinal: usize,
    kind: AverageStateKind,
) -> Result<ArrayRef> {
    let values = operations
        .iter()
        .map(
            |row| match (&row.entry.aggregates[ordinal], row.tombstone) {
                (_, true) => Ok(None),
                (AccumulatorValue::SignedAverage { sum, .. }, false)
                    if kind == AverageStateKind::Signed =>
                {
                    Ok(Some(sum.to_be_bytes()))
                }
                (AccumulatorValue::UnsignedAverage { sum, .. }, false)
                    if kind == AverageStateKind::Unsigned =>
                {
                    Ok(Some(sum.to_be_bytes()))
                }
                _ => Err(internal_error(format!(
                    "{} average state type mismatch",
                    kind.label()
                ))),
            },
        )
        .collect::<Result<Vec<_>>>()?;
    fixed_average_state_array(values, kind.label())
}

fn fixed_average_state_array(values: Vec<Option<[u8; 16]>>, label: &str) -> Result<ArrayRef> {
    FixedSizeBinaryArray::try_from_sparse_iter_with_size(values.into_iter(), 16)
        .map(|array| Arc::new(array) as ArrayRef)
        .map_err(|error| state_format(format!("{label} average state array failed: {error}")))
}

fn float_average_state_array(
    operations: &[StateOperationRow],
    ordinal: usize,
    operator_id: &str,
) -> Result<ArrayRef> {
    let values = operations
        .iter()
        .map(
            |row| match (&row.entry.aggregates[ordinal], row.tombstone) {
                (_, true) => Ok(None),
                (AccumulatorValue::FloatAverage { sum, .. }, false) => {
                    Ok(Some(ScalarValue::Float64(sum.sum.to_bits())))
                }
                _ => Err(internal_error("float average state type mismatch")),
            },
        )
        .collect::<Result<Vec<_>>>()?;
    scalar_array(&DataType::Float64, &values, operator_id)
}

fn float_correction_state_array(
    operations: &[StateOperationRow],
    ordinal: usize,
    operator_id: &str,
) -> Result<ArrayRef> {
    let values = operations
        .iter()
        .map(|row| {
            if row.tombstone {
                return Ok(None);
            }
            match &row.entry.aggregates[ordinal] {
                AccumulatorValue::FloatSum(Some(sum))
                | AccumulatorValue::FloatAverage { sum, .. } => {
                    Ok(Some(ScalarValue::Float64(sum.correction.to_bits())))
                }
                AccumulatorValue::FloatSum(None) => Ok(None),
                _ => Err(internal_error("float correction state type mismatch")),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    scalar_array(&DataType::Float64, &values, operator_id)
}

fn accumulator_state_scalar(accumulator: &AccumulatorValue) -> Result<Option<ScalarValue>> {
    match accumulator {
        AccumulatorValue::Count(value) => Ok(Some(ScalarValue::Unsigned(*value))),
        AccumulatorValue::SignedSum(value) => value
            .map(|value| {
                i64::try_from(value)
                    .map(ScalarValue::Signed)
                    .map_err(|_| internal_error("signed sum state escaped Int64"))
            })
            .transpose(),
        AccumulatorValue::UnsignedSum(value) => value
            .map(|value| {
                u64::try_from(value)
                    .map(ScalarValue::Unsigned)
                    .map_err(|_| internal_error("unsigned sum state escaped UInt64"))
            })
            .transpose(),
        AccumulatorValue::FloatSum(value) => {
            Ok(value.map(|value| ScalarValue::Float64(value.sum.to_bits())))
        }
        AccumulatorValue::Min(value) | AccumulatorValue::Max(value) => Ok(value.clone()),
        AccumulatorValue::SignedAverage { .. }
        | AccumulatorValue::UnsignedAverage { .. }
        | AccumulatorValue::FloatAverage { .. } => Err(internal_error(
            "average state requires two physical columns",
        )),
    }
}

fn average_count_array(operations: &[StateOperationRow], ordinal: usize) -> Result<ArrayRef> {
    let values = operations
        .iter()
        .map(|row| {
            if row.tombstone {
                return Ok(None);
            }
            match &row.entry.aggregates[ordinal] {
                AccumulatorValue::SignedAverage { count, .. }
                | AccumulatorValue::UnsignedAverage { count, .. }
                | AccumulatorValue::FloatAverage { count, .. } => Ok(Some(*count)),
                _ => Err(internal_error("average count state type mismatch")),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(UInt64Array::from(values)))
}

fn validate_snapshot_metadata(
    metadata: &WindowSnapshotMetadata,
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<StateInventory> {
    validate_snapshot_header(metadata, spec, compiled)?;
    let inventory = StateInventory::new(metadata.segment_inventory.clone())
        .map_err(|error| checkpoint_mismatch(error.to_string()))?;
    validate_snapshot_inventory(metadata, spec, compiled, &inventory)?;
    validate_snapshot_segment_set(snapshot, &inventory)?;
    validate_snapshot_identity(metadata, snapshot)?;
    Ok(inventory)
}

fn validate_snapshot_header(
    metadata: &WindowSnapshotMetadata,
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
) -> Result<()> {
    if !(WINDOW_STATE_LAYOUT_VERSION..=WINDOW_SEGMENT_LAYOUT_VERSION)
        .contains(&metadata.state_layout_version)
    {
        return Err(checkpoint_mismatch(format!(
            "window state layout version {} is unsupported",
            metadata.state_layout_version
        )));
    }
    if metadata.configuration_hash != compiled.configuration_hash {
        return Err(checkpoint_mismatch(
            "window operator configuration hash does not match the compiled operator",
        ));
    }
    let expected_fingerprint =
        state_schema_fingerprint_for_version(spec, compiled, metadata.state_layout_version);
    if metadata.state_schema_fingerprint != expected_fingerprint {
        return Err(checkpoint_mismatch(
            "window state schema fingerprint does not match the compiled operator",
        ));
    }
    Ok(())
}

fn validate_snapshot_segment_set(
    snapshot: &crate::OperatorStateSnapshot,
    inventory: &StateInventory,
) -> Result<()> {
    let expected_ids = inventory
        .segments()
        .iter()
        .map(|descriptor| descriptor.handle.segment_id().to_owned())
        .collect::<Vec<_>>();
    let actual_ids = snapshot.segments.keys().cloned().collect::<Vec<_>>();
    if expected_ids != actual_ids {
        return Err(checkpoint_mismatch(
            "window snapshot segment IDs are missing, extra, duplicated, or non-canonical",
        ));
    }
    Ok(())
}

fn validate_snapshot_identity(
    metadata: &WindowSnapshotMetadata,
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<()> {
    if !snapshot.segments.is_empty()
        && (metadata.pipeline_fingerprint.is_none() || metadata.operator_id.is_none())
    {
        return Err(checkpoint_mismatch(
            "window segments require pipeline and operator identity metadata",
        ));
    }
    validate_snapshot_pipeline_fingerprint(metadata.pipeline_fingerprint.as_deref())?;
    validate_snapshot_operator_id(metadata.operator_id.as_deref())
}

fn validate_snapshot_pipeline_fingerprint(fingerprint: Option<&str>) -> Result<()> {
    if let Some(fingerprint) = fingerprint
        && (fingerprint.len() != 64
            || !fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    {
        return Err(checkpoint_mismatch(
            "window pipeline fingerprint is not lowercase SHA-256",
        ));
    }
    Ok(())
}

fn validate_snapshot_operator_id(operator_id: Option<&str>) -> Result<()> {
    if operator_id.is_some_and(|operator_id| operator_id.is_empty() || operator_id.contains('\0')) {
        return Err(checkpoint_mismatch(
            "window operator ID is empty or contains NUL",
        ));
    }
    Ok(())
}

fn validate_snapshot_inventory(
    metadata: &WindowSnapshotMetadata,
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    inventory: &StateInventory,
) -> Result<()> {
    for descriptor in inventory.segments() {
        let version = descriptor.state_layout_version;
        if version != metadata.state_layout_version
            || descriptor.schema_fingerprint
                != state_schema_fingerprint_for_version(spec, compiled, version)
        {
            return Err(checkpoint_mismatch(
                "window segment inventory layout or schema does not match the compiled operator",
            ));
        }
        if descriptor.handle.epoch() > metadata.epoch {
            return Err(checkpoint_mismatch(
                "window segment inventory contains a future epoch",
            ));
        }
        if metadata.operator_id.as_deref() != Some(descriptor.handle.operator_id()) {
            return Err(checkpoint_mismatch(
                "window segment inventory operator does not match snapshot metadata",
            ));
        }
    }
    Ok(())
}

fn decode_state_segments(
    segments: &[(u32, Arc<Vec<u8>>)],
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    pipeline_fingerprint: Option<&str>,
    operator_id: Option<&str>,
) -> Result<BTreeMap<WindowKey, AccumulatorRow>> {
    if segments.is_empty() {
        return Ok(BTreeMap::new());
    }
    let pipeline_fingerprint = pipeline_fingerprint
        .ok_or_else(|| checkpoint_mismatch("window state is missing its pipeline fingerprint"))?;
    let operator_id = operator_id
        .ok_or_else(|| checkpoint_mismatch("window state is missing its operator ID"))?;
    let decoded = segments
        .iter()
        .map(|(version, bytes)| {
            let expected_schema = state_schema_for_version(
                spec,
                compiled,
                pipeline_fingerprint,
                operator_id,
                *version,
            );
            decode_state_segment(
                bytes,
                spec,
                compiled,
                &expected_schema,
                operator_id,
                *version,
            )
            .map(|operations| {
                operations
                    .into_iter()
                    .map(|(key, entry)| {
                        let operation =
                            entry.map_or(StateOperation::Tombstone, StateOperation::Upsert);
                        (key, operation)
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Result<Vec<_>>>()?;
    fold_state_segments(decoded)
}

#[allow(
    clippy::too_many_lines,
    reason = "durable state validation remains a single fail-before-install decoding transaction"
)]
fn decode_state_segment(
    bytes: &[u8],
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    expected_schema: &Schema,
    operator_id: &str,
    version: u32,
) -> Result<Vec<(WindowKey, Option<AccumulatorRow>)>> {
    if !bytes.starts_with(b"ARROW1") || !bytes.ends_with(b"ARROW1") {
        return Err(state_format(
            "window state segment is missing the Arrow IPC file magic",
        ));
    }
    let mut reader = FileReader::try_new(Cursor::new(bytes), None)
        .map_err(|error| state_format(format!("window state IPC is invalid: {error}")))?;
    if reader.schema().as_ref() != expected_schema {
        return Err(checkpoint_mismatch(
            "window state Arrow schema or metadata does not match the compiled operator",
        ));
    }
    if reader.num_batches() != 1 {
        return Err(state_format(format!(
            "window state segment must contain exactly one record batch, found {}",
            reader.num_batches()
        )));
    }
    let record = reader
        .next()
        .ok_or_else(|| state_format("window state segment has no record batch"))?
        .map_err(|error| state_format(format!("window state batch is invalid: {error}")))?;
    if record.num_rows() == 0 {
        return Err(state_format(
            "window state segment must contain at least one operation",
        ));
    }

    let operations = state_array::<UInt8Array>(&record, 0, "_operation")?;
    let starts = state_array::<TimestampMicrosecondArray>(&record, 1, "window_start")?;
    let ends = state_array::<TimestampMicrosecondArray>(&record, 2, "window_end")?;
    let stable_keys = state_array::<LargeBinaryArray>(&record, 3, "_stable_group_key")?;
    let mut decoded = Vec::with_capacity(record.num_rows());
    let mut previous_key = None::<WindowKey>;

    for row in 0..record.num_rows() {
        if operations.is_null(row)
            || starts.is_null(row)
            || ends.is_null(row)
            || stable_keys.is_null(row)
        {
            return Err(state_format(
                "window state operation and key columns must not be null",
            ));
        }
        let tombstone = match operations.value(row) {
            0 => false,
            1 => true,
            value => {
                return Err(state_format(format!(
                    "window state operation {value} is not 0 or 1"
                )));
            }
        };
        let key = WindowKey {
            start: EventTime::from_micros(starts.value(row)),
            end: EventTime::from_micros(ends.value(row)),
            stable_group_key: stable_keys.value(row).into(),
        };
        validate_restored_window_key(&key, compiled)?;
        if previous_key
            .as_ref()
            .is_some_and(|previous| previous >= &key)
        {
            return Err(state_format(
                "window state rows are not in strict key order or contain a duplicate key",
            ));
        }
        previous_key = Some(key.clone());

        let mut group_values = Vec::with_capacity(compiled.group_columns.len());
        for (ordinal, group) in compiled.group_columns.iter().enumerate() {
            group_values.push(
                scalar_at(
                    record.column(4 + ordinal).as_ref(),
                    &group.data_type,
                    row,
                    operator_id,
                )
                .map_err(|error| state_format(error.to_string()))?,
            );
        }
        let encoded_group = encode_group_values(&group_values, compiled)?;
        if *encoded_group != *key.stable_group_key {
            return Err(checkpoint_mismatch(
                "window state stable group key does not match its declared group values",
            ));
        }

        let mut column_index = 4 + compiled.group_columns.len();
        let mut accumulators = Vec::with_capacity(compiled.aggregates.len());
        for (ordinal, (aggregate, compiled_aggregate)) in
            spec.aggregates.iter().zip(&compiled.aggregates).enumerate()
        {
            let (accumulator, next_column) = decode_accumulator_state(
                &record,
                row,
                column_index,
                tombstone,
                aggregate.function,
                compiled_aggregate,
                operator_id,
                ordinal,
                version,
            )?;
            column_index = next_column;
            if let Some(accumulator) = accumulator {
                accumulators.push(accumulator);
            }
        }
        decoded.push((
            key,
            (!tombstone).then_some(AccumulatorRow {
                group_values,
                aggregates: accumulators,
            }),
        ));
    }
    Ok(decoded)
}

#[allow(
    clippy::too_many_arguments,
    reason = "state decoding names every durable aggregate coordinate explicitly"
)]
fn decode_accumulator_state(
    record: &RecordBatch,
    row: usize,
    column_index: usize,
    tombstone: bool,
    function: AggregateFunction,
    compiled: &CompiledAggregate,
    operator_id: &str,
    ordinal: usize,
    version: u32,
) -> Result<(Option<AccumulatorValue>, usize)> {
    let value = record.column(column_index);
    let compensated = version >= WINDOW_SEGMENT_LAYOUT_VERSION
        && ((function == AggregateFunction::Sum && compiled.output_type == DataType::Float64)
            || (function == AggregateFunction::Avg
                && matches!(compiled.input_type, DataType::Float32 | DataType::Float64)));
    let width = 1 + usize::from(function == AggregateFunction::Avg) + usize::from(compensated);
    if tombstone {
        if (0..width).any(|offset| !record.column(column_index + offset).is_null(row)) {
            return Err(state_format(format!(
                "window tombstone aggregate {ordinal} contains state"
            )));
        }
        return Ok((None, column_index + width));
    }

    let decoded = match function {
        AggregateFunction::Count => {
            let Some(ScalarValue::Unsigned(value)) =
                scalar_at(value.as_ref(), &DataType::UInt64, row, operator_id)
                    .map_err(|error| state_format(error.to_string()))?
            else {
                return Err(state_format(format!(
                    "window count aggregate {ordinal} has null or invalid state"
                )));
            };
            AccumulatorValue::Count(value)
        }
        AggregateFunction::Sum => match compiled.output_type {
            DataType::Int64 => AccumulatorValue::SignedSum(
                scalar_at(value.as_ref(), &DataType::Int64, row, operator_id)
                    .map_err(|error| state_format(error.to_string()))?
                    .map(|value| signed_value(&value).map(i128::from))
                    .transpose()
                    .map_err(state_format)?,
            ),
            DataType::UInt64 => AccumulatorValue::UnsignedSum(
                scalar_at(value.as_ref(), &DataType::UInt64, row, operator_id)
                    .map_err(|error| state_format(error.to_string()))?
                    .map(|value| unsigned_value(&value).map(u128::from))
                    .transpose()
                    .map_err(state_format)?,
            ),
            DataType::Float64 => {
                let sum = scalar_at(value.as_ref(), &DataType::Float64, row, operator_id)
                    .map_err(|error| state_format(error.to_string()))?
                    .map(|value| float_value(&value))
                    .transpose()
                    .map_err(state_format)?;
                let correction = if compensated {
                    match sum {
                        Some(_) => float_compensation_at(
                            record,
                            row,
                            column_index + 1,
                            operator_id,
                            ordinal,
                        )?,
                        None if record.column(column_index + 1).is_null(row) => 0.0,
                        None => {
                            return Err(state_format(format!(
                                "window float sum aggregate {ordinal} has correction without sum"
                            )));
                        }
                    }
                } else {
                    0.0
                };
                AccumulatorValue::FloatSum(sum.map(|sum| CompensatedSum { sum, correction }))
            }
            _ => unreachable!("sum output matrix validated at construction"),
        },
        AggregateFunction::Min | AggregateFunction::Max => {
            let scalar = scalar_at(value.as_ref(), &compiled.input_type, row, operator_id)
                .map_err(|error| state_format(error.to_string()))?;
            if function == AggregateFunction::Min {
                AccumulatorValue::Min(scalar)
            } else {
                AccumulatorValue::Max(scalar)
            }
        }
        AggregateFunction::Avg => decode_average_state(
            record,
            value.as_ref(),
            row,
            column_index,
            &compiled.input_type,
            operator_id,
            ordinal,
            version,
        )?,
    };
    Ok((Some(decoded), column_index + width))
}

fn float_compensation_at(
    record: &RecordBatch,
    row: usize,
    column_index: usize,
    operator_id: &str,
    ordinal: usize,
) -> Result<f64> {
    let value = scalar_at(
        record.column(column_index),
        &DataType::Float64,
        row,
        operator_id,
    )
    .map_err(|error| state_format(error.to_string()))?
    .ok_or_else(|| {
        state_format(format!(
            "window float aggregate {ordinal} has null correction"
        ))
    })?;
    float_value(&value).map_err(state_format)
}

/// Decode one average aggregate's sum column and its adjacent count column.
#[allow(clippy::too_many_arguments, reason = "names every durable coordinate")]
fn decode_average_state(
    record: &RecordBatch,
    value: &dyn Array,
    row: usize,
    column_index: usize,
    input_type: &DataType,
    operator_id: &str,
    ordinal: usize,
    version: u32,
) -> Result<AccumulatorValue> {
    let count_array = state_array::<UInt64Array>(
        record,
        column_index + 1,
        &format!("_agg_{ordinal:04}_count"),
    )?;
    if value.is_null(row) || count_array.is_null(row) {
        return Err(state_format(format!(
            "window average aggregate {ordinal} has null state"
        )));
    }
    let count = count_array.value(row);
    Ok(match input_type {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            let bytes = fixed_width_average_bytes(value, row, ordinal, "signed")?;
            AccumulatorValue::SignedAverage {
                sum: i128::from_be_bytes(bytes),
                count,
            }
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            let bytes = fixed_width_average_bytes(value, row, ordinal, "unsigned")?;
            AccumulatorValue::UnsignedAverage {
                sum: u128::from_be_bytes(bytes),
                count,
            }
        }
        DataType::Float32 | DataType::Float64 => {
            let Some(ScalarValue::Float64(bits)) =
                scalar_at(value, &DataType::Float64, row, operator_id)
                    .map_err(|error| state_format(error.to_string()))?
            else {
                return Err(state_format(format!(
                    "window average aggregate {ordinal} has invalid float state"
                )));
            };
            AccumulatorValue::FloatAverage {
                sum: CompensatedSum {
                    sum: f64::from_bits(bits),
                    correction: if version >= WINDOW_SEGMENT_LAYOUT_VERSION {
                        float_compensation_at(record, row, column_index + 2, operator_id, ordinal)?
                    } else {
                        0.0
                    },
                },
                count,
            }
        }
        _ => unreachable!("average input matrix validated at construction"),
    })
}

/// Read one 16-byte big-endian average sum from a fixed-size binary column.
fn fixed_width_average_bytes(
    value: &dyn Array,
    row: usize,
    ordinal: usize,
    kind: &str,
) -> Result<[u8; 16]> {
    value
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or_else(|| {
            state_format(format!(
                "window average aggregate {ordinal} has invalid binary state"
            ))
        })?
        .value(row)
        .try_into()
        .map_err(|_| state_format(format!("{kind} average state is not 16 bytes")))
}

fn validate_restored_window_key(key: &WindowKey, compiled: &CompiledWindowSpec) -> Result<()> {
    if key.stable_group_key.len() > MAX_GROUP_KEY_BYTES {
        return Err(state_format(
            "window state stable group key exceeds the 64-KiB bound",
        ));
    }
    let start = i128::from(key.start.as_micros());
    let end = i128::from(key.end.as_micros());
    if end - start != i128::from(compiled.geometry.size_micros)
        || start.rem_euclid(i128::from(compiled.geometry.slide_micros)) != 0
    {
        return Err(checkpoint_mismatch(
            "window state key does not match the compiled geometry",
        ));
    }
    Ok(())
}

fn encode_group_values(
    values: &[Option<ScalarValue>],
    compiled: &CompiledWindowSpec,
) -> Result<Vec<u8>> {
    if values.len() != compiled.group_columns.len() {
        return Err(state_format(
            "window state group value count does not match its schema",
        ));
    }
    let mut encoded = Vec::new();
    for (value, column) in values.iter().zip(&compiled.group_columns) {
        encode_group_scalar(&mut encoded, &column.data_type, value.as_ref())
            .map_err(state_format)?;
    }
    Ok(encoded)
}

fn state_array<'a, T: 'static>(record: &'a RecordBatch, index: usize, name: &str) -> Result<&'a T> {
    record
        .column(index)
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| state_format(format!("window state column {name:?} has invalid type")))
}

fn output_schema(
    input_schema: &Schema,
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
) -> SchemaRef {
    let utc_timestamp = DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC")));
    let mut fields = vec![
        Field::new("window_start", utc_timestamp.clone(), false),
        Field::new("window_end", utc_timestamp, false),
    ];
    fields.extend(
        compiled
            .group_columns
            .iter()
            .map(|column| input_schema.field(column.index).clone()),
    );
    fields.extend(
        spec.aggregates
            .iter()
            .zip(&compiled.aggregates)
            .map(|(aggregate, compiled)| {
                Field::new(
                    &aggregate.output,
                    compiled.output_type.clone(),
                    aggregate.function != AggregateFunction::Count,
                )
            }),
    );
    Arc::new(Schema::new(fields))
}

fn exact_field_index(schema: &Schema, column: &str) -> Result<usize> {
    let matches = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| field.name() == column)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => Err(compile_error(format!(
            "window column {column:?} does not exist in the input schema"
        ))),
        _ => Err(compile_error(format!(
            "window column {column:?} is ambiguous in the input schema"
        ))),
    }
}

fn validate_event_time_type(data_type: &DataType, column: &str) -> Result<()> {
    let DataType::Timestamp(_, timezone) = data_type else {
        return Err(compile_error(format!(
            "window event-time column {column:?} must be an Arrow timestamp, found {data_type}"
        )));
    };
    if timezone
        .as_deref()
        .is_some_and(|timezone| timezone != "UTC")
    {
        return Err(compile_error(format!(
            "window event-time column {column:?} must be timezone-naive or UTC"
        )));
    }
    Ok(())
}

fn supports_group_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Date32
            | DataType::Date64
    ) || matches!(
        data_type,
        DataType::Timestamp(TimeUnit::Microsecond, timezone)
            if timezone.as_deref().is_none_or(|timezone| timezone == "UTC")
    )
}

fn aggregate_output_type(function: AggregateFunction, input: &DataType) -> Option<DataType> {
    match function {
        AggregateFunction::Count => Some(DataType::UInt64),
        AggregateFunction::Sum => numeric_output_type(input),
        AggregateFunction::Avg if is_numeric(input) => Some(DataType::Float64),
        AggregateFunction::Min | AggregateFunction::Max if supports_ordered_aggregate(input) => {
            Some(input.clone())
        }
        AggregateFunction::Avg | AggregateFunction::Min | AggregateFunction::Max => None,
    }
}

fn numeric_output_type(input: &DataType) -> Option<DataType> {
    match input {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            Some(DataType::Int64)
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            Some(DataType::UInt64)
        }
        DataType::Float32 | DataType::Float64 => Some(DataType::Float64),
        _ => None,
    }
}

fn is_numeric(input: &DataType) -> bool {
    numeric_output_type(input).is_some()
}

fn supports_ordered_aggregate(input: &DataType) -> bool {
    is_numeric(input)
        || matches!(
            input,
            DataType::Boolean
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Date32
                | DataType::Date64
        )
        || matches!(
            input,
            DataType::Timestamp(TimeUnit::Microsecond, timezone)
                if timezone.as_deref().is_none_or(|timezone| timezone == "UTC")
        )
}

fn configuration(spec: &WindowSpec) -> Result<JsonMap> {
    let geometry = serde_json::to_value(spec.geometry).map_err(|error| format_error(&error))?;
    let aggregates =
        serde_json::to_value(&spec.aggregates).map_err(|error| format_error(&error))?;
    Ok(JsonMap::from([
        ("kind".into(), json!("window_aggregate")),
        (
            "state_layout_version".into(),
            json!(WINDOW_STATE_LAYOUT_VERSION),
        ),
        ("event_time_column".into(), json!(spec.event_time_column)),
        ("geometry".into(), geometry),
        ("group_by".into(), json!(spec.group_by)),
        ("aggregates".into(), aggregates),
        ("group_key_encoding".into(), json!("g1")),
        ("max_group_key_bytes".into(), json!(MAX_GROUP_KEY_BYTES)),
        ("null_event_time_policy".into(), json!("drop")),
    ]))
}

fn validate_geometry(geometry: WindowGeometry) -> Result<()> {
    let (size, slide) = match geometry {
        WindowGeometry::Tumbling { size_micros } => (size_micros, size_micros),
        WindowGeometry::Hopping {
            size_micros,
            slide_micros,
        } => (size_micros, slide_micros),
    };
    if size == 0 {
        return Err(invalid_argument(
            "window.geometry.size",
            "must be greater than zero",
        ));
    }
    if slide == 0 {
        return Err(invalid_argument(
            "window.geometry.slide",
            "must be greater than zero",
        ));
    }
    if size % slide != 0 {
        return Err(invalid_argument(
            "window.geometry",
            "size must be an exact multiple of slide",
        ));
    }
    if size / slide > MAX_WINDOW_OVERLAP {
        return Err(invalid_argument(
            "window.geometry",
            "window overlap exceeds MAX_WINDOW_OVERLAP",
        ));
    }
    Ok(())
}

fn exact_duration_micros(duration: Duration, field: &str) -> Result<u64> {
    let nanos = duration.as_nanos();
    if nanos == 0 {
        return Err(invalid_argument(field, "must be greater than zero"));
    }
    if nanos % 1_000 != 0 {
        return Err(invalid_argument(
            field,
            "must be an exact multiple of one microsecond",
        ));
    }
    u64::try_from(nanos / 1_000)
        .map_err(|_| invalid_argument(field, "exceeds the serialized microsecond range"))
}

fn is_reserved_output(value: &str) -> bool {
    matches!(value, "window_start" | "window_end")
}

fn invalid_argument(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: field.into(),
        message: message.into(),
    }
}

fn format_error(error: &serde_json::Error) -> CalcFlowError {
    CalcFlowError::Format {
        message: error.to_string(),
    }
}

fn deserialize_required_option<'de, D, T>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BatchMetadata, StateBudget};

    const HIGH_CARDINALITY_ROWS: usize = 400_000;
    const LEGACY_PROJECT_JSON_LIMIT: usize = 10 * 1024 * 1024;

    /// The pre-iterator eager assignment semantics the lazy iterator must match.
    fn eager_window_assignments(
        event_time: EventTime,
        geometry: CompiledWindowGeometry,
    ) -> std::result::Result<Vec<(EventTime, EventTime)>, String> {
        let time = i128::from(event_time.as_micros());
        let slide = i128::from(geometry.slide_micros);
        let latest_start = time.div_euclid(slide) * slide;
        (0..geometry.overlap)
            .rev()
            .map(|offset| {
                window_assignment(
                    latest_start,
                    slide,
                    i128::from(geometry.size_micros),
                    offset,
                )
            })
            .collect()
    }

    #[test]
    fn window_assignment_iterator_matches_eager_assignments_at_boundaries() {
        let geometries = [
            (1, 1),
            (100, 100),
            (100, 10),
            (1_024, 1),
            (1 << 62, 1 << 52),
            (1 << 63, 1 << 63),
            (1 << 63, 1 << 54),
            (u64::MAX, u64::MAX),
        ];
        let times = [
            i64::MIN,
            i64::MIN + 1,
            i64::MIN + 99,
            -101,
            -100,
            -1,
            0,
            1,
            99,
            100,
            i64::MAX - 1_023,
            i64::MAX - 1,
            i64::MAX,
        ];
        for (size_micros, slide_micros) in geometries {
            let geometry = CompiledWindowGeometry {
                size_micros,
                slide_micros,
                overlap: size_micros / slide_micros,
            };
            for time in times {
                let event_time = EventTime::from_micros(time);
                let lazy =
                    window_assignments(event_time, geometry).map(Iterator::collect::<Vec<_>>);
                assert_eq!(
                    lazy,
                    eager_window_assignments(event_time, geometry),
                    "size={size_micros} slide={slide_micros} time={time}"
                );
            }
        }
    }

    fn budget_test_batch(times: &[i64]) -> Batch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("amount", DataType::Int64, false),
        ]));
        let record = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(TimestampMicrosecondArray::from(times.to_vec())) as ArrayRef,
                Arc::new(Int64Array::from(vec![1; times.len()])) as ArrayRef,
            ],
        )
        .unwrap();
        Batch::table(vec![record], BatchMetadata::default()).unwrap()
    }

    fn budget_test_string_batch(value: &str) -> Batch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("value", DataType::Utf8, false),
        ]));
        let record = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(TimestampMicrosecondArray::from(vec![0])) as ArrayRef,
                Arc::new(StringArray::from(vec![value])) as ArrayRef,
            ],
        )
        .unwrap();
        Batch::table(vec![record], BatchMetadata::default()).unwrap()
    }

    fn budget_test_job() -> crate::StreamJobContext {
        crate::StreamJobContext::new(
            1,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        )
    }

    #[tokio::test]
    async fn test_checkpoint_disabled_window_releases_only_accepted_output() {
        let mut operator = checkpoint_segment_operator();
        operator
            .set_state_budget(StateBudget::new(1, 1_048_576).unwrap())
            .unwrap();
        let job = budget_test_job().with_checkpointing(false);
        let context = StreamOperatorContext::new(&job, "window", None);
        let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("input", budget_test_batch(&[0, 1]), &context, &mut output)
            .await
            .unwrap();
        assert_eq!(operator.state.accumulators.len(), 1);
        assert!(operator.state.accumulator_bytes > 0);
        assert!(operator.state.prepared_segments.is_empty());
        assert!(operator.state.dirty.is_empty());
        struct RejectOutput;
        #[async_trait]
        impl StreamCollector for RejectOutput {
            async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
                Err(operator_error("window", "output rejected"))
            }
        }
        let before_bytes = operator.state.accumulator_bytes;
        let error = operator
            .on_watermark(
                EventTime::from_micros(60_000_000),
                &context,
                &mut RejectOutput,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("output rejected"));
        assert_eq!(operator.state.accumulators.len(), 1);
        assert_eq!(operator.state.accumulator_bytes, before_bytes);
        assert!(operator.state.last_input_watermark.is_none());
        assert!(operator.state.emitted_pending_snapshot.is_empty());
        operator
            .on_watermark(EventTime::from_micros(60_000_000), &context, &mut output)
            .await
            .unwrap();
        let emitted = output.drain("output");
        assert_eq!(emitted[0].as_data().unwrap().num_rows(), 1);
        assert!(operator.state.accumulators.is_empty());
        assert_eq!(operator.state.accumulator_bytes, 0);
        assert!(operator.state.emitted_pending_snapshot.is_empty());
        assert!(operator.state.prepared_segments.is_empty());
        operator
            .process_data(
                "input",
                budget_test_batch(&[60_000_000]),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        operator.on_end(&context, &mut output).await.unwrap();
        assert_eq!(output.drain("output")[0].as_data().unwrap().num_rows(), 1);
        assert!(operator.state.accumulators.is_empty());
        assert_eq!(operator.state.accumulator_bytes, 0);
        assert!(operator.state.emitted_pending_snapshot.is_empty());
        assert!(operator.state.prepared_segments.is_empty());
    }

    #[tokio::test]
    async fn window_state_row_budget_rejects_whole_batch_before_installing_it() {
        let mut operator = checkpoint_segment_operator();
        operator
            .set_state_budget(StateBudget::new(1, 1_048_576).unwrap())
            .unwrap();
        let job = budget_test_job();
        let context = StreamOperatorContext::new(&job, "window", None);
        let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());

        let error = operator
            .process_data(
                "input",
                budget_test_batch(&[0, 60_000_000]),
                &context,
                &mut output,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("window state budget exceeded"));
        assert!(operator.state.accumulators.is_empty());

        operator
            .process_data("input", budget_test_batch(&[0]), &context, &mut output)
            .await
            .unwrap();
        assert_eq!(operator.state.accumulators.len(), 1);
        let previous_budget = operator.state_budget;
        assert!(
            operator
                .set_state_budget(StateBudget::new(1, 1).unwrap())
                .is_err()
        );
        assert_eq!(operator.state_budget, previous_budget);
    }

    #[tokio::test]
    async fn window_state_byte_budget_rejects_one_group() {
        let mut operator = checkpoint_segment_operator();
        operator
            .set_state_budget(StateBudget::new(10, 1).unwrap())
            .unwrap();
        let job = budget_test_job();
        let context = StreamOperatorContext::new(&job, "window", None);
        let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());

        let error = operator
            .process_data("input", budget_test_batch(&[0]), &context, &mut output)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("window state budget exceeded"));
        assert!(operator.state.accumulators.is_empty());
    }

    #[tokio::test]
    async fn window_state_budget_tracks_min_string_growth_without_changing_live_state() {
        let schema = budget_test_string_batch("z")
            .table_payload()
            .unwrap()
            .batches()[0]
            .schema();
        let spec = WindowSpec::tumbling("event_time", Duration::from_secs(60))
            .unwrap()
            .aggregate(AggregateFunction::Min, "value", "minimum")
            .unwrap();
        let mut operator = WindowAggregateOperator::new("window", schema, spec).unwrap();
        operator
            .set_state_budget(StateBudget::new(1, 250).unwrap())
            .unwrap();
        let job = budget_test_job();
        let context = StreamOperatorContext::new(&job, "window", None);
        let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "input",
                budget_test_string_batch("z"),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        let previous_bytes = operator.state.accumulator_bytes;

        let error = operator
            .process_data(
                "input",
                budget_test_string_batch(&"a".repeat(200)),
                &context,
                &mut output,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("window state budget exceeded"));
        assert_eq!(operator.state.accumulator_bytes, previous_bytes);
        let entry = operator.state.accumulators.values().next().unwrap();
        assert!(matches!(
            &entry.aggregates[0],
            AccumulatorValue::Min(Some(ScalarValue::String(value))) if value == "z"
        ));

        operator
            .process_data(
                "input",
                budget_test_string_batch("y"),
                &context,
                &mut output,
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn hopping_window_budget_rejects_mid_row_without_partial_state() {
        let schema = budget_test_batch(&[15]).table_payload().unwrap().batches()[0].schema();
        let spec = WindowSpec::hopping(
            "event_time",
            Duration::from_micros(60),
            Duration::from_micros(10),
        )
        .unwrap()
        .aggregate(AggregateFunction::Count, "amount", "count_amount")
        .unwrap();
        let mut operator = WindowAggregateOperator::new("window", schema, spec).unwrap();
        operator
            .set_state_budget(StateBudget::new(1, 1_048_576).unwrap())
            .unwrap();
        let job = budget_test_job();
        let context = StreamOperatorContext::new(&job, "window", None);
        let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());

        let error = operator
            .process_data("input", budget_test_batch(&[15]), &context, &mut output)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("window state budget exceeded"));
        assert!(operator.state.accumulators.is_empty());
        assert_eq!(operator.state.accumulator_bytes, 0);
    }

    #[tokio::test]
    async fn window_state_budget_releases_closed_groups_after_checkpoint() {
        let mut operator = checkpoint_segment_operator();
        operator
            .set_state_budget(StateBudget::new(1, 1_048_576).unwrap())
            .unwrap();
        let job = budget_test_job();
        let context = StreamOperatorContext::new(&job, "window", None);
        let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("input", budget_test_batch(&[0]), &context, &mut output)
            .await
            .unwrap();

        operator
            .on_watermark(EventTime::from_micros(60_000_000), &context, &mut output)
            .await
            .unwrap();
        assert_eq!(operator.state.accumulators.len(), 1);
        operator.checkpoint(crate::Epoch::INITIAL).unwrap();
        assert!(operator.state.accumulators.is_empty());
        assert_eq!(operator.state.accumulator_bytes, 0);

        operator
            .process_data(
                "input",
                budget_test_batch(&[60_000_000]),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        assert_eq!(operator.state.accumulators.len(), 1);
    }

    #[tokio::test]
    async fn window_state_budget_rejects_restore_without_replacing_live_state() {
        let mut source = checkpoint_segment_operator();
        let job = budget_test_job();
        let context = StreamOperatorContext::new(&job, "window", None);
        let mut output = crate::EdgeCollector::new(source.output_ports().to_vec());
        source
            .process_data(
                "input",
                budget_test_batch(&[0, 60_000_000]),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        let snapshot = source.checkpoint(crate::Epoch::INITIAL).unwrap();

        let mut restored = checkpoint_segment_operator();
        restored
            .set_state_budget(StateBudget::new(1, 1_048_576).unwrap())
            .unwrap();
        let error = restored.restore(&snapshot).unwrap_err();
        assert!(error.to_string().contains("window state budget exceeded"));
        assert!(restored.state.accumulators.is_empty());
    }

    #[tokio::test]
    async fn restored_window_state_budget_applies_to_next_batch() {
        let mut source = checkpoint_segment_operator();
        let job = budget_test_job();
        let context = StreamOperatorContext::new(&job, "window", None);
        let mut output = crate::EdgeCollector::new(source.output_ports().to_vec());
        source
            .process_data("input", budget_test_batch(&[0]), &context, &mut output)
            .await
            .unwrap();
        let snapshot = source.checkpoint(crate::Epoch::INITIAL).unwrap();

        let mut restored = checkpoint_segment_operator();
        restored
            .set_state_budget(StateBudget::new(1, 1_048_576).unwrap())
            .unwrap();
        restored.restore(&snapshot).unwrap();
        let previous_bytes = restored.state.accumulator_bytes;
        let mut output = crate::EdgeCollector::new(restored.output_ports().to_vec());
        let error = restored
            .process_data(
                "input",
                budget_test_batch(&[60_000_000]),
                &context,
                &mut output,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("window state budget exceeded"));
        assert_eq!(restored.state.accumulators.len(), 1);
        assert_eq!(restored.state.accumulator_bytes, previous_bytes);
    }

    #[test]
    fn float_sum_and_average_recover_low_order_terms() {
        let mut sum = AccumulatorValue::FloatSum(None);
        let mut average = AccumulatorValue::FloatAverage {
            sum: CompensatedSum::new(0.0),
            count: 0,
        };
        for value in [1.0e16, 1.0, -1.0e16] {
            let scalar = ScalarValue::Float64(f64::to_bits(value));
            update_accumulator(&mut sum, AggregateFunction::Sum, scalar.clone()).unwrap();
            update_accumulator(&mut average, AggregateFunction::Avg, scalar).unwrap();
        }
        let sum = finalize_accumulator(&sum).unwrap().unwrap();
        let average = finalize_accumulator(&average).unwrap().unwrap();
        assert_eq!(float_value(&sum).unwrap().to_bits(), 1.0_f64.to_bits());
        assert_eq!(
            float_value(&average).unwrap().to_bits(),
            (1.0_f64 / 3.0).to_bits()
        );
    }

    #[test]
    fn float_compensation_survives_checkpoint_segment_restore() {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("amount", DataType::Float64, false),
        ]));
        let spec = WindowSpec::tumbling("event_time", Duration::from_secs(60))
            .unwrap()
            .aggregate(AggregateFunction::Sum, "amount", "total")
            .unwrap()
            .aggregate(AggregateFunction::Avg, "amount", "mean")
            .unwrap();
        let mut source =
            WindowAggregateOperator::new("window", Arc::clone(&schema), spec.clone()).unwrap();
        let mut sum = AccumulatorValue::FloatSum(None);
        let mut average = AccumulatorValue::FloatAverage {
            sum: CompensatedSum::new(0.0),
            count: 0,
        };
        for value in [1.0e16, 1.0] {
            let scalar = ScalarValue::Float64(f64::to_bits(value));
            update_accumulator(&mut sum, AggregateFunction::Sum, scalar.clone()).unwrap();
            update_accumulator(&mut average, AggregateFunction::Avg, scalar).unwrap();
        }
        let operations = vec![StateOperationRow {
            key: WindowKey {
                start: EventTime::from_micros(0),
                end: EventTime::from_micros(60_000_000),
                stable_group_key: Arc::from([]),
            },
            entry: AccumulatorRow {
                group_values: Vec::new(),
                aggregates: vec![sum, average],
            },
            tombstone: false,
        }];
        let fingerprint = "a".repeat(64);
        let bytes =
            encode_state_segment(&operations, &spec, &source.compiled, &fingerprint, "window")
                .unwrap();
        source.state.pipeline_fingerprint = Some(fingerprint);
        source.state.operator_id = Some("window".into());
        source
            .state
            .accumulators
            .insert(operations[0].key.clone(), operations[0].entry.clone());
        source.state.prepared_segments.push(PreparedStateSegment {
            kind: SegmentKind::Delta,
            bytes,
        });
        let snapshot = source.checkpoint(crate::Epoch::INITIAL).unwrap();
        let mut restored = WindowAggregateOperator::new("window", schema, spec).unwrap();
        restored.restore(&snapshot).unwrap();
        let mut decoded = restored
            .state
            .accumulators
            .values()
            .next()
            .unwrap()
            .aggregates
            .clone();
        for (function, accumulator) in [AggregateFunction::Sum, AggregateFunction::Avg]
            .into_iter()
            .zip(&mut decoded)
        {
            update_accumulator(
                accumulator,
                function,
                ScalarValue::Float64((-1.0e16_f64).to_bits()),
            )
            .unwrap();
        }
        let sum = finalize_accumulator(&decoded[0]).unwrap().unwrap();
        let average = finalize_accumulator(&decoded[1]).unwrap().unwrap();
        assert_eq!(float_value(&sum).unwrap().to_bits(), 1.0_f64.to_bits());
        assert_eq!(
            float_value(&average).unwrap().to_bits(),
            (1.0_f64 / 3.0).to_bits()
        );
    }

    #[test]
    fn legacy_float_snapshot_is_rewritten_as_compensated_base() {
        let (mut operator, spec) = legacy_float_window_operator();
        operator.state.operator_id = Some("window".into());
        let fingerprint = "a".repeat(64);
        let legacy_schema = state_schema_for_version(
            &spec,
            &operator.compiled,
            &fingerprint,
            "window",
            WINDOW_STATE_LAYOUT_VERSION,
        );
        let operations = vec![StateOperationRow {
            key: WindowKey {
                start: EventTime::from_micros(0),
                end: EventTime::from_micros(60_000_000),
                stable_group_key: Arc::from([]),
            },
            entry: AccumulatorRow {
                group_values: Vec::new(),
                aggregates: vec![AccumulatorValue::FloatSum(Some(CompensatedSum::new(5.0)))],
            },
            tombstone: false,
        }];
        let mut arrays = state_key_arrays(&operations);
        arrays.push(
            scalar_array(
                &DataType::Float64,
                &[Some(ScalarValue::Float64(5.0_f64.to_bits()))],
                "window",
            )
            .unwrap(),
        );
        let segment = crate::StateSegment::new(write_state_ipc(&legacy_schema, arrays).unwrap());
        let segment_id = "delta-00000000000000000001-00000000";
        let mut descriptor = operator
            .snapshot_segment_descriptor(
                crate::Epoch::INITIAL,
                segment_id,
                SegmentKind::Delta,
                &segment,
            )
            .unwrap();
        descriptor.state_layout_version = WINDOW_STATE_LAYOUT_VERSION;
        descriptor.schema_fingerprint = state_schema_fingerprint_for_version(
            &spec,
            &operator.compiled,
            WINDOW_STATE_LAYOUT_VERSION,
        );
        let metadata = WindowSnapshotMetadata {
            state_layout_version: WINDOW_STATE_LAYOUT_VERSION,
            configuration_hash: operator.compiled.configuration_hash.clone(),
            state_schema_fingerprint: descriptor.schema_fingerprint.clone(),
            epoch: crate::Epoch::INITIAL,
            pipeline_fingerprint: Some(fingerprint),
            operator_id: Some("window".into()),
            last_input_watermark: None,
            next_output_sequence: 0,
            ended: false,
            metrics: LateMetricDelta::default(),
            segment_inventory: vec![descriptor],
        };
        let Value::Object(inline_metadata) = serde_json::to_value(metadata).unwrap() else {
            panic!("snapshot metadata must be an object");
        };
        let mut malformed_metadata = inline_metadata.clone();
        malformed_metadata.insert("state_layout_version".into(), serde_json::json!(2));
        malformed_metadata.insert(
            "state_schema_fingerprint".into(),
            serde_json::json!(operator.compiled.state_schema_fingerprint.clone()),
        );
        assert!(
            operator
                .restore(&crate::OperatorStateSnapshot {
                    inline_metadata: malformed_metadata.into_iter().collect(),
                    segments: BTreeMap::from([(segment_id.into(), segment.clone())]),
                })
                .is_err(),
            "a version 2 header cannot retain version 1 segments"
        );
        operator
            .restore(&crate::OperatorStateSnapshot {
                inline_metadata: inline_metadata.into_iter().collect(),
                segments: BTreeMap::from([(segment_id.into(), segment)]),
            })
            .unwrap();
        assert!(operator.state.replace_retained_on_checkpoint);
        let upgraded = operator.checkpoint(crate::Epoch::new(2).unwrap()).unwrap();
        let inventory = upgraded.inline_metadata["segment_inventory"]
            .as_array()
            .unwrap();
        assert_eq!(inventory.len(), 1);
        assert_eq!(inventory[0]["state_layout_version"], 2);
        assert_eq!(inventory[0]["kind"], "base");
        assert!(!upgraded.segments.contains_key(segment_id));
    }

    fn legacy_float_window_operator() -> (WindowAggregateOperator, WindowSpec) {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("amount", DataType::Float64, false),
        ]));
        let spec = WindowSpec::tumbling("event_time", Duration::from_secs(60))
            .unwrap()
            .aggregate(AggregateFunction::Sum, "amount", "total")
            .unwrap();
        (
            WindowAggregateOperator::new("window", schema, spec.clone()).unwrap(),
            spec,
        )
    }

    fn output_record(rows: usize) -> RecordBatch {
        RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(
                (0..rows)
                    .map(|value| i64::try_from(value).unwrap())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        )])
        .unwrap()
    }

    fn checkpoint_segment_operator() -> WindowAggregateOperator {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("amount", DataType::Int64, false),
        ]));
        let spec = WindowSpec::tumbling("event_time", Duration::from_secs(60))
            .unwrap()
            .aggregate(AggregateFunction::Sum, "amount", "total")
            .unwrap();
        WindowAggregateOperator::new("window", schema, spec).unwrap()
    }

    #[test]
    fn checkpoint_segment_bytes_round_trip_is_stable_across_decode_reencode() {
        // Build a compiled spec over a tumbling sum window and hand-construct
        // the state rows the codec carries; the operator stream path is not
        // needed to guard the encoding itself.
        let input_schema = Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("amount", DataType::Int64, false),
        ]);
        let spec = WindowSpec::tumbling("event_time", Duration::from_secs(60))
            .unwrap()
            .aggregate(AggregateFunction::Sum, "amount", "total")
            .unwrap();
        let compiled = compile_spec(&input_schema, &spec, &JsonMap::new()).unwrap();

        let operations = vec![
            StateOperationRow {
                key: WindowKey {
                    start: EventTime::from_micros(0),
                    end: EventTime::from_micros(60_000_000),
                    stable_group_key: Arc::from([]),
                },
                entry: AccumulatorRow {
                    group_values: Vec::new(),
                    aggregates: vec![AccumulatorValue::SignedSum(Some(150))],
                },
                tombstone: false,
            },
            StateOperationRow {
                key: WindowKey {
                    start: EventTime::from_micros(60_000_000),
                    end: EventTime::from_micros(120_000_000),
                    stable_group_key: Arc::from([]),
                },
                entry: AccumulatorRow {
                    group_values: Vec::new(),
                    aggregates: vec![AccumulatorValue::SignedSum(Some(42))],
                },
                tombstone: false,
            },
        ];

        let first_bytes =
            encode_state_segment(&operations, &spec, &compiled, &"a".repeat(64), "window").unwrap();
        assert!(!first_bytes.is_empty());

        // Decoding must restore the exact rows in order.
        let expected_schema = state_schema(&spec, &compiled, &"a".repeat(64), "window");
        let restored = decode_state_segment(
            &first_bytes,
            &spec,
            &compiled,
            &expected_schema,
            "window",
            WINDOW_SEGMENT_LAYOUT_VERSION,
        )
        .unwrap();
        assert_eq!(restored.len(), operations.len());
        for ((key, row), original) in restored.iter().zip(&operations) {
            assert_eq!(*key, original.key);
            let entry = row.as_ref().expect("every live row carries an entry");
            match (&entry.aggregates[..], &original.entry.aggregates[..]) {
                ([AccumulatorValue::SignedSum(left)], [AccumulatorValue::SignedSum(right)]) => {
                    assert_eq!(left, right);
                }
                _ => panic!("restored aggregate shape must match the encoded shape"),
            }
        }

        // Re-encoding the same rows must reproduce identical bytes. This is
        // the guard the planned codec consolidation must hold for every
        // checkpoint ever written.
        let second_bytes =
            encode_state_segment(&operations, &spec, &compiled, &"a".repeat(64), "window").unwrap();
        assert_eq!(
            first_bytes, second_bytes,
            "window checkpoint segment encoding must be deterministic byte-for-byte"
        );

        // A single-bit flip in the payload must not decode to the same rows.
        let mut corrupted = first_bytes.clone();
        let flip_at = corrupted.len() - 6;
        corrupted[flip_at] ^= 0x01;
        let decoded = decode_state_segment(
            &corrupted,
            &spec,
            &compiled,
            &expected_schema,
            "window",
            WINDOW_SEGMENT_LAYOUT_VERSION,
        );
        if let Ok(rows) = decoded {
            let same = rows.len() == operations.len()
                && rows.iter().zip(&operations).all(|((k, r), o)| {
                    let aggregates_match = matches!(
                        (r.as_ref().map(|e| &e.aggregates[..]), &o.entry.aggregates[..]),
                        (
                            Some([AccumulatorValue::SignedSum(left)]),
                            [AccumulatorValue::SignedSum(right)],
                        ) if left == right
                    );
                    *k == o.key && aggregates_match
                });
            assert!(
                !same,
                "a bit-flipped segment must not silently decode to the original rows"
            );
        }
    }

    #[test]
    fn checkpoint_segment_assembly_rejects_duplicate_and_missing_retained_data() {
        let mut operator = checkpoint_segment_operator();
        operator.state.operator_id = Some("window".into());
        let segment_id = "delta-00000000000000000001-00000000";
        let segment = crate::StateSegment::new(vec![1, 2, 3]);
        operator
            .state
            .retained_segments
            .insert(segment_id.into(), segment.clone());
        let duplicate = operator
            .next_snapshot_segments(
                &StateInventory::default(),
                BTreeMap::from([(segment_id.into(), segment.clone())]),
            )
            .unwrap_err();
        assert!(matches!(
            duplicate,
            CalcFlowError::CheckpointMismatch { message }
                if message == "window checkpoint produced a duplicate segment ID"
        ));

        operator.state.retained_segments.clear();
        let descriptor = operator
            .snapshot_segment_descriptor(
                crate::Epoch::INITIAL,
                segment_id,
                SegmentKind::Delta,
                &segment,
            )
            .unwrap();
        let inventory = StateInventory::new(vec![descriptor]).unwrap();
        let missing = operator
            .next_snapshot_segments(&inventory, BTreeMap::new())
            .unwrap_err();
        assert!(matches!(
            missing,
            CalcFlowError::CheckpointMismatch { message }
                if message == "window checkpoint segment data does not match its inventory"
        ));
    }

    #[test]
    fn replacement_checkpoint_without_prepared_segments_discards_old_inventory() {
        let mut operator = checkpoint_segment_operator();
        operator.state.operator_id = Some("window".into());
        let segment_id = "delta-00000000000000000001-00000000";
        let segment = crate::StateSegment::new(vec![1, 2, 3]);
        let descriptor = operator
            .snapshot_segment_descriptor(
                crate::Epoch::INITIAL,
                segment_id,
                SegmentKind::Delta,
                &segment,
            )
            .unwrap();
        operator.state.retained_inventory = StateInventory::new(vec![descriptor]).unwrap();
        operator
            .state
            .retained_segments
            .insert(segment_id.into(), segment);
        operator.state.replace_retained_on_checkpoint = true;

        let snapshot = operator.checkpoint(crate::Epoch::INITIAL).unwrap();
        let metadata = parse_snapshot_metadata(&snapshot).unwrap();

        assert!(metadata.segment_inventory.is_empty());
        assert!(snapshot.segments.is_empty());
        assert!(operator.state.retained_inventory.segments().is_empty());
        assert!(operator.state.retained_segments.is_empty());
    }

    #[test]
    fn output_chunking_preserves_rows_and_uses_consecutive_sequences() {
        let record = output_record(5);
        let chunks = chunk_output_record(
            &record,
            "window",
            7,
            crate::EdgeBudget {
                max_rows: 2,
                max_bytes: usize::MAX,
            },
        )
        .unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(
            chunks.iter().map(Batch::num_rows).collect::<Vec<_>>(),
            [2, 2, 1]
        );
        assert_eq!(
            chunks
                .iter()
                .map(|batch| batch.metadata().sequence())
                .collect::<Vec<_>>(),
            [7, 8, 9]
        );
    }

    #[test]
    fn one_oversized_output_row_fails_before_returning_any_chunk() {
        let record = output_record(1);
        let bytes = Batch::table(vec![record.clone()], BatchMetadata::default())
            .unwrap()
            .estimated_bytes()
            .unwrap();
        let error = chunk_output_record(
            &record,
            "window",
            0,
            crate::EdgeBudget {
                max_rows: 1,
                max_bytes: bytes - 1,
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CalcFlowError::InvalidArgument { field, .. } if field == "message.bytes"
        ));
    }

    #[test]
    fn date32_group_encoding_uses_one_signed_big_endian_payload() {
        let mut encoded = Vec::new();
        encode_group_scalar(
            &mut encoded,
            &DataType::Date32,
            Some(&ScalarValue::Date32(1)),
        )
        .unwrap();
        assert_eq!(encoded, [0x01, 0x80, 0x00, 0x00, 0x01]);
    }

    #[test]
    fn count_and_average_counts_reject_uint64_overflow() {
        let mut count = AccumulatorValue::Count(u64::MAX);
        assert_eq!(
            update_accumulator(
                &mut count,
                AggregateFunction::Count,
                ScalarValue::Unsigned(0),
            ),
            Err("count overflowed UInt64".into())
        );
        assert!(matches!(count, AccumulatorValue::Count(u64::MAX)));

        let mut average = AccumulatorValue::SignedAverage {
            sum: 7,
            count: u64::MAX,
        };
        assert_eq!(
            update_accumulator(&mut average, AggregateFunction::Avg, ScalarValue::Signed(1),),
            Err("average count overflowed UInt64".into())
        );
    }

    #[tokio::test]
    #[ignore = "M7 high-cardinality state soak; set CALC_FLOW_M7_WINDOW_STATE_SOAK=1"]
    async fn high_cardinality_window_state_exceeds_legacy_json_limit_and_restores() {
        if std::env::var("CALC_FLOW_M7_WINDOW_STATE_SOAK").as_deref() != Ok("1") {
            return;
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("account", DataType::Int64, false),
            Field::new("amount", DataType::Int64, false),
        ]));
        let input = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(TimestampMicrosecondArray::from(vec![
                    0;
                    HIGH_CARDINALITY_ROWS
                ])) as ArrayRef,
                Arc::new(Int64Array::from_iter_values(
                    0..i64::try_from(HIGH_CARDINALITY_ROWS).unwrap(),
                )),
                Arc::new(Int64Array::from(vec![1; HIGH_CARDINALITY_ROWS])),
            ],
        )
        .unwrap();
        let spec = WindowSpec::tumbling("event_time", Duration::from_secs(60))
            .unwrap()
            .group_by(["account"])
            .unwrap()
            .aggregate(AggregateFunction::Sum, "amount", "total")
            .unwrap();
        let mut source =
            WindowAggregateOperator::new("window", Arc::clone(&schema), spec.clone()).unwrap();
        let job = crate::StreamJobContext::new(
            1,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "window", None);
        let mut collector = crate::EdgeCollector::new(source.output_ports().to_vec());
        source
            .process_data(
                "input",
                Batch::table(vec![input], BatchMetadata::default()).unwrap(),
                &context,
                &mut collector,
            )
            .await
            .unwrap();

        let snapshot = source.checkpoint(crate::Epoch::INITIAL).unwrap();
        let segment_bytes = snapshot
            .segments
            .values()
            .map(|segment| segment.bytes().len())
            .sum::<usize>();
        assert!(
            segment_bytes > LEGACY_PROJECT_JSON_LIMIT,
            "high-cardinality state encoded only {segment_bytes} bytes"
        );
        let mut restored = WindowAggregateOperator::new("window", schema, spec).unwrap();
        restored.restore(&snapshot).unwrap();
        assert_eq!(restored.state.accumulators.len(), HIGH_CARDINALITY_ROWS);
    }
}
