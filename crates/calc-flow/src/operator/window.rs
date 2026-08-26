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
    Batch, BatchKind, BatchMetadata, CalcFlowError, EventTime, JsonMap, Port, Result, StateHandle,
    StreamCollector, StreamOperator, StreamOperatorContext, canonical_json,
    state::{SegmentDescriptor, SegmentKind, StateInventory, StateOperation, fold_state_segments},
};

use super::{LateMetricDelta, OperatorMetadata, accumulate_late_metrics, validate_operator_name};

/// Maximum number of concrete hopping-window assignments for one input row.
pub const MAX_WINDOW_OVERLAP: u64 = 1_024;

pub(crate) const WINDOW_STATE_LAYOUT_VERSION: u32 = 1;
const MAX_GROUP_KEY_BYTES: usize = 65_536;
const MAX_WINDOW_DELTA_SEGMENTS: usize = 32;

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
    dirty: BTreeSet<WindowKey>,
    emitted_pending_snapshot: BTreeSet<WindowKey>,
    last_input_watermark: Option<EventTime>,
    next_output_sequence: u64,
    ended: bool,
    metrics: LateMetricDelta,
    prepared_segments: Vec<PreparedStateSegment>,
    retained_inventory: StateInventory,
    replace_retained_on_checkpoint: bool,
    last_checkpoint_epoch: Option<crate::Epoch>,
    pipeline_fingerprint: Option<String>,
    operator_id: Option<String>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct WindowKey {
    start: EventTime,
    end: EventTime,
    stable_group_key: Vec<u8>,
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
    FloatSum(Option<f64>),
    Min(Option<ScalarValue>),
    Max(Option<ScalarValue>),
    SignedAverage { sum: i128, count: u64 },
    UnsignedAverage { sum: u128, count: u64 },
    FloatAverage { sum: f64, count: u64 },
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
    metrics: LateMetricDelta,
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
        })
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
        let mut scratch = BTreeMap::<WindowKey, AccumulatorRow>::new();
        let mut metrics = PreparedInputMetrics::default();

        for record in table.batches() {
            self.prepare_record(record, context, &mut scratch, &mut metrics)?;
        }

        Ok(InputBatchUpdate {
            accumulators: scratch,
            metrics: metrics.into_delta(),
        })
    }

    fn prepare_record(
        &self,
        record: &RecordBatch,
        context: &StreamOperatorContext<'_>,
        scratch: &mut BTreeMap<WindowKey, AccumulatorRow>,
        metrics: &mut PreparedInputMetrics,
    ) -> Result<()> {
        for row_index in 0..record.num_rows() {
            self.prepare_row(record, row_index, context, scratch, metrics)?;
        }
        Ok(())
    }

    fn prepare_row(
        &self,
        record: &RecordBatch,
        row_index: usize,
        context: &StreamOperatorContext<'_>,
        scratch: &mut BTreeMap<WindowKey, AccumulatorRow>,
        metrics: &mut PreparedInputMetrics,
    ) -> Result<()> {
        let Some(event_time) = self.row_event_time(record, row_index, context.operator_id())?
        else {
            record_null_event_time(metrics, context.operator_id())?;
            return Ok(());
        };
        let assignments = window_assignments(event_time, self.compiled.geometry)
            .map_err(|message| operator_error(context.operator_id(), &message))?;
        let open_assignments = partition_open_assignments(
            assignments,
            context.input_watermark(),
            metrics,
            context.operator_id(),
        )?;
        if open_assignments.is_empty() {
            return Ok(());
        }
        self.prepare_open_row(record, row_index, &open_assignments, context, scratch)
    }

    fn prepare_open_row(
        &self,
        record: &RecordBatch,
        row_index: usize,
        open_assignments: &[(EventTime, EventTime)],
        context: &StreamOperatorContext<'_>,
        scratch: &mut BTreeMap<WindowKey, AccumulatorRow>,
    ) -> Result<()> {
        let (stable_group_key, group_values) = encode_group_key(
            record,
            row_index,
            &self.compiled.group_columns,
            context.operator_id(),
            &self.spec.group_by,
        )?;
        for &(start, end) in open_assignments {
            self.prepare_assignment(
                record,
                row_index,
                start,
                end,
                &stable_group_key,
                &group_values,
                scratch,
                context.operator_id(),
            )?;
        }
        Ok(())
    }

    fn row_event_time(
        &self,
        record: &RecordBatch,
        row_index: usize,
        operator_id: &str,
    ) -> Result<Option<EventTime>> {
        event_time_at(
            record.column(self.compiled.event_time_index).as_ref(),
            record
                .schema()
                .field(self.compiled.event_time_index)
                .data_type(),
            row_index,
            operator_id,
            &self.spec.event_time_column,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "one prepared assignment carries its immutable row and key coordinates"
    )]
    fn prepare_assignment(
        &self,
        record: &RecordBatch,
        row_index: usize,
        start: EventTime,
        end: EventTime,
        stable_group_key: &[u8],
        group_values: &[Option<ScalarValue>],
        scratch: &mut BTreeMap<WindowKey, AccumulatorRow>,
        operator_id: &str,
    ) -> Result<()> {
        let key = WindowKey {
            start,
            end,
            stable_group_key: stable_group_key.to_vec(),
        };
        let accumulator = scratch.entry(key).or_insert_with(|| {
            self.state
                .accumulators
                .get(&WindowKey {
                    start,
                    end,
                    stable_group_key: stable_group_key.to_vec(),
                })
                .cloned()
                .unwrap_or_else(|| new_accumulator_row(&self.spec, &self.compiled, group_values))
        });
        update_accumulators(
            accumulator,
            record,
            row_index,
            &self.spec,
            &self.compiled,
            operator_id,
        )
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
        .map_err(|error| internal_error(&format!("window state encoder task failed: {error}")))?
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
        self.state
            .emitted_pending_snapshot
            .extend(keys.iter().cloned());
        Ok(())
    }
}

fn chunk_output_record(
    record: &RecordBatch,
    operator_id: &str,
    first_sequence: u64,
    budget: crate::EdgeBudget,
) -> Result<Vec<Batch>> {
    let row_costs = output_row_costs(record, budget)?;
    let ranges = output_chunk_ranges(&row_costs, record.num_rows(), budget)?;
    validate_output_sequence_range(operator_id, first_sequence, ranges.len())?;
    build_output_batches(record, operator_id, first_sequence, budget, ranges)
}

fn output_row_costs(record: &RecordBatch, budget: crate::EdgeBudget) -> Result<Vec<usize>> {
    let mut row_costs = Vec::with_capacity(record.num_rows());
    for index in 0..record.num_rows() {
        let row = record.slice(index, 1);
        let batch = Batch::table(vec![row], BatchMetadata::default())?;
        let bytes = batch.estimated_bytes()?;
        if bytes > budget.max_bytes {
            return Err(CalcFlowError::InvalidArgument {
                field: "message.bytes".into(),
                message: format!(
                    "one window output row requires {bytes} bytes, exceeding the effective edge byte budget {}",
                    budget.max_bytes
                ),
            });
        }
        row_costs.push(bytes);
    }
    Ok(row_costs)
}

fn output_chunk_ranges(
    row_costs: &[usize],
    row_count: usize,
    budget: crate::EdgeBudget,
) -> Result<Vec<(usize, usize)>> {
    let mut ranges = Vec::<(usize, usize)>::new();
    let mut start = 0;
    while start < row_count {
        let end = next_output_chunk_end(row_costs, start, row_count, budget);
        if end == start {
            return Err(internal_error(
                "validated output row did not fit the effective edge budget",
            ));
        }
        ranges.push((start, end));
        start = end;
    }
    Ok(ranges)
}

fn next_output_chunk_end(
    row_costs: &[usize],
    start: usize,
    row_count: usize,
    budget: crate::EdgeBudget,
) -> usize {
    let mut end = start;
    let mut bytes = 0_usize;
    while end < row_count && end - start < budget.max_rows {
        let Some(candidate_bytes) = bytes.checked_add(row_costs[end]) else {
            break;
        };
        if candidate_bytes > budget.max_bytes {
            break;
        }
        bytes = candidate_bytes;
        end += 1;
    }
    end
}

fn validate_output_sequence_range(
    operator_id: &str,
    first_sequence: u64,
    range_count: usize,
) -> Result<()> {
    let chunk_count = u64::try_from(range_count).map_err(|_| {
        operator_error(
            operator_id,
            "output chunk count does not fit the sequence range",
        )
    })?;
    first_sequence
        .checked_add(chunk_count)
        .ok_or_else(|| operator_error(operator_id, "output sequence overflowed before emission"))?;
    Ok(())
}

fn build_output_batches(
    record: &RecordBatch,
    operator_id: &str,
    first_sequence: u64,
    budget: crate::EdgeBudget,
    ranges: Vec<(usize, usize)>,
) -> Result<Vec<Batch>> {
    ranges
        .into_iter()
        .enumerate()
        .map(|(ordinal, (start, end))| {
            let ordinal = u64::try_from(ordinal)
                .map_err(|_| internal_error("output chunk ordinal does not fit UInt64"))?;
            let sequence = first_sequence
                .checked_add(ordinal)
                .expect("complete sequence range validated above");
            let metadata = BatchMetadata::new(operator_id, sequence, BTreeMap::new())?;
            let batch = Batch::table(vec![record.slice(start, end - start)], metadata)?;
            if batch.estimated_bytes()? > budget.max_bytes {
                return Err(internal_error(
                    "conservative row charges underreported a window output chunk",
                ));
            }
            Ok(batch)
        })
        .collect()
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
        let encoded = self
            .encode_operations(Self::upsert_operations(&update), context)
            .await?;
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
        let tombstones = self
            .encode_operations(self.tombstone_operations(&keys), context)
            .await?;
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
        let tombstones = self
            .encode_operations(self.tombstone_operations(&keys), context)
            .await?;
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
                "window checkpoint epoch did not advance strictly".into(),
            ));
        }
        let prepared = self.prepare_snapshot_segments(epoch)?;
        let retained_inventory = self.next_snapshot_inventory(prepared.descriptors)?;
        let metadata = WindowSnapshotMetadata {
            state_layout_version: WINDOW_STATE_LAYOUT_VERSION,
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
            self.state.accumulators.remove(&key);
            self.state.dirty.remove(&key);
        }
        self.state.dirty.clear();
        self.state.retained_inventory = retained_inventory;
        self.state.replace_retained_on_checkpoint = false;
        self.state.last_checkpoint_epoch = Some(epoch);
        Ok(crate::OperatorStateSnapshot {
            inline_metadata: inline_metadata.into_iter().collect(),
            segments: prepared.bytes,
        })
    }

    fn restore(&mut self, snapshot: &crate::OperatorStateSnapshot) -> Result<()> {
        if snapshot.inline_metadata.is_empty() && snapshot.segments.is_empty() {
            return self.reset();
        }
        let metadata = parse_snapshot_metadata(snapshot)?;
        let inventory = validate_snapshot_metadata(&metadata, &self.compiled, snapshot)?;
        let decoded = self.decode_snapshot_segments(snapshot, &metadata)?;
        self.install_restored_state(metadata, inventory, decoded);
        Ok(())
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
            checkpoint_mismatch("window segment is missing its operator identity".into())
        })?;
        let relative_path = format!(
            "committed/{operator_id}/{:020}-{segment_id}.arrow",
            epoch.as_u64()
        );
        let byte_len = u64::try_from(segment.bytes().len())
            .map_err(|_| internal_error("window segment length does not fit u64"))?;
        Ok(SegmentDescriptor {
            kind,
            state_layout_version: WINDOW_STATE_LAYOUT_VERSION,
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
        let replacement = self
            .state
            .retained_inventory
            .replacement_after_full_compaction(base.clone())?;
        let mut retained = replacement.segments().to_vec();
        retained.extend_from_slice(later);
        StateInventory::new(retained)
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
        for (key, accumulator) in update.accumulators {
            self.state.dirty.insert(key.clone());
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
        decoded: BTreeMap<WindowKey, AccumulatorRow>,
    ) {
        self.state = WindowState {
            accumulators: decoded,
            last_input_watermark: metadata.last_input_watermark,
            next_output_sequence: metadata.next_output_sequence,
            ended: metadata.ended,
            metrics: metadata.metrics,
            retained_inventory: inventory,
            last_checkpoint_epoch: Some(metadata.epoch),
            pipeline_fingerprint: metadata.pipeline_fingerprint,
            operator_id: metadata.operator_id,
            ..WindowState::default()
        };
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
) -> Result<Vec<Arc<Vec<u8>>>> {
    inventory
        .iter()
        .map(|descriptor| {
            let segment_id = descriptor.handle.segment_id();
            let segment = snapshot.segments.get(segment_id).ok_or_else(|| {
                checkpoint_mismatch(format!("window snapshot is missing segment {segment_id:?}"))
            })?;
            validate_snapshot_segment_bytes(descriptor, segment.bytes())?;
            Ok(segment.bytes_arc())
        })
        .collect()
}

fn validate_snapshot_segment_bytes(descriptor: &SegmentDescriptor, bytes: &[u8]) -> Result<()> {
    if u64::try_from(bytes.len()).ok() != Some(descriptor.handle.byte_len()) {
        return Err(checkpoint_mismatch(
            "window snapshot segment byte length does not match its handle".into(),
        ));
    }
    if hex::encode(Sha256::digest(bytes)) != descriptor.handle.sha256() {
        return Err(checkpoint_mismatch(
            "window snapshot segment checksum does not match its handle".into(),
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

fn partition_open_assignments(
    assignments: Vec<(EventTime, EventTime)>,
    watermark: Option<EventTime>,
    metrics: &mut PreparedInputMetrics,
    operator_id: &str,
) -> Result<Vec<(EventTime, EventTime)>> {
    let mut open = Vec::with_capacity(assignments.len());
    for assignment in assignments {
        if let Some(closing_watermark) = watermark.filter(|value| assignment.1 <= *value) {
            record_late_assignment(metrics, closing_watermark, assignment.1, operator_id)?;
        } else {
            open.push(assignment);
        }
    }
    Ok(open)
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

fn event_time_at(
    array: &dyn Array,
    data_type: &DataType,
    row: usize,
    operator_id: &str,
    column: &str,
) -> Result<Option<EventTime>> {
    if array.is_null(row) {
        return Ok(None);
    }
    let timestamp_value = match data_type {
        DataType::Timestamp(TimeUnit::Second, _) => downcast_array::<TimestampSecondArray>(
            array,
            operator_id,
            "event-time timestamp(second)",
        )?
        .value(row),
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            downcast_array::<TimestampMillisecondArray>(
                array,
                operator_id,
                "event-time timestamp(millisecond)",
            )?
            .value(row)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            downcast_array::<TimestampMicrosecondArray>(
                array,
                operator_id,
                "event-time timestamp(microsecond)",
            )?
            .value(row)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => downcast_array::<TimestampNanosecondArray>(
            array,
            operator_id,
            "event-time timestamp(nanosecond)",
        )?
        .value(row),
        _ => {
            return Err(operator_error(
                operator_id,
                "compiled event-time column is not a timestamp",
            ));
        }
    };
    EventTime::import_timestamp(timestamp_value, data_type, column)
        .map(Some)
        .map_err(|error| {
            operator_error(
                operator_id,
                &format!("event-time conversion failed: {error}"),
            )
        })
}

fn window_assignments(
    event_time: EventTime,
    geometry: CompiledWindowGeometry,
) -> std::result::Result<Vec<(EventTime, EventTime)>, String> {
    let time = i128::from(event_time.as_micros());
    let size = i128::from(geometry.size_micros);
    let slide = i128::from(geometry.slide_micros);
    let latest_start = time.div_euclid(slide) * slide;
    let mut assignments = Vec::with_capacity(
        usize::try_from(geometry.overlap)
            .map_err(|_| "window overlap does not fit usize".to_string())?,
    );
    for offset in (0..geometry.overlap).rev() {
        assignments.push(window_assignment(latest_start, slide, size, offset)?);
    }
    Ok(assignments)
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

fn encode_group_key(
    record: &RecordBatch,
    row: usize,
    columns: &[CompiledGroupColumn],
    operator_id: &str,
    names: &[String],
) -> Result<(Vec<u8>, Vec<Option<ScalarValue>>)> {
    let mut encoded = Vec::new();
    let mut values = Vec::with_capacity(columns.len());
    for (ordinal, column) in columns.iter().enumerate() {
        let value = scalar_at(
            record.column(column.index).as_ref(),
            &column.data_type,
            row,
            operator_id,
        )?;
        encode_group_scalar(&mut encoded, &column.data_type, value.as_ref()).map_err(
            |message| {
                operator_error(
                    operator_id,
                    &format!(
                        "window.group_by[{ordinal}] ({:?}) encoding failed: {message}",
                        names[ordinal]
                    ),
                )
            },
        )?;
        values.push(value);
    }
    Ok((encoded, values))
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
    group_values: &[Option<ScalarValue>],
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
                DataType::Float32 | DataType::Float64 => {
                    AccumulatorValue::FloatAverage { sum: 0.0, count: 0 }
                }
                _ => unreachable!("aggregate matrix validated at construction"),
            },
        })
        .collect();
    AccumulatorRow {
        group_values: group_values.to_vec(),
        aggregates,
    }
}

fn update_accumulators(
    row: &mut AccumulatorRow,
    record: &RecordBatch,
    row_index: usize,
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    operator_id: &str,
) -> Result<()> {
    for (ordinal, ((aggregate, compiled), accumulator)) in spec
        .aggregates
        .iter()
        .zip(&compiled.aggregates)
        .zip(&mut row.aggregates)
        .enumerate()
    {
        let array = record.column(compiled.input_index);
        if array.is_null(row_index) {
            continue;
        }
        if aggregate.function == AggregateFunction::Count {
            update_accumulator(accumulator, aggregate.function, ScalarValue::Unsigned(0)).map_err(
                |message| {
                    operator_error(
                        operator_id,
                        &format!("window.aggregates[{ordinal}] update failed: {message}"),
                    )
                },
            )?;
            continue;
        }
        let value = scalar_at(array.as_ref(), &compiled.input_type, row_index, operator_id)?
            .expect("non-null array row produces a scalar");
        update_accumulator(accumulator, aggregate.function, value).map_err(|message| {
            operator_error(
                operator_id,
                &format!("window.aggregates[{ordinal}] update failed: {message}"),
            )
        })?;
    }
    Ok(())
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
            *sum = Some(canonical_float_add(sum.unwrap_or(0.0), float_value(value)?));
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
            *sum = canonical_float_add(*sum, float_value(value)?);
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
            Ok(value.map(|value| ScalarValue::Float64(value.to_bits())))
        }
        AccumulatorValue::Min(value) | AccumulatorValue::Max(value) => Ok(value.clone()),
        AccumulatorValue::SignedAverage { sum, count } => {
            Ok((*count != 0).then(|| ScalarValue::Float64(signed_average(*sum, *count).to_bits())))
        }
        AccumulatorValue::UnsignedAverage { sum, count } => {
            Ok((*count != 0)
                .then(|| ScalarValue::Float64(unsigned_average(*sum, *count).to_bits())))
        }
        AccumulatorValue::FloatAverage { sum, count } => {
            Ok((*count != 0).then(|| ScalarValue::Float64(float_average(*sum, *count).to_bits())))
        }
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

fn canonical_float_add(left: f64, right: f64) -> f64 {
    canonicalize_float(left + right)
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
    scalar_non_null_at(array, data_type, row, operator_id).map(Some)
}

fn scalar_non_null_at(
    array: &dyn Array,
    data_type: &DataType,
    row: usize,
    operator_id: &str,
) -> Result<ScalarValue> {
    match data_type {
        DataType::Boolean => Ok(ScalarValue::Boolean(
            downcast_array::<BooleanArray>(array, operator_id, "Boolean")?.value(row),
        )),
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            signed_scalar_at(array, data_type, row, operator_id)
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            unsigned_scalar_at(array, data_type, row, operator_id)
        }
        DataType::Float32 | DataType::Float64 => {
            float_scalar_at(array, data_type, row, operator_id)
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            string_scalar_at(array, data_type, row, operator_id)
        }
        DataType::Date32 | DataType::Date64 | DataType::Timestamp(TimeUnit::Microsecond, _) => {
            temporal_scalar_at(array, data_type, row, operator_id)
        }
        _ => Err(operator_error(
            operator_id,
            &format!("compiled scalar type {data_type} is unsupported"),
        )),
    }
}

fn signed_scalar_at(
    array: &dyn Array,
    data_type: &DataType,
    row: usize,
    operator_id: &str,
) -> Result<ScalarValue> {
    let value = match data_type {
        DataType::Int8 => {
            i64::from(downcast_array::<Int8Array>(array, operator_id, "Int8")?.value(row))
        }
        DataType::Int16 => {
            i64::from(downcast_array::<Int16Array>(array, operator_id, "Int16")?.value(row))
        }
        DataType::Int32 => {
            i64::from(downcast_array::<Int32Array>(array, operator_id, "Int32")?.value(row))
        }
        DataType::Int64 => downcast_array::<Int64Array>(array, operator_id, "Int64")?.value(row),
        _ => return Err(internal_error("compiled signed scalar type mismatch")),
    };
    Ok(ScalarValue::Signed(value))
}

fn unsigned_scalar_at(
    array: &dyn Array,
    data_type: &DataType,
    row: usize,
    operator_id: &str,
) -> Result<ScalarValue> {
    let value = match data_type {
        DataType::UInt8 => {
            u64::from(downcast_array::<UInt8Array>(array, operator_id, "UInt8")?.value(row))
        }
        DataType::UInt16 => {
            u64::from(downcast_array::<UInt16Array>(array, operator_id, "UInt16")?.value(row))
        }
        DataType::UInt32 => {
            u64::from(downcast_array::<UInt32Array>(array, operator_id, "UInt32")?.value(row))
        }
        DataType::UInt64 => downcast_array::<UInt64Array>(array, operator_id, "UInt64")?.value(row),
        _ => return Err(internal_error("compiled unsigned scalar type mismatch")),
    };
    Ok(ScalarValue::Unsigned(value))
}

fn float_scalar_at(
    array: &dyn Array,
    data_type: &DataType,
    row: usize,
    operator_id: &str,
) -> Result<ScalarValue> {
    match data_type {
        DataType::Float32 => Ok(ScalarValue::Float32(
            downcast_array::<Float32Array>(array, operator_id, "Float32")?
                .value(row)
                .to_bits(),
        )),
        DataType::Float64 => Ok(ScalarValue::Float64(
            downcast_array::<Float64Array>(array, operator_id, "Float64")?
                .value(row)
                .to_bits(),
        )),
        _ => Err(internal_error("compiled float scalar type mismatch")),
    }
}

fn string_scalar_at(
    array: &dyn Array,
    data_type: &DataType,
    row: usize,
    operator_id: &str,
) -> Result<ScalarValue> {
    let value = match data_type {
        DataType::Utf8 => downcast_array::<StringArray>(array, operator_id, "Utf8")?.value(row),
        DataType::LargeUtf8 => {
            downcast_array::<LargeStringArray>(array, operator_id, "LargeUtf8")?.value(row)
        }
        _ => return Err(internal_error("compiled string scalar type mismatch")),
    };
    Ok(ScalarValue::String(value.into()))
}

fn temporal_scalar_at(
    array: &dyn Array,
    data_type: &DataType,
    row: usize,
    operator_id: &str,
) -> Result<ScalarValue> {
    match data_type {
        DataType::Date32 => Ok(ScalarValue::Date32(
            downcast_array::<Date32Array>(array, operator_id, "Date32")?.value(row),
        )),
        DataType::Date64 => Ok(ScalarValue::Date64(
            downcast_array::<Date64Array>(array, operator_id, "Date64")?.value(row),
        )),
        DataType::Timestamp(TimeUnit::Microsecond, _) => Ok(ScalarValue::Timestamp(
            downcast_array::<TimestampMicrosecondArray>(
                array,
                operator_id,
                "Timestamp(Microsecond)",
            )?
            .value(row),
        )),
        _ => Err(internal_error("compiled temporal scalar type mismatch")),
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

fn internal_error(message: &str) -> CalcFlowError {
    CalcFlowError::Internal {
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
    let schema = Schema::new(state_fields(spec, compiled));
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
    Schema::new_with_metadata(
        state_fields(spec, compiled),
        HashMap::from([
            (
                "calc_flow.state_layout_version".into(),
                WINDOW_STATE_LAYOUT_VERSION.to_string(),
            ),
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
                compiled.state_schema_fingerprint.clone(),
            ),
            ("calc_flow.group_key_encoding".into(), "g1".into()),
        ]),
    )
}

fn state_fields(spec: &WindowSpec, compiled: &CompiledWindowSpec) -> Vec<Field> {
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
            operations
                .iter()
                .map(|row| row.key.stable_group_key.as_slice()),
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
            arrays.push(signed_average_state_array(operations, ordinal)?);
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            arrays.push(unsigned_average_state_array(operations, ordinal)?);
        }
        DataType::Float32 | DataType::Float64 => {
            arrays.push(float_average_state_array(operations, ordinal, operator_id)?);
        }
        _ => unreachable!("average input matrix validated at construction"),
    }
    arrays.push(average_count_array(operations, ordinal)?);
    Ok(())
}

fn signed_average_state_array(
    operations: &[StateOperationRow],
    ordinal: usize,
) -> Result<ArrayRef> {
    let values = operations
        .iter()
        .map(
            |row| match (&row.entry.aggregates[ordinal], row.tombstone) {
                (_, true) => Ok(None),
                (AccumulatorValue::SignedAverage { sum, .. }, false) => Ok(Some(sum.to_be_bytes())),
                _ => Err(internal_error("signed average state type mismatch")),
            },
        )
        .collect::<Result<Vec<_>>>()?;
    fixed_average_state_array(values, "signed")
}

fn unsigned_average_state_array(
    operations: &[StateOperationRow],
    ordinal: usize,
) -> Result<ArrayRef> {
    let values = operations
        .iter()
        .map(
            |row| match (&row.entry.aggregates[ordinal], row.tombstone) {
                (_, true) => Ok(None),
                (AccumulatorValue::UnsignedAverage { sum, .. }, false) => {
                    Ok(Some(sum.to_be_bytes()))
                }
                _ => Err(internal_error("unsigned average state type mismatch")),
            },
        )
        .collect::<Result<Vec<_>>>()?;
    fixed_average_state_array(values, "unsigned")
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
                    Ok(Some(ScalarValue::Float64(sum.to_bits())))
                }
                _ => Err(internal_error("float average state type mismatch")),
            },
        )
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
            Ok(value.map(|value| ScalarValue::Float64(value.to_bits())))
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
    compiled: &CompiledWindowSpec,
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<StateInventory> {
    validate_snapshot_header(metadata, compiled)?;
    let inventory = StateInventory::new(metadata.segment_inventory.clone())
        .map_err(|error| checkpoint_mismatch(error.to_string()))?;
    validate_snapshot_inventory(metadata, compiled, &inventory)?;
    validate_snapshot_segment_set(snapshot, &inventory)?;
    validate_snapshot_identity(metadata, snapshot)?;
    Ok(inventory)
}

fn validate_snapshot_header(
    metadata: &WindowSnapshotMetadata,
    compiled: &CompiledWindowSpec,
) -> Result<()> {
    if metadata.state_layout_version != WINDOW_STATE_LAYOUT_VERSION {
        return Err(checkpoint_mismatch(format!(
            "window state layout version {} does not match expected {}",
            metadata.state_layout_version, WINDOW_STATE_LAYOUT_VERSION
        )));
    }
    if metadata.configuration_hash != compiled.configuration_hash {
        return Err(checkpoint_mismatch(
            "window operator configuration hash does not match the compiled operator".into(),
        ));
    }
    if metadata.state_schema_fingerprint != compiled.state_schema_fingerprint {
        return Err(checkpoint_mismatch(
            "window state schema fingerprint does not match the compiled operator".into(),
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
            "window snapshot segment IDs are missing, extra, duplicated, or non-canonical".into(),
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
            "window segments require pipeline and operator identity metadata".into(),
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
            "window pipeline fingerprint is not lowercase SHA-256".into(),
        ));
    }
    Ok(())
}

fn validate_snapshot_operator_id(operator_id: Option<&str>) -> Result<()> {
    if operator_id.is_some_and(|operator_id| operator_id.is_empty() || operator_id.contains('\0')) {
        return Err(checkpoint_mismatch(
            "window operator ID is empty or contains NUL".into(),
        ));
    }
    Ok(())
}

fn validate_snapshot_inventory(
    metadata: &WindowSnapshotMetadata,
    compiled: &CompiledWindowSpec,
    inventory: &StateInventory,
) -> Result<()> {
    for descriptor in inventory.segments() {
        if descriptor.state_layout_version != WINDOW_STATE_LAYOUT_VERSION
            || descriptor.schema_fingerprint != compiled.state_schema_fingerprint
        {
            return Err(checkpoint_mismatch(
                "window segment inventory layout or schema does not match the compiled operator"
                    .into(),
            ));
        }
        if descriptor.handle.epoch() > metadata.epoch {
            return Err(checkpoint_mismatch(
                "window segment inventory contains a future epoch".into(),
            ));
        }
        if metadata.operator_id.as_deref() != Some(descriptor.handle.operator_id()) {
            return Err(checkpoint_mismatch(
                "window segment inventory operator does not match snapshot metadata".into(),
            ));
        }
    }
    Ok(())
}

fn decode_state_segments(
    segments: &[Arc<Vec<u8>>],
    spec: &WindowSpec,
    compiled: &CompiledWindowSpec,
    pipeline_fingerprint: Option<&str>,
    operator_id: Option<&str>,
) -> Result<BTreeMap<WindowKey, AccumulatorRow>> {
    if segments.is_empty() {
        return Ok(BTreeMap::new());
    }
    let pipeline_fingerprint = pipeline_fingerprint.ok_or_else(|| {
        checkpoint_mismatch("window state is missing its pipeline fingerprint".into())
    })?;
    let operator_id = operator_id
        .ok_or_else(|| checkpoint_mismatch("window state is missing its operator ID".into()))?;
    let expected_schema = state_schema(spec, compiled, pipeline_fingerprint, operator_id);
    let decoded = segments
        .iter()
        .map(|bytes| {
            decode_state_segment(bytes, spec, compiled, &expected_schema, operator_id).map(
                |operations| {
                    operations
                        .into_iter()
                        .map(|(key, entry)| {
                            let operation =
                                entry.map_or(StateOperation::Tombstone, StateOperation::Upsert);
                            (key, operation)
                        })
                        .collect::<Vec<_>>()
                },
            )
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
            "window state Arrow schema or metadata does not match the compiled operator".into(),
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
            stable_group_key: stable_keys.value(row).to_vec(),
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
        if encoded_group != key.stable_group_key {
            return Err(checkpoint_mismatch(
                "window state stable group key does not match its declared group values".into(),
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
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "state decoding names every durable aggregate coordinate and aggregate matrix branch explicitly"
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
) -> Result<(Option<AccumulatorValue>, usize)> {
    let value = record.column(column_index);
    if tombstone {
        if !value.is_null(row)
            || (function == AggregateFunction::Avg && !record.column(column_index + 1).is_null(row))
        {
            return Err(state_format(format!(
                "window tombstone aggregate {ordinal} contains state"
            )));
        }
        return Ok((
            None,
            column_index + 1 + usize::from(function == AggregateFunction::Avg),
        ));
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
            DataType::Float64 => AccumulatorValue::FloatSum(
                scalar_at(value.as_ref(), &DataType::Float64, row, operator_id)
                    .map_err(|error| state_format(error.to_string()))?
                    .map(|value| float_value(&value))
                    .transpose()
                    .map_err(state_format)?,
            ),
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
        AggregateFunction::Avg => {
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
            match compiled.input_type {
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                    let bytes = value
                        .as_any()
                        .downcast_ref::<FixedSizeBinaryArray>()
                        .ok_or_else(|| {
                            state_format(format!(
                                "window average aggregate {ordinal} has invalid binary state"
                            ))
                        })?
                        .value(row)
                        .try_into()
                        .map_err(|_| state_format("signed average state is not 16 bytes"))?;
                    AccumulatorValue::SignedAverage {
                        sum: i128::from_be_bytes(bytes),
                        count,
                    }
                }
                DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                    let bytes = value
                        .as_any()
                        .downcast_ref::<FixedSizeBinaryArray>()
                        .ok_or_else(|| {
                            state_format(format!(
                                "window average aggregate {ordinal} has invalid binary state"
                            ))
                        })?
                        .value(row)
                        .try_into()
                        .map_err(|_| state_format("unsigned average state is not 16 bytes"))?;
                    AccumulatorValue::UnsignedAverage {
                        sum: u128::from_be_bytes(bytes),
                        count,
                    }
                }
                DataType::Float32 | DataType::Float64 => {
                    let Some(ScalarValue::Float64(bits)) =
                        scalar_at(value.as_ref(), &DataType::Float64, row, operator_id)
                            .map_err(|error| state_format(error.to_string()))?
                    else {
                        return Err(state_format(format!(
                            "window average aggregate {ordinal} has invalid float state"
                        )));
                    };
                    AccumulatorValue::FloatAverage {
                        sum: f64::from_bits(bits),
                        count,
                    }
                }
                _ => unreachable!("average input matrix validated at construction"),
            }
        }
    };
    Ok((
        Some(decoded),
        column_index + 1 + usize::from(function == AggregateFunction::Avg),
    ))
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
            "window state key does not match the compiled geometry".into(),
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

fn compile_error(message: String) -> CalcFlowError {
    CalcFlowError::Compile { message }
}

fn checkpoint_mismatch(message: String) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch { message }
}

fn state_format(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::Format {
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

    const HIGH_CARDINALITY_ROWS: usize = 400_000;
    const LEGACY_PROJECT_JSON_LIMIT: usize = 10 * 1024 * 1024;

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
