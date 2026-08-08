use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use datafusion::arrow::{
    array::{
        Array, ArrayRef, BooleanArray, Date32Array, Date64Array, Float32Array, Float64Array,
        Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray, StringArray,
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, EventTime, JsonMap, Port, Result,
    StreamCollector, StreamOperator, StreamOperatorContext, canonical_json,
};

use super::{OperatorMetadata, validate_operator_name};

/// Maximum number of concrete hopping-window assignments for one input row.
pub const MAX_WINDOW_OVERLAP: u64 = 1_024;

const WINDOW_STATE_LAYOUT_VERSION: u32 = 1;
const MAX_GROUP_KEY_BYTES: usize = 65_536;

/// Aggregate function supported by the first built-in window operator.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
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

        let mut group_names = BTreeSet::new();
        for (index, column) in self.group_by.iter().enumerate() {
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
            if !group_names.insert(column) {
                return Err(invalid_argument(
                    &field,
                    "duplicates an earlier group column",
                ));
            }
        }

        let mut aggregate_outputs = BTreeSet::new();
        for (index, aggregate) in self.aggregates.iter().enumerate() {
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
            if !aggregate_outputs.insert(&aggregate.output) {
                return Err(invalid_argument(
                    &output_field,
                    "duplicates an earlier aggregate output",
                ));
            }
        }
        Ok(())
    }
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
}

struct CompiledGroupColumn {
    index: usize,
    data_type: DataType,
}

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

    fn apply_input_batch(
        &mut self,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        if self.state.ended {
            return Err(operator_error(
                context.operator_id(),
                "received data after end-of-input",
            ));
        }
        let table = batch.table_payload()?;
        let mut scratch = BTreeMap::<WindowKey, AccumulatorRow>::new();
        let mut late_rows = 0_u64;
        let mut max_lateness = None::<u64>;

        for record in table.batches() {
            for row_index in 0..record.num_rows() {
                let Some(event_time) = event_time_at(
                    record.column(self.compiled.event_time_index).as_ref(),
                    record
                        .schema()
                        .field(self.compiled.event_time_index)
                        .data_type(),
                    row_index,
                    context.operator_id(),
                    &self.spec.event_time_column,
                )?
                else {
                    continue;
                };
                let assignments = window_assignments(event_time, self.compiled.geometry)
                    .map_err(|message| operator_error(context.operator_id(), &message))?;
                let mut open_assignments = Vec::with_capacity(assignments.len());
                for (start, end) in assignments {
                    if let Some(watermark) = context.input_watermark()
                        && end <= watermark
                    {
                        late_rows = late_rows.checked_add(1).ok_or_else(|| {
                            operator_error(context.operator_id(), "late row counter overflowed")
                        })?;
                        let lateness = watermark
                            .as_micros()
                            .checked_sub(end.as_micros())
                            .and_then(|value| u64::try_from(value).ok())
                            .ok_or_else(|| {
                                operator_error(
                                    context.operator_id(),
                                    "late assignment distance overflowed",
                                )
                            })?;
                        max_lateness = Some(max_lateness.map_or(lateness, |max| max.max(lateness)));
                    } else {
                        open_assignments.push((start, end));
                    }
                }
                if open_assignments.is_empty() {
                    continue;
                }

                let (stable_group_key, group_values) = encode_group_key(
                    record,
                    row_index,
                    &self.compiled.group_columns,
                    context.operator_id(),
                    &self.spec.group_by,
                )?;
                for (start, end) in open_assignments {
                    let key = WindowKey {
                        start,
                        end,
                        stable_group_key: stable_group_key.clone(),
                    };
                    if !scratch.contains_key(&key) {
                        let accumulator = self
                            .state
                            .accumulators
                            .get(&key)
                            .cloned()
                            .unwrap_or_else(|| {
                                new_accumulator_row(&self.spec, &self.compiled, &group_values)
                            });
                        scratch.insert(key.clone(), accumulator);
                    }
                    let accumulator = scratch
                        .get_mut(&key)
                        .expect("scratch accumulator inserted above");
                    update_accumulators(
                        accumulator,
                        record,
                        row_index,
                        &self.spec,
                        &self.compiled,
                        context.operator_id(),
                    )?;
                }
            }
        }

        for (key, accumulator) in scratch {
            self.state.dirty.insert(key.clone());
            self.state.accumulators.insert(key, accumulator);
        }
        if late_rows > 0 {
            context.record_late_rows(late_rows, max_lateness.map(Duration::from_micros))?;
        }
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
        let next_sequence = self
            .state
            .next_output_sequence
            .checked_add(1)
            .ok_or_else(|| operator_error(context.operator_id(), "output sequence overflowed"))?;
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
        let metadata = BatchMetadata::new(
            context.operator_id(),
            self.state.next_output_sequence,
            BTreeMap::new(),
        )?;
        let batch = Batch::table(vec![record], metadata)?;
        output.emit("output", batch).await?;
        self.state.next_output_sequence = next_sequence;
        self.state
            .emitted_pending_snapshot
            .extend(keys.iter().cloned());
        Ok(())
    }
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
        if ingress != "input" {
            return Err(CalcFlowError::Operator {
                node_id: self.name.clone(),
                message: format!("unknown ingress {ingress:?}; expected \"input\""),
            });
        }
        self.input_ports[0].validate(&batch, &format!("{}.input", self.name))?;
        self.apply_input_batch(&batch, context)
    }

    async fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
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
        let keys = self
            .state
            .accumulators
            .keys()
            .filter(|key| {
                key.end <= watermark && !self.state.emitted_pending_snapshot.contains(*key)
            })
            .cloned()
            .collect::<Vec<_>>();
        self.emit_keys(&keys, context, output).await?;
        self.state.last_input_watermark = Some(watermark);
        Ok(())
    }

    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if self.state.ended {
            return Ok(());
        }
        let keys = self
            .state
            .accumulators
            .keys()
            .filter(|key| !self.state.emitted_pending_snapshot.contains(*key))
            .cloned()
            .collect::<Vec<_>>();
        self.emit_keys(&keys, context, output).await?;
        self.state.ended = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.state = WindowState::default();
        Ok(())
    }
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
        let start = latest_start
            .checked_sub(i128::from(offset) * slide)
            .ok_or_else(|| "window assignment start overflowed".to_string())?;
        let end = start
            .checked_add(size)
            .ok_or_else(|| "window assignment end overflowed".to_string())?;
        let start = i64::try_from(start)
            .map_err(|_| "window assignment start is outside EventTime".to_string())?;
        let end = i64::try_from(end)
            .map_err(|_| "window assignment end is outside EventTime".to_string())?;
        assignments.push((EventTime::from_micros(start), EventTime::from_micros(end)));
    }
    Ok(assignments)
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
    match (data_type, value) {
        (DataType::Boolean, ScalarValue::Boolean(value)) => {
            extend_group_encoding(encoded, &[u8::from(*value)])?;
        }
        (DataType::Int8, ScalarValue::Signed(value)) => {
            let value =
                i8::try_from(*value).map_err(|_| "Int8 group scalar escaped its compiled range")?;
            extend_group_encoding(encoded, &[value.to_be_bytes()[0] ^ 0x80])?;
        }
        (DataType::Int16, ScalarValue::Signed(value)) => {
            let mut bytes = i16::try_from(*value)
                .map_err(|_| "Int16 group scalar escaped its compiled range")?
                .to_be_bytes();
            bytes[0] ^= 0x80;
            extend_group_encoding(encoded, &bytes)?;
        }
        (DataType::Int32, ScalarValue::Signed(value)) => {
            let mut bytes = i32::try_from(*value)
                .map_err(|_| "Int32 group scalar escaped its compiled range")?
                .to_be_bytes();
            bytes[0] ^= 0x80;
            extend_group_encoding(encoded, &bytes)?;
        }
        (DataType::Int64, ScalarValue::Signed(value)) => {
            let mut bytes = value.to_be_bytes();
            bytes[0] ^= 0x80;
            extend_group_encoding(encoded, &bytes)?;
        }
        (DataType::UInt8, ScalarValue::Unsigned(value)) => {
            let value = u8::try_from(*value)
                .map_err(|_| "UInt8 group scalar escaped its compiled range")?;
            extend_group_encoding(encoded, &[value])?;
        }
        (DataType::UInt16, ScalarValue::Unsigned(value)) => {
            let value = u16::try_from(*value)
                .map_err(|_| "UInt16 group scalar escaped its compiled range")?;
            extend_group_encoding(encoded, &value.to_be_bytes())?;
        }
        (DataType::UInt32, ScalarValue::Unsigned(value)) => {
            let value = u32::try_from(*value)
                .map_err(|_| "UInt32 group scalar escaped its compiled range")?;
            extend_group_encoding(encoded, &value.to_be_bytes())?;
        }
        (DataType::UInt64, ScalarValue::Unsigned(value)) => {
            extend_group_encoding(encoded, &value.to_be_bytes())?;
        }
        (DataType::Float32, ScalarValue::Float32(bits)) => {
            let ordered = if bits & (1 << 31) != 0 {
                !bits
            } else {
                bits | (1 << 31)
            };
            extend_group_encoding(encoded, &ordered.to_be_bytes())?;
        }
        (DataType::Float64, ScalarValue::Float64(bits)) => {
            let ordered = if bits & (1 << 63) != 0 {
                !bits
            } else {
                bits | (1 << 63)
            };
            extend_group_encoding(encoded, &ordered.to_be_bytes())?;
        }
        (DataType::Utf8 | DataType::LargeUtf8, ScalarValue::String(value)) => {
            for byte in value.as_bytes() {
                if *byte == 0 {
                    extend_group_encoding(encoded, &[0x00, 0xff])?;
                } else {
                    extend_group_encoding(encoded, &[*byte])?;
                }
            }
            extend_group_encoding(encoded, &[0x00, 0x00])?;
        }
        (DataType::Date32, ScalarValue::Date32(value)) => {
            let mut bytes = value.to_be_bytes();
            bytes[0] ^= 0x80;
            extend_group_encoding(encoded, &bytes)?;
        }
        (DataType::Date64, ScalarValue::Date64(value))
        | (DataType::Timestamp(TimeUnit::Microsecond, _), ScalarValue::Timestamp(value)) => {
            let mut bytes = value.to_be_bytes();
            bytes[0] ^= 0x80;
            extend_group_encoding(encoded, &bytes)?;
        }
        _ => return Err("compiled group scalar type mismatch".into()),
    }
    Ok(())
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
    match (function, accumulator) {
        (AggregateFunction::Count, AccumulatorValue::Count(count)) => {
            *count = count
                .checked_add(1)
                .ok_or_else(|| "count overflowed UInt64".to_string())?;
        }
        (AggregateFunction::Sum, AccumulatorValue::SignedSum(sum)) => {
            let value = signed_value(&value)?;
            let updated = sum
                .unwrap_or(0)
                .checked_add(i128::from(value))
                .ok_or_else(|| "signed sum overflowed its widened state".to_string())?;
            i64::try_from(updated).map_err(|_| "signed sum overflowed Int64".to_string())?;
            *sum = Some(updated);
        }
        (AggregateFunction::Sum, AccumulatorValue::UnsignedSum(sum)) => {
            let value = unsigned_value(&value)?;
            let updated = sum
                .unwrap_or(0)
                .checked_add(u128::from(value))
                .ok_or_else(|| "unsigned sum overflowed its widened state".to_string())?;
            u64::try_from(updated).map_err(|_| "unsigned sum overflowed UInt64".to_string())?;
            *sum = Some(updated);
        }
        (AggregateFunction::Sum, AccumulatorValue::FloatSum(sum)) => {
            *sum = Some(canonical_float_add(
                sum.unwrap_or(0.0),
                float_value(&value)?,
            ));
        }
        (AggregateFunction::Min, AccumulatorValue::Min(current)) => {
            if current
                .as_ref()
                .is_none_or(|current| scalar_total_cmp(&value, current) == Ordering::Less)
            {
                *current = Some(value);
            }
        }
        (AggregateFunction::Max, AccumulatorValue::Max(current)) => {
            if current
                .as_ref()
                .is_none_or(|current| scalar_total_cmp(&value, current) == Ordering::Greater)
            {
                *current = Some(value);
            }
        }
        (AggregateFunction::Avg, AccumulatorValue::SignedAverage { sum, count }) => {
            *sum = sum
                .checked_add(i128::from(signed_value(&value)?))
                .ok_or_else(|| "signed average sum overflowed Int128".to_string())?;
            *count = count
                .checked_add(1)
                .ok_or_else(|| "average count overflowed UInt64".to_string())?;
        }
        (AggregateFunction::Avg, AccumulatorValue::UnsignedAverage { sum, count }) => {
            *sum = sum
                .checked_add(u128::from(unsigned_value(&value)?))
                .ok_or_else(|| "unsigned average sum overflowed UInt128".to_string())?;
            *count = count
                .checked_add(1)
                .ok_or_else(|| "average count overflowed UInt64".to_string())?;
        }
        (AggregateFunction::Avg, AccumulatorValue::FloatAverage { sum, count }) => {
            *sum = canonical_float_add(*sum, float_value(&value)?);
            *count = count
                .checked_add(1)
                .ok_or_else(|| "average count overflowed UInt64".to_string())?;
        }
        _ => return Err("compiled aggregate accumulator type mismatch".into()),
    }
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
    let value = match data_type {
        DataType::Boolean => ScalarValue::Boolean(
            downcast_array::<BooleanArray>(array, operator_id, "Boolean")?.value(row),
        ),
        DataType::Int8 => ScalarValue::Signed(i64::from(
            downcast_array::<Int8Array>(array, operator_id, "Int8")?.value(row),
        )),
        DataType::Int16 => ScalarValue::Signed(i64::from(
            downcast_array::<Int16Array>(array, operator_id, "Int16")?.value(row),
        )),
        DataType::Int32 => ScalarValue::Signed(i64::from(
            downcast_array::<Int32Array>(array, operator_id, "Int32")?.value(row),
        )),
        DataType::Int64 => ScalarValue::Signed(
            downcast_array::<Int64Array>(array, operator_id, "Int64")?.value(row),
        ),
        DataType::UInt8 => ScalarValue::Unsigned(u64::from(
            downcast_array::<UInt8Array>(array, operator_id, "UInt8")?.value(row),
        )),
        DataType::UInt16 => ScalarValue::Unsigned(u64::from(
            downcast_array::<UInt16Array>(array, operator_id, "UInt16")?.value(row),
        )),
        DataType::UInt32 => ScalarValue::Unsigned(u64::from(
            downcast_array::<UInt32Array>(array, operator_id, "UInt32")?.value(row),
        )),
        DataType::UInt64 => ScalarValue::Unsigned(
            downcast_array::<UInt64Array>(array, operator_id, "UInt64")?.value(row),
        ),
        DataType::Float32 => ScalarValue::Float32(
            downcast_array::<Float32Array>(array, operator_id, "Float32")?
                .value(row)
                .to_bits(),
        ),
        DataType::Float64 => ScalarValue::Float64(
            downcast_array::<Float64Array>(array, operator_id, "Float64")?
                .value(row)
                .to_bits(),
        ),
        DataType::Utf8 => ScalarValue::String(
            downcast_array::<StringArray>(array, operator_id, "Utf8")?
                .value(row)
                .into(),
        ),
        DataType::LargeUtf8 => ScalarValue::String(
            downcast_array::<LargeStringArray>(array, operator_id, "LargeUtf8")?
                .value(row)
                .into(),
        ),
        DataType::Date32 => ScalarValue::Date32(
            downcast_array::<Date32Array>(array, operator_id, "Date32")?.value(row),
        ),
        DataType::Date64 => ScalarValue::Date64(
            downcast_array::<Date64Array>(array, operator_id, "Date64")?.value(row),
        ),
        DataType::Timestamp(TimeUnit::Microsecond, _) => ScalarValue::Timestamp(
            downcast_array::<TimestampMicrosecondArray>(
                array,
                operator_id,
                "Timestamp(Microsecond)",
            )?
            .value(row),
        ),
        _ => {
            return Err(operator_error(
                operator_id,
                &format!("compiled scalar type {data_type} is unsupported"),
            ));
        }
    };
    Ok(Some(value))
}

fn scalar_array(
    data_type: &DataType,
    values: &[Option<ScalarValue>],
    operator_id: &str,
) -> Result<ArrayRef> {
    macro_rules! primitive {
        ($array:ty, $pattern:pat => $value:expr) => {{
            let values = values
                .iter()
                .map(|value| match value {
                    None => Ok(None),
                    Some($pattern) => Ok(Some($value)),
                    Some(_) => Err(internal_error("output scalar type mismatch")),
                })
                .collect::<Result<Vec<_>>>()?;
            Arc::new(<$array>::from(values)) as ArrayRef
        }};
    }

    let array = match data_type {
        DataType::Boolean => primitive!(BooleanArray, ScalarValue::Boolean(value) => *value),
        DataType::Int8 => primitive!(Int8Array, ScalarValue::Signed(value) => i8::try_from(*value)
            .map_err(|_| internal_error("Int8 output scalar overflowed"))?),
        DataType::Int16 => {
            primitive!(Int16Array, ScalarValue::Signed(value) => i16::try_from(*value)
            .map_err(|_| internal_error("Int16 output scalar overflowed"))?)
        }
        DataType::Int32 => {
            primitive!(Int32Array, ScalarValue::Signed(value) => i32::try_from(*value)
            .map_err(|_| internal_error("Int32 output scalar overflowed"))?)
        }
        DataType::Int64 => primitive!(Int64Array, ScalarValue::Signed(value) => *value),
        DataType::UInt8 => {
            primitive!(UInt8Array, ScalarValue::Unsigned(value) => u8::try_from(*value)
            .map_err(|_| internal_error("UInt8 output scalar overflowed"))?)
        }
        DataType::UInt16 => {
            primitive!(UInt16Array, ScalarValue::Unsigned(value) => u16::try_from(*value)
            .map_err(|_| internal_error("UInt16 output scalar overflowed"))?)
        }
        DataType::UInt32 => {
            primitive!(UInt32Array, ScalarValue::Unsigned(value) => u32::try_from(*value)
            .map_err(|_| internal_error("UInt32 output scalar overflowed"))?)
        }
        DataType::UInt64 => primitive!(UInt64Array, ScalarValue::Unsigned(value) => *value),
        DataType::Float32 => {
            primitive!(Float32Array, ScalarValue::Float32(value) => f32::from_bits(*value))
        }
        DataType::Float64 => {
            primitive!(Float64Array, ScalarValue::Float64(value) => f64::from_bits(*value))
        }
        DataType::Utf8 => {
            let values = values
                .iter()
                .map(|value| match value {
                    None => Ok(None),
                    Some(ScalarValue::String(value)) => Ok(Some(value.as_str())),
                    Some(_) => Err(internal_error("Utf8 output scalar type mismatch")),
                })
                .collect::<Result<Vec<_>>>()?;
            Arc::new(StringArray::from(values))
        }
        DataType::LargeUtf8 => {
            let values = values
                .iter()
                .map(|value| match value {
                    None => Ok(None),
                    Some(ScalarValue::String(value)) => Ok(Some(value.as_str())),
                    Some(_) => Err(internal_error("LargeUtf8 output scalar type mismatch")),
                })
                .collect::<Result<Vec<_>>>()?;
            Arc::new(LargeStringArray::from(values))
        }
        DataType::Date32 => primitive!(Date32Array, ScalarValue::Date32(value) => *value),
        DataType::Date64 => primitive!(Date64Array, ScalarValue::Date64(value) => *value),
        DataType::Timestamp(TimeUnit::Microsecond, timezone) => {
            let values = values
                .iter()
                .map(|value| match value {
                    None => Ok(None),
                    Some(ScalarValue::Timestamp(value)) => Ok(Some(*value)),
                    Some(_) => Err(internal_error("timestamp output scalar type mismatch")),
                })
                .collect::<Result<Vec<_>>>()?;
            Arc::new(TimestampMicrosecondArray::from(values).with_timezone_opt(timezone.clone()))
        }
        _ => {
            return Err(operator_error(
                operator_id,
                &format!("cannot build window output array for type {data_type}"),
            ));
        }
    };
    Ok(array)
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

    let group_columns = spec
        .group_by
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
        .collect::<Result<Vec<_>>>()?;

    let aggregates = spec
        .aggregates
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
        .collect::<Result<Vec<_>>>()?;

    let (size_micros, slide_micros) = match spec.geometry {
        WindowGeometry::Tumbling { size_micros } => (size_micros, size_micros),
        WindowGeometry::Hopping {
            size_micros,
            slide_micros,
        } => (size_micros, slide_micros),
    };
    let canonical = canonical_json(&Value::Object(configuration.clone().into_iter().collect()))?;
    Ok(CompiledWindowSpec {
        event_time_index,
        group_columns,
        aggregates,
        geometry: CompiledWindowGeometry {
            size_micros,
            slide_micros,
            overlap: size_micros / slide_micros,
        },
        configuration_hash: hex::encode(Sha256::digest(canonical.as_bytes())),
    })
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

fn format_error(error: &serde_json::Error) -> CalcFlowError {
    CalcFlowError::Format {
        message: error.to_string(),
    }
}
