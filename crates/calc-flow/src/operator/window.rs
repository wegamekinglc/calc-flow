use std::{collections::BTreeSet, fmt, sync::Arc, time::Duration};

use async_trait::async_trait;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    Batch, BatchKind, CalcFlowError, EventTime, JsonMap, Port, Result, StreamCollector,
    StreamOperator, StreamOperatorContext, canonical_json,
};

use super::{OperatorMetadata, validate_operator_name};

/// Maximum number of concrete hopping-window assignments for one input row.
pub const MAX_WINDOW_OVERLAP: u64 = 1_024;

const WINDOW_STATE_LAYOUT_VERSION: u32 = 1;
const MAX_GROUP_KEY_BYTES: u64 = 65_536;

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
    #[allow(
        dead_code,
        reason = "the M4 execution work package consumes the compiled window declaration"
    )]
    compiled: CompiledWindowSpec,
}

#[allow(
    dead_code,
    reason = "the M4 execution work package consumes the compiled window declaration"
)]
struct CompiledWindowSpec {
    event_time_index: usize,
    group_columns: Vec<CompiledGroupColumn>,
    aggregates: Vec<CompiledAggregate>,
    geometry: CompiledWindowGeometry,
    configuration_hash: String,
}

#[allow(
    dead_code,
    reason = "the M4 execution work package consumes the compiled window declaration"
)]
struct CompiledGroupColumn {
    index: usize,
    data_type: DataType,
}

#[allow(
    dead_code,
    reason = "the M4 execution work package consumes the compiled window declaration"
)]
struct CompiledAggregate {
    input_index: usize,
    input_type: DataType,
    output_type: DataType,
}

#[derive(Clone, Copy)]
#[allow(
    dead_code,
    reason = "the M4 execution work package consumes the compiled window declaration"
)]
struct CompiledWindowGeometry {
    size_micros: u64,
    slide_micros: u64,
    overlap: u64,
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
        })
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
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if ingress != "input" {
            return Err(CalcFlowError::Operator {
                node_id: self.name.clone(),
                message: format!("unknown ingress {ingress:?}; expected \"input\""),
            });
        }
        self.input_ports[0].validate(&batch, &format!("{}.input", self.name))?;
        Err(CalcFlowError::Internal {
            message: "window execution is installed by the M4 execution work package".into(),
        })
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
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
