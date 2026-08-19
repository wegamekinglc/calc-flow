//! Shared Arrow/`PostgreSQL` type mapping (feature `postgresql`).
//!
//! Only the reviewed mapping pairs live here; unknown `PostgreSQL` types
//! fail closed at conversion time instead of degrading silently. The
//! module stores no client or pool types.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, FixedSizeBinaryArray,
    Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, StringArray,
    TimestampMicrosecondArray,
};
use arrow::datatypes::{DataType, Field, SchemaRef, TimeUnit};
use arrow::record_batch::RecordBatch;
use calc_flow::{ArrowFieldSpec, CalcFlowError, Result};
use tokio_postgres::Row;
use tokio_postgres::types::{ToSql, Type as PgType};

/// One discovered relation column.
#[derive(Clone, Debug)]
pub struct PgColumn {
    /// Column name.
    pub name: String,
    /// Resolved `PostgreSQL` type.
    pub data_type: PgType,
    /// Nullability from the catalog.
    pub nullable: bool,
}

/// Builds the Arrow schema for one relation's columns.
///
/// # Errors
///
/// Returns [`CalcFlowError::InvalidArgument`] when a column's
/// `PostgreSQL` type is outside the reviewed matrix.
pub fn arrow_schema(columns: &[PgColumn]) -> Result<SchemaRef> {
    let mut fields = Vec::with_capacity(columns.len());
    for column in columns {
        fields.push(Field::new(
            &column.name,
            arrow_data_type(&column.data_type)?,
            column.nullable,
        ));
    }
    Ok(Arc::new(arrow::datatypes::Schema::new(fields)))
}

/// Maps a `PostgreSQL` type onto the reviewed Arrow type matrix; NUMERIC
/// travels as its exact text form.
///
/// # Errors
///
/// Returns [`CalcFlowError::InvalidArgument`] naming the type for any
/// entry outside the matrix.
pub fn arrow_data_type(data_type: &PgType) -> Result<DataType> {
    let mapped = match data_type.clone() {
        PgType::BOOL => DataType::Boolean,
        PgType::INT2 => DataType::Int16,
        PgType::INT4 => DataType::Int32,
        PgType::INT8 => DataType::Int64,
        PgType::FLOAT4 => DataType::Float32,
        PgType::FLOAT8 => DataType::Float64,
        PgType::TEXT
        | PgType::VARCHAR
        | PgType::BPCHAR
        | PgType::NAME
        | PgType::NUMERIC
        | PgType::JSON
        | PgType::JSONB => DataType::Utf8,
        PgType::BYTEA => DataType::Binary,
        PgType::TIMESTAMP => DataType::Timestamp(TimeUnit::Microsecond, None),
        PgType::TIMESTAMPTZ => DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into())),
        PgType::DATE => DataType::Date32,
        PgType::UUID => DataType::FixedSizeBinary(16),
        other => {
            return Err(CalcFlowError::InvalidArgument {
                field: "column type".into(),
                message: format!(
                    "`PostgreSQL` type {} is outside the reviewed matrix",
                    other.name()
                ),
            });
        }
    };
    Ok(mapped)
}

/// Converts fetched rows into one Arrow record batch.
///
/// # Errors
///
/// Returns the conversion error for any cell outside the matrix.
pub fn record_batch(columns: &[PgColumn], rows: &[Row]) -> Result<RecordBatch> {
    let schema = arrow_schema(columns)?;
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len());
    for (index, column) in columns.iter().enumerate() {
        arrays.push(column_array(column, rows, index)?);
    }
    RecordBatch::try_new(schema, arrays).map_err(|error| CalcFlowError::InvalidArgument {
        field: "record batch".into(),
        message: error.to_string(),
    })
}

fn column_array(column: &PgColumn, rows: &[Row], index: usize) -> Result<ArrayRef> {
    let name = &column.name;
    macro_rules! typed {
        ($builder:ty, $cast:ty) => {{
            let mut values = Vec::with_capacity(rows.len());
            for row in rows {
                let value: Option<$cast> = row
                    .try_get::<_, Option<$cast>>(index)
                    .map_err(cell_error(name))?;
                values.push(value);
            }
            Ok(Arc::new(<$builder>::from(values)) as ArrayRef)
        }};
    }
    match column.data_type.clone() {
        PgType::BOOL => typed!(BooleanArray, bool),
        PgType::INT2 => typed!(Int16Array, i16),
        PgType::INT4 => typed!(Int32Array, i32),
        PgType::INT8 => typed!(Int64Array, i64),
        PgType::FLOAT4 => typed!(Float32Array, f32),
        PgType::FLOAT8 => typed!(Float64Array, f64),
        PgType::TEXT | PgType::VARCHAR | PgType::BPCHAR | PgType::NAME => {
            typed!(StringArray, String)
        }
        PgType::NUMERIC => numeric_array(rows, index, name),
        PgType::JSON | PgType::JSONB => json_array(rows, index, name),
        PgType::BYTEA => bytea_array(rows, index, name),
        PgType::DATE => date_array(rows, index, name),
        PgType::TIMESTAMP => timestamp_array(rows, index, name),
        PgType::TIMESTAMPTZ => timestamptz_array(rows, index, name),
        PgType::UUID => uuid_array(rows, index, name),
        other => Err(CalcFlowError::InvalidArgument {
            field: format!("column {name}"),
            message: format!(
                "`PostgreSQL` type {} is outside the reviewed matrix",
                other.name()
            ),
        }),
    }
}

fn numeric_array(rows: &[Row], index: usize, name: &str) -> Result<ArrayRef> {
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        let value: Option<rust_decimal::Decimal> = row
            .try_get::<_, Option<rust_decimal::Decimal>>(index)
            .map_err(cell_error(name))?;
        values.push(value.map(|value| value.to_string()));
    }
    Ok(Arc::new(StringArray::from(values)) as ArrayRef)
}

fn json_array(rows: &[Row], index: usize, name: &str) -> Result<ArrayRef> {
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        let value: Option<serde_json::Value> = row
            .try_get::<_, Option<serde_json::Value>>(index)
            .map_err(cell_error(name))?;
        values.push(value.map(|value| value.to_string()));
    }
    Ok(Arc::new(StringArray::from(values)) as ArrayRef)
}

fn bytea_array(rows: &[Row], index: usize, name: &str) -> Result<ArrayRef> {
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        let value: Option<Vec<u8>> = row
            .try_get::<_, Option<Vec<u8>>>(index)
            .map_err(cell_error(name))?;
        values.push(value);
    }
    let converted: Vec<Option<&[u8]>> = values.iter().map(|v| v.as_deref()).collect();
    Ok(Arc::new(BinaryArray::from_opt_vec(converted)) as ArrayRef)
}

fn date_array(rows: &[Row], index: usize, name: &str) -> Result<ArrayRef> {
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch is valid");
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        let value: Option<chrono::NaiveDate> = row
            .try_get::<_, Option<chrono::NaiveDate>>(index)
            .map_err(cell_error(name))?;
        values.push(value.map(|date| i32::try_from((date - epoch).num_days()).unwrap_or(i32::MAX)));
    }
    Ok(Arc::new(Date32Array::from(values)) as ArrayRef)
}

fn timestamp_array(rows: &[Row], index: usize, name: &str) -> Result<ArrayRef> {
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        let value: Option<chrono::NaiveDateTime> = row
            .try_get::<_, Option<chrono::NaiveDateTime>>(index)
            .map_err(cell_error(name))?;
        values.push(value.map(|stamp| stamp.and_utc().timestamp_micros()));
    }
    Ok(Arc::new(TimestampMicrosecondArray::from(values)) as ArrayRef)
}

fn timestamptz_array(rows: &[Row], index: usize, name: &str) -> Result<ArrayRef> {
    let mut values = Vec::with_capacity(rows.len());
    for row in rows {
        let value: Option<chrono::DateTime<chrono::Utc>> = row
            .try_get::<_, Option<chrono::DateTime<chrono::Utc>>>(index)
            .map_err(cell_error(name))?;
        values.push(value.map(|stamp| stamp.timestamp_micros()));
    }
    Ok(Arc::new(TimestampMicrosecondArray::from(values)) as ArrayRef)
}

fn uuid_array(rows: &[Row], index: usize, name: &str) -> Result<ArrayRef> {
    let mut values: Vec<Option<Vec<u8>>> = Vec::with_capacity(rows.len());
    for row in rows {
        let value: Option<uuid::Uuid> = row
            .try_get::<_, Option<uuid::Uuid>>(index)
            .map_err(cell_error(name))?;
        values.push(value.map(|id| id.as_bytes().to_vec()));
    }
    let mut builder = arrow::array::FixedSizeBinaryBuilder::with_capacity(values.len(), 16);
    for value in values {
        match value {
            Some(bytes) => {
                let _ = builder.append_value(bytes.as_slice());
            }
            None => builder.append_null(),
        }
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

fn cell_error(column: &str) -> impl Fn(tokio_postgres::Error) -> CalcFlowError + '_ {
    move |error| CalcFlowError::InvalidArgument {
        field: format!("column {column}"),
        message: error.to_string(),
    }
}

/// One cell extracted from an Arrow array, ready for parameterized
/// insertion.
#[derive(Clone, Debug, PartialEq)]
pub enum PgValue {
    /// SQL NULL.
    Null,
    /// A boolean cell.
    Boolean(bool),
    /// A 16-bit integer cell.
    Int16(i16),
    /// A 32-bit integer cell.
    Int32(i32),
    /// A 64-bit integer cell.
    Int64(i64),
    /// A 32-bit float cell.
    Float32(f32),
    /// A 64-bit float cell.
    Float64(f64),
    /// A text cell (also carries NUMERIC exact text).
    Text(String),
    /// A binary cell.
    Bytes(Vec<u8>),
    /// An exact decimal within the reviewed 28-digit boundary.
    Numeric(rust_decimal::Decimal),
    /// Days since the Unix epoch.
    Date32(i32),
    /// UTC-naive microseconds since the Unix epoch.
    Timestamp(i64),
    /// UTC microseconds since the Unix epoch.
    TimestampTz(i64),
    /// A UUID cell.
    Uuid(uuid::Uuid),
}

impl PgValue {
    pub(crate) fn to_evidence(&self) -> serde_json::Value {
        let (kind, value) = match self {
            Self::Null => ("null", serde_json::Value::Null),
            Self::Boolean(value) => ("bool", serde_json::Value::Bool(*value)),
            Self::Int16(value) => ("i16", serde_json::Value::from(*value)),
            Self::Int32(value) => ("i32", serde_json::Value::from(*value)),
            Self::Int64(value) => ("i64", serde_json::Value::from(*value)),
            Self::Float32(value) => ("f32_bits", serde_json::Value::from(value.to_bits())),
            Self::Float64(value) => ("f64_bits", serde_json::Value::from(value.to_bits())),
            Self::Text(value) => ("text", serde_json::Value::String(value.clone())),
            Self::Bytes(value) => (
                "bytes",
                serde_json::Value::Array(
                    value.iter().copied().map(serde_json::Value::from).collect(),
                ),
            ),
            Self::Numeric(value) => ("numeric", serde_json::Value::String(value.to_string())),
            Self::Date32(value) => ("date32", serde_json::Value::from(*value)),
            Self::Timestamp(value) => ("timestamp_us", serde_json::Value::from(*value)),
            Self::TimestampTz(value) => ("timestamptz_us", serde_json::Value::from(*value)),
            Self::Uuid(value) => ("uuid", serde_json::Value::String(value.to_string())),
        };
        serde_json::json!({"type": kind, "value": value})
    }

    // This is an exhaustive tagged-value decoder; keeping the tag matrix in one
    // place makes the durable evidence contract auditable.
    // #lizard forgives
    pub(crate) fn from_evidence(value: &serde_json::Value) -> Result<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| invalid_evidence("cell evidence must be an object"))?;
        if object.len() != 2 || !object.contains_key("type") || !object.contains_key("value") {
            return Err(invalid_evidence(
                "cell evidence must contain exactly type and value",
            ));
        }
        let kind = object["type"]
            .as_str()
            .ok_or_else(|| invalid_evidence("cell evidence type must be a string"))?;
        let cell = &object["value"];
        match kind {
            "null" if cell.is_null() => Ok(Self::Null),
            "bool" => cell
                .as_bool()
                .map(Self::Boolean)
                .ok_or_else(|| invalid_evidence("bool evidence must be a boolean")),
            "i16" => evidence_i64(cell, "i16")
                .and_then(|value| {
                    i16::try_from(value).map_err(|_| invalid_evidence("i16 overflow"))
                })
                .map(Self::Int16),
            "i32" => evidence_i64(cell, "i32")
                .and_then(|value| {
                    i32::try_from(value).map_err(|_| invalid_evidence("i32 overflow"))
                })
                .map(Self::Int32),
            "i64" => evidence_i64(cell, "i64").map(Self::Int64),
            "f32_bits" => evidence_u64(cell, "f32_bits")
                .and_then(|bits| {
                    u32::try_from(bits).map_err(|_| invalid_evidence("f32 bits overflow"))
                })
                .map(f32::from_bits)
                .map(Self::Float32),
            "f64_bits" => evidence_u64(cell, "f64_bits")
                .map(f64::from_bits)
                .map(Self::Float64),
            "text" => cell
                .as_str()
                .map(|value| Self::Text(value.to_string()))
                .ok_or_else(|| invalid_evidence("text evidence must be a string")),
            "bytes" => {
                let values = cell
                    .as_array()
                    .ok_or_else(|| invalid_evidence("bytes evidence must be an array"))?;
                values
                    .iter()
                    .map(|value| {
                        evidence_u64(value, "byte").and_then(|byte| {
                            u8::try_from(byte).map_err(|_| invalid_evidence("byte overflow"))
                        })
                    })
                    .collect::<Result<Vec<_>>>()
                    .map(Self::Bytes)
            }
            "numeric" => cell
                .as_str()
                .ok_or_else(|| invalid_evidence("numeric evidence must be a string"))
                .and_then(|value| {
                    value
                        .parse::<rust_decimal::Decimal>()
                        .map_err(|_| invalid_evidence("numeric evidence is invalid"))
                })
                .map(Self::Numeric),
            "date32" => evidence_i64(cell, "date32")
                .and_then(|value| {
                    i32::try_from(value).map_err(|_| invalid_evidence("date32 overflow"))
                })
                .map(Self::Date32),
            "timestamp_us" => evidence_i64(cell, "timestamp_us").map(Self::Timestamp),
            "timestamptz_us" => evidence_i64(cell, "timestamptz_us").map(Self::TimestampTz),
            "uuid" => cell
                .as_str()
                .ok_or_else(|| invalid_evidence("uuid evidence must be a string"))
                .and_then(|value| {
                    uuid::Uuid::parse_str(value)
                        .map_err(|_| invalid_evidence("uuid evidence is invalid"))
                })
                .map(Self::Uuid),
            _ => Err(invalid_evidence("unknown or mismatched cell evidence type")),
        }
    }
}

fn evidence_i64(value: &serde_json::Value, field: &str) -> Result<i64> {
    value
        .as_i64()
        .ok_or_else(|| invalid_evidence(&format!("{field} evidence must be an integer")))
}

fn evidence_u64(value: &serde_json::Value, field: &str) -> Result<u64> {
    value
        .as_u64()
        .ok_or_else(|| invalid_evidence(&format!("{field} evidence must be an unsigned integer")))
}

fn invalid_evidence(message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "postgresql pre-commit evidence".into(),
        message: message.into(),
    }
}

type ToSqlResult =
    std::result::Result<tokio_postgres::types::IsNull, Box<dyn std::error::Error + Send + Sync>>;

impl ToSql for PgValue {
    fn to_sql(
        &self,
        ty: &PgType,
        out: &mut tokio_postgres::types::private::BytesMut,
    ) -> ToSqlResult {
        use tokio_postgres::types::IsNull;
        match self {
            PgValue::Null => Ok(IsNull::Yes),
            PgValue::Boolean(v) => v.to_sql(&PgType::BOOL, out),
            PgValue::Int16(v) => v.to_sql(&PgType::INT2, out),
            PgValue::Int32(v) => v.to_sql(&PgType::INT4, out),
            PgValue::Int64(v) => v.to_sql(&PgType::INT8, out),
            PgValue::Float32(v) => v.to_sql(&PgType::FLOAT4, out),
            PgValue::Float64(v) => v.to_sql(&PgType::FLOAT8, out),
            PgValue::Text(v) if matches!(ty.clone(), PgType::JSON | PgType::JSONB) => {
                let value: serde_json::Value = serde_json::from_str(v)?;
                value.to_sql(ty, out)
            }
            PgValue::Text(v) if ty == &PgType::NUMERIC => {
                let value: rust_decimal::Decimal = v.parse()?;
                value.to_sql(ty, out)
            }
            PgValue::Text(v) => v.to_sql(&PgType::TEXT, out),
            PgValue::Bytes(v) => v.to_sql(&PgType::BYTEA, out),
            PgValue::Numeric(v) => v.to_sql(&PgType::NUMERIC, out),
            PgValue::Date32(v) => {
                let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch is valid");
                epoch
                    .checked_add_signed(chrono::Duration::days(i64::from(*v)))
                    .ok_or("date32 value is outside chrono's range")?
                    .to_sql(&PgType::DATE, out)
            }
            PgValue::Timestamp(v) => chrono::DateTime::from_timestamp_micros(*v)
                .ok_or("timestamp value is outside chrono's range")?
                .naive_utc()
                .to_sql(&PgType::TIMESTAMP, out),
            PgValue::TimestampTz(v) => chrono::DateTime::from_timestamp_micros(*v)
                .ok_or("timestamptz value is outside chrono's range")?
                .to_sql(&PgType::TIMESTAMPTZ, out),
            PgValue::Uuid(v) => v.to_sql(&PgType::UUID, out),
        }
    }

    fn accepts(_ty: &PgType) -> bool {
        true
    }

    tokio_postgres::types::to_sql_checked!();
}

/// Extracts one cell from a record batch column at the given row.
///
/// # Errors
///
/// Returns [`CalcFlowError::InvalidArgument`] when the column's Arrow
/// type has no reviewed `PostgreSQL` mapping.
#[allow(
    clippy::too_many_lines,
    reason = "one exhaustive Arrow-to-PostgreSQL type matrix stays reviewable as a single dispatch"
)]
pub fn cell_value(column: &dyn arrow::array::Array, row: usize) -> Result<PgValue> {
    use arrow::array::{
        BinaryArray, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
        StringArray,
    };
    if column.is_null(row) {
        return Ok(PgValue::Null);
    }
    let value = match column.data_type() {
        DataType::Boolean => PgValue::Boolean(
            column
                .as_any()
                .downcast_ref::<BooleanArray>()
                .is_some_and(|a| a.value(row)),
        ),
        DataType::Int16 => PgValue::Int16(
            column
                .as_any()
                .downcast_ref::<Int16Array>()
                .map(|a| a.value(row))
                .unwrap_or_default(),
        ),
        DataType::Int32 => PgValue::Int32(
            column
                .as_any()
                .downcast_ref::<Int32Array>()
                .map(|a| a.value(row))
                .unwrap_or_default(),
        ),
        DataType::Int64 => PgValue::Int64(
            column
                .as_any()
                .downcast_ref::<Int64Array>()
                .map(|a| a.value(row))
                .unwrap_or_default(),
        ),
        DataType::Float32 => PgValue::Float32(
            column
                .as_any()
                .downcast_ref::<Float32Array>()
                .map(|a| a.value(row))
                .unwrap_or_default(),
        ),
        DataType::Float64 => PgValue::Float64(
            column
                .as_any()
                .downcast_ref::<Float64Array>()
                .map(|a| a.value(row))
                .unwrap_or_default(),
        ),
        DataType::Utf8 => PgValue::Text(
            column
                .as_any()
                .downcast_ref::<StringArray>()
                .map(|a| a.value(row).to_string())
                .unwrap_or_default(),
        ),
        DataType::Binary => PgValue::Bytes(
            column
                .as_any()
                .downcast_ref::<BinaryArray>()
                .map(|a| a.value(row).to_vec())
                .unwrap_or_default(),
        ),
        DataType::Decimal128(_, scale) if *scale >= 0 => {
            let value = column
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .map(|array| array.value(row))
                .unwrap_or_default();
            let scale = u32::try_from(*scale).map_err(|_| CalcFlowError::InvalidArgument {
                field: "column".into(),
                message: "Decimal128 uses a negative scale unsupported by PostgreSQL".into(),
            })?;
            PgValue::Numeric(
                rust_decimal::Decimal::try_from_i128_with_scale(value, scale).map_err(|_| {
                    CalcFlowError::InvalidArgument {
                        field: "column".into(),
                        message: "Decimal128 value exceeds the PostgreSQL sink's reviewed 28-digit boundary".into(),
                    }
                })?,
            )
        }
        DataType::Date32 => PgValue::Date32(
            column
                .as_any()
                .downcast_ref::<Date32Array>()
                .map(|array| array.value(row))
                .unwrap_or_default(),
        ),
        DataType::Timestamp(TimeUnit::Microsecond, timezone) => {
            let value = column
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .map(|array| array.value(row))
                .unwrap_or_default();
            if timezone.is_some() {
                PgValue::TimestampTz(value)
            } else {
                PgValue::Timestamp(value)
            }
        }
        DataType::FixedSizeBinary(16) => {
            let value = column
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .map(|array| array.value(row))
                .unwrap_or_default();
            PgValue::Uuid(uuid::Uuid::from_slice(value).map_err(|_| {
                CalcFlowError::InvalidArgument {
                    field: "column".into(),
                    message: "FixedSizeBinary(16) cell is not a UUID".into(),
                }
            })?)
        }
        other => {
            return Err(CalcFlowError::InvalidArgument {
                field: "column".into(),
                message: format!("Arrow type {other:?} has no reviewed `PostgreSQL` mapping"),
            });
        }
    };
    Ok(value)
}

/// Validates an SQL identifier against the lowercase vocabulary.
///
/// # Errors
///
/// Returns [`CalcFlowError::InvalidArgument`] when the name is not a
/// plain lowercase identifier.
pub fn pg_identifier(name: &str) -> Result<String> {
    let valid = !name.is_empty()
        && name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_lowercase() || c == '_')
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        && name.len() <= 63;
    if !valid {
        return Err(CalcFlowError::InvalidArgument {
            field: "identifier".into(),
            message: format!("{name:?} is not a lowercase `PostgreSQL` identifier"),
        });
    }
    Ok(name.to_string())
}

/// Validates explicit field specifications as identifiers.
///
/// # Errors
///
/// Returns the identifier error for the first invalid field name.
pub fn pg_identifiers(fields: &[ArrowFieldSpec]) -> Result<Vec<String>> {
    fields
        .iter()
        .map(|field| pg_identifier(&field.name))
        .collect()
}
