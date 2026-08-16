//! Shared Arrow/PostgreSQL type mapping (feature `postgresql`).
//!
//! Only the reviewed mapping pairs live here; unknown PostgreSQL types
//! fail closed at conversion time instead of degrading silently. The
//! module stores no client or pool types.

use std::sync::Arc;

use arrow::array::{
    ArrayRef, BinaryArray, BooleanArray, Date32Array, FixedSizeBinaryArray, Float32Array,
    Float64Array, Int16Array, Int32Array, Int64Array, StringArray, TimestampMicrosecondArray,
};
use arrow::datatypes::{DataType, Field, SchemaRef, TimeUnit};
use arrow::record_batch::RecordBatch;
use calc_flow::{ArrowFieldSpec, CalcFlowError, Result};
use tokio_postgres::Row;
use tokio_postgres::types::Type as PgType;

/// One discovered relation column.
#[derive(Clone, Debug)]
pub struct PgColumn {
    /// Column name.
    pub name: String,
    /// Resolved PostgreSQL type.
    pub data_type: PgType,
    /// Nullability from the catalog.
    pub nullable: bool,
}

/// Builds the Arrow schema for one relation's columns.
///
/// # Errors
///
/// Returns [`CalcFlowError::InvalidArgument`] when a column's
/// PostgreSQL type is outside the reviewed matrix.
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

/// Maps a PostgreSQL type onto the reviewed Arrow type matrix; NUMERIC
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
        PgType::TEXT | PgType::VARCHAR | PgType::BPCHAR | PgType::NAME => DataType::Utf8,
        PgType::BYTEA => DataType::Binary,
        PgType::NUMERIC => DataType::Utf8,
        PgType::TIMESTAMP => DataType::Timestamp(TimeUnit::Microsecond, None),
        PgType::TIMESTAMPTZ => DataType::Timestamp(TimeUnit::Microsecond, Some("+00:00".into())),
        PgType::DATE => DataType::Date32,
        PgType::UUID => DataType::FixedSizeBinary(16),
        other => {
            return Err(CalcFlowError::InvalidArgument {
                field: "column type".into(),
                message: format!(
                    "PostgreSQL type {} is outside the reviewed matrix",
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
    let get = |row: &Row| -> Option<tokio_postgres::types::Type> {
        row.columns().get(index).map(|c| c.type_().clone())
    };
    let _ = get;
    macro_rules! typed {
        ($variant:ident, $builder:ty, $cast:ty) => {{
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
        PgType::BOOL => typed!(BOOL, BooleanArray, bool),
        PgType::INT2 => typed!(INT2, Int16Array, i16),
        PgType::INT4 => typed!(INT4, Int32Array, i32),
        PgType::INT8 => typed!(INT8, Int64Array, i64),
        PgType::FLOAT4 => typed!(FLOAT4, Float32Array, f32),
        PgType::FLOAT8 => typed!(FLOAT8, Float64Array, f64),
        PgType::TEXT | PgType::VARCHAR | PgType::BPCHAR | PgType::NAME | PgType::NUMERIC => {
            typed!(TEXT, StringArray, String)
        }
        PgType::BYTEA => {
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
        PgType::DATE => {
            let mut values = Vec::with_capacity(rows.len());
            for row in rows {
                let value: Option<chrono::NaiveDate> = row
                    .try_get::<_, Option<chrono::NaiveDate>>(index)
                    .map_err(cell_error(name))?;
                values.push(value.map(|date| {
                    (date - chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch")).num_days()
                        as i32
                }));
            }
            Ok(Arc::new(Date32Array::from(values)) as ArrayRef)
        }
        PgType::TIMESTAMP | PgType::TIMESTAMPTZ => {
            let mut values = Vec::with_capacity(rows.len());
            for row in rows {
                let value: Option<chrono::NaiveDateTime> = row
                    .try_get::<_, Option<chrono::NaiveDateTime>>(index)
                    .map_err(cell_error(name))?;
                values.push(value.map(|stamp| stamp.and_utc().timestamp_micros()));
            }
            Ok(Arc::new(TimestampMicrosecondArray::from(values)) as ArrayRef)
        }
        PgType::UUID => {
            let mut values = Vec::with_capacity(rows.len());
            for row in rows {
                let value: Option<uuid::Uuid> = row
                    .try_get::<_, Option<uuid::Uuid>>(index)
                    .map_err(cell_error(name))?;
                values.push(value.map(|id| id.as_bytes().to_vec()));
            }
            let converted: Vec<Option<[u8; 16]>> = values
                .into_iter()
                .map(|value| {
                    value.map(|bytes| {
                        let mut fixed = [0u8; 16];
                        fixed.copy_from_slice(&bytes);
                        fixed
                    })
                })
                .collect();
            let mut builder =
                arrow::array::FixedSizeBinaryBuilder::with_capacity(converted.len(), 16);
            for value in converted {
                match value {
                    Some(bytes) => {
                        builder.append_value(&bytes);
                    }
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        }
        other => Err(CalcFlowError::InvalidArgument {
            field: format!("column {name}"),
            message: format!(
                "PostgreSQL type {} is outside the reviewed matrix",
                other.name()
            ),
        }),
    }
}

fn cell_error(column: &str) -> impl Fn(tokio_postgres::Error) -> CalcFlowError + '_ {
    move |error| CalcFlowError::InvalidArgument {
        field: format!("column {column}"),
        message: error.to_string(),
    }
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
            message: format!("{name:?} is not a lowercase PostgreSQL identifier"),
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
