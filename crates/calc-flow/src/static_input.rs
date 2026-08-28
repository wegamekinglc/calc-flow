//! Immutable static stream inputs (SCE-11).
//!
//! A static input is a declared, engine-owned immutable [`Batch`] side value
//! latched exactly once per streaming job before any source opens. This
//! module owns the declaration types, the canonical digest-v1 byte encoding
//! (API note §7), and the digest entry points. Payload digests never enter
//! lineage identity; only declarations participate in the plan fingerprint.

use datafusion::arrow::{
    array::{
        Array, BooleanArray, Date32Array, Date64Array, DictionaryArray, Float32Array, Float64Array,
        Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray, StringArray,
        Time32SecondArray, Time64MicrosecondArray, TimestampMicrosecondArray,
        TimestampMillisecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, TimeUnit},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use std::collections::BTreeMap;

use crate::{ArrowFieldSpec, Batch, CalcFlowError, Result, TableBatch, batch::LatchedArrayPayload};

/// The digest version string for the canonical static-input encoding.
pub const STATIC_INPUT_DIGEST_VERSION: &str = "calc_flow.static_input.digest.v1";

/// The declared mutability of one static input; only `Static` exists in v1.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StaticMutability {
    /// The payload is latched once and never mutates for the job's lifetime.
    Static,
}

/// The declared external-input descriptor for one static input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StaticInputSpec {
    /// An Arrow table value with an exact declared schema.
    Table {
        /// Declared external input name.
        name: String,
        /// Declared mutability; always `Static` in v1.
        mutability: StaticMutability,
        /// Exact declared Arrow schema in field order.
        schema: Vec<ArrowFieldSpec>,
    },
    /// An Array API value with an exact declared backend, dtype, and shape.
    Array {
        /// Declared external input name.
        name: String,
        /// Declared mutability; always `Static` in v1.
        mutability: StaticMutability,
        /// Declared array backend identifier (for example `numpy`).
        backend: String,
        /// Declared dtype spelling (for example `float64`).
        dtype: String,
        /// Declared logical shape in element dimensions.
        shape: Vec<u64>,
    },
}

impl StaticInputSpec {
    /// Returns the declared input name.
    pub fn name(&self) -> &str {
        match self {
            Self::Table { name, .. } | Self::Array { name, .. } => name,
        }
    }
}

/// The bounded digest evidence recorded for one latched static input.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StaticInputDigest {
    /// The digest version string (`calc_flow.static_input.digest.v1`).
    pub digest_version: String,
    /// Lowercase hexadecimal SHA-256 over the canonical tagged bytes.
    pub sha256: String,
}

/// Growing byte buffer with the §7 checked-length primitives.
struct DigestWriter {
    bytes: Vec<u8>,
}

impl DigestWriter {
    fn new() -> Self {
        let mut bytes = STATIC_INPUT_DIGEST_VERSION.as_bytes().to_vec();
        bytes.push(0x00);
        Self { bytes }
    }

    fn tag(&mut self, tag: u8) {
        self.bytes.push(tag);
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn checked_len(&mut self, value: usize) -> Result<()> {
        self.u64(
            u64::try_from(value).map_err(|_| CalcFlowError::InvalidArgument {
                field: "static_input.length".into(),
                message: "length exceeds the u64 digest range".into(),
            })?,
        );
        Ok(())
    }

    fn text(&mut self, value: &[u8]) -> Result<()> {
        self.checked_len(value.len())?;
        self.bytes.extend_from_slice(value);
        Ok(())
    }

    fn scalar(&mut self, bytes: &[u8]) {
        self.tag(0x31);
        self.bytes.extend_from_slice(bytes);
    }

    fn finish(self) -> StaticInputDigest {
        StaticInputDigest {
            digest_version: STATIC_INPUT_DIGEST_VERSION.into(),
            sha256: hex::encode(Sha256::digest(&self.bytes)),
        }
    }
}

/// Returns the digest-v1 type tag for one Arrow type, or `None` when the
/// type is outside the digest-v1 table set.
fn table_type_tag(data_type: &DataType) -> Option<u8> {
    match data_type {
        DataType::Boolean => Some(0x40),
        DataType::Int8 => Some(0x41),
        DataType::Int16 => Some(0x42),
        DataType::Int32 => Some(0x43),
        DataType::Int64 => Some(0x44),
        DataType::UInt8 => Some(0x45),
        DataType::UInt16 => Some(0x46),
        DataType::UInt32 => Some(0x47),
        DataType::UInt64 => Some(0x48),
        DataType::Float32 => Some(0x49),
        DataType::Float64 => Some(0x4a),
        DataType::Utf8 => Some(0x4b),
        DataType::LargeUtf8 => Some(0x4c),
        DataType::Date32 => Some(0x4d),
        DataType::Date64 => Some(0x4e),
        DataType::Time32(TimeUnit::Second) => Some(0x4f),
        DataType::Time64(TimeUnit::Microsecond) => Some(0x50),
        DataType::Timestamp(TimeUnit::Millisecond, _) => Some(0x51),
        DataType::Timestamp(TimeUnit::Microsecond, None) => Some(0x52),
        DataType::Timestamp(TimeUnit::Microsecond, Some(zone)) if zone.as_ref() == "UTC" => {
            Some(0x53)
        }
        DataType::Dictionary(_, _) => Some(0x54),
        _ => None,
    }
}

/// Returns the digest-v1 type tag for one array dtype spelling.
fn array_dtype_tag(dtype: &str) -> Option<u8> {
    Some(match dtype {
        "bool" => 0x40,
        "int8" => 0x41,
        "int16" => 0x42,
        "int32" => 0x43,
        "int64" => 0x44,
        "uint8" => 0x45,
        "uint16" => 0x46,
        "uint32" => 0x47,
        "uint64" => 0x48,
        "float32" => 0x49,
        "float64" => 0x4a,
        _ => return None,
    })
}

/// Returns whether `dtype` names a digest-v1 array dtype spelling.
pub(crate) fn is_supported_array_dtype(dtype: &str) -> bool {
    array_dtype_tag(dtype).is_some()
}

fn unsupported_table_type(data_type: &DataType) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "static_inputs.schema".into(),
        message: format!("Arrow type {data_type} is outside the digest-v1 table set"),
    }
}

/// Writes the type descriptor for one Arrow type.
fn write_table_type(writer: &mut DigestWriter, data_type: &DataType) -> Result<()> {
    write_table_type_ordered(writer, data_type, None)
}

/// Writes the type descriptor; `dictionary_ordered` supplies the Arrow
/// ordered flag for the outermost dictionary type (Arrow 58 stores the flag
/// on the array, not the type).
fn write_table_type_ordered(
    writer: &mut DigestWriter,
    data_type: &DataType,
    dictionary_ordered: Option<bool>,
) -> Result<()> {
    let tag = table_type_tag(data_type).ok_or_else(|| unsupported_table_type(data_type))?;
    writer.tag(tag);
    if let DataType::Dictionary(index, value) = data_type {
        if !matches!(
            **index,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
        ) {
            return Err(CalcFlowError::InvalidArgument {
                field: "static_inputs.schema".into(),
                message: "dictionary index types are restricted to integer types in digest v1"
                    .into(),
            });
        }
        if table_type_tag(value).is_none_or(|tag| tag == 0x54) {
            return Err(CalcFlowError::InvalidArgument {
                field: "static_inputs.schema".into(),
                message: "dictionary values must use a non-dictionary digest-v1 type".into(),
            });
        }
        write_table_type(writer, index)?;
        write_table_type(writer, value)?;
        writer.tag(u8::from(dictionary_ordered.unwrap_or(false)));
    }
    Ok(())
}

fn f32_bits(value: f32) -> [u8; 4] {
    (if value.is_nan() {
        0x7fc0_0000u32
    } else {
        value.to_bits()
    })
    .to_be_bytes()
}

fn f64_bits(value: f64) -> [u8; 8] {
    (if value.is_nan() {
        0x7ff8_0000_0000_0000u64
    } else {
        value.to_bits()
    })
    .to_be_bytes()
}

/// Writes one logical cell of a non-dictionary array at `row`.
fn write_flat_cell(writer: &mut DigestWriter, array: &dyn Array, row: usize) -> Result<()> {
    if array.is_null(row) {
        writer.tag(0x30);
        return Ok(());
    }
    let data_type = array.data_type();
    let tag = table_type_tag(data_type).ok_or_else(|| unsupported_table_type(data_type))?;
    debug_assert_ne!(tag, 0x54, "dictionary cells resolve before this seam");
    macro_rules! scalar {
        ($array:ident, $rule:expr) => {{
            let typed = array
                .as_any()
                .downcast_ref::<$array>()
                .expect("the tag was derived from this array's own type");
            writer.scalar(&$rule(typed.value(row)));
        }};
    }
    macro_rules! text_scalar {
        ($array:ident) => {{
            let typed = array
                .as_any()
                .downcast_ref::<$array>()
                .expect("the tag was derived from this array's own type");
            writer.tag(0x31);
            writer.text(typed.value(row).as_bytes())?;
        }};
    }
    match data_type {
        DataType::Boolean => scalar!(BooleanArray, |value: bool| [u8::from(value)]),
        DataType::Int8 => scalar!(Int8Array, |value: i8| value.to_be_bytes()),
        DataType::Int16 => scalar!(Int16Array, |value: i16| value.to_be_bytes()),
        DataType::Int32 => scalar!(Int32Array, |value: i32| value.to_be_bytes()),
        DataType::Int64 => scalar!(Int64Array, |value: i64| value.to_be_bytes()),
        DataType::UInt8 => scalar!(UInt8Array, |value: u8| value.to_be_bytes()),
        DataType::UInt16 => scalar!(UInt16Array, |value: u16| value.to_be_bytes()),
        DataType::UInt32 => scalar!(UInt32Array, |value: u32| value.to_be_bytes()),
        DataType::UInt64 => scalar!(UInt64Array, |value: u64| value.to_be_bytes()),
        DataType::Float32 => scalar!(Float32Array, f32_bits),
        DataType::Float64 => scalar!(Float64Array, f64_bits),
        DataType::Utf8 => text_scalar!(StringArray),
        DataType::LargeUtf8 => text_scalar!(LargeStringArray),
        DataType::Date32 => scalar!(Date32Array, |value: i32| value.to_be_bytes()),
        DataType::Date64 => scalar!(Date64Array, |value: i64| value.to_be_bytes()),
        DataType::Time32(TimeUnit::Second) => {
            scalar!(Time32SecondArray, |value: i32| value.to_be_bytes());
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            scalar!(Time64MicrosecondArray, |value: i64| value.to_be_bytes());
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            scalar!(TimestampMillisecondArray, |value: i64| value.to_be_bytes());
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            scalar!(TimestampMicrosecondArray, |value: i64| value.to_be_bytes());
        }
        _ => return Err(unsupported_table_type(data_type)),
    }
    Ok(())
}

/// Returns the array-carried ordered flag for one dictionary column.
fn dictionary_ordered_flag(array: &dyn Array) -> Option<bool> {
    if !matches!(array.data_type(), DataType::Dictionary(_, _)) {
        return None;
    }
    macro_rules! ordered {
        ($key:ty) => {
            if let Some(dictionary) = array.as_any().downcast_ref::<DictionaryArray<$key>>() {
                return Some(dictionary.is_ordered());
            }
        };
    }
    ordered!(datafusion::arrow::datatypes::Int8Type);
    ordered!(datafusion::arrow::datatypes::Int16Type);
    ordered!(datafusion::arrow::datatypes::Int32Type);
    ordered!(datafusion::arrow::datatypes::Int64Type);
    ordered!(datafusion::arrow::datatypes::UInt8Type);
    ordered!(datafusion::arrow::datatypes::UInt16Type);
    ordered!(datafusion::arrow::datatypes::UInt32Type);
    ordered!(datafusion::arrow::datatypes::UInt64Type);
    None
}

/// Writes one logical cell, resolving dictionary values to their logical
/// scalar form.
fn write_cell(writer: &mut DigestWriter, array: &dyn Array, row: usize) -> Result<()> {
    if let DataType::Dictionary(_, _) = array.data_type() {
        macro_rules! dictionary {
            ($key:ty) => {
                if let Some(dictionary) = array.as_any().downcast_ref::<DictionaryArray<$key>>() {
                    let index = usize::try_from(dictionary.keys().value(row)).map_err(|_| {
                        CalcFlowError::InvalidArgument {
                            field: "static_inputs.schema".into(),
                            message: "dictionary index does not fit usize".into(),
                        }
                    })?;
                    return write_flat_cell(writer, dictionary.values().as_ref(), index);
                }
            };
        }
        dictionary!(datafusion::arrow::datatypes::Int8Type);
        dictionary!(datafusion::arrow::datatypes::Int16Type);
        dictionary!(datafusion::arrow::datatypes::Int32Type);
        dictionary!(datafusion::arrow::datatypes::Int64Type);
        dictionary!(datafusion::arrow::datatypes::UInt8Type);
        dictionary!(datafusion::arrow::datatypes::UInt16Type);
        dictionary!(datafusion::arrow::datatypes::UInt32Type);
        dictionary!(datafusion::arrow::datatypes::UInt64Type);
        return Err(CalcFlowError::InvalidArgument {
            field: "static_inputs.schema".into(),
            message: "dictionary index type is not an integer type".into(),
        });
    }
    write_flat_cell(writer, array, row)
}

fn write_table_schema(writer: &mut DigestWriter, table: &TableBatch) -> Result<usize> {
    let schema = table.schema();
    let fields = schema.fields();
    writer.checked_len(fields.len())?;
    let first_chunk_columns = table
        .batches()
        .first()
        .map(datafusion::arrow::record_batch::RecordBatch::columns);
    for (index, field) in fields.iter().enumerate() {
        writer.tag(0x20);
        writer.text(field.name().as_bytes())?;
        let ordered = first_chunk_columns.and_then(|columns| {
            columns
                .get(index)
                .and_then(|column| dictionary_ordered_flag(column.as_ref()))
        });
        write_table_type_ordered(writer, field.data_type(), ordered)?;
        writer.tag(u8::from(field.is_nullable()));
    }
    Ok(fields.len())
}

fn table_row_offsets(table: &TableBatch) -> Result<(Vec<usize>, usize)> {
    let mut offsets = Vec::with_capacity(table.batches().len() + 1);
    let mut total = 0_usize;
    for chunk in table.batches() {
        offsets.push(total);
        total =
            total
                .checked_add(chunk.num_rows())
                .ok_or_else(|| CalcFlowError::InvalidArgument {
                    field: "static_input.rows".into(),
                    message: "row count overflowed usize".into(),
                })?;
    }
    offsets.push(total);
    Ok((offsets, total))
}

fn write_table_rows(
    writer: &mut DigestWriter,
    table: &TableBatch,
    offsets: &[usize],
    total: usize,
    column_count: usize,
) -> Result<()> {
    let mut chunk_index = 0_usize;
    for row in 0..total {
        while row >= offsets[chunk_index + 1] {
            chunk_index += 1;
        }
        let local = row - offsets[chunk_index];
        for column in 0..column_count {
            write_cell(writer, table.batches()[chunk_index].column(column), local)?;
        }
    }
    Ok(())
}

/// Builds the canonical digest-v1 byte string for one table payload.
///
/// Cells are encoded in logical row-major order across every chunk; schema
/// metadata other than field name, logical type, and nullability is excluded.
pub(crate) fn table_digest(name: &str, table: &TableBatch) -> Result<StaticInputDigest> {
    let mut writer = DigestWriter::new();
    writer.tag(0x01);
    writer.text(name.as_bytes())?;
    writer.tag(0x10);
    let column_count = write_table_schema(&mut writer, table)?;
    let (offsets, total) = table_row_offsets(table)?;
    writer.checked_len(total)?;
    write_table_rows(&mut writer, table, &offsets, total, column_count)?;
    Ok(writer.finish())
}

fn array_digest_tag(payload: &LatchedArrayPayload) -> Result<u8> {
    array_dtype_tag(payload.dtype()).ok_or_else(|| CalcFlowError::InvalidArgument {
        field: "static_inputs.dtype".into(),
        message: format!(
            "array dtype {:?} is outside the digest-v1 set",
            payload.dtype()
        ),
    })
}

fn write_array_descriptor(writer: &mut DigestWriter, payload: &LatchedArrayPayload) -> Result<()> {
    writer.text(payload.backend().as_bytes())?;
    writer.tag(array_digest_tag(payload)?);
    writer.checked_len(payload.shape().len())?;
    for dimension in payload.shape() {
        writer.u64(*dimension);
    }
    writer.checked_len(payload.element_count())
}

fn write_array_cells(writer: &mut DigestWriter, payload: &LatchedArrayPayload) -> Result<()> {
    let mut next_value = 0_usize;
    for position in 0..payload.element_count() {
        if payload.is_null(position) {
            writer.tag(0x30);
            continue;
        }
        let cell = payload
            .cell(next_value)
            .ok_or_else(|| CalcFlowError::Internal {
                message: "latched array values do not cover the non-null cells".into(),
            })?;
        writer.scalar(&cell);
        next_value += 1;
    }
    Ok(())
}

/// Builds the canonical digest-v1 byte string for one latched array payload.
pub(crate) fn array_digest(name: &str, payload: &LatchedArrayPayload) -> Result<StaticInputDigest> {
    let mut writer = DigestWriter::new();
    writer.tag(0x01);
    writer.text(name.as_bytes())?;
    writer.tag(0x11);
    write_array_descriptor(&mut writer, payload)?;
    write_array_cells(&mut writer, payload)?;
    Ok(writer.finish())
}

/// Computes the canonical digest for one static input value by kind.
///
/// Table payloads digest directly over Arrow; array payloads must already be
/// engine-latched — a provider payload that was never latched cannot satisfy
/// a static declaration and fails here instead of hashing through a
/// provider-specific fallback.
pub(crate) fn digest_for_name(name: &str, batch: &Batch) -> Result<StaticInputDigest> {
    match batch.table_payload() {
        Ok(table) => table_digest(name, table),
        Err(_) => batch
            .latched_array_payload()
            .ok_or_else(|| CalcFlowError::InvalidArgument {
                field: format!("static_inputs.{name}.backend"),
                message: "array static inputs must be latched engine-owned values".into(),
            })
            .and_then(|payload| array_digest(name, payload)),
    }
}

/// The validated, latched static inputs for one streaming job (SCE-11).
///
/// `latched` holds engine-owned immutable batch handles installed before any
/// source opens; `digests` is the bounded evidence recorded into checkpoint
/// manifests and prepared-job identity. The value drops with the job driver,
/// which releases the handles exactly once on every exit path.
#[derive(Clone, Debug, Default)]
pub(crate) struct PreparedStaticInputs {
    pub(crate) latched: BTreeMap<String, Batch>,
    pub(crate) digests: BTreeMap<String, StaticInputDigest>,
}

/// Returns the canonical project-v3 spelling for one Arrow type, or `None`
/// for types the strict schema vocabulary cannot express.
fn canonical_type_string(data_type: &DataType) -> Option<String> {
    Some(
        match data_type {
            DataType::Boolean => "bool",
            DataType::Date32 => "date32",
            DataType::Date64 => "date64",
            DataType::Float32 => "float32",
            DataType::Float64 => "float64",
            DataType::Int8 => "int8",
            DataType::Int16 => "int16",
            DataType::Int32 => "int32",
            DataType::Int64 => "int64",
            DataType::LargeUtf8 => "large_string",
            DataType::Utf8 => "string",
            DataType::Time32(TimeUnit::Second) => "time32[s]",
            DataType::Time64(TimeUnit::Microsecond) => "time64[us]",
            DataType::Timestamp(TimeUnit::Millisecond, _) => "timestamp[ms]",
            DataType::Timestamp(TimeUnit::Microsecond, None) => "timestamp[us]",
            DataType::Timestamp(TimeUnit::Microsecond, Some(zone)) if zone.as_ref() == "UTC" => {
                "timestamp[us, UTC]"
            }
            DataType::UInt8 => "uint8",
            DataType::UInt16 => "uint16",
            DataType::UInt32 => "uint32",
            DataType::UInt64 => "uint64",
            _ => return None,
        }
        .to_owned(),
    )
}

fn static_error(path: String, message: String) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: path,
        message,
    }
}

fn qualify_static_input_error(name: &str, error: CalcFlowError) -> CalcFlowError {
    let CalcFlowError::InvalidArgument { field, message } = error else {
        return error;
    };
    let input_root = format!("static_inputs.{name}");
    let path = if field == input_root || field.starts_with(&format!("{input_root}.")) {
        field
    } else if let Some(suffix) = field.strip_prefix("static_inputs.") {
        format!("{input_root}.{suffix}")
    } else {
        format!("{input_root}.{field}")
    };
    static_error(path, message)
}

fn validate_array_descriptor(
    name: &str,
    backend: &str,
    dtype: &str,
    shape: &[u64],
    payload: &LatchedArrayPayload,
) -> Result<()> {
    if payload.backend() != backend {
        return Err(static_error(
            format!("static_inputs.{name}.backend"),
            format!(
                "declared {backend:?} but the value has {:?}",
                payload.backend()
            ),
        ));
    }
    if payload.dtype() != dtype {
        return Err(static_error(
            format!("static_inputs.{name}.dtype"),
            format!("declared {dtype:?} but the value has {:?}", payload.dtype()),
        ));
    }
    if !is_supported_array_dtype(dtype) {
        return Err(static_error(
            format!("static_inputs.{name}.dtype"),
            format!("array dtype {dtype:?} is outside the digest-v1 set"),
        ));
    }
    if payload.shape() != shape {
        return Err(static_error(
            format!("static_inputs.{name}.shape"),
            format!("declared {shape:?} but the value has {:?}", payload.shape()),
        ));
    }
    Ok(())
}

fn validate_array_against_spec(
    name: &str,
    backend: &str,
    dtype: &str,
    shape: &[u64],
    batch: &Batch,
) -> Result<()> {
    if batch.table_payload().is_ok() {
        return Err(static_error(
            format!("static_inputs.{name}.kind"),
            "declared an array static input but the value is a table batch".into(),
        ));
    }
    let payload = batch.latched_array_payload().ok_or_else(|| {
        static_error(
            format!("static_inputs.{name}.backend"),
            "array static inputs must be latched engine-owned values".into(),
        )
    })?;
    validate_array_descriptor(name, backend, dtype, shape, payload)
}

/// Validates one supplied batch against its declaration and latches it.
fn validate_against_spec(name: &str, spec: &StaticInputSpec, batch: &Batch) -> Result<()> {
    match spec {
        StaticInputSpec::Table { schema, .. } => validate_table_against_schema(name, schema, batch),
        StaticInputSpec::Array {
            backend,
            dtype,
            shape,
            ..
        } => validate_array_against_spec(name, backend, dtype, shape, batch),
    }
}

fn validate_table_field(
    name: &str,
    index: usize,
    declared: &ArrowFieldSpec,
    field: &datafusion::arrow::datatypes::Field,
) -> Result<()> {
    if field.name() != declared.name.as_str() {
        return Err(static_error(
            format!("static_inputs.{name}.schema[{index}].name"),
            format!(
                "declared field {:?} but the value has {:?}",
                declared.name,
                field.name()
            ),
        ));
    }
    let canonical = canonical_type_string(field.data_type()).ok_or_else(|| {
        static_error(
            format!("static_inputs.{name}.schema[{index}].data_type"),
            format!(
                "Arrow type {} has no strict schema spelling",
                field.data_type()
            ),
        )
    })?;
    if canonical != declared.data_type {
        return Err(static_error(
            format!("static_inputs.{name}.schema[{index}].data_type"),
            format!(
                "declared {:?} but the value has {canonical:?}",
                declared.data_type
            ),
        ));
    }
    if field.is_nullable() != declared.nullable {
        return Err(static_error(
            format!("static_inputs.{name}.schema[{index}].nullable"),
            format!(
                "declared {} but the value has {}",
                declared.nullable,
                field.is_nullable()
            ),
        ));
    }
    Ok(())
}

/// Validates an Arrow table value against its exact declared schema.
fn validate_table_against_schema(
    name: &str,
    schema: &[ArrowFieldSpec],
    batch: &Batch,
) -> Result<()> {
    let table = batch.table_payload().map_err(|_| {
        static_error(
            format!("static_inputs.{name}.kind"),
            "declared a table static input but the value is an array batch".into(),
        )
    })?;
    let fields = table.schema().fields();
    if fields.len() != schema.len() {
        return Err(static_error(
            format!("static_inputs.{name}.schema"),
            format!(
                "declared {} field(s) but the value has {}",
                schema.len(),
                fields.len()
            ),
        ));
    }
    for (index, declared) in schema.iter().enumerate() {
        validate_table_field(name, index, declared, fields[index].as_ref())?;
    }
    Ok(())
}

fn validate_static_input_names(
    declared: &BTreeMap<String, StaticInputSpec>,
    supplied: &BTreeMap<String, Batch>,
) -> Result<()> {
    if let Some(name) = declared.keys().find(|name| !supplied.contains_key(*name)) {
        return Err(static_error(
            format!("static_inputs.{name}"),
            "required static input is missing".into(),
        ));
    }
    if let Some(name) = supplied.keys().find(|name| !declared.contains_key(*name)) {
        return Err(static_error(
            format!("static_inputs.{name}"),
            "unexpected static input is not declared by the plan".into(),
        ));
    }
    Ok(())
}

fn prepare_static_input(
    name: &str,
    spec: &StaticInputSpec,
    batch: &Batch,
) -> Result<(Batch, StaticInputDigest)> {
    validate_against_spec(name, spec, batch)?;
    let digest =
        digest_for_name(name, batch).map_err(|error| qualify_static_input_error(name, error))?;
    Ok((batch.clone(), digest))
}

/// Validates the exact static input mapping, latches engine-owned handles,
/// and computes every canonical digest (API note section 8 order, steps 1-3).
///
/// # Errors
///
/// Returns a stable `static_inputs.{name}`-pathed error for a missing,
/// unexpected, or mismatched value; nothing is latched on failure.
pub(crate) fn prepare_static_inputs(
    declared: &BTreeMap<String, StaticInputSpec>,
    supplied: &BTreeMap<String, Batch>,
) -> Result<PreparedStaticInputs> {
    validate_static_input_names(declared, supplied)?;
    let mut prepared = PreparedStaticInputs::default();
    for (name, spec) in declared {
        let (batch, digest) = prepare_static_input(name, spec, &supplied[name])?;
        prepared.digests.insert(name.clone(), digest);
        prepared.latched.insert(name.clone(), batch);
    }
    Ok(prepared)
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use datafusion::arrow::{
        array::{
            ArrayRef, BooleanArray, Date32Array, Date64Array, DictionaryArray, Float32Array,
            Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray,
            StringArray, Time32SecondArray, Time64MicrosecondArray, TimestampMicrosecondArray,
            TimestampMillisecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
        },
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use sha2::{Digest, Sha256};

    use super::{STATIC_INPUT_DIGEST_VERSION, StaticInputDigest, digest_for_name};
    use crate::{Batch, BatchMetadata};

    fn text(bytes: &mut Vec<u8>, value: &[u8]) {
        bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
        bytes.extend_from_slice(value);
    }

    fn u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    fn table_batch(schema: Schema, columns: Vec<ArrayRef>) -> Batch {
        Batch::table(
            vec![RecordBatch::try_new(Arc::new(schema), columns).unwrap()],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    #[test]
    fn digest_version_string_is_frozen() {
        assert_eq!(
            STATIC_INPUT_DIGEST_VERSION,
            "calc_flow.static_input.digest.v1"
        );
    }

    #[test]
    fn table_digest_matches_the_canonical_tagged_bytes() {
        let batch = table_batch(
            Schema::new(vec![Field::new("factor", DataType::Float64, false)]),
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]))],
        );

        let mut expected = Vec::new();
        expected.extend_from_slice(b"calc_flow.static_input.digest.v1");
        expected.push(0x00);
        expected.push(0x01);
        text(&mut expected, b"weights");
        expected.push(0x10);
        u64(&mut expected, 1);
        expected.push(0x20);
        text(&mut expected, b"factor");
        expected.push(0x4a);
        expected.push(0x00);
        u64(&mut expected, 3);
        for value in [1.0f64, 2.0, 3.0] {
            expected.push(0x31);
            expected.extend_from_slice(&value.to_bits().to_be_bytes());
        }

        let digest = digest_for_name("weights", &batch).unwrap();
        assert_eq!(
            digest,
            StaticInputDigest {
                digest_version: STATIC_INPUT_DIGEST_VERSION.into(),
                sha256: sha256_hex(&expected),
            }
        );
    }

    #[test]
    fn table_digest_encodes_nulls_and_multiple_fields_in_row_major_order() {
        let schema = Schema::new(vec![
            Field::new("left", DataType::Int64, true),
            Field::new("right", DataType::Boolean, true),
        ]);
        let batch = table_batch(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![Some(7), None])),
                Arc::new(BooleanArray::from(vec![None, Some(true)])),
            ],
        );

        let mut expected = Vec::new();
        expected.extend_from_slice(b"calc_flow.static_input.digest.v1");
        expected.push(0x00);
        expected.push(0x01);
        text(&mut expected, b"t");
        expected.push(0x10);
        u64(&mut expected, 2);
        expected.push(0x20);
        text(&mut expected, b"left");
        expected.push(0x44);
        expected.push(0x01);
        expected.push(0x20);
        text(&mut expected, b"right");
        expected.push(0x40);
        expected.push(0x01);
        u64(&mut expected, 2);
        expected.push(0x31);
        expected.extend_from_slice(&7i64.to_be_bytes());
        expected.push(0x30);
        expected.push(0x30);
        expected.push(0x31);
        expected.push(0x01);

        assert_eq!(
            digest_for_name("t", &batch).unwrap().sha256,
            sha256_hex(&expected)
        );
    }

    #[test]
    fn table_digest_resolves_dictionary_cells_to_logical_values() {
        let values = StringArray::from(vec!["a", "b"]);
        let indices = Int32Array::from(vec![0, 1, 0]);
        let dictionary = DictionaryArray::new(indices, Arc::new(values));
        let batch = table_batch(
            Schema::new(vec![Field::new(
                "sym",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                false,
            )]),
            vec![Arc::new(dictionary)],
        );

        let mut expected = Vec::new();
        expected.extend_from_slice(b"calc_flow.static_input.digest.v1");
        expected.push(0x00);
        expected.push(0x01);
        text(&mut expected, b"d");
        expected.push(0x10);
        u64(&mut expected, 1);
        expected.push(0x20);
        text(&mut expected, b"sym");
        expected.push(0x54);
        expected.push(0x43);
        expected.push(0x4b);
        expected.push(0x00);
        expected.push(0x00);
        u64(&mut expected, 3);
        for value in ["a", "b", "a"] {
            expected.push(0x31);
            text(&mut expected, value.as_bytes());
        }

        assert_eq!(
            digest_for_name("d", &batch).unwrap().sha256,
            sha256_hex(&expected)
        );
    }

    #[test]
    fn table_digest_canonicalizes_float_nans_and_ignores_chunk_boundaries() {
        let schema = Schema::new(vec![Field::new("v", DataType::Float64, true)]);
        let whole = table_batch(
            schema.clone(),
            vec![Arc::new(Float64Array::from(vec![
                Some(f64::NAN),
                Some(2.0),
            ]))],
        );
        let chunked = Batch::table(
            vec![
                RecordBatch::try_new(
                    Arc::new(schema),
                    vec![Arc::new(Float64Array::from(vec![Some(f64::NAN)]))],
                )
                .unwrap(),
                RecordBatch::try_new(
                    Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)])),
                    vec![Arc::new(Float64Array::from(vec![Some(2.0)]))],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap();
        let negative_nan = table_batch(
            Schema::new(vec![Field::new("v", DataType::Float64, true)]),
            vec![Arc::new(Float64Array::from(vec![
                Some(-f64::NAN),
                Some(2.0),
            ]))],
        );

        let digest = digest_for_name("x", &whole).unwrap();
        assert_eq!(digest_for_name("x", &chunked).unwrap(), digest);
        assert_eq!(digest_for_name("x", &negative_nan).unwrap(), digest);

        let mut expected = Vec::new();
        expected.extend_from_slice(b"calc_flow.static_input.digest.v1");
        expected.push(0x00);
        expected.push(0x01);
        text(&mut expected, b"x");
        expected.push(0x10);
        u64(&mut expected, 1);
        expected.push(0x20);
        text(&mut expected, b"v");
        expected.push(0x4a);
        expected.push(0x01);
        u64(&mut expected, 2);
        expected.push(0x31);
        expected.extend_from_slice(&0x7ff8_0000_0000_0000u64.to_be_bytes());
        expected.push(0x31);
        expected.extend_from_slice(&2.0f64.to_bits().to_be_bytes());
        assert_eq!(digest.sha256, sha256_hex(&expected));
    }

    #[test]
    fn table_digest_accepts_every_frozen_primitive_type() {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(BooleanArray::from(vec![true])),
            Arc::new(Int8Array::from(vec![1_i8])),
            Arc::new(Int16Array::from(vec![2_i16])),
            Arc::new(Int32Array::from(vec![3_i32])),
            Arc::new(Int64Array::from(vec![4_i64])),
            Arc::new(UInt8Array::from(vec![5_u8])),
            Arc::new(UInt16Array::from(vec![6_u16])),
            Arc::new(UInt32Array::from(vec![7_u32])),
            Arc::new(UInt64Array::from(vec![8_u64])),
            Arc::new(Float32Array::from(vec![9.0_f32])),
            Arc::new(Float64Array::from(vec![10.0_f64])),
            Arc::new(StringArray::from(vec!["text"])),
            Arc::new(LargeStringArray::from(vec!["large"])),
            Arc::new(Date32Array::from(vec![11_i32])),
            Arc::new(Date64Array::from(vec![12_i64])),
            Arc::new(Time32SecondArray::from(vec![13_i32])),
            Arc::new(Time64MicrosecondArray::from(vec![14_i64])),
            Arc::new(TimestampMillisecondArray::from(vec![15_i64]).with_timezone("UTC")),
            Arc::new(TimestampMicrosecondArray::from(vec![16_i64])),
            Arc::new(TimestampMicrosecondArray::from(vec![17_i64]).with_timezone("UTC")),
        ];
        let fields = columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                Field::new(format!("value_{index}"), column.data_type().clone(), false)
            })
            .collect::<Vec<_>>();
        let batch = table_batch(Schema::new(fields), columns);

        let digest = digest_for_name("all_types", &batch).unwrap();

        assert_eq!(digest.digest_version, STATIC_INPUT_DIGEST_VERSION);
        assert_eq!(digest.sha256.len(), 64);
    }

    #[test]
    fn table_digest_resolves_every_frozen_dictionary_key_type() {
        let values: ArrayRef = Arc::new(StringArray::from(vec!["value"]));
        macro_rules! dictionary_column {
            ($array:ident, $value:expr) => {
                Arc::new(DictionaryArray::new(
                    $array::from(vec![$value]),
                    Arc::clone(&values),
                )) as ArrayRef
            };
        }
        let columns = vec![
            dictionary_column!(Int8Array, 0_i8),
            dictionary_column!(Int16Array, 0_i16),
            dictionary_column!(Int32Array, 0_i32),
            dictionary_column!(Int64Array, 0_i64),
            dictionary_column!(UInt8Array, 0_u8),
            dictionary_column!(UInt16Array, 0_u16),
            dictionary_column!(UInt32Array, 0_u32),
            dictionary_column!(UInt64Array, 0_u64),
        ];
        let fields = columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                Field::new(
                    format!("dictionary_{index}"),
                    column.data_type().clone(),
                    false,
                )
            })
            .collect::<Vec<_>>();
        let batch = table_batch(Schema::new(fields), columns);

        assert_eq!(digest_for_name("keys", &batch).unwrap().sha256.len(), 64);
    }

    #[test]
    fn array_digest_encodes_descriptor_and_canonical_float32_cells() {
        let batch = Batch::static_array_float(
            "numpy",
            "float32",
            vec![2, 2],
            None,
            vec![1.0, -0.0, f64::INFINITY, f64::NAN],
        )
        .unwrap();

        let mut expected = Vec::new();
        expected.extend_from_slice(b"calc_flow.static_input.digest.v1");
        expected.push(0x00);
        expected.push(0x01);
        text(&mut expected, b"w");
        expected.push(0x11);
        text(&mut expected, b"numpy");
        expected.push(0x49);
        u64(&mut expected, 2);
        u64(&mut expected, 2);
        u64(&mut expected, 2);
        u64(&mut expected, 4);
        for bits in [
            1.0f32.to_bits(),
            (-0.0f32).to_bits(),
            f32::INFINITY.to_bits(),
            0x7fc0_0000u32,
        ] {
            expected.push(0x31);
            expected.extend_from_slice(&bits.to_be_bytes());
        }

        assert_eq!(
            digest_for_name("w", &batch).unwrap().sha256,
            sha256_hex(&expected)
        );
    }

    #[test]
    fn array_digest_encodes_null_cells_and_unsigned_values() {
        let batch = Batch::static_array_uint(
            "numpy",
            "uint64",
            vec![3],
            Some(vec![false, true, false]),
            vec![u64::MAX, 4],
        )
        .unwrap();

        let mut expected = Vec::new();
        expected.extend_from_slice(b"calc_flow.static_input.digest.v1");
        expected.push(0x00);
        expected.push(0x01);
        text(&mut expected, b"u");
        expected.push(0x11);
        text(&mut expected, b"numpy");
        expected.push(0x48);
        u64(&mut expected, 1);
        u64(&mut expected, 3);
        u64(&mut expected, 3);
        expected.push(0x31);
        expected.extend_from_slice(&u64::MAX.to_be_bytes());
        expected.push(0x30);
        expected.push(0x31);
        expected.extend_from_slice(&4u64.to_be_bytes());

        assert_eq!(
            digest_for_name("u", &batch).unwrap().sha256,
            sha256_hex(&expected)
        );
    }

    #[test]
    fn array_constructors_reject_unsupported_dtypes_and_shape_mismatches() {
        assert!(
            Batch::static_array_float("numpy", "complex128", vec![1], None, vec![1.0]).is_err()
        );
        assert!(
            Batch::static_array_int("numpy", "int32", vec![2, 2], None, vec![1, 2, 3]).is_err()
        );
        assert!(Batch::static_array_bool("numpy", vec![2], Some(vec![true]), vec![true]).is_err());
        assert!(Batch::static_array_int("numpy", "int8", vec![1], None, vec![300]).is_err());
        let empty = Batch::static_array_float("numpy", "float64", vec![0], None, vec![]).unwrap();
        assert_eq!(empty.num_rows(), 0);
    }

    fn declared_table() -> super::StaticInputSpec {
        super::StaticInputSpec::Table {
            name: "weights".into(),
            mutability: super::StaticMutability::Static,
            schema: vec![super::ArrowFieldSpec {
                name: "factor".into(),
                data_type: "float64".into(),
                nullable: false,
            }],
        }
    }

    fn declared_array() -> super::StaticInputSpec {
        super::StaticInputSpec::Array {
            name: "w".into(),
            mutability: super::StaticMutability::Static,
            backend: "numpy".into(),
            dtype: "float32".into(),
            shape: vec![2],
        }
    }

    fn error_text(error: crate::CalcFlowError) -> String {
        match error {
            crate::CalcFlowError::InvalidArgument { field, message } => {
                format!("{field}: {message}")
            }
            other => other.to_string(),
        }
    }

    #[test]
    fn preflight_reports_missing_and_unexpected_names_first() {
        let declared = BTreeMap::from([("weights".to_string(), declared_table())]);
        let error = super::prepare_static_inputs(&declared, &BTreeMap::new()).unwrap_err();
        assert_eq!(
            error_text(error),
            "static_inputs.weights: required static input is missing"
        );

        let supplied = BTreeMap::from([
            (
                "weights".to_string(),
                table_batch(
                    Schema::new(vec![Field::new("factor", DataType::Float64, false)]),
                    vec![Arc::new(Float64Array::from(vec![1.0]))],
                ),
            ),
            (
                "other".to_string(),
                table_batch(
                    Schema::new(vec![Field::new("factor", DataType::Float64, false)]),
                    vec![Arc::new(Float64Array::from(vec![1.0]))],
                ),
            ),
        ]);
        let error = super::prepare_static_inputs(&declared, &supplied).unwrap_err();
        assert_eq!(
            error_text(error),
            "static_inputs.other: unexpected static input is not declared by the plan"
        );
    }

    #[test]
    fn preflight_validates_table_schemas_at_the_precise_path() {
        let declared = BTreeMap::from([("weights".to_string(), declared_table())]);
        let wrong_count = table_batch(
            Schema::new(vec![
                Field::new("factor", DataType::Float64, false),
                Field::new("extra", DataType::Float64, false),
            ]),
            vec![
                Arc::new(Float64Array::from(vec![1.0])),
                Arc::new(Float64Array::from(vec![2.0])),
            ],
        );
        let error = super::prepare_static_inputs(
            &declared,
            &BTreeMap::from([("weights".to_string(), wrong_count)]),
        )
        .unwrap_err();
        assert!(error_text(error).starts_with("static_inputs.weights.schema"));

        let wrong_name = table_batch(
            Schema::new(vec![Field::new("coefficient", DataType::Float64, false)]),
            vec![Arc::new(Float64Array::from(vec![1.0]))],
        );
        let error = super::prepare_static_inputs(
            &declared,
            &BTreeMap::from([("weights".to_string(), wrong_name)]),
        )
        .unwrap_err();
        assert!(error_text(error).starts_with("static_inputs.weights.schema[0].name"));

        let unsupported_type = table_batch(
            Schema::new(vec![Field::new("factor", DataType::Binary, false)]),
            vec![Arc::new(datafusion::arrow::array::BinaryArray::from(vec![
                b"x".as_slice(),
            ]))],
        );
        let error = super::prepare_static_inputs(
            &declared,
            &BTreeMap::from([("weights".to_string(), unsupported_type)]),
        )
        .unwrap_err();
        assert!(error_text(error).starts_with("static_inputs.weights.schema[0].data_type"));

        let wrong_type = table_batch(
            Schema::new(vec![Field::new("factor", DataType::Int64, false)]),
            vec![Arc::new(Int64Array::from(vec![1]))],
        );
        let error = super::prepare_static_inputs(
            &declared,
            &BTreeMap::from([("weights".to_string(), wrong_type)]),
        )
        .unwrap_err();
        assert!(error_text(error).starts_with("static_inputs.weights.schema[0].data_type"));

        let wrong_nullable = table_batch(
            Schema::new(vec![Field::new("factor", DataType::Float64, true)]),
            vec![Arc::new(Float64Array::from(vec![1.0]))],
        );
        let error = super::prepare_static_inputs(
            &declared,
            &BTreeMap::from([("weights".to_string(), wrong_nullable)]),
        )
        .unwrap_err();
        assert!(error_text(error).starts_with("static_inputs.weights.schema[0].nullable"));

        let wrong_kind =
            Batch::static_array_float("numpy", "float64", vec![1], None, vec![1.0]).unwrap();
        let error = super::prepare_static_inputs(
            &declared,
            &BTreeMap::from([("weights".to_string(), wrong_kind)]),
        )
        .unwrap_err();
        assert!(error_text(error).starts_with("static_inputs.weights.kind"));
    }

    #[test]
    fn digest_error_qualification_keeps_one_static_input_root() {
        let mut writer = super::DigestWriter::new();
        let error = super::write_table_type(&mut writer, &DataType::Binary).unwrap_err();

        let qualified = super::qualify_static_input_error("weights", error);
        let rendered = error_text(qualified);

        assert!(rendered.starts_with("static_inputs.weights.schema:"));
        assert_eq!(rendered.matches("static_inputs").count(), 1);
    }

    #[derive(Debug)]
    struct UnlatchedArray;

    impl crate::ExternalPayload for UnlatchedArray {
        fn backend(&self) -> &str {
            "numpy"
        }

        fn len(&self) -> usize {
            1
        }

        fn estimated_bytes(&self) -> usize {
            8
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn digest_rejects_an_unlatched_external_array_at_the_named_backend_path() {
        let batch = Batch::external(Arc::new(UnlatchedArray), BatchMetadata::default()).unwrap();

        let error = digest_for_name("weights", &batch).unwrap_err();

        assert_eq!(
            error_text(error),
            "static_inputs.weights.backend: array static inputs must be latched engine-owned values"
        );
    }

    #[test]
    fn preflight_validates_array_backend_dtype_and_shape() {
        let declared = BTreeMap::from([("w".to_string(), declared_array())]);
        let wrong_backend =
            Batch::static_array_float("jax", "float32", vec![2], None, vec![1.0, 2.0]).unwrap();
        let error = super::prepare_static_inputs(
            &declared,
            &BTreeMap::from([("w".to_string(), wrong_backend)]),
        )
        .unwrap_err();
        assert!(error_text(error).starts_with("static_inputs.w.backend"));

        let wrong_dtype =
            Batch::static_array_float("numpy", "float64", vec![2], None, vec![1.0, 2.0]).unwrap();
        let error = super::prepare_static_inputs(
            &declared,
            &BTreeMap::from([("w".to_string(), wrong_dtype)]),
        )
        .unwrap_err();
        assert!(error_text(error).starts_with("static_inputs.w.dtype"));

        let wrong_shape =
            Batch::static_array_float("numpy", "float32", vec![3], None, vec![1.0, 2.0, 3.0])
                .unwrap();
        let error = super::prepare_static_inputs(
            &declared,
            &BTreeMap::from([("w".to_string(), wrong_shape)]),
        )
        .unwrap_err();
        assert!(error_text(error).starts_with("static_inputs.w.shape"));

        let wrong_table_kind = table_batch(
            Schema::new(vec![Field::new("factor", DataType::Float64, false)]),
            vec![Arc::new(Float64Array::from(vec![1.0]))],
        );
        let error = super::prepare_static_inputs(
            &declared,
            &BTreeMap::from([("w".to_string(), wrong_table_kind)]),
        )
        .unwrap_err();
        assert!(error_text(error).starts_with("static_inputs.w.kind"));
    }

    #[test]
    fn preflight_latches_and_digests_the_exact_declared_set() {
        let declared = BTreeMap::from([
            ("weights".to_string(), declared_table()),
            ("w".to_string(), declared_array()),
        ]);
        let supplied = BTreeMap::from([
            (
                "weights".to_string(),
                table_batch(
                    Schema::new(vec![Field::new("factor", DataType::Float64, false)]),
                    vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]))],
                ),
            ),
            (
                "w".to_string(),
                Batch::static_array_float("numpy", "float32", vec![2], None, vec![1.0, 2.0])
                    .unwrap(),
            ),
        ]);
        let prepared = super::prepare_static_inputs(&declared, &supplied).unwrap();
        assert_eq!(prepared.digests.len(), 2);
        assert_eq!(prepared.latched.len(), 2);
        for digest in prepared.digests.values() {
            assert_eq!(digest.digest_version, STATIC_INPUT_DIGEST_VERSION);
            assert_eq!(digest.sha256.len(), 64);
        }
    }

    #[test]
    fn array_constructors_keep_values_engine_owned_and_detached() {
        let mut values = vec![1.0f64, 2.0];
        let batch =
            Batch::static_array_float("numpy", "float64", vec![2], None, values.clone()).unwrap();
        values[0] = 99.0;
        let digest_before = digest_for_name("k", &batch).unwrap();
        assert_eq!(digest_for_name("k", &batch).unwrap(), digest_before);
    }
}
