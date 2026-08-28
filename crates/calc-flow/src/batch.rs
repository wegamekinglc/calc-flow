use std::{any::Any, collections::BTreeMap, fmt::Debug, sync::Arc};

use datafusion::arrow::{datatypes::SchemaRef, record_batch::RecordBatch};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{CalcFlowError, JsonMap, Result};

/// Adds `addend` to `total`, reporting overflow as a typed error (spec S10.2:
/// byte and row sums use checked arithmetic; overflow is a typed error).
pub(crate) fn checked_accumulate(
    total: usize,
    addend: usize,
    field: &'static str,
) -> Result<usize> {
    total
        .checked_add(addend)
        .ok_or_else(|| CalcFlowError::InvalidArgument {
            field: field.into(),
            message: "size sum overflowed usize".into(),
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BatchKind {
    Table,
    Array,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct BatchMetadata {
    source: String,
    sequence: u64,
    attributes: JsonMap,
}

impl BatchMetadata {
    /// Creates metadata for a batch.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when `source` contains a NUL
    /// character.
    pub fn new(
        source: impl Into<String>,
        sequence: u64,
        attributes: BTreeMap<String, Value>,
    ) -> Result<Self> {
        let source = source.into();
        if source.contains('\0') {
            return Err(CalcFlowError::InvalidArgument {
                field: "metadata.source".into(),
                message: "must not contain NUL".into(),
            });
        }
        Ok(Self {
            source,
            sequence,
            attributes,
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn attributes(&self) -> &JsonMap {
        &self.attributes
    }
}

#[derive(Clone, Debug)]
pub struct TableBatch {
    schema: SchemaRef,
    batches: Arc<[RecordBatch]>,
    rows: usize,
}

impl TableBatch {
    fn new(batches: Vec<RecordBatch>) -> Result<Self> {
        let schema = batches.first().map(RecordBatch::schema).ok_or_else(|| {
            CalcFlowError::InvalidArgument {
                field: "batches".into(),
                message: "must contain at least one RecordBatch; represent an empty table with one zero-row batch".into(),
            }
        })?;
        if batches.iter().any(|batch| batch.schema() != schema) {
            return Err(CalcFlowError::InvalidArgument {
                field: "batches".into(),
                message: "schemas must match".into(),
            });
        }
        let rows = batches.iter().try_fold(0_usize, |rows, batch| {
            checked_accumulate(rows, batch.num_rows(), "batches")
        })?;
        Ok(Self {
            schema,
            batches: batches.into(),
            rows,
        })
    }

    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    /// Estimates the in-memory cost of the visible Arrow slices in bytes.
    ///
    /// Each column of each record batch is charged its Arrow slice memory
    /// size, so sliced arrays sharing a larger backing allocation are charged
    /// only for their visible window. The estimate is a logical queue charge,
    /// not a process RSS measurement.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when Arrow cannot measure a
    /// column or the summed size overflows `usize`.
    pub fn estimated_bytes(&self) -> Result<usize> {
        self.batches.iter().try_fold(0_usize, |total, batch| {
            batch.columns().iter().try_fold(total, |total, column| {
                let bytes = column.to_data().get_slice_memory_size().map_err(|error| {
                    CalcFlowError::InvalidArgument {
                        field: "batch".into(),
                        message: format!("Arrow slice memory could not be measured: {error}"),
                    }
                })?;
                checked_accumulate(total, bytes, "batch")
            })
        })
    }
}

#[allow(clippy::len_without_is_empty)]
pub trait ExternalPayload: Any + Debug + Send + Sync {
    fn backend(&self) -> &str;
    fn len(&self) -> usize;
    /// Returns an exact or conservative estimate of the payload's visible
    /// in-memory cost in bytes.
    ///
    /// Implementations must never under-report the visible payload cost;
    /// there is no opt-out (spec S10.2). The estimate is a logical queue
    /// charge used for backpressure accounting, not a process RSS
    /// measurement, and shared payloads are charged per consumer.
    ///
    /// An opaque host object that exposes no byte size has no observable
    /// cost to under-report; its charge is defined by convention as a
    /// logical per-element estimate (the built-in hosts charge one `u64`
    /// per element). That convention prices queue occupancy for such hosts
    /// and is a documented charging rule, not a memory bound: it can sit
    /// below a large non-array host's true footprint.
    fn estimated_bytes(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
}

#[derive(Clone, Debug)]
enum BatchPayload {
    Table(TableBatch),
    External(Arc<dyn ExternalPayload>),
}

/// The engine-owned immutable storage behind one latched static array
/// (SCE-11). Values are copied out of any host array at latch time, so the
/// payload can never alias caller-mutable memory.
#[derive(Debug)]
pub(crate) struct LatchedArrayPayload {
    backend: String,
    dtype: String,
    shape: Vec<u64>,
    nulls: Option<Vec<bool>>,
    values: LatchedArrayValues,
}

#[derive(Debug)]
enum LatchedArrayValues {
    Bool(Vec<bool>),
    Int(Vec<i64>),
    Uint(Vec<u64>),
    Float(Vec<f64>),
}

impl LatchedArrayPayload {
    pub(crate) fn backend(&self) -> &str {
        &self.backend
    }

    pub(crate) fn dtype(&self) -> &str {
        &self.dtype
    }

    pub(crate) fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Total logical element count including null positions.
    pub(crate) fn element_count(&self) -> usize {
        self.shape.iter().fold(1_usize, |total, dimension| {
            total.saturating_mul(usize::try_from(*dimension).unwrap_or(usize::MAX))
        })
    }

    pub(crate) fn is_null(&self, position: usize) -> bool {
        self.nulls.as_ref().is_some_and(|nulls| nulls[position])
    }

    /// Returns the canonical scalar bytes for the next non-null value.
    ///
    /// Constructors already reject values outside the declared dtype's
    /// range, so the narrowing casts here are lossless.
    pub(crate) fn cell(&self, value_index: usize) -> Option<Vec<u8>> {
        match &self.values {
            LatchedArrayValues::Bool(values) => {
                values.get(value_index).map(|value| vec![u8::from(*value)])
            }
            LatchedArrayValues::Int(values) => {
                values
                    .get(value_index)
                    .map(|value| match self.dtype.as_str() {
                        "int8" => i8::try_from(*value)
                            .expect("constructors validate the declared width")
                            .to_be_bytes()
                            .to_vec(),
                        "int16" => i16::try_from(*value)
                            .expect("constructors validate the declared width")
                            .to_be_bytes()
                            .to_vec(),
                        "int32" => i32::try_from(*value)
                            .expect("constructors validate the declared width")
                            .to_be_bytes()
                            .to_vec(),
                        _ => value.to_be_bytes().to_vec(),
                    })
            }
            LatchedArrayValues::Uint(values) => {
                values
                    .get(value_index)
                    .map(|value| match self.dtype.as_str() {
                        "uint8" => u8::try_from(*value)
                            .expect("constructors validate the declared width")
                            .to_be_bytes()
                            .to_vec(),
                        "uint16" => u16::try_from(*value)
                            .expect("constructors validate the declared width")
                            .to_be_bytes()
                            .to_vec(),
                        "uint32" => u32::try_from(*value)
                            .expect("constructors validate the declared width")
                            .to_be_bytes()
                            .to_vec(),
                        _ => value.to_be_bytes().to_vec(),
                    })
            }
            LatchedArrayValues::Float(values) => values.get(value_index).map(|value| {
                if self.dtype == "float32" {
                    // f64 to f32 narrowing is the documented lossless carrier
                    // rule for latched float32 values (digest v1).
                    #[allow(clippy::cast_possible_truncation)]
                    let narrowed = *value as f32;
                    let bits = if value.is_nan() {
                        0x7fc0_0000u32
                    } else {
                        narrowed.to_bits()
                    };
                    bits.to_be_bytes().to_vec()
                } else {
                    let bits = if value.is_nan() {
                        0x7ff8_0000_0000_0000u64
                    } else {
                        value.to_bits()
                    };
                    bits.to_be_bytes().to_vec()
                }
            }),
        }
    }
}

impl ExternalPayload for LatchedArrayPayload {
    fn backend(&self) -> &str {
        &self.backend
    }

    fn len(&self) -> usize {
        self.element_count()
    }

    fn estimated_bytes(&self) -> usize {
        let per_element = match &self.values {
            LatchedArrayValues::Bool(values) => size_of::<bool>() * values.len(),
            LatchedArrayValues::Int(values) => size_of::<i64>() * values.len(),
            LatchedArrayValues::Uint(values) => size_of::<u64>() * values.len(),
            LatchedArrayValues::Float(values) => size_of::<f64>() * values.len(),
        };
        per_element
            + self.backend.len()
            + self.dtype.len()
            + self.shape.len().saturating_mul(size_of::<u64>())
            + self
                .nulls
                .as_ref()
                .map_or(0, |nulls| nulls.len().saturating_mul(size_of::<bool>()))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Family tag used to check that the value carrier matches the dtype
/// spelling supplied by the latch caller.
#[derive(Clone, Copy, PartialEq)]
enum LatchedValuesFamily {
    Bool,
    Int,
    Uint,
    Float,
}

fn latched_family(dtype: &str) -> Option<LatchedValuesFamily> {
    Some(match dtype {
        "bool" => LatchedValuesFamily::Bool,
        "int8" | "int16" | "int32" | "int64" => LatchedValuesFamily::Int,
        "uint8" | "uint16" | "uint32" | "uint64" => LatchedValuesFamily::Uint,
        "float32" | "float64" => LatchedValuesFamily::Float,
        _ => return None,
    })
}

fn validate_latched_descriptor(
    backend: &str,
    dtype: &str,
    family: LatchedValuesFamily,
) -> Result<()> {
    if backend.is_empty() {
        return Err(CalcFlowError::InvalidArgument {
            field: "static_inputs.backend".into(),
            message: "must not be empty".into(),
        });
    }
    if latched_family(dtype) != Some(family) {
        return Err(CalcFlowError::InvalidArgument {
            field: "static_inputs.dtype".into(),
            message: format!("dtype {dtype:?} does not match the supplied value family"),
        });
    }
    Ok(())
}

fn latched_element_count(shape: &[u64]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |element_count, dimension| {
        let dimension =
            usize::try_from(*dimension).map_err(|_| CalcFlowError::InvalidArgument {
                field: "static_inputs.shape".into(),
                message: "dimension exceeds the platform address range".into(),
            })?;
        element_count
            .checked_mul(dimension)
            .ok_or_else(|| CalcFlowError::InvalidArgument {
                field: "static_inputs.shape".into(),
                message: "element count overflowed usize".into(),
            })
    })
}

fn validate_latched_counts(
    nulls: Option<&[bool]>,
    value_count: usize,
    element_count: usize,
) -> Result<()> {
    if nulls.is_some_and(|nulls| nulls.len() != element_count) {
        return Err(CalcFlowError::InvalidArgument {
            field: "static_inputs.shape".into(),
            message: "null mask length must equal the element count".into(),
        });
    }
    let null_count = nulls.map_or(0, |nulls| nulls.iter().filter(|null| **null).count());
    if value_count != element_count - null_count {
        return Err(CalcFlowError::InvalidArgument {
            field: "static_inputs.shape".into(),
            message: "value count must equal the non-null element count".into(),
        });
    }
    Ok(())
}

fn validate_latched_shape(
    backend: &str,
    dtype: &str,
    family: LatchedValuesFamily,
    shape: &[u64],
    nulls: Option<&[bool]>,
    value_count: usize,
) -> Result<usize> {
    validate_latched_descriptor(backend, dtype, family)?;
    let element_count = latched_element_count(shape)?;
    validate_latched_counts(nulls, value_count, element_count)?;
    Ok(element_count)
}

fn validate_int_widths(dtype: &str, values: &[i64]) -> Result<()> {
    let out_of_range = |value: i64| match dtype {
        "int8" => !(i64::from(i8::MIN)..=i64::from(i8::MAX)).contains(&value),
        "int16" => !(i64::from(i16::MIN)..=i64::from(i16::MAX)).contains(&value),
        "int32" => !(i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&value),
        _ => false,
    };
    values
        .iter()
        .find(|value| out_of_range(**value))
        .map_or(Ok(()), |value| {
            Err(CalcFlowError::InvalidArgument {
                field: "static_inputs.dtype".into(),
                message: format!("value {value} does not fit the declared {dtype} width"),
            })
        })
}

fn validate_uint_widths(dtype: &str, values: &[u64]) -> Result<()> {
    let out_of_range = |value: u64| match dtype {
        "uint8" => value > u64::from(u8::MAX),
        "uint16" => value > u64::from(u16::MAX),
        "uint32" => value > u64::from(u32::MAX),
        _ => false,
    };
    values
        .iter()
        .find(|value| out_of_range(**value))
        .map_or(Ok(()), |value| {
            Err(CalcFlowError::InvalidArgument {
                field: "static_inputs.dtype".into(),
                message: format!("value {value} does not fit the declared {dtype} width"),
            })
        })
}

#[derive(Clone, Debug)]
pub struct Batch {
    payload: BatchPayload,
    metadata: BatchMetadata,
}

impl Batch {
    /// Creates a table batch from one or more identically shaped Arrow batches.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when `batches` is empty or
    /// contains mismatched schemas.
    pub fn table(batches: Vec<RecordBatch>, metadata: BatchMetadata) -> Result<Self> {
        Ok(Self {
            payload: BatchPayload::Table(TableBatch::new(batches)?),
            metadata,
        })
    }

    /// Creates an array batch from an externally owned payload.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the payload backend is
    /// empty.
    pub fn external(payload: Arc<dyn ExternalPayload>, metadata: BatchMetadata) -> Result<Self> {
        if payload.backend().is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "backend".into(),
                message: "must not be empty".into(),
            });
        }
        Ok(Self {
            payload: BatchPayload::External(payload),
            metadata,
        })
    }

    /// Latches a boolean static array into engine-owned immutable storage.
    ///
    /// The values are copied at construction, so the returned batch can never
    /// alias caller-mutable memory (SCE-11).
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the backend is empty,
    /// the shape does not match the value and mask lengths, or the element
    /// count overflows.
    pub fn static_array_bool(
        backend: &str,
        shape: Vec<u64>,
        nulls: Option<Vec<bool>>,
        values: Vec<bool>,
    ) -> Result<Self> {
        validate_latched_shape(
            backend,
            "bool",
            LatchedValuesFamily::Bool,
            &shape,
            nulls.as_deref(),
            values.len(),
        )?;
        Ok(Self::latched(LatchedArrayPayload {
            backend: backend.into(),
            dtype: "bool".into(),
            shape,
            nulls,
            values: LatchedArrayValues::Bool(values),
        }))
    }

    /// Latches a signed-integer static array (`int8`–`int64`).
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for an empty backend, an
    /// unsupported dtype spelling, a width overflow, or a shape, mask, and
    /// value-length disagreement.
    pub fn static_array_int(
        backend: &str,
        dtype: &str,
        shape: Vec<u64>,
        nulls: Option<Vec<bool>>,
        values: Vec<i64>,
    ) -> Result<Self> {
        validate_latched_shape(
            backend,
            dtype,
            LatchedValuesFamily::Int,
            &shape,
            nulls.as_deref(),
            values.len(),
        )?;
        validate_int_widths(dtype, &values)?;
        Ok(Self::latched(LatchedArrayPayload {
            backend: backend.into(),
            dtype: dtype.into(),
            shape,
            nulls,
            values: LatchedArrayValues::Int(values),
        }))
    }

    /// Latches an unsigned-integer static array (`uint8`–`uint64`).
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for an empty backend, an
    /// unsupported dtype spelling, a width overflow, or a shape, mask, and
    /// value-length disagreement.
    pub fn static_array_uint(
        backend: &str,
        dtype: &str,
        shape: Vec<u64>,
        nulls: Option<Vec<bool>>,
        values: Vec<u64>,
    ) -> Result<Self> {
        validate_latched_shape(
            backend,
            dtype,
            LatchedValuesFamily::Uint,
            &shape,
            nulls.as_deref(),
            values.len(),
        )?;
        validate_uint_widths(dtype, &values)?;
        Ok(Self::latched(LatchedArrayPayload {
            backend: backend.into(),
            dtype: dtype.into(),
            shape,
            nulls,
            values: LatchedArrayValues::Uint(values),
        }))
    }

    /// Latches a floating-point static array (`float32`/`float64`).
    ///
    /// `float32` values are carried at `f64` width losslessly and re-derived
    /// for digesting; NaN payloads canonicalize per digest v1.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for an empty backend, an
    /// unsupported dtype spelling, or a shape, mask, and value-length
    /// disagreement.
    pub fn static_array_float(
        backend: &str,
        dtype: &str,
        shape: Vec<u64>,
        nulls: Option<Vec<bool>>,
        values: Vec<f64>,
    ) -> Result<Self> {
        validate_latched_shape(
            backend,
            dtype,
            LatchedValuesFamily::Float,
            &shape,
            nulls.as_deref(),
            values.len(),
        )?;
        Ok(Self::latched(LatchedArrayPayload {
            backend: backend.into(),
            dtype: dtype.into(),
            shape,
            nulls,
            values: LatchedArrayValues::Float(values),
        }))
    }

    fn latched(payload: LatchedArrayPayload) -> Self {
        Self {
            payload: BatchPayload::External(Arc::new(payload)),
            metadata: BatchMetadata::default(),
        }
    }

    pub fn kind(&self) -> BatchKind {
        match &self.payload {
            BatchPayload::Table(_) => BatchKind::Table,
            BatchPayload::External(_) => BatchKind::Array,
        }
    }

    pub fn num_rows(&self) -> usize {
        match &self.payload {
            BatchPayload::Table(table) => table.rows,
            BatchPayload::External(payload) => payload.len(),
        }
    }

    /// Estimates this batch's in-memory cost in bytes before it enters a
    /// stream queue.
    ///
    /// Table batches are charged the Arrow memory size of their visible
    /// slices; external batches are charged the payload-provided exact or
    /// conservative estimate (spec S10.2). The result is a logical queue
    /// charge, not a process RSS measurement.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when a table batch cannot
    /// be measured or the summed size overflows `usize`.
    pub fn estimated_bytes(&self) -> Result<usize> {
        match &self.payload {
            BatchPayload::Table(table) => table.estimated_bytes(),
            BatchPayload::External(payload) => Ok(payload.estimated_bytes()),
        }
    }

    pub fn metadata(&self) -> &BatchMetadata {
        &self.metadata
    }

    /// Returns this batch's Arrow table payload.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when this is an array batch.
    pub fn table_payload(&self) -> Result<&TableBatch> {
        match &self.payload {
            BatchPayload::Table(table) => Ok(table),
            BatchPayload::External(_) => Err(CalcFlowError::InvalidArgument {
                field: "batch".into(),
                message: "expected table batch".into(),
            }),
        }
    }

    /// Returns this batch's externally owned payload.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when this is a table batch.
    pub fn external_payload(&self) -> Result<&Arc<dyn ExternalPayload>> {
        match &self.payload {
            BatchPayload::External(payload) => Ok(payload),
            BatchPayload::Table(_) => Err(CalcFlowError::InvalidArgument {
                field: "batch".into(),
                message: "expected array batch".into(),
            }),
        }
    }

    /// Returns the engine-owned latched static array payload, when this batch
    /// carries one (SCE-11).
    #[must_use]
    pub(crate) fn latched_array_payload(&self) -> Option<&LatchedArrayPayload> {
        self.external_payload()
            .ok()?
            .as_any()
            .downcast_ref::<LatchedArrayPayload>()
    }

    #[must_use]
    pub fn with_metadata(&self, metadata: BatchMetadata) -> Self {
        Self {
            payload: self.payload.clone(),
            metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invalid_argument_field(result: Result<Batch>) -> String {
        match result.unwrap_err() {
            CalcFlowError::InvalidArgument { field, .. } => field,
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn checked_accumulate_sums_within_usize() {
        assert_eq!(checked_accumulate(2, 3, "batch").unwrap(), 5);
        assert_eq!(checked_accumulate(0, 0, "batch").unwrap(), 0);
    }

    #[test]
    fn checked_accumulate_rejects_usize_overflow_with_a_typed_error() {
        assert!(matches!(
            checked_accumulate(usize::MAX, 1, "batch"),
            Err(CalcFlowError::InvalidArgument { ref field, .. }) if field == "batch"
        ));
    }

    #[test]
    fn latched_arrays_reject_invalid_descriptors_shapes_and_counts() {
        assert_eq!(
            invalid_argument_field(Batch::static_array_float(
                "",
                "float64",
                vec![1],
                None,
                vec![1.0],
            )),
            "static_inputs.backend"
        );
        assert_eq!(
            invalid_argument_field(Batch::static_array_int(
                "numpy",
                "float64",
                vec![1],
                None,
                vec![1],
            )),
            "static_inputs.dtype"
        );
        assert_eq!(
            invalid_argument_field(Batch::static_array_bool(
                "numpy",
                vec![u64::MAX, 2],
                None,
                vec![],
            )),
            "static_inputs.shape"
        );
        assert_eq!(
            invalid_argument_field(Batch::static_array_uint(
                "numpy",
                "uint8",
                vec![2],
                Some(vec![false]),
                vec![1, 2],
            )),
            "static_inputs.shape"
        );
        assert_eq!(
            invalid_argument_field(Batch::static_array_uint(
                "numpy",
                "uint8",
                vec![2],
                Some(vec![false, true]),
                vec![1, 2],
            )),
            "static_inputs.shape"
        );
        assert_eq!(
            invalid_argument_field(Batch::static_array_uint(
                "numpy",
                "uint8",
                vec![1],
                None,
                vec![u64::from(u8::MAX) + 1],
            )),
            "static_inputs.dtype"
        );
    }

    #[test]
    fn latched_signed_array_cells_preserve_each_declared_width() {
        let cases = [
            ("int8", i64::from(i8::MIN), i8::MIN.to_be_bytes().to_vec()),
            (
                "int16",
                i64::from(i16::MIN),
                i16::MIN.to_be_bytes().to_vec(),
            ),
            (
                "int32",
                i64::from(i32::MIN),
                i32::MIN.to_be_bytes().to_vec(),
            ),
            ("int64", i64::MIN, i64::MIN.to_be_bytes().to_vec()),
        ];

        for (dtype, value, expected) in cases {
            let batch =
                Batch::static_array_int("numpy", dtype, vec![1], None, vec![value]).unwrap();
            let payload = batch.latched_array_payload().unwrap();

            assert_eq!(payload.dtype(), dtype);
            assert_eq!(payload.cell(0).as_deref(), Some(expected.as_slice()));
            assert_eq!(payload.cell(1), None);
        }
    }

    #[test]
    fn latched_unsigned_array_cells_preserve_each_declared_width() {
        let cases = [
            ("uint8", u64::from(u8::MAX), u8::MAX.to_be_bytes().to_vec()),
            (
                "uint16",
                u64::from(u16::MAX),
                u16::MAX.to_be_bytes().to_vec(),
            ),
            (
                "uint32",
                u64::from(u32::MAX),
                u32::MAX.to_be_bytes().to_vec(),
            ),
            ("uint64", u64::MAX, u64::MAX.to_be_bytes().to_vec()),
        ];

        for (dtype, value, expected) in cases {
            let batch =
                Batch::static_array_uint("numpy", dtype, vec![1], None, vec![value]).unwrap();
            let payload = batch.latched_array_payload().unwrap();

            assert_eq!(payload.dtype(), dtype);
            assert_eq!(payload.cell(0).as_deref(), Some(expected.as_slice()));
            assert_eq!(payload.cell(1), None);
        }
    }

    #[test]
    fn latched_bool_and_float64_cells_preserve_canonical_values_and_nulls() {
        let boolean =
            Batch::static_array_bool("numpy", vec![2], Some(vec![false, true]), vec![true])
                .unwrap();
        let boolean_payload = boolean.latched_array_payload().unwrap();
        assert_eq!(boolean_payload.element_count(), 2);
        assert!(!boolean_payload.is_null(0));
        assert!(boolean_payload.is_null(1));
        assert_eq!(boolean_payload.cell(0), Some(vec![1]));

        let nan = f64::from_bits(0xfff8_0000_0000_0001);
        let float =
            Batch::static_array_float("numpy", "float64", vec![1], None, vec![nan]).unwrap();
        assert_eq!(
            float.latched_array_payload().unwrap().cell(0),
            Some(0x7ff8_0000_0000_0000u64.to_be_bytes().to_vec())
        );
    }

    #[test]
    fn latched_arrays_report_owned_lengths_backends_and_byte_estimates() {
        let cases = [
            (
                Batch::static_array_bool("numpy", vec![1], None, vec![true]).unwrap(),
                size_of::<bool>() + 5 + 4 + size_of::<u64>(),
            ),
            (
                Batch::static_array_int("numpy", "int8", vec![1], None, vec![1]).unwrap(),
                size_of::<i64>() + 5 + 4 + size_of::<u64>(),
            ),
            (
                Batch::static_array_uint("numpy", "uint8", vec![1], None, vec![1]).unwrap(),
                size_of::<u64>() + 5 + 5 + size_of::<u64>(),
            ),
            (
                Batch::static_array_float(
                    "numpy",
                    "float64",
                    vec![1],
                    Some(vec![false]),
                    vec![1.0],
                )
                .unwrap(),
                size_of::<f64>() + 5 + 7 + size_of::<u64>() + size_of::<bool>(),
            ),
        ];

        for (batch, expected_bytes) in cases {
            assert_eq!(batch.num_rows(), 1);
            assert_eq!(batch.external_payload().unwrap().backend(), "numpy");
            assert_eq!(batch.estimated_bytes().unwrap(), expected_bytes);
        }
    }
}
