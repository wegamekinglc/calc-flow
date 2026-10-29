//! Private Arrow schema and buffer bridges without Python package dependencies.

use std::{collections::BTreeMap, sync::Arc};

use datafusion::arrow::{
    array::{
        Array, ArrayRef, Int64Array,
        types::{
            Float16Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
            UInt8Type, UInt16Type, UInt32Type, UInt64Type,
        },
    },
    compute::cast,
    datatypes::{DataType, Schema, SchemaRef},
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyCapsule},
};
use pyo3_arrow::{PySchema, ffi::to_schema_pycapsule};

#[pyclass(name = "_ArrowSchema", frozen, module = "calc_flow._native")]
pub(crate) struct ArrowSchema(pub(crate) SchemaRef);

pub(crate) fn guard_input(value: &Bound<'_, PyAny>) -> PyResult<()> {
    for base in value.get_type().getattr("__mro__")?.try_iter()? {
        let module = base?.getattr("__module__")?.extract::<String>()?;
        if module == "pyarrow" || module.starts_with("pyarrow.") {
            require_pyarrow(value.py())?;
            break;
        }
    }
    Ok(())
}

pub(crate) fn require_pyarrow(py: Python<'_>) -> PyResult<()> {
    use pyo3::exceptions::PyImportError;
    let module = py.import("pyarrow").map_err(|_| {
        PyImportError::new_err("PyArrow output and Python UDFs require calc-flow-python[pyarrow]")
    })?;
    let version = module.getattr("__version__")?.extract::<String>()?;
    let major = version
        .split('.')
        .next()
        .and_then(|v| v.parse::<u32>().ok());
    let expected = if py.version_info().minor < 10 { 21 } else { 24 };
    if major != Some(expected) {
        return Err(PyImportError::new_err(
            "unsupported PyArrow version; install calc-flow-python[pyarrow] (PyArrow 25 capsule lifetimes are not supported)",
        ));
    }
    Ok(())
}

pub(crate) fn schema(value: &Bound<'_, PyAny>) -> PyResult<SchemaRef> {
    guard_input(value)?;
    Ok(value.extract::<PySchema>()?.into_inner())
}

#[pymethods]
impl ArrowSchema {
    #[new]
    fn new(fields: Vec<(String, String, bool)>) -> PyResult<Self> {
        let fields = fields
            .into_iter()
            .map(|(name, data_type, nullable)| {
                calc_flow::ArrowFieldSpec {
                    name,
                    data_type,
                    nullable,
                }
                .to_arrow()
                .map_err(crate::error::to_py_err)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self(Arc::new(Schema::new(fields))))
    }

    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        to_schema_pycapsule(py, self.0.as_ref()).map_err(Into::into)
    }

    fn equals(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let other = schema(other)?;
        Ok(self.0.fields().len() == other.fields().len()
            && self.0.fields().iter().zip(other.fields()).all(|(a, b)| {
                a.name() == b.name()
                    && a.data_type() == b.data_type()
                    && a.is_nullable() == b.is_nullable()
            }))
    }
}

#[pyfunction]
fn _schema_fields(value: &Bound<'_, PyAny>) -> PyResult<Vec<(String, String, bool)>> {
    schema(value)?
        .fields()
        .iter()
        .map(|f| {
            let dtype = calc_flow::ArrowFieldSpec::from_arrow(f)
                .map_or_else(|_| f.data_type().to_string(), |s| s.data_type);
            Ok((f.name().clone(), dtype, f.is_nullable()))
        })
        .collect()
}

#[pyfunction]
fn _schema_key(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let schema = schema(value)?;
    // Schema cache identity includes metadata, with deterministic map ordering.
    let fields = schema
        .fields()
        .iter()
        .map(|f| {
            (
                f.name(),
                f.data_type(),
                f.is_nullable(),
                f.metadata().iter().collect::<BTreeMap<_, _>>(),
            )
        })
        .collect::<Vec<_>>();
    Ok(format!(
        "{fields:?}:{:?}",
        schema.metadata().iter().collect::<BTreeMap<_, _>>()
    ))
}

pub(crate) fn validate_time(
    batch: &calc_flow::Batch,
    column: &str,
    last: Option<i64>,
) -> PyResult<Option<i64>> {
    let table = batch.table_payload().map_err(crate::error::to_py_err)?;
    let mut previous = last;
    for record in table.batches() {
        let values = record
            .column_by_name(column)
            .ok_or_else(|| PyValueError::new_err("missing event time column"))?;
        if values.null_count() != 0 {
            return Err(PyValueError::new_err("expected non-null event time"));
        }
        let values =
            cast(values, &DataType::Int64).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let values = values
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| PyTypeError::new_err("expected timestamp column"))?;
        for &current in values.values() {
            if previous.is_some_and(|p| current < p) {
                return Err(PyValueError::new_err(
                    "expected nondecreasing event time; select an explicit watermark policy for out-of-order input",
                ));
            }
            previous = Some(current);
        }
    }
    Ok(previous)
}

pub(crate) fn numpy_array(value: &Bound<'_, PyAny>) -> PyResult<ArrayRef> {
    use datafusion::arrow::array::PrimitiveArray;
    use numpy::PyArrayMethods;
    macro_rules! convert {
        ($native:ty, $arrow:ty) => {{
            let array = value.cast::<numpy::PyArray1<$native>>()?.try_readonly()?;
            Ok(Arc::new(PrimitiveArray::<$arrow>::from_iter_values(
                array.as_array().iter().copied(),
            )) as ArrayRef)
        }};
    }
    match value
        .getattr("dtype")?
        .getattr("name")?
        .extract::<String>()?
        .as_str()
    {
        "float32" => convert!(f32, Float32Type),
        "float64" => convert!(f64, Float64Type),
        "int8" => convert!(i8, Int8Type),
        "int16" => convert!(i16, Int16Type),
        "int32" => convert!(i32, Int32Type),
        "int64" => convert!(i64, Int64Type),
        "uint8" => convert!(u8, UInt8Type),
        "uint16" => convert!(u16, UInt16Type),
        "uint32" => convert!(u32, UInt32Type),
        "uint64" => convert!(u64, UInt64Type),
        "bool" => {
            let array = value.cast::<numpy::PyArray1<bool>>()?.try_readonly()?;
            Ok(Arc::new(
                array
                    .as_array()
                    .iter()
                    .copied()
                    .map(Some)
                    .collect::<datafusion::arrow::array::BooleanArray>(),
            ))
        }
        _ => Err(PyTypeError::new_err("unsupported array-to-table dtype")),
    }
}

#[pyfunction]
fn _batch_from_ipc(data: &Bound<'_, PyBytes>) -> PyResult<crate::batch::PyBatch> {
    use datafusion::arrow::ipc::reader::StreamReader;
    let reader = StreamReader::try_new(std::io::Cursor::new(data.as_bytes()), None)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let schema = reader.schema();
    let mut records = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if records.is_empty() {
        records.push(datafusion::arrow::record_batch::RecordBatch::new_empty(
            schema,
        ));
    }
    let batch = calc_flow::Batch::table(records, calc_flow::BatchMetadata::default())
        .map_err(crate::error::to_py_err)?;
    Ok(crate::batch::PyBatch::from_inner(batch))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ArrowSchema>()?;
    module.add(
        "_ArrowValidationError",
        module.py().get_type::<ArrowValidationError>(),
    )?;
    module.add_function(wrap_pyfunction!(_stream_input_batch, module)?)?;
    module.add_function(wrap_pyfunction!(_schema_fields, module)?)?;
    module.add_function(wrap_pyfunction!(_schema_key, module)?)?;
    module.add_function(wrap_pyfunction!(_batch_from_ipc, module)?)?;
    Ok(())
}

pub(crate) fn column_index(schema: &Schema, name: &str) -> PyResult<usize> {
    let indices = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| f.name() == name)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    match indices.as_slice() {
        [index] => Ok(*index),
        [] => Err(PyValueError::new_err(format!(
            "invalid table_matmul columns: selected column {name:?} is missing"
        ))),
        _ => Err(PyValueError::new_err(format!(
            "invalid table_matmul columns: selected column {name:?} is ambiguous"
        ))),
    }
}

pub(crate) fn table_matrix(
    py: Python<'_>,
    batch: &calc_flow::Batch,
    columns: Vec<String>,
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    use datafusion::arrow::array::PrimitiveArray;
    use numpy::PyArrayMethods;
    let table = batch.table_payload().map_err(crate::error::to_py_err)?;
    let column_count = columns.len();
    let indices = columns
        .into_iter()
        .map(|name| column_index(table.schema(), &name))
        .collect::<PyResult<Vec<_>>>()?;
    validate_matrix_casts(py, table.schema(), &indices, dtype)?;
    macro_rules! matrix {
        ($dst:ty) => {{
            let object = allocate_matrix(py, batch.num_rows(), column_count, dtype)?;
            {
                let mut output = object
                    .bind(py)
                    .cast::<numpy::PyArray2<$dst>>()?
                    .try_readwrite()?;
                let mut output = output.as_array_mut();
                for (column, &index) in indices.iter().enumerate() {
                    let mut offset = 0;
                    for record in table.batches() {
                        let values = record.column(index);
                        if values.null_count() != 0 {
                            return Err(PyValueError::new_err(
                                "invalid table_matmul columns: contains nulls",
                            ));
                        }
                        macro_rules! copy {
                            ($src:ty) => {{
                                let values = values
                                    .as_any()
                                    .downcast_ref::<PrimitiveArray<$src>>()
                                    .ok_or_else(|| {
                                        PyTypeError::new_err("invalid numeric column")
                                    })?;
                                for (row, &value) in values.values().iter().enumerate() {
                                    #[allow(
                                        clippy::cast_possible_truncation,
                                        clippy::cast_possible_wrap,
                                        clippy::cast_precision_loss,
                                        clippy::cast_sign_loss,
                                        clippy::cast_lossless,
                                        reason = "NumPy safe-cast preflight admits this conversion; preserve np.copyto semantics without intermediate buffers"
                                    )]
                                    let converted = value as $dst;
                                    output[(offset + row, column)] = converted;
                                }
                            }};
                        }
                        match values.data_type() {
                            DataType::Float16 => {
                                let values = values.as_any().downcast_ref::<PrimitiveArray<Float16Type>>()
                                    .ok_or_else(|| PyTypeError::new_err("invalid float16 column"))?;
                                for (row, &value) in values.values().iter().enumerate() {
                                    #[allow(
                                        clippy::cast_possible_truncation,
                                        clippy::cast_precision_loss,
                                        clippy::cast_sign_loss,
                                        reason = "safe-cast preflight permits float16 only into an admitted floating dtype"
                                    )]
                                    let converted = value.to_f64() as $dst;
                                    output[(offset + row, column)] = converted;
                                }
                            }
                            DataType::Float32 => copy!(Float32Type),
                            DataType::Float64 => copy!(Float64Type),
                            DataType::Int8 => copy!(Int8Type),
                            DataType::Int16 => copy!(Int16Type),
                            DataType::Int32 => copy!(Int32Type),
                            DataType::Int64 => copy!(Int64Type),
                            DataType::UInt8 => copy!(UInt8Type),
                            DataType::UInt16 => copy!(UInt16Type),
                            DataType::UInt32 => copy!(UInt32Type),
                            DataType::UInt64 => copy!(UInt64Type),
                            _ => {
                                return Err(PyTypeError::new_err(
                                    "invalid table_matmul columns: unsupported Arrow dtype",
                                ));
                            }
                        }
                        offset += record.num_rows();
                    }
                }
            }
            Ok(object)
        }};
    }
    match dtype {
        "float32" => matrix!(f32),
        "float64" => matrix!(f64),
        "int8" => matrix!(i8),
        "int16" => matrix!(i16),
        "int32" => matrix!(i32),
        "int64" => matrix!(i64),
        "uint8" => matrix!(u8),
        "uint16" => matrix!(u16),
        "uint32" => matrix!(u32),
        "uint64" => matrix!(u64),
        _ => Err(PyTypeError::new_err("invalid table_matmul dtype")),
    }
}

pub(crate) fn export_stream<'py>(
    py: Python<'py>,
    batch: &calc_flow::Batch,
    requested: Option<Bound<'py, PyCapsule>>,
) -> PyResult<Bound<'py, PyCapsule>> {
    use datafusion::arrow::{array::StructArray, datatypes::Field};
    use pyo3_arrow::ffi::{ArrayIterator, to_stream_pycapsule};
    let table = batch
        .table_payload()
        .map_err(|_| PyTypeError::new_err("array Batch cannot export an Arrow table"))?;
    let field = Arc::new(
        Field::new_struct("", table.schema().fields().clone(), false)
            .with_metadata(table.schema().metadata().clone()),
    );
    let records = table
        .batches()
        .to_vec()
        .into_iter()
        .map(|b| Ok(Arc::new(StructArray::from(b)) as ArrayRef));
    to_stream_pycapsule(py, Box::new(ArrayIterator::new(records, field)), requested)
        .map_err(Into::into)
}

fn validate_matrix_casts(
    py: Python<'_>,
    schema: &Schema,
    indices: &[usize],
    dtype: &str,
) -> PyResult<()> {
    let np = py.import("numpy")?;
    for &index in indices {
        let field = &schema.fields()[index];
        let source = if field.data_type() == &DataType::Float16 {
            "float16".into()
        } else {
            calc_flow::ArrowFieldSpec::from_arrow(field)
                .map_err(crate::error::to_py_err)?
                .data_type
        };
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("casting", "safe")?;
        if !np
            .getattr("can_cast")?
            .call((source, dtype), Some(&kwargs))?
            .extract::<bool>()?
        {
            return Err(PyTypeError::new_err(
                "invalid table_matmul dtype: unsafe conversion",
            ));
        }
    }
    Ok(())
}

pub(crate) fn import_table(
    value: &Bound<'_, PyAny>,
    ordered_column: Option<&str>,
) -> PyResult<Vec<datafusion::arrow::record_batch::RecordBatch>> {
    use datafusion::arrow::{
        array::StructArray,
        record_batch::{RecordBatch, RecordBatchOptions},
    };
    guard_input(value)?;
    let (arrays, field) = if value.hasattr("__arrow_c_stream__")? {
        value.extract::<pyo3_arrow::PyChunkedArray>()?.into_inner()
    } else if value.hasattr("__arrow_c_array__")? {
        let (array, field) = value.extract::<pyo3_arrow::PyArray>()?.into_inner();
        (vec![array], field)
    } else {
        return Err(PyTypeError::new_err(
            "table must implement the Arrow C stream or struct array interface",
        ));
    };
    let DataType::Struct(fields) = field.data_type() else {
        return Err(PyTypeError::new_err(
            "Arrow record batch requires a struct root",
        ));
    };
    let schema = Arc::new(Schema::new_with_metadata(
        fields.clone(),
        field.metadata().clone(),
    ));
    if arrays.is_empty() {
        return Ok(vec![RecordBatch::new_empty(schema)]);
    }
    arrays
        .iter()
        .map(|values| {
            let values = values
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| PyTypeError::new_err("Arrow record batch requires a struct root"))?;
            if values.null_count() != 0 {
                return Err(PyTypeError::new_err(
                    "Arrow record batch requires a non-null struct root",
                ));
            }
            if let Some(column) = ordered_column
                && values
                    .column_by_name(column)
                    .is_some_and(|v| v.null_count() != 0)
            {
                return Err(ArrowValidationError::new_err(
                    ".event_time: expected non-null event time",
                ));
            }
            RecordBatch::try_new_with_options(
                schema.clone(),
                values.columns().to_vec(),
                &RecordBatchOptions::new().with_row_count(Some(values.len())),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))
        })
        .collect()
}

pyo3::create_exception!(_native, ArrowValidationError, PyValueError);

#[pyfunction]
fn _stream_input_batch(
    value: &Bound<'_, PyAny>,
    ordered_column: Option<&str>,
) -> PyResult<crate::batch::PyBatch> {
    let batch = calc_flow::Batch::table(
        import_table(value, ordered_column)?,
        calc_flow::BatchMetadata::default(),
    )
    .map_err(crate::error::to_py_err)?;
    Ok(crate::batch::PyBatch::from_inner(batch))
}

fn allocate_matrix(
    py: Python<'_>,
    rows: usize,
    columns: usize,
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    let (object, token) = py
        .get_type::<crate::batch::PyBatch>()
        .call_method1("_new_owned_numpy", (vec![rows, columns], dtype))?
        .extract::<(Py<PyAny>, Py<crate::batch::OwnedArrayToken>)>()?;
    token.borrow(py).consume(object.bind(py))?;
    Ok(object)
}
