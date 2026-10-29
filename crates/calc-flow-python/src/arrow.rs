//! Private Arrow schema and buffer bridges without Python package dependencies.

use std::{collections::BTreeMap, sync::Arc};

use datafusion::arrow::{
    array::{Array, ArrayRef, Int64Array, StructArray},
    compute::cast,
    datatypes::{DataType, FieldRef, Schema, SchemaRef},
    record_batch::{RecordBatch, RecordBatchOptions},
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyCapsule},
};
use pyo3_arrow::{PySchema, ffi::to_schema_pycapsule};

mod buffers;
pub(crate) use buffers::{numpy_array, table_matrix};

#[pyclass(name = "_ArrowSchema", frozen, module = "calc_flow._native")]
pub(crate) struct ArrowSchema(pub(crate) SchemaRef);

pub(crate) fn guard_input(value: &Bound<'_, PyAny>) -> PyResult<()> {
    for base in value.get_type().getattr("__mro__")?.try_iter()? {
        if is_pyarrow_type(&base?)? {
            require_pyarrow(value.py())?;
            break;
        }
    }
    Ok(())
}

fn is_pyarrow_type(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let module = value.getattr("__module__")?.extract::<String>()?;
    Ok(module == "pyarrow" || module.starts_with("pyarrow."))
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
        let values = event_time_values(record, column)?;
        let values = values
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| PyTypeError::new_err("expected timestamp column"))?;
        previous = ordered_event_time(values.values(), previous)?;
    }
    Ok(previous)
}

fn event_time_values(record: &RecordBatch, column: &str) -> PyResult<ArrayRef> {
    let values = record
        .column_by_name(column)
        .ok_or_else(|| PyValueError::new_err("missing event time column"))?;
    if values.null_count() != 0 {
        return Err(PyValueError::new_err("expected non-null event time"));
    }
    cast(values, &DataType::Int64).map_err(|error| PyValueError::new_err(error.to_string()))
}

fn ordered_event_time(values: &[i64], last: Option<i64>) -> PyResult<Option<i64>> {
    let mut previous = last;
    for &current in values {
        if previous.is_some_and(|p| current < p) {
            return Err(PyValueError::new_err(
                "expected nondecreasing event time; select an explicit watermark policy for out-of-order input",
            ));
        }
        previous = Some(current);
    }
    Ok(previous)
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
        records.push(RecordBatch::new_empty(schema));
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
    register_stream(module)?;
    register_schema(module)
}

fn register_stream(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_stream_input_batch, module)?)?;
    module.add_function(wrap_pyfunction!(_batch_from_ipc, module)?)?;
    Ok(())
}

fn register_schema(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_schema_fields, module)?)?;
    module.add_function(wrap_pyfunction!(_schema_key, module)?)?;
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

pub(crate) fn import_table(
    value: &Bound<'_, PyAny>,
    ordered_column: Option<&str>,
) -> PyResult<Vec<RecordBatch>> {
    guard_input(value)?;
    let (arrays, field) = import_arrays(value)?;
    let schema = table_schema(&field)?;
    if arrays.is_empty() {
        return Ok(vec![RecordBatch::new_empty(schema)]);
    }
    arrays
        .iter()
        .map(|values| table_record(values, &schema, ordered_column))
        .collect()
}

fn import_arrays(value: &Bound<'_, PyAny>) -> PyResult<(Vec<ArrayRef>, FieldRef)> {
    if value.hasattr("__arrow_c_stream__")? {
        Ok(value.extract::<pyo3_arrow::PyChunkedArray>()?.into_inner())
    } else if value.hasattr("__arrow_c_array__")? {
        let (array, field) = value.extract::<pyo3_arrow::PyArray>()?.into_inner();
        Ok((vec![array], field))
    } else {
        Err(PyTypeError::new_err(
            "table must implement the Arrow C stream or struct array interface",
        ))
    }
}

fn table_schema(field: &datafusion::arrow::datatypes::Field) -> PyResult<SchemaRef> {
    let DataType::Struct(fields) = field.data_type() else {
        return Err(PyTypeError::new_err(
            "Arrow record batch requires a struct root",
        ));
    };
    Ok(Arc::new(Schema::new_with_metadata(
        fields.clone(),
        field.metadata().clone(),
    )))
}

fn table_record(
    values: &ArrayRef,
    schema: &SchemaRef,
    ordered_column: Option<&str>,
) -> PyResult<RecordBatch> {
    let values = values
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| PyTypeError::new_err("Arrow record batch requires a struct root"))?;
    validate_struct(values, ordered_column)?;
    RecordBatch::try_new_with_options(
        schema.clone(),
        values.columns().to_vec(),
        &RecordBatchOptions::new().with_row_count(Some(values.len())),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn validate_struct(values: &StructArray, ordered_column: Option<&str>) -> PyResult<()> {
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
    Ok(())
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
