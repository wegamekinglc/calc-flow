use std::{collections::HashSet, fmt, sync::Arc};

use datafusion::arrow::record_batch::RecordBatch;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    intern,
    prelude::*,
    types::{PyAny, PyDict, PyList, PyString, PyTuple},
};
use pyo3_arrow::PyTable;

const MAX_METADATA_DEPTH: usize = calc_flow::MAX_JSON_DEPTH;
const METADATA_TYPE_MESSAGE: &str = "metadata must be a JSON-compatible mapping";

pub(crate) struct PythonPayload {
    #[allow(
        dead_code,
        reason = "the stored reference owns the Python payload for the full Batch lifetime"
    )]
    pub(crate) object: Py<PyAny>,
    backend: String,
    len: usize,
}

impl fmt::Debug for PythonPayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PythonPayload")
            .field("backend", &self.backend)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl calc_flow::ExternalPayload for PythonPayload {
    fn backend(&self) -> &str {
        &self.backend
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[pyclass(name = "Batch", frozen, module = "calc_flow._native")]
pub(crate) struct PyBatch {
    pub(crate) inner: calc_flow::Batch,
}

#[pymethods]
impl PyBatch {
    #[staticmethod]
    #[pyo3(signature = (table, metadata=None))]
    fn from_pyarrow(
        py: Python<'_>,
        table: &Bound<'_, PyAny>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let table = table.extract::<PyTable>().map_err(|error| {
            PyTypeError::new_err(format!(
                "table must implement the Arrow C stream interface: {error}"
            ))
        })?;
        let (mut batches, schema) = table.into_inner();
        if batches.is_empty() {
            batches.push(RecordBatch::new_empty(schema));
        }
        let metadata = metadata_from_python(py, metadata)?;
        let inner = calc_flow::Batch::table(batches, metadata).map_err(crate::error::to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn _from_external(
        object: Py<PyAny>,
        backend: String,
        len: usize,
        metadata: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let metadata = metadata_from_python(metadata.py(), Some(metadata))?;
        let payload = PythonPayload {
            object,
            backend,
            len,
        };
        let inner = calc_flow::Batch::external(Arc::new(payload), metadata)
            .map_err(crate::error::to_py_err)?;
        Ok(Self { inner })
    }

    fn to_pyarrow<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let table = self
            .inner
            .table_payload()
            .map_err(|_| PyTypeError::new_err("array batches do not contain a PyArrow table"))?;
        PyTable::try_new(table.batches().to_vec(), table.schema().clone())?.into_pyarrow(py)
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind() {
            calc_flow::BatchKind::Table => "table",
            calc_flow::BatchKind::Array => "array",
        }
    }

    #[getter]
    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        metadata_to_python(py, self.inner.metadata())
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBatch>()
}

pub(crate) fn metadata_from_python(
    py: Python<'_>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<calc_flow::BatchMetadata> {
    let Some(value) = value else {
        return Ok(calc_flow::BatchMetadata::default());
    };

    let mapping_type = py
        .import(intern!(py, "collections.abc"))?
        .getattr(intern!(py, "Mapping"))?;
    if !value.is_instance(&mapping_type)? {
        return Err(PyTypeError::new_err(METADATA_TYPE_MESSAGE));
    }

    let copied = py.get_type::<PyDict>().call1((value,))?;
    validate_json_containers(&copied, 0, &mut HashSet::new())?;

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "allow_nan"), false)?;
    let encoded: String = py
        .import(intern!(py, "json"))?
        .getattr(intern!(py, "dumps"))?
        .call((copied,), Some(&kwargs))?
        .extract()?;
    let attributes = serde_json::from_str(&encoded)
        .map_err(|error| PyTypeError::new_err(format!("{METADATA_TYPE_MESSAGE}: {error}")))?;
    calc_flow::BatchMetadata::new("", 0, attributes).map_err(crate::error::to_py_err)
}

pub(crate) fn metadata_to_python<'py>(
    py: Python<'py>,
    metadata: &calc_flow::BatchMetadata,
) -> PyResult<Bound<'py, PyAny>> {
    let encoded = serde_json::to_string(metadata.attributes())
        .map_err(|error| PyTypeError::new_err(error.to_string()))?;
    py.import(intern!(py, "json"))?
        .getattr(intern!(py, "loads"))?
        .call1((encoded,))
}

fn validate_json_containers(
    value: &Bound<'_, PyAny>,
    depth: usize,
    ancestors: &mut HashSet<usize>,
) -> PyResult<()> {
    if depth > MAX_METADATA_DEPTH {
        return Err(PyTypeError::new_err(format!(
            "{METADATA_TYPE_MESSAGE}: nesting exceeds {MAX_METADATA_DEPTH} levels"
        )));
    }

    if let Ok(mapping) = value.cast::<PyDict>() {
        validate_container(value, ancestors, |ancestors| {
            for (key, nested) in mapping {
                if !key.is_instance_of::<PyString>() {
                    return Err(PyTypeError::new_err(
                        "metadata JSON object keys must be strings",
                    ));
                }
                validate_json_containers(&nested, depth + 1, ancestors)?;
            }
            Ok(())
        })
    } else if let Ok(items) = value.cast::<PyList>() {
        validate_container(value, ancestors, |ancestors| {
            for nested in items {
                validate_json_containers(&nested, depth + 1, ancestors)?;
            }
            Ok(())
        })
    } else if let Ok(items) = value.cast::<PyTuple>() {
        validate_container(value, ancestors, |ancestors| {
            for nested in items {
                validate_json_containers(&nested, depth + 1, ancestors)?;
            }
            Ok(())
        })
    } else {
        Ok(())
    }
}

fn validate_container(
    value: &Bound<'_, PyAny>,
    ancestors: &mut HashSet<usize>,
    validate_children: impl FnOnce(&mut HashSet<usize>) -> PyResult<()>,
) -> PyResult<()> {
    let identity = value.as_ptr() as usize;
    if !ancestors.insert(identity) {
        return Err(PyValueError::new_err(
            "metadata must be an acyclic JSON-compatible mapping",
        ));
    }
    let result = validate_children(ancestors);
    ancestors.remove(&identity);
    result
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::{
        array::{Array, ArrayRef, Int64Array},
        datatypes::{DataType, Field, Schema},
    };
    use pyo3::{
        exceptions::{PyTypeError, PyValueError},
        types::{PyDict, PyList},
    };

    use super::*;

    fn table() -> (PyTable, ArrayRef) {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(schema.clone(), vec![values.clone()]).unwrap();
        (PyTable::try_new(vec![batch], schema).unwrap(), values)
    }

    #[test]
    fn table_batch_preserves_buffers_and_metadata() {
        Python::initialize();
        Python::attach(|py| {
            let metadata = PyDict::new(py);
            metadata
                .set_item("nested", PyDict::new(py))
                .expect("metadata should be valid");
            let (table, values) = table();
            let table = table.into_pyarrow(py).unwrap();

            let batch = PyBatch::from_pyarrow(py, &table, Some(metadata.as_any())).unwrap();

            assert_eq!(batch.kind(), "table");
            assert_eq!(batch.num_rows(), 3);
            let inner_values = batch.inner.table_payload().unwrap().batches()[0].column(0);
            assert_eq!(
                inner_values.to_data().buffers()[0].as_ptr(),
                values.to_data().buffers()[0].as_ptr()
            );
            let returned = batch.metadata(py).unwrap();
            assert!(
                returned
                    .cast::<PyDict>()
                    .unwrap()
                    .contains("nested")
                    .unwrap()
            );
            assert_eq!(batch.to_pyarrow(py).unwrap().len().unwrap(), 3);
        });
    }

    #[test]
    fn empty_table_gets_a_schema_preserving_zero_row_batch() {
        Python::initialize();
        Python::attach(|py| {
            let schema = Arc::new(Schema::new(vec![Field::new(
                "value",
                DataType::Int64,
                false,
            )]));
            let batch = PyBatch::from_pyarrow(
                py,
                &PyTable::try_new(Vec::new(), schema.clone())
                    .unwrap()
                    .into_pyarrow(py)
                    .unwrap(),
                None,
            )
            .unwrap();

            let table = batch.inner.table_payload().unwrap();
            assert_eq!(table.schema(), &schema);
            assert_eq!(table.batches().len(), 1);
            assert_eq!(table.batches()[0].num_rows(), 0);
        });
    }

    #[test]
    fn external_batch_owns_and_describes_python_payload() {
        Python::initialize();
        Python::attach(|py| {
            let object = PyList::new(py, [1, 2]).unwrap().unbind().into_any();
            let metadata = PyDict::new(py);
            let batch =
                PyBatch::_from_external(object, "numpy".into(), 2, metadata.as_any()).unwrap();

            assert_eq!(batch.kind(), "array");
            assert_eq!(batch.num_rows(), 2);
            let payload = batch.inner.external_payload().unwrap();
            let payload = payload.as_any().downcast_ref::<PythonPayload>().unwrap();
            assert_eq!(payload.backend, "numpy");
            assert_eq!(payload.len, 2);
            assert_eq!(payload.object.bind(py).len().unwrap(), 2);
            assert!(format!("{payload:?}").contains("numpy"));
            let error = batch.to_pyarrow(py).unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn metadata_conversion_validates_shape_keys_cycles_and_depth() {
        Python::initialize();
        Python::attach(|py| {
            assert_eq!(
                metadata_from_python(py, None).unwrap(),
                calc_flow::BatchMetadata::default()
            );

            let not_mapping = PyList::empty(py);
            let error = metadata_from_python(py, Some(not_mapping.as_any())).unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));

            let invalid_key = PyDict::new(py);
            invalid_key.set_item(1, "value").unwrap();
            let error = metadata_from_python(py, Some(invalid_key.as_any())).unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));

            let circular = PyDict::new(py);
            circular.set_item("self", &circular).unwrap();
            let error = metadata_from_python(py, Some(circular.as_any())).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));

            let root = PyList::empty(py);
            let mut current = root.clone().into_any();
            for _ in 0..=MAX_METADATA_DEPTH {
                let parent = PyList::empty(py);
                parent.append(current).unwrap();
                current = parent.into_any();
            }
            let deep = PyDict::new(py);
            deep.set_item("value", current).unwrap();
            let error = metadata_from_python(py, Some(deep.as_any())).unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn invalid_external_backend_uses_core_error_mapping() {
        Python::initialize();
        Python::attach(|py| {
            let object = PyList::empty(py).unbind().into_any();
            let metadata = PyDict::new(py);
            let error = match PyBatch::_from_external(object, String::new(), 0, metadata.as_any()) {
                Ok(_) => panic!("an empty backend must be rejected"),
                Err(error) => error,
            };
            assert!(error.is_instance_of::<crate::error::ConfigError>(py));
        });
    }
}
