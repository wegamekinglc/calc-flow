use std::{
    collections::HashSet,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use datafusion::arrow::record_batch::RecordBatch;
use ndarray::{ArrayD, IxDyn};
use num_complex::{Complex32, Complex64};
use numpy::{Element, IntoPyArray};
use parking_lot::RwLock;
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    intern,
    prelude::*,
    types::{PyAny, PyBool, PyDict, PyInt, PyList, PyString, PyTuple},
};
use pyo3_arrow::PyTable;

const MAX_METADATA_DEPTH: usize = calc_flow::MAX_JSON_DEPTH;
const METADATA_TYPE_MESSAGE: &str = "metadata must be a JSON-compatible mapping";
const METADATA_INTEGER_MESSAGE: &str = "metadata integers must be in the portable JSON range -9223372036854775808 to 18446744073709551615";
const CLEARED_BATCH_MESSAGE: &str = "Batch has been cleared by garbage collection";
const MAX_OWNED_NUMPY_RANK: usize = 16;
const MAX_OWNED_NUMPY_DIMENSION: usize = 1_000_000;
const MAX_OWNED_NUMPY_ELEMENTS: usize = 10_000_000;

fn validate_owned_numpy_shape(shape: &[usize]) -> Result<(), &'static str> {
    if shape.is_empty() {
        return Err("owned NumPy shape must have at least one dimension");
    }
    if shape.len() > MAX_OWNED_NUMPY_RANK {
        return Err("owned NumPy shape must have at most 16 dimensions");
    }

    let mut elements = 1_usize;
    for &dimension in shape {
        if dimension == 0 {
            return Err("owned NumPy shape dimensions must be positive");
        }
        if dimension > MAX_OWNED_NUMPY_DIMENSION {
            return Err("owned NumPy shape dimension exceeds 1000000");
        }
        if elements > MAX_OWNED_NUMPY_ELEMENTS / dimension {
            return Err("owned NumPy shape element count exceeds 10000000");
        }
        elements *= dimension;
    }
    Ok(())
}

#[pyclass(name = "_OwnedArrayToken", frozen, module = "calc_flow._native")]
struct OwnedArrayToken {
    object: RwLock<Option<Py<PyAny>>>,
    consumed: AtomicBool,
}

impl OwnedArrayToken {
    fn consume(&self, object: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut expected = self.object.write();
        let Some(expected_object) = expected.as_ref() else {
            return Err(PyValueError::new_err(
                "owned array token was already consumed",
            ));
        };
        if !expected_object.bind(object.py()).is(object) {
            return Err(PyValueError::new_err(
                "owned array token does not match the array",
            ));
        }
        self.consumed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| PyValueError::new_err("owned array token was already consumed"))?;
        expected.take();
        Ok(())
    }
}

fn owned_numpy<T>(py: Python<'_>, shape: &[usize]) -> PyResult<(Py<PyAny>, Py<OwnedArrayToken>)>
where
    T: Clone + Default + Element,
{
    let array = ArrayD::<T>::default(IxDyn(shape)).into_pyarray(py);
    let object = array.into_any().unbind();
    let token = Py::new(
        py,
        OwnedArrayToken {
            object: RwLock::new(Some(object.clone_ref(py))),
            consumed: AtomicBool::new(false),
        },
    )?;
    Ok((object, token))
}

pub(crate) struct PythonPayload {
    #[allow(
        dead_code,
        reason = "the stored reference owns the Python payload for the full Batch lifetime"
    )]
    pub(crate) object: Py<PyAny>,
    backend: String,
    len: usize,
}

impl PythonPayload {
    pub(crate) fn backend(&self) -> &str {
        &self.backend
    }
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
    inner: RwLock<Option<calc_flow::Batch>>,
}

impl PyBatch {
    pub(crate) fn from_inner(inner: calc_flow::Batch) -> Self {
        Self {
            inner: RwLock::new(Some(inner)),
        }
    }

    pub(crate) fn clone_inner(&self) -> PyResult<calc_flow::Batch> {
        self.inner
            .read()
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err(CLEARED_BATCH_MESSAGE))
    }

    pub(crate) fn from_inner_python(py: Python<'_>, inner: calc_flow::Batch) -> PyResult<Self> {
        Ok(Self::from_inner(rehome_python_payload(py, inner)?))
    }

    pub(crate) fn python_payload(&self) -> PyResult<calc_flow::Batch> {
        let batch = self.clone_inner()?;
        let payload = batch.external_payload().map_err(|_| {
            PyTypeError::new_err("table batches do not contain a Python array payload")
        })?;
        if payload.as_any().downcast_ref::<PythonPayload>().is_none() {
            return Err(PyTypeError::new_err(
                "array batch payload was not created by the Python host",
            ));
        }
        Ok(batch)
    }
}

pub(crate) fn python_payload_root(batch: &calc_flow::Batch) -> Option<&Py<PyAny>> {
    batch
        .external_payload()
        .ok()?
        .as_any()
        .downcast_ref::<PythonPayload>()
        .map(|payload| &payload.object)
}

pub(crate) fn rehome_python_payload(
    py: Python<'_>,
    batch: calc_flow::Batch,
) -> PyResult<calc_flow::Batch> {
    let Ok(payload) = batch.external_payload() else {
        return Ok(batch);
    };
    let Some(payload) = payload.as_any().downcast_ref::<PythonPayload>() else {
        return Ok(batch);
    };
    let payload = PythonPayload {
        object: payload.object.clone_ref(py),
        backend: payload.backend.clone(),
        len: payload.len,
    };
    calc_flow::Batch::external(Arc::new(payload), batch.metadata().clone())
        .map_err(crate::error::to_py_err)
}

#[pymethods]
impl PyBatch {
    #[staticmethod]
    fn _new_owned_numpy(
        py: Python<'_>,
        shape: &Bound<'_, PyAny>,
        dtype: &str,
    ) -> PyResult<(Py<PyAny>, Py<OwnedArrayToken>)> {
        let shape: Vec<usize> = shape.extract()?;
        validate_owned_numpy_shape(&shape).map_err(PyValueError::new_err)?;
        match dtype {
            "int8" => owned_numpy::<i8>(py, &shape),
            "int16" => owned_numpy::<i16>(py, &shape),
            "int32" => owned_numpy::<i32>(py, &shape),
            "int64" => owned_numpy::<i64>(py, &shape),
            "uint8" => owned_numpy::<u8>(py, &shape),
            "uint16" => owned_numpy::<u16>(py, &shape),
            "uint32" => owned_numpy::<u32>(py, &shape),
            "uint64" => owned_numpy::<u64>(py, &shape),
            "float32" => owned_numpy::<f32>(py, &shape),
            "float64" => owned_numpy::<f64>(py, &shape),
            "complex64" => owned_numpy::<Complex32>(py, &shape),
            "complex128" => owned_numpy::<Complex64>(py, &shape),
            _ => Err(PyValueError::new_err(format!(
                "owned NumPy arrays require a supported native numeric dtype; received {dtype}"
            ))),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (array, *, backend, token, metadata))]
    fn _from_owned_array(
        py: Python<'_>,
        array: Py<PyAny>,
        backend: String,
        token: Option<PyRef<'_, OwnedArrayToken>>,
        metadata: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let len = match (backend.as_str(), token) {
            ("numpy", Some(token)) => {
                token.consume(array.bind(py))?;
                let flags = array.bind(py).getattr("flags")?;
                if flags.getattr("owndata")?.extract::<bool>()? {
                    return Err(PyValueError::new_err(
                        "owned NumPy array must use binding-owned storage",
                    ));
                }
                array.bind(py).call_method1("setflags", (false,))?;
                let shape: Vec<usize> = array.bind(py).getattr("shape")?.extract()?;
                shape.first().copied().unwrap_or(1)
            }
            ("numpy", None) => {
                return Err(PyTypeError::new_err(
                    "NumPy owned arrays require an ownership token",
                ));
            }
            ("jax", None) => {
                let jax = py.import("jax")?;
                if !array.bind(py).is_instance(&jax.getattr("Array")?)? {
                    return Err(PyTypeError::new_err("JAX owned arrays require a jax.Array"));
                }
                let shape: Vec<usize> = array.bind(py).getattr("shape")?.extract()?;
                shape.first().copied().unwrap_or(1)
            }
            ("jax", Some(_)) => {
                return Err(PyTypeError::new_err(
                    "JAX owned arrays do not accept a NumPy ownership token",
                ));
            }
            _ => {
                return Err(PyValueError::new_err(
                    "owned array backend must be 'numpy' or 'jax'",
                ));
            }
        };
        let metadata = metadata_from_python(py, Some(metadata))?;
        let payload = PythonPayload {
            object: array,
            backend,
            len,
        };
        let inner = calc_flow::Batch::external(Arc::new(payload), metadata)
            .map_err(crate::error::to_py_err)?;
        Ok(Self::from_inner(inner))
    }

    #[staticmethod]
    #[pyo3(signature = (table, metadata=None))]
    fn from_pyarrow(
        py: Python<'_>,
        table: &Bound<'_, PyAny>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if !table.hasattr(intern!(py, "__arrow_c_stream__"))? {
            return Err(PyTypeError::new_err(
                "table must implement the Arrow C stream interface",
            ));
        }
        let table = table.extract::<PyTable>()?;
        let (mut batches, schema) = table.into_inner();
        if batches.is_empty() {
            batches.push(RecordBatch::new_empty(schema));
        }
        let metadata = metadata_from_python(py, metadata)?;
        let inner = calc_flow::Batch::table(batches, metadata).map_err(crate::error::to_py_err)?;
        Ok(Self::from_inner(inner))
    }

    #[staticmethod]
    #[pyo3(signature = (array, *, backend, metadata=None))]
    fn from_array(
        py: Python<'_>,
        array: &Bound<'_, PyAny>,
        backend: String,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let prepared = py
            .import(intern!(py, "calc_flow.array"))?
            .getattr(intern!(py, "_prepare_array"))?
            .call1((array, &backend))?;
        let (object, len): (Py<PyAny>, usize) = prepared.extract()?;
        let metadata = metadata_from_python(py, metadata)?;
        let payload = PythonPayload {
            object,
            backend,
            len,
        };
        let inner = calc_flow::Batch::external(Arc::new(payload), metadata)
            .map_err(crate::error::to_py_err)?;
        Ok(Self::from_inner(inner))
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
        Ok(Self::from_inner(inner))
    }

    fn to_pyarrow<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.clone_inner()?;
        let table = inner
            .table_payload()
            .map_err(|_| PyTypeError::new_err("array batches do not contain a PyArrow table"))?;
        PyTable::try_new(table.batches().to_vec(), table.schema().clone())?.into_pyarrow(py)
    }

    #[getter]
    fn array(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = self.clone_inner()?;
        let payload = inner
            .external_payload()
            .map_err(|_| PyTypeError::new_err("table batches do not contain an array"))?;
        let payload = payload
            .as_any()
            .downcast_ref::<PythonPayload>()
            .ok_or_else(|| {
                PyTypeError::new_err("array batch payload was not created by the Python host")
            })?;
        Ok(payload.object.clone_ref(py))
    }

    #[getter]
    fn backend(&self) -> PyResult<String> {
        let inner = self.clone_inner()?;
        let payload = inner
            .external_payload()
            .map_err(|_| PyTypeError::new_err("table batches do not have an array backend"))?;
        let payload = payload
            .as_any()
            .downcast_ref::<PythonPayload>()
            .ok_or_else(|| {
                PyTypeError::new_err("array batch payload was not created by the Python host")
            })?;
        Ok(payload.backend().into())
    }

    #[getter]
    fn kind(&self) -> PyResult<&'static str> {
        Ok(match self.clone_inner()?.kind() {
            calc_flow::BatchKind::Table => "table",
            calc_flow::BatchKind::Array => "array",
        })
    }

    #[getter]
    fn num_rows(&self) -> PyResult<usize> {
        Ok(self.clone_inner()?.num_rows())
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.clone_inner()?;
        metadata_to_python(py, inner.metadata())
    }

    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3's garbage-collector protocol requires PyVisit by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        let guard = self.inner.read();
        let Some(inner) = guard.as_ref() else {
            return Ok(());
        };
        let Ok(payload) = inner.external_payload() else {
            return Ok(());
        };
        let Some(payload) = payload.as_any().downcast_ref::<PythonPayload>() else {
            return Ok(());
        };
        visit.call(&payload.object)
    }

    fn __clear__(&self) {
        let inner = self.inner.write().take();
        drop(inner);
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<OwnedArrayToken>()?;
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
    } else if value.is_instance_of::<PyBool>() {
        Ok(())
    } else if value.is_instance_of::<PyInt>() {
        if value.extract::<i64>().is_ok() || value.extract::<u64>().is_ok() {
            Ok(())
        } else {
            Err(PyTypeError::new_err(METADATA_INTEGER_MESSAGE))
        }
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
        types::{PyDict, PyList, PyTuple},
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

            assert_eq!(batch.kind().unwrap(), "table");
            assert_eq!(batch.num_rows().unwrap(), 3);
            let inner = batch.clone_inner().unwrap();
            let inner_values = inner.table_payload().unwrap().batches()[0].column(0);
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

            let inner = batch.clone_inner().unwrap();
            let table = inner.table_payload().unwrap();
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

            assert_eq!(batch.kind().unwrap(), "array");
            assert_eq!(batch.num_rows().unwrap(), 2);
            let inner = batch.clone_inner().unwrap();
            let payload = inner.external_payload().unwrap();
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

    #[test]
    fn cleared_batch_rejects_every_payload_accessor() {
        Python::initialize();
        Python::attach(|py| {
            let (table, _) = table();
            let table = table.into_pyarrow(py).unwrap();
            let batch = PyBatch::from_pyarrow(py, &table, None).unwrap();

            batch.__clear__();

            for error in [
                batch.clone_inner().unwrap_err(),
                batch.kind().unwrap_err(),
                batch.num_rows().unwrap_err(),
                batch.metadata(py).unwrap_err(),
                batch.to_pyarrow(py).unwrap_err(),
            ] {
                assert!(error.is_instance_of::<PyRuntimeError>(py));
                assert_eq!(error.value(py).to_string(), CLEARED_BATCH_MESSAGE);
            }
            batch.__clear__();
        });
    }

    #[test]
    fn external_batch_payload_can_drop_on_a_background_thread() {
        Python::initialize();
        let inner = Python::attach(|py| {
            let object = PyList::new(py, [1, 2]).unwrap().unbind().into_any();
            let metadata = PyDict::new(py);
            PyBatch::_from_external(object, "numpy".into(), 2, metadata.as_any())
                .unwrap()
                .clone_inner()
                .unwrap()
        });

        std::thread::spawn(move || drop(inner)).join().unwrap();
        Python::attach(|_| {});
    }

    #[test]
    fn owned_numpy_shape_validation_rejects_every_allocation_bound() {
        assert_eq!(
            validate_owned_numpy_shape(&[]).unwrap_err(),
            "owned NumPy shape must have at least one dimension"
        );
        assert_eq!(
            validate_owned_numpy_shape(&[1; 17]).unwrap_err(),
            "owned NumPy shape must have at most 16 dimensions"
        );
        assert_eq!(
            validate_owned_numpy_shape(&[0]).unwrap_err(),
            "owned NumPy shape dimensions must be positive"
        );
        assert_eq!(
            validate_owned_numpy_shape(&[1_000_001]).unwrap_err(),
            "owned NumPy shape dimension exceeds 1000000"
        );
        assert_eq!(
            validate_owned_numpy_shape(&[1_000_000, 11]).unwrap_err(),
            "owned NumPy shape element count exceeds 10000000"
        );
    }

    #[test]
    fn owned_numpy_adoption_preserves_identity_and_irreversible_read_only_state() {
        Python::initialize();
        Python::attach(|py| {
            let shape = PyTuple::new(py, [2, 2]).unwrap();
            let (object, token) = PyBatch::_new_owned_numpy(py, shape.as_any(), "float64").unwrap();
            let identity = object.bind(py).as_ptr();
            let pointer: usize = object
                .bind(py)
                .getattr("__array_interface__")
                .unwrap()
                .get_item("data")
                .unwrap()
                .get_item(0)
                .unwrap()
                .extract()
                .unwrap();
            let metadata = PyDict::new(py);

            let batch = PyBatch::_from_owned_array(
                py,
                object,
                "numpy".into(),
                Some(token.bind(py).borrow()),
                metadata.as_any(),
            )
            .unwrap();
            let output = batch.array(py).unwrap();

            assert_eq!(output.bind(py).as_ptr(), identity);
            assert_eq!(
                output
                    .bind(py)
                    .getattr("__array_interface__")
                    .unwrap()
                    .get_item("data")
                    .unwrap()
                    .get_item(0)
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                pointer
            );
            let flags = output.bind(py).getattr("flags").unwrap();
            assert!(!flags.getattr("owndata").unwrap().extract::<bool>().unwrap());
            assert!(
                !flags
                    .getattr("writeable")
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
            let error = output
                .bind(py)
                .call_method1("setflags", (true,))
                .unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));

            let error = match PyBatch::_from_owned_array(
                py,
                output,
                "numpy".into(),
                Some(token.bind(py).borrow()),
                metadata.as_any(),
            ) {
                Ok(_) => panic!("an ownership token must be one-use"),
                Err(error) => error,
            };
            assert_eq!(
                error.value(py).to_string(),
                "owned array token was already consumed"
            );
        });
    }
}
