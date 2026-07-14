use std::{path::PathBuf, sync::Arc};

use async_trait::async_trait;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use tokio::sync::OnceCell;

use calc_flow::{CheckpointStore as _, ProjectStore as _};

struct LazyProjectStore {
    directory: PathBuf,
    inner: OnceCell<calc_flow::FileProjectStore>,
}

impl LazyProjectStore {
    fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            inner: OnceCell::new(),
        }
    }

    async fn get(&self) -> calc_flow::Result<&calc_flow::FileProjectStore> {
        self.inner
            .get_or_try_init(|| calc_flow::FileProjectStore::new(&self.directory))
            .await
    }
}

pub(crate) struct LazyCheckpointStore {
    directory: PathBuf,
    inner: OnceCell<calc_flow::FileCheckpointStore>,
}

impl LazyCheckpointStore {
    fn new(directory: PathBuf) -> Self {
        Self {
            directory,
            inner: OnceCell::new(),
        }
    }

    async fn get(&self) -> calc_flow::Result<&calc_flow::FileCheckpointStore> {
        self.inner
            .get_or_try_init(|| calc_flow::FileCheckpointStore::new(&self.directory))
            .await
    }
}

#[async_trait]
impl calc_flow::CheckpointStore for LazyCheckpointStore {
    async fn load(&self, pipeline_name: &str) -> calc_flow::Result<Option<calc_flow::Checkpoint>> {
        self.get().await?.load(pipeline_name).await
    }

    async fn save(&self, checkpoint: &calc_flow::Checkpoint) -> calc_flow::Result<()> {
        self.get().await?.save(checkpoint).await
    }

    async fn delete(&self, pipeline_name: &str) -> calc_flow::Result<()> {
        self.get().await?.delete(pipeline_name).await
    }
}

fn parse_project(document: &str) -> PyResult<calc_flow::ProjectSpec> {
    calc_flow::import_project_json(document.as_bytes()).map_err(crate::error::to_py_err)
}

fn parse_checkpoint(document: &str) -> PyResult<calc_flow::Checkpoint> {
    serde_json::from_str(document).map_err(|error| {
        crate::error::to_py_err(calc_flow::CalcFlowError::Format {
            message: format!("invalid checkpoint document: {error}"),
        })
    })
}

fn encode_checkpoint(checkpoint: &calc_flow::Checkpoint) -> PyResult<String> {
    serde_json::to_string(checkpoint).map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

#[pyclass(name = "_FileProjectStore", frozen, module = "calc_flow._native")]
pub(crate) struct PyFileProjectStore {
    inner: Arc<LazyProjectStore>,
}

#[pymethods]
impl PyFileProjectStore {
    #[new]
    fn new(directory: PathBuf) -> Self {
        Self {
            inner: Arc::new(LazyProjectStore::new(directory)),
        }
    }

    fn create<'py>(&self, py: Python<'py>, project_json: &str) -> PyResult<Bound<'py, PyAny>> {
        let project = parse_project(project_json)?;
        let store = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = store.get().await.map_err(crate::error::to_py_err)?;
            store
                .create(&project)
                .await
                .map_err(crate::error::to_py_err)
        })
    }

    fn put<'py>(&self, py: Python<'py>, project_json: &str) -> PyResult<Bound<'py, PyAny>> {
        let project = parse_project(project_json)?;
        let store = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = store.get().await.map_err(crate::error::to_py_err)?;
            store.put(&project).await.map_err(crate::error::to_py_err)
        })
    }

    fn get<'py>(&self, py: Python<'py>, project_id: String) -> PyResult<Bound<'py, PyAny>> {
        let store = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = store.get().await.map_err(crate::error::to_py_err)?;
            let project = store
                .get(&project_id)
                .await
                .map_err(crate::error::to_py_err)?;
            calc_flow::export_project_json(&project).map_err(crate::error::to_py_err)
        })
    }

    fn list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let store = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = store.get().await.map_err(crate::error::to_py_err)?;
            store
                .list()
                .await
                .map_err(crate::error::to_py_err)?
                .iter()
                .map(calc_flow::export_project_json)
                .collect::<calc_flow::Result<Vec<_>>>()
                .map_err(crate::error::to_py_err)
        })
    }

    fn delete<'py>(&self, py: Python<'py>, project_id: String) -> PyResult<Bound<'py, PyAny>> {
        let store = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let store = store.get().await.map_err(crate::error::to_py_err)?;
            store
                .delete(&project_id)
                .await
                .map_err(crate::error::to_py_err)
        })
    }
}

#[pyclass(name = "_FileCheckpointStore", frozen, module = "calc_flow._native")]
pub(crate) struct PyFileCheckpointStore {
    inner: Arc<LazyCheckpointStore>,
}

impl PyFileCheckpointStore {
    pub(crate) fn from_directory(directory: PathBuf) -> Self {
        Self {
            inner: Arc::new(LazyCheckpointStore::new(directory)),
        }
    }

    pub(crate) fn clone_store(&self) -> Arc<dyn calc_flow::CheckpointStore> {
        Arc::clone(&self.inner) as Arc<dyn calc_flow::CheckpointStore>
    }
}

#[pymethods]
impl PyFileCheckpointStore {
    #[new]
    fn new(directory: PathBuf) -> Self {
        Self::from_directory(directory)
    }

    fn load<'py>(&self, py: Python<'py>, pipeline_name: String) -> PyResult<Bound<'py, PyAny>> {
        let store = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            store
                .load(&pipeline_name)
                .await
                .map_err(crate::error::to_py_err)?
                .as_ref()
                .map(encode_checkpoint)
                .transpose()
        })
    }

    fn save<'py>(&self, py: Python<'py>, checkpoint_json: &str) -> PyResult<Bound<'py, PyAny>> {
        let checkpoint = parse_checkpoint(checkpoint_json)?;
        let store = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            store
                .save(&checkpoint)
                .await
                .map_err(crate::error::to_py_err)
        })
    }

    fn delete<'py>(&self, py: Python<'py>, pipeline_name: String) -> PyResult<Bound<'py, PyAny>> {
        let store = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            store
                .delete(&pipeline_name)
                .await
                .map_err(crate::error::to_py_err)
        })
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyFileProjectStore>()?;
    module.add_class::<PyFileCheckpointStore>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        ffi::CString,
        sync::atomic::{AtomicU64, Ordering},
    };

    use pyo3::types::PyDict;

    use super::*;

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    fn directory(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "calc-flow-python-task20-{label}-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn project_json(name: &str) -> String {
        format!(
            r#"{{"format_version":2,"id":"{name}","name":"{name}","pipeline":{{"name":"{name}","nodes":[{{"id":"calc","operator":{{"kind":"expression","expression":"b = a + 1"}}}}]}}}}"#
        )
    }

    fn checkpoint_json(name: &str) -> String {
        format!(
            r#"{{"created_at":"2026-07-14T00:00:00Z","format_version":2,"pipeline_fingerprint":"fingerprint","pipeline_name":"{name}","sequence":4,"source_cursor":{{"offset":3}},"state":{{"calc":null}}}}"#
        )
    }

    #[test]
    fn native_stores_cover_async_crud_and_strict_documents() {
        Python::initialize();
        Python::attach(|py| {
            let project_directory = directory("projects");
            let checkpoint_directory = directory("checkpoints");
            let projects = Py::new(py, PyFileProjectStore::new(project_directory.clone())).unwrap();
            let checkpoints =
                Py::new(py, PyFileCheckpointStore::new(checkpoint_directory.clone())).unwrap();
            let locals = PyDict::new(py);
            locals.set_item("projects", projects).unwrap();
            locals.set_item("checkpoints", checkpoints).unwrap();
            locals
                .set_item("project_document", project_json("stored"))
                .unwrap();
            locals
                .set_item("checkpoint_document", checkpoint_json("stored"))
                .unwrap();
            py.run(
                &CString::new(
                    "import asyncio, json\nasync def exercise():\n    await projects.create(project_document)\n    try:\n        await projects.create(project_document)\n    except Exception as error:\n        assert 'already exists' in str(error)\n    assert json.loads(await projects.get('stored'))['id'] == 'stored'\n    assert len(await projects.list()) == 1\n    await projects.put(project_document)\n    await projects.delete('stored')\n    try:\n        await projects.get('stored')\n    except Exception as error:\n        assert 'not found' in str(error)\n    await checkpoints.save(checkpoint_document)\n    assert json.loads(await checkpoints.load('stored'))['sequence'] == 4\n    await checkpoints.delete('stored')\n    await checkpoints.delete('stored')\n    assert await checkpoints.load('stored') is None\nasyncio.run(exercise())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();

            assert!(parse_project(r#"{"format_version":1}"#).is_err());
            assert!(parse_checkpoint("[]").is_err());
            assert!(parse_checkpoint(r#"{"format_version":2}"#).is_err());
            let checkpoint = parse_checkpoint(&checkpoint_json("encoded")).unwrap();
            assert!(encode_checkpoint(&checkpoint).unwrap().contains("encoded"));

            std::fs::remove_dir_all(project_directory).unwrap();
            std::fs::remove_dir_all(checkpoint_directory).unwrap();
        });
    }

    #[test]
    fn native_store_registration_exposes_both_private_classes() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_native").unwrap();
            register(&module).unwrap();
            assert!(module.getattr("_FileProjectStore").is_ok());
            assert!(module.getattr("_FileCheckpointStore").is_ok());
        });
    }
}
