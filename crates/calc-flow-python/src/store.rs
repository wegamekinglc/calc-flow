use std::{path::PathBuf, sync::Arc};

use pyo3::prelude::*;
use tokio::sync::OnceCell;

use calc_flow::ProjectStore as _;

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
fn parse_project(document: &str) -> PyResult<calc_flow::ProjectSpec> {
    calc_flow::import_project_json(document.as_bytes()).map_err(crate::error::to_py_err)
}

#[pyfunction]
fn import_project_json(py: Python<'_>, document: &[u8]) -> PyResult<String> {
    let document = document.to_vec();
    py.detach(move || {
        let project = calc_flow::import_project_json(&document).map_err(crate::error::to_py_err)?;
        calc_flow::export_project_json(&project).map_err(crate::error::to_py_err)
    })
}

#[pyfunction]
fn import_project_yaml(py: Python<'_>, document: &[u8]) -> PyResult<String> {
    let document = document.to_vec();
    py.detach(move || {
        let project = calc_flow::import_project_yaml(&document).map_err(crate::error::to_py_err)?;
        calc_flow::export_project_json(&project).map_err(crate::error::to_py_err)
    })
}

#[pyfunction]
fn export_project_json(py: Python<'_>, project_json: &str) -> PyResult<String> {
    let project_json = project_json.to_owned();
    py.detach(move || {
        let project = parse_project(&project_json)?;
        calc_flow::export_project_json(&project).map_err(crate::error::to_py_err)
    })
}

#[pyfunction]
fn export_project_yaml(py: Python<'_>, project_json: &str) -> PyResult<String> {
    let project_json = project_json.to_owned();
    py.detach(move || {
        let project = parse_project(&project_json)?;
        calc_flow::export_project_yaml(&project).map_err(crate::error::to_py_err)
    })
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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyFileProjectStore>()?;
    module.add_function(wrap_pyfunction!(import_project_json, module)?)?;
    module.add_function(wrap_pyfunction!(import_project_yaml, module)?)?;
    module.add_function(wrap_pyfunction!(export_project_json, module)?)?;
    module.add_function(wrap_pyfunction!(export_project_yaml, module)?)?;
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
            "calc-flow-python-project-store-{label}-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn project_json(name: &str) -> String {
        format!(
            r#"{{"format_version":2,"id":"{name}","name":"{name}","pipeline":{{"name":"{name}","nodes":[{{"id":"calc","operator":{{"kind":"expression","expression":"b = a + 1"}}}}]}}}}"#
        )
    }

    #[test]
    fn native_project_store_covers_async_crud_and_strict_documents() {
        Python::initialize();
        Python::attach(|py| {
            let project_directory = directory("crud");
            let projects = Py::new(py, PyFileProjectStore::new(project_directory.clone())).unwrap();
            let locals = PyDict::new(py);
            locals.set_item("projects", projects).unwrap();
            locals
                .set_item("project_document", project_json("stored"))
                .unwrap();
            py.run(
                &CString::new(
                    "import asyncio, json\nasync def exercise():\n    await projects.create(project_document)\n    try:\n        await projects.create(project_document)\n    except Exception as error:\n        assert 'already exists' in str(error)\n    assert json.loads(await projects.get('stored'))['id'] == 'stored'\n    assert len(await projects.list()) == 1\n    await projects.put(project_document)\n    await projects.delete('stored')\n    try:\n        await projects.get('stored')\n    except Exception as error:\n        assert 'not found' in str(error)\nasyncio.run(exercise())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();

            assert!(parse_project(r#"{"format_version":1}"#).is_err());
            std::fs::remove_dir_all(project_directory).unwrap();
        });
    }

    #[test]
    fn native_project_store_registration_and_transforms_are_strict() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_native").unwrap();
            register(&module).unwrap();
            assert!(module.getattr("_FileProjectStore").is_ok());
            assert!(module.getattr("_FileCheckpointStore").is_err());
            assert!(module.getattr("import_project_json").is_ok());
            assert!(module.getattr("import_project_yaml").is_ok());
            assert!(module.getattr("export_project_json").is_ok());
            assert!(module.getattr("export_project_yaml").is_ok());

            let project = project_json("portable");
            let imported = import_project_json(py, project.as_bytes()).unwrap();
            assert!(imported.ends_with('\n'));
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&imported).unwrap()["format_version"],
                2
            );

            let yaml = export_project_yaml(py, &project).unwrap();
            assert!(yaml.contains("format_version: 2"));
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(
                    &import_project_yaml(py, yaml.as_bytes()).unwrap(),
                )
                .unwrap()["id"],
                "portable"
            );
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(
                    &export_project_json(py, &project).unwrap()
                )
                .unwrap()["id"],
                "portable"
            );
            assert!(
                import_project_json(py, &vec![b'x'; calc_flow::MAX_PROJECT_DOCUMENT_BYTES + 1])
                    .is_err()
            );
            assert!(
                import_project_yaml(
                    py,
                    b"format_version: 2\nid: aliases\nname: &name aliases\ndescription: *name\npipeline: {name: p, nodes: []}\n",
                )
                .is_err()
            );
        });
    }
}
