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
