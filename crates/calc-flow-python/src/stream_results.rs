use std::sync::Arc;

use calc_flow::{Batch, CalcFlowError, Result, StreamSink};
use parking_lot::Mutex;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use tokio::sync::mpsc;

use crate::batch::PyBatch;

type Output = (String, Batch);

struct NativeResultQueue {
    sender: Mutex<Option<mpsc::Sender<Output>>>,
    receiver: tokio::sync::Mutex<mpsc::Receiver<Output>>,
}

impl NativeResultQueue {
    fn sender(&self) -> PyResult<mpsc::Sender<Output>> {
        self.sender
            .lock()
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("native stream result queue is closed"))
    }
}

#[pyclass(name = "_ArrowResultQueue", frozen, module = "calc_flow._native")]
pub(crate) struct PyArrowResultQueue {
    inner: Arc<NativeResultQueue>,
}

fn python_output(py: Python<'_>, output: Output) -> PyResult<(String, Py<PyBatch>)> {
    let (name, batch) = output;
    Ok((name, Py::new(py, PyBatch::from_inner(batch))?))
}

#[pymethods]
impl PyArrowResultQueue {
    #[new]
    fn new(capacity: usize) -> PyResult<Self> {
        if capacity == 0 {
            return Err(PyRuntimeError::new_err(
                "native stream result queue capacity must be positive",
            ));
        }
        let (sender, receiver) = mpsc::channel(capacity);
        Ok(Self {
            inner: Arc::new(NativeResultQueue {
                sender: Mutex::new(Some(sender)),
                receiver: tokio::sync::Mutex::new(receiver),
            }),
        })
    }

    fn seal(&self) {
        self.inner.sender.lock().take();
    }

    fn try_next(&self, py: Python<'_>) -> PyResult<Option<(String, Py<PyBatch>)>> {
        let mut receiver =
            self.inner.receiver.try_lock().map_err(|_| {
                PyRuntimeError::new_err("native stream result read is already pending")
            })?;
        match receiver.try_recv() {
            Ok(output) => python_output(py, output).map(Some),
            Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {
                Ok(None)
            }
        }
    }

    fn next_async<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let queue = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let output = queue.receiver.lock().await.recv().await;
            Python::attach(|py| output.map(|output| python_output(py, output)).transpose())
        })
    }

    fn close_async<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let queue = Arc::clone(&self.inner);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            queue.sender.lock().take();
            let mut receiver = queue.receiver.lock().await;
            receiver.close();
            while receiver.try_recv().is_ok() {}
            Ok::<(), PyErr>(())
        })
    }
}

#[derive(Clone)]
pub(crate) struct ArrowResultSink {
    name: String,
    sender: mpsc::Sender<Output>,
}

#[pyclass(name = "_ArrowResultSink", frozen, module = "calc_flow._native")]
pub(crate) struct PyArrowResultSink {
    inner: ArrowResultSink,
}

impl PyArrowResultSink {
    pub(crate) fn native(&self) -> ArrowResultSink {
        self.inner.clone()
    }
}

#[pymethods]
impl PyArrowResultSink {
    #[new]
    fn new(queue: &PyArrowResultQueue, name: String) -> PyResult<Self> {
        Ok(Self {
            inner: ArrowResultSink {
                name,
                sender: queue.inner.sender()?,
            },
        })
    }
}

#[async_trait::async_trait]
impl StreamSink for ArrowResultSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.sender
            .send((self.name.clone(), batch.clone()))
            .await
            .map_err(|_| CalcFlowError::Internal {
                message: "native stream result queue is closed".into(),
            })
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyArrowResultQueue>()?;
    module.add_class::<PyArrowResultSink>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use calc_flow::BatchMetadata;
    use datafusion::arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };

    use super::*;

    fn batch() -> Batch {
        Batch::table(
            vec![
                RecordBatch::try_new(
                    Arc::new(Schema::new(vec![Field::new(
                        "value",
                        DataType::Int64,
                        false,
                    )])),
                    vec![Arc::new(Int64Array::from(vec![1, 2]))],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn bounded_result_sink_preserves_order_and_applies_backpressure() {
        let (sender, mut receiver) = mpsc::channel(1);
        let mut sink = ArrowResultSink {
            name: "result".into(),
            sender,
        };
        sink.write(&batch()).await.unwrap();
        let mut blocked = sink.clone();
        let writing = tokio::spawn(async move { blocked.write(&batch()).await });
        tokio::task::yield_now().await;
        assert!(!writing.is_finished());

        let (name, first) = receiver.recv().await.unwrap();
        assert_eq!(name, "result");
        assert_eq!(first.num_rows(), 2);
        writing.await.unwrap().unwrap();
        let (name, second) = receiver.recv().await.unwrap();
        assert_eq!(name, "result");
        assert_eq!(second.num_rows(), 2);
    }
}
