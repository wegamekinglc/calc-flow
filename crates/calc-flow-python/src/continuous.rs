use std::{
    collections::BTreeMap,
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use parking_lot::Mutex;
use pyo3::{
    IntoPyObjectExt, PyTraverseError, PyVisit,
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    intern,
    prelude::*,
    sync::PyOnceLock,
    types::{PyAny, PyCFunction, PyCapsule, PyDict, PyDictMethods, PyList, PyTuple, PyType},
};

use crate::{
    batch::PyBatch,
    config::PythonRoot,
    pipeline::PyStreamExecutionPlan,
    runtime::{PythonAsyncContext, PythonAwaitRegistry, python_json, resolve_python_in_context},
};

const SAFE_EXCEPTION_STORAGE: &str = "_calc_flow_safe_fields";
const NATIVE_EXCEPTION_STORAGE: &str = "_calc_flow_native_safe_fields";
const SAFE_EXCEPTION_FIELDS: [&str; 10] = [
    "category",
    "reason_code",
    "message",
    "job_id",
    "epoch",
    "checkpoint_phase",
    "component_kind",
    "component_id",
    "diagnostic_id",
    "position",
];
static STREAMING_RUNTIME_ERROR_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static CHECKPOINT_PUBLICATION_UNKNOWN_ERROR_TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static OBSERVER_RESULT_RESOLVER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

macro_rules! set_py_items {
    ($mapping:expr, { $($key:literal => $value:expr),+ $(,)? }) => {{
        $($mapping.set_item($key, $value)?;)+
    }};
}

macro_rules! py_tuple {
    ($py:expr, [$($value:expr),+ $(,)?]) => {
        PyTuple::new($py, [$($value.into_py_any($py)?,)+])
    };
}

struct SafeStreamingErrorFields {
    category: String,
    reason_code: Option<String>,
    message: String,
    job_id: Option<u64>,
    epoch: Option<u64>,
    checkpoint_phase: Option<String>,
    component_kind: Option<String>,
    component_id: Option<String>,
    diagnostic_id: Option<u64>,
    position: u32,
}

impl SafeStreamingErrorFields {
    fn internal(message: &str) -> Self {
        Self {
            category: "internal".to_owned(),
            reason_code: None,
            message: message.to_owned(),
            job_id: None,
            epoch: None,
            checkpoint_phase: None,
            component_kind: None,
            component_id: None,
            diagnostic_id: None,
            position: 0,
        }
    }

    fn from_streaming(error: &calc_flow::StreamingError) -> Self {
        Self {
            category: streaming_error_category_name(error.category()).to_owned(),
            reason_code: error
                .reason_code()
                .and_then(streaming_failure_reason_name)
                .map(str::to_owned),
            message: error.message().to_owned(),
            job_id: error.job_id(),
            epoch: error.epoch().map(calc_flow::Epoch::as_u64),
            checkpoint_phase: error
                .checkpoint_phase()
                .map(checkpoint_phase_name)
                .map(str::to_owned),
            component_kind: error
                .component_kind()
                .map(component_kind_name)
                .map(str::to_owned),
            component_id: error.component_id().map(str::to_owned),
            diagnostic_id: error.diagnostic_id(),
            position: error.position(),
        }
    }
}

#[pyclass(name = "_StreamingRunner", frozen, module = "calc_flow._native")]
pub(crate) struct PyContinuousStreamingRunner {
    inner: Arc<RunnerStartState>,
}

struct RunnerStartState {
    runner: Mutex<Option<calc_flow::StreamingRunner>>,
    roots: Mutex<Vec<Arc<PythonRoot>>>,
    awaits: Arc<PythonAwaitRegistry>,
    context: Arc<Mutex<Option<Arc<PythonAsyncContext>>>>,
    ownership: Arc<ConnectorOwnership>,
}

#[pyclass(frozen, module = "calc_flow._native")]
struct PyStreamingStartAwaitable {
    inner: Arc<RunnerStartState>,
    started: AtomicBool,
}

#[pyclass(name = "_StreamingJob", frozen, module = "calc_flow._native")]
pub(crate) struct PyStreamingJob {
    inner: Mutex<Option<Arc<calc_flow::StreamingJob>>>,
    roots: Arc<Mutex<Vec<Arc<PythonRoot>>>>,
    awaits: Arc<PythonAwaitRegistry>,
    context: Arc<Mutex<Option<Arc<PythonAsyncContext>>>>,
}

#[pyclass(
    name = "_ManagedCheckpointRuntime",
    frozen,
    module = "calc_flow._native"
)]
pub(crate) struct PyManagedCheckpointRuntime {
    inner: Mutex<Option<calc_flow::ManagedCheckpointRuntime>>,
}

#[pyclass(frozen, module = "calc_flow._native")]
struct PyStreamingJobAwaitable {
    inner: Arc<calc_flow::StreamingJob>,
    awaits: Arc<PythonAwaitRegistry>,
    operation: JobOperation,
    started: AtomicBool,
}

#[derive(Clone, Copy)]
enum JobOperation {
    TriggerCheckpoint,
    Shutdown,
    Cancel,
    Wait,
}

fn connector_error(component: &str, error: impl std::fmt::Display) -> calc_flow::CalcFlowError {
    calc_flow::CalcFlowError::ExternalProvider {
        provider: "python".into(),
        name: component.into(),
        version: "1".into(),
        message: error.to_string(),
    }
}

fn required_item<'py>(mapping: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    mapping
        .get_item(key)?
        .ok_or_else(|| PyTypeError::new_err(format!("missing required field {key:?}")))
}

fn replay_positioning(value: &Bound<'_, PyDict>) -> PyResult<calc_flow::ReplayPositioning> {
    match required_item(value, "replay_positioning")?
        .extract::<String>()?
        .as_str()
    {
        "exact_pause_report_and_seek" => Ok(calc_flow::ReplayPositioning::ExactPauseReportAndSeek),
        "unsupported" => Ok(calc_flow::ReplayPositioning::Unsupported),
        other => Err(PyTypeError::new_err(format!(
            "unsupported replay positioning {other:?}"
        ))),
    }
}

fn source_delivery(value: &Bound<'_, PyDict>) -> PyResult<calc_flow::SourceDeliveryCapability> {
    match required_item(value, "delivery")?
        .extract::<String>()?
        .as_str()
    {
        "lossless" => Ok(calc_flow::SourceDeliveryCapability::Lossless),
        "lossy" => Ok(calc_flow::SourceDeliveryCapability::Lossy),
        other => Err(PyTypeError::new_err(format!(
            "unsupported source delivery {other:?}"
        ))),
    }
}

fn source_schema(value: &Bound<'_, PyDict>) -> PyResult<calc_flow::SourceSchema> {
    let schema = required_item(value, "schema")?;
    if schema.is_none() {
        return Ok(calc_flow::SourceSchema::DynamicOrUnknown);
    }
    let capsule = schema
        .call_method0("__arrow_c_schema__")?
        .cast_into::<PyCapsule>()?;
    let schema = pyo3_arrow::PySchema::from_arrow_pycapsule(&capsule)?;
    Ok(calc_flow::SourceSchema::Exact(schema.into()))
}

fn native_watermarks(value: &Bound<'_, PyDict>) -> PyResult<calc_flow::NativeWatermarkCapability> {
    match required_item(value, "native_watermarks")?
        .extract::<String>()?
        .as_str()
    {
        "never_emits" => Ok(calc_flow::NativeWatermarkCapability::NeverEmits),
        "emits_native" => Ok(calc_flow::NativeWatermarkCapability::EmitsNative),
        "runtime_toggleable" => Ok(calc_flow::NativeWatermarkCapability::RuntimeToggleable),
        "unknown" => Ok(calc_flow::NativeWatermarkCapability::Unknown),
        other => Err(PyTypeError::new_err(format!(
            "unsupported native watermark capability {other:?}"
        ))),
    }
}

fn parse_source_capabilities(value: &Bound<'_, PyDict>) -> PyResult<calc_flow::SourceCapabilities> {
    let (max_batch_rows, max_batch_bytes) = source_batch_bounds(value)?;
    Ok(calc_flow::SourceCapabilities {
        replay_positioning: replay_positioning(value)?,
        delivery: source_delivery(value)?,
        max_batch_rows,
        max_batch_bytes,
        schema: source_schema(value)?,
        native_watermarks: native_watermarks(value)?,
    })
}

fn source_batch_bounds(value: &Bound<'_, PyDict>) -> PyResult<(usize, usize)> {
    Ok((
        required_item(value, "max_batch_rows")?.extract()?,
        required_item(value, "max_batch_bytes")?.extract()?,
    ))
}

fn source_cursor_arguments(
    cursor: Option<calc_flow::Cursor>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    Python::attach(|py| {
        let Some(cursor) = cursor else {
            return Ok((py.None(), py.None(), py.None()));
        };
        let payload = serde_json::to_string(cursor.payload())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok((
            cursor.source_id().into_py_any(py)?,
            cursor.order().into_py_any(py)?,
            crate::config::json_to_python(py, &payload)?.unbind(),
        ))
    })
}

fn source_event_batch(
    py: Python<'_>,
    value: &Bound<'_, PyTuple>,
) -> calc_flow::Result<calc_flow::Batch> {
    value
        .get_item(1)
        .map_err(|error| connector_error("source.next", error))?
        .extract::<PyRef<'_, PyBatch>>()
        .map_err(|error| connector_error("source.next", error))?
        .clone_inner()
        .and_then(|batch| crate::batch::rehome_python_payload(py, batch))
        .map_err(|error| connector_error("source.next", error))
}

fn source_event_cursor(value: &Bound<'_, PyTuple>) -> calc_flow::Result<calc_flow::Cursor> {
    let source_id = value
        .get_item(2)
        .and_then(|item| item.extract::<Option<String>>())
        .map_err(|error| connector_error("source.next", error))?;
    let order = value
        .get_item(3)
        .and_then(|item| item.extract::<Vec<u8>>())
        .map_err(|error| connector_error("source.next", error))?;
    let payload = value
        .get_item(4)
        .map_err(|error| connector_error("source.next", error))?;
    let payload = json_map(&payload, "source.next cursor")?;
    match source_id {
        Some(source_id) => calc_flow::Cursor::new(source_id, order, payload),
        None => calc_flow::Cursor::unbound(order, payload),
    }
}

fn data_source_event(
    py: Python<'_>,
    value: &Bound<'_, PyTuple>,
) -> calc_flow::Result<calc_flow::SourceEvent> {
    Ok(calc_flow::SourceEvent::Data {
        batch: source_event_batch(py, value)?,
        cursor: source_event_cursor(value)?,
    })
}

fn watermark_source_event(value: &Bound<'_, PyTuple>) -> calc_flow::Result<calc_flow::SourceEvent> {
    let micros = value
        .get_item(1)
        .and_then(|item| item.extract::<i64>())
        .map_err(|error| connector_error("source.next", error))?;
    Ok(calc_flow::SourceEvent::Watermark(
        calc_flow::EventTime::from_micros(micros),
    ))
}

fn source_event_from_tuple(
    py: Python<'_>,
    value: &Bound<'_, PyTuple>,
) -> calc_flow::Result<calc_flow::SourceEvent> {
    let tag = value
        .get_item(0)
        .and_then(|item| item.extract::<String>())
        .map_err(|error| connector_error("source.next", error))?;
    match tag.as_str() {
        "data" if value.len() == 5 => data_source_event(py, value),
        "watermark" if value.len() == 2 => watermark_source_event(value),
        "idle" if value.len() == 1 => Ok(calc_flow::SourceEvent::Idle),
        _ => Err(connector_error("source.next", "invalid source event")),
    }
}

fn decode_source_event(value: &Py<PyAny>) -> calc_flow::Result<Option<calc_flow::SourceEvent>> {
    Python::attach(|py| {
        let value = value.bind(py);
        if value.is_none() {
            return Ok(None);
        }
        let value = value
            .cast::<PyTuple>()
            .map_err(|_| connector_error("source.next", "invalid source event"))?;
        source_event_from_tuple(py, value).map(Some)
    })
}

fn json_map(value: &Bound<'_, PyAny>, label: &str) -> calc_flow::Result<calc_flow::JsonMap> {
    let value = python_json(value, label)?;
    serde_json::from_value(value).map_err(|error| connector_error(label, error))
}

struct PythonContinuousSource {
    binding: Arc<PythonRoot>,
    awaits: Arc<PythonAwaitRegistry>,
    capability_error: Arc<Mutex<Option<String>>>,
    context: Arc<Mutex<Option<Arc<PythonAsyncContext>>>>,
    _ownership: ConnectorOwnershipLease,
}

struct ConnectorOwnership {
    pending: AtomicUsize,
    idle: tokio::sync::Notify,
}

impl ConnectorOwnership {
    fn new() -> Self {
        Self {
            pending: AtomicUsize::new(0),
            idle: tokio::sync::Notify::new(),
        }
    }

    fn retain(self: &Arc<Self>) -> ConnectorOwnershipLease {
        self.pending.fetch_add(1, Ordering::AcqRel);
        ConnectorOwnershipLease(Arc::clone(self))
    }

    async fn wait_idle(&self) {
        loop {
            let notified = self.idle.notified();
            if self.pending.load(Ordering::Acquire) == 0 {
                return;
            }
            notified.await;
        }
    }
}

struct ConnectorOwnershipLease(Arc<ConnectorOwnership>);

type PythonSourceBindings = (
    BTreeMap<String, calc_flow::SourceBinding>,
    Vec<Arc<PythonRoot>>,
);
type PythonSinkBindings = (
    BTreeMap<String, Vec<calc_flow::SinkBinding>>,
    Vec<Arc<PythonRoot>>,
);

impl Drop for ConnectorOwnershipLease {
    fn drop(&mut self) {
        let previous = self.0.pending.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
        if previous == 1 {
            self.0.idle.notify_waiters();
        }
    }
}

async fn resolve_connector(
    value: Py<PyAny>,
    callback_name: &str,
    awaits: &Arc<PythonAwaitRegistry>,
    context: &Arc<Mutex<Option<Arc<PythonAsyncContext>>>>,
) -> calc_flow::Result<Py<PyAny>> {
    let context = context
        .lock()
        .clone()
        .ok_or_else(|| connector_error(callback_name, "Python event loop is unavailable"))?;
    resolve_python_in_context(value, callback_name, awaits, &context).await
}

impl PythonContinuousSource {
    fn parsed_capabilities(&self) -> PyResult<calc_flow::SourceCapabilities> {
        Python::attach(|py| {
            let value = self
                .binding
                .object()
                .bind(py)
                .call_method0("_native_capabilities")?;
            let value = value.cast::<PyDict>()?;
            parse_source_capabilities(value)
        })
    }
}

#[async_trait::async_trait]
impl calc_flow::StreamSource for PythonContinuousSource {
    fn capabilities(&self) -> calc_flow::SourceCapabilities {
        self.parsed_capabilities().unwrap_or_else(|error| {
            self.capability_error.lock().replace(error.to_string());
            calc_flow::SourceCapabilities {
                replay_positioning: calc_flow::ReplayPositioning::ExactPauseReportAndSeek,
                delivery: calc_flow::SourceDeliveryCapability::Lossless,
                max_batch_rows: usize::MAX,
                max_batch_bytes: usize::MAX,
                schema: calc_flow::SourceSchema::DynamicOrUnknown,
                native_watermarks: calc_flow::NativeWatermarkCapability::NeverEmits,
            }
        })
    }

    async fn open(&mut self, cursor: Option<calc_flow::Cursor>) -> calc_flow::Result<()> {
        if let Some(error) = self.capability_error.lock().take() {
            return Err(connector_error("source.capabilities", error));
        }
        let (source_id, order, payload) = source_cursor_arguments(cursor)
            .map_err(|error| connector_error("source.open", error))?;
        let value = Python::attach(|py| {
            self.binding
                .object()
                .bind(py)
                .call_method1("_native_open", (source_id, order, payload))
                .map(Bound::unbind)
        })
        .map_err(|error| connector_error("source.open", error))?;
        resolve_connector(value, "source.open", &self.awaits, &self.context).await?;
        Ok(())
    }

    async fn next(&mut self) -> calc_flow::Result<Option<calc_flow::SourceEvent>> {
        let value = Python::attach(|py| {
            self.binding
                .object()
                .bind(py)
                .call_method0("_native_next")
                .map(Bound::unbind)
        })
        .map_err(|error| connector_error("source.next", error))?;
        let value = resolve_connector(value, "source.next", &self.awaits, &self.context).await?;
        decode_source_event(&value)
    }

    async fn close(&mut self) -> calc_flow::Result<()> {
        let value = Python::attach(|py| {
            self.binding
                .object()
                .bind(py)
                .call_method0("_native_close")
                .map(Bound::unbind)
        })
        .map_err(|error| connector_error("source.close", error))?;
        resolve_connector(value, "source.close", &self.awaits, &self.context).await?;
        Ok(())
    }
}

struct PythonContinuousSink {
    binding: Arc<PythonRoot>,
    awaits: Arc<PythonAwaitRegistry>,
    context: Arc<Mutex<Option<Arc<PythonAsyncContext>>>>,
    _ownership: ConnectorOwnershipLease,
}

impl PythonContinuousSink {
    async fn call0(&self, method: &str) -> calc_flow::Result<()> {
        let value = Python::attach(|py| {
            self.binding
                .object()
                .bind(py)
                .call_method0(method)
                .map(Bound::unbind)
        })
        .map_err(|error| connector_error(method, error))?;
        resolve_connector(value, method, &self.awaits, &self.context).await?;
        Ok(())
    }

    async fn call_args(&self, method: &str, args: Py<PyTuple>) -> calc_flow::Result<Py<PyAny>> {
        let value = Python::attach(|py| {
            self.binding
                .object()
                .bind(py)
                .call_method1(method, args.bind(py))
                .map(Bound::unbind)
        })
        .map_err(|error| connector_error(method, error))?;
        resolve_connector(value, method, &self.awaits, &self.context).await
    }

    async fn write_batch(&self, batch: &calc_flow::Batch) -> calc_flow::Result<()> {
        let args = Python::attach(|py| {
            let batch = Py::new(py, PyBatch::from_inner_python(py, batch.clone())?)?;
            PyTuple::new(py, [batch]).map(Bound::unbind)
        })
        .map_err(|error| connector_error("sink.write", error))?;
        self.call_args("_native_write", args).await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl calc_flow::StreamSink for PythonContinuousSink {
    async fn open(&mut self) -> calc_flow::Result<()> {
        self.call0("_native_open").await
    }

    async fn write(&mut self, batch: &calc_flow::Batch) -> calc_flow::Result<()> {
        self.write_batch(batch).await
    }

    async fn close(&mut self) -> calc_flow::Result<()> {
        self.call0("_native_close").await
    }
}

#[async_trait::async_trait]
impl calc_flow::TransactionalStreamSink for PythonContinuousSink {
    async fn open(&mut self) -> calc_flow::Result<()> {
        self.call0("_native_open").await
    }

    async fn begin_epoch(&mut self, epoch: calc_flow::Epoch) -> calc_flow::Result<()> {
        let args = Python::attach(|py| PyTuple::new(py, [epoch.as_u64()]).map(Bound::unbind))
            .map_err(|error| connector_error("sink.begin_epoch", error))?;
        self.call_args("_native_begin_epoch", args).await?;
        Ok(())
    }

    async fn write(&mut self, batch: &calc_flow::Batch) -> calc_flow::Result<()> {
        self.write_batch(batch).await
    }

    async fn pre_commit(
        &mut self,
        epoch: calc_flow::Epoch,
    ) -> calc_flow::Result<calc_flow::JsonMap> {
        let args = Python::attach(|py| PyTuple::new(py, [epoch.as_u64()]).map(Bound::unbind))
            .map_err(|error| connector_error("sink.pre_commit", error))?;
        let value = self.call_args("_native_pre_commit", args).await?;
        Python::attach(|py| json_map(value.bind(py), "sink.pre_commit"))
    }

    async fn commit(
        &mut self,
        epoch: calc_flow::Epoch,
        pre_commit: &calc_flow::JsonMap,
    ) -> calc_flow::Result<()> {
        let args = python_epoch_json_args(epoch, Some(pre_commit), "sink.commit")?;
        self.call_args("_native_commit", args).await?;
        Ok(())
    }

    async fn abort(
        &mut self,
        epoch: calc_flow::Epoch,
        pre_commit: Option<&calc_flow::JsonMap>,
    ) -> calc_flow::Result<()> {
        let args = python_epoch_json_args(epoch, pre_commit, "sink.abort")?;
        self.call_args("_native_abort", args).await?;
        Ok(())
    }

    async fn recover(&mut self, recovery: &calc_flow::SinkRecovery) -> calc_flow::Result<()> {
        let args = Python::attach(|py| -> PyResult<_> {
            let delivery = sink_delivery_to_py(py, recovery.delivery())?;
            let encoded = serde_json::to_string(recovery.pre_commit())
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            let pre_commit = crate::config::json_to_python(py, &encoded)?;
            PyTuple::new(
                py,
                [
                    recovery.epoch().as_u64().into_py_any(py)?,
                    recovery.terminal().into_py_any(py)?,
                    delivery.into_any().unbind(),
                    pre_commit.into_any().unbind(),
                ],
            )
            .map(Bound::unbind)
        })
        .map_err(|error| connector_error("sink.recover", error))?;
        self.call_args("_native_recover", args).await?;
        Ok(())
    }

    async fn close(&mut self) -> calc_flow::Result<()> {
        self.call0("_native_close").await
    }
}

fn python_epoch_json_args(
    epoch: calc_flow::Epoch,
    value: Option<&calc_flow::JsonMap>,
    label: &str,
) -> calc_flow::Result<Py<PyTuple>> {
    Python::attach(|py| {
        let value = match value {
            Some(value) => {
                let encoded = serde_json::to_string(value)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
                crate::config::json_to_python(py, &encoded)?.unbind()
            }
            None => py.None(),
        };
        PyTuple::new(py, [epoch.as_u64().into_py_any(py)?, value]).map(Bound::unbind)
    })
    .map_err(|error| connector_error(label, error))
}

impl PyContinuousStreamingRunner {
    #[allow(
        dead_code,
        reason = "unit tests construct already-validated runners without Python binding assembly"
    )]
    pub(crate) fn from_inner(inner: calc_flow::StreamingRunner) -> Self {
        Self {
            inner: Arc::new(RunnerStartState {
                runner: Mutex::new(Some(inner)),
                roots: Mutex::new(Vec::new()),
                awaits: Arc::new(PythonAwaitRegistry::new()),
                context: Arc::new(Mutex::new(None)),
                ownership: Arc::new(ConnectorOwnership::new()),
            }),
        }
    }
}

impl PyManagedCheckpointRuntime {
    fn take(&self) -> PyResult<calc_flow::ManagedCheckpointRuntime> {
        self.inner.lock().take().ok_or_else(|| {
            PyRuntimeError::new_err("managed checkpoint runtime has already been consumed")
        })
    }
}

#[pymethods]
impl PyManagedCheckpointRuntime {
    #[new]
    #[pyo3(signature = (directory, /))]
    fn new(directory: &str) -> PyResult<Self> {
        let inner =
            calc_flow::ManagedCheckpointRuntime::new(directory).map_err(crate::error::to_py_err)?;
        Ok(Self {
            inner: Mutex::new(Some(inner)),
        })
    }
}

fn optional_duration(value: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Duration>> {
    required_item(value, key)?
        .extract::<Option<u64>>()
        .map(|value| value.map(Duration::from_micros))
}

fn bounded_watermark_policy(value: &Bound<'_, PyDict>) -> PyResult<calc_flow::WatermarkPolicy> {
    Ok(calc_flow::WatermarkPolicy::BoundedOutOfOrderness {
        event_time_column: required_item(value, "event_time_column")?.extract()?,
        max_out_of_orderness: Duration::from_micros(
            required_item(value, "max_out_of_orderness_micros")?.extract()?,
        ),
        emit_interval: Duration::from_micros(
            required_item(value, "emit_interval_micros")?.extract()?,
        ),
        idle_timeout: optional_duration(value, "idle_timeout_micros")?,
    })
}

fn source_policy(binding: &Bound<'_, PyAny>) -> PyResult<calc_flow::WatermarkPolicy> {
    let value = binding.call_method0("_native_policy")?;
    let value = value.cast::<PyDict>()?;
    match required_item(value, "kind")?.extract::<String>()?.as_str() {
        "source_provided" => Ok(calc_flow::WatermarkPolicy::SourceProvided),
        "bounded_out_of_orderness" => bounded_watermark_policy(value),
        "disabled" => Ok(calc_flow::WatermarkPolicy::Disabled {
            idle_timeout: optional_duration(value, "idle_timeout_micros")?,
        }),
        kind => Err(PyTypeError::new_err(format!(
            "unsupported watermark policy {kind:?}"
        ))),
    }
}

fn build_sources(
    sources: &Bound<'_, PyDict>,
    awaits: &Arc<PythonAwaitRegistry>,
    context: &Arc<Mutex<Option<Arc<PythonAsyncContext>>>>,
    ownership: &Arc<ConnectorOwnership>,
) -> PyResult<PythonSourceBindings> {
    let mut native = BTreeMap::new();
    let mut roots = Vec::new();
    for (source_id, binding) in sources {
        let source_id = source_id
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("source binding IDs must be strings"))?;
        let policy = source_policy(&binding)?;
        let root = Arc::new(PythonRoot::new(binding.unbind()));
        let source = PythonContinuousSource {
            binding: Arc::clone(&root),
            awaits: Arc::clone(awaits),
            capability_error: Arc::new(Mutex::new(None)),
            context: Arc::clone(context),
            _ownership: ownership.retain(),
        };
        native.insert(
            source_id,
            calc_flow::SourceBinding::new(source).with_watermark_policy(policy),
        );
        roots.push(root);
    }
    Ok((native, roots))
}

fn sink_descriptor<'py>(binding: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    binding
        .call_method0("_native_descriptor")?
        .cast_into::<PyDict>()
        .map_err(Into::into)
}

fn sink_retention(descriptor: &Bound<'_, PyDict>) -> PyResult<calc_flow::RetentionClass> {
    match required_item(descriptor, "retention")?
        .extract::<String>()?
        .as_str()
    {
        "bounded" => Ok(calc_flow::RetentionClass::Bounded),
        "unbounded" => Ok(calc_flow::RetentionClass::Unbounded),
        value => Err(PyTypeError::new_err(format!(
            "unsupported retention {value:?}"
        ))),
    }
}

fn native_sink_binding(
    kind: &str,
    descriptor: &Bound<'_, PyDict>,
    sink_id: &str,
    sink: PythonContinuousSink,
) -> PyResult<calc_flow::SinkBinding> {
    let binding = match kind {
        "ordinary" => calc_flow::SinkBinding::ordinary(sink_id, sink),
        "transactional" => calc_flow::SinkBinding::transactional(sink_id, sink),
        "epoch_idempotent" => {
            let mechanism = required_item(descriptor, "mechanism")?.extract::<String>()?;
            calc_flow::SinkBinding::epoch_idempotent(
                sink_id,
                sink,
                &mechanism,
                sink_retention(descriptor)?,
            )
        }
        value => {
            return Err(PyTypeError::new_err(format!(
                "unsupported sink delivery {value:?}"
            )));
        }
    };
    binding.map_err(crate::error::to_py_err)
}

fn build_one_sink(
    binding: Bound<'_, PyAny>,
    awaits: &Arc<PythonAwaitRegistry>,
    context: &Arc<Mutex<Option<Arc<PythonAsyncContext>>>>,
    ownership: &Arc<ConnectorOwnership>,
) -> PyResult<(calc_flow::SinkBinding, Arc<PythonRoot>)> {
    let descriptor = sink_descriptor(&binding)?;
    let kind = required_item(&descriptor, "kind")?.extract::<String>()?;
    let sink_id = binding.getattr("sink_id")?.extract::<String>()?;
    let root = Arc::new(PythonRoot::new(binding.unbind()));
    let sink = PythonContinuousSink {
        binding: Arc::clone(&root),
        awaits: Arc::clone(awaits),
        context: Arc::clone(context),
        _ownership: ownership.retain(),
    };
    let binding = native_sink_binding(&kind, &descriptor, &sink_id, sink)?;
    Ok((binding, root))
}

fn build_sinks(
    sinks: &Bound<'_, PyDict>,
    awaits: &Arc<PythonAwaitRegistry>,
    context: &Arc<Mutex<Option<Arc<PythonAsyncContext>>>>,
    ownership: &Arc<ConnectorOwnership>,
) -> PyResult<PythonSinkBindings> {
    let mut native = BTreeMap::new();
    let mut roots = Vec::new();
    for (output_id, bindings) in sinks {
        let output_id = output_id
            .extract::<String>()
            .map_err(|_| PyTypeError::new_err("sink output IDs must be strings"))?;
        let bindings = if let Ok(values) = bindings.cast::<PyList>() {
            values.iter().collect::<Vec<_>>()
        } else if let Ok(values) = bindings.cast::<PyTuple>() {
            values.iter().collect::<Vec<_>>()
        } else {
            return Err(PyTypeError::new_err(format!(
                "sinks[{output_id:?}] must be a list or tuple"
            )));
        };
        let mut output_bindings = Vec::new();
        for binding in bindings {
            let (binding, root) = build_one_sink(binding, awaits, context, ownership)?;
            output_bindings.push(binding);
            roots.push(root);
        }
        native.insert(output_id, output_bindings);
    }
    Ok((native, roots))
}

fn duration_config(config: &Bound<'_, PyDict>, key: &str) -> PyResult<Duration> {
    required_item(config, key)?
        .extract::<u64>()
        .map(Duration::from_micros)
}

fn edge_budget_config(config: &Bound<'_, PyDict>) -> PyResult<calc_flow::EdgeBudget> {
    calc_flow::EdgeBudget::new(
        required_item(config, "edge_max_rows")?.extract()?,
        required_item(config, "edge_max_bytes")?.extract()?,
    )
    .map_err(crate::error::to_py_err)
}

fn checked_static_array_shape(name: &str, shape: &[usize]) -> PyResult<Vec<u64>> {
    shape
        .iter()
        .map(|dimension| {
            u64::try_from(*dimension).map_err(|_| {
                PyValueError::new_err(format!(
                    "static_inputs.{name}.shape: dimension exceeds the u64 range"
                ))
            })
        })
        .collect()
}

fn latch_static_array_values(
    name: &str,
    backend: &str,
    dtype: &str,
    shape: Vec<u64>,
    values: &[Bound<'_, PyAny>],
) -> PyResult<calc_flow::Batch> {
    let latched = match dtype {
        "bool" => calc_flow::Batch::static_array_bool(
            backend,
            shape,
            None,
            values
                .iter()
                .map(PyAnyMethods::extract::<bool>)
                .collect::<PyResult<Vec<_>>>()?,
        ),
        "int8" | "int16" | "int32" | "int64" => calc_flow::Batch::static_array_int(
            backend,
            dtype,
            shape,
            None,
            values
                .iter()
                .map(PyAnyMethods::extract::<i64>)
                .collect::<PyResult<Vec<_>>>()?,
        ),
        "uint8" | "uint16" | "uint32" | "uint64" => calc_flow::Batch::static_array_uint(
            backend,
            dtype,
            shape,
            None,
            values
                .iter()
                .map(PyAnyMethods::extract::<u64>)
                .collect::<PyResult<Vec<_>>>()?,
        ),
        "float32" | "float64" => calc_flow::Batch::static_array_float(
            backend,
            dtype,
            shape,
            None,
            values
                .iter()
                .map(PyAnyMethods::extract::<f64>)
                .collect::<PyResult<Vec<_>>>()?,
        ),
        other => {
            return Err(PyValueError::new_err(format!(
                "static_inputs.{name}.dtype: array dtype {other:?} is outside the digest-v1 set"
            )));
        }
    };
    latched.map_err(|error| PyValueError::new_err(format!("static_inputs.{name}: {error}")))
}

fn static_array_object(name: &str, batch: &calc_flow::Batch) -> PyResult<(String, Py<PyAny>)> {
    let payload = batch.external_payload().map_err(|_| {
        PyTypeError::new_err(format!(
            "static_inputs.{name}: table batches do not carry an array payload"
        ))
    })?;
    let backend = payload.backend().to_owned();
    let object = payload
        .as_any()
        .downcast_ref::<crate::batch::PythonPayload>()
        .map(|payload| Python::attach(|py| payload.object.clone_ref(py)))
        .ok_or_else(|| {
            PyTypeError::new_err(format!(
                "static_inputs.{name}.backend: array static inputs must be latched engine-owned values"
            ))
        })?;
    Ok((backend, object))
}

fn static_array_descriptor(name: &str, array: &Bound<'_, PyAny>) -> PyResult<(String, Vec<u64>)> {
    let dtype = array
        .getattr(intern!(array.py(), "dtype"))?
        .str()?
        .extract()?;
    let shape: Vec<usize> = array.getattr(intern!(array.py(), "shape"))?.extract()?;
    Ok((dtype, checked_static_array_shape(name, &shape)?))
}

fn flattened_static_array_values<'py>(
    array: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    array
        .call_method0(intern!(array.py(), "ravel"))?
        .call_method0(intern!(array.py(), "tolist"))?
        .extract()
}

fn latch_attached_static_array(
    name: &str,
    backend: &str,
    array: &Bound<'_, PyAny>,
) -> PyResult<calc_flow::Batch> {
    let (dtype, shape) = static_array_descriptor(name, array)?;
    let values = flattened_static_array_values(array)?;
    latch_static_array_values(name, backend, &dtype, shape, &values)
}

/// Latches one Python host array batch into engine-owned storage (SCE-11).
///
/// This is the trusted provider boundary of API note section 7.3: the
/// declared backend, dtype, and shape are extracted from the host array and
/// the logical C-order values are copied out of caller-mutable memory, so
/// the returned batch can never alias it.
fn latch_static_array(name: &str, batch: &calc_flow::Batch) -> PyResult<calc_flow::Batch> {
    let (backend, object) = static_array_object(name, batch)?;
    Python::attach(|py| latch_attached_static_array(name, &backend, object.bind(py)))
}

/// Converts the adapter-supplied static input mapping into engine-owned
/// batches. Table batches pass through unchanged; array batches are
/// snapshotted out of Python memory at this seam.
fn build_static_inputs(
    static_inputs: &Bound<'_, PyDict>,
) -> PyResult<BTreeMap<String, calc_flow::Batch>> {
    let mut converted = BTreeMap::new();
    for (key, value) in static_inputs {
        let name: String = key.extract()?;
        let batch = value
            .extract::<PyRef<'_, PyBatch>>()
            .map_err(|_| {
                PyTypeError::new_err("static_inputs must be a mapping of calc_flow.Batch values")
            })?
            .clone_inner()?;
        let batch = if batch.kind() == calc_flow::BatchKind::Array {
            latch_static_array(&name, &batch)?
        } else {
            batch
        };
        converted.insert(name, batch);
    }
    Ok(converted)
}

fn runtime_config(config: &Bound<'_, PyDict>) -> PyResult<calc_flow::StreamRuntimeConfig> {
    Ok(calc_flow::StreamRuntimeConfig {
        checkpoint_interval: duration_config(config, "checkpoint_interval_micros")?,
        checkpoint_timeout: duration_config(config, "checkpoint_timeout_micros")?,
        edge_budget: edge_budget_config(config)?,
        retained_epochs: required_item(config, "retained_epochs")?.extract()?,
    })
}

#[pymethods]
impl PyContinuousStreamingRunner {
    #[new]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3 constructor extraction owns its PyRef boundary values"
    )]
    fn new(
        plan: PyRef<'_, PyStreamExecutionPlan>,
        sources: &Bound<'_, PyDict>,
        sinks: &Bound<'_, PyDict>,
        checkpoints: PyRef<'_, PyManagedCheckpointRuntime>,
        config: &Bound<'_, PyDict>,
        static_inputs: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let awaits = Arc::new(PythonAwaitRegistry::new());
        let context = Arc::new(Mutex::new(None));
        let ownership = Arc::new(ConnectorOwnership::new());
        let (sources, mut roots) = build_sources(sources, &awaits, &context, &ownership)?;
        let (sinks, sink_roots) = build_sinks(sinks, &awaits, &context, &ownership)?;
        roots.extend(sink_roots);
        let config = runtime_config(config)?;
        let (plan, plan_owner) = plan.take()?;
        roots.push(Arc::new(PythonRoot::new(plan_owner)));
        let checkpoints = checkpoints.take()?;
        let static_inputs = build_static_inputs(static_inputs)?;
        let runner = calc_flow::StreamingRunner::new(plan, sources, sinks, checkpoints)
            .and_then(|runner| runner.with_runtime_config(config))
            .and_then(|runner| runner.with_static_inputs(static_inputs))
            .map_err(streaming_py_err)?;
        Ok(Self {
            inner: Arc::new(RunnerStartState {
                runner: Mutex::new(Some(runner)),
                roots: Mutex::new(roots),
                awaits,
                context,
                ownership,
            }),
        })
    }

    fn start_async(&self, py: Python<'_>) -> PyResult<Py<PyStreamingStartAwaitable>> {
        self.inner
            .context
            .lock()
            .replace(Arc::new(PythonAsyncContext::capture(py)?));
        Py::new(
            py,
            PyStreamingStartAwaitable {
                inner: Arc::clone(&self.inner),
                started: AtomicBool::new(false),
            },
        )
    }

    fn __repr__(&self) -> String {
        let consumed = self.inner.runner.lock().is_none();
        format!("<calc_flow._native._StreamingRunner consumed={consumed}>")
    }

    #[allow(clippy::needless_pass_by_value)]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for root in self.inner.roots.lock().iter() {
            visit.call(root.object())?;
        }
        if let Some(context) = self.inner.context.lock().as_ref() {
            context.traverse(&visit)?;
        }
        Ok(())
    }

    fn __clear__(&self) {
        let runner = self.inner.runner.lock().take();
        self.inner.roots.lock().clear();
        if runner.is_some() {
            drop(self.inner.context.lock().take());
        }
        drop(runner);
    }

    fn _release_roots(&self) {
        self.inner.roots.lock().clear();
    }

    fn _wait_start_cleanup_async<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ownership = Arc::clone(&self.inner.ownership);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            ownership.wait_idle().await;
            Ok(())
        })
    }
}

#[pymethods]
impl PyStreamingStartAwaitable {
    fn __await__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.started.swap(true, Ordering::AcqRel) {
            return Err(PyRuntimeError::new_err(
                "streaming start awaitable has already been awaited",
            ));
        }
        let runner = self.inner.runner.lock().take().ok_or_else(|| {
            PyRuntimeError::new_err("streaming runner has already been consumed by start()")
        })?;
        let state = Arc::clone(&self.inner);
        let observer = observer_result_future(py, async move {
            let job = runner.start().await.map_err(streaming_py_err)?;
            let roots = state.roots.lock().clone();
            let awaits = Arc::clone(&state.awaits);
            let context = Arc::clone(&state.context);
            Python::attach(|py| {
                Py::new(
                    py,
                    PyStreamingJob::from_inner_with_roots(job, roots, awaits, context),
                )
            })
        })?;
        resolved_observer_await(py, observer)
    }
}

impl PyStreamingJob {
    fn from_inner_with_roots(
        inner: calc_flow::StreamingJob,
        roots: Vec<Arc<PythonRoot>>,
        awaits: Arc<PythonAwaitRegistry>,
        context: Arc<Mutex<Option<Arc<PythonAsyncContext>>>>,
    ) -> Self {
        Self {
            inner: Mutex::new(Some(Arc::new(inner))),
            roots: Arc::new(Mutex::new(roots)),
            awaits,
            context,
        }
    }

    fn job(&self) -> PyResult<Arc<calc_flow::StreamingJob>> {
        self.inner
            .lock()
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("StreamingJob has been cleared"))
    }
}

#[pymethods]
impl PyStreamingJob {
    #[getter]
    fn id(&self) -> PyResult<u64> {
        self.job().map(|job| job.id())
    }

    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let job = self.job()?;
        let status = job_status_to_py(py, &job.status())?;
        let joins = job.stream_join_status();
        let join_values = stream_join_status_to_py(py, &joins)?;
        status.set_item("stream_joins", join_values)?;
        Ok(status)
    }

    fn trigger_checkpoint_async(&self, py: Python<'_>) -> PyResult<Py<PyStreamingJobAwaitable>> {
        self.operation_awaitable(py, JobOperation::TriggerCheckpoint)
    }

    fn shutdown_async(&self, py: Python<'_>) -> PyResult<Py<PyStreamingJobAwaitable>> {
        self.operation_awaitable(py, JobOperation::Shutdown)
    }

    fn cancel_async(&self, py: Python<'_>) -> PyResult<Py<PyStreamingJobAwaitable>> {
        self.operation_awaitable(py, JobOperation::Cancel)
    }

    fn wait_async(&self, py: Python<'_>) -> PyResult<Py<PyStreamingJobAwaitable>> {
        self.operation_awaitable(py, JobOperation::Wait)
    }

    fn __repr__(&self) -> String {
        self.job().map_or_else(
            |_| "<calc_flow._native._StreamingJob cleared=True>".into(),
            |job| {
                format!(
                    "<calc_flow._native._StreamingJob id={} state={}>",
                    job.id(),
                    job_state_name(job.status().state)
                )
            },
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for root in self.roots.lock().iter() {
            visit.call(root.object())?;
        }
        if let Some(context) = self.context.lock().as_ref() {
            context.traverse(&visit)?;
        }
        Ok(())
    }

    fn __clear__(&self) {
        drop(self.inner.lock().take());
        self.roots.lock().clear();
    }

    fn _release_roots(&self) {
        self.roots.lock().clear();
        drop(self.context.lock().take());
    }
}

impl PyStreamingJob {
    fn operation_awaitable(
        &self,
        py: Python<'_>,
        operation: JobOperation,
    ) -> PyResult<Py<PyStreamingJobAwaitable>> {
        Py::new(
            py,
            PyStreamingJobAwaitable {
                inner: self.job()?,
                awaits: Arc::clone(&self.awaits),
                operation,
                started: AtomicBool::new(false),
            },
        )
    }
}

#[pymethods]
impl PyStreamingJobAwaitable {
    fn __await__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.started.swap(true, Ordering::AcqRel) {
            return Err(PyRuntimeError::new_err(
                "streaming job awaitable has already been awaited",
            ));
        }
        let job = Arc::clone(&self.inner);
        let awaits = Arc::clone(&self.awaits);
        let observer = match self.operation {
            JobOperation::TriggerCheckpoint => persistent_future(py, async move {
                job.trigger_checkpoint()
                    .await
                    .map(calc_flow::Epoch::as_u64)
                    .map_err(streaming_py_err)
            })?,
            JobOperation::Shutdown => persistent_future(py, async move {
                let outcome = job.shutdown().await;
                awaits.wait_idle().await;
                Python::attach(|py| job_outcome_to_py(py, &outcome).map(Bound::unbind))
            })?,
            JobOperation::Cancel => persistent_future(py, async move {
                let outcome = job.cancel().await;
                awaits.wait_idle().await;
                Python::attach(|py| job_outcome_to_py(py, &outcome).map(Bound::unbind))
            })?,
            JobOperation::Wait => observer_result_future(py, async move {
                let outcome = job.wait().await;
                awaits.wait_idle().await;
                Python::attach(|py| job_outcome_to_py(py, &outcome).map(Bound::unbind))
            })?,
        };
        resolved_observer_await(py, observer)
    }
}

fn resolved_observer_await<'py>(
    py: Python<'py>,
    observer: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let resolver = OBSERVER_RESULT_RESOLVER.get_or_try_init(py, || {
        let namespace = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                "async def resolve_observer_result(observer):\n    succeeded, value = await observer\n    if succeeded:\n        return value\n    try:\n        raise value\n    except BaseException as error:\n        BaseException.__context__.__set__(error, None)\n        raise"
            ),
            Some(&namespace),
            Some(&namespace),
        )?;
        namespace
            .get_item("resolve_observer_result")?
            .ok_or_else(|| {
                PyRuntimeError::new_err(
                    "failed to initialize streaming observer resolver",
                )
            })
            .map(Bound::unbind)
    })?;
    resolver
        .bind(py)
        .call1((observer,))?
        .call_method0("__await__")
}

fn observer_result_future<'py, F, T>(py: Python<'py>, future: F) -> PyResult<Bound<'py, PyAny>>
where
    F: Future<Output = PyResult<T>> + Send + 'static,
    T: for<'a> IntoPyObject<'a> + Send + 'static,
{
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let result = future.await;
        Python::attach(|py| match result {
            Ok(value) => py_tuple!(py, [true, value]).map(Bound::unbind),
            Err(error) => py_tuple!(py, [false, error.value(py)]).map(Bound::unbind),
        })
    })
}

fn persistent_future<'py, F, T>(py: Python<'py>, future: F) -> PyResult<Bound<'py, PyAny>>
where
    F: Future<Output = PyResult<T>> + Send + 'static,
    T: for<'a> IntoPyObject<'a> + Send + 'static,
{
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let observer = observer_result_future(py, async move {
        receiver
            .await
            .map_err(|_| internal_streaming_py_err("streaming lifecycle task failed"))?
    })?;
    pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
        let result = future.await;
        let _ = sender.send(result);
    });
    Ok(observer)
}

fn streaming_py_err(error: calc_flow::CalcFlowError) -> PyErr {
    match error {
        calc_flow::CalcFlowError::Streaming(error) => streaming_error_to_py_err(&error),
        _ => internal_streaming_py_err("streaming runtime failed"),
    }
}

fn streaming_error_to_py_err(error: &calc_flow::StreamingError) -> PyErr {
    let fields = SafeStreamingErrorFields::from_streaming(error);
    let checkpoint_publication_unknown =
        error.category() == calc_flow::StreamingErrorCategory::CheckpointPublicationUnknown;
    structured_streaming_py_err(fields, checkpoint_publication_unknown)
}

fn internal_streaming_py_err(message: &str) -> PyErr {
    structured_streaming_py_err(SafeStreamingErrorFields::internal(message), false)
}

fn structured_streaming_py_err(
    fields: SafeStreamingErrorFields,
    checkpoint_publication_unknown: bool,
) -> PyErr {
    let fallback_message = fields.message.clone();
    Python::attach(|py| -> PyResult<PyErr> {
        let exception_type = if checkpoint_publication_unknown {
            checkpoint_publication_unknown_error_type(py)?
        } else {
            streaming_runtime_error_type(py)?
        };
        let exception = PyErr::from_type(exception_type, fallback_message.clone());
        let values = py_tuple!(
            py,
            [
                fields.category,
                fields.reason_code,
                fields.message,
                fields.job_id,
                fields.epoch,
                fields.checkpoint_phase,
                fields.component_kind,
                fields.component_id,
                fields.diagnostic_id,
                fields.position,
            ]
        )?;
        exception
            .value(py)
            .setattr(NATIVE_EXCEPTION_STORAGE, values)?;
        exception.set_cause(py, None);
        Ok(exception)
    })
    .unwrap_or_else(|_| crate::error::CalcFlowError::new_err(fallback_message))
}

fn streaming_runtime_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    STREAMING_RUNTIME_ERROR_TYPE
        .get_or_try_init(py, || {
            let namespace = structured_exception_namespace(py)?;
            create_exception_type(
                py,
                "StreamingRuntimeError",
                &py.get_type::<crate::error::CalcFlowError>(),
                namespace,
            )
        })
        .map(|exception_type| exception_type.bind(py).clone())
}

fn checkpoint_publication_unknown_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    CHECKPOINT_PUBLICATION_UNKNOWN_ERROR_TYPE
        .get_or_try_init(py, || {
            let namespace = PyDict::new(py);
            namespace.set_item("__slots__", PyTuple::empty(py))?;
            create_exception_type(
                py,
                "CheckpointPublicationUnknownError",
                &streaming_runtime_error_type(py)?,
                namespace,
            )
        })
        .map(|exception_type| exception_type.bind(py).clone())
}

fn create_exception_type(
    py: Python<'_>,
    name: &str,
    base: &Bound<'_, PyType>,
    namespace: Bound<'_, PyDict>,
) -> PyResult<Py<PyType>> {
    namespace.set_item("__module__", "calc_flow._native")?;
    let bases = PyTuple::new(py, [base])?;
    py.import("builtins")?
        .getattr("type")?
        .call1((name, bases, namespace))?
        .cast_into::<PyType>()
        .map(Bound::unbind)
        .map_err(Into::into)
}

fn structured_exception_namespace(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let namespace = PyDict::new(py);
    set_exception_storage_slot(py, &namespace)?;
    set_exception_attribute_guards(py, &namespace)?;
    for (index, field) in SAFE_EXCEPTION_FIELDS.into_iter().enumerate() {
        set_exception_field_property(py, &namespace, field, index)?;
    }
    set_exception_storage_property(py, &namespace)?;
    Ok(namespace)
}

fn set_exception_storage_slot(py: Python<'_>, namespace: &Bound<'_, PyDict>) -> PyResult<()> {
    namespace.set_item("__slots__", PyTuple::new(py, [NATIVE_EXCEPTION_STORAGE])?)
}

fn set_exception_attribute_guards(py: Python<'_>, namespace: &Bound<'_, PyDict>) -> PyResult<()> {
    let property = py.import("builtins")?.getattr("property")?;
    namespace.set_item("__setattr__", property.call1((exception_setattr(py)?,))?)?;
    namespace.set_item("__delattr__", property.call1((exception_delattr(py)?,))?)
}

fn set_exception_field_property(
    py: Python<'_>,
    namespace: &Bound<'_, PyDict>,
    field: &str,
    index: usize,
) -> PyResult<()> {
    let getter = exception_field_getter(py, index)?;
    set_exception_property(py, namespace, field, &getter)
}

fn set_exception_storage_property(py: Python<'_>, namespace: &Bound<'_, PyDict>) -> PyResult<()> {
    let getter = exception_storage_getter(py)?;
    set_exception_property(py, namespace, SAFE_EXCEPTION_STORAGE, &getter)
}

fn set_exception_property(
    py: Python<'_>,
    namespace: &Bound<'_, PyDict>,
    name: &str,
    getter: &Bound<'_, PyCFunction>,
) -> PyResult<()> {
    let property = py.import("builtins")?.getattr("property")?;
    namespace.set_item(name, property.call1((getter,))?)
}

fn exception_setattr(py: Python<'_>) -> PyResult<Bound<'_, PyCFunction>> {
    PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
        let instance = args.get_item(0)?.unbind();
        PyCFunction::new_closure(args.py(), None, None, move |args, _kwargs| {
            let instance = instance.bind(args.py());
            let name = args.get_item(0)?;
            if name.extract::<&str>()? == NATIVE_EXCEPTION_STORAGE
                && instance.getattr(NATIVE_EXCEPTION_STORAGE).is_ok()
            {
                return Err(pyo3::exceptions::PyAttributeError::new_err(
                    "native safe exception backing is read-only",
                ));
            }
            let value = args.get_item(1)?;
            args.py()
                .get_type::<pyo3::exceptions::PyBaseException>()
                .getattr("__setattr__")?
                .call1((instance, name, value))
                .map(Bound::unbind)
        })
        .map(Bound::unbind)
    })
}

fn exception_delattr(py: Python<'_>) -> PyResult<Bound<'_, PyCFunction>> {
    PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
        let instance = args.get_item(0)?.unbind();
        PyCFunction::new_closure(args.py(), None, None, move |args, _kwargs| {
            let instance = instance.bind(args.py());
            let name = args.get_item(0)?;
            if name.extract::<&str>()? == NATIVE_EXCEPTION_STORAGE {
                return Err(pyo3::exceptions::PyAttributeError::new_err(
                    "native safe exception backing is read-only",
                ));
            }
            args.py()
                .get_type::<pyo3::exceptions::PyBaseException>()
                .getattr("__delattr__")?
                .call1((instance, name))
                .map(Bound::unbind)
        })
        .map(Bound::unbind)
    })
}

fn exception_field_getter(py: Python<'_>, index: usize) -> PyResult<Bound<'_, PyCFunction>> {
    PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
        let instance = args.get_item(0)?;
        instance
            .getattr(NATIVE_EXCEPTION_STORAGE)?
            .get_item(index)
            .map(Bound::unbind)
    })
}

fn exception_storage_getter(py: Python<'_>) -> PyResult<Bound<'_, PyCFunction>> {
    PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
        let instance = args.get_item(0)?;
        let values = instance.getattr(NATIVE_EXCEPTION_STORAGE)?;
        let fields = PyDict::new(args.py());
        for (index, field) in SAFE_EXCEPTION_FIELDS.into_iter().enumerate() {
            fields.set_item(field, values.get_item(index)?)?;
        }
        args.py()
            .import("types")?
            .getattr("MappingProxyType")?
            .call1((fields,))
            .map(Bound::unbind)
    })
}

fn job_outcome_to_py<'py>(
    py: Python<'py>,
    outcome: &calc_flow::JobOutcome,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("state", job_state_name(outcome.state))?;
    value.set_item("cause", terminal_cause_name(outcome.cause))?;
    value.set_item(
        "completed_epoch",
        outcome.completed_epoch.map(calc_flow::Epoch::as_u64),
    )?;
    let errors = outcome
        .errors
        .iter()
        .map(|error| streaming_error_value_to_py(py, error).map(Bound::unbind))
        .collect::<PyResult<Vec<_>>>()?;
    value.set_item("errors", PyTuple::new(py, errors)?)?;
    Ok(value)
}

fn streaming_error_value_to_py<'py>(
    py: Python<'py>,
    error: &calc_flow::StreamingError,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    set_py_items!(value, {
        "category" => streaming_error_category_name(error.category()),
        "reason_code" => error.reason_code().and_then(streaming_failure_reason_name),
        "message" => error.message(),
        "job_id" => error.job_id(),
        "epoch" => error.epoch().map(calc_flow::Epoch::as_u64),
        "checkpoint_phase" => error.checkpoint_phase().map(checkpoint_phase_name),
        "component_kind" => error.component_kind().map(component_kind_name),
        "component_id" => error.component_id(),
        "diagnostic_id" => error.diagnostic_id(),
        "position" => error.position(),
    });
    Ok(value)
}

fn job_status_to_py<'py>(
    py: Python<'py>,
    status: &calc_flow::JobStatus,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    set_py_items!(value, {
        "job_id" => status.job_id,
        "state" => job_state_name(status.state),
        "terminal_cause" => status.terminal_cause.map(terminal_cause_name),
        "delivery" => delivery_status_to_py(py, &status.delivery)?,
        "task_count" => status.task_count,
        "task_errors" => status.task_errors,
        "metrics_overflowed" => status.metrics_overflowed,
        "watermark_micros" => status.watermark.map(calc_flow::EventTime::as_micros),
        "edges" => edge_status_to_py(py, &status.edges)?,
        "sources" => source_status_to_py(py, &status.sources)?,
        "operators" => operator_status_to_py(py, &status.operators)?,
        "sinks" => sink_status_to_py(py, &status.sinks)?,
        "checkpoint" => checkpoint_status_to_py(py, &status.checkpoint)?,
    });
    Ok(value)
}

/// Projects the per-node Join status mapping (api note "Payload-free Join
/// status").
fn stream_join_status_to_py<'py>(
    py: Python<'py>,
    statuses: &BTreeMap<String, calc_flow::StreamJoinStatus>,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    for (node_id, status) in statuses {
        values.set_item(node_id, stream_join_status_value_to_py(py, status)?)?;
    }
    Ok(values)
}

fn stream_join_status_value_to_py<'py>(
    py: Python<'py>,
    status: &calc_flow::StreamJoinStatus,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    set_py_items!(value, {
        "left" => stream_join_side_to_py(py, &status.left)?,
        "right" => stream_join_side_to_py(py, &status.right)?,
        "emitted_match_rows" => status.emitted_match_rows,
        "state_limit_failures" => status.state_limit_failures,
        "match_limit_failures" => status.match_limit_failures,
    });
    Ok(value)
}

fn stream_join_side_to_py<'py>(
    py: Python<'py>,
    side: &calc_flow::StreamJoinSideStatus,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    set_py_items!(value, {
        "retained_rows" => side.retained_rows,
        "retained_bytes" => side.retained_bytes,
        "evicted_rows" => side.evicted_rows,
        "late_rows" => side.late_rows,
        "late_affected_batches" => side.late_affected_batches,
        "max_lateness_micros" => side.max_lateness.map(duration_micros),
        "null_event_time_rows" => side.null_event_time_rows,
        "null_key_rows" => side.null_key_rows,
    });
    Ok(value)
}

fn delivery_status_to_py<'py>(
    py: Python<'py>,
    statuses: &BTreeMap<String, calc_flow::OutputDeliveryStatus>,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    for (output_id, status) in statuses {
        let value = PyDict::new(py);
        value.set_item("requested", delivery_guarantee_name(status.requested))?;
        value.set_item("effective", delivery_guarantee_name(status.effective))?;
        values.set_item(output_id, value)?;
    }
    Ok(values)
}

fn edge_status_to_py<'py>(
    py: Python<'py>,
    statuses: &BTreeMap<String, calc_flow::EdgeStatus>,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    for (edge_id, status) in statuses {
        let value = PyDict::new(py);
        set_py_items!(value, {
            "current_envelopes" => status.current_envelopes,
            "current_rows" => status.current_rows,
            "current_bytes" => status.current_bytes,
            "high_water_envelopes" => status.high_water_envelopes,
            "high_water_rows" => status.high_water_rows,
            "high_water_bytes" => status.high_water_bytes,
            "blocked_sends" => status.blocked_sends,
            "blocked_duration_micros" => duration_micros(status.blocked_duration),
            "envelope_limit" => status.envelope_limit,
            "row_limit" => status.row_limit,
            "byte_limit" => status.byte_limit,
        });
        values.set_item(edge_id, value)?;
    }
    Ok(values)
}

fn source_status_to_py<'py>(
    py: Python<'py>,
    statuses: &BTreeMap<String, calc_flow::SourceStatus>,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    for (source_id, status) in statuses {
        let value = PyDict::new(py);
        set_py_items!(value, {
            "replay_positioning" => replay_positioning_name(status.replay_positioning),
            "delivery" => source_delivery_name(status.delivery),
            "max_batch_rows" => status.max_batch_rows,
            "max_batch_bytes" => status.max_batch_bytes,
            "next_sequence" => status.next_sequence,
            "ended" => status.ended,
            "polls" => status.polls,
            "data_batches" => status.data_batches,
            "data_rows" => status.data_rows,
            "data_bytes" => status.data_bytes,
            "fanned_out_batches" => status.fanned_out_batches,
            "fanned_out_rows" => status.fanned_out_rows,
            "fanned_out_bytes" => status.fanned_out_bytes,
            "errors" => status.errors,
        });
        values.set_item(source_id, value)?;
    }
    Ok(values)
}

fn operator_status_to_py<'py>(
    py: Python<'py>,
    statuses: &BTreeMap<String, calc_flow::OperatorStatus>,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    for (operator_id, status) in statuses {
        let value = PyDict::new(py);
        set_py_items!(value, {
            "input_batches" => status.input_batches,
            "input_rows" => status.input_rows,
            "input_bytes" => status.input_bytes,
            "fanned_out_batches" => status.fanned_out_batches,
            "fanned_out_rows" => status.fanned_out_rows,
            "fanned_out_bytes" => status.fanned_out_bytes,
            "processing_duration_micros" => duration_micros(status.processing_duration),
            "errors" => status.errors,
            "ended" => status.ended,
            "late_rows" => status.late_rows,
            "late_affected_batches" => status.late_affected_batches,
            "max_lateness_micros" => status.max_lateness.map(duration_micros),
            "null_event_time_rows" => status.null_event_time_rows,
            "null_event_time_batches" => status.null_event_time_batches,
            "datafusion_runtime_created" => status.datafusion_runtime_created,
        });
        values.set_item(operator_id, value)?;
    }
    Ok(values)
}

fn sink_status_to_py<'py>(
    py: Python<'py>,
    statuses: &BTreeMap<String, calc_flow::SinkStatus>,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    for (sink_id, status) in statuses {
        let value = PyDict::new(py);
        set_py_items!(value, {
            "output_id" => &status.output_id,
            "effective_delivery" => sink_delivery_to_py(py, &status.effective_delivery)?,
            "delivered_batches" => status.delivered_batches,
            "delivered_rows" => status.delivered_rows,
            "delivered_bytes" => status.delivered_bytes,
            "write_duration_micros" => duration_micros(status.write_duration),
            "errors" => status.errors,
            "ended" => status.ended,
        });
        values.set_item(sink_id, value)?;
    }
    Ok(values)
}

fn checkpoint_status_to_py<'py>(
    py: Python<'py>,
    status: &calc_flow::CheckpointStatus,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    set_py_items!(value, {
        "current_epoch" => status.current_epoch.map(calc_flow::Epoch::as_u64),
        "phase" => status.phase.map(checkpoint_phase_name),
        "terminal" => status.terminal,
        "source_acknowledgements" => status.source_acknowledgements,
        "expected_sources" => status.expected_sources,
        "operator_acknowledgements" => status.operator_acknowledgements,
        "expected_operators" => status.expected_operators,
        "sink_precommit_acknowledgements" => status.sink_precommit_acknowledgements,
        "expected_sink_precommits" => status.expected_sink_precommits,
        "sink_commit_acknowledgements" => status.sink_commit_acknowledgements,
        "expected_sink_commits" => status.expected_sink_commits,
        "elapsed_micros" => status.elapsed.map(duration_micros),
        "last_completed_epoch" => status.last_completed_epoch.map(calc_flow::Epoch::as_u64),
        "installed_unknown_epoch" => status.installed_unknown_epoch.map(calc_flow::Epoch::as_u64),
        "failure_category" => status.failure_category.map(streaming_error_category_name),
        "runtime_config_changed" => status.runtime_config_changed,
    });
    Ok(value)
}

fn sink_delivery_to_py<'py>(
    py: Python<'py>,
    delivery: &calc_flow::SinkDelivery,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    match delivery {
        calc_flow::SinkDelivery::Ordinary => value.set_item("kind", "ordinary")?,
        calc_flow::SinkDelivery::EpochIdempotent {
            mechanism,
            retention,
        } => {
            value.set_item("kind", "epoch_idempotent")?;
            value.set_item("mechanism", mechanism)?;
            value.set_item(
                "retention",
                match retention {
                    calc_flow::RetentionClass::Bounded => "bounded",
                    calc_flow::RetentionClass::Unbounded => "unbounded",
                },
            )?;
        }
        calc_flow::SinkDelivery::Transactional => {
            value.set_item("kind", "transactional")?;
        }
    }
    Ok(value)
}

const fn duration_micros(duration: Duration) -> u128 {
    duration.as_micros()
}

const fn job_state_name(state: calc_flow::JobState) -> &'static str {
    match state {
        calc_flow::JobState::Running => "running",
        calc_flow::JobState::Draining => "draining",
        calc_flow::JobState::Completed => "completed",
        calc_flow::JobState::Cancelled => "cancelled",
        calc_flow::JobState::Failed => "failed",
        calc_flow::JobState::RecoveryRequired => "recovery_required",
    }
}

const fn terminal_cause_name(cause: calc_flow::TerminalCause) -> &'static str {
    match cause {
        calc_flow::TerminalCause::NaturalEnd => "natural_end",
        calc_flow::TerminalCause::GracefulShutdown => "graceful_shutdown",
        calc_flow::TerminalCause::ExplicitCancel => "explicit_cancel",
        calc_flow::TerminalCause::DeadlineExceeded => "deadline_exceeded",
        calc_flow::TerminalCause::Failure => "failure",
    }
}

const fn streaming_error_category_name(
    category: calc_flow::StreamingErrorCategory,
) -> &'static str {
    match category {
        calc_flow::StreamingErrorCategory::Validation => "validation",
        calc_flow::StreamingErrorCategory::Compile => "compile",
        calc_flow::StreamingErrorCategory::Conflict => "conflict",
        calc_flow::StreamingErrorCategory::Cancelled => "cancelled",
        calc_flow::StreamingErrorCategory::CheckpointTimeout => "checkpoint_timeout",
        calc_flow::StreamingErrorCategory::CheckpointMismatch => "checkpoint_mismatch",
        calc_flow::StreamingErrorCategory::CheckpointPublicationUnknown => {
            "checkpoint_publication_unknown"
        }
        calc_flow::StreamingErrorCategory::Io => "io",
        calc_flow::StreamingErrorCategory::Operator => "operator",
        calc_flow::StreamingErrorCategory::Connector => "connector",
        calc_flow::StreamingErrorCategory::TaskPanicked => "task_panicked",
        calc_flow::StreamingErrorCategory::Internal => "internal",
    }
}

const fn streaming_failure_reason_name(
    reason: calc_flow::StreamingFailureReason,
) -> Option<&'static str> {
    match reason {
        calc_flow::StreamingFailureReason::JoinStateLimitExceeded => {
            Some("join_state_limit_exceeded")
        }
        calc_flow::StreamingFailureReason::JoinMatchLimitExceeded => {
            Some("join_match_limit_exceeded")
        }
        calc_flow::StreamingFailureReason::JoinCounterOverflow => Some("join_counter_overflow"),
        calc_flow::StreamingFailureReason::JoinTimeConversionFailed => {
            Some("join_time_conversion_failed")
        }
        _ => None,
    }
}

const fn checkpoint_phase_name(phase: calc_flow::CheckpointPhase) -> &'static str {
    match phase {
        calc_flow::CheckpointPhase::Requested => "requested",
        calc_flow::CheckpointPhase::SourcesCut => "sources_cut",
        calc_flow::CheckpointPhase::OperatorsSnapshotted => "operators_snapshotted",
        calc_flow::CheckpointPhase::SinksPrecommitted => "sinks_precommitted",
        calc_flow::CheckpointPhase::ManifestInstalled => "manifest_installed",
        calc_flow::CheckpointPhase::ManifestDurable => "manifest_durable",
        calc_flow::CheckpointPhase::SinksCommitted => "sinks_committed",
        calc_flow::CheckpointPhase::Completed => "completed",
    }
}

const fn component_kind_name(kind: calc_flow::ComponentKind) -> &'static str {
    match kind {
        calc_flow::ComponentKind::Job => "job",
        calc_flow::ComponentKind::Edge => "edge",
        calc_flow::ComponentKind::Source => "source",
        calc_flow::ComponentKind::Operator => "operator",
        calc_flow::ComponentKind::Sink => "sink",
        calc_flow::ComponentKind::Checkpoint => "checkpoint",
    }
}

const fn replay_positioning_name(capability: calc_flow::ReplayPositioning) -> &'static str {
    match capability {
        calc_flow::ReplayPositioning::ExactPauseReportAndSeek => "exact_pause_report_and_seek",
        calc_flow::ReplayPositioning::Unsupported => "unsupported",
    }
}

const fn source_delivery_name(capability: calc_flow::SourceDeliveryCapability) -> &'static str {
    match capability {
        calc_flow::SourceDeliveryCapability::Lossless => "lossless",
        calc_flow::SourceDeliveryCapability::Lossy => "lossy",
    }
}

const fn delivery_guarantee_name(guarantee: calc_flow::DeliveryGuarantee) -> &'static str {
    match guarantee {
        calc_flow::DeliveryGuarantee::BestEffort => "best_effort",
        calc_flow::DeliveryGuarantee::AtLeastOnce => "at_least_once",
        calc_flow::DeliveryGuarantee::ExactlyOnce => "exactly_once",
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PyManagedCheckpointRuntime>()?;
    module.add_class::<PyContinuousStreamingRunner>()?;
    module.add_class::<PyStreamingJob>()?;
    let streaming_runtime_error = streaming_runtime_error_type(py)?;
    module.add("StreamingRuntimeError", streaming_runtime_error)?;
    let checkpoint_publication_unknown = checkpoint_publication_unknown_error_type(py)?;
    module.add(
        "CheckpointPublicationUnknownError",
        checkpoint_publication_unknown,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        ffi::CString,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };

    use async_trait::async_trait;
    use calc_flow::{
        Batch, CalcFlowError, Cursor, ExpressionOperator, ManagedCheckpointRuntime,
        NativeWatermarkCapability, PipelineBuilder, ReplayPositioning, Result, SinkBinding,
        SinkRecovery, SourceBinding, SourceCapabilities, SourceDeliveryCapability, SourceEvent,
        SourceSchema, StreamExecutionPlan, StreamRequirements, StreamSink, StreamSource,
        StreamingRunner, TransactionalStreamSink, UdfRegistry,
    };
    use datafusion::arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use pyo3::{
        Bound, Py, Python, pyclass, pymethods,
        types::{PyAnyMethods, PyDict, PyDictMethods},
    };

    use super::{
        PyContinuousStreamingRunner, PyManagedCheckpointRuntime, checked_static_array_shape,
    };
    use crate::{batch::PyBatch, pipeline::PyStreamExecutionPlan};

    struct PendingSource {
        closed: Arc<AtomicBool>,
    }

    #[test]
    fn checked_static_array_shape_preserves_dimensions() {
        assert_eq!(
            checked_static_array_shape("weights", &[2, 3]).unwrap(),
            vec![2_u64, 3_u64]
        );
    }

    struct FailingOpenSource;

    struct BlockedOpenSource {
        dropped: Arc<AtomicBool>,
    }

    struct BlockingCloseSource {
        closed: Arc<AtomicBool>,
        probe: Arc<CheckpointProbe>,
    }

    impl Drop for BlockedOpenSource {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Release);
        }
    }

    #[async_trait]
    impl StreamSource for PendingSource {
        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
                delivery: SourceDeliveryCapability::Lossless,
                max_batch_rows: 1,
                max_batch_bytes: 1024,
                schema: SourceSchema::DynamicOrUnknown,
                native_watermarks: NativeWatermarkCapability::EmitsNative,
            }
        }

        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            std::future::pending().await
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.store(true, Ordering::Release);
            Ok(())
        }
    }

    #[async_trait]
    impl StreamSource for FailingOpenSource {
        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
                delivery: SourceDeliveryCapability::Lossless,
                max_batch_rows: 1,
                max_batch_bytes: 1024,
                schema: SourceSchema::DynamicOrUnknown,
                native_watermarks: NativeWatermarkCapability::EmitsNative,
            }
        }

        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Err(CalcFlowError::ExternalProvider {
                provider: "python-private-provider".into(),
                name: "private-source".into(),
                version: "private-version".into(),
                message: "private-connector-payload-redaction-sentinel".into(),
            })
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            std::future::pending().await
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl StreamSource for BlockedOpenSource {
        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
                delivery: SourceDeliveryCapability::Lossless,
                max_batch_rows: 1,
                max_batch_bytes: 1024,
                schema: SourceSchema::DynamicOrUnknown,
                native_watermarks: NativeWatermarkCapability::EmitsNative,
            }
        }

        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            std::future::pending().await
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            std::future::pending().await
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl StreamSource for BlockingCloseSource {
        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
                delivery: SourceDeliveryCapability::Lossless,
                max_batch_rows: 1,
                max_batch_bytes: 1024,
                schema: SourceSchema::DynamicOrUnknown,
                native_watermarks: NativeWatermarkCapability::EmitsNative,
            }
        }

        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            std::future::pending().await
        }

        async fn close(&mut self) -> Result<()> {
            let released = self.probe.release.notified();
            self.probe.entered.store(true, Ordering::Release);
            released.await;
            self.closed.store(true, Ordering::Release);
            Ok(())
        }
    }

    struct NoopSink;

    struct CheckpointProbe {
        entered: AtomicBool,
        release: tokio::sync::Notify,
        commits: AtomicUsize,
    }

    impl CheckpointProbe {
        fn new() -> Self {
            Self {
                entered: AtomicBool::new(false),
                release: tokio::sync::Notify::new(),
                commits: AtomicUsize::new(0),
            }
        }
    }

    #[pyclass]
    struct PyCheckpointProbe {
        inner: Arc<CheckpointProbe>,
    }

    #[pymethods]
    impl PyCheckpointProbe {
        #[getter]
        fn entered(&self) -> bool {
            self.inner.entered.load(Ordering::Acquire)
        }

        fn fire(&self) {
            self.inner.release.notify_waiters();
        }
    }

    struct BlockingCheckpointSink {
        probe: Arc<CheckpointProbe>,
    }

    #[async_trait]
    impl StreamSink for NoopSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl TransactionalStreamSink for BlockingCheckpointSink {
        async fn open(&mut self) -> Result<()> {
            Ok(())
        }

        async fn begin_epoch(&mut self, _epoch: calc_flow::Epoch) -> Result<()> {
            Ok(())
        }

        async fn write(&mut self, _batch: &Batch) -> Result<()> {
            Ok(())
        }

        async fn pre_commit(&mut self, _epoch: calc_flow::Epoch) -> Result<calc_flow::JsonMap> {
            if !self.probe.entered.load(Ordering::Acquire) {
                let released = self.probe.release.notified();
                self.probe.entered.store(true, Ordering::Release);
                released.await;
            }
            Ok(calc_flow::JsonMap::new())
        }

        async fn commit(
            &mut self,
            _epoch: calc_flow::Epoch,
            _pre_commit: &calc_flow::JsonMap,
        ) -> Result<()> {
            self.probe.commits.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }

        async fn abort(
            &mut self,
            _epoch: calc_flow::Epoch,
            _pre_commit: Option<&calc_flow::JsonMap>,
        ) -> Result<()> {
            Ok(())
        }

        async fn recover(&mut self, _recovery: &SinkRecovery) -> Result<()> {
            Ok(())
        }

        async fn close(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn test_stream_plan(name: &str) -> StreamExecutionPlan {
        PipelineBuilder::new(name)
            .unwrap()
            .add_node(
                "operator",
                Box::new(
                    ExpressionOperator::new(
                        "operator",
                        "result = value + 1",
                        Vec::new(),
                        None,
                        Vec::new(),
                    )
                    .unwrap(),
                ),
            )
            .unwrap()
            .compile_stream(
                &UdfRegistry::new().snapshot(),
                &StreamRequirements::default(),
            )
            .unwrap()
    }

    fn python_batch() -> PyBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let values = Arc::new(Int64Array::from(vec![1_i64]));
        let record = RecordBatch::try_new(schema, vec![values]).unwrap();
        PyBatch::from_inner(
            Batch::table(vec![record], calc_flow::BatchMetadata::default()).unwrap(),
        )
    }

    fn assert_native_source_variants(py: Python<'_>, locals: &Bound<'_, PyDict>) {
        let policy =
            |name: &str| super::source_policy(&locals.get_item(name).unwrap().unwrap()).unwrap();
        assert!(matches!(
            policy("source_provided"),
            calc_flow::WatermarkPolicy::SourceProvided
        ));
        assert!(matches!(
            policy("bounded"),
            calc_flow::WatermarkPolicy::BoundedOutOfOrderness {
                event_time_column,
                max_out_of_orderness,
                emit_interval,
                idle_timeout: Some(idle_timeout),
            } if event_time_column == "event_time"
                && max_out_of_orderness == std::time::Duration::from_micros(5)
                && emit_interval == std::time::Duration::from_micros(7)
                && idle_timeout == std::time::Duration::from_micros(11)
        ));
        assert!(matches!(
            policy("disabled"),
            calc_flow::WatermarkPolicy::Disabled { idle_timeout: None }
        ));
        assert!(
            super::source_policy(&locals.get_item("invalid_policy").unwrap().unwrap()).is_err()
        );

        let awaits = Arc::new(crate::runtime::PythonAwaitRegistry::new());
        let context = Arc::new(super::Mutex::new(None));
        let ownership = Arc::new(super::ConnectorOwnership::new());
        let python_source = |name: &str| super::PythonContinuousSource {
            binding: Arc::new(crate::config::PythonRoot::new(
                locals.get_item(name).unwrap().unwrap().unbind(),
            )),
            awaits: Arc::clone(&awaits),
            capability_error: Arc::new(super::Mutex::new(None)),
            context: Arc::clone(&context),
            _ownership: ownership.retain(),
        };
        let capabilities = python_source("good_capabilities")
            .parsed_capabilities()
            .unwrap();
        assert_eq!(
            capabilities.replay_positioning,
            ReplayPositioning::Unsupported
        );
        assert_eq!(capabilities.delivery, SourceDeliveryCapability::Lossy);
        assert_eq!(capabilities.max_batch_rows, 3);
        assert_eq!(capabilities.max_batch_bytes, 5);
        assert_eq!(
            capabilities.native_watermarks,
            NativeWatermarkCapability::RuntimeToggleable
        );
        assert!(
            python_source("invalid_delivery")
                .parsed_capabilities()
                .is_err()
        );
        assert!(
            python_source("invalid_watermarks")
                .parsed_capabilities()
                .is_err()
        );
        let mut invalid_source = python_source("invalid_replay");
        assert_eq!(invalid_source.capabilities().max_batch_rows, usize::MAX);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let error = py
            .detach(|| runtime.block_on(invalid_source.open(None)))
            .unwrap_err();
        assert!(error.to_string().contains("source.capabilities"));
    }

    fn assert_native_sink_variants(py: Python<'_>, locals: &Bound<'_, PyDict>) {
        let awaits = Arc::new(crate::runtime::PythonAwaitRegistry::new());
        let context = Arc::new(super::Mutex::new(None));
        let ownership = Arc::new(super::ConnectorOwnership::new());
        let sinks = PyDict::new(py);
        sinks
            .set_item(
                "output",
                (
                    locals.get_item("bounded_sink").unwrap().unwrap(),
                    locals.get_item("unbounded_sink").unwrap().unwrap(),
                ),
            )
            .unwrap();
        let (native, roots) = super::build_sinks(&sinks, &awaits, &context, &ownership).unwrap();
        assert_eq!(native["output"].len(), 2);
        assert_eq!(roots.len(), 2);
        drop(native);
        drop(roots);

        assert!(
            super::build_one_sink(
                locals.get_item("invalid_sink").unwrap().unwrap(),
                &awaits,
                &context,
                &ownership,
            )
            .is_err()
        );
        let invalid_shape = PyDict::new(py);
        invalid_shape.set_item("output", 1).unwrap();
        assert!(super::build_sinks(&invalid_shape, &awaits, &context, &ownership).is_err());

        for retention in [
            calc_flow::RetentionClass::Bounded,
            calc_flow::RetentionClass::Unbounded,
        ] {
            let delivery = calc_flow::SinkDelivery::EpochIdempotent {
                mechanism: "key".into(),
                retention,
            };
            let value = super::sink_delivery_to_py(py, &delivery).unwrap();
            assert_eq!(
                value
                    .get_item("kind")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "epoch_idempotent"
            );
        }
    }

    #[test]
    fn native_binding_descriptors_cover_delivery_and_watermark_variants() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    "class Policy:\n    def __init__(self, value): self.value = value\n    def _native_policy(self): return self.value\nsource_provided = Policy({'kind': 'source_provided'})\nbounded = Policy({'kind': 'bounded_out_of_orderness', 'event_time_column': 'event_time', 'max_out_of_orderness_micros': 5, 'emit_interval_micros': 7, 'idle_timeout_micros': 11})\ndisabled = Policy({'kind': 'disabled', 'idle_timeout_micros': None})\ninvalid_policy = Policy({'kind': 'invalid'})\nclass Capabilities:\n    def __init__(self, **overrides):\n        self.value = {'replay_positioning': 'unsupported', 'delivery': 'lossy', 'max_batch_rows': 3, 'max_batch_bytes': 5, 'schema': None, 'native_watermarks': 'runtime_toggleable'}\n        self.value.update(overrides)\n    def _native_capabilities(self): return self.value\ngood_capabilities = Capabilities()\ninvalid_replay = Capabilities(replay_positioning='invalid')\ninvalid_delivery = Capabilities(delivery='invalid')\ninvalid_watermarks = Capabilities(native_watermarks='invalid')\nclass Sink:\n    def __init__(self, sink_id, descriptor):\n        self.sink_id = sink_id\n        self.descriptor = descriptor\n    def _native_descriptor(self): return self.descriptor\nbounded_sink = Sink('bounded', {'kind': 'epoch_idempotent', 'mechanism': 'key', 'retention': 'bounded'})\nunbounded_sink = Sink('unbounded', {'kind': 'epoch_idempotent', 'mechanism': 'key', 'retention': 'unbounded'})\ninvalid_sink = Sink('invalid', {'kind': 'epoch_idempotent', 'mechanism': 'key', 'retention': 'invalid'})"
                ),
                Some(&locals),
                None,
            )
            .unwrap();
            assert_native_source_variants(py, &locals);
            assert_native_sink_variants(py, &locals);
        });
    }

    #[test]
    fn native_projection_names_cover_job_and_error_variants() {
        assert_eq!(
            [
                calc_flow::JobState::Running,
                calc_flow::JobState::Draining,
                calc_flow::JobState::Completed,
                calc_flow::JobState::Cancelled,
                calc_flow::JobState::Failed,
                calc_flow::JobState::RecoveryRequired,
            ]
            .map(super::job_state_name),
            [
                "running",
                "draining",
                "completed",
                "cancelled",
                "failed",
                "recovery_required",
            ]
        );
        assert_eq!(
            [
                calc_flow::TerminalCause::NaturalEnd,
                calc_flow::TerminalCause::GracefulShutdown,
                calc_flow::TerminalCause::ExplicitCancel,
                calc_flow::TerminalCause::DeadlineExceeded,
                calc_flow::TerminalCause::Failure,
            ]
            .map(super::terminal_cause_name),
            [
                "natural_end",
                "graceful_shutdown",
                "explicit_cancel",
                "deadline_exceeded",
                "failure",
            ]
        );
        assert_eq!(
            [
                calc_flow::StreamingErrorCategory::Validation,
                calc_flow::StreamingErrorCategory::Compile,
                calc_flow::StreamingErrorCategory::Conflict,
                calc_flow::StreamingErrorCategory::Cancelled,
                calc_flow::StreamingErrorCategory::CheckpointTimeout,
                calc_flow::StreamingErrorCategory::CheckpointMismatch,
                calc_flow::StreamingErrorCategory::CheckpointPublicationUnknown,
                calc_flow::StreamingErrorCategory::Io,
                calc_flow::StreamingErrorCategory::Operator,
                calc_flow::StreamingErrorCategory::Connector,
                calc_flow::StreamingErrorCategory::TaskPanicked,
                calc_flow::StreamingErrorCategory::Internal,
            ]
            .map(super::streaming_error_category_name),
            [
                "validation",
                "compile",
                "conflict",
                "cancelled",
                "checkpoint_timeout",
                "checkpoint_mismatch",
                "checkpoint_publication_unknown",
                "io",
                "operator",
                "connector",
                "task_panicked",
                "internal",
            ]
        );
    }

    #[test]
    fn native_projection_names_cover_checkpoint_and_delivery_variants() {
        assert_eq!(
            [
                calc_flow::CheckpointPhase::Requested,
                calc_flow::CheckpointPhase::SourcesCut,
                calc_flow::CheckpointPhase::OperatorsSnapshotted,
                calc_flow::CheckpointPhase::SinksPrecommitted,
                calc_flow::CheckpointPhase::ManifestInstalled,
                calc_flow::CheckpointPhase::ManifestDurable,
                calc_flow::CheckpointPhase::SinksCommitted,
                calc_flow::CheckpointPhase::Completed,
            ]
            .map(super::checkpoint_phase_name),
            [
                "requested",
                "sources_cut",
                "operators_snapshotted",
                "sinks_precommitted",
                "manifest_installed",
                "manifest_durable",
                "sinks_committed",
                "completed",
            ]
        );
        assert_eq!(
            [
                calc_flow::ComponentKind::Job,
                calc_flow::ComponentKind::Edge,
                calc_flow::ComponentKind::Source,
                calc_flow::ComponentKind::Operator,
                calc_flow::ComponentKind::Sink,
                calc_flow::ComponentKind::Checkpoint,
            ]
            .map(super::component_kind_name),
            ["job", "edge", "source", "operator", "sink", "checkpoint"]
        );
        assert_eq!(
            [
                ReplayPositioning::ExactPauseReportAndSeek,
                ReplayPositioning::Unsupported,
            ]
            .map(super::replay_positioning_name),
            ["exact_pause_report_and_seek", "unsupported"]
        );
        assert_eq!(
            [
                SourceDeliveryCapability::Lossless,
                SourceDeliveryCapability::Lossy,
            ]
            .map(super::source_delivery_name),
            ["lossless", "lossy"]
        );
        assert_eq!(
            [
                calc_flow::DeliveryGuarantee::AtLeastOnce,
                calc_flow::DeliveryGuarantee::ExactlyOnce,
            ]
            .map(super::delivery_guarantee_name),
            ["at_least_once", "exactly_once"]
        );
    }

    #[test]
    fn native_python_connectors_run_to_completion_and_project_status() {
        Python::initialize();
        let directory = tempfile::tempdir().unwrap();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item("batch", Py::new(py, python_batch()).unwrap())
                .unwrap();
            py.run(
                pyo3::ffi::c_str!(
                    "import asyncio\nevents = []\nclass NativeSource:\n    def __init__(self):\n        self.emitted = False\n        self.release = asyncio.Event()\n    def _native_capabilities(self):\n        return {'replay_positioning': 'exact_pause_report_and_seek', 'delivery': 'lossless', 'max_batch_rows': 1, 'max_batch_bytes': 1024, 'schema': None, 'native_watermarks': 'emits_native'}\n    def _native_policy(self): return {'kind': 'source_provided'}\n    async def _native_open(self, source_id, order, payload):\n        assert source_id is order is payload is None\n        events.append('source.open')\n    async def _native_next(self):\n        if self.emitted:\n            await self.release.wait()\n            return None\n        self.emitted = True\n        return ('data', batch, None, b'1', {'offset': 1})\n    async def _native_close(self): events.append('source.close')\nclass NativeSink:\n    sink_id = 'archive'\n    def _native_descriptor(self): return {'kind': 'ordinary'}\n    async def _native_open(self): events.append('sink.open')\n    async def _native_write(self, value): events.append(f'sink.write:{value.num_rows}')\n    async def _native_close(self): events.append('sink.close')\nclass NativeTransactionalSink:\n    sink_id = 'transactional-archive'\n    def _native_descriptor(self): return {'kind': 'transactional'}\n    async def _native_open(self): events.append('transactional.open')\n    async def _native_begin_epoch(self, epoch): events.append(f'transactional.begin:{epoch}')\n    async def _native_write(self, value): events.append(f'transactional.write:{value.num_rows}')\n    async def _native_pre_commit(self, epoch):\n        events.append(f'transactional.pre_commit:{epoch}')\n        return {'epoch': epoch}\n    async def _native_commit(self, epoch, pre_commit):\n        assert pre_commit == {'epoch': epoch}\n        events.append(f'transactional.commit:{epoch}')\n    async def _native_abort(self, epoch, pre_commit): events.append(f'transactional.abort:{epoch}')\n    async def _native_recover(self, epoch, terminal, delivery, pre_commit): events.append(f'transactional.recover:{epoch}')\n    async def _native_close(self): events.append('transactional.close')\nsource = NativeSource()\nsink = NativeSink()\ntransactional_sink = NativeTransactionalSink()"
                ),
                Some(&locals),
                None,
            )
            .unwrap();

            let sources = PyDict::new(py);
            sources
                .set_item("input", locals.get_item("source").unwrap().unwrap())
                .unwrap();
            let sinks = PyDict::new(py);
            sinks
                .set_item(
                    "output",
                    [
                        locals.get_item("sink").unwrap().unwrap(),
                        locals.get_item("transactional_sink").unwrap().unwrap(),
                    ],
                )
                .unwrap();
            let config = PyDict::new(py);
            config
                .set_item("checkpoint_interval_micros", 60_000_000_u64)
                .unwrap();
            config
                .set_item("checkpoint_timeout_micros", 600_000_000_u64)
                .unwrap();
            config.set_item("edge_max_rows", 10_000_u64).unwrap();
            config.set_item("edge_max_bytes", 64_u64 << 20).unwrap();
            config.set_item("retained_epochs", 2_u64).unwrap();

            let plan = Py::new(
                py,
                PyStreamExecutionPlan::new(test_stream_plan("python-native-connectors"), py.None()),
            )
            .unwrap();
            let checkpoints = Py::new(
                py,
                PyManagedCheckpointRuntime::new(directory.path().to_str().unwrap()).unwrap(),
            )
            .unwrap();
            let static_inputs = PyDict::new(py);
            let runner = PyContinuousStreamingRunner::new(
                plan.borrow(py),
                &sources,
                &sinks,
                checkpoints.borrow(py),
                &config,
                &static_inputs,
            )
            .unwrap();
            locals
                .set_item("runner", Py::new(py, runner).unwrap())
                .unwrap();
            py.run(
                &CString::new(
                    "import asyncio\nasync def exercise():\n    assert 'consumed=false' in repr(runner)\n    job = await runner.start_async()\n    assert job.id > 0\n    while 'transactional.write:1' not in events:\n        await asyncio.sleep(0)\n    epoch = await job.trigger_checkpoint_async()\n    assert epoch >= 1\n    source.release.set()\n    status = job.status()\n    assert status['job_id'] == job.id\n    assert set(status) == {'job_id', 'state', 'terminal_cause', 'delivery', 'task_count', 'task_errors', 'metrics_overflowed', 'watermark_micros', 'edges', 'sources', 'operators', 'sinks', 'checkpoint', 'stream_joins'}\n    assert status['stream_joins'] == {}
    outcome = await job.wait_async()\n    assert outcome['state'] == 'completed', outcome\n    assert outcome['cause'] == 'natural_end'\n    assert outcome['errors'] == ()\n    assert 'state=completed' in repr(job)\nasyncio.run(exercise())\nassert 'source.open' in events\nassert 'sink.open' in events\nassert 'sink.write:1' in events\nassert 'source.close' in events\nassert 'sink.close' in events\nassert any(value.startswith('transactional.begin:') for value in events)\nassert any(value.startswith('transactional.pre_commit:') for value in events)\nassert any(value.startswith('transactional.commit:') for value in events)\nassert 'transactional.close' in events",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    fn pending_runner(managed_root: &std::path::Path, closed: Arc<AtomicBool>) -> StreamingRunner {
        let plan = test_stream_plan("python-owning-job");
        let source_id = plan.source_binding_ids()[0].to_owned();
        let output_id = plan.sink_binding_ids()[0].to_owned();
        StreamingRunner::new(
            plan,
            BTreeMap::from([(source_id, SourceBinding::new(PendingSource { closed }))]),
            BTreeMap::from([(
                output_id,
                vec![SinkBinding::ordinary("sink", NoopSink).unwrap()],
            )]),
            ManagedCheckpointRuntime::new(managed_root).unwrap(),
        )
        .unwrap()
    }

    fn checkpoint_runner(
        managed_root: &std::path::Path,
        closed: Arc<AtomicBool>,
        probe: Arc<CheckpointProbe>,
    ) -> StreamingRunner {
        let plan = test_stream_plan("python-checkpoint-observer");
        let source_id = plan.source_binding_ids()[0].to_owned();
        let output_id = plan.sink_binding_ids()[0].to_owned();
        StreamingRunner::new(
            plan,
            BTreeMap::from([(source_id, SourceBinding::new(PendingSource { closed }))]),
            BTreeMap::from([(
                output_id,
                vec![SinkBinding::transactional("sink", BlockingCheckpointSink { probe }).unwrap()],
            )]),
            ManagedCheckpointRuntime::new(managed_root).unwrap(),
        )
        .unwrap()
    }

    fn failing_runner(managed_root: &std::path::Path) -> StreamingRunner {
        let plan = test_stream_plan("python-safe-error");
        let source_id = plan.source_binding_ids()[0].to_owned();
        let output_id = plan.sink_binding_ids()[0].to_owned();
        StreamingRunner::new(
            plan,
            BTreeMap::from([(source_id, SourceBinding::new(FailingOpenSource))]),
            BTreeMap::from([(
                output_id,
                vec![SinkBinding::ordinary("sink", NoopSink).unwrap()],
            )]),
            ManagedCheckpointRuntime::new(managed_root).unwrap(),
        )
        .unwrap()
    }

    fn blocked_start_runner(
        managed_root: &std::path::Path,
        dropped: Arc<AtomicBool>,
    ) -> StreamingRunner {
        let plan = test_stream_plan("python-cancelled-start");
        let source_id = plan.source_binding_ids()[0].to_owned();
        let output_id = plan.sink_binding_ids()[0].to_owned();
        StreamingRunner::new(
            plan,
            BTreeMap::from([(source_id, SourceBinding::new(BlockedOpenSource { dropped }))]),
            BTreeMap::from([(
                output_id,
                vec![SinkBinding::ordinary("sink", NoopSink).unwrap()],
            )]),
            ManagedCheckpointRuntime::new(managed_root).unwrap(),
        )
        .unwrap()
    }

    fn blocking_close_runner(
        managed_root: &std::path::Path,
        closed: Arc<AtomicBool>,
        probe: Arc<CheckpointProbe>,
    ) -> StreamingRunner {
        let plan = test_stream_plan("python-lifecycle-observer");
        let source_id = plan.source_binding_ids()[0].to_owned();
        let output_id = plan.sink_binding_ids()[0].to_owned();
        StreamingRunner::new(
            plan,
            BTreeMap::from([(
                source_id,
                SourceBinding::new(BlockingCloseSource { closed, probe }),
            )]),
            BTreeMap::from([(
                output_id,
                vec![SinkBinding::ordinary("sink", NoopSink).unwrap()],
            )]),
            ManagedCheckpointRuntime::new(managed_root).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn owning_job_outlives_runner_and_waiter_cancellation() {
        Python::initialize();
        let directory = tempfile::tempdir().unwrap();
        let closed = Arc::new(AtomicBool::new(false));
        Python::attach(|py| {
            let locals = PyDict::new(py);
            let runner = Py::new(
                py,
                PyContinuousStreamingRunner::from_inner(pending_runner(
                    directory.path(),
                    Arc::clone(&closed),
                )),
            )
            .unwrap();
            locals.set_item("runner", runner).unwrap();
            py.run(
                &CString::new(
                    "import asyncio, gc\nasync def exercise():\n    owned_runner = runner\n    globals().pop('runner')\n    job = await owned_runner.start_async()\n    del owned_runner\n    gc.collect()\n    first_status = job.status()\n    assert first_status['state'] == 'running'\n    first_status['state'] = 'tampered'\n    assert job.status()['state'] == 'running'\n    waiter = asyncio.ensure_future(job.wait_async())\n    await asyncio.sleep(0)\n    waiter.cancel()\n    try:\n        await waiter\n    except asyncio.CancelledError:\n        pass\n    else:\n        raise AssertionError('cancelled wait observer returned')\n    assert job.status()['state'] == 'running'\n    cancelled = await job.cancel_async()\n    repeated = await job.cancel_async()\n    waited = await job.wait_async()\n    assert cancelled == repeated == waited\n    assert waited['state'] == 'cancelled'\n    assert waited['cause'] == 'explicit_cancel'\nasyncio.run(exercise())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();
        });
        assert!(closed.load(Ordering::Acquire));
    }

    #[test]
    fn pending_native_wait_does_not_hold_the_gil() {
        Python::initialize();
        let directory = tempfile::tempdir().unwrap();
        let closed = Arc::new(AtomicBool::new(false));
        Python::attach(|py| {
            let locals = PyDict::new(py);
            let runner = Py::new(
                py,
                PyContinuousStreamingRunner::from_inner(pending_runner(
                    directory.path(),
                    Arc::clone(&closed),
                )),
            )
            .unwrap();
            locals.set_item("runner", runner).unwrap();
            py.run(
                &CString::new(
                    "import asyncio, threading\nasync def exercise():\n    job = await runner.start_async()\n    waiter = asyncio.ensure_future(job.wait_async())\n    await asyncio.sleep(0.05)\n    assert not waiter.done()\n    marker = []\n    worker = threading.Thread(target=lambda: marker.append('ran'))\n    worker.start()\n    worker.join(timeout=1)\n    assert marker == ['ran']\n    assert not worker.is_alive()\n    assert job.status()['state'] == 'running'\n    waiter.cancel()\n    try:\n        await waiter\n    except asyncio.CancelledError:\n        pass\n    else:\n        raise AssertionError('cancelled wait observer returned')\n    assert job.status()['state'] == 'running'\n    await job.cancel_async()\nasyncio.run(exercise())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();
        });
        assert!(closed.load(Ordering::Acquire));
    }

    #[test]
    fn unawaited_start_does_not_consume_runner() {
        Python::initialize();
        let directory = tempfile::tempdir().unwrap();
        let closed = Arc::new(AtomicBool::new(false));
        Python::attach(|py| {
            let locals = PyDict::new(py);
            let runner = Py::new(
                py,
                PyContinuousStreamingRunner::from_inner(pending_runner(
                    directory.path(),
                    Arc::clone(&closed),
                )),
            )
            .unwrap();
            locals.set_item("runner", runner).unwrap();
            py.run(
                &CString::new(
                    "import asyncio, gc\nasync def exercise():\n    unstarted = runner.start_async()\n    del unstarted\n    gc.collect()\n    job = await runner.start_async()\n    outcome = await job.cancel_async()\n    assert outcome['state'] == 'cancelled'\nasyncio.run(exercise())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();
        });
        assert!(closed.load(Ordering::Acquire));
    }

    #[test]
    fn cancelled_start_reaps_provisional_runtime_and_releases_lease() {
        Python::initialize();
        let directory = tempfile::tempdir().unwrap();
        let dropped = Arc::new(AtomicBool::new(false));
        Python::attach(|py| {
            let locals = PyDict::new(py);
            let runner = Py::new(
                py,
                PyContinuousStreamingRunner::from_inner(blocked_start_runner(
                    directory.path(),
                    Arc::clone(&dropped),
                )),
            )
            .unwrap();
            locals.set_item("runner", runner).unwrap();
            py.run(
                &CString::new(
                    "import asyncio\nasync def exercise():\n    launch = asyncio.ensure_future(runner.start_async())\n    await asyncio.sleep(0.05)\n    assert not launch.done()\n    launch.cancel()\n    try:\n        await launch\n    except asyncio.CancelledError:\n        pass\n    else:\n        raise AssertionError('cancelled start observer returned')\nasyncio.run(exercise())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();
        });

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while !dropped.load(Ordering::Acquire) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("cancelled start must drop provisional connector ownership");

            let replacement_closed = Arc::new(AtomicBool::new(false));
            let replacement = pending_runner(directory.path(), Arc::clone(&replacement_closed))
                .start()
                .await
                .expect("cancelled start must release the managed checkpoint lease");
            assert_eq!(
                replacement.cancel().await.state,
                calc_flow::JobState::Cancelled
            );
            assert!(replacement_closed.load(Ordering::Acquire));
        });
    }

    #[test]
    fn dropped_waiter_and_event_loop_release_job_and_checkpoint_lease() {
        Python::initialize();
        let directory = tempfile::tempdir().unwrap();
        let closed = Arc::new(AtomicBool::new(false));
        Python::attach(|py| {
            let locals = PyDict::new(py);
            let runner = Py::new(
                py,
                PyContinuousStreamingRunner::from_inner(pending_runner(
                    directory.path(),
                    Arc::clone(&closed),
                )),
            )
            .unwrap();
            locals.set_item("runner", runner).unwrap();
            py.run(
                &CString::new(
                    "import asyncio, gc\nasync def exercise():\n    owned_runner = runner\n    globals().pop('runner')\n    job = await owned_runner.start_async()\n    del owned_runner\n    wait_awaitable = job.wait_async()\n    native_waiter = asyncio.ensure_future(wait_awaitable)\n    await asyncio.sleep(0.05)\n    assert not native_waiter.done()\n    try:\n        wait_awaitable.__await__()\n    except RuntimeError as error:\n        assert 'already been awaited' in str(error)\n    else:\n        raise AssertionError('native wait observer was not started')\n    del wait_awaitable, native_waiter, job\n    gc.collect()\nasyncio.run(exercise())\ngc.collect()",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();
        });

        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while !closed.load(Ordering::Acquire) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("job GC must settle the source after event-loop close");

            let replacement_closed = Arc::new(AtomicBool::new(false));
            let replacement = pending_runner(directory.path(), Arc::clone(&replacement_closed))
                .start()
                .await
                .expect("the managed checkpoint lease must be reusable after GC");
            let outcome = replacement.cancel().await;
            assert_eq!(outcome.state, calc_flow::JobState::Cancelled);
            assert!(replacement_closed.load(Ordering::Acquire));
        });
    }

    #[test]
    fn checkpoint_awaiter_is_deferred_and_cancellation_detaches_only_observer() {
        Python::initialize();
        let directory = tempfile::tempdir().unwrap();
        let closed = Arc::new(AtomicBool::new(false));
        let probe = Arc::new(CheckpointProbe::new());
        Python::attach(|py| {
            let locals = PyDict::new(py);
            let runner = Py::new(
                py,
                PyContinuousStreamingRunner::from_inner(checkpoint_runner(
                    directory.path(),
                    Arc::clone(&closed),
                    Arc::clone(&probe),
                )),
            )
            .unwrap();
            let python_probe = Py::new(
                py,
                PyCheckpointProbe {
                    inner: Arc::clone(&probe),
                },
            )
            .unwrap();
            locals.set_item("runner", runner).unwrap();
            locals.set_item("probe", python_probe).unwrap();
            py.run(
                &CString::new(
                    "import asyncio, gc\nasync def exercise():\n    job = await runner.start_async()\n    unstarted = job.trigger_checkpoint_async()\n    await asyncio.sleep(0.05)\n    started_without_await = probe.entered\n    if started_without_await:\n        probe.fire()\n    del unstarted\n    gc.collect()\n    assert not started_without_await\n    checkpoint = asyncio.ensure_future(job.trigger_checkpoint_async())\n    while not probe.entered:\n        await asyncio.sleep(0)\n    checkpoint.cancel()\n    try:\n        await checkpoint\n    except asyncio.CancelledError:\n        pass\n    else:\n        raise AssertionError('cancelled checkpoint observer returned')\n    probe.fire()\n    outcome = await job.shutdown_async()\n    assert outcome['state'] == 'completed'\n    assert outcome['completed_epoch'] >= 1\nasyncio.run(exercise())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();
        });
        assert!(closed.load(Ordering::Acquire));
        assert!(probe.commits.load(Ordering::Acquire) >= 1);
    }

    #[test]
    fn lifecycle_awaiter_cancellation_preserves_native_convergence() {
        Python::initialize();
        let shutdown_directory = tempfile::tempdir().unwrap();
        let cancel_directory = tempfile::tempdir().unwrap();
        let shutdown_closed = Arc::new(AtomicBool::new(false));
        let cancel_closed = Arc::new(AtomicBool::new(false));
        let shutdown_probe = Arc::new(CheckpointProbe::new());
        let cancel_probe = Arc::new(CheckpointProbe::new());
        Python::attach(|py| {
            let locals = PyDict::new(py);
            let shutdown_runner = Py::new(
                py,
                PyContinuousStreamingRunner::from_inner(blocking_close_runner(
                    shutdown_directory.path(),
                    Arc::clone(&shutdown_closed),
                    Arc::clone(&shutdown_probe),
                )),
            )
            .unwrap();
            let cancel_runner = Py::new(
                py,
                PyContinuousStreamingRunner::from_inner(blocking_close_runner(
                    cancel_directory.path(),
                    Arc::clone(&cancel_closed),
                    Arc::clone(&cancel_probe),
                )),
            )
            .unwrap();
            let python_shutdown_probe = Py::new(
                py,
                PyCheckpointProbe {
                    inner: Arc::clone(&shutdown_probe),
                },
            )
            .unwrap();
            let python_cancel_probe = Py::new(
                py,
                PyCheckpointProbe {
                    inner: Arc::clone(&cancel_probe),
                },
            )
            .unwrap();
            locals.set_item("shutdown_runner", shutdown_runner).unwrap();
            locals.set_item("cancel_runner", cancel_runner).unwrap();
            locals
                .set_item("shutdown_probe", python_shutdown_probe)
                .unwrap();
            locals
                .set_item("cancel_probe", python_cancel_probe)
                .unwrap();
            py.run(
                &CString::new(
                    "import asyncio, gc\nasync def cancel_observer(operation, probe):\n    observer = asyncio.ensure_future(operation())\n    while not probe.entered:\n        await asyncio.sleep(0)\n    observer.cancel()\n    try:\n        await observer\n    except asyncio.CancelledError:\n        pass\n    else:\n        raise AssertionError('cancelled lifecycle observer returned')\n    probe.fire()\n    return await operation()\nasync def exercise():\n    shutdown_job = await shutdown_runner.start_async()\n    unstarted = shutdown_job.shutdown_async()\n    await asyncio.sleep(0.05)\n    assert shutdown_job.status()['state'] == 'running'\n    del unstarted\n    gc.collect()\n    shutdown = await cancel_observer(shutdown_job.shutdown_async, shutdown_probe)\n    assert shutdown['state'] == 'completed'\n    assert shutdown['cause'] == 'graceful_shutdown'\n    cancel_job = await cancel_runner.start_async()\n    cancelled = await cancel_observer(cancel_job.cancel_async, cancel_probe)\n    assert cancelled['state'] == 'cancelled'\n    assert cancelled['cause'] == 'explicit_cancel'\nasyncio.run(exercise())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();
        });
        assert!(shutdown_closed.load(Ordering::Acquire));
        assert!(cancel_closed.load(Ordering::Acquire));
    }

    #[test]
    fn start_error_is_frozen_and_contains_only_safe_fields() {
        Python::initialize();
        let directory = tempfile::tempdir().unwrap();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            let module = pyo3::types::PyModule::new(py, "_native").unwrap();
            super::register(&module).unwrap();
            let runner = Py::new(
                py,
                PyContinuousStreamingRunner::from_inner(failing_runner(directory.path())),
            )
            .unwrap();
            locals.set_item("runner", runner).unwrap();
            py.run(
                &CString::new(
                    "import asyncio, traceback\nasync def exercise():\n    try:\n        await runner.start_async()\n    except Exception as error:\n        assert type(error).__name__ == 'StreamingRuntimeError'\n        assert error.category == 'connector'\n        assert error.message == str(error)\n        assert error.job_id is not None\n        assert error.epoch is None\n        assert error.checkpoint_phase is None\n        assert error.component_kind == 'source'\n        assert error.component_id == 'input'\n        assert error.diagnostic_id is None\n        assert error.position == 0\n        assert error.__cause__ is None\n        assert error.__context__ is None\n        rendered = repr(error) + str(error) + ''.join(traceback.format_exception(error)) + repr(vars(error))\n        for sentinel in ('private-connector-payload-redaction-sentinel', 'python-private-provider', 'private-source', 'private-version'):\n            assert sentinel not in rendered\n        try:\n            error.category = 'tampered'\n        except AttributeError:\n            pass\n        else:\n            raise AssertionError('safe exception fields must be read-only')\n        backing = error._calc_flow_safe_fields\n        try:\n            backing['category'] = 'tampered-through-dict'\n        except TypeError:\n            pass\n        else:\n            raise AssertionError('safe exception backing must be immutable')\n        try:\n            error._calc_flow_safe_fields = {'category': 'tampered-through-replacement'}\n        except AttributeError:\n            pass\n        else:\n            raise AssertionError('safe exception backing must not be replaceable')\n        vars(error)['_calc_flow_safe_fields'] = {'category': 'shadowed'}\n        assert error._calc_flow_safe_fields['category'] == 'connector'\n        assert error.category == 'connector'\n        assert error.message == str(error)\n    else:\n        raise AssertionError('start unexpectedly succeeded')\nasyncio.run(exercise())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    #[test]
    fn start_error_clears_active_exception_context() {
        Python::initialize();
        let directory = tempfile::tempdir().unwrap();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            let runner = Py::new(
                py,
                PyContinuousStreamingRunner::from_inner(failing_runner(directory.path())),
            )
            .unwrap();
            locals.set_item("runner", runner).unwrap();
            py.run(
                &CString::new(
                    "import asyncio, traceback\nACTIVE_CONTEXT_SENTINEL = 'active-exception-sensitive-sentinel'\nasync def exercise():\n    try:\n        raise RuntimeError(ACTIVE_CONTEXT_SENTINEL)\n    except RuntimeError:\n        try:\n            await runner.start_async()\n        except Exception as error:\n            assert type(error).__name__ == 'StreamingRuntimeError'\n            assert error.__cause__ is None\n            assert BaseException.__context__.__get__(error) is None\n            assert error.__context__ is None\n            rendered = ''.join(traceback.format_exception(error))\n            assert ACTIVE_CONTEXT_SENTINEL not in rendered\n        else:\n            raise AssertionError('start unexpectedly succeeded')\nasyncio.run(exercise())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    #[test]
    fn start_error_native_backing_cannot_be_replaced() {
        Python::initialize();
        let directory = tempfile::tempdir().unwrap();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            let runner = Py::new(
                py,
                PyContinuousStreamingRunner::from_inner(failing_runner(directory.path())),
            )
            .unwrap();
            locals.set_item("runner", runner).unwrap();
            py.run(
                &CString::new(
                    "import asyncio\nasync def exercise():\n    try:\n        await runner.start_async()\n    except Exception as error:\n        original_fields = dict(error._calc_flow_safe_fields)\n        assert error.category == 'connector'\n        try:\n            error._calc_flow_native_safe_fields = ('tampered',) * 9\n        except AttributeError:\n            pass\n        else:\n            raise AssertionError('native safe exception backing must not be replaceable')\n        assert error.category == 'connector'\n        assert dict(error._calc_flow_safe_fields) == original_fields\n        try:\n            del error._calc_flow_native_safe_fields\n        except AttributeError:\n            pass\n        else:\n            raise AssertionError('native safe exception backing must not be deletable')\n        assert error.category == 'connector'\n        assert dict(error._calc_flow_safe_fields) == original_fields\n    else:\n        raise AssertionError('start unexpectedly succeeded')\nasyncio.run(exercise())",
                )
                .unwrap(),
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }
}
