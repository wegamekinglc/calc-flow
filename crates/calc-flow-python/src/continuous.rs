use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use parking_lot::Mutex;
use pyo3::{
    IntoPyObjectExt,
    prelude::*,
    sync::PyOnceLock,
    types::{PyAny, PyCFunction, PyDict, PyDictMethods, PyTuple, PyType},
};

const SAFE_EXCEPTION_STORAGE: &str = "_calc_flow_safe_fields";
const NATIVE_EXCEPTION_STORAGE: &str = "_calc_flow_native_safe_fields";
const SAFE_EXCEPTION_FIELDS: [&str; 9] = [
    "category",
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

struct SafeStreamingErrorFields {
    category: String,
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

    fn from_streaming(error: &calc_flow::continuous::StreamingError) -> Self {
        Self {
            category: streaming_error_category_name(error.category()).to_owned(),
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

#[pyclass(
    name = "_ContinuousStreamingRunner",
    frozen,
    module = "calc_flow._native"
)]
pub(crate) struct PyContinuousStreamingRunner {
    inner: Arc<RunnerStartState>,
}

struct RunnerStartState {
    runner: Mutex<Option<calc_flow::continuous::StreamingRunner>>,
}

#[pyclass(frozen, module = "calc_flow._native")]
struct PyStreamingStartAwaitable {
    inner: Arc<RunnerStartState>,
    started: AtomicBool,
}

#[pyclass(name = "_StreamingJob", frozen, module = "calc_flow._native")]
pub(crate) struct PyStreamingJob {
    inner: Arc<calc_flow::continuous::StreamingJob>,
}

#[pyclass(frozen, module = "calc_flow._native")]
struct PyStreamingJobAwaitable {
    inner: Arc<calc_flow::continuous::StreamingJob>,
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

impl PyContinuousStreamingRunner {
    #[allow(
        dead_code,
        reason = "constructed by the native connector bindings added in downstream A6-11"
    )]
    pub(crate) fn from_inner(inner: calc_flow::continuous::StreamingRunner) -> Self {
        Self {
            inner: Arc::new(RunnerStartState {
                runner: Mutex::new(Some(inner)),
            }),
        }
    }
}

#[pymethods]
impl PyContinuousStreamingRunner {
    fn start_async(&self, py: Python<'_>) -> PyResult<Py<PyStreamingStartAwaitable>> {
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
        format!("<calc_flow._native._ContinuousStreamingRunner consumed={consumed}>")
    }
}

#[pymethods]
impl PyStreamingStartAwaitable {
    fn __await__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.started.swap(true, Ordering::AcqRel) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "streaming start awaitable has already been awaited",
            ));
        }
        let runner = self.inner.runner.lock().take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "streaming runner has already been consumed by start()",
            )
        })?;
        let observer = pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let job = runner.start().await.map_err(streaming_py_err)?;
            Python::attach(|py| Py::new(py, PyStreamingJob::from_inner(job)))
        })?;
        observer.call_method0("__await__")
    }
}

impl PyStreamingJob {
    fn from_inner(inner: calc_flow::continuous::StreamingJob) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

#[pymethods]
impl PyStreamingJob {
    #[getter]
    fn id(&self) -> u64 {
        self.inner.id()
    }

    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        job_status_to_py(py, &self.inner.status())
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
        format!(
            "<calc_flow._native._StreamingJob id={} state={}>",
            self.inner.id(),
            job_state_name(self.inner.status().state)
        )
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
                inner: Arc::clone(&self.inner),
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
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "streaming job awaitable has already been awaited",
            ));
        }
        let job = Arc::clone(&self.inner);
        let observer = match self.operation {
            JobOperation::TriggerCheckpoint => persistent_future(py, async move {
                job.trigger_checkpoint()
                    .await
                    .map(calc_flow::Epoch::as_u64)
                    .map_err(streaming_py_err)
            })?,
            JobOperation::Shutdown => persistent_future(py, async move {
                let outcome = job.shutdown().await;
                Python::attach(|py| job_outcome_to_py(py, &outcome).map(Bound::unbind))
            })?,
            JobOperation::Cancel => persistent_future(py, async move {
                let outcome = job.cancel().await;
                Python::attach(|py| job_outcome_to_py(py, &outcome).map(Bound::unbind))
            })?,
            JobOperation::Wait => pyo3_async_runtimes::tokio::future_into_py(py, async move {
                let outcome = job.wait().await;
                Python::attach(|py| job_outcome_to_py(py, &outcome).map(Bound::unbind))
            })?,
        };
        observer.call_method0("__await__")
    }
}

fn persistent_future<'py, F, T>(py: Python<'py>, future: F) -> PyResult<Bound<'py, PyAny>>
where
    F: Future<Output = PyResult<T>> + Send + 'static,
    T: for<'a> IntoPyObject<'a> + Send + 'static,
{
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let observer = pyo3_async_runtimes::tokio::future_into_py(py, async move {
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

fn streaming_error_to_py_err(error: &calc_flow::continuous::StreamingError) -> PyErr {
    let fields = SafeStreamingErrorFields::from_streaming(error);
    let checkpoint_publication_unknown = error.category()
        == calc_flow::continuous::StreamingErrorCategory::CheckpointPublicationUnknown;
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
        let values = PyTuple::new(
            py,
            [
                fields.category.into_py_any(py)?,
                fields.message.into_py_any(py)?,
                fields.job_id.into_py_any(py)?,
                fields.epoch.into_py_any(py)?,
                fields.checkpoint_phase.into_py_any(py)?,
                fields.component_kind.into_py_any(py)?,
                fields.component_id.into_py_any(py)?,
                fields.diagnostic_id.into_py_any(py)?,
                fields.position.into_py_any(py)?,
            ],
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
    namespace.set_item("__slots__", PyTuple::new(py, [NATIVE_EXCEPTION_STORAGE])?)?;
    let property = py.import("builtins")?.getattr("property")?;
    for (index, field) in SAFE_EXCEPTION_FIELDS.into_iter().enumerate() {
        let getter = PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
            let instance = args.get_item(0)?;
            instance
                .getattr(NATIVE_EXCEPTION_STORAGE)?
                .get_item(index)
                .map(Bound::unbind)
        })?;
        namespace.set_item(field, property.call1((getter,))?)?;
    }
    let storage_getter = PyCFunction::new_closure(py, None, None, move |args, _kwargs| {
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
    })?;
    namespace.set_item(SAFE_EXCEPTION_STORAGE, property.call1((storage_getter,))?)?;
    Ok(namespace)
}

fn job_outcome_to_py<'py>(
    py: Python<'py>,
    outcome: &calc_flow::continuous::JobOutcome,
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
    error: &calc_flow::continuous::StreamingError,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("category", streaming_error_category_name(error.category()))?;
    value.set_item("message", error.message())?;
    value.set_item("job_id", error.job_id())?;
    value.set_item("epoch", error.epoch().map(calc_flow::Epoch::as_u64))?;
    value.set_item(
        "checkpoint_phase",
        error.checkpoint_phase().map(checkpoint_phase_name),
    )?;
    value.set_item(
        "component_kind",
        error.component_kind().map(component_kind_name),
    )?;
    value.set_item("component_id", error.component_id())?;
    value.set_item("diagnostic_id", error.diagnostic_id())?;
    value.set_item("position", error.position())?;
    Ok(value)
}

fn job_status_to_py<'py>(
    py: Python<'py>,
    status: &calc_flow::continuous::JobStatus,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item("job_id", status.job_id)?;
    value.set_item("state", job_state_name(status.state))?;
    value.set_item(
        "terminal_cause",
        status.terminal_cause.map(terminal_cause_name),
    )?;
    value.set_item("delivery", delivery_status_to_py(py, &status.delivery)?)?;
    value.set_item("task_count", status.task_count)?;
    value.set_item("task_errors", status.task_errors)?;
    value.set_item("metrics_overflowed", status.metrics_overflowed)?;
    value.set_item("edges", edge_status_to_py(py, &status.edges)?)?;
    value.set_item("sources", source_status_to_py(py, &status.sources)?)?;
    value.set_item("operators", operator_status_to_py(py, &status.operators)?)?;
    value.set_item("sinks", sink_status_to_py(py, &status.sinks)?)?;
    value.set_item(
        "checkpoint",
        checkpoint_status_to_py(py, &status.checkpoint)?,
    )?;
    Ok(value)
}

fn delivery_status_to_py<'py>(
    py: Python<'py>,
    statuses: &std::collections::BTreeMap<String, calc_flow::continuous::OutputDeliveryStatus>,
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
    statuses: &std::collections::BTreeMap<String, calc_flow::continuous::EdgeStatus>,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    for (edge_id, status) in statuses {
        let value = PyDict::new(py);
        value.set_item("current_envelopes", status.current_envelopes)?;
        value.set_item("current_rows", status.current_rows)?;
        value.set_item("current_bytes", status.current_bytes)?;
        value.set_item("high_water_envelopes", status.high_water_envelopes)?;
        value.set_item("high_water_rows", status.high_water_rows)?;
        value.set_item("high_water_bytes", status.high_water_bytes)?;
        value.set_item("blocked_sends", status.blocked_sends)?;
        value.set_item(
            "blocked_duration_micros",
            duration_micros(status.blocked_duration),
        )?;
        value.set_item("envelope_limit", status.envelope_limit)?;
        value.set_item("row_limit", status.row_limit)?;
        value.set_item("byte_limit", status.byte_limit)?;
        values.set_item(edge_id, value)?;
    }
    Ok(values)
}

fn source_status_to_py<'py>(
    py: Python<'py>,
    statuses: &std::collections::BTreeMap<String, calc_flow::continuous::SourceStatus>,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    for (source_id, status) in statuses {
        let value = PyDict::new(py);
        value.set_item(
            "replay_positioning",
            replay_positioning_name(status.replay_positioning),
        )?;
        value.set_item("delivery", source_delivery_name(status.delivery))?;
        value.set_item("max_batch_rows", status.max_batch_rows)?;
        value.set_item("max_batch_bytes", status.max_batch_bytes)?;
        value.set_item("next_sequence", status.next_sequence)?;
        value.set_item("ended", status.ended)?;
        value.set_item("polls", status.polls)?;
        value.set_item("data_batches", status.data_batches)?;
        value.set_item("data_rows", status.data_rows)?;
        value.set_item("data_bytes", status.data_bytes)?;
        value.set_item("fanned_out_batches", status.fanned_out_batches)?;
        value.set_item("fanned_out_rows", status.fanned_out_rows)?;
        value.set_item("fanned_out_bytes", status.fanned_out_bytes)?;
        value.set_item("errors", status.errors)?;
        values.set_item(source_id, value)?;
    }
    Ok(values)
}

fn operator_status_to_py<'py>(
    py: Python<'py>,
    statuses: &std::collections::BTreeMap<String, calc_flow::continuous::OperatorStatus>,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    for (operator_id, status) in statuses {
        let value = PyDict::new(py);
        value.set_item("input_batches", status.input_batches)?;
        value.set_item("input_rows", status.input_rows)?;
        value.set_item("input_bytes", status.input_bytes)?;
        value.set_item("fanned_out_batches", status.fanned_out_batches)?;
        value.set_item("fanned_out_rows", status.fanned_out_rows)?;
        value.set_item("fanned_out_bytes", status.fanned_out_bytes)?;
        value.set_item(
            "processing_duration_micros",
            duration_micros(status.processing_duration),
        )?;
        value.set_item("errors", status.errors)?;
        value.set_item("ended", status.ended)?;
        value.set_item("late_rows", status.late_rows)?;
        value.set_item("late_affected_batches", status.late_affected_batches)?;
        value.set_item(
            "max_lateness_micros",
            status.max_lateness.map(duration_micros),
        )?;
        value.set_item("null_event_time_rows", status.null_event_time_rows)?;
        value.set_item("null_event_time_batches", status.null_event_time_batches)?;
        value.set_item(
            "datafusion_runtime_created",
            status.datafusion_runtime_created,
        )?;
        values.set_item(operator_id, value)?;
    }
    Ok(values)
}

fn sink_status_to_py<'py>(
    py: Python<'py>,
    statuses: &std::collections::BTreeMap<String, calc_flow::continuous::SinkStatus>,
) -> PyResult<Bound<'py, PyDict>> {
    let values = PyDict::new(py);
    for (sink_id, status) in statuses {
        let value = PyDict::new(py);
        value.set_item("output_id", &status.output_id)?;
        value.set_item(
            "effective_delivery",
            sink_delivery_to_py(py, &status.effective_delivery)?,
        )?;
        value.set_item("delivered_batches", status.delivered_batches)?;
        value.set_item("delivered_rows", status.delivered_rows)?;
        value.set_item("delivered_bytes", status.delivered_bytes)?;
        value.set_item(
            "write_duration_micros",
            duration_micros(status.write_duration),
        )?;
        value.set_item("errors", status.errors)?;
        value.set_item("ended", status.ended)?;
        values.set_item(sink_id, value)?;
    }
    Ok(values)
}

fn checkpoint_status_to_py<'py>(
    py: Python<'py>,
    status: &calc_flow::continuous::CheckpointStatus,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    value.set_item(
        "current_epoch",
        status.current_epoch.map(calc_flow::Epoch::as_u64),
    )?;
    value.set_item("phase", status.phase.map(checkpoint_phase_name))?;
    value.set_item("terminal", status.terminal)?;
    value.set_item("source_acknowledgements", status.source_acknowledgements)?;
    value.set_item("expected_sources", status.expected_sources)?;
    value.set_item(
        "operator_acknowledgements",
        status.operator_acknowledgements,
    )?;
    value.set_item("expected_operators", status.expected_operators)?;
    value.set_item(
        "sink_precommit_acknowledgements",
        status.sink_precommit_acknowledgements,
    )?;
    value.set_item("expected_sink_precommits", status.expected_sink_precommits)?;
    value.set_item(
        "sink_commit_acknowledgements",
        status.sink_commit_acknowledgements,
    )?;
    value.set_item("expected_sink_commits", status.expected_sink_commits)?;
    value.set_item("elapsed_micros", status.elapsed.map(duration_micros))?;
    value.set_item(
        "last_completed_epoch",
        status.last_completed_epoch.map(calc_flow::Epoch::as_u64),
    )?;
    value.set_item(
        "installed_unknown_epoch",
        status.installed_unknown_epoch.map(calc_flow::Epoch::as_u64),
    )?;
    value.set_item(
        "failure_category",
        status.failure_category.map(streaming_error_category_name),
    )?;
    value.set_item("runtime_config_changed", status.runtime_config_changed)?;
    Ok(value)
}

fn sink_delivery_to_py<'py>(
    py: Python<'py>,
    delivery: &calc_flow::continuous::SinkDelivery,
) -> PyResult<Bound<'py, PyDict>> {
    let value = PyDict::new(py);
    match delivery {
        calc_flow::continuous::SinkDelivery::Ordinary => value.set_item("kind", "ordinary")?,
        calc_flow::continuous::SinkDelivery::EpochIdempotent {
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
        calc_flow::continuous::SinkDelivery::Transactional => {
            value.set_item("kind", "transactional")?;
        }
    }
    Ok(value)
}

const fn duration_micros(duration: Duration) -> u128 {
    duration.as_micros()
}

const fn job_state_name(state: calc_flow::continuous::JobState) -> &'static str {
    match state {
        calc_flow::continuous::JobState::Running => "running",
        calc_flow::continuous::JobState::Draining => "draining",
        calc_flow::continuous::JobState::Completed => "completed",
        calc_flow::continuous::JobState::Cancelled => "cancelled",
        calc_flow::continuous::JobState::Failed => "failed",
        calc_flow::continuous::JobState::RecoveryRequired => "recovery_required",
    }
}

const fn terminal_cause_name(cause: calc_flow::continuous::TerminalCause) -> &'static str {
    match cause {
        calc_flow::continuous::TerminalCause::NaturalEnd => "natural_end",
        calc_flow::continuous::TerminalCause::GracefulShutdown => "graceful_shutdown",
        calc_flow::continuous::TerminalCause::ExplicitCancel => "explicit_cancel",
        calc_flow::continuous::TerminalCause::DeadlineExceeded => "deadline_exceeded",
        calc_flow::continuous::TerminalCause::Failure => "failure",
    }
}

const fn streaming_error_category_name(
    category: calc_flow::continuous::StreamingErrorCategory,
) -> &'static str {
    match category {
        calc_flow::continuous::StreamingErrorCategory::Validation => "validation",
        calc_flow::continuous::StreamingErrorCategory::Compile => "compile",
        calc_flow::continuous::StreamingErrorCategory::Conflict => "conflict",
        calc_flow::continuous::StreamingErrorCategory::Cancelled => "cancelled",
        calc_flow::continuous::StreamingErrorCategory::CheckpointTimeout => "checkpoint_timeout",
        calc_flow::continuous::StreamingErrorCategory::CheckpointMismatch => "checkpoint_mismatch",
        calc_flow::continuous::StreamingErrorCategory::CheckpointPublicationUnknown => {
            "checkpoint_publication_unknown"
        }
        calc_flow::continuous::StreamingErrorCategory::Io => "io",
        calc_flow::continuous::StreamingErrorCategory::Operator => "operator",
        calc_flow::continuous::StreamingErrorCategory::Connector => "connector",
        calc_flow::continuous::StreamingErrorCategory::TaskPanicked => "task_panicked",
        calc_flow::continuous::StreamingErrorCategory::Internal => "internal",
    }
}

const fn checkpoint_phase_name(phase: calc_flow::continuous::CheckpointPhase) -> &'static str {
    match phase {
        calc_flow::continuous::CheckpointPhase::Requested => "requested",
        calc_flow::continuous::CheckpointPhase::SourcesCut => "sources_cut",
        calc_flow::continuous::CheckpointPhase::OperatorsSnapshotted => "operators_snapshotted",
        calc_flow::continuous::CheckpointPhase::SinksPrecommitted => "sinks_precommitted",
        calc_flow::continuous::CheckpointPhase::ManifestInstalled => "manifest_installed",
        calc_flow::continuous::CheckpointPhase::ManifestDurable => "manifest_durable",
        calc_flow::continuous::CheckpointPhase::SinksCommitted => "sinks_committed",
        calc_flow::continuous::CheckpointPhase::Completed => "completed",
    }
}

const fn component_kind_name(kind: calc_flow::continuous::ComponentKind) -> &'static str {
    match kind {
        calc_flow::continuous::ComponentKind::Job => "job",
        calc_flow::continuous::ComponentKind::Edge => "edge",
        calc_flow::continuous::ComponentKind::Source => "source",
        calc_flow::continuous::ComponentKind::Operator => "operator",
        calc_flow::continuous::ComponentKind::Sink => "sink",
        calc_flow::continuous::ComponentKind::Checkpoint => "checkpoint",
    }
}

const fn replay_positioning_name(
    capability: calc_flow::continuous::ReplayPositioning,
) -> &'static str {
    match capability {
        calc_flow::continuous::ReplayPositioning::ExactPauseReportAndSeek => {
            "exact_pause_report_and_seek"
        }
        calc_flow::continuous::ReplayPositioning::Unsupported => "unsupported",
    }
}

const fn source_delivery_name(
    capability: calc_flow::continuous::SourceDeliveryCapability,
) -> &'static str {
    match capability {
        calc_flow::continuous::SourceDeliveryCapability::Lossless => "lossless",
        calc_flow::continuous::SourceDeliveryCapability::Lossy => "lossy",
    }
}

const fn delivery_guarantee_name(guarantee: calc_flow::DeliveryGuarantee) -> &'static str {
    match guarantee {
        calc_flow::DeliveryGuarantee::AtLeastOnce => "at_least_once",
        calc_flow::DeliveryGuarantee::ExactlyOnce => "exactly_once",
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
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
        Batch, CalcFlowError, ExpressionOperator, PipelineBuilder, Result, StreamExecutionPlan,
        StreamRequirements, UdfRegistry,
        continuous::{
            Cursor, ManagedCheckpointRuntime, NativeWatermarkCapability, ReplayPositioning,
            SinkBinding, SinkRecovery, SourceBinding, SourceCapabilities, SourceDeliveryCapability,
            SourceEvent, SourceSchema, StreamSink, StreamSource, StreamingRunner,
            TransactionalStreamSink,
        },
    };
    use pyo3::{
        Py, Python, pyclass, pymethods,
        types::{PyDict, PyDictMethods},
    };

    use super::PyContinuousStreamingRunner;

    struct PendingSource {
        closed: Arc<AtomicBool>,
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
                        "value = value",
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
                calc_flow::continuous::JobState::Cancelled
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
            assert_eq!(outcome.state, calc_flow::continuous::JobState::Cancelled);
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
}
