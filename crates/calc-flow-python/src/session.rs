//! Python binding for the caller-driven `StreamSession`.

use std::{collections::BTreeMap, time::Duration};

use datafusion::arrow::record_batch::RecordBatch;
use parking_lot::Mutex;
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    intern,
    prelude::*,
    types::{PyAny, PyDict, PyList, PyTuple},
};
use pyo3_arrow::{PyRecordBatch, PyTable};

use crate::{batch::PyBatch, pipeline::PyStreamExecutionPlan};

/// The binding lifecycle. A call swaps `Idle` for `Busy` and runs without the
/// state lock or the GIL; `close` during `Busy` wins over the call's result.
#[derive(Debug)]
enum SessionState<S> {
    Idle(S),
    Busy,
    Closed,
    Failed(String),
    Finished,
}

/// How a finished native call leaves the session.
#[derive(Debug)]
enum Completion {
    Open,
    Finished,
    Failed(String),
}

impl<S> SessionState<S> {
    fn begin(&mut self) -> Result<S, String> {
        match std::mem::replace(self, Self::Busy) {
            Self::Idle(session) => Ok(session),
            other => {
                let message = other.unavailable();
                *self = other;
                Err(message)
            }
        }
    }

    /// Stores the next state; returns the session when it must be dropped.
    fn complete(&mut self, session: S, completion: Completion) -> Option<S> {
        if !matches!(self, Self::Busy) {
            return Some(session);
        }
        match completion {
            Completion::Open => {
                *self = Self::Idle(session);
                None
            }
            Completion::Finished => {
                *self = Self::Finished;
                Some(session)
            }
            Completion::Failed(message) => {
                *self = Self::Failed(message);
                Some(session)
            }
        }
    }

    /// Closes idempotently; returns an idle session to drop.
    fn close(&mut self) -> Option<S> {
        match std::mem::replace(self, Self::Closed) {
            Self::Idle(session) => Some(session),
            _ => None,
        }
    }

    fn unavailable(&self) -> String {
        match self {
            Self::Idle(_) => "StreamSession is idle".into(),
            Self::Busy => "StreamSession is busy: another call is running".into(),
            Self::Closed => "StreamSession is closed".into(),
            Self::Failed(message) => {
                format!("StreamSession failed earlier ({message}); create a new session")
            }
            Self::Finished => "StreamSession is finished".into(),
        }
    }
}

/// Owns one call's `Busy` state. A call that unwinds before completing
/// stores `Failed` so a panic cannot leave the session busy forever.
struct CallGuard<'a, S> {
    state: &'a Mutex<SessionState<S>>,
    completed: bool,
}

impl<'a, S> CallGuard<'a, S> {
    fn begin(state: &'a Mutex<SessionState<S>>) -> Result<(Self, S), String> {
        let session = state.lock().begin()?;
        Ok((
            Self {
                state,
                completed: false,
            },
            session,
        ))
    }

    fn complete(mut self, session: S, completion: Completion) -> Option<S> {
        self.completed = true;
        self.state.lock().complete(session, completion)
    }
}

impl<S> Drop for CallGuard<'_, S> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let mut state = self.state.lock();
        if matches!(*state, SessionState::Busy) {
            *state = SessionState::Failed("a session call panicked".into());
        }
    }
}

type NativeSession = Box<calc_flow::StreamSession>;
type NativeResult = calc_flow::Result<calc_flow::SessionOutput>;

#[pyclass(name = "_StreamSession", frozen, module = "calc_flow._native")]
pub(crate) struct PyStreamSession {
    state: Mutex<SessionState<NativeSession>>,
    input_ids: Vec<String>,
    output_ids: Vec<String>,
    owner: Mutex<Option<Py<PyAny>>>,
}

impl PyStreamSession {
    /// Runs one native call on the caller's thread with the GIL released.
    fn call(
        &self,
        py: Python<'_>,
        finishes: bool,
        operation: impl FnOnce(&mut calc_flow::StreamSession) -> NativeResult + Send,
    ) -> PyResult<Py<PyDict>> {
        let (guard, mut session) =
            CallGuard::begin(&self.state).map_err(PyRuntimeError::new_err)?;
        let (session, result) = py.detach(move || {
            let result = operation(&mut session);
            (session, result)
        });
        let failed = session.is_failed();
        let completion = match &result {
            Ok(_) if finishes => Completion::Finished,
            Err(error) if failed => Completion::Failed(error.to_string()),
            _ => Completion::Open,
        };
        let released = guard.complete(session, completion);
        drop(released);
        let output = result.map_err(|error| session_py_err(error, failed))?;
        outputs_to_python(py, output)
    }
}

#[pymethods]
impl PyStreamSession {
    #[new]
    #[allow(
        clippy::needless_pass_by_value,
        reason = "PyO3 constructor extraction owns its PyRef boundary value"
    )]
    fn new(
        py: Python<'_>,
        plan: PyRef<'_, PyStreamExecutionPlan>,
        config: &Bound<'_, PyDict>,
        watermarks: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let config = crate::continuous::runtime_config(config)?;
        let watermarks = session_watermarks(watermarks)?;
        let (plan, owner) = plan.take()?;
        let session = py
            .detach(move || calc_flow::StreamSession::new(plan, &config, watermarks))
            .map_err(crate::error::to_py_err)?;
        let input_ids = session.input_ids().into_iter().map(str::to_owned).collect();
        let output_ids = session
            .output_ids()
            .into_iter()
            .map(str::to_owned)
            .collect();
        Ok(Self {
            state: Mutex::new(SessionState::Idle(Box::new(session))),
            input_ids,
            output_ids,
            owner: Mutex::new(Some(owner)),
        })
    }

    #[pyo3(signature = (input, batch, watermark_micros=None))]
    fn push(
        &self,
        py: Python<'_>,
        input: String,
        batch: &Bound<'_, PyAny>,
        watermark_micros: Option<i64>,
    ) -> PyResult<Py<PyDict>> {
        let batch = input_batch(batch)?;
        let watermark = watermark_micros.map(calc_flow::EventTime::from_micros);
        self.call(py, false, move |session| {
            session.push(&input, batch, watermark)
        })
    }

    fn advance(
        &self,
        py: Python<'_>,
        input: String,
        watermark_micros: i64,
    ) -> PyResult<Py<PyDict>> {
        let watermark = calc_flow::EventTime::from_micros(watermark_micros);
        self.call(py, false, move |session| session.advance(&input, watermark))
    }

    fn finish(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.call(py, true, calc_flow::StreamSession::finish)
    }

    /// Closes without flushing; idempotent. A running call drops its session.
    fn close(&self) {
        let released = self.state.lock().close();
        drop(released);
        drop(self.owner.lock().take());
    }

    #[getter]
    fn input_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.input_ids)
    }

    #[getter]
    fn output_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.output_ids)
    }

    fn __repr__(&self) -> String {
        let state = match &*self.state.lock() {
            SessionState::Idle(_) => "idle",
            SessionState::Busy => "busy",
            SessionState::Closed => "closed",
            SessionState::Failed(_) => "failed",
            SessionState::Finished => "finished",
        };
        format!("<calc_flow._native._StreamSession state={state}>")
    }

    #[allow(clippy::needless_pass_by_value)]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(owner) = self.owner.lock().as_ref() {
            visit.call(owner)?;
        }
        Ok(())
    }

    fn __clear__(&self) {
        drop(self.owner.lock().take());
    }
}

/// Lifecycle and Tokio-context errors raise `RuntimeError` and a rejected
/// input schema raises `ValueError`; the rest keep the ordinary error classes.
fn session_py_err(error: calc_flow::CalcFlowError, failed: bool) -> PyErr {
    match &error {
        calc_flow::CalcFlowError::InvalidArgument { field, message } if field == "session" => {
            PyRuntimeError::new_err(format!("StreamSession: {message}"))
        }
        // Validation compares the input batch before any operator runs.
        calc_flow::CalcFlowError::Compile { message } if !failed => {
            PyValueError::new_err(message.clone())
        }
        _ => crate::error::to_py_err(error),
    }
}

fn input_batch(batch: &Bound<'_, PyAny>) -> PyResult<calc_flow::Batch> {
    if let Ok(native) = batch.cast::<PyBatch>() {
        return native.get().clone_inner();
    }
    let py = batch.py();
    let records = if batch.hasattr(intern!(py, "__arrow_c_array__"))? {
        vec![batch.extract::<PyRecordBatch>()?.into_inner()]
    } else if batch.hasattr(intern!(py, "__arrow_c_stream__"))? {
        let (mut records, schema) = batch.extract::<PyTable>()?.into_inner();
        if records.is_empty() {
            records.push(RecordBatch::new_empty(schema));
        }
        records
    } else {
        return Err(PyTypeError::new_err(
            "session input must be a Batch or implement the Arrow C data or stream interface",
        ));
    };
    calc_flow::Batch::table(records, calc_flow::BatchMetadata::default())
        .map_err(crate::error::to_py_err)
}

fn outputs_to_python(py: Python<'_>, output: calc_flow::SessionOutput) -> PyResult<Py<PyDict>> {
    let outputs = PyDict::new(py);
    for (name, batches) in output.into_batches() {
        let records = PyList::empty(py);
        for batch in batches {
            let table = batch.table_payload().map_err(|_| {
                PyTypeError::new_err(format!("session output {name:?} emitted an array batch"))
            })?;
            for record in table.batches() {
                records.append(PyRecordBatch::new(record.clone()).into_pyarrow(py)?)?;
            }
        }
        outputs.set_item(name, records)?;
    }
    Ok(outputs.unbind())
}

fn session_watermarks(
    watermarks: &Bound<'_, PyDict>,
) -> PyResult<BTreeMap<String, calc_flow::SessionWatermark>> {
    watermarks
        .iter()
        .map(|(name, policy)| {
            let name: String = name.extract()?;
            let policy = policy.cast_into::<PyDict>().map_err(|_| {
                PyTypeError::new_err(format!("session watermark {name:?} must be a dict"))
            })?;
            Ok((name, session_watermark(&policy)?))
        })
        .collect()
}

fn session_watermark(policy: &Bound<'_, PyDict>) -> PyResult<calc_flow::SessionWatermark> {
    let item = |key: &str| {
        policy
            .get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("session watermark requires {key:?}")))
    };
    let kind: String = item("kind")?.extract()?;
    Ok(match kind.as_str() {
        "explicit" => calc_flow::SessionWatermark::Explicit,
        "disabled" => calc_flow::SessionWatermark::Disabled,
        "ordered" => calc_flow::SessionWatermark::Ordered {
            event_time_column: item("event_time_column")?.extract()?,
        },
        "bounded_out_of_orderness" => calc_flow::SessionWatermark::BoundedOutOfOrderness {
            event_time_column: item("event_time_column")?.extract()?,
            max_out_of_orderness: Duration::from_micros(
                item("max_out_of_orderness_micros")?.extract()?,
            ),
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported session watermark kind {other:?}"
            )));
        }
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStreamSession>()
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use parking_lot::Mutex;

    use super::{CallGuard, Completion, SessionState};

    fn idle() -> SessionState<&'static str> {
        SessionState::Idle("session")
    }

    #[test]
    fn test_call_moves_idle_to_busy_and_back() {
        let mut state = idle();
        assert_eq!(state.begin(), Ok("session"));
        assert!(matches!(state, SessionState::Busy));
        assert_eq!(state.complete("session", Completion::Open), None);
        assert!(matches!(state, SessionState::Idle("session")));
    }

    #[test]
    fn test_busy_rejects_concurrent_calls() {
        let mut state = idle();
        state.begin().unwrap();
        assert_eq!(
            state.begin(),
            Err("StreamSession is busy: another call is running".into())
        );
        assert!(matches!(state, SessionState::Busy));
    }

    #[test]
    fn test_finish_and_failure_are_terminal() {
        let mut state = idle();
        state.begin().unwrap();
        assert_eq!(
            state.complete("session", Completion::Finished),
            Some("session")
        );
        assert_eq!(state.begin(), Err("StreamSession is finished".into()));

        let mut state = idle();
        state.begin().unwrap();
        assert_eq!(
            state.complete("session", Completion::Failed("node rolling failed".into())),
            Some("session")
        );
        assert_eq!(
            state.begin(),
            Err("StreamSession failed earlier (node rolling failed); create a new session".into())
        );
    }

    #[test]
    fn test_close_during_busy_drops_the_session_on_completion() {
        let mut state = idle();
        state.begin().unwrap();
        assert_eq!(state.close(), None);
        assert!(matches!(state, SessionState::Closed));
        assert_eq!(state.complete("session", Completion::Open), Some("session"));
        assert!(matches!(state, SessionState::Closed));
        assert_eq!(state.begin(), Err("StreamSession is closed".into()));
    }

    #[test]
    fn test_close_is_idempotent_from_every_state() {
        let mut state = idle();
        assert_eq!(state.close(), Some("session"));
        assert_eq!(state.close(), None);
        assert!(matches!(state, SessionState::Closed));
        for mut state in [
            SessionState::<&str>::Finished,
            SessionState::Failed("error".into()),
        ] {
            assert_eq!(state.close(), None);
            assert!(matches!(state, SessionState::Closed));
        }
    }

    #[test]
    fn test_panic_during_call_stores_failed_instead_of_busy() {
        let state = Mutex::new(idle());
        let unwound = catch_unwind(AssertUnwindSafe(|| {
            let (_guard, _session) = CallGuard::begin(&state).unwrap();
            panic!("export failed");
        }));
        assert!(unwound.is_err());
        assert_eq!(
            state.lock().begin(),
            Err(
                "StreamSession failed earlier (a session call panicked); create a new session"
                    .into()
            )
        );
    }

    #[test]
    fn test_completed_or_closed_call_guard_keeps_its_state() {
        let state = Mutex::new(idle());
        let (guard, session) = CallGuard::begin(&state).unwrap();
        assert_eq!(guard.complete(session, Completion::Open), None);
        assert!(matches!(*state.lock(), SessionState::Idle("session")));

        let unwound = catch_unwind(AssertUnwindSafe(|| {
            let (_guard, _session) = CallGuard::begin(&state).unwrap();
            assert_eq!(state.lock().close(), None);
            panic!("export failed after close");
        }));
        assert!(unwound.is_err());
        assert!(matches!(*state.lock(), SessionState::Closed));
    }
}
