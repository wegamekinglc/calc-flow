use std::collections::{BTreeMap, HashSet};

use chrono::{Datelike, NaiveDate, TimeZone, Timelike, Utc};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple},
};
use serde_json::{Number, Value};

fn settings_type_error(message: impl Into<String>) -> PyErr {
    PyTypeError::new_err(format!("execution settings {}", message.into()))
}

fn settings_value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(format!("execution settings {}", message.into()))
}

fn deadline_type_error(message: impl Into<String>) -> PyErr {
    PyTypeError::new_err(format!("deadline {}", message.into()))
}

fn invalid_deadline() -> PyErr {
    PyValueError::new_err("deadline must be a timezone-aware UTC datetime")
}

fn strict_settings(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<calc_flow::JsonMap> {
    let mapping = py
        .import(pyo3::intern!(py, "collections.abc"))?
        .getattr(pyo3::intern!(py, "Mapping"))?;
    if !source.is_instance(&mapping)? {
        return Err(settings_type_error("must be a mapping"));
    }
    let copied = py
        .get_type::<PyDict>()
        .call1((source,))?
        .cast_into::<PyDict>()?;
    let mut ancestors = HashSet::new();
    strict_dict(&copied, 0, &mut ancestors)
}

fn strict_dict(
    value: &Bound<'_, PyDict>,
    depth: usize,
    ancestors: &mut HashSet<usize>,
) -> PyResult<calc_flow::JsonMap> {
    if depth > calc_flow::MAX_JSON_DEPTH {
        return Err(settings_value_error(format!(
            "exceed the maximum JSON depth of {}",
            calc_flow::MAX_JSON_DEPTH
        )));
    }
    let identity = value.as_ptr() as usize;
    if !ancestors.insert(identity) {
        return Err(settings_value_error("must not contain cycles"));
    }
    let result = (|| {
        let mut converted = BTreeMap::new();
        for (key, child) in value {
            if !key.is_exact_instance_of::<PyString>() {
                return Err(settings_value_error("keys must be exact strings"));
            }
            let key = key.extract::<String>()?;
            converted.insert(key, strict_value(&child, depth + 1, ancestors)?);
        }
        Ok(converted)
    })();
    ancestors.remove(&identity);
    result
}

fn strict_list(
    value: &Bound<'_, PyList>,
    depth: usize,
    ancestors: &mut HashSet<usize>,
) -> PyResult<Vec<Value>> {
    if depth > calc_flow::MAX_JSON_DEPTH {
        return Err(settings_value_error(format!(
            "exceed the maximum JSON depth of {}",
            calc_flow::MAX_JSON_DEPTH
        )));
    }
    let identity = value.as_ptr() as usize;
    if !ancestors.insert(identity) {
        return Err(settings_value_error("must not contain cycles"));
    }
    let result = value
        .iter()
        .map(|child| strict_value(&child, depth + 1, ancestors))
        .collect();
    ancestors.remove(&identity);
    result
}

fn strict_value(
    value: &Bound<'_, PyAny>,
    depth: usize,
    ancestors: &mut HashSet<usize>,
) -> PyResult<Value> {
    if depth > calc_flow::MAX_JSON_DEPTH {
        return Err(settings_value_error(format!(
            "exceed the maximum JSON depth of {}",
            calc_flow::MAX_JSON_DEPTH
        )));
    }
    if value.is_none() {
        return Ok(Value::Null);
    }
    if value.is_exact_instance_of::<PyBool>() {
        return value.extract::<bool>().map(Value::Bool);
    }
    if value.is_exact_instance_of::<PyInt>() {
        if let Ok(integer) = value.extract::<i64>() {
            return Ok(Value::Number(Number::from(integer)));
        }
        if let Ok(integer) = value.extract::<u64>() {
            return Ok(Value::Number(Number::from(integer)));
        }
        return Err(settings_value_error(
            "integers must be in the range -2**63 through 2**64 - 1",
        ));
    }
    if value.is_exact_instance_of::<PyFloat>() {
        let number = value.extract::<f64>()?;
        if !number.is_finite() {
            return Err(settings_value_error("numbers must be finite"));
        }
        return Ok(Value::Number(
            Number::from_f64(number).expect("finite floats are valid JSON numbers"),
        ));
    }
    if value.is_exact_instance_of::<PyString>() {
        return value.extract::<String>().map(Value::String);
    }
    if value.is_exact_instance_of::<PyList>() {
        return strict_list(value.cast::<PyList>()?, depth, ancestors).map(Value::Array);
    }
    if value.is_exact_instance_of::<PyDict>() {
        let converted = strict_dict(value.cast::<PyDict>()?, depth, ancestors)?;
        return Ok(Value::Object(converted.into_iter().collect()));
    }
    Err(settings_value_error(
        "values must use exact built-in JSON types",
    ))
}

fn parse_deadline(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<chrono::DateTime<Utc>>> {
    if value.is_none() {
        return Ok(None);
    }
    let datetime_module = py.import(pyo3::intern!(py, "datetime"))?;
    let datetime_type = datetime_module.getattr(pyo3::intern!(py, "datetime"))?;
    if !value.is_instance(&datetime_type)? {
        return Err(deadline_type_error("must be a datetime or None"));
    }
    let offset = value
        .call_method0(pyo3::intern!(py, "utcoffset"))
        .map_err(|_| invalid_deadline())?;
    let timedelta_type = datetime_module.getattr(pyo3::intern!(py, "timedelta"))?;
    if offset.is_none() || !offset.is_instance(&timedelta_type)? {
        return Err(invalid_deadline());
    }
    let offset_is_zero = offset
        .getattr(pyo3::intern!(py, "days"))?
        .extract::<i32>()?
        == 0
        && offset
            .getattr(pyo3::intern!(py, "seconds"))?
            .extract::<i32>()?
            == 0
        && offset
            .getattr(pyo3::intern!(py, "microseconds"))?
            .extract::<i32>()?
            == 0;
    if !offset_is_zero {
        return Err(invalid_deadline());
    }

    let year = value.getattr(pyo3::intern!(py, "year"))?.extract::<i32>()?;
    let month = value
        .getattr(pyo3::intern!(py, "month"))?
        .extract::<u32>()?;
    let day = value.getattr(pyo3::intern!(py, "day"))?.extract::<u32>()?;
    let hour = value.getattr(pyo3::intern!(py, "hour"))?.extract::<u32>()?;
    let minute = value
        .getattr(pyo3::intern!(py, "minute"))?
        .extract::<u32>()?;
    let second = value
        .getattr(pyo3::intern!(py, "second"))?
        .extract::<u32>()?;
    let microsecond = value
        .getattr(pyo3::intern!(py, "microsecond"))?
        .extract::<u32>()?;
    let naive = NaiveDate::from_ymd_opt(year, month, day)
        .and_then(|date| date.and_hms_micro_opt(hour, minute, second, microsecond))
        .ok_or_else(invalid_deadline)?;
    Ok(Some(Utc.from_utc_datetime(&naive)))
}

fn deadline_to_python<'py>(
    py: Python<'py>,
    deadline: Option<&chrono::DateTime<Utc>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Some(deadline) = deadline else {
        return Ok(None);
    };
    let datetime_module = py.import(pyo3::intern!(py, "datetime"))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(
        pyo3::intern!(py, "tzinfo"),
        datetime_module.getattr(pyo3::intern!(py, "UTC"))?,
    )?;
    datetime_module
        .getattr(pyo3::intern!(py, "datetime"))?
        .call(
            (
                deadline.year(),
                deadline.month(),
                deadline.day(),
                deadline.hour(),
                deadline.minute(),
                deadline.second(),
                deadline.timestamp_subsec_micros(),
            ),
            Some(&kwargs),
        )
        .map(Some)
}

fn settings_to_python<'py>(
    py: Python<'py>,
    settings: &calc_flow::JsonMap,
) -> PyResult<Bound<'py, PyAny>> {
    let encoded = serde_json::to_string(settings)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    crate::config::json_to_python(py, &encoded)
}

#[pyclass(name = "ExecutionOptions", frozen, module = "calc_flow._native")]
pub(crate) struct PyExecutionOptions {
    settings: calc_flow::JsonMap,
    deadline: Option<chrono::DateTime<Utc>>,
}

impl PyExecutionOptions {
    pub(crate) fn to_core(&self) -> calc_flow::ExecutionOptions {
        calc_flow::ExecutionOptions {
            settings: self.settings.clone(),
            deadline: self.deadline,
            cancellation: calc_flow::CancellationToken::new(),
        }
    }

    #[cfg(test)]
    pub(crate) const fn for_test(
        settings: calc_flow::JsonMap,
        deadline: Option<chrono::DateTime<Utc>>,
    ) -> Self {
        Self { settings, deadline }
    }
}

#[pymethods]
impl PyExecutionOptions {
    #[new]
    #[pyo3(
        signature = (*args, **kwargs),
        text_signature = "(settings={}, deadline=None)"
    )]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if args.len() > 2 {
            return Err(PyTypeError::new_err(format!(
                "ExecutionOptions expected at most 2 arguments, got {}",
                args.len()
            )));
        }
        let mut settings = (args.len() >= 1).then(|| args.get_item(0)).transpose()?;
        let mut deadline = (args.len() >= 2).then(|| args.get_item(1)).transpose()?;
        if let Some(kwargs) = kwargs {
            for (key, _) in kwargs {
                let name = key.extract::<String>()?;
                if name != "settings" && name != "deadline" {
                    return Err(PyTypeError::new_err(format!(
                        "ExecutionOptions got an unexpected keyword argument {name:?}"
                    )));
                }
            }
            if let Some(value) = kwargs.get_item("settings")? {
                if settings.is_some() {
                    return Err(PyTypeError::new_err(
                        "ExecutionOptions got multiple values for argument 'settings'",
                    ));
                }
                settings = Some(value);
            }
            if let Some(value) = kwargs.get_item("deadline")? {
                if deadline.is_some() {
                    return Err(PyTypeError::new_err(
                        "ExecutionOptions got multiple values for argument 'deadline'",
                    ));
                }
                deadline = Some(value);
            }
        }
        let settings =
            settings.map_or_else(|| Ok(BTreeMap::new()), |value| strict_settings(py, &value))?;
        let deadline = deadline.map_or(Ok(None), |value| parse_deadline(py, &value))?;
        Ok(Self { settings, deadline })
    }

    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        settings_to_python(py, &self.settings)
    }

    #[getter]
    fn deadline<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        deadline_to_python(py, self.deadline.as_ref())
    }
}

#[pyclass(name = "ProviderContext", frozen, module = "calc_flow._native")]
pub(crate) struct PyProviderContext {
    settings: calc_flow::JsonMap,
    deadline: Option<chrono::DateTime<Utc>>,
}

impl PyProviderContext {
    pub(crate) fn from_run(run: &calc_flow::RunContext) -> Self {
        Self {
            settings: run.settings().clone(),
            deadline: run.deadline().copied(),
        }
    }
}

#[pymethods]
impl PyProviderContext {
    #[getter]
    fn settings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        settings_to_python(py, &self.settings)
    }

    #[getter]
    fn deadline<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        deadline_to_python(py, self.deadline.as_ref())
    }
}

#[pyclass(frozen, module = "calc_flow._native")]
pub(crate) struct PyExecutionCancellation {
    cancellation: calc_flow::CancellationToken,
}

impl PyExecutionCancellation {
    pub(crate) const fn new(cancellation: calc_flow::CancellationToken) -> Self {
        Self { cancellation }
    }
}

#[pymethods]
impl PyExecutionCancellation {
    fn cancel(&self) {
        self.cancellation.cancel();
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyExecutionOptions>()?;
    module.add_class::<PyProviderContext>()?;
    Ok(())
}
