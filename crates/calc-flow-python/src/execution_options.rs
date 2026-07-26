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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use pyo3::ffi::c_str;
    use serde_json::json;
    use std::sync::Mutex;

    static PYTHON_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn with_python(test: impl FnOnce(Python<'_>)) {
        let _guard = PYTHON_TEST_LOCK.lock().unwrap();
        Python::initialize();
        Python::attach(test);
    }

    fn expect_error<T>(result: PyResult<T>) -> PyErr {
        match result {
            Ok(_) => panic!("expected Python error"),
            Err(error) => error,
        }
    }

    #[test]
    fn strict_settings_copies_every_supported_json_value() {
        with_python(|py| {
            let source = py
                .eval(
                    c_str!(
                        "{'none': None, 'bool': True, 'signed': -7, \
                         'unsigned': 9223372036854775808, 'float': 1.5, \
                         'text': 'value', 'list': [1, {'nested': False}]}"
                    ),
                    None,
                    None,
                )
                .unwrap();
            let user_dict = py
                .import("collections")
                .unwrap()
                .getattr("UserDict")
                .unwrap()
                .call1((&source,))
                .unwrap();

            let settings = strict_settings(py, &user_dict).unwrap();

            assert_eq!(
                settings,
                serde_json::from_value(json!({
                    "bool": true,
                    "float": 1.5,
                    "list": [1, {"nested": false}],
                    "none": null,
                    "signed": -7,
                    "text": "value",
                    "unsigned": 9_223_372_036_854_775_808_u64,
                }))
                .unwrap()
            );
            source
                .cast::<PyDict>()
                .unwrap()
                .set_item("text", "changed")
                .unwrap();
            assert_eq!(settings["text"], json!("value"));

            let copied = settings_to_python(py, &settings).unwrap();
            copied
                .cast::<PyDict>()
                .unwrap()
                .set_item("text", "local")
                .unwrap();
            assert_eq!(settings["text"], json!("value"));
        });
    }

    #[test]
    fn strict_settings_rejects_invalid_shapes_cycles_values_and_depth() {
        with_python(|py| {
            let not_mapping = PyList::empty(py);
            let error = expect_error(strict_settings(py, not_mapping.as_any()));
            assert!(error.is_instance_of::<PyTypeError>(py), "{error}");

            let non_string_key = PyDict::new(py);
            non_string_key.set_item(1, "value").unwrap();
            let error = expect_error(strict_settings(py, non_string_key.as_any()));
            assert!(error.is_instance_of::<PyValueError>(py));

            let cyclic_dict = PyDict::new(py);
            cyclic_dict.set_item("self", &cyclic_dict).unwrap();
            let error = expect_error(strict_settings(py, cyclic_dict.as_any()));
            assert!(error.is_instance_of::<PyValueError>(py));

            let cyclic_list = PyList::empty(py);
            cyclic_list.append(&cyclic_list).unwrap();
            let list_root = PyDict::new(py);
            list_root.set_item("value", &cyclic_list).unwrap();
            let error = expect_error(strict_settings(py, list_root.as_any()));
            assert!(error.is_instance_of::<PyValueError>(py));

            for expression in [c_str!("2**64"), c_str!("float('nan')"), c_str!("(1, 2)")] {
                let root = PyDict::new(py);
                root.set_item("value", py.eval(expression, None, None).unwrap())
                    .unwrap();
                let error = expect_error(strict_settings(py, root.as_any()));
                assert!(error.is_instance_of::<PyValueError>(py));
            }

            let mut ancestors = HashSet::new();
            let error = expect_error(strict_dict(
                &PyDict::new(py),
                calc_flow::MAX_JSON_DEPTH + 1,
                &mut ancestors,
            ));
            assert!(error.is_instance_of::<PyValueError>(py));
            let error = expect_error(strict_list(
                &PyList::empty(py),
                calc_flow::MAX_JSON_DEPTH + 1,
                &mut ancestors,
            ));
            assert!(error.is_instance_of::<PyValueError>(py));
            let none = py.None();
            let error = expect_error(strict_value(
                none.bind(py),
                calc_flow::MAX_JSON_DEPTH + 1,
                &mut ancestors,
            ));
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn deadlines_accept_exact_zero_offset_and_reject_invalid_datetime_values() {
        with_python(|py| {
            let none = py.None();
            assert_eq!(parse_deadline(py, none.bind(py)).unwrap(), None);

            let integer = 1_i64.into_pyobject(py).unwrap().into_any();
            let error = expect_error(parse_deadline(py, &integer));
            assert!(error.is_instance_of::<PyTypeError>(py));

            let locals = PyDict::new(py);
            py.run(
                c_str!(
                    "import datetime\nvalid = datetime.datetime(\n    2027, 4, 5, 6, 7, 8, 123456,\n    tzinfo=datetime.timezone(datetime.timedelta(0), 'zero'),\n)\nnaive = datetime.datetime(2027, 4, 5)\nnonzero = datetime.datetime(\n    2027, 4, 5,\n    tzinfo=datetime.timezone(datetime.timedelta(hours=1)),\n)\nclass BrokenOffset(datetime.datetime):\n    def utcoffset(self):\n        raise RuntimeError('broken offset')\nbroken = BrokenOffset(2027, 4, 5, tzinfo=datetime.UTC)\nclass InvalidOffset(datetime.datetime):\n    def utcoffset(self):\n        return object()\ninvalid_offset = InvalidOffset(2027, 4, 5, tzinfo=datetime.UTC)"
                ),
                Some(&locals),
                None,
            )
            .unwrap();

            let valid = locals.get_item("valid").unwrap().unwrap();
            let deadline = parse_deadline(py, &valid).unwrap().unwrap();
            assert_eq!(
                deadline,
                Utc.with_ymd_and_hms(2027, 4, 5, 6, 7, 8)
                    .single()
                    .unwrap()
                    .with_nanosecond(123_456_000)
                    .unwrap()
            );

            for name in ["naive", "nonzero", "broken", "invalid_offset"] {
                let value = locals.get_item(name).unwrap().unwrap();
                let error = expect_error(parse_deadline(py, &value));
                assert!(error.is_instance_of::<PyValueError>(py));
            }

            assert!(deadline_to_python(py, None).unwrap().is_none());
            let round_trip = deadline_to_python(py, Some(&deadline)).unwrap().unwrap();
            assert!(
                round_trip.getattr("tzinfo").unwrap().is(py
                    .import("datetime")
                    .unwrap()
                    .getattr("UTC")
                    .unwrap())
            );
            assert_eq!(
                round_trip
                    .getattr("microsecond")
                    .unwrap()
                    .extract::<u32>()
                    .unwrap(),
                123_456
            );
        });
    }

    #[test]
    fn execution_options_constructor_getters_and_core_values_are_isolated() {
        with_python(|py| {
            let module = PyModule::new(py, "execution_options_test").unwrap();
            register(&module).unwrap();
            let options_type = module.getattr("ExecutionOptions").unwrap();
            let source = py
                .eval(c_str!("{'request': {'attempts': [1, 2]}}"), None, None)
                .unwrap();
            let datetime = py.import("datetime").unwrap();
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("tzinfo", datetime.getattr("UTC").unwrap())
                .unwrap();
            let deadline = datetime
                .getattr("datetime")
                .unwrap()
                .call((2028, 9, 10, 11, 12, 13, 456_789), Some(&kwargs))
                .unwrap();

            let options = options_type.call1((&source, &deadline)).unwrap();
            source
                .cast::<PyDict>()
                .unwrap()
                .set_item("request", "changed")
                .unwrap();
            let first_settings = options.getattr("settings").unwrap();
            first_settings
                .cast::<PyDict>()
                .unwrap()
                .set_item("request", "local")
                .unwrap();
            let second_settings = options.getattr("settings").unwrap();
            assert_eq!(
                second_settings
                    .get_item("request")
                    .unwrap()
                    .get_item("attempts")
                    .unwrap()
                    .extract::<Vec<i64>>()
                    .unwrap(),
                vec![1, 2]
            );
            assert!(options.getattr("deadline").unwrap().eq(&deadline).unwrap());

            let options_ref: PyRef<'_, PyExecutionOptions> = options.extract().unwrap();
            let first_core = options_ref.to_core();
            let second_core = options_ref.to_core();
            assert_eq!(first_core.settings["request"]["attempts"], json!([1, 2]));
            assert_eq!(
                first_core.deadline,
                Some(
                    Utc.with_ymd_and_hms(2028, 9, 10, 11, 12, 13)
                        .single()
                        .unwrap()
                        .with_nanosecond(456_789_000)
                        .unwrap()
                )
            );
            first_core.cancellation.cancel();
            assert!(!second_core.cancellation.is_cancelled());

            let default_options = options_type.call0().unwrap();
            assert!(
                default_options
                    .getattr("settings")
                    .unwrap()
                    .cast::<PyDict>()
                    .unwrap()
                    .is_empty()
            );
            assert!(default_options.getattr("deadline").unwrap().is_none());
            assert!(module.getattr("ProviderContext").is_ok());
        });
    }

    #[test]
    fn execution_options_constructor_preserves_argument_errors() {
        with_python(|py| {
            let module = PyModule::new(py, "execution_options_errors").unwrap();
            register(&module).unwrap();
            let options_type = module.getattr("ExecutionOptions").unwrap();
            let settings = PyDict::new(py);
            let none = py.None();

            let error = expect_error(options_type.call1((&settings, none.bind(py), "extra")));
            assert!(error.is_instance_of::<PyTypeError>(py));

            let unexpected = PyDict::new(py);
            unexpected.set_item("unexpected", true).unwrap();
            let error = expect_error(options_type.call((), Some(&unexpected)));
            assert!(error.is_instance_of::<PyTypeError>(py));

            let duplicate_settings = PyDict::new(py);
            duplicate_settings.set_item("settings", &settings).unwrap();
            let error = expect_error(options_type.call((&settings,), Some(&duplicate_settings)));
            assert!(error.is_instance_of::<PyTypeError>(py));

            let duplicate_deadline = PyDict::new(py);
            duplicate_deadline
                .set_item("deadline", none.bind(py))
                .unwrap();
            let error = expect_error(
                options_type.call((&settings, none.bind(py)), Some(&duplicate_deadline)),
            );
            assert!(error.is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn provider_context_and_cancellation_preserve_run_state() {
        with_python(|py| {
            let deadline = Utc.with_ymd_and_hms(2030, 1, 2, 3, 4, 5).single().unwrap();
            let settings = BTreeMap::from([("request".into(), json!({"id": 7}))]);
            let run = calc_flow::RunContext::new(
                settings.clone(),
                Some(deadline),
                calc_flow::CancellationToken::new(),
            )
            .unwrap();
            let context = PyProviderContext::from_run(&run);

            let first = context.settings(py).unwrap();
            first
                .cast::<PyDict>()
                .unwrap()
                .set_item("request", "local")
                .unwrap();
            assert_eq!(
                context
                    .settings(py)
                    .unwrap()
                    .get_item("request")
                    .unwrap()
                    .get_item("id")
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                7
            );
            assert_eq!(
                context
                    .deadline(py)
                    .unwrap()
                    .unwrap()
                    .getattr("year")
                    .unwrap()
                    .extract::<i32>()
                    .unwrap(),
                2030
            );

            let token = calc_flow::CancellationToken::new();
            let cancellation = PyExecutionCancellation::new(token.clone());
            cancellation.cancel();
            assert!(token.is_cancelled());
        });
    }
}
