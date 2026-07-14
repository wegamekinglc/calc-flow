use std::{fmt, sync::Arc};

use datafusion::{
    arrow::{array::ArrayRef, datatypes::DataType},
    common::ScalarValue,
    error::DataFusionError,
    logical_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarFunctionImplementation, ScalarUDF, ScalarUDFImpl,
        Signature, SimpleScalarUDF, Volatility,
    },
};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    intern,
    prelude::*,
    types::{PyAny, PyTuple},
};
use pyo3_arrow::{PyArray, PyDataType};

use crate::config::{PythonRoot, UdfCatalogMetadata};

const MAX_UDF_ARGUMENTS: usize = 64;

pub(crate) struct PreparedPythonUdf {
    pub(crate) reference: calc_flow::UdfReference,
    pub(crate) udf: Arc<ScalarUDF>,
    pub(crate) root: Arc<PythonRoot>,
    pub(crate) metadata: UdfCatalogMetadata,
}

#[allow(
    clippy::too_many_arguments,
    reason = "the constructor validates every field of the explicit Python UDF contract"
)]
pub(crate) fn prepare_python_scalar_udf(
    py: Python<'_>,
    provider: &str,
    name: &str,
    version: &str,
    input_type_names: Vec<String>,
    return_type_name: String,
    volatility_name: String,
    function: Py<PyAny>,
) -> PyResult<PreparedPythonUdf> {
    if input_type_names.len() > MAX_UDF_ARGUMENTS {
        return Err(PyValueError::new_err(format!(
            "input_types must contain at most {MAX_UDF_ARGUMENTS} entries"
        )));
    }
    if !function.bind(py).is_callable() {
        return Err(PyTypeError::new_err("function must be callable"));
    }
    let reference = calc_flow::UdfReference::new(
        provider,
        name,
        version,
        calc_flow::UdfKind::DataFusionScalar,
    )
    .map_err(crate::error::to_py_err)?;
    let input_types = input_type_names
        .iter()
        .map(|name| {
            parse_arrow_type(name)
                .ok_or_else(|| PyValueError::new_err(format!("unsupported Arrow type {name:?}")))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let return_type = parse_arrow_type(&return_type_name).ok_or_else(|| {
        PyValueError::new_err(format!("unsupported Arrow type {return_type_name:?}"))
    })?;
    let volatility = parse_volatility(&volatility_name)?;
    let root = Arc::new(PythonRoot::new(function));
    let udf = python_scalar_udf(
        &reference,
        input_types,
        return_type,
        volatility,
        Arc::clone(&root),
    );
    Ok(PreparedPythonUdf {
        reference,
        udf,
        root,
        metadata: UdfCatalogMetadata {
            provider: provider.into(),
            name: name.into(),
            version: version.into(),
            input_types: input_type_names,
            return_type: return_type_name,
            volatility: volatility_name,
        },
    })
}

fn python_scalar_udf(
    reference: &calc_flow::UdfReference,
    input_types: Vec<DataType>,
    return_type: DataType,
    volatility: Volatility,
    root: Arc<PythonRoot>,
) -> Arc<ScalarUDF> {
    let name = reference.name().to_owned();
    let identity = format!(
        "{}:{}@{}",
        reference.provider(),
        reference.name(),
        reference.version()
    );
    let argument_count = input_types.len();
    let expected_inputs = input_types.clone();
    let expected_return = return_type.clone();
    let callback_identity = identity.clone();
    let implementation: ScalarFunctionImplementation = Arc::new(move |arguments| {
        invoke_python_udf(
            &callback_identity,
            &root,
            &expected_inputs,
            &expected_return,
            argument_count,
            arguments,
        )
        .map_err(|error| {
            DataFusionError::Execution(format!("python UDF {callback_identity} failed: {error}"))
        })
    });
    let inner = SimpleScalarUDF::new_with_signature(
        &name,
        Signature::user_defined(volatility),
        return_type,
        implementation,
    );
    Arc::new(ScalarUDF::from(ExactPythonScalarUdf {
        name,
        identity,
        signature: Signature::user_defined(volatility),
        expected_inputs: input_types,
        inner,
    }))
}

#[derive(PartialEq, Eq, Hash)]
struct ExactPythonScalarUdf {
    name: String,
    identity: String,
    signature: Signature,
    expected_inputs: Vec<DataType>,
    inner: SimpleScalarUDF,
}

impl fmt::Debug for ExactPythonScalarUdf {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExactPythonScalarUdf")
            .field("name", &self.name)
            .field("identity", &self.identity)
            .field("signature", &self.signature)
            .field("expected_inputs", &self.expected_inputs)
            .field("inner", &"<PYTHON CALLBACK REDACTED>")
            .finish()
    }
}

impl ScalarUDFImpl for ExactPythonScalarUdf {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> datafusion::common::Result<DataType> {
        self.inner.return_type(arg_types)
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> datafusion::common::Result<Vec<DataType>> {
        if arg_types == self.expected_inputs {
            return Ok(self.expected_inputs.clone());
        }
        let expected = describe_argument_types(&self.expected_inputs);
        let actual = describe_argument_types(arg_types);
        Err(DataFusionError::Plan(format!(
            "{} requires exact Arrow input types {expected}; received {actual}",
            self.identity
        )))
    }

    fn invoke_with_args(
        &self,
        arguments: ScalarFunctionArgs,
    ) -> datafusion::common::Result<ColumnarValue> {
        self.inner.invoke_with_args(arguments)
    }
}

fn describe_argument_types(types: &[DataType]) -> String {
    match types {
        [] => "zero arguments".into(),
        [single] => arrow_type_name(single).into(),
        _ => format!(
            "{} arguments ({})",
            types.len(),
            types
                .iter()
                .map(arrow_type_name)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn invoke_python_udf(
    identity: &str,
    root: &Arc<PythonRoot>,
    expected_inputs: &[DataType],
    expected_return: &DataType,
    argument_count: usize,
    arguments: &[ColumnarValue],
) -> Result<ColumnarValue, String> {
    let (callback_arguments, length, all_scalar) =
        normalize_arguments(identity, expected_inputs, argument_count, arguments)?;
    Python::attach(|py| {
        let py_arguments = callback_arguments
            .into_iter()
            .map(|array| {
                PyArray::from_array_ref(array)
                    .to_pyarrow(py)
                    .map(Bound::unbind)
            })
            .collect::<PyResult<Vec<_>>>()?;
        let tuple = PyTuple::new(py, py_arguments)?;
        let output = root.object().call1(py, tuple)?;
        python_output_to_columnar(py, output.bind(py), expected_return, length, all_scalar)
            .map_err(PyValueError::new_err)
    })
    .map_err(|error| error.to_string())
}

fn normalize_arguments(
    identity: &str,
    expected_inputs: &[DataType],
    argument_count: usize,
    arguments: &[ColumnarValue],
) -> Result<(Vec<ArrayRef>, usize, bool), String> {
    if argument_count == 0 {
        let length = match arguments {
            [] | [ColumnarValue::Scalar(ScalarValue::Null)] => 1,
            [ColumnarValue::Array(array)] if array.data_type() == &DataType::Null => array.len(),
            _ => {
                return Err(format!(
                    "{identity} expected zero arguments; received {}",
                    arguments.len()
                ));
            }
        };
        return Ok((Vec::new(), length, arguments.is_empty()));
    }
    if arguments.len() != argument_count {
        return Err(format!(
            "{identity} expected {argument_count} arguments; received {}",
            arguments.len()
        ));
    }
    let array_lengths = arguments
        .iter()
        .filter_map(|value| match value {
            ColumnarValue::Array(array) => Some(array.len()),
            ColumnarValue::Scalar(_) => None,
        })
        .collect::<Vec<_>>();
    let length = array_lengths.first().copied().unwrap_or(1);
    if array_lengths.iter().any(|candidate| *candidate != length) {
        return Err(format!("{identity} received arrays with unequal lengths"));
    }
    let all_scalar = array_lengths.is_empty();
    let arrays = arguments
        .iter()
        .zip(expected_inputs)
        .map(|(value, expected)| {
            let actual = match value {
                ColumnarValue::Array(array) => array.data_type().clone(),
                ColumnarValue::Scalar(value) => value.data_type(),
            };
            if &actual != expected {
                return Err(format!(
                    "{identity} argument Arrow type must be {}; received {actual}",
                    arrow_type_name(expected)
                ));
            }
            match value {
                ColumnarValue::Array(array) => Ok(Arc::clone(array)),
                ColumnarValue::Scalar(value) => value
                    .to_array_of_size(length)
                    .map_err(|error| error.to_string()),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((arrays, length, all_scalar))
}

fn python_scalar_value<'py>(
    output: &Bound<'py, PyAny>,
    scalar_type: &Bound<'py, PyAny>,
    expected_type: &DataType,
) -> Result<Bound<'py, PyAny>, String> {
    if !output
        .is_instance(scalar_type)
        .map_err(|error| error.to_string())?
    {
        return Ok(output.clone());
    }
    let actual_type = output
        .getattr(intern!(output.py(), "type"))
        .and_then(|value| value.extract::<PyDataType>())
        .map(PyDataType::into_inner)
        .map_err(|error| error.to_string())?;
    if &actual_type != expected_type {
        return Err(format!(
            "output Arrow type must be {}; received {actual_type}",
            arrow_type_name(expected_type)
        ));
    }
    output
        .call_method0(intern!(output.py(), "as_py"))
        .map_err(|error| error.to_string())
}

fn python_output_to_columnar(
    py: Python<'_>,
    output: &Bound<'_, PyAny>,
    expected_type: &DataType,
    length: usize,
    scalar_mode: bool,
) -> Result<ColumnarValue, String> {
    let pyarrow = py
        .import(intern!(py, "pyarrow"))
        .map_err(|error| error.to_string())?;
    let array_type = pyarrow
        .getattr(intern!(py, "Array"))
        .map_err(|error| error.to_string())?;
    let scalar_type = pyarrow
        .getattr(intern!(py, "Scalar"))
        .map_err(|error| error.to_string())?;
    let (array, scalar_result) = if output
        .is_instance(&array_type)
        .map_err(|error| error.to_string())?
    {
        let imported = output
            .extract::<PyArray>()
            .map_err(|error| error.to_string())?;
        (Arc::clone(imported.array()), false)
    } else {
        if output
            .hasattr(intern!(py, "__arrow_c_stream__"))
            .map_err(|error| error.to_string())?
        {
            return Err(
                "output must be a PyArrow array or scalar, not a chunked/table value".into(),
            );
        }
        let value = python_scalar_value(output, &scalar_type, expected_type)?;
        let data_type = PyDataType::new(expected_type.clone())
            .into_pyarrow(py)
            .map_err(|error| error.to_string())?;
        let values = pyo3::types::PyList::new(py, [value]).map_err(|error| error.to_string())?;
        let keyword_arguments = pyo3::types::PyDict::new(py);
        keyword_arguments
            .set_item(intern!(py, "type"), data_type)
            .map_err(|error| error.to_string())?;
        let converted = pyarrow
            .getattr(intern!(py, "array"))
            .and_then(|constructor| constructor.call((values,), Some(&keyword_arguments)))
            .map_err(|error| error.to_string())?;
        let imported = converted
            .extract::<PyArray>()
            .map_err(|error| error.to_string())?;
        (Arc::clone(imported.array()), true)
    };
    if array.data_type() != expected_type {
        return Err(format!(
            "output Arrow type must be {}; received {}",
            arrow_type_name(expected_type),
            array.data_type()
        ));
    }
    if scalar_result {
        let scalar =
            ScalarValue::try_from_array(array.as_ref(), 0).map_err(|error| error.to_string())?;
        return if scalar_mode {
            Ok(ColumnarValue::Scalar(scalar))
        } else {
            scalar
                .to_array_of_size(length)
                .map(ColumnarValue::Array)
                .map_err(|error| error.to_string())
        };
    }
    if array.len() != length {
        return Err(format!(
            "output length must be {length}; received {}",
            array.len()
        ));
    }
    if scalar_mode {
        if array.len() != 1 {
            return Err(format!(
                "output length must be 1 in scalar mode; received {}",
                array.len()
            ));
        }
        ScalarValue::try_from_array(array.as_ref(), 0)
            .map(ColumnarValue::Scalar)
            .map_err(|error| error.to_string())
    } else {
        Ok(ColumnarValue::Array(array))
    }
}

fn parse_volatility(value: &str) -> PyResult<Volatility> {
    match value {
        "immutable" => Ok(Volatility::Immutable),
        "stable" => Ok(Volatility::Stable),
        "volatile" => Ok(Volatility::Volatile),
        _ => Err(PyValueError::new_err(
            "volatility must be 'immutable', 'stable', or 'volatile'",
        )),
    }
}

pub(crate) fn parse_arrow_type(value: &str) -> Option<DataType> {
    use datafusion::arrow::datatypes::TimeUnit;
    Some(match value {
        "bool" => DataType::Boolean,
        "date32" => DataType::Date32,
        "date64" => DataType::Date64,
        "float32" => DataType::Float32,
        "float64" => DataType::Float64,
        "int8" => DataType::Int8,
        "int16" => DataType::Int16,
        "int32" => DataType::Int32,
        "int64" => DataType::Int64,
        "large_string" => DataType::LargeUtf8,
        "string" => DataType::Utf8,
        "time32[s]" => DataType::Time32(TimeUnit::Second),
        "time64[us]" => DataType::Time64(TimeUnit::Microsecond),
        "timestamp[ms]" => DataType::Timestamp(TimeUnit::Millisecond, None),
        "timestamp[us]" => DataType::Timestamp(TimeUnit::Microsecond, None),
        "uint8" => DataType::UInt8,
        "uint16" => DataType::UInt16,
        "uint32" => DataType::UInt32,
        "uint64" => DataType::UInt64,
        _ => return None,
    })
}

pub(crate) fn arrow_type_name(value: &DataType) -> &'static str {
    use datafusion::arrow::datatypes::TimeUnit;
    match value {
        DataType::Boolean => "bool",
        DataType::Date32 => "date32",
        DataType::Date64 => "date64",
        DataType::Float32 => "float32",
        DataType::Float64 => "float64",
        DataType::Int8 => "int8",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::LargeUtf8 => "large_string",
        DataType::Utf8 => "string",
        DataType::Time32(TimeUnit::Second) => "time32[s]",
        DataType::Time64(TimeUnit::Microsecond) => "time64[us]",
        DataType::Timestamp(TimeUnit::Millisecond, None) => "timestamp[ms]",
        DataType::Timestamp(TimeUnit::Microsecond, None) => "timestamp[us]",
        DataType::UInt8 => "uint8",
        DataType::UInt16 => "uint16",
        DataType::UInt32 => "uint32",
        DataType::UInt64 => "uint64",
        _ => "unsupported",
    }
}

#[cfg(test)]
mod tests {
    use datafusion::{
        arrow::{
            array::{Float64Array, Int64Array, NullArray},
            datatypes::Field,
        },
        config::ConfigOptions,
        logical_expr::ScalarFunctionArgs,
    };

    use super::*;

    #[test]
    fn supported_arrow_type_names_round_trip_exactly() {
        for name in [
            "bool",
            "date32",
            "date64",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "large_string",
            "string",
            "time32[s]",
            "time64[us]",
            "timestamp[ms]",
            "timestamp[us]",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        ] {
            let parsed = parse_arrow_type(name).unwrap();
            assert_eq!(arrow_type_name(&parsed), name);
        }
        assert!(parse_arrow_type("object").is_none());
    }

    #[test]
    fn argument_normalization_broadcasts_scalars_and_checks_types_and_lengths() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        let scalar = ScalarValue::Int64(Some(3));
        let (arguments, length, all_scalar) = normalize_arguments(
            "python:add@1",
            &[DataType::Int64, DataType::Int64],
            2,
            &[
                ColumnarValue::Array(Arc::clone(&values)),
                ColumnarValue::Scalar(scalar),
            ],
        )
        .unwrap();
        assert_eq!(arguments.len(), 2);
        assert_eq!(arguments[1].len(), 2);
        assert_eq!(length, 2);
        assert!(!all_scalar);

        let wrong: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        let error = normalize_arguments(
            "python:add@1",
            &[DataType::Int64],
            1,
            &[ColumnarValue::Array(wrong)],
        )
        .unwrap_err();
        assert!(error.contains("argument Arrow type"));

        let short: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        let error = normalize_arguments(
            "python:add@1",
            &[DataType::Int64, DataType::Int64],
            2,
            &[ColumnarValue::Array(values), ColumnarValue::Array(short)],
        )
        .unwrap_err();
        assert!(error.contains("unequal lengths"));
    }

    #[test]
    fn zero_argument_dummy_preserves_datafusion_row_count() {
        let dummy: ArrayRef = Arc::new(NullArray::new(4));
        let (arguments, length, all_scalar) =
            normalize_arguments("python:constant@1", &[], 0, &[ColumnarValue::Array(dummy)])
                .unwrap();
        assert!(arguments.is_empty());
        assert_eq!(length, 4);
        assert!(!all_scalar);
    }

    #[test]
    fn pyarrow_output_conversion_handles_arrays_scalars_and_rejects_streams() {
        Python::initialize();
        Python::attach(|py| {
            let pyarrow = py.import("pyarrow").unwrap();
            let array = pyarrow
                .getattr("array")
                .unwrap()
                .call1((vec![1_i64, 2],))
                .unwrap();
            let converted =
                python_output_to_columnar(py, &array, &DataType::Int64, 2, false).unwrap();
            assert!(matches!(converted, ColumnarValue::Array(_)));

            let scalar = pyarrow.getattr("scalar").unwrap().call1((3_i64,)).unwrap();
            let converted =
                python_output_to_columnar(py, &scalar, &DataType::Int64, 5, false).unwrap();
            let ColumnarValue::Array(values) = converted else {
                panic!("an array invocation must broadcast scalar output");
            };
            assert_eq!(values.len(), 5);

            let chunked = pyarrow
                .getattr("chunked_array")
                .unwrap()
                .call1((vec![vec![1_i64], vec![2]],))
                .unwrap();
            let error =
                python_output_to_columnar(py, &chunked, &DataType::Int64, 2, false).unwrap_err();
            assert!(error.contains("chunked/table"));
        });
    }

    #[test]
    fn prepared_udf_and_runtime_share_one_python_root_allocation() {
        Python::initialize();
        Python::attach(|py| {
            let function = py
                .eval(pyo3::ffi::c_str!("lambda value: value"), None, None)
                .unwrap()
                .unbind();
            let prepared = prepare_python_scalar_udf(
                py,
                "python",
                "identity",
                "1",
                vec!["int64".into()],
                "int64".into(),
                "immutable".into(),
                function,
            )
            .unwrap();
            assert!(Arc::strong_count(&prepared.root) >= 2);
        });
    }

    #[test]
    fn preparation_rejects_every_invalid_registration_before_rooting() {
        Python::initialize();
        Python::attach(|py| {
            let callback = || {
                py.eval(pyo3::ffi::c_str!("lambda value: value"), None, None)
                    .unwrap()
                    .unbind()
            };
            let prepare = |input_types: Vec<String>,
                           return_type: &str,
                           volatility: &str,
                           function: Py<PyAny>| {
                prepare_python_scalar_udf(
                    py,
                    "python",
                    "identity",
                    "1",
                    input_types,
                    return_type.into(),
                    volatility.into(),
                    function,
                )
            };

            assert!(
                prepare(
                    vec!["int64".into(); MAX_UDF_ARGUMENTS + 1],
                    "int64",
                    "immutable",
                    callback()
                )
                .is_err()
            );
            assert!(prepare(vec![], "int64", "immutable", py.None()).is_err());
            assert!(prepare(vec!["object".into()], "int64", "immutable", callback()).is_err());
            assert!(prepare(vec!["int64".into()], "object", "immutable", callback()).is_err());
            assert!(prepare(vec!["int64".into()], "int64", "sometimes", callback()).is_err());
            assert!(
                prepare_python_scalar_udf(
                    py,
                    "not valid!",
                    "identity",
                    "1",
                    vec!["int64".into()],
                    "int64".into(),
                    "immutable".into(),
                    callback(),
                )
                .is_err()
            );
            assert!(matches!(parse_volatility("stable"), Ok(Volatility::Stable)));
            assert!(matches!(
                parse_volatility("volatile"),
                Ok(Volatility::Volatile)
            ));
        });
    }

    #[test]
    fn native_scalar_udf_invocation_uses_capsules_and_wraps_callback_failures() {
        Python::initialize();
        Python::attach(|py| {
            let function = py
                .eval(pyo3::ffi::c_str!("lambda value: value"), None, None)
                .unwrap()
                .unbind();
            let prepared = prepare_python_scalar_udf(
                py,
                "python",
                "identity",
                "1",
                vec!["int64".into()],
                "int64".into(),
                "immutable".into(),
                function,
            )
            .unwrap();
            let values: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
            let result = prepared
                .udf
                .invoke_with_args(ScalarFunctionArgs {
                    args: vec![ColumnarValue::Array(Arc::clone(&values))],
                    arg_fields: vec![Arc::new(Field::new("value", DataType::Int64, true))],
                    number_rows: 2,
                    return_field: Arc::new(Field::new("result", DataType::Int64, true)),
                    config_options: Arc::new(ConfigOptions::new()),
                })
                .unwrap();
            let ColumnarValue::Array(result) = result else {
                panic!("array input must produce array output");
            };
            assert_eq!(result.len(), 2);

            let failing = py
                .eval(
                    pyo3::ffi::c_str!("lambda value: (_ for _ in ()).throw(RuntimeError('boom'))"),
                    None,
                    None,
                )
                .unwrap()
                .unbind();
            let root = Arc::new(PythonRoot::new(failing));
            let error = invoke_python_udf(
                "python:fail@1",
                &root,
                &[DataType::Int64],
                &DataType::Int64,
                1,
                &[ColumnarValue::Array(values)],
            )
            .unwrap_err();
            assert!(error.contains("boom"));
        });
    }

    #[test]
    fn output_and_argument_errors_cover_exact_scalar_array_contracts() {
        Python::initialize();
        Python::attach(|py| {
            let pyarrow = py.import("pyarrow").unwrap();
            let array =
                |values: Vec<i64>| pyarrow.getattr("array").unwrap().call1((values,)).unwrap();
            let wrong_type = pyarrow
                .getattr("array")
                .unwrap()
                .call1((vec![1.0_f64],))
                .unwrap();
            assert!(
                python_output_to_columnar(py, &wrong_type, &DataType::Int64, 1, false)
                    .unwrap_err()
                    .contains("output Arrow type")
            );
            let wrong_scalar = pyarrow
                .getattr("scalar")
                .unwrap()
                .call1((1.0_f64,))
                .unwrap();
            assert!(
                python_output_to_columnar(py, &wrong_scalar, &DataType::Int64, 1, false)
                    .unwrap_err()
                    .contains("output Arrow type")
            );
            assert!(
                python_output_to_columnar(py, &array(vec![1, 2]), &DataType::Int64, 1, false)
                    .unwrap_err()
                    .contains("output length")
            );
            assert!(
                python_output_to_columnar(py, &array(vec![1, 2]), &DataType::Int64, 2, true)
                    .unwrap_err()
                    .contains("scalar mode")
            );
            let scalar = python_output_to_columnar(
                py,
                &1_i64.into_pyobject(py).unwrap().into_any(),
                &DataType::Int64,
                1,
                true,
            )
            .unwrap();
            assert!(matches!(scalar, ColumnarValue::Scalar(_)));
        });

        assert!(
            normalize_arguments(
                "python:constant@1",
                &[],
                0,
                &[ColumnarValue::Scalar(ScalarValue::Int64(Some(1)))],
            )
            .unwrap_err()
            .contains("expected zero arguments")
        );
        assert!(
            normalize_arguments("python:identity@1", &[DataType::Int64], 1, &[])
                .unwrap_err()
                .contains("expected 1 arguments")
        );
        assert_eq!(arrow_type_name(&DataType::Binary), "unsupported");
    }
}
