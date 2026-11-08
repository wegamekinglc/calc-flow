use std::sync::Arc;

use calc_flow::array::{
    ArrayDType, ArrayExecutionOptions, ArrayView, EwmaOptions, NativeArray, ReductionOptions,
    RollingMeanOptions,
};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyCapsule, PyTuple},
};
use pyo3_arrow::PyArray;

/// Immutable CPU array owner with read-only Arrow backing buffers.
#[pyclass(name = "NativeArray", frozen, module = "calc_flow._native")]
pub(crate) struct PyNativeArray {
    pub(crate) inner: NativeArray,
}

impl PyNativeArray {
    pub(crate) fn from_inner(inner: NativeArray) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyNativeArray {
    #[getter]
    fn dtype(&self) -> PyResult<&'static str> {
        match self.inner.dtype() {
            ArrayDType::Float32 => Ok("float32"),
            ArrayDType::Float64 => Ok("float64"),
            ArrayDType::Boolean => Ok("bool"),
            _ => Err(invalid("input.dtype", "unsupported native array dtype")),
        }
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.shape().iter().copied())
    }

    #[getter]
    fn estimated_bytes(&self) -> usize {
        self.inner.estimated_bytes()
    }

    #[getter]
    fn retained_buffer_bytes(&self) -> usize {
        self.inner.retained_buffer_bytes()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    /// Export Arrow values and preserved validity without importing `NumPy`.
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyArray::from_array_ref(Arc::clone(self.inner.as_arrow()))
            .to_pyarrow(py)?
            .unbind())
    }

    /// Copy into immutable `NumPy` storage; nulls require an explicit `fill_value`.
    #[pyo3(signature = (*, fill_value=None))]
    fn to_numpy(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        fill_value: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(py
            .import("calc_flow._native_array")?
            .getattr("_to_numpy")?
            .call1((slf, fill_value))?
            .unbind())
    }
}

fn invalid(field: &str, message: &str) -> PyErr {
    crate::error::to_py_err(calc_flow::CalcFlowError::InvalidArgument {
        field: field.into(),
        message: message.into(),
    })
}

#[pyfunction]
fn _native_array_from_bytes(
    values: &Bound<'_, PyBytes>,
    dtype: &str,
    shape: Vec<usize>,
) -> PyResult<PyNativeArray> {
    let values = values.as_bytes();
    let width = float_width(dtype)?;
    if !values.len().is_multiple_of(width) {
        return Err(invalid("input.bytes", "incomplete typed value"));
    }
    let len = values.len() / width;
    if shape.len() > 1 || shape.first().copied().unwrap_or(1) != len {
        return Err(invalid(
            "input.shape",
            "expected scalar or matching vector shape",
        ));
    }
    let array = match dtype {
        "float32" => NativeArray::from_f32(
            values
                .chunks_exact(4)
                .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
            shape,
            None,
        ),
        _ => NativeArray::from_f64(
            values
                .chunks_exact(8)
                .map(|chunk| {
                    f64::from_ne_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect(),
            shape,
            None,
        ),
    }
    .map_err(crate::error::to_py_err)?;
    Ok(PyNativeArray::from_inner(array))
}

fn float_width(dtype: &str) -> PyResult<usize> {
    match dtype {
        "float32" => Ok(4),
        "float64" => Ok(8),
        _ => Err(invalid("input.dtype", "expected float32 or float64")),
    }
}

#[pyfunction]
fn _native_array_from_arrow(
    values: &Bound<'_, PyAny>,
    shape: Vec<usize>,
) -> PyResult<PyNativeArray> {
    let (array, _) = import_arrow(values)?.into_inner();
    NativeArray::from_arrow(array, shape)
        .map(PyNativeArray::from_inner)
        .map_err(crate::error::to_py_err)
}

fn import_arrow(values: &Bound<'_, PyAny>) -> PyResult<PyArray> {
    let capsules = values.call_method0("__arrow_c_array__")?;
    let (schema, array): (Bound<'_, PyCapsule>, Bound<'_, PyCapsule>) = capsules.extract()?;
    PyArray::from_arrow_pycapsule(&schema, &array)
}

#[pyfunction]
fn _native_array_reduce(
    py: Python<'_>,
    array: &PyNativeArray,
    operation: &str,
    skip_nulls: bool,
    max_elements: usize,
    max_allocation_bytes: usize,
) -> PyResult<PyNativeArray> {
    let function = match operation {
        "sum" => calc_flow::array::sum,
        "mean" => calc_flow::array::mean,
        "min" => calc_flow::array::min,
        "max" => calc_flow::array::max,
        _ => return Err(invalid("operation", "unsupported native reduction")),
    };
    let input = array.inner.clone();
    let execution = ArrayExecutionOptions::default()
        .with_max_elements(max_elements)
        .with_max_allocation_bytes(max_allocation_bytes);
    py.detach(move || {
        function(
            input.float_view()?,
            ReductionOptions::default().with_skip_nulls(skip_nulls),
            &execution,
        )
    })
    .map(PyNativeArray::from_inner)
    .map_err(crate::error::to_py_err)
}

#[pyfunction]
fn _native_array_ewma(
    py: Python<'_>,
    array: &PyNativeArray,
    span: u64,
    min_periods: u64,
    max_elements: usize,
    max_allocation_bytes: usize,
) -> PyResult<PyNativeArray> {
    let options = EwmaOptions::new(span)
        .and_then(|options| options.with_min_periods(min_periods))
        .map_err(crate::error::to_py_err)?;
    let execution = ArrayExecutionOptions::default()
        .with_max_elements(max_elements)
        .with_max_allocation_bytes(max_allocation_bytes);
    let input = array.inner.clone();
    py.detach(move || calc_flow::array::ewma(input.float_view()?, options, &execution))
        .map(PyNativeArray::from_inner)
        .map_err(crate::error::to_py_err)
}

#[pyfunction]
fn _native_array_rolling_mean(
    py: Python<'_>,
    array: &PyNativeArray,
    window: u64,
    min_periods: u64,
    numerical_profile: &str,
    max_elements: usize,
    max_allocation_bytes: usize,
) -> PyResult<PyNativeArray> {
    let profile = match numerical_profile {
        "stable_v1" => calc_flow::RollingNumericalProfile::StableV1,
        "stable_v2" => calc_flow::RollingNumericalProfile::StableV2Preview,
        _ => {
            return Err(invalid(
                "options.numerical_profile",
                "expected stable_v1 or stable_v2",
            ));
        }
    };
    let options = RollingMeanOptions::new(window)
        .and_then(|options| options.with_min_periods(min_periods))
        .map_err(crate::error::to_py_err)?
        .with_numerical_profile(profile);
    let input = array.inner.clone();
    let execution = execution_limits(max_elements, max_allocation_bytes);
    py.detach(move || calc_flow::array::rolling_mean(input.float_view()?, options, &execution))
        .map(PyNativeArray::from_inner)
        .map_err(crate::error::to_py_err)
}

fn execution_limits(max_elements: usize, max_allocation_bytes: usize) -> ArrayExecutionOptions {
    ArrayExecutionOptions::default()
        .with_max_elements(max_elements)
        .with_max_allocation_bytes(max_allocation_bytes)
}

type UnaryFunction =
    for<'a, 'b> fn(ArrayView<'a>, &'b ArrayExecutionOptions) -> calc_flow::Result<NativeArray>;
type BinaryFunction = for<'a, 'b, 'c> fn(
    ArrayView<'a>,
    ArrayView<'b>,
    &'c ArrayExecutionOptions,
) -> calc_flow::Result<NativeArray>;

fn unary_function(operation: &str) -> PyResult<UnaryFunction> {
    Ok(match operation {
        "neg" => calc_flow::array::neg,
        "abs" => calc_flow::array::abs,
        "sqrt" => calc_flow::array::sqrt,
        "log" => calc_flow::array::log,
        "exp" => calc_flow::array::exp,
        "logical_not" => calc_flow::array::logical_not,
        _ => return Err(invalid("operation", "unsupported native unary operation")),
    })
}

fn binary_function(operation: &str) -> PyResult<BinaryFunction> {
    Ok(match operation {
        "add" => calc_flow::array::add,
        "subtract" => calc_flow::array::subtract,
        "multiply" => calc_flow::array::multiply,
        "divide" => calc_flow::array::divide,
        "pow" => calc_flow::array::pow,
        _ => return comparison_function(operation),
    })
}

fn comparison_function(operation: &str) -> PyResult<BinaryFunction> {
    Ok(match operation {
        "equal" => calc_flow::array::equal,
        "not_equal" => calc_flow::array::not_equal,
        "less" => calc_flow::array::less,
        "less_equal" => calc_flow::array::less_equal,
        "greater" => calc_flow::array::greater,
        "greater_equal" => calc_flow::array::greater_equal,
        _ => return logical_function(operation),
    })
}

fn logical_function(operation: &str) -> PyResult<BinaryFunction> {
    Ok(match operation {
        "logical_and" => calc_flow::array::logical_and,
        "logical_or" => calc_flow::array::logical_or,
        "logical_xor" => calc_flow::array::logical_xor,
        _ => return Err(invalid("operation", "unsupported native binary operation")),
    })
}

#[pyfunction]
fn _native_array_unary(
    py: Python<'_>,
    array: &PyNativeArray,
    operation: &str,
    max_elements: usize,
    max_allocation_bytes: usize,
) -> PyResult<PyNativeArray> {
    let function = unary_function(operation)?;
    let input = array.inner.clone();
    let execution = execution_limits(max_elements, max_allocation_bytes);
    py.detach(move || function(input.view(), &execution))
        .map(PyNativeArray::from_inner)
        .map_err(crate::error::to_py_err)
}

#[pyfunction]
fn _native_array_binary(
    py: Python<'_>,
    left: &PyNativeArray,
    right: &PyNativeArray,
    operation: &str,
    max_elements: usize,
    max_allocation_bytes: usize,
) -> PyResult<PyNativeArray> {
    let function = binary_function(operation)?;
    let left = left.inner.clone();
    let right = right.inner.clone();
    let execution = execution_limits(max_elements, max_allocation_bytes);
    py.detach(move || function(left.view(), right.view(), &execution))
        .map(PyNativeArray::from_inner)
        .map_err(crate::error::to_py_err)
}

#[pyfunction]
fn _native_array_where(
    py: Python<'_>,
    condition: &PyNativeArray,
    if_true: &PyNativeArray,
    if_false: &PyNativeArray,
    max_elements: usize,
    max_allocation_bytes: usize,
) -> PyResult<PyNativeArray> {
    let condition = condition.inner.clone();
    let if_true = if_true.inner.clone();
    let if_false = if_false.inner.clone();
    let execution = execution_limits(max_elements, max_allocation_bytes);
    py.detach(move || {
        calc_flow::array::where_(
            condition.view(),
            if_true.view(),
            if_false.view(),
            &execution,
        )
    })
    .map(PyNativeArray::from_inner)
    .map_err(crate::error::to_py_err)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativeArray>()?;
    module.add_function(wrap_pyfunction!(_native_array_from_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(_native_array_from_arrow, module)?)?;
    register_calculations(module)
}

fn register_calculations(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_native_array_reduce, module)?)?;
    module.add_function(wrap_pyfunction!(_native_array_ewma, module)?)?;
    module.add_function(wrap_pyfunction!(_native_array_rolling_mean, module)?)?;
    register_elementwise(module)
}

fn register_elementwise(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_native_array_unary, module)?)?;
    module.add_function(wrap_pyfunction!(_native_array_binary, module)?)?;
    module.add_function(wrap_pyfunction!(_native_array_where, module)?)
}
