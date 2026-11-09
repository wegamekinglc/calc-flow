use calc_flow::array::{ArrayExecutionOptions, AtrOptions, MacdOptions, RsiOptions};
use pyo3::prelude::*;

use crate::array::PyNativeArray;

fn execution_limits(max_elements: usize, max_allocation_bytes: usize) -> ArrayExecutionOptions {
    ArrayExecutionOptions::default()
        .with_max_elements(max_elements)
        .with_max_allocation_bytes(max_allocation_bytes)
}

#[pyfunction(name = "_native_array_rsi")]
fn rsi(
    py: Python<'_>,
    array: &PyNativeArray,
    period: u64,
    max_elements: usize,
    max_allocation_bytes: usize,
) -> PyResult<PyNativeArray> {
    let options = RsiOptions::new(period).map_err(crate::error::to_py_err)?;
    let execution = execution_limits(max_elements, max_allocation_bytes);
    let input = &array.inner;
    py.detach(move || calc_flow::array::rsi(input.view(), options, &execution))
        .map(PyNativeArray::from_inner)
        .map_err(crate::error::to_py_err)
}

#[pyfunction(name = "_native_array_atr")]
fn atr(
    py: Python<'_>,
    high: &PyNativeArray,
    low: &PyNativeArray,
    close: &PyNativeArray,
    period: u64,
    max_elements: usize,
    max_allocation_bytes: usize,
) -> PyResult<PyNativeArray> {
    let options = AtrOptions::new(period).map_err(crate::error::to_py_err)?;
    let execution = execution_limits(max_elements, max_allocation_bytes);
    let (high, low, close) = (&high.inner, &low.inner, &close.inner);
    py.detach(move || {
        calc_flow::array::atr(high.view(), low.view(), close.view(), options, &execution)
    })
    .map(PyNativeArray::from_inner)
    .map_err(crate::error::to_py_err)
}

#[pyfunction(name = "_native_array_macd_components")]
fn macd_components(
    py: Python<'_>,
    array: &PyNativeArray,
    options: (u64, u64, u64, u64, u64),
    max_elements: usize,
    max_allocation_bytes: usize,
) -> PyResult<(PyNativeArray, PyNativeArray, PyNativeArray)> {
    let options = MacdOptions::new(options.0, options.1, options.2)
        .and_then(|value| value.with_min_periods(options.3))
        .and_then(|value| value.with_signal_min_periods(options.4))
        .map_err(crate::error::to_py_err)?;
    let execution = execution_limits(max_elements, max_allocation_bytes);
    let input = &array.inner;
    let output = py
        .detach(move || calc_flow::array::macd_components(input.view(), options, &execution))
        .map_err(crate::error::to_py_err)?;
    let (line, signal, histogram) = output.into_arrays();
    Ok((
        PyNativeArray::from_inner(line),
        PyNativeArray::from_inner(signal),
        PyNativeArray::from_inner(histogram),
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(rsi, module)?)?;
    module.add_function(wrap_pyfunction!(atr, module)?)?;
    module.add_function(wrap_pyfunction!(macd_components, module)?)
}
