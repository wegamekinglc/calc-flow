use calc_flow::array::{ArrayExecutionOptions, AtrOptions, RsiOptions};
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

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(rsi, module)?)?;
    module.add_function(wrap_pyfunction!(atr, module)?)
}
