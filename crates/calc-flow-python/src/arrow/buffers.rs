//! Typed NumPy/Arrow copies with unchanged snapshot and matrix allocation bounds.

use std::sync::Arc;

use datafusion::arrow::{
    array::{
        Array, ArrayRef, PrimitiveArray,
        types::{
            Float16Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
            UInt8Type, UInt16Type, UInt32Type, UInt64Type,
        },
    },
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
};
use ndarray::ArrayViewMut2;
use numpy::{Element, PyArrayMethods};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyModule},
};

macro_rules! numpy_primitive {
    ($value:expr, $native:ty, $arrow:ty) => {{
        let array = $value.cast::<numpy::PyArray1<$native>>()?.try_readonly()?;
        Ok(Arc::new(PrimitiveArray::<$arrow>::from_iter_values(
            array.as_array().iter().copied(),
        )) as ArrayRef)
    }};
}

pub(crate) fn numpy_array(value: &Bound<'_, PyAny>) -> PyResult<ArrayRef> {
    let dtype = value
        .getattr("dtype")?
        .getattr("name")?
        .extract::<String>()?;
    match dtype.as_bytes().first() {
        Some(b'f') => floating_array(value, &dtype),
        Some(b'i') => signed_array(value, &dtype),
        Some(b'u') => unsigned_array(value, &dtype),
        _ => boolean_array(value, &dtype),
    }
}

fn floating_array(value: &Bound<'_, PyAny>, dtype: &str) -> PyResult<ArrayRef> {
    match dtype {
        "float32" => numpy_primitive!(value, f32, Float32Type),
        "float64" => numpy_primitive!(value, f64, Float64Type),
        _ => Err(unsupported_array_dtype()),
    }
}

fn signed_array(value: &Bound<'_, PyAny>, dtype: &str) -> PyResult<ArrayRef> {
    match dtype {
        "int8" => numpy_primitive!(value, i8, Int8Type),
        "int16" => numpy_primitive!(value, i16, Int16Type),
        "int32" => numpy_primitive!(value, i32, Int32Type),
        "int64" => numpy_primitive!(value, i64, Int64Type),
        _ => Err(unsupported_array_dtype()),
    }
}

fn unsigned_array(value: &Bound<'_, PyAny>, dtype: &str) -> PyResult<ArrayRef> {
    match dtype {
        "uint8" => numpy_primitive!(value, u8, UInt8Type),
        "uint16" => numpy_primitive!(value, u16, UInt16Type),
        "uint32" => numpy_primitive!(value, u32, UInt32Type),
        "uint64" => numpy_primitive!(value, u64, UInt64Type),
        _ => Err(unsupported_array_dtype()),
    }
}

fn boolean_array(value: &Bound<'_, PyAny>, dtype: &str) -> PyResult<ArrayRef> {
    if dtype != "bool" {
        return Err(unsupported_array_dtype());
    }
    let array = value.cast::<numpy::PyArray1<bool>>()?.try_readonly()?;
    Ok(Arc::new(
        array
            .as_array()
            .iter()
            .copied()
            .map(Some)
            .collect::<datafusion::arrow::array::BooleanArray>(),
    ))
}

fn unsupported_array_dtype() -> PyErr {
    PyTypeError::new_err("unsupported array-to-table dtype")
}

macro_rules! copy_primitive {
    ($values:expr, $output:expr, $column:expr, $offset:expr, $src:ty, $dst:ty) => {{
        let values = $values.as_any().downcast_ref::<PrimitiveArray<$src>>()
            .ok_or_else(|| PyTypeError::new_err("invalid numeric column"))?;
        for (row, &value) in values.values().iter().enumerate() {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_lossless,
                reason = "NumPy safe-cast preflight admits this conversion; preserve np.copyto semantics without intermediate buffers")]
            let converted = value as $dst;
            $output[($offset + row, $column)] = converted;
        }
        Ok(())
    }};
}

// Each destination has the same three small source-family dispatchers. Generic
// matrix ownership and chunk traversal stay outside the numeric specialization.
macro_rules! matrix_copy {
    ($copy:ident, $signed:ident, $unsigned:ident, $floating:ident, $dst:ty) => {
        fn $copy(values: &ArrayRef, output: &mut ArrayViewMut2<'_, $dst>, column: usize, offset: usize) -> PyResult<()> {
            if values.data_type().is_signed_integer() {
                $signed(values, output, column, offset)
            } else if values.data_type().is_unsigned_integer() {
                $unsigned(values, output, column, offset)
            } else {
                $floating(values, output, column, offset)
            }
        }

        fn $signed(values: &ArrayRef, output: &mut ArrayViewMut2<'_, $dst>, column: usize, offset: usize) -> PyResult<()> {
            match values.data_type() {
                DataType::Int8 => copy_primitive!(values, output, column, offset, Int8Type, $dst),
                DataType::Int16 => copy_primitive!(values, output, column, offset, Int16Type, $dst),
                DataType::Int32 => copy_primitive!(values, output, column, offset, Int32Type, $dst),
                DataType::Int64 => copy_primitive!(values, output, column, offset, Int64Type, $dst),
                _ => Err(unsupported_matrix_column()),
            }
        }

        fn $unsigned(values: &ArrayRef, output: &mut ArrayViewMut2<'_, $dst>, column: usize, offset: usize) -> PyResult<()> {
            match values.data_type() {
                DataType::UInt8 => copy_primitive!(values, output, column, offset, UInt8Type, $dst),
                DataType::UInt16 => copy_primitive!(values, output, column, offset, UInt16Type, $dst),
                DataType::UInt32 => copy_primitive!(values, output, column, offset, UInt32Type, $dst),
                DataType::UInt64 => copy_primitive!(values, output, column, offset, UInt64Type, $dst),
                _ => Err(unsupported_matrix_column()),
            }
        }

        fn $floating(values: &ArrayRef, output: &mut ArrayViewMut2<'_, $dst>, column: usize, offset: usize) -> PyResult<()> {
            match values.data_type() {
                DataType::Float16 => {
                    let values = values.as_any().downcast_ref::<PrimitiveArray<Float16Type>>()
                        .ok_or_else(|| PyTypeError::new_err("invalid float16 column"))?;
                    for (row, &value) in values.values().iter().enumerate() {
                        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss,
                            reason = "safe-cast preflight permits float16 only into an admitted floating dtype")]
                        let converted = value.to_f64() as $dst;
                        output[(offset + row, column)] = converted;
                    }
                    Ok(())
                }
                DataType::Float32 => copy_primitive!(values, output, column, offset, Float32Type, $dst),
                DataType::Float64 => copy_primitive!(values, output, column, offset, Float64Type, $dst),
                _ => Err(unsupported_matrix_column()),
            }
        }
    };
}

matrix_copy!(
    copy_f32,
    copy_f32_signed,
    copy_f32_unsigned,
    copy_f32_float,
    f32
);
matrix_copy!(
    copy_f64,
    copy_f64_signed,
    copy_f64_unsigned,
    copy_f64_float,
    f64
);
matrix_copy!(copy_i8, copy_i8_signed, copy_i8_unsigned, copy_i8_float, i8);
matrix_copy!(
    copy_i16,
    copy_i16_signed,
    copy_i16_unsigned,
    copy_i16_float,
    i16
);
matrix_copy!(
    copy_i32,
    copy_i32_signed,
    copy_i32_unsigned,
    copy_i32_float,
    i32
);
matrix_copy!(
    copy_i64,
    copy_i64_signed,
    copy_i64_unsigned,
    copy_i64_float,
    i64
);
matrix_copy!(copy_u8, copy_u8_signed, copy_u8_unsigned, copy_u8_float, u8);
matrix_copy!(
    copy_u16,
    copy_u16_signed,
    copy_u16_unsigned,
    copy_u16_float,
    u16
);
matrix_copy!(
    copy_u32,
    copy_u32_signed,
    copy_u32_unsigned,
    copy_u32_float,
    u32
);
matrix_copy!(
    copy_u64,
    copy_u64_signed,
    copy_u64_unsigned,
    copy_u64_float,
    u64
);

fn unsupported_matrix_column() -> PyErr {
    PyTypeError::new_err("invalid table_matmul columns: unsupported Arrow dtype")
}

pub(crate) fn table_matrix(
    py: Python<'_>,
    batch: &calc_flow::Batch,
    columns: Vec<String>,
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    let table = batch.table_payload().map_err(crate::error::to_py_err)?;
    let indices = columns
        .into_iter()
        .map(|name| super::column_index(table.schema(), &name))
        .collect::<PyResult<Vec<_>>>()?;
    validate_matrix_casts(py, table.schema(), &indices, dtype)?;
    matrix_for_dtype(py, batch.num_rows(), table.batches(), &indices, dtype)
}

fn matrix_for_dtype(
    py: Python<'_>,
    rows: usize,
    records: &[RecordBatch],
    indices: &[usize],
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    match dtype.as_bytes().first() {
        Some(b'f') => floating_matrix(py, rows, records, indices, dtype),
        Some(b'i') => signed_matrix(py, rows, records, indices, dtype),
        Some(b'u') => unsigned_matrix(py, rows, records, indices, dtype),
        _ => Err(PyTypeError::new_err("invalid table_matmul dtype")),
    }
}

fn floating_matrix(
    py: Python<'_>,
    rows: usize,
    records: &[RecordBatch],
    indices: &[usize],
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    match dtype {
        "float32" => typed_matrix::<f32>(py, rows, records, indices, dtype, copy_f32),
        "float64" => typed_matrix::<f64>(py, rows, records, indices, dtype, copy_f64),
        _ => Err(PyTypeError::new_err("invalid table_matmul dtype")),
    }
}

fn signed_matrix(
    py: Python<'_>,
    rows: usize,
    records: &[RecordBatch],
    indices: &[usize],
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    match dtype {
        "int8" => typed_matrix::<i8>(py, rows, records, indices, dtype, copy_i8),
        "int16" => typed_matrix::<i16>(py, rows, records, indices, dtype, copy_i16),
        "int32" => typed_matrix::<i32>(py, rows, records, indices, dtype, copy_i32),
        "int64" => typed_matrix::<i64>(py, rows, records, indices, dtype, copy_i64),
        _ => Err(PyTypeError::new_err("invalid table_matmul dtype")),
    }
}

fn unsigned_matrix(
    py: Python<'_>,
    rows: usize,
    records: &[RecordBatch],
    indices: &[usize],
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    match dtype {
        "uint8" => typed_matrix::<u8>(py, rows, records, indices, dtype, copy_u8),
        "uint16" => typed_matrix::<u16>(py, rows, records, indices, dtype, copy_u16),
        "uint32" => typed_matrix::<u32>(py, rows, records, indices, dtype, copy_u32),
        "uint64" => typed_matrix::<u64>(py, rows, records, indices, dtype, copy_u64),
        _ => Err(PyTypeError::new_err("invalid table_matmul dtype")),
    }
}

fn typed_matrix<T: Element>(
    py: Python<'_>,
    rows: usize,
    records: &[RecordBatch],
    indices: &[usize],
    dtype: &str,
    copy_column: impl Fn(&ArrayRef, &mut ArrayViewMut2<'_, T>, usize, usize) -> PyResult<()>,
) -> PyResult<Py<PyAny>> {
    let object = allocate_matrix(py, rows, indices.len(), dtype)?;
    {
        let mut output = object
            .bind(py)
            .cast::<numpy::PyArray2<T>>()?
            .try_readwrite()?;
        let mut output = output.as_array_mut();
        for (column, &index) in indices.iter().enumerate() {
            let mut offset = 0;
            for record in records {
                let values = record.column(index);
                if values.null_count() != 0 {
                    return Err(PyValueError::new_err(
                        "invalid table_matmul columns: contains nulls",
                    ));
                }
                copy_column(values, &mut output, column, offset)?;
                offset += record.num_rows();
            }
        }
    }
    Ok(object)
}

fn validate_matrix_casts(
    py: Python<'_>,
    schema: &Schema,
    indices: &[usize],
    dtype: &str,
) -> PyResult<()> {
    let np = py.import("numpy")?;
    for &index in indices {
        let source = matrix_source_dtype(&schema.fields()[index])?;
        validate_matrix_cast(&np, &source, dtype)?;
    }
    Ok(())
}

fn matrix_source_dtype(field: &datafusion::arrow::datatypes::Field) -> PyResult<String> {
    if field.data_type() == &DataType::Float16 {
        Ok("float16".into())
    } else {
        calc_flow::ArrowFieldSpec::from_arrow(field)
            .map(|spec| spec.data_type)
            .map_err(crate::error::to_py_err)
    }
}

fn validate_matrix_cast(np: &Bound<'_, PyModule>, source: &str, dtype: &str) -> PyResult<()> {
    let kwargs = PyDict::new(np.py());
    kwargs.set_item("casting", "safe")?;
    if !np
        .getattr("can_cast")?
        .call((source, dtype), Some(&kwargs))?
        .extract::<bool>()?
    {
        return Err(PyTypeError::new_err(
            "invalid table_matmul dtype: unsafe conversion",
        ));
    }
    Ok(())
}

fn allocate_matrix(
    py: Python<'_>,
    rows: usize,
    columns: usize,
    dtype: &str,
) -> PyResult<Py<PyAny>> {
    let (object, token) = py
        .get_type::<crate::batch::PyBatch>()
        .call_method1("_new_owned_numpy", (vec![rows, columns], dtype))?
        .extract::<(Py<PyAny>, Py<crate::batch::OwnedArrayToken>)>()?;
    token.borrow(py).consume(object.bind(py))?;
    Ok(object)
}
