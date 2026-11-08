//! Packed native matrix multiplication and axis reductions without Python.

use calc_flow::{
    Result,
    array::{self, ArrayExecutionOptions, NativeArray, Reduction, ReductionOptions},
};
use datafusion::arrow::array::Float64Array;
use std::sync::Arc;

fn main() -> Result<()> {
    let limits = ArrayExecutionOptions::default();
    let left = NativeArray::from_f64(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], None)?;
    let right = NativeArray::from_f64(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2], None)?;
    let product = array::matmul(left.view(), right.view(), &limits)?;
    let means = array::reduce(
        product.view(),
        Reduction::Mean,
        ReductionOptions::default().with_axis(Some(1)),
        &limits,
    )?;
    let transposed = array::transpose(&product, &limits)?;
    let flat = array::reshape(&product, &[4], &limits)?;
    assert_eq!(product.len(), 2);
    assert_eq!(product.element_count(), 4);
    assert_eq!(transposed.shape(), &[2, 2]);
    assert!(Arc::ptr_eq(product.as_arrow(), flat.as_arrow()));
    let product = product
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("floating matmul returns Float64");
    let means = means
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("mean returns Float64");
    assert_eq!(
        product
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [4.0_f64, 5.0, 10.0, 11.0].map(f64::to_bits)
    );
    assert_eq!(
        means
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [4.5_f64, 10.5].map(f64::to_bits)
    );
    println!("Matrix product: {:?}", product.values().as_ref());
    println!("Row means: {:?}", means.values().as_ref());
    Ok(())
}
