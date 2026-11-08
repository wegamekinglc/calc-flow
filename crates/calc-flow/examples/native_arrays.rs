//! Native array calculation without a Python interpreter or provider callback.

use std::sync::Arc;

use calc_flow::{
    Result,
    array::{
        self, ArrayExecutionOptions, ArrayView, NativeArray, ReductionOptions, RollingMeanOptions,
    },
};
use datafusion::arrow::array::Float64Array;

fn main() -> Result<()> {
    let values = NativeArray::from_arrow(
        Arc::new(Float64Array::from(vec![
            Some(2.0),
            None,
            Some(4.0),
            Some(8.0),
        ])),
        vec![4],
    )?;
    let execution = ArrayExecutionOptions::default();
    let result = filtered_average(&values, &execution)?;
    let total = array::sum(
        result.float_view()?,
        ReductionOptions::default(),
        &execution,
    )?;
    let actual = result
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("floating selection preserves Float64");
    assert_eq!(
        actual
            .iter()
            .map(|value| value.map(f64::to_bits))
            .collect::<Vec<_>>(),
        [0.0_f64, 0.0, 4.0, 6.0].map(|value| Some(value.to_bits()))
    );
    let total = total
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("sum produces Float64")
        .value(0);
    assert_eq!(total.to_bits(), 10.0_f64.to_bits());
    println!("Native values: {:?}", actual.values());
    println!("Native sum: {total}");
    Ok(())
}

fn filtered_average(
    values: &NativeArray,
    execution: &ArrayExecutionOptions,
) -> Result<NativeArray> {
    let average =
        array::rolling_mean(values.float_view()?, RollingMeanOptions::new(2)?, execution)?;
    let condition =
        array::greater_equal(average.view(), ArrayView::scalar_f64(Some(4.0)), execution)?;
    array::where_(
        condition.view(),
        average.view(),
        ArrayView::scalar_f64(Some(0.0)),
        execution,
    )
}
