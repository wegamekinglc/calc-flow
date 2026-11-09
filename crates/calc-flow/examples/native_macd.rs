use calc_flow::{
    Result,
    array::{
        ArrayExecutionOptions, ArrayStateOptions, ArrayView, FloatValues, FloatView, MacdOptions,
        MacdOutputs, MacdState, NativeArray,
    },
};

fn component(array: &NativeArray, expected: &[f64]) -> Result<()> {
    let FloatValues::Float64(values) = array.float_view()?.values() else {
        unreachable!("MACD returns float64");
    };
    assert_eq!(values.len(), expected.len());
    for (value, expected) in values.iter().zip(expected) {
        assert_eq!(value.to_bits(), expected.to_bits());
    }
    Ok(())
}
fn components(output: MacdOutputs) -> Result<()> {
    let (line, signal, histogram) = output.into_arrays();
    component(&line, &[2.0, 3.0])?;
    component(&signal, &[1.5, 2.25])?;
    component(&histogram, &[0.5, 0.75])
}
fn main() -> Result<()> {
    let execution = ArrayExecutionOptions::default();
    let mut state = MacdState::new(MacdOptions::new(1, 3, 3)?, ArrayStateOptions::default())?;
    let first = state.process(
        ArrayView::from_float(FloatView::from_f64(&[10.0, 14.0])),
        &execution,
    )?;
    let next = state.process(
        ArrayView::from_float(FloatView::from_f64(&[f64::NAN, 18.0])),
        &execution,
    )?;
    components(next)?;
    state.reset();
    assert_eq!(first.line().shape(), &[2]);
    println!("MACD carries the ready line through a missing price and advances signal.");
    Ok(())
}
