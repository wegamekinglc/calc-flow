use calc_flow::{
    Result,
    array::{
        ArrayExecutionOptions, ArrayStateOptions, ArrayView, AtrOptions, FloatValues, FloatView,
        RsiOptions, RsiState, atr,
    },
};

fn view(values: &[f64]) -> ArrayView<'_> {
    ArrayView::from_float(FloatView::from_f64(values))
}

fn rsi_chunks(execution: &ArrayExecutionOptions) -> Result<()> {
    let mut state = RsiState::new(RsiOptions::new(2)?, ArrayStateOptions::default())?;
    state.process(view(&[1.0, 2.0]), execution)?;
    let output = state.process(view(&[1.0, 3.0]), execution)?;
    assert_eq!(state.state_bytes(), 48);
    assert!(output.float_view()?.validity().unwrap().is_valid(0));
    Ok(())
}

fn true_ranges(execution: &ArrayExecutionOptions) -> Result<()> {
    let output = atr(
        view(&[10.0, 12.0]),
        view(&[8.0, 9.0]),
        view(&[9.0, 10.0]),
        AtrOptions::new(1)?,
        execution,
    )?;
    let FloatValues::Float64(values) = output.float_view()?.values() else {
        unreachable!("ATR returns float64");
    };
    assert_eq!(values[1].to_bits(), 3.0_f64.to_bits());
    Ok(())
}

fn main() -> Result<()> {
    let execution = ArrayExecutionOptions::default();
    rsi_chunks(&execution)?;
    true_ranges(&execution)
}
