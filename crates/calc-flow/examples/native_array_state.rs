//! Owned native sequence states: chunk continuity, independent outputs and reset.

use calc_flow::{
    CalcFlowError, Result,
    array::{
        ArrayExecutionOptions, ArrayStateOptions, ArrayView, EwmaOptions, EwmaState, FloatValues,
        FloatView, NativeArray, RollingMeanOptions, RollingMeanState,
    },
};

fn view(values: &[f64]) -> ArrayView<'_> {
    ArrayView::from_float(FloatView::from_f64(values))
}

fn verify(output: &NativeArray, expected: &[f64]) -> Result<()> {
    let floats = output.float_view()?;
    let FloatValues::Float64(values) = floats.values() else {
        return Err(example_error("expected float64 output"));
    };
    let valid = floats.validity().is_none_or(|mask| mask.null_count() == 0);
    let equal = values
        .iter()
        .map(|value| value.to_bits())
        .eq(expected.iter().map(|value| value.to_bits()));
    if !valid || !equal {
        return Err(example_error("unexpected sequence result"));
    }
    Ok(())
}

fn example_error(message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "example.output".into(),
        message: message.into(),
    }
}

fn mean_chunks(execution: &ArrayExecutionOptions) -> Result<NativeArray> {
    let options = RollingMeanOptions::new(2)?;
    let mut state = RollingMeanState::new(options, ArrayStateOptions::default())?;
    let first = state.process(view(&[1.0, 3.0]), execution)?;
    verify(&first, &[1.0, 2.0])?;
    let second = state.process(view(&[5.0]), execution)?;
    verify(&second, &[4.0])?;
    state.reset();
    Ok(second)
}

fn ewma_chunks(execution: &ArrayExecutionOptions) -> Result<()> {
    let options = EwmaOptions::new(3)?;
    let mut state = EwmaState::new(options, ArrayStateOptions::default())?;
    let first = state.process(view(&[1.0, 3.0]), execution)?;
    verify(&first, &[1.0, 2.0])?;
    let second = state.process(view(&[5.0]), execution)?;
    verify(&second, &[3.5])
}

fn main() -> Result<()> {
    let execution = ArrayExecutionOptions::default();
    let retained = mean_chunks(&execution)?;
    ewma_chunks(&execution)?;
    // Output buffers outlive their input borrow and the state that produced them.
    verify(&retained, &[4.0])?;
    println!("Native mean and EWMA chunks verified; owned output survives state drop.");
    Ok(())
}
