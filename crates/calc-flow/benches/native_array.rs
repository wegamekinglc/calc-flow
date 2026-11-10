//! Native Array library calls, with input preparation and oracles outside timers.

use std::{hint::black_box, time::Duration};

use calc_flow::array::{
    self, ArrayExecutionOptions, MacdOptions, NativeArray, ReductionOptions, RollingMeanOptions,
    RsiOptions,
};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::arrow::array::{Array, Float64Array};

fn validate_constant(array: &NativeArray, shape: &[usize], expected: f64, warmup: usize) {
    let values = array
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("native numerical output is Float64");
    assert_eq!(array.shape(), shape);
    assert_eq!(values.len(), shape.iter().product::<usize>());
    assert_eq!(values.null_count(), warmup);
    assert!(values.iter().take(warmup).all(|value| value.is_none()));
    assert!(
        values
            .iter()
            .skip(warmup)
            .all(|value| { value.is_some_and(|value| value.to_bits() == expected.to_bits()) })
    );
}

fn sum(c: &mut Criterion, input: &NativeArray, limits: &ArrayExecutionOptions) {
    let options = ReductionOptions::default();
    let values = input.float_view().unwrap();
    validate_constant(
        &array::sum(values, options, limits).unwrap(),
        &[],
        1024.0,
        0,
    );
    c.bench_function("native_array/sum_f64_1024_ones", |b| {
        b.iter(|| black_box(array::sum(black_box(values), options, limits).unwrap()));
    });
}

fn rolling_mean(c: &mut Criterion, input: &NativeArray, limits: &ArrayExecutionOptions) {
    let options = RollingMeanOptions::new(20).unwrap();
    let values = input.float_view().unwrap();
    validate_constant(
        &array::rolling_mean(values, options, limits).unwrap(),
        &[1024],
        1.0,
        0,
    );
    c.bench_function(
        "native_array/rolling_mean_f64_1024_window20_constant",
        |b| {
            b.iter(|| black_box(array::rolling_mean(black_box(values), options, limits).unwrap()));
        },
    );
}

fn rsi(c: &mut Criterion, input: &NativeArray, limits: &ArrayExecutionOptions) {
    let options = RsiOptions::new(14).unwrap();
    validate_constant(
        &array::rsi(input.view(), options, limits).unwrap(),
        &[1024],
        50.0,
        14,
    );
    c.bench_function("native_array/rsi_f64_1024_period14_flat", |b| {
        b.iter(|| black_box(array::rsi(black_box(input.view()), options, limits).unwrap()));
    });
}

fn macd(c: &mut Criterion, input: &NativeArray, limits: &ArrayExecutionOptions) {
    let options = MacdOptions::new(12, 26, 9).unwrap();
    let expected = array::macd_components(input.view(), options, limits).unwrap();
    for array in [expected.line(), expected.signal(), expected.histogram()] {
        validate_constant(array, &[1024], 0.0, 0);
    }
    c.bench_function("native_array/macd_components_f64_1024_constant", |b| {
        b.iter(|| {
            black_box(array::macd_components(black_box(input.view()), options, limits).unwrap())
        });
    });
}

fn matmul(c: &mut Criterion, limits: &ArrayExecutionOptions) {
    let input = NativeArray::from_f64(vec![1.0; 256], vec![16, 16], None).unwrap();
    let result = array::matmul(input.view(), input.view(), limits).unwrap();
    assert_eq!(result.shape(), &[16, 16]);
    validate_constant(&result, &[16, 16], 16.0, 0);
    c.bench_function("native_array/matmul_f64_16x16_ones", |b| {
        b.iter(|| {
            black_box(
                array::matmul(black_box(input.view()), black_box(input.view()), limits).unwrap(),
            )
        });
    });
}

fn native_array(c: &mut Criterion) {
    let input = NativeArray::from_f64(vec![1.0; 1024], vec![1024], None).unwrap();
    let limits = ArrayExecutionOptions::default();
    sum(c, &input, &limits);
    rolling_mean(c, &input, &limits);
    rsi(c, &input, &limits);
    macd(c, &input, &limits);
    matmul(c, &limits);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(200));
    targets = native_array
}
criterion_main!(benches);
