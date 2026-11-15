use calc_flow::array::{
    ArrayExecutionOptions, FloatValues, FloatView, NativeArray, RollingMeanOptions, rolling_mean,
};
use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, ExecutionOptions, PipelineBuilder,
    RollingNumericalProfile, RollingOperator, RollingSpec, UdfRegistry,
};
use datafusion::arrow::{
    array::{Array, ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    buffer::NullBuffer,
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

fn output_bits(output: &NativeArray) -> Vec<Option<u64>> {
    let view = output.float_view().unwrap();
    let FloatValues::Float64(values) = view.values() else {
        panic!("expected Float64 output");
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            view.validity()
                .is_none_or(|validity| validity.is_valid(index))
                .then(|| value.to_bits())
        })
        .collect()
}

#[test]
fn test_rolling_mean_keeps_window_positions_and_minimum_periods() {
    let input = [1.0, f64::NAN, 3.0, 5.0, 7.0];
    let options = RollingMeanOptions::new(3)
        .unwrap()
        .with_min_periods(2)
        .unwrap();
    let output = rolling_mean(
        FloatView::from_f64(&input),
        options,
        &ArrayExecutionOptions::default(),
    )
    .unwrap();
    assert_eq!(output.shape(), &[5]);
    assert_eq!(
        output_bits(&output),
        [None, None, Some(2.0_f64), Some(4.0), Some(5.0)].map(|value| value.map(f64::to_bits))
    );
}

#[test]
fn test_rolling_mean_preserves_profiles_and_non_finite_readouts() {
    let options = RollingMeanOptions::new(2).unwrap();
    assert_eq!(options.window(), 2);
    assert_eq!(options.min_periods(), 1);
    assert_eq!(
        options.numerical_profile(),
        RollingNumericalProfile::StableV1
    );
    let output = rolling_mean(
        FloatView::from_f64(&[-1e308, 1e308]),
        options,
        &ArrayExecutionOptions::default(),
    )
    .unwrap();
    assert_eq!(
        output_bits(&output),
        [Some((-1e308_f64).to_bits()), Some(f64::INFINITY.to_bits())]
    );
    for profile in [
        RollingNumericalProfile::StableV1,
        RollingNumericalProfile::StableV2Preview,
    ] {
        let output = rolling_mean(
            FloatView::from_f64(&[f64::INFINITY, f64::NEG_INFINITY, 2.0, 4.0]),
            options.with_numerical_profile(profile),
            &ArrayExecutionOptions::default(),
        )
        .unwrap();
        let bits = output_bits(&output);
        assert_eq!(bits[0], Some(f64::INFINITY.to_bits()));
        assert!(f64::from_bits(bits[1].unwrap()).is_nan());
        assert_eq!(bits[2], Some(f64::NEG_INFINITY.to_bits()));
        assert_eq!(bits[3], Some(3.0_f64.to_bits()));
    }
}

#[test]
fn test_rolling_mean_widens_float32_and_preserves_sliced_null_positions() {
    let values = [1.0_f32, 123_456.75, 3.0, 5.0];
    let bitmap = NullBuffer::from(vec![true, true, false, true, true, false]).slice(1, 4);
    let output = rolling_mean(
        FloatView::try_new(FloatValues::Float32(&values), Some(&bitmap)).unwrap(),
        RollingMeanOptions::new(2).unwrap(),
        &ArrayExecutionOptions::default(),
    )
    .unwrap();
    assert_eq!(
        output_bits(&output),
        [Some(1.0_f64), Some(1.0), Some(3.0), Some(4.0)].map(|value| value.map(f64::to_bits))
    );
    assert_eq!(bitmap.offset(), 1);
    assert_eq!(values[1].to_bits(), 123_456.75_f32.to_bits());
}

fn assert_field(error: CalcFlowError, expected: &str) {
    assert!(matches!(error, CalcFlowError::InvalidArgument { field, .. } if field == expected));
}

#[test]
fn test_rolling_mean_validates_options_and_budget_and_bounds_large_windows() {
    assert_field(RollingMeanOptions::new(0).unwrap_err(), "options.window");
    for minimum in [0, 4] {
        assert_field(
            RollingMeanOptions::new(3)
                .unwrap()
                .with_min_periods(minimum)
                .unwrap_err(),
            "options.min_periods",
        );
    }
    let values = [1.0, 3.0];
    let input = FloatView::from_f64(&values);
    let options = RollingMeanOptions::new(u64::MAX).unwrap();
    let defaults = ArrayExecutionOptions::default();
    assert_field(
        rolling_mean(input, options, &defaults.with_max_elements(1)).unwrap_err(),
        "outputs.output.elements",
    );
    assert_field(
        rolling_mean(input, options, &defaults.with_max_allocation_bytes(0)).unwrap_err(),
        "outputs.output.bytes",
    );
    let output = rolling_mean(input, options, &defaults).unwrap();
    assert_eq!(
        output_bits(&output),
        [Some(1.0_f64.to_bits()), Some(2.0_f64.to_bits())]
    );
    let empty = rolling_mean(
        FloatView::from_f64(&[]),
        options,
        &defaults.with_max_elements(0),
    )
    .unwrap();
    assert_eq!(empty.shape(), &[0]);
}

#[test]
fn test_rolling_mean_retains_m2_repair_and_expires_missing_positions() {
    let repaired = rolling_mean(
        FloatView::from_f64(&[1e155, 1e155, 1e155, -1e155]),
        RollingMeanOptions::new(2).unwrap(),
        &ArrayExecutionOptions::default(),
    )
    .unwrap();
    assert_eq!(output_bits(&repaired)[3], Some(0.0_f64.to_bits()));
    let missing = rolling_mean(
        FloatView::from_f64(&[1.0, f64::NAN, f64::NAN, f64::NAN, 5.0]),
        RollingMeanOptions::new(2).unwrap(),
        &ArrayExecutionOptions::default(),
    )
    .unwrap();
    assert_eq!(
        output_bits(&missing),
        [Some(1.0_f64), Some(1.0), None, None, Some(5.0)].map(|value| value.map(f64::to_bits))
    );
}

fn table_record(prices: &Float64Array) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
    ]));
    let rows = u64::try_from(prices.len()).unwrap();
    let times = (0..rows)
        .map(|value| i64::try_from(value).unwrap())
        .collect::<Vec<_>>();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")) as ArrayRef,
            Arc::new(StringArray::from(vec!["a"; prices.len()])),
            Arc::new(UInt64Array::from_iter_values(0..rows)),
            Arc::new(prices.clone()),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn test_rolling_mean_matches_native_table_profiles_bit_for_bit() {
    let prices = Float64Array::from(
        (0..130_u32)
            .map(|index| {
                if index % 17 == 0 {
                    None
                } else if index % 9 == 0 {
                    Some(f64::NAN)
                } else {
                    Some(1.0e12 + f64::from(index % 11) * 0.125)
                }
            })
            .collect::<Vec<_>>(),
    );
    let record = table_record(&prices);
    for profile in [
        RollingNumericalProfile::StableV1,
        RollingNumericalProfile::StableV2Preview,
    ] {
        for window in [2_u64, 13] {
            let spec: RollingSpec = serde_json::from_value(serde_json::json!({
                "configuration_version":1,"state_layout_version":1,"numerical_profile":profile,
                "partition_by":["symbol"],"event_time":"ts","sequence_by":["sequence"],
                "outputs":[{"kind":"mean","primitive_version":1,"input":"price","output":"average",
                    "frame":{"kind":"rows","size":window},"min_periods":2}],
                "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
                "value_policy":"stateful_numeric_v1"
            }))
            .unwrap();
            let operator = RollingOperator::new("mean", record.schema(), spec).unwrap();
            let plan = PipelineBuilder::new("native array rolling mean parity")
                .unwrap()
                .add_node("mean", operator)
                .unwrap()
                .compile_batch(&UdfRegistry::new().snapshot())
                .unwrap();
            let result = plan
                .execute(
                    BTreeMap::from([(
                        "input".into(),
                        Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap(),
                    )]),
                    ExecutionOptions::default(),
                )
                .await
                .unwrap();
            let output = result.outputs["output"].table_payload().unwrap().batches()[0]
                .column_by_name("average")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let native = rolling_mean(
                FloatView::try_new(FloatValues::Float64(prices.values()), prices.nulls()).unwrap(),
                RollingMeanOptions::new(window)
                    .unwrap()
                    .with_min_periods(2)
                    .unwrap()
                    .with_numerical_profile(profile),
                &ArrayExecutionOptions::default(),
            )
            .unwrap();
            assert_eq!(
                output
                    .iter()
                    .map(|value| value.map(f64::to_bits))
                    .collect::<Vec<_>>(),
                output_bits(&native),
                "profile={profile:?}, window={window}",
            );
        }
    }
}
