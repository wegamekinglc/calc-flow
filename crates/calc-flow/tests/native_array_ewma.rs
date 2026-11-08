use std::{collections::BTreeMap, mem::size_of, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, ExecutionOptions, PipelineBuilder, RollingOperator,
    RollingSpec, UdfRegistry,
    array::{
        ArrayDType, ArrayExecutionOptions, EwmaOptions, FloatValues, FloatView, NativeArray, ewma,
    },
};
use datafusion::arrow::{
    array::{Array, ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    buffer::NullBuffer,
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

fn output_bits(output: &NativeArray) -> Vec<Option<u64>> {
    assert_eq!(output.dtype(), ArrayDType::Float64);
    output
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .iter()
        .map(|value| value.map(f64::to_bits))
        .collect()
}

fn bits(values: &[Option<f64>]) -> Vec<Option<u64>> {
    values.iter().map(|value| value.map(f64::to_bits)).collect()
}

#[test]
fn test_ewma_preserves_table_seed_skip_and_readout_contract() {
    let values = [10.0, 123_456.75, 14.0, f64::NAN, 18.0, 10.0];
    let validity = NullBuffer::from(vec![true, false, true, true, true, true]);
    let input = FloatView::try_new(FloatValues::Float64(&values), Some(&validity)).unwrap();
    let options = EwmaOptions::new(3).unwrap().with_min_periods(2).unwrap();
    assert_eq!(options.span(), 3);
    assert_eq!(options.min_periods(), 2);
    let output = ewma(input, options, &ArrayExecutionOptions::default()).unwrap();
    assert_eq!(output.shape(), &[6]);
    assert_eq!(
        output_bits(&output),
        bits(&[None, None, Some(12.0), Some(12.0), Some(15.0), Some(12.5)])
    );
    assert_eq!(values[1].to_bits(), 123_456.75_f64.to_bits());
}

#[test]
fn test_ewma_widens_float32_and_accepts_min_periods_above_span() {
    let values = [1.0_f32, 3.0, 5.0, 7.0];
    let validity = NullBuffer::new_valid(6).slice(2, 4);
    let input = FloatView::try_new(FloatValues::Float32(&values), Some(&validity)).unwrap();
    let output = ewma(
        input,
        EwmaOptions::new(3).unwrap().with_min_periods(4).unwrap(),
        &ArrayExecutionOptions::default(),
    )
    .unwrap();
    assert_eq!(output_bits(&output), bits(&[None, None, None, Some(5.25)]));
    assert_eq!(validity.offset(), 2);
}

#[test]
fn test_ewma_preserves_span_one_and_valid_nan_results_from_infinity() {
    let values = [-0.0, f64::INFINITY, 2.0, f64::NEG_INFINITY, f64::NAN, 3.0];
    let output = ewma(
        FloatView::from_f64(&values),
        EwmaOptions::new(1).unwrap(),
        &ArrayExecutionOptions::default(),
    )
    .unwrap();
    assert_eq!(
        output_bits(&output),
        bits(&[
            Some(-0.0),
            Some(f64::INFINITY),
            Some(2.0),
            Some(f64::NEG_INFINITY),
            Some(f64::NEG_INFINITY),
            Some(3.0)
        ])
    );

    let output = ewma(
        FloatView::from_f64(&[f64::INFINITY, 2.0, f64::NAN, 3.0]),
        EwmaOptions::new(3).unwrap(),
        &ArrayExecutionOptions::default(),
    )
    .unwrap();
    let values = output
        .as_arrow()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(values.null_count(), 0);
    assert_eq!(values.value(0).to_bits(), f64::INFINITY.to_bits());
    assert!(values.value(1).is_nan());
    assert!(values.value(2).is_nan());
    assert!(values.value(3).is_nan());

    let output = ewma(
        FloatView::from_f64(&[1.0e308, -1.0e308]),
        EwmaOptions::new(3).unwrap(),
        &ArrayExecutionOptions::default(),
    )
    .unwrap();
    assert_eq!(
        output_bits(&output),
        bits(&[Some(1.0e308), Some(f64::NEG_INFINITY)])
    );
}

fn assert_field(error: CalcFlowError, expected: &str) {
    assert!(matches!(error, CalcFlowError::InvalidArgument { field, .. } if field == expected));
}

#[test]
fn test_ewma_validates_options_and_allocations_before_execution() {
    assert_field(EwmaOptions::new(0).unwrap_err(), "options.span");
    assert_field(
        EwmaOptions::new(3)
            .unwrap()
            .with_min_periods(0)
            .unwrap_err(),
        "options.min_periods",
    );
    let input = FloatView::from_f64(&[1.0, 3.0, 5.0]);
    let options = EwmaOptions::new(3).unwrap();
    let defaults = ArrayExecutionOptions::default();
    assert_field(
        ewma(input, options, &defaults.with_max_elements(2)).unwrap_err(),
        "outputs.output.elements",
    );
    let output_bytes = 3 * size_of::<f64>() + 1 + size_of::<usize>();
    let scratch_bytes = size_of::<(u64, f64)>();
    assert_field(
        ewma(
            input,
            options,
            &defaults.with_max_allocation_bytes(output_bytes - 1),
        )
        .unwrap_err(),
        "outputs.output.bytes",
    );
    assert_field(
        ewma(
            input,
            options,
            &defaults.with_max_allocation_bytes(output_bytes + scratch_bytes - 1),
        )
        .unwrap_err(),
        "scratch.bytes",
    );
    let output = ewma(
        input,
        options,
        &defaults.with_max_allocation_bytes(output_bytes + scratch_bytes),
    )
    .unwrap();
    assert_eq!(output.retained_buffer_bytes(), output_bytes);
}

#[test]
fn test_ewma_accepts_empty_input_and_maximum_span() {
    let output = ewma(
        FloatView::from_f32(&[]),
        EwmaOptions::new(3).unwrap(),
        &ArrayExecutionOptions::default().with_max_elements(0),
    )
    .unwrap();
    assert_eq!(output.shape(), &[0]);
    assert!(output.is_empty());
    let output = ewma(
        FloatView::from_f64(&[1.0, 3.0]),
        EwmaOptions::new(u64::MAX).unwrap(),
        &ArrayExecutionOptions::default(),
    )
    .unwrap();
    assert_eq!(output_bits(&output), bits(&[Some(1.0), Some(1.0)]));
}

#[tokio::test]
async fn test_ewma_matches_native_table_outputs_bit_for_bit() {
    let prices = vec![
        None,
        Some(10.0),
        None,
        Some(14.0),
        Some(f64::NAN),
        Some(18.0),
        Some(-0.0),
    ];
    let price_column = Float64Array::from(prices);
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
    let rows = u64::try_from(price_column.len()).unwrap();
    let times = (0..rows)
        .map(|value| i64::try_from(value).unwrap())
        .collect::<Vec<_>>();
    let record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")) as ArrayRef,
            Arc::new(StringArray::from(vec!["a"; price_column.len()])),
            Arc::new(UInt64Array::from_iter_values(0..rows)),
            Arc::new(price_column.clone()),
        ],
    )
    .unwrap();
    for span in [3_u64, 6] {
        let spec: RollingSpec = serde_json::from_value(serde_json::json!({
            "configuration_version":1,"state_layout_version":2,"partition_by":["symbol"],
            "event_time":"ts","sequence_by":["sequence"],"outputs":[{
                "kind":"ewma","primitive_version":1,"input":"price","output":"ema","span":span,"min_periods":2
            }],"allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},
            "value_policy":"stateful_numeric_v1"
        })).unwrap();
        let operator = RollingOperator::new("ewma", Arc::clone(&schema), spec).unwrap();
        let plan = PipelineBuilder::new("native array table parity")
            .unwrap()
            .add_node("ewma", operator)
            .unwrap()
            .compile_batch(&UdfRegistry::new().snapshot())
            .unwrap();
        let run = plan
            .execute(
                BTreeMap::from([(
                    "input".into(),
                    Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap(),
                )]),
                ExecutionOptions::default(),
            )
            .await
            .unwrap();
        let output = run.outputs["output"].table_payload().unwrap().batches()[0]
            .column_by_name("ema")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let native = ewma(
            FloatView::try_new(
                FloatValues::Float64(price_column.values()),
                price_column.nulls(),
            )
            .unwrap(),
            EwmaOptions::new(span).unwrap().with_min_periods(2).unwrap(),
            &ArrayExecutionOptions::default(),
        )
        .unwrap();
        assert_eq!(
            output
                .iter()
                .map(|v| v.map(f64::to_bits))
                .collect::<Vec<_>>(),
            output_bits(&native)
        );
    }
}
