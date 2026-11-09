//! Full MACD declarations share table ordering and stream state.

use std::{collections::BTreeMap, sync::Arc};

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, EventTime, ExecutionOptions, JsonMap,
    OperatorMetadata, PipelineBuilder, Result, RollingOperator, RollingSpec, StreamJobContext,
    StreamOperator, StreamOperatorContext, UdfRegistry,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
        UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::json;

fn schema(dtype: DataType) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("entity", DataType::Utf8, false),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", dtype, true),
    ]))
}
fn operator(dtype: DataType) -> RollingOperator {
    let outputs = [
        ("line", "line"),
        ("signal", "signal"),
        ("histogram", "histogram"),
        ("alias", "line"),
    ]
    .map(|(output, component)| {
        json!({
            "kind":"macd_component","primitive_version":1,"input":"price","output":output,
            "fast_span":1,"slow_span":3,"signal_span":3,
            "min_periods":1,"signal_min_periods":1,"component":component
        })
    });
    let spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":6,
        "partition_by":["entity"],"event_time":"ts","sequence_by":["seq"],
        "outputs":outputs,"allowed_lateness_micros":0,
        "late_policy":{"kind":"error","scope":"envelope"},"value_policy":"stateful_numeric_v1"
    }))
    .unwrap();
    RollingOperator::new("features", schema(dtype), spec).unwrap()
}
fn batch(times: Vec<i64>, entities: Vec<&str>, values: ArrayRef) -> Batch {
    let input_schema = schema(values.data_type().clone());
    let sequences = times
        .iter()
        .map(|time| u64::try_from(*time).unwrap())
        .collect::<Vec<_>>();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
        Arc::new(StringArray::from(entities)),
        Arc::new(UInt64Array::from(sequences)),
        values,
    ];
    Batch::table(
        vec![RecordBatch::try_new(input_schema, columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}
fn values(batch: &Batch, name: &str) -> Vec<Option<f64>> {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            record
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
        })
        .collect()
}
fn assert_components(batch: &Batch, expected: [&[Option<f64>]; 3]) {
    for (name, expected) in ["line", "signal", "histogram"].into_iter().zip(expected) {
        assert_eq!(values(batch, name), expected, "{name}");
    }
    assert_eq!(values(batch, "line"), values(batch, "alias"));
}

#[tokio::test]
async fn test_integer_table_macd_emits_three_components_and_duplicate_alias() -> Result<()> {
    let operator = operator(DataType::Int64);
    let input = batch(
        vec![1, 2, 3, 4],
        vec!["a"; 4],
        Arc::new(Int64Array::from(vec![Some(10), Some(14), None, None])),
    );
    let result = PipelineBuilder::new("table-macd")?
        .add_node("features", operator)?
        .compile_batch(&UdfRegistry::new().snapshot())?
        .execute(
            BTreeMap::from([("input".into(), input)]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_components(
        &result.outputs["output"],
        [
            &[Some(0.0), Some(2.0), Some(2.0), Some(2.0)],
            &[Some(0.0), Some(1.0), Some(1.5), Some(1.75)],
            &[Some(0.0), Some(1.0), Some(0.5), Some(0.25)],
        ],
    );
    Ok(())
}

#[tokio::test]
async fn test_stream_macd_keeps_each_entity_state_across_watermark_and_missing_values() -> Result<()>
{
    let mut operator = operator(DataType::Float64);
    let job = StreamJobContext::new(
        1,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let first = batch(
        vec![1, 2, 3],
        vec!["a", "b", "a"],
        Arc::new(Float64Array::from(vec![Some(10.0), Some(2.0), Some(14.0)])),
    );
    operator
        .process_data("input", first, &context, &mut output)
        .await?;
    operator
        .on_watermark(EventTime::from_micros(3), &context, &mut output)
        .await?;
    let first = output.drain("output");
    assert_eq!(first.len(), 1);
    assert_components(
        first[0].as_data().unwrap(),
        [
            &[Some(0.0), Some(0.0), Some(2.0)],
            &[Some(0.0), Some(0.0), Some(1.0)],
            &[Some(0.0), Some(0.0), Some(1.0)],
        ],
    );
    let second = batch(
        vec![4, 5, 6, 7],
        vec!["b", "a", "b", "a"],
        Arc::new(Float64Array::from(vec![
            Some(6.0),
            None,
            Some(10.0),
            Some(f64::NAN),
        ])),
    );
    operator
        .process_data("input", second, &context, &mut output)
        .await?;
    operator.on_end(&context, &mut output).await?;
    let second = output.drain("output");
    assert_eq!(second.len(), 1);
    assert_components(
        second[0].as_data().unwrap(),
        [
            &[Some(2.0), Some(2.0), Some(3.0), Some(2.0)],
            &[Some(1.0), Some(1.5), Some(2.0), Some(1.75)],
            &[Some(1.0), Some(0.5), Some(1.0), Some(0.25)],
        ],
    );
    Ok(())
}

#[tokio::test]
async fn test_table_macd_checkpoint_stores_one_complete_104_byte_group() -> Result<()> {
    use calc_flow::Epoch;
    use datafusion::arrow::{array::BinaryArray, ipc::reader::FileReader};
    use std::io::Cursor;

    let mut operator = operator(DataType::Int64);
    let job = StreamJobContext::new(
        1,
        "a".repeat(64),
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "features", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "input",
            batch(
                vec![1, 2, 3, 4],
                vec!["a"; 4],
                Arc::new(Int64Array::from(vec![Some(10), Some(14), None, None])),
            ),
            &context,
            &mut output,
        )
        .await?;
    operator.on_end(&context, &mut output).await?;
    operator.prepare_checkpoint_async(&context).await?;
    let snapshot = operator.checkpoint(Epoch::new(1).unwrap())?;
    let bytes = snapshot.segments.values().next().unwrap().bytes();
    let mut reader = FileReader::try_new(Cursor::new(bytes), None).unwrap();
    let record = reader.next().unwrap().unwrap();
    let states = record
        .column_by_name("_indicator_state")
        .unwrap()
        .as_any()
        .downcast_ref::<BinaryArray>()
        .unwrap();
    let payloads = states.iter().flatten().collect::<Vec<_>>();
    assert_eq!(
        payloads.len(),
        1,
        "aliases and readouts share one recurrence"
    );
    let payload = payloads[0];
    assert_eq!(payload.len(), 104);
    assert_eq!(&payload[..8], &[1, 0, 0, 0, 5, 0, 0, 0]);
    let expected = [
        4_u64,
        1,
        3,
        3,
        1,
        1,
        2,
        14.0_f64.to_bits(),
        2,
        12.0_f64.to_bits(),
        4,
        1.75_f64.to_bits(),
    ];
    for (chunk, expected) in payload[8..].chunks_exact(8).zip(expected) {
        assert_eq!(u64::from_le_bytes(chunk.try_into().unwrap()), expected);
    }
    let mut restored = self::operator(DataType::Int64);
    restored.restore_async(&snapshot, &context).await?;
    Ok(())
}

fn declaration(outputs: &serde_json::Value, layout: u32) -> RollingSpec {
    serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":layout,
        "partition_by":["entity"],"event_time":"ts","sequence_by":["seq"],
        "outputs":outputs,"allowed_lateness_micros":0,
        "late_policy":{"kind":"error","scope":"envelope"},"value_policy":"stateful_numeric_v1"
    }))
    .unwrap()
}
fn component(output: &str, kind: &str) -> serde_json::Value {
    json!({"kind":"macd_component","primitive_version":1,"input":"price","output":output,"component":kind})
}
fn assert_argument(error: calc_flow::CalcFlowError, expected: &str) {
    let calc_flow::CalcFlowError::InvalidArgument { field, .. } = error else {
        panic!("expected a structured argument error");
    };
    assert_eq!(field, expected);
}
#[test]
fn test_macd_layout_parameters_and_numeric_input_admission_are_strict() {
    let outputs = json!([component("histogram", "histogram")]);
    let old = declaration(&outputs, 5);
    assert_argument(
        RollingOperator::new("features", schema(DataType::Float64), old).unwrap_err(),
        "rolling.state_layout_version",
    );
    let mut invalid = outputs.clone();
    invalid[0]["fast_span"] = json!(26);
    assert_argument(
        RollingOperator::new(
            "features",
            schema(DataType::Float64),
            declaration(&invalid, 6),
        )
        .unwrap_err(),
        "rolling.outputs[0].fast_span",
    );
    for field in [
        "fast_span",
        "slow_span",
        "signal_span",
        "min_periods",
        "signal_min_periods",
    ] {
        let mut invalid = outputs.clone();
        invalid[0][field] = json!(0);
        assert_argument(
            RollingOperator::new(
                "features",
                schema(DataType::Float64),
                declaration(&invalid, 6),
            )
            .unwrap_err(),
            &format!("rolling.outputs[0].{field}"),
        );
    }
    assert!(
        RollingOperator::new(
            "features",
            schema(DataType::Boolean),
            declaration(&outputs, 6)
        )
        .is_err()
    );
    assert!(
        RollingOperator::new(
            "features",
            schema(DataType::Float64),
            declaration(&outputs, 7)
        )
        .is_err()
    );
}
#[tokio::test]
async fn test_histogram_subset_preserves_independent_minima_larger_than_spans() -> Result<()> {
    let mut histogram = component("histogram", "histogram");
    for (name, value) in [
        ("fast_span", 1),
        ("slow_span", 3),
        ("signal_span", 1),
        ("min_periods", 3),
        ("signal_min_periods", 2),
    ] {
        histogram[name] = json!(value);
    }
    let mut operator = RollingOperator::new(
        "features",
        schema(DataType::Int64),
        declaration(&json!([histogram]), 6),
    )?;
    operator.set_state_budget(calc_flow::StateBudget::new(1, 16 * 1024 * 1024)?)?;
    let input = batch(
        vec![1, 2, 3, 4, 5, 6],
        vec!["a"; 6],
        Arc::new(Int64Array::from(vec![
            Some(10),
            Some(14),
            None,
            Some(18),
            Some(22),
            None,
        ])),
    );
    let result = finite(operator, input).await?;
    assert_eq!(
        values(&result, "histogram"),
        [None, None, None, None, Some(0.0), Some(0.0)]
    );
    Ok(())
}

async fn finite(operator: RollingOperator, input: Batch) -> Result<Batch> {
    let mut result = PipelineBuilder::new("histogram-subset")?
        .add_node("features", operator)?
        .compile_batch(&UdfRegistry::new().snapshot())?
        .execute(
            BTreeMap::from([("input".into(), input)]),
            ExecutionOptions::default(),
        )
        .await?;
    Ok(result.outputs.remove("output").unwrap())
}

#[tokio::test]
async fn test_macd_unsigned_inputs_preserve_the_existing_ewma_float_conversion() -> Result<()> {
    use datafusion::arrow::array::{UInt8Array, UInt16Array, UInt32Array};

    let arrays: Vec<ArrayRef> = vec![
        Arc::new(UInt8Array::from(vec![Some(10), Some(14), None, None])),
        Arc::new(UInt16Array::from(vec![Some(10), Some(14), None, None])),
        Arc::new(UInt32Array::from(vec![Some(10), Some(14), None, None])),
        Arc::new(UInt64Array::from(vec![
            Some((1_u64 << 53) + 1),
            Some((1_u64 << 53) + 5),
            None,
            None,
        ])),
    ];
    for array in arrays {
        let declared = operator(array.data_type().clone());
        let result = finite(declared, batch(vec![1, 2, 3, 4], vec!["a"; 4], array)).await?;
        assert_components(
            &result,
            [
                &[Some(0.0), Some(2.0), Some(2.0), Some(2.0)],
                &[Some(0.0), Some(1.0), Some(1.5), Some(1.75)],
                &[Some(0.0), Some(1.0), Some(0.5), Some(0.25)],
            ],
        );
    }
    Ok(())
}

#[test]
fn test_macd_component_public_enum_keeps_snake_case_serialization() {
    use calc_flow::MacdComponent;

    let selectors = [
        MacdComponent::Line,
        MacdComponent::Signal,
        MacdComponent::Histogram,
    ];
    assert_eq!(json!(selectors), json!(["line", "signal", "histogram"]));
}
