use std::sync::Arc;

use calc_flow::{
    Batch, BatchKind, BatchMetadata, EdgeCollector, Epoch, ExternalOperatorSpec, JsonMap, Port,
    StreamJobContext, StreamOperator, StreamOperatorContext, StreamOperatorFactory,
    array::{FloatValues, NativeArray, NativeArraySequenceFactory},
};
use serde_json::{Value, json};

fn ports() -> (Vec<Port>, Vec<Port>) {
    (
        vec![Port::new("input", BatchKind::Array, true, None).unwrap()],
        vec![Port::new("output", BatchKind::Array, true, None).unwrap()],
    )
}
fn options(version: u64, kernel: Value) -> JsonMap {
    JsonMap::from([
        ("ir_version".into(), version.into()),
        ("input_dtype".into(), "float64".into()),
        ("kernel".into(), kernel),
    ])
}
fn create(configured: JsonMap) -> calc_flow::Result<Box<dyn StreamOperator>> {
    let spec = ExternalOperatorSpec::new("native", "array_sequence", "1", configured)?;
    let (inputs, outputs) = ports();
    NativeArraySequenceFactory::default().create(&spec, inputs, outputs)
}
fn make_operator(atr: bool) -> Box<dyn StreamOperator> {
    let kernel = if atr {
        json!({"function":"atr","period":1})
    } else {
        json!({"function":"rsi","period":2})
    };
    let mut configured = options(2, kernel);
    if atr {
        configured.insert("execution".into(), json!({"max_elements":2}));
    }
    create(configured).unwrap()
}
fn job() -> StreamJobContext {
    StreamJobContext::new(
        707,
        "indicators",
        JsonMap::new(),
        None,
        calc_flow::CancellationToken::new(),
    )
}
fn batch(values: Vec<f64>, shape: Vec<usize>) -> Batch {
    Batch::external(
        Arc::new(NativeArray::from_f64(values, shape, None).unwrap()),
        BatchMetadata::new("indicator", 42, JsonMap::new()).unwrap(),
    )
    .unwrap()
}
fn bits(array: &NativeArray) -> Vec<Option<u64>> {
    let view = array.float_view().unwrap();
    let FloatValues::Float64(values) = view.values() else {
        panic!("expected f64");
    };
    values
        .iter()
        .enumerate()
        .map(|(i, value)| {
            view.validity()
                .unwrap()
                .is_valid(i)
                .then(|| value.to_bits())
        })
        .collect()
}
async fn output(
    operator: &mut dyn StreamOperator,
    input: Batch,
    context: &StreamOperatorContext<'_>,
) -> Vec<Option<u64>> {
    let mut collector = EdgeCollector::new(ports().1);
    operator
        .process_data("input", input, context, &mut collector)
        .await
        .unwrap();
    let messages = collector.drain("output");
    assert_eq!(messages.len(), 1);
    let batch = messages[0].as_data().unwrap();
    assert_eq!(batch.metadata().sequence(), 42);
    bits(
        batch
            .external_payload()
            .unwrap()
            .as_any()
            .downcast_ref::<NativeArray>()
            .unwrap(),
    )
}

#[test]
fn test_ir2_admits_indicator_and_legacy_kernels_while_ir1_stays_closed() {
    for kernel in [
        json!({"function":"rsi","period":2}),
        json!({"function":"atr","period":1}),
        json!({"function":"rolling_mean","window":3}),
        json!({"function":"ewma","span":3}),
    ] {
        create(options(2, kernel)).unwrap();
    }
    for function in ["rsi", "atr"] {
        assert!(create(options(1, json!({"function":function,"period":2}))).is_err());
    }
}

#[test]
fn test_ir2_period_and_field_validation_is_strict() {
    for kernel in [
        json!({"function":"rsi"}),
        json!({"function":"rsi","period":0}),
        json!({"function":"atr","period":true}),
        json!({"function":"rsi","period":1.5}),
        json!({"function":"atr","period":2,"min_periods":1}),
        json!({"function":"macd_components","period":1}),
    ] {
        assert!(create(options(2, kernel)).is_err());
    }
}

#[tokio::test]
async fn test_rsi_stream_checkpoint_continuation_has_exact_layout_and_values() {
    let mut operator = make_operator(false);
    let job = job();
    let context = StreamOperatorContext::new(&job, "price_rsi", None);
    assert_eq!(
        output(operator.as_mut(), batch(vec![1.0, 2.0], vec![2]), &context).await,
        vec![None, None]
    );
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["layout"], 2);
    assert_eq!(snapshot.segments["state"].bytes().len(), 116);
    let mut restored = make_operator(false);
    restored.restore_async(&snapshot, &context).await.unwrap();
    assert_eq!(
        output(restored.as_mut(), batch(vec![1.0, 3.0], vec![2]), &context).await,
        vec![Some(50.0_f64.to_bits()), Some((100.0_f64 / 1.2).to_bits())]
    );
}

#[tokio::test]
async fn test_atr_stream_reads_packed_rows_and_rejects_wrong_shapes_atomically() {
    let mut operator = make_operator(true);
    let job = job();
    let context = StreamOperatorContext::new(&job, "price_atr", None);
    assert_eq!(
        output(
            operator.as_mut(),
            batch(vec![10.0, 8.0, 9.0, 12.0, 9.0, 10.0], vec![2, 3]),
            &context
        )
        .await,
        vec![None, Some(3.0_f64.to_bits())]
    );
    assert!(
        output(operator.as_mut(), batch(vec![], vec![0, 3]), &context)
            .await
            .is_empty()
    );
    for shape in [vec![0], vec![0, 2]] {
        let mut collector = EdgeCollector::new(ports().1);
        let error = operator
            .process_data("input", batch(vec![], shape), &context, &mut collector)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("shape"));
        assert!(collector.drain("output").is_empty());
    }
    operator.prepare_checkpoint_async(&context).await.unwrap();
    assert_eq!(
        operator.checkpoint(Epoch::INITIAL).unwrap().segments["state"]
            .bytes()
            .len(),
        108
    );
    assert_eq!(
        output(
            operator.as_mut(),
            batch(vec![14.0, 11.0, 13.0], vec![1, 3]),
            &context
        )
        .await,
        vec![Some(4.0_f64.to_bits())]
    );
}

#[test]
fn test_indicator_descriptor_keeps_old_ir_branch_and_adds_layout_two() {
    let descriptor = NativeArraySequenceFactory::capability_descriptor();
    assert_eq!(descriptor["state_layouts"], json!([1, 2, 3]));
    assert_eq!(
        descriptor["array_rules"]["shape_rules"],
        json!([{"name":"native_sequence_vector","version":"1"},{"name":"native_sequence_indicators","version":"2"},{"name":"native_sequence_indicators","version":"3"}])
    );
    let schema: Value =
        serde_json::from_str(descriptor["options_json_schema"].as_str().unwrap()).unwrap();
    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(schema["oneOf"][0]["properties"]["ir_version"]["const"], 1);
    assert_eq!(
        schema["oneOf"][0]["properties"]["kernel"]["oneOf"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        schema["oneOf"][1]["properties"]["kernel"]["oneOf"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
}

#[tokio::test]
async fn test_rsi_float32_sequence_widens_and_persists_declared_dtype() {
    let mut configured = options(2, json!({"function":"rsi","period":1}));
    configured.insert("input_dtype".into(), "float32".into());
    let mut operator = create(configured).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "f32_rsi", None);
    let input = Batch::external(
        Arc::new(NativeArray::from_f32(vec![1.0, 2.0, 0.0], vec![3], None).unwrap()),
        BatchMetadata::new("indicator", 42, JsonMap::new()).unwrap(),
    )
    .unwrap();
    assert_eq!(
        output(operator.as_mut(), input, &context).await,
        vec![None, Some(100.0_f64.to_bits()), Some(0.0_f64.to_bits())]
    );
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(&snapshot.segments["state"].bytes()[16..20], &[3, 1, 0, 0]);
}
