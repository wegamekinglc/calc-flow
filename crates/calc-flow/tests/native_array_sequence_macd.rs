use calc_flow::{
    BatchKind, ExternalOperatorSpec, JsonMap, Port, StreamOperator, StreamOperatorFactory,
    array::NativeArraySequenceFactory,
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
#[test]
fn test_ir3_admits_full_macd_defaults_and_old_kernels_without_widening_old_versions() {
    for kernel in [
        json!({"function":"macd_components"}),
        json!({"function":"macd_components","fast_span":1,"slow_span":3,"signal_span":3,
            "min_periods":9,"signal_min_periods":12}),
        json!({"function":"rsi","period":2}),
        json!({"function":"atr","period":1}),
        json!({"function":"rolling_mean","window":3}),
        json!({"function":"ewma","span":3}),
    ] {
        create(options(3, kernel)).unwrap();
    }
    for version in [1, 2] {
        let error = create(options(version, json!({"function":"macd_components"})))
            .err()
            .unwrap();
        assert!(error.to_string().contains("options.kernel.function"));
    }
}

fn configured() -> JsonMap {
    options(
        3,
        json!({"function":"macd_components","fast_span":1,"slow_span":3,"signal_span":3}),
    )
}
fn batch(values: Vec<f64>, shape: Vec<usize>) -> calc_flow::Batch {
    calc_flow::Batch::external(
        std::sync::Arc::new(calc_flow::array::NativeArray::from_f64(values, shape, None).unwrap()),
        calc_flow::BatchMetadata::new("macd", 42, JsonMap::new()).unwrap(),
    )
    .unwrap()
}
fn job() -> calc_flow::StreamJobContext {
    calc_flow::StreamJobContext::new(
        709,
        "macd",
        JsonMap::new(),
        None,
        calc_flow::CancellationToken::new(),
    )
}
async fn output(
    operator: &mut dyn StreamOperator,
    input: calc_flow::Batch,
    context: &calc_flow::StreamOperatorContext<'_>,
) -> calc_flow::Result<calc_flow::Batch> {
    let mut collector = calc_flow::EdgeCollector::new(ports().1);
    operator
        .process_data("input", input, context, &mut collector)
        .await?;
    let messages = collector.drain("output");
    assert_eq!(messages.len(), 1);
    Ok(messages[0].as_data().unwrap().clone())
}
fn packed(batch: &calc_flow::Batch, shape: &[usize]) -> Vec<Option<u64>> {
    let payload = batch.external_payload().unwrap();
    let array = payload
        .as_any()
        .downcast_ref::<calc_flow::array::NativeArray>()
        .unwrap();
    assert_eq!(array.shape(), shape);
    let arrow = array.as_arrow();
    let values = arrow
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Float64Array>()
        .unwrap();
    let validity = arrow.nulls().unwrap();
    values
        .values()
        .iter()
        .enumerate()
        .map(|(index, value)| validity.is_valid(index).then(|| value.to_bits()))
        .collect()
}
#[tokio::test]
async fn test_macd_stream_uses_packed_output_and_restores_all_three_recurrences() {
    let mut operator = create(configured()).unwrap();
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "macd", None);
    let first = output(
        operator.as_mut(),
        batch(vec![10.0, 14.0], vec![2]),
        &context,
    )
    .await
    .unwrap();
    assert_eq!(first.metadata().sequence(), 42);
    assert_eq!(
        packed(&first, &[2, 3]),
        [0.0_f64, 0.0, 0.0, 2.0, 1.0, 1.0].map(|x| Some(x.to_bits()))
    );
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(calc_flow::Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["layout"], 3);
    assert_eq!(snapshot.inline_metadata["kernel"], 5);
    assert_eq!(snapshot.segments["state"].bytes().len(), 156);
    let mut restored = create(configured()).unwrap();
    restored.restore_async(&snapshot, &context).await.unwrap();
    let tail = output(
        restored.as_mut(),
        batch(vec![f64::NAN, 18.0], vec![2]),
        &context,
    )
    .await
    .unwrap();
    assert_eq!(
        packed(&tail, &[2, 3]),
        [2.0_f64, 1.5, 0.5, 3.0, 2.25, 0.75].map(|x| Some(x.to_bits()))
    );
    assert_eq!(
        packed(
            &output(restored.as_mut(), batch(vec![], vec![0]), &context)
                .await
                .unwrap(),
            &[0, 3]
        ),
        []
    );
}

#[test]
fn test_macd_ir3_count_fields_are_exact_positive_u64_without_capped_minima() {
    for field in [
        "fast_span",
        "slow_span",
        "signal_span",
        "min_periods",
        "signal_min_periods",
    ] {
        for invalid in [
            json!(0),
            json!(-1),
            json!(true),
            json!(1.5),
            Value::Null,
            json!(18_446_744_073_709_551_616.0),
        ] {
            let mut values = configured();
            values.get_mut("kernel").unwrap()[field] = invalid;
            let error = create(values).err().unwrap();
            assert!(error.to_string().contains(field), "{field}: {error}");
        }
    }
    let mut values = configured();
    values.get_mut("kernel").unwrap()["fast_span"] = 3.into();
    assert!(create(values).is_err());
    for field in ["period", "component", "alpha"] {
        let mut values = configured();
        values.get_mut("kernel").unwrap()[field] = 1.into();
        assert!(create(values).err().unwrap().to_string().contains(field));
    }
}

#[tokio::test]
async fn test_macd_ir3_default_and_explicit_parameters_share_checkpoint_identity() {
    let mut default = create(options(3, json!({"function":"macd_components"}))).unwrap();
    let mut explicit = create(options(
        3,
        json!({"function":"macd_components","fast_span":12,"slow_span":26,
        "signal_span":9,"min_periods":1,"signal_min_periods":1}),
    ))
    .unwrap();
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "macd", None);
    default.prepare_checkpoint_async(&context).await.unwrap();
    explicit.prepare_checkpoint_async(&context).await.unwrap();
    let left = default.checkpoint(calc_flow::Epoch::INITIAL).unwrap();
    let right = explicit.checkpoint(calc_flow::Epoch::INITIAL).unwrap();
    assert_eq!(left.inline_metadata, right.inline_metadata);
    assert_eq!(
        left.segments["state"].bytes(),
        right.segments["state"].bytes()
    );
}

#[test]
fn test_macd_descriptor_keeps_old_branches_and_adds_only_layout_three_rule() {
    let descriptor = NativeArraySequenceFactory::capability_descriptor();
    assert_eq!(descriptor["state_layouts"], json!([1, 2, 3]));
    assert_eq!(descriptor["state_version"], 1);
    assert_eq!(
        descriptor["array_rules"]["shape_rules"],
        json!([
            {"name":"native_sequence_vector","version":"1"},
            {"name":"native_sequence_indicators","version":"2"},
            {"name":"native_sequence_indicators","version":"3"}
        ])
    );
    let schema: Value =
        serde_json::from_str(descriptor["options_json_schema"].as_str().unwrap()).unwrap();
    assert_eq!(schema["additionalProperties"], false);
    for (index, count) in [(0, 2), (1, 4), (2, 5)] {
        assert_eq!(
            schema["oneOf"][index]["properties"]["ir_version"]["const"],
            index + 1
        );
        assert_eq!(
            schema["oneOf"][index]["properties"]["kernel"]["oneOf"]
                .as_array()
                .unwrap()
                .len(),
            count
        );
    }
    let macd = &schema["oneOf"][2]["properties"]["kernel"]["oneOf"][4];
    assert_eq!(macd["required"], json!(["function"]));
    assert_eq!(macd["additionalProperties"], false);
    for (field, value) in [
        ("fast_span", 12),
        ("slow_span", 26),
        ("signal_span", 9),
        ("min_periods", 1),
        ("signal_min_periods", 1),
    ] {
        assert_eq!(macd["properties"][field]["default"], value);
        assert_eq!(macd["properties"][field]["minimum"], 1);
        assert_eq!(macd["properties"][field]["maximum"], u64::MAX);
    }
}

#[tokio::test]
async fn test_macd_stream_f32_rank_admission_keeps_declared_dtype_and_committed_state() {
    let mut configured = configured();
    configured.insert("input_dtype".into(), "float32".into());
    let mut operator = create(configured).unwrap();
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "macd", None);
    let first = f32_batch(vec![10.0], vec![1]);
    assert_eq!(
        packed(
            &output(operator.as_mut(), first, &context).await.unwrap(),
            &[1, 3]
        ),
        vec![Some(0); 3]
    );
    reject(
        operator.as_mut(),
        f32_batch(vec![14.0], vec![]),
        &context,
        "shape",
    )
    .await;
    reject(
        operator.as_mut(),
        f32_batch(vec![14.0], vec![1, 1]),
        &context,
        "shape",
    )
    .await;
    reject(
        operator.as_mut(),
        batch(vec![14.0], vec![1]),
        &context,
        "dtype",
    )
    .await;
    let tail = output(operator.as_mut(), f32_batch(vec![14.0], vec![1]), &context)
        .await
        .unwrap();
    assert_eq!(
        packed(&tail, &[1, 3]),
        [2.0_f64, 1.0, 1.0].map(|x| Some(x.to_bits()))
    );
    operator.prepare_checkpoint_async(&context).await.unwrap();
    let snapshot = operator.checkpoint(calc_flow::Epoch::INITIAL).unwrap();
    assert_eq!(&snapshot.segments["state"].bytes()[16..20], &[5, 1, 0, 0]);
}
fn f32_batch(values: Vec<f32>, shape: Vec<usize>) -> calc_flow::Batch {
    calc_flow::Batch::external(
        std::sync::Arc::new(calc_flow::array::NativeArray::from_f32(values, shape, None).unwrap()),
        calc_flow::BatchMetadata::default(),
    )
    .unwrap()
}
async fn reject(
    operator: &mut dyn StreamOperator,
    input: calc_flow::Batch,
    context: &calc_flow::StreamOperatorContext<'_>,
    field: &str,
) {
    let mut collector = calc_flow::EdgeCollector::new(ports().1);
    let error = operator
        .process_data("input", input, context, &mut collector)
        .await
        .unwrap_err();
    assert!(error.to_string().contains(field), "{error}");
    assert!(collector.drain("output").is_empty());
}
#[tokio::test]
async fn test_macd_stream_counts_three_elements_and_exact_packed_peak_before_commit() {
    let job = job();
    let context = calc_flow::StreamOperatorContext::new(&job, "macd", None);
    for (execution, field) in [
        (json!({"max_elements":3}), "outputs.elements"),
        (
            json!({"max_allocation_bytes":153}),
            "execution.max_allocation_bytes",
        ),
    ] {
        let mut configured = configured();
        configured.insert("execution".into(), execution);
        let mut operator = create(configured).unwrap();
        output(operator.as_mut(), batch(vec![10.0], vec![1]), &context)
            .await
            .unwrap();
        reject(
            operator.as_mut(),
            batch(vec![100.0, 200.0], vec![2]),
            &context,
            field,
        )
        .await;
        let tail = output(operator.as_mut(), batch(vec![14.0], vec![1]), &context)
            .await
            .unwrap();
        assert_eq!(
            packed(&tail, &[1, 3]),
            [2.0_f64, 1.0, 1.0].map(|x| Some(x.to_bits()))
        );
    }
}
